//! CPU-side DiffusionGemma canvas state and deterministic confidence commits.
//!
//! This module deliberately does not implement model logits or sampling. Those
//! are produced by the model execution path; this layer validates and applies
//! the resulting candidate updates without owning device tensors.

use thiserror::Error;

/// DiffusionGemma's audited initial canvas length.
pub const GEMMA_CANVAS_TOKENS: usize = 256;

/// Lifecycle of one canvas position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanvasTokenState {
    /// Position has not yet been committed by a denoising iteration.
    Masked,
    /// A candidate was produced but did not meet the commit threshold.
    Provisional,
    /// Token is accepted and immutable for subsequent canvas iterations.
    Committed,
}

/// One token slot in the diffusion canvas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanvasPosition {
    token_id: Option<u32>,
    confidence: f32,
    state: CanvasTokenState,
}

impl CanvasPosition {
    pub const fn token_id(&self) -> Option<u32> {
        self.token_id
    }

    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    pub const fn state(&self) -> CanvasTokenState {
        self.state
    }
}

/// Counters returned after applying one model-produced denoising iteration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CanvasMetrics {
    pub candidates_seen: usize,
    pub newly_committed: usize,
    pub remaining_masked: usize,
    pub remaining_provisional: usize,
}

/// Mutable canvas state. It contains token IDs and scalar confidence metadata,
/// never tensors, buffers, weights, or CUDA resources.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaCanvas {
    positions: Vec<CanvasPosition>,
    iteration: u32,
}

impl GemmaCanvas {
    pub fn new() -> Self {
        Self {
            positions: vec![
                CanvasPosition {
                    token_id: None,
                    confidence: 0.0,
                    state: CanvasTokenState::Masked,
                };
                GEMMA_CANVAS_TOKENS
            ],
            iteration: 0,
        }
    }

    pub fn with_length(length: usize) -> Result<Self, CanvasError> {
        if length == 0 {
            return Err(CanvasError::EmptyCanvas);
        }
        Ok(Self {
            positions: vec![
                CanvasPosition {
                    token_id: None,
                    confidence: 0.0,
                    state: CanvasTokenState::Masked,
                };
                length
            ],
            iteration: 0,
        })
    }

    pub fn positions(&self) -> &[CanvasPosition] {
        &self.positions
    }

    pub const fn iteration(&self) -> u32 {
        self.iteration
    }

    pub fn committed_count(&self) -> usize {
        self.positions
            .iter()
            .filter(|p| p.state == CanvasTokenState::Committed)
            .count()
    }

    pub fn is_complete(&self) -> bool {
        self.positions
            .iter()
            .all(|p| p.state == CanvasTokenState::Committed)
    }

    /// Apply a complete set of candidate updates for one denoising iteration.
    ///
    /// Duplicate positions, invalid confidence values and updates to committed
    /// positions are rejected before any state is mutated (all-or-nothing).
    pub fn apply_iteration<I>(
        &mut self,
        candidates: I,
        commit_threshold: f32,
    ) -> Result<CanvasMetrics, CanvasError>
    where
        I: IntoIterator<Item = (usize, u32, f32)>,
    {
        let accepted = vec![true; self.positions.len()];
        self.apply_iteration_with_mask(candidates, &accepted, commit_threshold)
    }

    /// Apply one iteration with an explicit acceptance mask.
    ///
    /// The mask carries the reference acceptance decision (for DiffusionGemma:
    /// the entropy-bound prefix of the pinned sampler). A position is committed
    /// only when the mask accepts it *and* its confidence reaches the
    /// threshold; every other candidate stays provisional and is re-noised by
    /// the adapter.
    pub fn apply_iteration_with_mask<I>(
        &mut self,
        candidates: I,
        accepted: &[bool],
        commit_threshold: f32,
    ) -> Result<CanvasMetrics, CanvasError>
    where
        I: IntoIterator<Item = (usize, u32, f32)>,
    {
        if !commit_threshold.is_finite() || !(0.0..=1.0).contains(&commit_threshold) {
            return Err(CanvasError::InvalidThreshold);
        }
        if accepted.len() != self.positions.len() {
            return Err(CanvasError::AcceptanceMaskLength {
                expected: self.positions.len(),
                actual: accepted.len(),
            });
        }

        let next_iteration = self
            .iteration
            .checked_add(1)
            .ok_or(CanvasError::IterationOverflow)?;
        let updates: Vec<_> = candidates.into_iter().collect();
        let mut seen = std::collections::BTreeSet::new();

        for (index, _, confidence) in &updates {
            if *index >= self.positions.len() {
                return Err(CanvasError::PositionOutOfBounds {
                    index: *index,
                    length: self.positions.len(),
                });
            }
            if !seen.insert(*index) {
                return Err(CanvasError::DuplicatePosition(*index));
            }
            if !confidence.is_finite() || !(0.0..=1.0).contains(confidence) {
                return Err(CanvasError::InvalidConfidence(*index));
            }
            if self.positions[*index].state == CanvasTokenState::Committed {
                return Err(CanvasError::AlreadyCommitted(*index));
            }
        }

        let mut metrics = CanvasMetrics {
            candidates_seen: updates.len(),
            ..CanvasMetrics::default()
        };

        for (index, token_id, confidence) in updates {
            let position = &mut self.positions[index];
            position.confidence = confidence;
            if accepted[index] && confidence >= commit_threshold {
                position.token_id = Some(token_id);
                position.state = CanvasTokenState::Committed;
                metrics.newly_committed += 1;
            } else {
                // The model adapter is responsible for re-noising and producing
                // the next candidate; no stale token is retained here.
                position.token_id = None;
                position.state = CanvasTokenState::Provisional;
            }
        }

        self.iteration = next_iteration;
        metrics.remaining_masked = self
            .positions
            .iter()
            .filter(|p| p.state == CanvasTokenState::Masked)
            .count();
        metrics.remaining_provisional = self
            .positions
            .iter()
            .filter(|p| p.state == CanvasTokenState::Provisional)
            .count();
        Ok(metrics)
    }
}

impl Default for GemmaCanvas {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CanvasError {
    #[error("canvas must contain at least one position")]
    EmptyCanvas,
    #[error("confidence threshold must be finite and within [0, 1]")]
    InvalidThreshold,
    #[error("acceptance mask has {actual} entries, expected {expected}")]
    AcceptanceMaskLength { expected: usize, actual: usize },
    #[error("candidate position {index} is outside canvas length {length}")]
    PositionOutOfBounds { index: usize, length: usize },
    #[error("candidate position {0} appears more than once in an iteration")]
    DuplicatePosition(usize),
    #[error("candidate confidence at position {0} must be finite and within [0, 1]")]
    InvalidConfidence(usize),
    #[error("canvas position {0} is already committed")]
    AlreadyCommitted(usize),
    #[error("canvas iteration counter overflow")]
    IterationOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_canvas_has_audited_length_and_starts_masked() {
        let canvas = GemmaCanvas::new();
        assert_eq!(canvas.positions().len(), 256);
        assert_eq!(canvas.committed_count(), 0);
        assert!(!canvas.is_complete());
        assert!(canvas
            .positions()
            .iter()
            .all(|p| p.state() == CanvasTokenState::Masked));
    }

    #[test]
    fn iteration_commits_only_confident_positions() {
        let mut canvas = GemmaCanvas::with_length(3).unwrap();
        let metrics = canvas
            .apply_iteration([(0, 11, 0.95), (1, 12, 0.4)], 0.8)
            .unwrap();
        assert_eq!(metrics.newly_committed, 1);
        assert_eq!(metrics.remaining_provisional, 1);
        assert_eq!(canvas.positions()[0].token_id(), Some(11));
        assert_eq!(canvas.positions()[1].token_id(), None);
        assert_eq!(canvas.iteration(), 1);
    }

    #[test]
    fn invalid_batch_is_rejected_without_partial_mutation() {
        let mut canvas = GemmaCanvas::with_length(2).unwrap();
        let before = canvas.clone();
        assert_eq!(
            canvas.apply_iteration([(0, 1, 0.9), (0, 2, 0.8)], 0.7),
            Err(CanvasError::DuplicatePosition(0))
        );
        assert_eq!(canvas, before);
    }

    #[test]
    fn committed_positions_cannot_be_overwritten() {
        let mut canvas = GemmaCanvas::with_length(1).unwrap();
        canvas.apply_iteration([(0, 4, 1.0)], 0.9).unwrap();
        assert_eq!(
            canvas.apply_iteration([(0, 5, 1.0)], 0.9),
            Err(CanvasError::AlreadyCommitted(0))
        );
        assert_eq!(canvas.positions()[0].token_id(), Some(4));
    }

    #[test]
    fn invalid_confidence_and_threshold_are_rejected() {
        let mut canvas = GemmaCanvas::with_length(1).unwrap();
        assert_eq!(
            canvas.apply_iteration([(0, 1, f32::NAN)], 0.5),
            Err(CanvasError::InvalidConfidence(0))
        );
        assert_eq!(
            canvas.apply_iteration([], f32::INFINITY),
            Err(CanvasError::InvalidThreshold)
        );
        assert_eq!(canvas.iteration(), 0);
    }
}
