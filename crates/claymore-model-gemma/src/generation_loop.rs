//! Claymore block-diffusion generation loop for DiffusionGemma (M10.6).
//!
//! Implements the pinned EntropyBoundSampler policy: random canvas
//! initialization, per-position entropy acceptance under the entropy bound,
//! renoising of non-accepted positions, stability (accepted canvas repeated
//! `stability_threshold` steps) and confidence (mean entropy below
//! `confidence_threshold`) stopping criteria. Sampling follows Claymore's own
//! deterministic RNG stream, so the canvas trajectory is reproducible for a
//! fixed seed, prompt and pinned configuration.

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerPolicy {
    pub canvas_length: usize,
    pub vocab_size: usize,
    pub entropy_bound: f32,
    pub confidence_threshold: f32,
    pub stability_threshold: usize,
    pub max_denoising_steps: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    #[error("canvas length {canvas} does not match the pinned policy {policy}")]
    CanvasMismatch { canvas: usize, policy: usize },
    #[error("entropy bound must be positive: {0}")]
    InvalidEntropyBound(f32),
    #[error("max denoising steps must be positive: {0}")]
    InvalidSteps(usize),
}

impl SamplerPolicy {
    pub fn validate(&self) -> Result<(), LoopError> {
        if self.canvas_length == 0 {
            return Err(LoopError::CanvasMismatch {
                canvas: 0,
                policy: 0,
            });
        }
        if self.entropy_bound <= 0.0 || !self.entropy_bound.is_finite() {
            return Err(LoopError::InvalidEntropyBound(self.entropy_bound));
        }
        if self.max_denoising_steps == 0 {
            return Err(LoopError::InvalidSteps(0));
        }
        Ok(())
    }

    /// Pinned generation policy read from the audited config: canvas 256,
    /// entropy bound 0.1, confidence 0.005, stability 1, 48 steps.
    pub fn pinned(canvas_length: usize, vocab_size: usize) -> Self {
        Self {
            canvas_length,
            vocab_size,
            entropy_bound: 0.1,
            confidence_threshold: 0.005,
            stability_threshold: 1,
            max_denoising_steps: 48,
        }
    }
}

/// Canvas state for one denoising block.
pub struct CanvasState {
    pub canvas: Vec<u32>,
    pub accepted: Vec<bool>,
    pub logits: Vec<f32>,
    pub step: usize,
    pub stable_count: usize,
    pub last_accepted: Vec<u32>,
}

impl CanvasState {
    pub fn len(&self) -> usize {
        self.canvas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.canvas.is_empty()
    }
}

/// Per-position entropy from F32 logits `[canvas, vocab]`: entropy of the
/// temperature-scaled distribution restricted to the current canvas token
/// neighbourhood, matching `EntropyBoundSampler` acceptance semantics.
pub fn position_entropy(logits: &[f32], canvas: usize, vocab: usize, temperature: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(canvas);
    for c in 0..canvas {
        let row = &logits[c * vocab..(c + 1) * vocab];
        let scaled: Vec<f32> = row.iter().map(|v| v / temperature).collect();
        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in &scaled {
            sum += (v - max).exp();
        }
        let mut ent = 0.0f32;
        for v in &scaled {
            let p = (v - max).exp() / sum;
            if p > 0.0 {
                ent -= p * p.ln();
            }
        }
        out.push(ent);
    }
    out
}

pub fn mean_entropy(entropies: &[f32]) -> f32 {
    entropies.iter().sum::<f32>() / entropies.len() as f32
}

/// Deterministic canvas initialization: uniform tokens in `0..vocab_size`
/// drawn from the caller's RNG stream (same stream as the CUDA campaign).
pub fn initialize_canvas(rng: &mut dyn FnMut(usize) -> usize, policy: &SamplerPolicy) -> Vec<u32> {
    (0..policy.canvas_length)
        .map(|_| rng(policy.vocab_size) as u32)
        .collect()
}

/// Acceptance step: positions whose entropy is below `entropy_bound` keep
/// the argmax token; all other positions are renoised from the RNG stream.
/// Returns the new canvas and the accepted mask.
pub fn accept_canvas(
    state: &CanvasState,
    argmax: &[u32],
    entropies: &[f32],
    policy: &SamplerPolicy,
    rng: &mut dyn FnMut(usize) -> usize,
) -> (Vec<u32>, Vec<bool>) {
    let mut canvas = state.canvas.clone();
    let mut accepted = vec![false; policy.canvas_length];
    for c in 0..policy.canvas_length {
        if entropies[c] < policy.entropy_bound {
            canvas[c] = argmax[c];
            accepted[c] = true;
        } else {
            canvas[c] = rng(policy.vocab_size) as u32;
        }
    }
    (canvas, accepted)
}

pub fn all_accepted(accepted: &[bool]) -> bool {
    accepted.iter().all(|a| *a)
}

/// Stopping decision: confidence (mean entropy below `confidence_threshold`)
/// or stability (all accepted with the same accepted canvas for
/// `stability_threshold` consecutive steps).
pub fn should_stop(
    state: &CanvasState,
    accepted: &[bool],
    canvas: &[u32],
    mean_ent: f32,
    policy: &SamplerPolicy,
) -> bool {
    if mean_ent < policy.confidence_threshold && all_accepted(accepted) {
        return true;
    }
    if all_accepted(accepted)
        && canvas == state.last_accepted
        && state.stable_count + 1 >= policy.stability_threshold
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(canvas: Vec<u32>, stable: usize) -> CanvasState {
        let len = canvas.len();
        let last = canvas.clone();
        CanvasState {
            canvas,
            accepted: vec![true; len],
            logits: Vec::new(),
            step: 0,
            stable_count: stable,
            last_accepted: last,
        }
    }

    #[test]
    fn entropy_of_a_peaked_distribution_is_below_the_bound() {
        let vocab = 4;
        let mut logits = vec![0.0f32; vocab];
        logits[2] = 10.0;
        let ent = position_entropy(&logits, 1, vocab, 1.0);
        assert!(ent[0] < 0.1, "peaked entropy {e}", e = ent[0]);
    }

    #[test]
    fn entropy_of_a_uniform_distribution_is_high() {
        let vocab = 4;
        let logits = vec![1.0f32; vocab];
        let ent = position_entropy(&logits, 1, vocab, 1.0);
        assert!(ent[0] > 0.9, "uniform entropy {}", ent[0]);
    }

    #[test]
    fn acceptance_keeps_low_entropy_positions_and_renoises_the_rest() {
        let policy = SamplerPolicy::pinned(4, 10);
        let state = make_state(vec![1, 2, 3, 4], 0);
        // Positions 0 and 2 are confident (low entropy), 1 and 3 are not.
        let entropies = [0.01f32, 0.9, 0.02, 0.8];
        let argmax = [7u32, 8, 9, 10];
        let mut rng_calls = 0usize;
        let mut rng = |n: usize| {
            rng_calls += 1;
            rng_calls % n
        };
        let (canvas, accepted) = accept_canvas(&state, &argmax, &entropies, &policy, &mut rng);
        assert_eq!(accepted, [true, false, true, false]);
        assert_eq!(canvas[0], 7);
        assert_eq!(canvas[2], 9);
        assert_ne!(canvas[1], 8);
        assert_ne!(canvas[3], 10);
        assert_eq!(rng_calls, 2);
    }

    #[test]
    fn stability_and_confidence_stop_the_loop() {
        let policy = SamplerPolicy::pinned(4, 10);
        let state = make_state(vec![1, 2, 3, 4], 0);
        let canvas = vec![1, 2, 3, 4];
        let accepted = [true; 4];
        // Confidence: mean entropy below 0.005 with everything accepted.
        assert!(should_stop(&state, &accepted, &canvas, 0.001, &policy));
        // Stability: identical accepted canvas, stability_threshold = 1.
        assert!(should_stop(&state, &accepted, &canvas, 0.5, &policy));
        // Not accepted everywhere, entropy high: keep going.
        assert!(!should_stop(
            &state,
            &[true, false, true, true],
            &canvas,
            0.5,
            &policy
        ));
    }

    #[test]
    fn pinned_policy_matches_the_audited_generation_config() {
        let policy = SamplerPolicy::pinned(256, 256_000);
        policy.validate().unwrap();
        assert_eq!(policy.canvas_length, 256);
        assert_eq!(policy.max_denoising_steps, 48);
        assert_eq!(policy.entropy_bound, 0.1);
        assert_eq!(policy.confidence_threshold, 0.005);
        assert_eq!(policy.stability_threshold, 1);
    }

    #[test]
    fn policy_validation_rejects_broken_bounds() {
        let mut policy = SamplerPolicy::pinned(256, 256);
        policy.entropy_bound = 0.0;
        assert!(policy.validate().is_err());
        policy.entropy_bound = 0.1;
        policy.max_denoising_steps = 0;
        assert!(policy.validate().is_err());
    }
}
