//! M9 `DiffusionExecutor` adapter for the DiffusionGemma graph.
//!
//! The adapter owns the *policy* (canvas state, temperature schedule,
//! entropy-bound acceptance, re-noising, adaptive stopping) while the model
//! backend owns the *computation*. Every backend call is bounded to one M9
//! step and must return only after the kernels and transfers it started have
//! completed: the adapter never reports a step as finished on the sole basis
//! of an enqueue.
//!
//! No tensor, weight or CUDA handle is ever placed in an M9 scheduler payload:
//! the scheduler only carries step descriptors, while the canvas and the
//! backend state stay inside this adapter.

use claymore_runtime::diffusion::{DiffusionExecutor, DiffusionStep};
use claymore_runtime::CancellationToken;
use thiserror::Error;

use crate::canvas::{CanvasError, GemmaCanvas};
use crate::generation::{GenerationConfigError, GenerationPolicy};
use crate::rng::SplitMix64;
use crate::sampling::{
    argmax_rows, mean_entropy, sample_rows, token_entropies, EntropyBoundSampler, SamplingError,
    StableAndConfident,
};

/// Identifier of a diffusion block, borrowed from the M9 runtime.
pub type BlockId = claymore_runtime::diffusion::BlockId;

/// Read-only view of the block state handed to the backend.
#[derive(Debug, Clone)]
pub struct BlockState<'a> {
    pub block: BlockId,
    pub iteration: u32,
    /// Current canvas tokens (re-noised between iterations).
    pub canvas: &'a [u32],
    /// Latest argmax prediction; this is the canvas emitted by the reference.
    pub argmax_canvas: &'a [u32],
    /// Number of positions accepted by the entropy-bound rule so far.
    pub accepted_positions: usize,
    /// Mean entropy of the last processed logits.
    pub mean_entropy: f32,
}

/// Model-side computation boundary.
///
/// Implementations own the weights, the device buffers and the KV cache. They
/// must return only when the step's work is complete and safe to release.
pub trait DiffusionGemmaBackend {
    /// Prepare a block: encode the prompt, build the KV cache and allocate the
    /// bounded working set.
    fn prepare_block(&mut self, block: BlockId) -> Result<(), GemmaExecutorError>;

    /// Run one denoising iteration and return the FP32 logits of the canvas,
    /// shaped `[canvas_length, vocabulary]` in row-major order.
    fn denoise(
        &mut self,
        block: BlockId,
        iteration: u32,
        canvas: &[u32],
    ) -> Result<Vec<f32>, GemmaExecutorError>;

    /// Validate the block before it is committed.
    fn validate_block(
        &mut self,
        block: BlockId,
        state: &BlockState<'_>,
    ) -> Result<(), GemmaExecutorError>;

    /// Commit the block and return the emitted tokens.
    fn commit_block(
        &mut self,
        block: BlockId,
        state: &BlockState<'_>,
    ) -> Result<Vec<u32>, GemmaExecutorError>;
}

/// Per-iteration diagnostics exposed to the audit harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationMetrics {
    pub iteration: u32,
    pub remaining_step: u32,
    pub temperature: f32,
    pub mean_entropy: f32,
    pub accepted: usize,
    pub committed: usize,
    pub finished: bool,
}

/// A committed block and the tokens it emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedBlock {
    pub block: BlockId,
    pub tokens: Vec<u32>,
}

/// Typed failures raised by the adapter.
#[derive(Debug, Error)]
pub enum GemmaExecutorError {
    #[error("backend failure: {0}")]
    Backend(String),
    #[error("sampling policy failure: {0}")]
    Sampling(#[from] SamplingError),
    #[error("canvas failure: {0}")]
    Canvas(#[from] CanvasError),
    #[error("generation policy failure: {0}")]
    Generation(#[from] GenerationConfigError),
    #[error("cancelled at a safe step boundary")]
    Cancelled,
    #[error("invalid executor state: {0}")]
    InvalidState(String),
}

/// DiffusionGemma adapter implementing the M9 executor contract.
pub struct GemmaDiffusionExecutor<B> {
    backend: B,
    policy: GenerationPolicy,
    sampler: EntropyBoundSampler,
    stopping: StableAndConfident,
    rng: SplitMix64,
    canvas: GemmaCanvas,
    tokens: Vec<u32>,
    argmax_canvas: Vec<u32>,
    vocabulary: usize,
    canvas_length: usize,
    commit_threshold: f32,
    iteration: u32,
    current_block: Option<BlockId>,
    prepared: bool,
    validated: bool,
    finished: bool,
    mean_entropy: f32,
    accepted_positions: usize,
    metrics: Vec<IterationMetrics>,
    committed: Vec<CommittedBlock>,
    cancellation: CancellationToken,
    last_error: Option<GemmaExecutorError>,
}

impl<B> GemmaDiffusionExecutor<B> {
    /// Build an executor with the pinned generation policy.
    pub fn new(
        backend: B,
        policy: GenerationPolicy,
        vocabulary: usize,
        canvas_length: usize,
        seed: u64,
    ) -> Result<Self, GemmaExecutorError> {
        if vocabulary == 0 {
            return Err(GemmaExecutorError::InvalidState(
                "vocabulary must be non-zero".to_owned(),
            ));
        }
        if canvas_length == 0 {
            return Err(GemmaExecutorError::InvalidState(
                "canvas length must be non-zero".to_owned(),
            ));
        }
        let sampler = EntropyBoundSampler::from_policy(&policy, canvas_length, vocabulary);
        let stopping = StableAndConfident::from_policy(&policy);
        Ok(Self {
            backend,
            policy,
            sampler,
            stopping,
            rng: SplitMix64::new(seed),
            canvas: GemmaCanvas::with_length(canvas_length)?,
            tokens: vec![0; canvas_length],
            argmax_canvas: vec![0; canvas_length],
            vocabulary,
            canvas_length,
            commit_threshold: 0.0,
            iteration: 0,
            current_block: None,
            prepared: false,
            validated: false,
            finished: false,
            mean_entropy: 0.0,
            accepted_positions: 0,
            metrics: Vec::new(),
            committed: Vec::new(),
            cancellation: CancellationToken::new(),
            last_error: None,
        })
    }

    /// Additional confidence gate applied on top of the reference acceptance
    /// rule. The pinned policy commits exactly the entropy-bound prefix, so the
    /// default threshold is `0.0`.
    pub fn with_commit_threshold(mut self, threshold: f32) -> Result<Self, GemmaExecutorError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(GemmaExecutorError::InvalidState(format!(
                "commit threshold {threshold} must be finite and within [0, 1]"
            )));
        }
        self.commit_threshold = threshold;
        Ok(self)
    }

    /// Share a cancellation token with the runtime.
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn canvas(&self) -> &GemmaCanvas {
        &self.canvas
    }

    pub fn current_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn argmax_canvas(&self) -> &[u32] {
        &self.argmax_canvas
    }

    pub fn iteration_metrics(&self) -> &[IterationMetrics] {
        &self.metrics
    }

    pub fn committed_blocks(&self) -> &[CommittedBlock] {
        &self.committed
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    pub fn last_error(&self) -> Option<&GemmaExecutorError> {
        self.last_error.as_ref()
    }

    pub fn policy(&self) -> &GenerationPolicy {
        &self.policy
    }
}

impl<B: DiffusionGemmaBackend> GemmaDiffusionExecutor<B> {
    fn prepare(&mut self, block: BlockId) -> Result<(), GemmaExecutorError> {
        if self.current_block == Some(block) && self.prepared {
            return Err(GemmaExecutorError::InvalidState(format!(
                "block {block:?} is already prepared"
            )));
        }
        self.backend.prepare_block(block)?;
        // The reference initializes the canvas with uniform random tokens.
        self.tokens = self.sampler.initialize_canvas(&mut self.rng);
        self.argmax_canvas = self.tokens.clone();
        self.canvas = GemmaCanvas::with_length(self.canvas_length)?;
        self.iteration = 0;
        self.prepared = true;
        self.validated = false;
        self.finished = false;
        self.mean_entropy = 0.0;
        self.accepted_positions = 0;
        self.metrics.clear();
        self.sampler =
            EntropyBoundSampler::from_policy(&self.policy, self.canvas_length, self.vocabulary);
        self.stopping = StableAndConfident::from_policy(&self.policy);
        self.current_block = Some(block);
        Ok(())
    }

    fn denoise(&mut self, block: BlockId, iteration: u32) -> Result<(), GemmaExecutorError> {
        if !self.prepared || self.current_block != Some(block) {
            return Err(GemmaExecutorError::InvalidState(format!(
                "block {block:?} is not prepared"
            )));
        }
        if iteration != self.iteration {
            return Err(GemmaExecutorError::InvalidState(format!(
                "expected denoising iteration {}, got {iteration}",
                self.iteration
            )));
        }
        let logits = self.backend.denoise(block, iteration, &self.tokens)?;
        let expected = self
            .canvas_length
            .checked_mul(self.vocabulary)
            .ok_or_else(|| {
                GemmaExecutorError::InvalidState("canvas logits shape overflows usize".to_owned())
            })?;
        if logits.len() != expected {
            return Err(GemmaExecutorError::InvalidState(format!(
                "backend returned {} logits, expected {expected}",
                logits.len()
            )));
        }

        // The reference iterates the reverse diffusion schedule from
        // `max_denoising_steps` down to 1.
        let remaining_step = self
            .policy
            .max_denoising_steps
            .checked_sub(iteration)
            .ok_or_else(|| {
                GemmaExecutorError::InvalidState(
                    "denoising iteration exceeds the schedule".to_owned(),
                )
            })?;
        let temperature = self.policy.temperature(remaining_step)?;
        let processed = crate::sampling::apply_temperature(&logits, self.vocabulary, temperature)?;
        let entropies = token_entropies(&processed, self.vocabulary, self.canvas_length)?;
        let sampled = sample_rows(
            &processed,
            self.vocabulary,
            self.canvas_length,
            &mut self.rng,
        )?;
        let argmax = argmax_rows(&processed, self.vocabulary, self.canvas_length)?;
        let acceptance = self.sampler.accept(&entropies)?;

        // Confidence is the largest temperature-scaled probability: it is the
        // natural per-position score in [0, 1] and is reported next to the
        // reference acceptance decision.
        let candidates: Vec<(usize, u32, f32)> = (0..self.canvas_length)
            .map(|index| {
                let row = &processed[index * self.vocabulary..(index + 1) * self.vocabulary];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|value| (*value - max).exp()).sum();
                let probability = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                (index, sampled[index], probability.clamp(0.0, 1.0))
            })
            .collect();
        let metrics = self.canvas.apply_iteration_with_mask(
            candidates,
            &acceptance.mask,
            self.commit_threshold,
        )?;
        self.accepted_positions = self
            .canvas
            .positions()
            .iter()
            .filter(|position| position.state() == crate::canvas::CanvasTokenState::Committed)
            .count();

        // The reference keeps the denoiser token at accepted positions and
        // re-noises every other position uniformly.
        let accepted_canvas = self.sampler.accept_canvas(&self.tokens, &sampled)?;
        self.tokens = self.sampler.renoise(&accepted_canvas, &mut self.rng)?;
        self.argmax_canvas = argmax;

        let mean = mean_entropy(&entropies)?;
        self.mean_entropy = mean;
        let finished = self.stopping.evaluate(&self.argmax_canvas, mean)?;
        self.finished = finished;
        self.metrics.push(IterationMetrics {
            iteration,
            remaining_step,
            temperature,
            mean_entropy: mean,
            accepted: acceptance.accepted_count,
            committed: metrics.newly_committed,
            finished,
        });
        self.iteration = self.iteration.saturating_add(1);
        Ok(())
    }

    fn validate(&mut self, block: BlockId) -> Result<(), GemmaExecutorError> {
        if !self.prepared || self.current_block != Some(block) {
            return Err(GemmaExecutorError::InvalidState(format!(
                "block {block:?} is not prepared"
            )));
        }
        if self.validated {
            return Err(GemmaExecutorError::InvalidState(format!(
                "block {block:?} is already validated"
            )));
        }
        let state = BlockState {
            block,
            iteration: self.iteration,
            canvas: &self.tokens,
            argmax_canvas: &self.argmax_canvas,
            accepted_positions: self.accepted_positions,
            mean_entropy: self.mean_entropy,
        };
        self.backend.validate_block(block, &state)?;
        self.validated = true;
        Ok(())
    }

    fn commit(&mut self, block: BlockId) -> Result<(), GemmaExecutorError> {
        if !self.validated || self.current_block != Some(block) {
            return Err(GemmaExecutorError::InvalidState(format!(
                "block {block:?} is not validated"
            )));
        }
        let state = BlockState {
            block,
            iteration: self.iteration,
            canvas: &self.tokens,
            argmax_canvas: &self.argmax_canvas,
            accepted_positions: self.accepted_positions,
            mean_entropy: self.mean_entropy,
        };
        let tokens = self.backend.commit_block(block, &state)?;
        self.committed.push(CommittedBlock { block, tokens });
        self.prepared = false;
        self.validated = false;
        Ok(())
    }
}

impl<B: DiffusionGemmaBackend> DiffusionExecutor for GemmaDiffusionExecutor<B> {
    fn execute_step(&mut self, step: DiffusionStep) -> Result<(), String> {
        if self.cancellation.is_cancelled() {
            let error = GemmaExecutorError::Cancelled;
            self.last_error = Some(GemmaExecutorError::Cancelled);
            return Err(error.to_string());
        }
        let outcome = match step {
            DiffusionStep::PrepareBlock { block } => self.prepare(block),
            DiffusionStep::Denoise { block, iteration } => self.denoise(block, iteration),
            DiffusionStep::ValidateBlock { block } => self.validate(block),
            DiffusionStep::CommitBlock { block } => self.commit(block),
        };
        match outcome {
            Ok(()) => Ok(()),
            Err(error) => {
                let message = error.to_string();
                self.last_error = Some(error);
                Err(message)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::CanvasTokenState;
    use claymore_runtime::diffusion::BlockId;

    /// Deterministic synthetic backend: the logits encode a peaked
    /// distribution whose peak token depends on the canvas content.
    struct SyntheticBackend {
        vocabulary: usize,
        canvas_length: usize,
        prepared: bool,
        validated: bool,
        committed: bool,
        fail_at: Option<&'static str>,
        calls: Vec<&'static str>,
    }

    impl SyntheticBackend {
        fn new(vocabulary: usize, canvas_length: usize) -> Self {
            Self {
                vocabulary,
                canvas_length,
                prepared: false,
                validated: false,
                committed: false,
                fail_at: None,
                calls: Vec::new(),
            }
        }

        fn logits(&self, canvas: &[u32]) -> Vec<f32> {
            let mut logits = vec![0.0f32; self.canvas_length * self.vocabulary];
            for (index, token) in canvas.iter().enumerate() {
                let row = &mut logits[index * self.vocabulary..(index + 1) * self.vocabulary];
                row[(*token as usize) % self.vocabulary] = 20.0;
            }
            logits
        }
    }

    impl DiffusionGemmaBackend for SyntheticBackend {
        fn prepare_block(&mut self, _block: BlockId) -> Result<(), GemmaExecutorError> {
            self.calls.push("prepare");
            if self.fail_at == Some("prepare") {
                return Err(GemmaExecutorError::Backend("injected".to_owned()));
            }
            self.prepared = true;
            Ok(())
        }

        fn denoise(
            &mut self,
            _block: BlockId,
            _iteration: u32,
            canvas: &[u32],
        ) -> Result<Vec<f32>, GemmaExecutorError> {
            self.calls.push("denoise");
            if self.fail_at == Some("denoise") {
                return Err(GemmaExecutorError::Backend("injected".to_owned()));
            }
            Ok(self.logits(canvas))
        }

        fn validate_block(
            &mut self,
            _block: BlockId,
            _state: &BlockState<'_>,
        ) -> Result<(), GemmaExecutorError> {
            self.calls.push("validate");
            if self.fail_at == Some("validate") {
                return Err(GemmaExecutorError::Backend("injected".to_owned()));
            }
            self.validated = true;
            Ok(())
        }

        fn commit_block(
            &mut self,
            _block: BlockId,
            state: &BlockState<'_>,
        ) -> Result<Vec<u32>, GemmaExecutorError> {
            self.calls.push("commit");
            if self.fail_at == Some("commit") {
                return Err(GemmaExecutorError::Backend("injected".to_owned()));
            }
            self.committed = true;
            Ok(state.argmax_canvas.to_vec())
        }
    }

    fn policy() -> GenerationPolicy {
        GenerationPolicy::new(4, 16, vec![1], 0, 0.1, 0.005, 1, 0.8, 0.4).unwrap()
    }

    fn executor() -> GemmaDiffusionExecutor<SyntheticBackend> {
        GemmaDiffusionExecutor::new(SyntheticBackend::new(8, 4), policy(), 8, 4, 7).unwrap()
    }

    #[test]
    fn denoising_commits_the_entropy_bound_prefix_and_renoises_the_rest() {
        let mut executor = executor();
        let block = BlockId(0);
        executor
            .execute_step(DiffusionStep::PrepareBlock { block })
            .unwrap();
        assert_eq!(executor.canvas().committed_count(), 0);

        executor
            .execute_step(DiffusionStep::Denoise {
                block,
                iteration: 0,
            })
            .unwrap();
        let metrics = executor.iteration_metrics();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].remaining_step, 4);
        assert_eq!(metrics[0].temperature, 0.8);
        assert!(metrics[0].accepted >= 1);
        assert_eq!(executor.canvas().committed_count(), metrics[0].committed);
        // Non accepted positions stay provisional and are re-noised.
        assert_eq!(
            executor.canvas().committed_count()
                + executor
                    .canvas()
                    .positions()
                    .iter()
                    .filter(|p| p.state() == CanvasTokenState::Provisional)
                    .count(),
            4
        );
    }

    #[test]
    fn the_reference_emits_the_argmax_canvas() {
        let mut executor = executor();
        let block = BlockId(0);
        executor
            .execute_step(DiffusionStep::PrepareBlock { block })
            .unwrap();
        executor
            .execute_step(DiffusionStep::Denoise {
                block,
                iteration: 0,
            })
            .unwrap();
        executor
            .execute_step(DiffusionStep::ValidateBlock { block })
            .unwrap();
        executor
            .execute_step(DiffusionStep::CommitBlock { block })
            .unwrap();
        let committed = executor.committed_blocks();
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].tokens, executor.argmax_canvas());
    }

    #[test]
    fn steps_out_of_order_are_refused_with_typed_errors() {
        let mut executor = executor();
        let block = BlockId(0);
        let error = executor
            .execute_step(DiffusionStep::Denoise {
                block,
                iteration: 0,
            })
            .unwrap_err();
        assert!(error.contains("not prepared"), "{error}");

        executor
            .execute_step(DiffusionStep::PrepareBlock { block })
            .unwrap();
        let error = executor
            .execute_step(DiffusionStep::Denoise {
                block,
                iteration: 3,
            })
            .unwrap_err();
        assert!(error.contains("expected denoising iteration 0"), "{error}");
    }

    #[test]
    fn backend_failures_are_typed_and_recorded() {
        let mut backend = SyntheticBackend::new(8, 4);
        backend.fail_at = Some("denoise");
        let mut executor = GemmaDiffusionExecutor::new(backend, policy(), 8, 4, 7).unwrap();
        let block = BlockId(0);
        executor
            .execute_step(DiffusionStep::PrepareBlock { block })
            .unwrap();
        let error = executor
            .execute_step(DiffusionStep::Denoise {
                block,
                iteration: 0,
            })
            .unwrap_err();
        assert!(error.contains("backend failure"), "{error}");
        assert!(matches!(
            executor.last_error(),
            Some(GemmaExecutorError::Backend(_))
        ));
    }

    #[test]
    fn cancellation_is_observed_at_the_step_boundary() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let mut executor = executor().with_cancellation(cancellation);
        let error = executor
            .execute_step(DiffusionStep::PrepareBlock { block: BlockId(0) })
            .unwrap_err();
        assert!(error.contains("cancelled"), "{error}");
    }

    #[test]
    fn the_whole_block_runs_through_the_m9_orchestrator() {
        use claymore_runtime::diffusion::{DiffusionOrchestrator, DiffusionPlan};
        use claymore_scheduler::{SchedulerConfig, TaskPriority};

        let mut executor = executor();
        let plan = DiffusionPlan::new(vec![
            DiffusionStep::PrepareBlock { block: BlockId(0) },
            DiffusionStep::Denoise {
                block: BlockId(0),
                iteration: 0,
            },
            DiffusionStep::ValidateBlock { block: BlockId(0) },
            DiffusionStep::CommitBlock { block: BlockId(0) },
        ])
        .unwrap();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let session = orchestrator
            .submit(plan, CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        let mut outcomes = Vec::new();
        loop {
            let outcome = orchestrator.run_one(&mut executor).unwrap();
            let terminal = matches!(
                outcome,
                claymore_runtime::diffusion::DiffusionRunOutcome::SessionCompleted { .. }
            );
            outcomes.push(outcome);
            if terminal {
                break;
            }
        }
        assert_eq!(outcomes.len(), 4);
        assert_eq!(
            orchestrator.session(session).unwrap().state,
            claymore_runtime::diffusion::DiffusionSessionState::Completed
        );
        assert_eq!(executor.committed_blocks().len(), 1);
        assert!(orchestrator.metrics().steps_completed == 4);
    }
}
