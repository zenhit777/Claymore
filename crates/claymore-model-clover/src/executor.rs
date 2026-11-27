//! M9 DiffusionExecutor adapter for Clover block LowConfidence generation.
//!
//! The executor owns request-local block state and sampling policy. The graph
//! backend owns model weights, tensor/device buffers, clean GDN/KV state and
//! expert residency. M9's scheduler carries only DiffusionStep/TaskId values.

use claymore_runtime::diffusion::{
    BlockId, DiffusionExecutor, DiffusionSessionId, DiffusionSessionState, DiffusionStep,
};
use thiserror::Error;

use crate::generation::{predict_with_confidence, CloverPenaltyState, CloverRng};
use crate::{
    CloverGenerationError, CloverGenerationOptions, CloverGenerationPlan, CloverGraph,
    CloverGraphBackend, CloverGraphError, CloverInvalidTokenPolicy,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloverGenerationOutput {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub pad_token_id: u32,
}

impl CloverGenerationOutput {
    pub fn sequence(&self) -> Vec<u32> {
        let mut sequence = self.prompt_tokens.clone();
        sequence.extend_from_slice(&self.generated_tokens);
        sequence
    }
}

#[derive(Debug, Error)]
pub enum CloverExecutorError {
    #[error(transparent)]
    Generation(#[from] CloverGenerationError),
    #[error(transparent)]
    Graph(#[from] CloverGraphError),
    #[error("prompt must contain at least one token")]
    EmptyPrompt,
    #[error("prompt token {0} is outside the tokenizer-addressable vocabulary")]
    InvalidPromptToken(u32),
    #[error("prompt plus complete preview blocks exceeds Clover's maximum context")]
    ContextExceeded,
    #[error("unexpected diffusion step {0:?}")]
    UnexpectedStep(DiffusionStep),
    #[error("block {actual:?} does not match active block {expected:?}")]
    BlockMismatch { expected: BlockId, actual: BlockId },
    #[error("block {0:?} has no prepared state")]
    MissingBlock(BlockId),
    #[error("block {0:?} still contains unresolved mask positions")]
    UnresolvedBlock(BlockId),
    #[error("denoising iteration {actual} is not the expected iteration {expected}")]
    IterationMismatch { expected: u32, actual: u32 },
    #[error("backend did not return preview logits")]
    MissingLogits,
    #[error("model predicted token {0}, outside tokenizer-addressable vocabulary")]
    UnusableOutputToken(u32),
    #[error("executor state arithmetic overflow")]
    SizeOverflow,
}

/// One request-local denoising block. The entire configured block is always
/// previewed, even when only a prefix of it is requested by max_new_tokens.
struct ActiveBlock {
    id: BlockId,
    tokens: Vec<u32>,
    pending: Vec<bool>,
    remaining_output: usize,
    next_iteration: u32,
}

/// Clover adapter for the shared M9 DiffusionOrchestrator.
///
/// Use one adapter instance per M9 session. Its backend and clean caches are
/// request-local; concurrent requests must use distinct instances.
pub struct CloverDiffusionExecutor<B: CloverGraphBackend> {
    graph: CloverGraph,
    backend: B,
    prompt: Vec<u32>,
    options: CloverGenerationOptions,
    plan: CloverGenerationPlan,
    terminal_ids: Vec<u32>,
    pad_token_id: u32,
    penalties: CloverPenaltyState,
    rng: CloverRng,
    generated: Vec<u32>,
    committed_prefix_len: usize,
    next_block: u32,
    prefilled: bool,
    stopped: bool,
    active: Option<ActiveBlock>,
}

impl<B: CloverGraphBackend> CloverDiffusionExecutor<B> {
    pub fn new(
        graph: CloverGraph,
        backend: B,
        prompt: Vec<u32>,
        options: CloverGenerationOptions,
    ) -> Result<Self, CloverExecutorError> {
        if prompt.is_empty() {
            return Err(CloverExecutorError::EmptyPrompt);
        }
        options.validate(graph.config())?;
        if let Some(&token) = prompt
            .iter()
            .find(|&&token| token as usize >= options.tokenizer_vocab_size)
        {
            return Err(CloverExecutorError::InvalidPromptToken(token));
        }
        let preview_blocks = options
            .max_new_tokens
            .checked_add(options.block_size - 1)
            .ok_or(CloverExecutorError::SizeOverflow)?
            / options.block_size;
        let preview_tokens = preview_blocks
            .checked_mul(options.block_size)
            .ok_or(CloverExecutorError::SizeOverflow)?;
        if prompt
            .len()
            .checked_add(preview_tokens)
            .ok_or(CloverExecutorError::SizeOverflow)?
            > graph.config().max_position_embeddings
        {
            return Err(CloverExecutorError::ContextExceeded);
        }
        let plan = CloverGenerationPlan::new(graph.config(), &options)?;
        let terminal_ids = options.resolved_terminal_ids(graph.config())?;
        let pad_token_id = options.resolved_pad_token_id(graph.config());
        let penalties = CloverPenaltyState::new(graph.config().vocab_size);
        let rng = CloverRng::new(options.seed);
        Ok(Self {
            graph,
            backend,
            prompt,
            options,
            plan,
            terminal_ids,
            pad_token_id,
            penalties,
            rng,
            generated: Vec::new(),
            committed_prefix_len: 0,
            next_block: 0,
            prefilled: false,
            stopped: false,
            active: None,
        })
    }

    pub fn plan(&self) -> &CloverGenerationPlan {
        &self.plan
    }
    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn output(&self) -> CloverGenerationOutput {
        CloverGenerationOutput {
            prompt_tokens: self.prompt.clone(),
            generated_tokens: self.generated.clone(),
            pad_token_id: self.pad_token_id,
        }
    }

    fn execute_step_inner(&mut self, step: DiffusionStep) -> Result<(), CloverExecutorError> {
        match step {
            DiffusionStep::PrepareBlock { block } => self.prepare_block(block),
            DiffusionStep::Denoise { block, iteration } => self.denoise(block, iteration),
            DiffusionStep::ValidateBlock { block } => self.validate_block(block),
            DiffusionStep::CommitBlock { block } => self.commit_block(block),
        }
    }

    fn prepare_block(&mut self, block: BlockId) -> Result<(), CloverExecutorError> {
        if self.stopped {
            return Ok(());
        }
        if block.0 != self.next_block {
            return Err(CloverExecutorError::UnexpectedStep(
                DiffusionStep::PrepareBlock { block },
            ));
        }
        if self.active.is_some() {
            return Err(CloverExecutorError::UnexpectedStep(
                DiffusionStep::PrepareBlock { block },
            ));
        }
        if !self.prefilled {
            let valid = vec![true; self.prompt.len()];
            let positions: Vec<usize> = (0..self.prompt.len()).collect();
            self.graph
                .forward_causal(&mut self.backend, &self.prompt, &valid, &[], &positions)?;
            self.prefilled = true;
            self.committed_prefix_len = self.prompt.len();
        }
        let remaining = self
            .options
            .max_new_tokens
            .saturating_sub(self.generated.len());
        if remaining == 0 {
            self.stopped = true;
            return Ok(());
        }
        let block_size = self.options.block_size;
        self.active = Some(ActiveBlock {
            id: block,
            tokens: vec![self.graph.config().mask_token_id; block_size],
            pending: vec![true; block_size],
            remaining_output: remaining.min(block_size),
            next_iteration: 0,
        });
        Ok(())
    }

    fn denoise(&mut self, block: BlockId, iteration: u32) -> Result<(), CloverExecutorError> {
        if self.stopped {
            return Ok(());
        }
        let Some(active) = self.active.as_ref() else {
            return Err(CloverExecutorError::MissingBlock(block));
        };
        if active.id != block {
            return Err(CloverExecutorError::BlockMismatch {
                expected: active.id,
                actual: block,
            });
        }
        if active.next_iteration != iteration {
            return Err(CloverExecutorError::IterationMismatch {
                expected: active.next_iteration,
                actual: iteration,
            });
        }
        if !active.pending.iter().any(|pending| *pending) {
            return Ok(());
        }
        let tokens = active.tokens.clone();
        let pending = active.pending.clone();
        let block_size = tokens.len();
        let prefix_valid = vec![true; self.committed_prefix_len];
        let positions: Vec<usize> = (0..block_size)
            .map(|index| self.committed_prefix_len + index)
            .collect();
        let valid = vec![true; block_size];
        let output = self.graph.forward_preview(
            &mut self.backend,
            &tokens,
            &valid,
            &prefix_valid,
            &positions,
        )?;
        let mut logits = output.logits.ok_or(CloverExecutorError::MissingLogits)?;
        let vocab = self.graph.config().vocab_size;
        let mask_id = self.graph.config().mask_token_id;
        for row in logits.chunks_exact_mut(vocab) {
            row[mask_id as usize] = f32::NEG_INFINITY;
        }
        self.penalties.apply(
            &mut logits,
            &tokens,
            mask_id,
            &self.options,
            &self.terminal_ids,
        )?;
        for row in logits.chunks_exact_mut(vocab) {
            row[mask_id as usize] = f32::NEG_INFINITY;
            if self.options.invalid_token_policy == Some(CloverInvalidTokenPolicy::Mask) {
                for value in &mut row[self.options.tokenizer_vocab_size..] {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
        let (predictions, confidence) =
            predict_with_confidence(&logits, block_size, vocab, &self.options, &mut self.rng)?;

        let mut accepted = Vec::new();
        for index in 0..block_size {
            if pending[index] && confidence[index] > self.options.threshold {
                accepted.push(index);
            }
        }
        if accepted.is_empty() {
            let mut best: Option<usize> = None;
            for index in 0..block_size {
                if pending[index]
                    && best.is_none_or(|current| confidence[index] > confidence[current])
                {
                    best = Some(index);
                }
            }
            if let Some(index) = best {
                accepted.push(index);
            }
        }
        let active = self
            .active
            .as_mut()
            .ok_or(CloverExecutorError::MissingBlock(block))?;
        for index in accepted {
            let token = predictions[index];
            if token as usize >= self.options.tokenizer_vocab_size || token == mask_id {
                return Err(CloverExecutorError::UnusableOutputToken(token));
            }
            active.tokens[index] = token;
            active.pending[index] = false;
        }
        active.next_iteration = active
            .next_iteration
            .checked_add(1)
            .ok_or(CloverExecutorError::SizeOverflow)?;
        Ok(())
    }

    fn validate_block(&mut self, block: BlockId) -> Result<(), CloverExecutorError> {
        if self.stopped {
            return Ok(());
        }
        let active = self
            .active
            .as_ref()
            .ok_or(CloverExecutorError::MissingBlock(block))?;
        if active.id != block {
            return Err(CloverExecutorError::BlockMismatch {
                expected: active.id,
                actual: block,
            });
        }
        if active.pending.iter().any(|pending| *pending) {
            return Err(CloverExecutorError::UnresolvedBlock(block));
        }
        if active
            .tokens
            .iter()
            .any(|&token| token as usize >= self.options.tokenizer_vocab_size)
        {
            return Err(CloverExecutorError::UnusableOutputToken(
                *active
                    .tokens
                    .iter()
                    .find(|&&token| token as usize >= self.options.tokenizer_vocab_size)
                    .unwrap(),
            ));
        }
        Ok(())
    }

    fn commit_block(&mut self, block: BlockId) -> Result<(), CloverExecutorError> {
        if self.stopped {
            return Ok(());
        }
        let active = self
            .active
            .take()
            .ok_or(CloverExecutorError::MissingBlock(block))?;
        if active.id != block {
            let expected = active.id;
            self.active = Some(active);
            return Err(CloverExecutorError::BlockMismatch {
                expected,
                actual: block,
            });
        }
        if active.pending.iter().any(|pending| *pending) {
            self.active = Some(active);
            return Err(CloverExecutorError::UnresolvedBlock(block));
        }

        let mut emitted = Vec::with_capacity(active.remaining_output);
        for &token in active.tokens.iter().take(active.remaining_output) {
            emitted.push(token);
            self.generated.push(token);
            if self.terminal_ids.binary_search(&token).is_ok() {
                self.stopped = true;
                break;
            }
        }
        self.penalties.commit(&emitted);

        // Only a fully emitted, nonterminal block becomes the causal prefix for
        // another preview. A final partial block or EOS block needs no replay.
        if !self.stopped && self.generated.len() < self.options.max_new_tokens {
            let prefix_valid = vec![true; self.committed_prefix_len];
            let valid = vec![true; emitted.len()];
            let positions: Vec<usize> = (0..emitted.len())
                .map(|index| self.committed_prefix_len + index)
                .collect();
            self.graph.forward_causal(
                &mut self.backend,
                &emitted,
                &valid,
                &prefix_valid,
                &positions,
            )?;
            self.committed_prefix_len = self
                .committed_prefix_len
                .checked_add(emitted.len())
                .ok_or(CloverExecutorError::SizeOverflow)?;
        }
        self.next_block = self
            .next_block
            .checked_add(1)
            .ok_or(CloverExecutorError::SizeOverflow)?;
        Ok(())
    }
}

impl<B: CloverGraphBackend> DiffusionExecutor for CloverDiffusionExecutor<B> {
    fn execute_step(&mut self, step: DiffusionStep) -> Result<(), String> {
        self.execute_step_inner(step)
            .map_err(|error| error.to_string())
    }

    fn on_session_end(&mut self, _session_id: DiffusionSessionId, _state: DiffusionSessionState) {
        self.backend.reset_session();
        self.active = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CloverActivation, CloverConfig, CloverGenerationOptions, CloverGraphBackend,
        CloverNormKind, LayerType,
    };
    use claymore_runtime::{
        diffusion::{DiffusionOrchestrator, DiffusionRunOutcome},
        CancellationToken,
    };
    use claymore_scheduler::{SchedulerConfig, TaskPriority};
    use claymore_tensor::Bf16;

    const CONFIG: &str = include_str!("../../../docs/audits/fixtures/clover-config.json");

    #[derive(Default)]
    struct SyntheticBackend {
        events: Vec<String>,
        hidden: usize,
        vocab: usize,
        terminal_logit: Option<(usize, f32)>,
        cancel_on_lm_head: Option<CancellationToken>,
        fail_moe_at: Option<usize>,
        reset_count: usize,
    }

    impl CloverGraphBackend for SyntheticBackend {
        fn embed_tokens(&mut self, ids: &[u32]) -> Result<CloverActivation, String> {
            self.events.push(format!("prefill:{}", ids.len()));
            CloverActivation::new(
                ids.len(),
                self.hidden,
                vec![Bf16::from_f32(0.0); ids.len() * self.hidden],
            )
            .map_err(|e| e.to_string())
        }
        fn rms_norm(
            &mut self,
            _layer: usize,
            _kind: CloverNormKind,
            input: &CloverActivation,
        ) -> Result<CloverActivation, String> {
            Ok(input.clone())
        }
        fn forward_gdn(
            &mut self,
            layer: usize,
            input: &CloverActivation,
            _valid: &[bool],
            preview: bool,
        ) -> Result<CloverActivation, String> {
            self.events.push(format!("gdn:{layer}:{preview}"));
            CloverActivation::new(
                input.tokens(),
                input.hidden_size(),
                vec![Bf16::from_f32(0.0); input.values().len()],
            )
            .map_err(|e| e.to_string())
        }
        fn forward_full_attention(
            &mut self,
            layer: usize,
            input: &CloverActivation,
            _valid: &[bool],
            _prefix: &[bool],
            _positions: &[usize],
            preview: bool,
        ) -> Result<CloverActivation, String> {
            self.events.push(format!("attention:{layer}:{preview}"));
            CloverActivation::new(
                input.tokens(),
                input.hidden_size(),
                vec![Bf16::from_f32(0.0); input.values().len()],
            )
            .map_err(|e| e.to_string())
        }
        fn router_logits(
            &mut self,
            _layer: usize,
            input: &CloverActivation,
        ) -> Result<Vec<f32>, String> {
            Ok(vec![0.0; input.tokens() * 4])
        }
        fn forward_moe(
            &mut self,
            layer: usize,
            input: &CloverActivation,
            _routing: &crate::CloverRoutingPlan,
            _valid: &[bool],
            _preview: bool,
        ) -> Result<CloverActivation, String> {
            if self.fail_moe_at == Some(layer) {
                return Err("synthetic expert failure".to_owned());
            }
            CloverActivation::new(
                input.tokens(),
                input.hidden_size(),
                vec![Bf16::from_f32(0.0); input.values().len()],
            )
            .map_err(|e| e.to_string())
        }
        fn final_norm(&mut self, input: &CloverActivation) -> Result<CloverActivation, String> {
            Ok(input.clone())
        }
        fn lm_head(&mut self, input: &CloverActivation) -> Result<Vec<f32>, String> {
            self.events.push(format!("preview:{}", input.tokens()));
            if let Some(token) = &self.cancel_on_lm_head {
                token.cancel();
            }
            let mut logits = vec![0.0; input.tokens() * self.vocab];
            for row in logits.chunks_exact_mut(self.vocab) {
                row[1] = 8.0;
                if let Some((token, value)) = self.terminal_logit {
                    row[token] = value;
                }
            }
            Ok(logits)
        }
        fn reset_session(&mut self) {
            self.reset_count += 1;
        }
    }

    fn tiny_graph() -> CloverGraph {
        let mut c = CloverConfig::from_json(CONFIG).unwrap();
        c.hidden_size = 4;
        c.num_hidden_layers = 2;
        c.layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];
        c.num_experts = 4;
        c.num_experts_per_tok = 2;
        c.vocab_size = 9;
        c.mask_token_id = 7;
        c.eos_token_id = 2;
        c.block_size = 4;
        c.max_position_embeddings = 32;
        c.num_attention_heads = 1;
        c.num_key_value_heads = 1;
        c.linear_num_key_heads = 1;
        c.linear_num_value_heads = 1;
        c.moe_intermediate_size = 2;
        c.shared_expert_intermediate_size = 2;
        CloverGraph::new(c).unwrap()
    }

    fn options(max_new_tokens: usize) -> CloverGenerationOptions {
        CloverGenerationOptions {
            threshold: 1.0,
            block_size: 2,
            invalid_token_policy: Some(CloverInvalidTokenPolicy::Reject),
            ..CloverGenerationOptions::greedy(max_new_tokens, 8)
        }
    }

    #[test]
    fn executes_multiple_lowconfidence_blocks_with_causal_replay() {
        let backend = SyntheticBackend {
            hidden: 4,
            vocab: 9,
            ..Default::default()
        };
        let mut executor =
            CloverDiffusionExecutor::new(tiny_graph(), backend, vec![3], options(3)).unwrap();
        let plan = executor.plan().diffusion_plan().clone();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan, CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        loop {
            match orchestrator.run_one(&mut executor).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Completed
        );
        assert_eq!(executor.generated_tokens(), &[1, 1, 1]);
        assert_eq!(
            executor
                .backend()
                .events
                .iter()
                .filter(|e| e.as_str() == "prefill:1")
                .count(),
            1
        );
        assert_eq!(
            executor
                .backend()
                .events
                .iter()
                .filter(|e| e.starts_with("preview:"))
                .count(),
            4
        );
        assert_eq!(executor.backend().reset_count, 1);
    }

    #[test]
    fn eos_is_checked_in_output_order_after_the_whole_block_resolves() {
        let backend = SyntheticBackend {
            hidden: 4,
            vocab: 9,
            terminal_logit: Some((2, 9.0)),
            ..Default::default()
        };
        let mut opts = options(6);
        opts.eos_token_ids = Some(vec![2]);
        let mut executor =
            CloverDiffusionExecutor::new(tiny_graph(), backend, vec![3], opts).unwrap();
        let plan = executor.plan().diffusion_plan().clone();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        orchestrator
            .submit(plan, CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        loop {
            match orchestrator.run_one(&mut executor).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        assert_eq!(executor.generated_tokens(), &[2]);
        assert_eq!(
            executor
                .backend()
                .events
                .iter()
                .filter(|e| e.starts_with("preview:"))
                .count(),
            2
        );
    }

    #[test]
    fn concurrent_requests_keep_independent_generation_and_backend_state() {
        let mut first = CloverDiffusionExecutor::new(
            tiny_graph(),
            SyntheticBackend {
                hidden: 4,
                vocab: 9,
                ..Default::default()
            },
            vec![3],
            options(2),
        )
        .unwrap();
        let mut second = CloverDiffusionExecutor::new(
            tiny_graph(),
            SyntheticBackend {
                hidden: 4,
                vocab: 9,
                ..Default::default()
            },
            vec![4],
            options(2),
        )
        .unwrap();
        let mut first_runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let mut second_runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let first_id = first_runtime
            .submit(
                first.plan().diffusion_plan().clone(),
                CancellationToken::new(),
                TaskPriority::Normal,
            )
            .unwrap();
        let second_id = second_runtime
            .submit(
                second.plan().diffusion_plan().clone(),
                CancellationToken::new(),
                TaskPriority::Normal,
            )
            .unwrap();

        let mut first_done = false;
        let mut second_done = false;
        while !first_done || !second_done {
            if !first_done {
                match first_runtime.run_one(&mut first).unwrap() {
                    DiffusionRunOutcome::SessionCompleted { .. } => first_done = true,
                    DiffusionRunOutcome::StepCompleted { .. } => {}
                    other => panic!("unexpected first-session outcome: {other:?}"),
                }
            }
            if !second_done {
                match second_runtime.run_one(&mut second).unwrap() {
                    DiffusionRunOutcome::SessionCompleted { .. } => second_done = true,
                    DiffusionRunOutcome::StepCompleted { .. } => {}
                    other => panic!("unexpected second-session outcome: {other:?}"),
                }
            }
        }

        assert_eq!(
            first_runtime.session(first_id).unwrap().state,
            DiffusionSessionState::Completed
        );
        assert_eq!(
            second_runtime.session(second_id).unwrap().state,
            DiffusionSessionState::Completed
        );
        assert_eq!(first.generated_tokens(), &[1, 1]);
        assert_eq!(second.generated_tokens(), &[1, 1]);
        assert_eq!(
            first
                .backend()
                .events
                .iter()
                .filter(|e| e.as_str() == "prefill:1")
                .count(),
            1
        );
        assert_eq!(
            second
                .backend()
                .events
                .iter()
                .filter(|e| e.as_str() == "prefill:1")
                .count(),
            1
        );
    }

    #[test]
    fn cancellation_before_first_step_releases_request_state() {
        let backend = SyntheticBackend {
            hidden: 4,
            vocab: 9,
            ..Default::default()
        };
        let mut executor =
            CloverDiffusionExecutor::new(tiny_graph(), backend, vec![3], options(2)).unwrap();
        let token = CancellationToken::new();
        token.cancel();
        let mut runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = runtime
            .submit(
                executor.plan().diffusion_plan().clone(),
                token,
                TaskPriority::Normal,
            )
            .unwrap();

        assert_eq!(
            runtime.run_one(&mut executor).unwrap(),
            DiffusionRunOutcome::Idle
        );
        assert_eq!(
            runtime.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
        assert_eq!(executor.backend().reset_count, 1);
        assert!(executor.backend().events.is_empty());
    }

    #[test]
    fn cancellation_during_preview_is_observed_after_the_safe_step() {
        let token = CancellationToken::new();
        let backend = SyntheticBackend {
            hidden: 4,
            vocab: 9,
            cancel_on_lm_head: Some(token.clone()),
            ..Default::default()
        };
        let mut executor =
            CloverDiffusionExecutor::new(tiny_graph(), backend, vec![3], options(2)).unwrap();
        let mut runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = runtime
            .submit(
                executor.plan().diffusion_plan().clone(),
                token,
                TaskPriority::Normal,
            )
            .unwrap();

        assert!(matches!(
            runtime.run_one(&mut executor).unwrap(),
            DiffusionRunOutcome::StepCompleted { .. }
        ));
        assert!(matches!(
            runtime.run_one(&mut executor).unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        assert_eq!(
            runtime.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
        assert!(executor.generated_tokens().is_empty());
        assert_eq!(executor.backend().reset_count, 1);
    }

    #[test]
    fn cancellation_after_a_committed_block_keeps_only_that_block() {
        let mut executor = CloverDiffusionExecutor::new(
            tiny_graph(),
            SyntheticBackend {
                hidden: 4,
                vocab: 9,
                ..Default::default()
            },
            vec![3],
            options(4),
        )
        .unwrap();
        let mut runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = runtime
            .submit(
                executor.plan().diffusion_plan().clone(),
                CancellationToken::new(),
                TaskPriority::Normal,
            )
            .unwrap();
        while executor.generated_tokens().len() < 2 {
            match runtime.run_one(&mut executor).unwrap() {
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome before block boundary: {other:?}"),
            }
        }
        runtime.cancel_with_executor(id, &mut executor).unwrap();
        assert_eq!(
            runtime.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
        assert_eq!(executor.generated_tokens(), &[1, 1]);
        assert_eq!(executor.backend().reset_count, 1);
    }

    #[test]
    fn expert_failure_is_terminal_and_releases_clean_state() {
        let backend = SyntheticBackend {
            hidden: 4,
            vocab: 9,
            fail_moe_at: Some(0),
            ..Default::default()
        };
        let mut executor =
            CloverDiffusionExecutor::new(tiny_graph(), backend, vec![3], options(2)).unwrap();
        let mut runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = runtime
            .submit(
                executor.plan().diffusion_plan().clone(),
                CancellationToken::new(),
                TaskPriority::Normal,
            )
            .unwrap();

        assert!(matches!(
            runtime.run_one(&mut executor).unwrap(),
            DiffusionRunOutcome::StepFailed { .. }
        ));
        assert_eq!(
            runtime.session(id).unwrap().state,
            DiffusionSessionState::Failed
        );
        assert_eq!(executor.backend().reset_count, 1);
    }

    #[test]
    fn rejects_prompt_when_complete_preview_blocks_exceed_context() {
        let result = CloverDiffusionExecutor::new(
            tiny_graph(),
            SyntheticBackend {
                hidden: 4,
                vocab: 9,
                ..Default::default()
            },
            vec![3; 31],
            options(2),
        );
        assert!(matches!(result, Err(CloverExecutorError::ContextExceeded)));
    }

    #[test]
    fn unmapped_head_rows_follow_only_the_explicit_request_policy() {
        let backend = SyntheticBackend {
            hidden: 4,
            vocab: 9,
            terminal_logit: Some((8, 10.0)),
            ..Default::default()
        };
        let mut mask_options = options(2);
        mask_options.invalid_token_policy = Some(CloverInvalidTokenPolicy::Mask);
        let mut mask_executor =
            CloverDiffusionExecutor::new(tiny_graph(), backend, vec![3], mask_options).unwrap();
        let mut mask_runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let mask_id = mask_runtime
            .submit(
                mask_executor.plan().diffusion_plan().clone(),
                CancellationToken::new(),
                TaskPriority::Normal,
            )
            .unwrap();
        loop {
            match mask_runtime.run_one(&mut mask_executor).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected masked-policy outcome: {other:?}"),
            }
        }
        assert_eq!(
            mask_runtime.session(mask_id).unwrap().state,
            DiffusionSessionState::Completed
        );
        assert_eq!(mask_executor.generated_tokens(), &[1, 1]);
        assert_eq!(mask_executor.output().sequence(), vec![3, 1, 1]);
        assert_eq!(mask_executor.output().pad_token_id, 2);

        let backend = SyntheticBackend {
            hidden: 4,
            vocab: 9,
            terminal_logit: Some((8, 10.0)),
            ..Default::default()
        };
        let mut reject_executor =
            CloverDiffusionExecutor::new(tiny_graph(), backend, vec![3], options(2)).unwrap();
        let mut reject_runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let reject_id = reject_runtime
            .submit(
                reject_executor.plan().diffusion_plan().clone(),
                CancellationToken::new(),
                TaskPriority::Normal,
            )
            .unwrap();
        assert!(matches!(
            reject_runtime.run_one(&mut reject_executor).unwrap(),
            DiffusionRunOutcome::StepCompleted { .. }
        ));
        assert!(matches!(
            reject_runtime.run_one(&mut reject_executor).unwrap(),
            DiffusionRunOutcome::StepFailed { .. }
        ));
        assert_eq!(
            reject_runtime.session(reject_id).unwrap().state,
            DiffusionSessionState::Failed
        );
    }
}
