//! Clover's pinned block LowConfidence generation policy.
//!
//! The policy mirrors the authenticated upstream sampling helper: response-only
//! penalties are committed once per resolved block, confidence is always
//! measured from unscaled FP32 logits, and a fallback position guarantees
//! progress when no pending token clears the strict threshold.

use claymore_runtime::diffusion::{BlockId, DiffusionError, DiffusionPlan, DiffusionStep};
use thiserror::Error;

use crate::CloverConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct CloverPenaltyConfig {
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub length_penalty: f32,
    pub length_penalty_start: usize,
    pub length_penalty_window: usize,
    pub think_end_bias: f32,
    pub think_end_start: usize,
    pub think_end_window: usize,
    pub think_end_token_id: Option<u32>,
}

impl Default for CloverPenaltyConfig {
    fn default() -> Self {
        Self {
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            length_penalty: 0.0,
            length_penalty_start: 8192,
            length_penalty_window: 8192,
            think_end_bias: 0.0,
            think_end_start: 4096,
            think_end_window: 4096,
            think_end_token_id: None,
        }
    }
}

impl CloverPenaltyConfig {
    fn validate(&self, vocab_size: usize) -> Result<(), CloverGenerationError> {
        for (name, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
                return Err(CloverGenerationError::InvalidPenalty(name));
            }
        }
        if !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
            || self.repetition_penalty > 2.0
        {
            return Err(CloverGenerationError::InvalidPenalty("repetition_penalty"));
        }
        if !self.length_penalty.is_finite() || self.length_penalty < 0.0 {
            return Err(CloverGenerationError::InvalidPenalty("length_penalty"));
        }
        if self.length_penalty_window == 0 {
            return Err(CloverGenerationError::InvalidPenalty(
                "length_penalty_window",
            ));
        }
        if !self.think_end_bias.is_finite() || self.think_end_bias < 0.0 {
            return Err(CloverGenerationError::InvalidPenalty("think_end_bias"));
        }
        if self.think_end_window == 0 {
            return Err(CloverGenerationError::InvalidPenalty("think_end_window"));
        }
        if let Some(token) = self.think_end_token_id {
            if token as usize >= vocab_size {
                return Err(CloverGenerationError::InvalidTokenId(token));
            }
        } else if self.think_end_bias != 0.0 {
            return Err(CloverGenerationError::MissingThinkEndToken);
        }
        Ok(())
    }
}

/// Explicit policy for the 243 output rows that exist in the model head but
/// have no tokenizer IDs. A2-B deliberately leaves this choice to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloverInvalidTokenPolicy {
    Reject,
    Mask,
}

/// Explicit per-request generation controls. EOS/PAD are kept separate from
/// the model config; an omitted EOS list falls back to config.eos_token_id,
/// then stop_token_ids are unioned as in the pinned upstream implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct CloverGenerationOptions {
    pub max_new_tokens: usize,
    pub block_size: usize,
    pub threshold: f32,
    /// Number of tokenizer-addressable IDs. The 243 model-head rows beyond this
    /// bound are not silently treated as valid output tokens.
    pub tokenizer_vocab_size: usize,
    /// Must be explicit until the architect closes A2-B.
    pub invalid_token_policy: Option<CloverInvalidTokenPolicy>,
    pub eos_token_ids: Option<Vec<u32>>,
    pub stop_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub penalties: CloverPenaltyConfig,
    pub seed: u64,
}

impl CloverGenerationOptions {
    pub fn greedy(max_new_tokens: usize, tokenizer_vocab_size: usize) -> Self {
        Self {
            max_new_tokens,
            block_size: 32,
            threshold: 0.95,
            tokenizer_vocab_size,
            invalid_token_policy: None,
            eos_token_ids: None,
            stop_token_ids: Vec::new(),
            pad_token_id: None,
            do_sample: false,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            penalties: CloverPenaltyConfig::default(),
            seed: 0,
        }
    }

    pub fn validate(&self, config: &CloverConfig) -> Result<(), CloverGenerationError> {
        if self.max_new_tokens == 0 {
            return Err(CloverGenerationError::ZeroMaxNewTokens);
        }
        if self.block_size == 0 || config.block_size % self.block_size != 0 {
            return Err(CloverGenerationError::InvalidBlockSize {
                requested: self.block_size,
                configured: config.block_size,
            });
        }
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Err(CloverGenerationError::InvalidThreshold(self.threshold));
        }
        if self.tokenizer_vocab_size == 0
            || self.tokenizer_vocab_size > config.vocab_size
            || config.mask_token_id as usize >= self.tokenizer_vocab_size
        {
            return Err(CloverGenerationError::InvalidTokenizerVocabulary {
                tokenizer: self.tokenizer_vocab_size,
                model: config.vocab_size,
            });
        }
        if self.invalid_token_policy.is_none() {
            return Err(CloverGenerationError::InvalidTokenPolicyUndecided);
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(CloverGenerationError::InvalidTemperature(self.temperature));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(CloverGenerationError::InvalidTopP(self.top_p));
        }
        self.penalties.validate(config.vocab_size)?;
        let terminal = self.resolved_terminal_ids(config)?;
        if terminal.contains(&config.mask_token_id)
            || self.penalties.think_end_token_id == Some(config.mask_token_id)
        {
            return Err(CloverGenerationError::ReservedMaskToken);
        }
        for id in terminal
            .iter()
            .copied()
            .chain(self.stop_token_ids.iter().copied())
        {
            if id as usize >= self.tokenizer_vocab_size {
                return Err(CloverGenerationError::InvalidTokenId(id));
            }
        }
        let pad = self
            .pad_token_id
            .or(config.pad_token_id)
            .or_else(|| terminal.first().copied())
            .unwrap_or(0);
        if pad as usize >= self.tokenizer_vocab_size {
            return Err(CloverGenerationError::InvalidTokenId(pad));
        }
        Ok(())
    }

    pub fn resolved_terminal_ids(
        &self,
        config: &CloverConfig,
    ) -> Result<Vec<u32>, CloverGenerationError> {
        let mut ids = self
            .eos_token_ids
            .clone()
            .unwrap_or_else(|| vec![config.eos_token_id]);
        ids.extend(self.stop_token_ids.iter().copied());
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    pub fn resolved_pad_token_id(&self, config: &CloverConfig) -> u32 {
        self.pad_token_id
            .or(config.pad_token_id)
            .or_else(|| {
                self.resolved_terminal_ids(config)
                    .ok()
                    .and_then(|ids| ids.first().copied())
            })
            .unwrap_or(0)
    }
}

#[derive(Debug, Error)]
pub enum CloverGenerationError {
    #[error(transparent)]
    Diffusion(#[from] DiffusionError),
    #[error("max_new_tokens must be greater than zero")]
    ZeroMaxNewTokens,
    #[error("requested block size {requested} must be positive and divide configured block size {configured}")]
    InvalidBlockSize { requested: usize, configured: usize },
    #[error("LowConfidence threshold must be finite and in [0, 1], got {0}")]
    InvalidThreshold(f32),
    #[error("tokenizer vocabulary {tokenizer} is incompatible with model vocabulary {model}")]
    InvalidTokenizerVocabulary { tokenizer: usize, model: usize },
    #[error("temperature must be finite and positive, got {0}")]
    InvalidTemperature(f32),
    #[error("top_p must be in (0, 1], got {0}")]
    InvalidTopP(f32),
    #[error("invalid Clover penalty setting: {0}")]
    InvalidPenalty(&'static str),
    #[error("think_end_bias requires think_end_token_id")]
    MissingThinkEndToken,
    #[error("token ID {0} is outside the tokenizer-addressable vocabulary")]
    InvalidTokenId(u32),
    #[error("the policy for the 243 non-tokenizer model-head rows must be explicit (A2-B)")]
    InvalidTokenPolicyUndecided,
    #[error("the mask token cannot be configured as EOS, stop, or think-end")]
    ReservedMaskToken,
    #[error("generation plan exceeds the supported block identifier range")]
    TooManyBlocks,
    #[error("generation plan step count overflow")]
    PlanSizeOverflow,
    #[error("logits must contain finite values after the mask-token exclusion")]
    InvalidLogits,
    #[error("sampling failed because no finite candidate remains")]
    NoSamplingCandidate,
}

/// The immutable M9 step plan for one Clover generation request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloverGenerationPlan {
    diffusion: DiffusionPlan,
    block_size: usize,
    block_count: usize,
}

impl CloverGenerationPlan {
    pub fn new(
        config: &CloverConfig,
        options: &CloverGenerationOptions,
    ) -> Result<Self, CloverGenerationError> {
        options.validate(config)?;
        let block_count = options
            .max_new_tokens
            .checked_add(options.block_size - 1)
            .ok_or(CloverGenerationError::PlanSizeOverflow)?
            / options.block_size;
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| CloverGenerationError::TooManyBlocks)?;
        let steps_per_block = options
            .block_size
            .checked_add(3)
            .ok_or(CloverGenerationError::PlanSizeOverflow)?;
        let capacity = block_count
            .checked_mul(steps_per_block)
            .ok_or(CloverGenerationError::PlanSizeOverflow)?;
        let mut steps = Vec::with_capacity(capacity);
        for block in 0..block_count_u32 {
            let id = BlockId(block);
            steps.push(DiffusionStep::PrepareBlock { block: id });
            for iteration in 0..options.block_size {
                steps.push(DiffusionStep::Denoise {
                    block: id,
                    iteration: u32::try_from(iteration)
                        .map_err(|_| CloverGenerationError::TooManyBlocks)?,
                });
            }
            steps.push(DiffusionStep::ValidateBlock { block: id });
            steps.push(DiffusionStep::CommitBlock { block: id });
        }
        Ok(Self {
            diffusion: DiffusionPlan::new(steps)?,
            block_size: options.block_size,
            block_count,
        })
    }

    pub fn diffusion_plan(&self) -> &DiffusionPlan {
        &self.diffusion
    }
    pub fn into_diffusion_plan(self) -> DiffusionPlan {
        self.diffusion
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn block_count(&self) -> usize {
        self.block_count
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CloverPenaltyState {
    counts: Vec<u32>,
    committed_length: usize,
}

impl CloverPenaltyState {
    pub(crate) fn new(vocab_size: usize) -> Self {
        Self {
            counts: vec![0; vocab_size],
            committed_length: 0,
        }
    }

    pub(crate) fn apply(
        &self,
        logits: &mut [f32],
        block: &[u32],
        mask_token_id: u32,
        options: &CloverGenerationOptions,
        terminal_ids: &[u32],
    ) -> Result<(), CloverGenerationError> {
        let vocab = self.counts.len();
        if logits.len()
            != block
                .len()
                .checked_mul(vocab)
                .ok_or(CloverGenerationError::PlanSizeOverflow)?
        {
            return Err(CloverGenerationError::InvalidLogits);
        }
        let seen: Vec<bool> = self.counts.iter().map(|&count| count > 0).collect();
        let penalties = &options.penalties;
        let think_close = penalties.think_end_token_id.map(|id| id as usize);
        let visible_close = think_close
            .map(|close| {
                block
                    .iter()
                    .any(|&token| token != mask_token_id && token as usize == close)
            })
            .unwrap_or(false);
        let think_enabled = think_close
            .map(|close| !seen[close] && !visible_close)
            .unwrap_or(false);

        for (position, row) in logits.chunks_exact_mut(vocab).enumerate() {
            if penalties.frequency_penalty != 0.0 {
                for (id, value) in row.iter_mut().enumerate() {
                    *value -= self.counts[id] as f32 * penalties.frequency_penalty;
                }
            }
            if penalties.presence_penalty != 0.0 {
                for (id, value) in row.iter_mut().enumerate() {
                    if seen[id] {
                        *value -= penalties.presence_penalty;
                    }
                }
            }
            if penalties.repetition_penalty != 1.0 {
                for (id, value) in row.iter_mut().enumerate() {
                    if seen[id] {
                        *value = if *value < 0.0 {
                            *value * penalties.repetition_penalty
                        } else {
                            *value / penalties.repetition_penalty
                        };
                    }
                }
            }
            let response_position = self
                .committed_length
                .checked_add(position + 1)
                .ok_or(CloverGenerationError::PlanSizeOverflow)?;
            if penalties.length_penalty != 0.0 {
                let ramp = ramp(
                    response_position,
                    penalties.length_penalty_start,
                    penalties.length_penalty_window,
                    penalties.length_penalty,
                );
                for &id in terminal_ids {
                    row[id as usize] += ramp;
                }
            }
            if think_enabled {
                if let Some(close) = think_close {
                    row[close] += ramp(
                        response_position,
                        penalties.think_end_start,
                        penalties.think_end_window,
                        penalties.think_end_bias,
                    );
                }
            }
        }
        Ok(())
    }

    pub(crate) fn commit(&mut self, tokens: &[u32]) {
        self.committed_length = self.committed_length.saturating_add(tokens.len());
        for &token in tokens {
            if let Some(count) = self.counts.get_mut(token as usize) {
                *count = count.saturating_add(1);
            }
        }
    }
}

fn ramp(position: usize, start: usize, window: usize, strength: f32) -> f32 {
    ((position as f32 - start as f32) / window as f32).clamp(0.0, 1.0) * strength
}

#[derive(Debug, Clone)]
pub(crate) struct CloverRng {
    state: u64,
}

impl CloverRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Return per-position token predictions and confidence. Sampling filters affect
/// prediction only; confidence always uses the post-penalty, unscaled logits.
pub(crate) fn predict_with_confidence(
    logits: &[f32],
    rows: usize,
    vocab: usize,
    options: &CloverGenerationOptions,
    rng: &mut CloverRng,
) -> Result<(Vec<u32>, Vec<f32>), CloverGenerationError> {
    if rows == 0
        || vocab == 0
        || logits.len()
            != rows
                .checked_mul(vocab)
                .ok_or(CloverGenerationError::PlanSizeOverflow)?
    {
        return Err(CloverGenerationError::InvalidLogits);
    }
    let mut predictions = Vec::with_capacity(rows);
    let mut confidences = Vec::with_capacity(rows);
    for row in logits.chunks_exact(vocab) {
        let original_probs = softmax(row)?;
        let prediction = if options.do_sample {
            let mut scaled: Vec<f32> = row
                .iter()
                .map(|value| *value / options.temperature)
                .collect();
            if options.top_k > 0 && options.top_k < vocab {
                let mut ordered: Vec<usize> = (0..vocab).collect();
                ordered.sort_by(|&a, &b| scaled[b].total_cmp(&scaled[a]).then_with(|| a.cmp(&b)));
                let cutoff = scaled[ordered[options.top_k - 1]];
                for value in &mut scaled {
                    if *value < cutoff {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }
            if options.top_p < 1.0 {
                let probabilities = softmax(&scaled)?;
                let mut ordered: Vec<usize> = (0..vocab).collect();
                ordered.sort_by(|&a, &b| scaled[b].total_cmp(&scaled[a]).then_with(|| a.cmp(&b)));
                let mut preceding = 0.0f32;
                for id in ordered {
                    if preceding > options.top_p {
                        scaled[id] = f32::NEG_INFINITY;
                    }
                    preceding += probabilities[id];
                }
            }
            let probabilities = softmax(&scaled)?;
            let draw = rng.unit_f64();
            let mut cumulative = 0.0f64;
            let mut selected = None;
            for (id, probability) in probabilities.iter().enumerate() {
                cumulative += *probability as f64;
                if draw < cumulative {
                    selected = Some(id);
                    break;
                }
            }
            selected
                .or_else(|| probabilities.iter().rposition(|p| *p > 0.0))
                .ok_or(CloverGenerationError::NoSamplingCandidate)?
        } else {
            let mut best = 0usize;
            for id in 1..vocab {
                if row[id] > row[best] {
                    best = id;
                }
            }
            best
        };
        predictions
            .push(u32::try_from(prediction).map_err(|_| CloverGenerationError::PlanSizeOverflow)?);
        confidences.push(original_probs[prediction]);
    }
    Ok((predictions, confidences))
}

fn softmax(logits: &[f32]) -> Result<Vec<f32>, CloverGenerationError> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return Err(CloverGenerationError::NoSamplingCandidate);
    }
    let mut probabilities: Vec<f32> = logits
        .iter()
        .map(|value| {
            if value.is_finite() {
                (*value - maximum).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f32 = probabilities.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(CloverGenerationError::NoSamplingCandidate);
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CloverConfig, LayerType};

    const CONFIG: &str = include_str!("../../../docs/audits/fixtures/clover-config.json");

    fn tiny_config() -> CloverConfig {
        let mut c = CloverConfig::from_json(CONFIG).unwrap();
        c.hidden_size = 4;
        c.num_hidden_layers = 2;
        c.layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];
        c.num_experts = 4;
        c.num_experts_per_tok = 2;
        c.vocab_size = 8;
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
        c.validate().unwrap();
        c
    }

    #[test]
    fn plan_schedules_multiple_complete_denoising_blocks() {
        let config = tiny_config();
        let options = CloverGenerationOptions {
            block_size: 2,
            invalid_token_policy: Some(CloverInvalidTokenPolicy::Reject),
            ..CloverGenerationOptions::greedy(5, 8)
        };
        let plan = CloverGenerationPlan::new(&config, &options).unwrap();
        assert_eq!(plan.block_count(), 3);
        assert_eq!(plan.block_size(), 2);
        assert_eq!(plan.diffusion_plan().block_count(), 3);
        assert_eq!(plan.diffusion_plan().steps().len(), 15);
    }

    #[test]
    fn eos_and_stop_ids_are_unioned_without_collapsing_config_metadata() {
        let config = tiny_config();
        let options = CloverGenerationOptions {
            eos_token_ids: Some(vec![2, 3]),
            stop_token_ids: vec![3, 4],
            ..CloverGenerationOptions::greedy(4, 8)
        };
        assert_eq!(
            options.resolved_terminal_ids(&config).unwrap(),
            vec![2, 3, 4]
        );
        let fallback = CloverGenerationOptions::greedy(4, 7);
        assert_eq!(fallback.resolved_terminal_ids(&config).unwrap(), vec![2]);
    }

    #[test]
    fn penalty_state_counts_only_committed_response_tokens() {
        let options = CloverGenerationOptions {
            penalties: CloverPenaltyConfig {
                frequency_penalty: 0.5,
                presence_penalty: 0.25,
                ..CloverPenaltyConfig::default()
            },
            ..CloverGenerationOptions::greedy(4, 7)
        };
        let mut state = CloverPenaltyState::new(8);
        let mut logits = vec![2.0; 16];
        state
            .apply(&mut logits, &[1, 2], 7, &options, &[3])
            .unwrap();
        assert_eq!(logits[1], 2.0);
        state.commit(&[1]);
        let mut next = vec![2.0; 8];
        state.apply(&mut next, &[7], 7, &options, &[3]).unwrap();
        assert_eq!(next[1], 1.25);
    }

    #[test]
    fn sampling_confidence_uses_unscaled_logits() {
        let options = CloverGenerationOptions {
            do_sample: true,
            temperature: 0.5,
            top_k: 1,
            ..CloverGenerationOptions::greedy(2, 7)
        };
        let mut rng = CloverRng::new(42);
        let (tokens, confidence) =
            predict_with_confidence(&[2.0, 0.0, -1.0], 1, 3, &options, &mut rng).unwrap();
        assert_eq!(tokens, vec![0]);
        let expected = 1.0 / (1.0 + (-2.0f32).exp() + (-3.0f32).exp());
        assert!((confidence[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn refuses_to_guess_policy_for_unmapped_model_head_rows() {
        let config = tiny_config();
        let options = CloverGenerationOptions {
            block_size: 2,
            ..CloverGenerationOptions::greedy(4, 8)
        };
        assert!(matches!(
            options.validate(&config),
            Err(CloverGenerationError::InvalidTokenPolicyUndecided)
        ));
    }

    #[test]
    fn think_end_bias_is_disabled_by_committed_or_visible_close_tokens() {
        let options = CloverGenerationOptions {
            penalties: CloverPenaltyConfig {
                think_end_bias: 2.0,
                think_end_start: 0,
                think_end_window: 2,
                think_end_token_id: Some(3),
                ..CloverPenaltyConfig::default()
            },
            ..CloverGenerationOptions::greedy(4, 8)
        };
        let mut state = CloverPenaltyState::new(8);
        let mut logits = vec![0.0; 16];
        state
            .apply(&mut logits, &[7, 7], 7, &options, &[2])
            .unwrap();
        assert_eq!(logits[3], 1.0);
        assert_eq!(logits[8 + 3], 2.0);

        let mut visible = vec![0.0; 16];
        state
            .apply(&mut visible, &[3, 7], 7, &options, &[2])
            .unwrap();
        assert_eq!(visible[3], 0.0);
        assert_eq!(visible[8 + 3], 0.0);

        state.commit(&[3]);
        let mut committed = vec![0.0; 8];
        state
            .apply(&mut committed, &[7], 7, &options, &[2])
            .unwrap();
        assert_eq!(committed[3], 0.0);
    }

    #[test]
    fn length_penalty_ramps_only_over_committed_response_positions() {
        let options = CloverGenerationOptions {
            penalties: CloverPenaltyConfig {
                length_penalty: 2.0,
                length_penalty_start: 0,
                length_penalty_window: 4,
                ..CloverPenaltyConfig::default()
            },
            ..CloverGenerationOptions::greedy(4, 8)
        };
        let mut state = CloverPenaltyState::new(8);
        let mut first = vec![0.0; 8];
        state.apply(&mut first, &[7], 7, &options, &[2]).unwrap();
        assert_eq!(first[2], 0.5);
        state.commit(&[1, 1]);
        let mut second = vec![0.0; 8];
        state.apply(&mut second, &[7], 7, &options, &[2]).unwrap();
        assert_eq!(second[2], 1.5);
    }
}
