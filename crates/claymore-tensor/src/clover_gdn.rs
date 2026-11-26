//! CPU reference primitives for Clover's Gated DeltaNet (GDN).
//!
//! These operations are deliberately small, backend-independent references
//! derived from the immutable Clover release. All recurrent state is FP32;
//! activation outputs are explicitly rounded to F32 or BF16.

use crate::Bf16;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDimensions {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl GdnDimensions {
    fn validate(self) -> Result<Self, GdnError> {
        if self.key_heads == 0
            || self.value_heads == 0
            || self.key_dim == 0
            || self.value_dim == 0
            || self.value_heads % self.key_heads != 0
        {
            return Err(GdnError::InvalidDimensions);
        }
        Ok(self)
    }

    fn state_len(self) -> Result<usize, GdnError> {
        self.value_heads
            .checked_mul(self.value_dim)
            .and_then(|n| n.checked_mul(self.key_dim))
            .ok_or(GdnError::SizeOverflow)
    }
}

/// Output precision for the activation values. Recurrent state always stays FP32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnOutputDType {
    F32,
    BF16,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GdnError {
    #[error("GDN dimensions must be non-zero and value_heads must be divisible by key_heads")]
    InvalidDimensions,
    #[error("GDN input or state length does not match the declared dimensions")]
    ShapeMismatch,
    #[error("GDN size arithmetic overflow")]
    SizeOverflow,
    #[error("GDN scale must be finite and positive")]
    InvalidScale,
}


/// Build Clover full-attention additive mask for one active block.
///
/// Prefix keys are visible only when marked valid. During preview every valid
/// block key is visible to every valid query (bidirectional denoising). During
/// commit, block keys are causal within the current block. Invalid query rows
/// and invalid keys are fully masked; the attention reference maps an all-masked
/// row to zero probabilities, matching Clover's eager backend.
pub fn full_attention_mask(
    prefix_valid: &[bool],
    block_valid: &[bool],
    preview: bool,
) -> Result<Vec<f32>, GdnError> {
    let query_tokens = block_valid.len();
    let key_tokens = prefix_valid
        .len()
        .checked_add(query_tokens)
        .ok_or(GdnError::SizeOverflow)?;
    let len = query_tokens
        .checked_mul(key_tokens)
        .ok_or(GdnError::SizeOverflow)?;
    let mut mask = vec![f32::NEG_INFINITY; len];
    for query in 0..query_tokens {
        if !block_valid[query] {
            continue;
        }
        let row = query * key_tokens;
        for (key, &valid) in prefix_valid.iter().enumerate() {
            if valid {
                mask[row + key] = 0.0;
            }
        }
        for key in 0..query_tokens {
            if block_valid[key] && (preview || key <= query) {
                mask[row + prefix_valid.len() + key] = 0.0;
            }
        }
    }
    Ok(mask)
}

/// Causal depthwise convolution state transition used by committed GDN tokens.
///
/// Inputs use row-major layouts: input [tokens, channels], history
/// [channels, kernel_width - 1], weights [channels, kernel_width]. The state
/// is copied and returned; the caller's cache is never mutated on error.
/// Invalid tokens emit zeros and do not advance their channel history.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv_silu(
    input: &[f32],
    tokens: usize,
    channels: usize,
    history: &[f32],
    weights: &[f32],
    kernel_width: usize,
    valid: &[bool],
    output_dtype: GdnOutputDType,
) -> Result<(Vec<f32>, Vec<f32>), GdnError> {
    validate_conv(input, tokens, channels, history, weights, kernel_width, valid)?;
    let mut next_history = history.to_vec();
    let mut output = vec![0.0; input.len()];
    let mut window = vec![0.0; kernel_width];
    for token in 0..tokens {
        for channel in 0..channels {
            if !valid[token] {
                continue;
            }
            let hist_base = channel * (kernel_width - 1);
            window[..kernel_width - 1]
                .copy_from_slice(&next_history[hist_base..hist_base + kernel_width - 1]);
            window[kernel_width - 1] = input[token * channels + channel];
            let mut sum = 0.0f32;
            for tap in 0..kernel_width {
                sum += round_product(
                    window[tap],
                    weights[channel * kernel_width + tap],
                    output_dtype,
                );
            }
            output[token * channels + channel] = silu_scalar(sum);
            if kernel_width > 1 {
                next_history[hist_base..hist_base + kernel_width - 2]
                    .copy_from_slice(&next_history[hist_base + 1..hist_base + kernel_width - 1]);
                next_history[hist_base + kernel_width - 2] = input[token * channels + channel];
            }
        }
    }
    Ok((round_output(output, output_dtype), next_history))
}

/// Bidirectional block-preview convolution. The clean causal history is read
/// only; right-context taps are applied from future valid tokens in the same
/// block. The returned activation is masked and rounded to the input dtype.
#[allow(clippy::too_many_arguments)]
pub fn block_conv_silu(
    input: &[f32],
    tokens: usize,
    channels: usize,
    clean_history: &[f32],
    causal_weights: &[f32],
    right_weights: &[f32],
    kernel_width: usize,
    valid: &[bool],
    output_dtype: GdnOutputDType,
) -> Result<Vec<f32>, GdnError> {
    validate_conv(
        input, tokens, channels, clean_history, causal_weights, kernel_width, valid,
    )?;
    let right_len = channels
        .checked_mul(kernel_width.saturating_sub(1))
        .ok_or(GdnError::SizeOverflow)?;
    if right_weights.len() != right_len {
        return Err(GdnError::ShapeMismatch);
    }
    let joined_len = kernel_width - 1 + tokens;
    let mut joined = vec![0.0f32; channels.checked_mul(joined_len).ok_or(GdnError::SizeOverflow)?];
    for channel in 0..channels {
        let base = channel * joined_len;
        joined[base..base + kernel_width - 1].copy_from_slice(
            &clean_history[channel * (kernel_width - 1)..(channel + 1) * (kernel_width - 1)],
        );
        for token in 0..tokens {
            joined[base + kernel_width - 1 + token] = if valid[token] {
                input[token * channels + channel]
            } else {
                0.0
            };
        }
    }
    let mut output = vec![0.0f32; input.len()];
    for token in 0..tokens {
        if !valid[token] {
            continue;
        }
        for channel in 0..channels {
            let base = channel * joined_len + token;
            let mut sum = 0.0f32;
            for tap in 0..kernel_width {
                sum += joined[base + tap] * causal_weights[channel * kernel_width + tap];
            }
            for distance in 1..kernel_width {
                let future = token + distance;
                if future < tokens && valid[future] {
                    sum += input[future * channels + channel]
                        * right_weights[channel * (kernel_width - 1) + distance - 1];
                }
            }
            output[token * channels + channel] = silu_scalar(sum);
        }
    }
    Ok(round_output(output, output_dtype))
}

fn validate_conv(
    input: &[f32],
    tokens: usize,
    channels: usize,
    history: &[f32],
    weights: &[f32],
    kernel_width: usize,
    valid: &[bool],
) -> Result<(), GdnError> {
    if tokens == 0 || channels == 0 || kernel_width == 0 {
        return Err(GdnError::InvalidDimensions);
    }
    let input_len = tokens.checked_mul(channels).ok_or(GdnError::SizeOverflow)?;
    let history_len = channels
        .checked_mul(kernel_width - 1)
        .ok_or(GdnError::SizeOverflow)?;
    let weight_len = channels.checked_mul(kernel_width).ok_or(GdnError::SizeOverflow)?;
    if input.len() != input_len
        || history.len() != history_len
        || weights.len() != weight_len
        || valid.len() != tokens
    {
        return Err(GdnError::ShapeMismatch);
    }
    Ok(())
}

fn round_product(lhs: f32, rhs: f32, dtype: GdnOutputDType) -> f32 {
    match dtype {
        GdnOutputDType::F32 => lhs * rhs,
        GdnOutputDType::BF16 => Bf16::from_f32(lhs * rhs).to_f32(),
    }
}

fn silu_scalar(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == f32::NEG_INFINITY {
        return -0.0;
    }
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    value * sigmoid
}

/// Commit a causal prefix token-by-token using Clover's recurrent delta rule.
///
/// Flattened layouts:
/// - q/k: [tokens, key_heads, key_dim]
/// - v: [tokens, value_heads, value_dim]
/// - g/beta: [tokens, value_heads]
/// - state: [value_heads, value_dim, key_dim] (FP32)
///
/// Invalid tokens emit zeros and leave the recurrent state unchanged. All
/// lengths are checked before the caller's state is mutated.
#[allow(clippy::too_many_arguments)]
pub fn recurrent_commit(
    dims: GdnDimensions,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    valid: &[bool],
    output_dtype: GdnOutputDType,
    state: &mut [f32],
) -> Result<Vec<f32>, GdnError> {
    let dims = dims.validate()?;
    let tokens = valid.len();
    let qk_len = tokens
        .checked_mul(dims.key_heads)
        .and_then(|n| n.checked_mul(dims.key_dim))
        .ok_or(GdnError::SizeOverflow)?;
    let v_len = tokens
        .checked_mul(dims.value_heads)
        .and_then(|n| n.checked_mul(dims.value_dim))
        .ok_or(GdnError::SizeOverflow)?;
    let gate_len = tokens
        .checked_mul(dims.value_heads)
        .ok_or(GdnError::SizeOverflow)?;
    let state_len = dims.state_len()?;
    if q.len() != qk_len
        || k.len() != qk_len
        || v.len() != v_len
        || g.len() != gate_len
        || beta.len() != gate_len
        || state.len() != state_len
    {
        return Err(GdnError::ShapeMismatch);
    }

    // Work on a private copy so a future fallible operation cannot leave the
    // caller with a partially committed sequence.
    let mut next_state = state.to_vec();
    let mut output = vec![0.0f32; v_len];
    let repeat = dims.value_heads / dims.key_heads;
    let query_scale = (dims.key_dim as f32).sqrt().recip();

    for token in 0..tokens {
        if !valid[token] {
            continue;
        }
        for value_head in 0..dims.value_heads {
            let key_head = value_head / repeat;
            let q_base = (token * dims.key_heads + key_head) * dims.key_dim;
            let k_base = q_base;
            let state_base = value_head * dims.value_dim * dims.key_dim;
            let gate_index = token * dims.value_heads + value_head;

            let q_norm = l2_normalize(&q[q_base..q_base + dims.key_dim]);
            let k_norm = l2_normalize(&k[k_base..k_base + dims.key_dim]);
            let decay = g[gate_index].exp();

            for value_index in 0..dims.value_dim {
                let row_base = state_base + value_index * dims.key_dim;
                let row = &mut next_state[row_base..row_base + dims.key_dim];
                for item in row.iter_mut() {
                    *item *= decay;
                }
                let remembered = dot(row, &k_norm);
                let value = v[(token * dims.value_heads + value_head) * dims.value_dim
                    + value_index];
                let delta = (value - remembered) * beta[gate_index];
                for key_index in 0..dims.key_dim {
                    row[key_index] += delta * k_norm[key_index];
                }
                let query_value = dot(row, &q_norm) * query_scale;
                output[(token * dims.value_heads + value_head) * dims.value_dim + value_index] =
                    query_value;
            }
        }
    }

    state.copy_from_slice(&next_state);
    Ok(round_output(output, output_dtype))
}

/// Read-only bidirectional block preview over the clean recurrent state.
///
/// Normalized q/k form local block logits, and a single history bucket
/// competes with valid local keys. The history bucket is calibrated by
/// ln(valid_tokens / 32).
///
/// Flattened layouts:
/// - q/k: [tokens, key_heads, key_dim]
/// - v: [tokens, value_heads, value_dim]
/// - g: [tokens, value_heads]
/// - clean_state: [value_heads, value_dim, key_dim] (FP32)
/// - per-head calibration arrays: [value_heads]
///
/// The clean state is never modified. The returned output is [tokens,
/// value_heads, value_dim] in FP32.
#[allow(clippy::too_many_arguments)]
pub fn block_preview(
    dims: GdnDimensions,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    valid: &[bool],
    clean_state: &[f32],
    scale: f32,
    output_dtype: GdnOutputDType,
    logit_scale: &[f32],
    history_bias: &[f32],
    history_c: &[f32],
) -> Result<Vec<f32>, GdnError> {
    let dims = dims.validate()?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(GdnError::InvalidScale);
    }
    let tokens = valid.len();
    let qk_len = tokens
        .checked_mul(dims.key_heads)
        .and_then(|n| n.checked_mul(dims.key_dim))
        .ok_or(GdnError::SizeOverflow)?;
    let v_len = tokens
        .checked_mul(dims.value_heads)
        .and_then(|n| n.checked_mul(dims.value_dim))
        .ok_or(GdnError::SizeOverflow)?;
    let gate_len = tokens
        .checked_mul(dims.value_heads)
        .ok_or(GdnError::SizeOverflow)?;
    let state_len = dims.state_len()?;
    if q.len() != qk_len
        || k.len() != qk_len
        || v.len() != v_len
        || g.len() != gate_len
        || clean_state.len() != state_len
        || logit_scale.len() != dims.value_heads
        || history_bias.len() != dims.value_heads
        || history_c.len() != dims.value_heads
    {
        return Err(GdnError::ShapeMismatch);
    }

    let repeat = dims.value_heads / dims.key_heads;
    let valid_count = valid.iter().filter(|&&is_valid| is_valid).count().max(1);
    let block_log_ratio = (valid_count as f32 / 32.0).ln();
    let mut output = vec![0.0f32; v_len];
    let mut logits = vec![f32::NEG_INFINITY; tokens + 1];

    for value_head in 0..dims.value_heads {
        let state_base = value_head * dims.value_dim * dims.key_dim;
        let state = &clean_state[state_base..state_base + dims.value_dim * dims.key_dim];
        let head_logit_scale = logit_scale[value_head];
        let history_logit =
            history_bias[value_head] + history_c[value_head] * block_log_ratio;

        for query_token in 0..tokens {
            if !valid[query_token] {
                continue;
            }
            let query_head = value_head / repeat;
            let q_base = (query_token * dims.key_heads + query_head) * dims.key_dim;
            let q_norm = l2_normalize(&q[q_base..q_base + dims.key_dim]);

            let mut history = vec![0.0f32; dims.value_dim];
            let decay = g[query_token * dims.value_heads + value_head].exp();
            for value_index in 0..dims.value_dim {
                let row_base = value_index * dims.key_dim;
                history[value_index] = dot(
                    &state[row_base..row_base + dims.key_dim],
                    &q_norm,
                ) * scale
                    * decay;
            }

            for key_token in 0..tokens {
                if valid[key_token] {
                    let key_head = value_head / repeat;
                    let k_base = (key_token * dims.key_heads + key_head) * dims.key_dim;
                    let k_norm = l2_normalize(&k[k_base..k_base + dims.key_dim]);
                    logits[key_token] = dot(&q_norm, &k_norm) * head_logit_scale;
                } else {
                    logits[key_token] = f32::NEG_INFINITY;
                }
            }
            logits[tokens] = history_logit;
            let weights = stable_softmax(&logits);

            for value_index in 0..dims.value_dim {
                let mut local = 0.0f32;
                for key_token in 0..tokens {
                    if valid[key_token] {
                        let value = v[(key_token * dims.value_heads + value_head)
                            * dims.value_dim
                            + value_index];
                        local += weights[key_token] * value;
                    }
                }
                let result = weights[tokens] * history[value_index] + local;
                output[(query_token * dims.value_heads + value_head) * dims.value_dim
                    + value_index] = result;
            }
        }
    }
    Ok(round_output(output, output_dtype))
}

fn round_output(values: Vec<f32>, dtype: GdnOutputDType) -> Vec<f32> {
    values
        .into_iter()
        .map(|value| match dtype {
            GdnOutputDType::F32 => value,
            GdnOutputDType::BF16 => Bf16::from_f32(value).to_f32(),
        })
        .collect()
}

fn l2_normalize(values: &[f32]) -> Vec<f32> {
    let sum = values.iter().map(|value| value * value).sum::<f32>();
    let inverse = (sum + 1e-6).sqrt().recip();
    values.iter().map(|value| value * inverse).collect()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .fold(0.0f32, |acc, (&left, &right)| left.mul_add(right, acc))
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return vec![0.0; logits.len()];
    }
    let exponents: Vec<f32> = logits.iter().map(|value| (*value - max).exp()).collect();
    let denominator = exponents.iter().sum::<f32>();
    exponents
        .into_iter()
        .map(|value| value / denominator)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> GdnDimensions {
        GdnDimensions {
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 1,
        }
    }

    #[test]
    fn full_attention_mask_distinguishes_bidirectional_preview_and_causal_commit() {
        let prefix = [true, false, true];
        let block = [true, true, false];
        let preview = full_attention_mask(&prefix, &block, true).unwrap();
        let commit = full_attention_mask(&prefix, &block, false).unwrap();
        let width = prefix.len() + block.len();
        assert_eq!(&preview[0..width], &[0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0, f32::NEG_INFINITY]);
        assert_eq!(&commit[0..width], &[0.0, f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert!(commit[2 * width..].iter().all(|value| value.is_infinite()));
    }

    #[test]
    fn causal_conv_advances_only_valid_tokens_and_preserves_history_on_padding() {
        let (output, history) = causal_conv_silu(
            &[1.0, 2.0, 3.0],
            3,
            1,
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0],
            4,
            &[true, false, true],
            GdnOutputDType::F32,
        ).unwrap();
        assert_eq!(output[1], 0.0);
        assert!(output[0].is_finite() && output[2].is_finite());
        assert_eq!(history, vec![0.0, 1.0, 3.0]);
    }

    #[test]
    fn preview_convolution_reads_future_context_without_mutating_history() {
        let history = vec![0.0, 0.0, 0.0];
        let before = history.clone();
        let output = block_conv_silu(
            &[1.0, 2.0, 3.0],
            3,
            1,
            &history,
            &[0.0, 0.0, 1.0, 0.0],
            &[1.0, 1.0, 1.0],
            4,
            &[true, true, true],
            GdnOutputDType::F32,
        ).unwrap();
        assert!(output[0] > output[1] && output[1] > output[2]);
        assert_eq!(history, before);
    }

    #[test]
    fn recurrent_commit_updates_fp32_state_and_emits_value_reads() {
        let mut state = vec![0.0; 4];
        let output = recurrent_commit(
            dims(),
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[3.0, 5.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[true],
            GdnOutputDType::F32,
            &mut state,
        )
        .unwrap();
        let norm = (1.0f32 + 1e-6).sqrt();
        assert_eq!(state, vec![3.0 / norm, 0.0, 5.0 / norm, 0.0]);
        assert!((output[0] - 3.0 / (norm * norm * 2.0f32.sqrt())).abs() < 1e-5);
        assert!((output[1] - 5.0 / (norm * norm * 2.0f32.sqrt())).abs() < 1e-5);
    }

    #[test]
    fn invalid_tokens_do_not_change_recurrent_state() {
        let mut state = vec![0.25, -0.5, 0.75, 1.0];
        let before = state.clone();
        let output = recurrent_commit(
            dims(),
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[3.0, 5.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[false],
            GdnOutputDType::F32,
            &mut state,
        )
        .unwrap();
        assert_eq!(state, before);
        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn shape_error_does_not_partially_mutate_state() {
        let mut state = vec![0.5; 4];
        let before = state.clone();
        let result = recurrent_commit(
            dims(),
            &[1.0],
            &[1.0, 0.0],
            &[3.0, 5.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[true],
            GdnOutputDType::F32,
            &mut state,
        );
        assert_eq!(result, Err(GdnError::ShapeMismatch));
        assert_eq!(state, before);
    }

    #[test]
    fn preview_is_read_only_and_masks_invalid_tokens() {
        let state = vec![0.1, 0.2, -0.3, 0.4];
        let before = state.clone();
        let output = block_preview(
            dims(),
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[2.0, 4.0, 9.0, 9.0],
            &[0.0; 4],
            &[true, false],
            &state,
            1.0 / 2.0f32.sqrt(),
            GdnOutputDType::F32,
            &[10.0, 10.0],
            &[5.0, 5.0],
            &[1.0, 1.0],
        )
        .unwrap();
        assert_eq!(state, before);
        assert_eq!(&output[2..], &[0.0, 0.0]);
        assert!(output[..2].iter().all(|value| value.is_finite()));
    }

    #[test]
    fn preview_with_all_tokens_invalid_returns_zero_without_nan() {
        let output = block_preview(
            dims(),
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            &[false, false],
            &[0.0; 4],
            1.0 / 2.0f32.sqrt(),
            GdnOutputDType::F32,
            &[10.0, 10.0],
            &[5.0, 5.0],
            &[1.0, 1.0],
        )
        .unwrap();
        assert_eq!(output, vec![0.0; 4]);
    }

    #[test]
    fn bf16_preview_output_is_rounded_once_after_fp32_accumulation() {
        let output = block_preview(
            dims(),
            &[1.0, 0.25],
            &[0.5, 1.0],
            &[1.234567, -0.765432],
            &[0.0, 0.0],
            &[true],
            &[0.1, 0.2, -0.3, 0.4],
            1.0 / 2.0f32.sqrt(),
            GdnOutputDType::BF16,
            &[10.0, 10.0],
            &[5.0, 5.0],
            &[1.0, 1.0],
        )
        .unwrap();
        assert!(output.iter().all(|value| {
            Bf16::from_f32(*value).to_f32().to_bits() == value.to_bits()
        }));
    }

    #[test]
    fn preview_rejects_bad_calibration_lengths() {
        let result = block_preview(
            dims(),
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 2],
            &[true],
            &[0.0; 4],
            1.0,
            GdnOutputDType::F32,
            &[10.0],
            &[5.0, 5.0],
            &[1.0, 1.0],
        );
        assert_eq!(result, Err(GdnError::ShapeMismatch));
    }
}
