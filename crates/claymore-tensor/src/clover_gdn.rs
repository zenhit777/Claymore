//! CPU reference primitives for Clover's Gated DeltaNet (GDN).
//!
//! These operations are deliberately small, backend-independent references
//! derived from the immutable Clover release. All recurrent state is FP32;
//! activation outputs are explicitly rounded to F32 or BF16.

use crate::activations::{sigmoid_scalar, silu_scalar};
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

/// Derive Clover's per-token GDN decay and update gates from the two
/// projected gate streams. `a` and `b` are row-major [tokens, value_heads];
/// `a_log` and `dt_bias` are [value_heads]. The decay is retained
/// in FP32, while beta is rounded to the activation dtype before the recurrent
/// reference widens it again, matching the pinned PyTorch path.
pub fn gdn_gates(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    tokens: usize,
    value_heads: usize,
    output_dtype: GdnOutputDType,
) -> Result<(Vec<f32>, Vec<f32>), GdnError> {
    let len = tokens
        .checked_mul(value_heads)
        .ok_or(GdnError::SizeOverflow)?;
    if tokens == 0
        || value_heads == 0
        || a.len() != len
        || b.len() != len
        || a_log.len() != value_heads
        || dt_bias.len() != value_heads
    {
        return Err(GdnError::ShapeMismatch);
    }
    let mut decay = vec![0.0f32; len];
    let mut beta = vec![0.0f32; len];
    for token in 0..tokens {
        for head in 0..value_heads {
            let index = token * value_heads + head;
            let preactivation = a[index] + dt_bias[head];
            let softplus = preactivation.max(0.0) + (-preactivation.abs()).exp().ln_1p();
            decay[index] = -a_log[head].exp() * softplus;
            let sigmoid = sigmoid_scalar(b[index]);
            beta[index] = match output_dtype {
                GdnOutputDType::F32 => sigmoid,
                GdnOutputDType::BF16 => Bf16::from_f32(sigmoid).to_f32(),
            };
        }
    }
    Ok((decay, beta))
}

/// Build Clover full-attention additive mask for one active block.
///
/// Prefix keys are visible only when marked valid. During preview every valid
/// block key is visible to every valid query (bidirectional denoising). During
/// commit, block keys are causal within the current block. Invalid query rows
/// are fully masked in both modes, matching the pinned construction in
/// `modeling_clover_moe.py:428-437`.
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
/// Invalid tokens are multiplied by the validity mask and do not advance history.
/// Finite values therefore emit zero; NaNs retain IEEE `NaN * 0 = NaN` semantics.
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
    validate_conv(
        input,
        tokens,
        channels,
        history,
        weights,
        kernel_width,
        valid,
    )?;
    let mut next_history = history.to_vec();
    let mut output = vec![0.0f32; input.len()];
    let history_width = kernel_width - 1;
    for token in 0..tokens {
        let mask = if valid[token] { 1.0 } else { 0.0 };
        for channel in 0..channels {
            let hist_base = channel * history_width;
            let mut window = vec![0.0f32; kernel_width];
            if history_width > 0 {
                window[..history_width]
                    .copy_from_slice(&next_history[hist_base..hist_base + history_width]);
            }
            let current = round_product(input[token * channels + channel], mask, output_dtype);
            window[history_width] = current;
            let mut sum = 0.0f32;
            for tap in 0..kernel_width {
                sum += round_product(
                    window[tap],
                    weights[channel * kernel_width + tap],
                    output_dtype,
                );
            }
            let activated = round_value(silu_scalar(sum), output_dtype);
            output[token * channels + channel] = round_product(activated, mask, output_dtype);
            if valid[token] && history_width > 0 {
                next_history.copy_within(hist_base + 1..hist_base + history_width, hist_base);
                next_history[hist_base + history_width - 1] = current;
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
        input,
        tokens,
        channels,
        clean_history,
        causal_weights,
        kernel_width,
        valid,
    )?;
    let right_len = channels
        .checked_mul(kernel_width.saturating_sub(1))
        .ok_or(GdnError::SizeOverflow)?;
    if right_weights.len() != right_len {
        return Err(GdnError::ShapeMismatch);
    }
    let joined_len = kernel_width - 1 + tokens;
    let mut joined = vec![
        0.0f32;
        channels
            .checked_mul(joined_len)
            .ok_or(GdnError::SizeOverflow)?
    ];
    for channel in 0..channels {
        let base = channel * joined_len;
        joined[base..base + kernel_width - 1].copy_from_slice(
            &clean_history[channel * (kernel_width - 1)..(channel + 1) * (kernel_width - 1)],
        );
        for token in 0..tokens {
            let mask = if valid[token] { 1.0 } else { 0.0 };
            joined[base + kernel_width - 1 + token] =
                round_product(input[token * channels + channel], mask, output_dtype);
        }
    }
    let mut output = vec![0.0f32; input.len()];
    for token in 0..tokens {
        let mask = if valid[token] { 1.0 } else { 0.0 };
        for channel in 0..channels {
            let base = channel * joined_len + token;
            let mut sum = 0.0f32;
            for tap in 0..kernel_width {
                sum += joined[base + tap] * causal_weights[channel * kernel_width + tap];
            }
            for distance in 1..kernel_width {
                let future = token + distance;
                if future < tokens {
                    let future_value = joined[channel * joined_len + kernel_width - 1 + future];
                    sum +=
                        future_value * right_weights[channel * (kernel_width - 1) + distance - 1];
                }
            }
            output[token * channels + channel] = round_value(silu_scalar(sum) * mask, output_dtype);
        }
    }
    Ok(output)
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
    let weight_len = channels
        .checked_mul(kernel_width)
        .ok_or(GdnError::SizeOverflow)?;
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
    round_value(lhs * rhs, dtype)
}

fn round_value(value: f32, dtype: GdnOutputDType) -> f32 {
    match dtype {
        GdnOutputDType::F32 => value,
        GdnOutputDType::BF16 => Bf16::from_f32(value).to_f32(),
    }
}

/// Commit a causal prefix token-by-token using Clover's recurrent delta rule.
///
/// Flattened layouts:
/// - q/k: [tokens, key_heads, key_dim]
/// - v: [tokens, value_heads, value_dim]
/// - g/beta: [tokens, value_heads]
/// - state: [value_heads, value_dim, key_dim] (FP32)
///
/// Invalid tokens leave the recurrent state unchanged and multiply their
/// output by zero (preserving IEEE NaN propagation). All lengths are checked
/// before the caller's state is mutated.
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
        let mask = if valid[token] { 1.0 } else { 0.0 };
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
                let current_row = next_state[row_base..row_base + dims.key_dim].to_vec();
                let mut decayed_row: Vec<f32> =
                    current_row.iter().map(|value| value * decay).collect();
                let remembered = dot(&decayed_row, &k_norm);
                let value =
                    v[(token * dims.value_heads + value_head) * dims.value_dim + value_index];
                let delta = (value - remembered) * beta[gate_index];
                for key_index in 0..dims.key_dim {
                    decayed_row[key_index] += delta * k_norm[key_index];
                }
                if valid[token] {
                    next_state[row_base..row_base + dims.key_dim].copy_from_slice(&decayed_row);
                }
                let output_state = if valid[token] {
                    &decayed_row
                } else {
                    &current_row
                };
                let query_value = dot(output_state, &q_norm) * query_scale * mask;
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
        let history_logit = history_bias[value_head] + history_c[value_head] * block_log_ratio;

        for query_token in 0..tokens {
            let query_head = value_head / repeat;
            let q_base = (query_token * dims.key_heads + query_head) * dims.key_dim;
            let q_norm = l2_normalize(&q[q_base..q_base + dims.key_dim]);

            let mut history = vec![0.0f32; dims.value_dim];
            let decay = g[query_token * dims.value_heads + value_head].exp();
            for (value_index, history_value) in history.iter_mut().enumerate() {
                let row_base = value_index * dims.key_dim;
                *history_value =
                    dot(&state[row_base..row_base + dims.key_dim], &q_norm) * scale * decay;
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
                    let value = v[(key_token * dims.value_heads + value_head) * dims.value_dim
                        + value_index];
                    let key_mask = if valid[key_token] { 1.0 } else { 0.0 };
                    local += weights[key_token] * (value * key_mask);
                }
                let result = weights[tokens] * history[value_index] + local;
                let query_mask = if valid[query_token] { 1.0 } else { 0.0 };
                output[(query_token * dims.value_heads + value_head) * dims.value_dim
                    + value_index] = result * query_mask;
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

    /// The generated fixture constants must stay pinned to the audited upstream
    /// revision and to the environment recorded in
    /// `docs/audits/m11.2-cpu-reference-manifest.json`. Regenerating the
    /// fixtures against a different upstream revision, or with a different
    /// PyTorch build, must fail here until the manifest is consciously updated.
    #[test]
    fn upstream_fixture_provenance_is_pinned() {
        use crate::clover_gdn_upstream_fixtures;
        assert_eq!(
            clover_gdn_upstream_fixtures::SOURCE_REVISION,
            "4b0402b65e523d0183a14c431955438714cb16ff"
        );
        assert_eq!(
            clover_gdn_upstream_fixtures::SOURCE_SHA256,
            "2584bd5502d6109c2cea9bee2f2ec98f147bbb8954c43858c0315eb4af9ffb0c"
        );
        assert_eq!(clover_gdn_upstream_fixtures::TORCH_VERSION, "2.12.0+cu130");
    }

    /// Commit-mode mask differential fixture against the pinned upstream mask
    /// construction (`modeling_clover_moe.py:428-437`): prefix length 3 with a
    /// hole, block length 4 straddling the block boundary, an invalid query
    /// (index 2) whose row must be fully masked, and intra-block causality.
    #[test]
    fn clover_full_attention_mask_commit_matches_pinned_upstream_fixture() {
        let prefix_valid = [true, false, true];
        let block_valid = [true, true, false, true];
        let mask = full_attention_mask(&prefix_valid, &block_valid, false).unwrap();
        let fixture = crate::clover_gdn_upstream_fixtures::MASK_COMMIT_ALLOW;
        assert_eq!(mask.len(), fixture.len());
        for (index, (&allowed, &actual)) in fixture.iter().zip(mask.iter()).enumerate() {
            let expected = if allowed > 0.5 {
                0.0
            } else {
                f32::NEG_INFINITY
            };
            assert_eq!(
                actual,
                expected,
                "row {} key {} (allowed fixture bit {allowed})",
                index / 7,
                index % 7
            );
        }
        // Explicit semantics: the invalid query row is fully masked.
        assert!(mask[14..21].iter().all(|value| *value == f32::NEG_INFINITY));
        // Intra-block causality: query 0 sees no block key past index 0 even
        // though block key 3 is valid.
        assert_eq!(mask[4], f32::NEG_INFINITY); // block key 1 for query 0
        assert_eq!(mask[3 * 7 + 6], 0.0); // block key 3 for query 3 (boundary)
    }

    fn dims() -> GdnDimensions {
        GdnDimensions {
            key_heads: 1,
            value_heads: 2,
            key_dim: 2,
            value_dim: 1,
        }
    }

    #[test]
    fn gdn_gate_derivation_matches_softplus_decay_and_sigmoid_beta() {
        let (decay, beta) = gdn_gates(
            &[0.0, 1.0],
            &[0.0, 100.0],
            &[0.0, 0.0],
            &[1.0, 0.0],
            1,
            2,
            GdnOutputDType::F32,
        )
        .unwrap();
        let softplus_one = 1.0f32 + (-1.0f32).exp().ln_1p();
        assert!((decay[0] + softplus_one).abs() < 1e-6);
        assert!((decay[1] + (1.0 + (-1.0f32).exp().ln_1p())).abs() < 1e-6);
        assert_eq!(beta, [0.5, 1.0]);
    }

    #[test]
    fn gdn_gate_bf16_beta_is_rounded_explicitly() {
        let (_, beta) = gdn_gates(
            &[0.0],
            &[0.123456],
            &[0.0],
            &[0.0],
            1,
            1,
            GdnOutputDType::BF16,
        )
        .unwrap();
        assert_eq!(
            beta[0].to_bits(),
            Bf16::from_f32(beta[0]).to_f32().to_bits()
        );
    }

    #[test]
    fn full_attention_mask_supports_empty_prefix_and_partial_final_block() {
        let preview = full_attention_mask(&[], &[true, false], true).unwrap();
        assert_eq!(
            preview,
            vec![0.0, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]
        );
        let commit = full_attention_mask(&[], &[true, false], false).unwrap();
        assert_eq!(
            commit,
            vec![0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
    }

    #[test]
    fn gdn_gates_propagate_nan_and_define_infinite_limits() {
        let (decay, beta) = gdn_gates(
            &[f32::INFINITY, f32::NAN, 0.0],
            &[f32::NEG_INFINITY, f32::NAN, f32::INFINITY],
            &[0.0],
            &[0.0],
            3,
            1,
            GdnOutputDType::F32,
        )
        .unwrap();
        assert_eq!(decay[0], f32::NEG_INFINITY);
        assert!(decay[1].is_nan());
        assert!(decay[2].is_finite());
        assert_eq!(beta[0], 0.0);
        assert!(beta[1].is_nan());
        assert_eq!(beta[2], 1.0);
    }

    #[test]
    fn causal_conv_propagates_nan_and_infinite_limits() {
        let (output, _) = causal_conv_silu(
            &[f32::NEG_INFINITY, f32::INFINITY, f32::NAN],
            3,
            1,
            &[],
            &[1.0],
            1,
            &[true, true, true],
            GdnOutputDType::F32,
        )
        .unwrap();
        assert_eq!(output[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(output[1], f32::INFINITY);
        assert!(output[2].is_nan());
    }

    #[test]
    fn convolution_shape_error_does_not_mutate_caller_history() {
        let history = vec![0.25, -0.5];
        let before = history.clone();
        let result = causal_conv_silu(
            &[1.0],
            2,
            1,
            &history,
            &[1.0, 1.0],
            3,
            &[true, true],
            GdnOutputDType::F32,
        );
        assert_eq!(result, Err(GdnError::ShapeMismatch));
        assert_eq!(history, before);
    }

    #[test]
    fn full_attention_mask_distinguishes_bidirectional_preview_and_causal_commit() {
        let prefix = [true, false, true];
        let block = [true, true, false];
        let preview = full_attention_mask(&prefix, &block, true).unwrap();
        let commit = full_attention_mask(&prefix, &block, false).unwrap();
        let width = prefix.len() + block.len();
        assert_eq!(
            &preview[0..width],
            &[0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0, f32::NEG_INFINITY]
        );
        assert_eq!(
            &preview[width..2 * width],
            &[0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0, f32::NEG_INFINITY]
        );
        assert!(preview[2 * width..]
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_negative()));
        assert_eq!(
            &commit[0..width],
            &[
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY
            ]
        );
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
        )
        .unwrap();
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
        )
        .unwrap();
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
    fn block_preview_competes_local_value_with_gate_decayed_history_bucket() {
        let dimensions = GdnDimensions {
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 1,
        };
        let output = block_preview(
            dimensions,
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[2.0],
            &[0.0],
            &[true],
            &[3.0, 0.0],
            1.0 / 2.0f32.sqrt(),
            GdnOutputDType::F32,
            &[0.0],
            &[0.0],
            &[0.0],
        )
        .unwrap();
        let expected = 1.0 + 1.5 / 2.0f32.sqrt();
        assert!((output[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn recurrent_commit_chunking_preserves_prefix_state_across_block_boundary() {
        let dimensions = GdnDimensions {
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 1,
        };
        let mut whole_state = vec![0.0; 2];
        let whole_output = recurrent_commit(
            dimensions,
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[2.0, 3.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[true, true],
            GdnOutputDType::F32,
            &mut whole_state,
        )
        .unwrap();

        let mut chunked_state = vec![0.0; 2];
        let first = recurrent_commit(
            dimensions,
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[2.0],
            &[0.0],
            &[1.0],
            &[true],
            GdnOutputDType::F32,
            &mut chunked_state,
        )
        .unwrap();
        let second = recurrent_commit(
            dimensions,
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[3.0],
            &[0.0],
            &[1.0],
            &[true],
            GdnOutputDType::F32,
            &mut chunked_state,
        )
        .unwrap();

        assert_eq!(chunked_state, whole_state);
        assert_eq!(vec![first[0], second[0]], whole_output);
    }

    #[test]
    fn recurrent_commit_preserves_nan_output_for_invalid_query_without_state_commit() {
        let dimensions = GdnDimensions {
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 1,
        };
        let mut state = vec![0.25, -0.5];
        let before_invalid = {
            let mut first_state = state.clone();
            recurrent_commit(
                dimensions,
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[2.0],
                &[0.0],
                &[1.0],
                &[true],
                GdnOutputDType::F32,
                &mut first_state,
            )
            .unwrap();
            first_state
        };
        let output = recurrent_commit(
            dimensions,
            &[1.0, 0.0, f32::NAN, 0.0],
            &[1.0, 0.0, 1.0, 0.0],
            &[2.0, 3.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[true, false],
            GdnOutputDType::F32,
            &mut state,
        )
        .unwrap();
        assert_eq!(state, before_invalid);
        assert!(output[1].is_nan());
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
    fn block_preview_preserves_nan_from_invalid_queries_and_values() {
        let dimensions = GdnDimensions {
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 1,
        };
        let state = [0.2, -0.3];
        let invalid_query = block_preview(
            dimensions,
            &[1.0, 0.0, f32::NAN, 0.0],
            &[1.0, 0.0, 1.0, 0.0],
            &[2.0, 3.0],
            &[0.0, 0.0],
            &[true, false],
            &state,
            1.0 / 2.0f32.sqrt(),
            GdnOutputDType::F32,
            &[10.0],
            &[5.0],
            &[1.0],
        )
        .unwrap();
        assert!(invalid_query[1].is_nan());

        let invalid_value = block_preview(
            dimensions,
            &[1.0, 0.0, 1.0, 0.0],
            &[1.0, 0.0, 1.0, 0.0],
            &[2.0, f32::NAN],
            &[0.0, 0.0],
            &[true, false],
            &state,
            1.0 / 2.0f32.sqrt(),
            GdnOutputDType::F32,
            &[10.0],
            &[5.0],
            &[1.0],
        )
        .unwrap();
        assert!(invalid_value[0].is_nan());
    }

    #[test]
    fn causal_conv_masks_invalid_values_but_preserves_nan_multiplication() {
        let (output, history) = causal_conv_silu(
            &[1.0, f32::NAN],
            2,
            1,
            &[],
            &[1.0],
            1,
            &[true, false],
            GdnOutputDType::F32,
        )
        .unwrap();
        assert!((output[0] - silu_scalar(1.0)).abs() < 1e-6);
        assert!(output[1].is_nan());
        assert!(history.is_empty());
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
        assert!(output
            .iter()
            .all(|value| { Bf16::from_f32(*value).to_f32().to_bits() == value.to_bits() }));
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
    fn bf16_values(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| Bf16::from_f32(*value).to_f32())
            .collect()
    }

    /// BF16 fixtures are compared with a relative bound of 2⁻⁷ (≤ 2 BF16
    /// ULPs of the expected value), not a single-ULP bound.
    fn assert_bf16_within_relative_bound(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let relative_bound = expected.abs().max(f32::MIN_POSITIVE) / 128.0;
            assert!(
                (actual - expected).abs() <= relative_bound,
                "BF16 reference mismatch at {index}: actual={actual:?}, expected={expected:?}, relative_bound(2^-7)={relative_bound}"
            );
        }
    }

    fn assert_reference_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "reference mismatch at {index}: actual={actual:?}, expected={expected:?}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn recurrent_commit_matches_pinned_upstream_cpu_fixture() {
        let q = bf16_values(&[0.3, -0.7, 1.1, 0.2, -0.4, 0.9]);
        let k = bf16_values(&[-0.2, 0.8, 0.5, -0.3, 1.0, 0.4]);
        let v = bf16_values(&[
            0.2, 0.7, -0.5, 0.3, 0.8, -0.1, 0.4, 0.9, -0.6, 0.2, 0.1, -0.8,
        ]);
        let g = [-0.2, -0.1, -0.4, -0.3, -0.1, -0.25];
        let beta = bf16_values(&[0.6, 0.4, 0.7, 0.5, 0.3, 0.8]);
        let mut state = vec![0.1, -0.2, 0.3, 0.05, -0.15, 0.25, 0.2, -0.1];
        let output = recurrent_commit(
            GdnDimensions {
                key_heads: 1,
                value_heads: 2,
                key_dim: 2,
                value_dim: 2,
            },
            &q,
            &k,
            &v,
            &g,
            &beta,
            &[true, false, true],
            GdnOutputDType::BF16,
            &mut state,
        )
        .unwrap();
        assert_bf16_within_relative_bound(&output, crate::clover_gdn_upstream_fixtures::REC_OUTPUT);
        assert_reference_close(
            &state,
            crate::clover_gdn_upstream_fixtures::REC_STATE_OUT,
            2e-6,
        );
    }

    #[test]
    fn gdn_block_preview_matches_pinned_upstream_cpu_fixture() {
        let q = bf16_values(&[0.3, -0.7, 1.1, 0.2, -0.4, 0.9]);
        let k = bf16_values(&[-0.2, 0.8, 0.5, -0.3, 1.0, 0.4]);
        let v = bf16_values(&[
            0.2, 0.7, -0.5, 0.3, 0.8, -0.1, 0.4, 0.9, -0.6, 0.2, 0.1, -0.8,
        ]);
        let g = [-0.2, -0.1, -0.4, -0.3, -0.1, -0.25];
        let state = [0.1, -0.2, 0.3, 0.05, -0.15, 0.25, 0.2, -0.1];
        let output = block_preview(
            GdnDimensions {
                key_heads: 1,
                value_heads: 2,
                key_dim: 2,
                value_dim: 2,
            },
            &q,
            &k,
            &v,
            &g,
            &[true, false, true],
            &state,
            1.0 / 2.0f32.sqrt(),
            GdnOutputDType::BF16,
            &[10.0, 9.0],
            &[5.0, 4.5],
            &[1.0, 0.75],
        )
        .unwrap();
        assert_bf16_within_relative_bound(
            &output,
            crate::clover_gdn_upstream_fixtures::PREVIEW_OUTPUT,
        );
        assert_eq!(state, [0.1, -0.2, 0.3, 0.05, -0.15, 0.25, 0.2, -0.1]);
    }

    #[test]
    fn causal_and_preview_convolutions_match_pinned_upstream_cpu_fixtures() {
        let raw = bf16_values(&[0.2, -0.4, 0.0, 0.0, 0.7, 0.1, -0.3, 0.6]);
        let history = bf16_values(&[0.1, -0.2, 0.3, -0.4, 0.2, 0.05]);
        let weights = bf16_values(&[0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3]);
        let (causal, next_history) = causal_conv_silu(
            &raw,
            4,
            2,
            &history,
            &weights,
            4,
            &[true, false, true, true],
            GdnOutputDType::BF16,
        )
        .unwrap();
        assert_bf16_within_relative_bound(
            &causal,
            crate::clover_gdn_upstream_fixtures::CAUSAL_OUTPUT,
        );
        assert_bf16_within_relative_bound(
            &next_history,
            crate::clover_gdn_upstream_fixtures::CAUSAL_STATE_OUT,
        );

        let block_raw = bf16_values(&[0.2, -0.4, 0.7, 0.1, -0.3, 0.6]);
        let right = bf16_values(&[0.15, -0.25, 0.35, -0.1, 0.2, 0.3]);
        let block = block_conv_silu(
            &block_raw,
            3,
            2,
            &history,
            &weights,
            &right,
            4,
            &[true, true, false],
            GdnOutputDType::BF16,
        )
        .unwrap();
        assert_bf16_within_relative_bound(
            &block,
            crate::clover_gdn_upstream_fixtures::BLOCK_OUTPUT,
        );
        assert_eq!(history, bf16_values(&[0.1, -0.2, 0.3, -0.4, 0.2, 0.05]));
    }

    #[test]
    fn gdn_gates_match_pinned_upstream_cpu_fixture() {
        let a = bf16_values(&[0.2, -0.4, 0.7, 0.1, -0.3, 0.6]);
        let b = bf16_values(&[0.3, -0.8, 0.5, 0.2, -0.1, 0.9]);
        let (decay, beta) = gdn_gates(
            &a,
            &b,
            &[0.1, -0.2],
            &[1.0, 0.8],
            3,
            2,
            GdnOutputDType::BF16,
        )
        .unwrap();
        assert_reference_close(
            &decay,
            crate::clover_gdn_upstream_fixtures::GATE_DECAY,
            1e-6,
        );
        assert_bf16_within_relative_bound(&beta, crate::clover_gdn_upstream_fixtures::GATE_BETA);
    }
}
