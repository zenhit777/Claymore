//! CPU reference primitives required by the DiffusionGemma graph (M10.2).
//!
//! These operations are deliberately graph-specific: they cover exactly the
//! pieces audited in the pinned revision (linear projections, rotary
//! embeddings, additive masks, grouped-query attention, logit soft-capping)
//! and nothing else. No generic tensor framework is introduced.
//!
//! Numerical policy (identical to the M8 operations):
//! - inputs are read in their exact stored representation and widened to FP32;
//! - every product and reduction is accumulated in FP32;
//! - the result is rounded back to the input dtype once, at the output;
//! - FP16 is never accepted and nothing is silently cast or sanitised.
//!
//! Layouts (batch dimension omitted, the model runs with batch size 1):
//! - projections: `input [tokens, in_features]`, `weight [out_features, in_features]`;
//! - attention: `q [query_tokens, heads, head_dim]`, `k/v [key_tokens, kv_heads, head_dim]`,
//!   optional additive `mask [query_tokens, key_tokens]`;
//! - rotary tables: `cos/sin [tokens, head_dim]`.

use claymore_device::{DType, Device};

use crate::ops::{read_numeric, write_numeric, OpError};
use crate::Tensor;

/// RoPE variants used by the pinned checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// `transformers` `_compute_default_rope_parameters`.
    Default,
    /// `transformers` `_compute_proportional_rope_parameters`.
    Proportional,
}

impl RopeType {
    pub const fn name(self) -> &'static str {
        match self {
            RopeType::Default => "default",
            RopeType::Proportional => "proportional",
        }
    }
}

/// Additive attention mask value used for blocked keys.
pub const MASK_BLOCKED: f32 = f32::NEG_INFINITY;

/// Inverse RoPE frequencies for one head dimension.
///
/// `default` follows the upstream implementation:
/// `inv_freq[j] = base ** (-2j / dim)` with `dim = int(head_dim * partial)`.
/// `proportional` keeps the full `head_dim / 2` frequencies but zeroes the
/// trailing ones, so only the first `partial` fraction of the head is rotated.
pub fn rope_inv_freq(
    rope_type: RopeType,
    head_dim: usize,
    theta: f64,
    partial_factor: f64,
) -> Result<Vec<f32>, OpError> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(OpError::InvalidRopeParameter(format!(
            "head_dim must be a non-zero even number, got {head_dim}"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(OpError::InvalidRopeParameter(format!(
            "rope theta must be finite and positive, got {theta}"
        )));
    }
    if !partial_factor.is_finite() || partial_factor <= 0.0 || partial_factor > 1.0 {
        return Err(OpError::InvalidRopeParameter(format!(
            "partial rotary factor must be within (0, 1], got {partial_factor}"
        )));
    }

    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f32; half];
    match rope_type {
        RopeType::Default => {
            let dim = (head_dim as f64 * partial_factor).floor() as usize;
            if dim == 0 || dim % 2 != 0 || dim > head_dim {
                return Err(OpError::InvalidRopeParameter(format!(
                    "default RoPE dimension {dim} is invalid for head_dim {head_dim}"
                )));
            }
            for (index, value) in inv_freq.iter_mut().enumerate().take(dim / 2) {
                let exponent = (2 * index) as f64 / dim as f64;
                *value = (theta.powf(-exponent)) as f32;
            }
            for value in inv_freq.iter_mut().skip(dim / 2) {
                *value = 0.0;
            }
        }
        RopeType::Proportional => {
            let rope_angles = (partial_factor * head_dim as f64).floor() as usize / 2;
            for (index, value) in inv_freq.iter_mut().enumerate().take(rope_angles) {
                let exponent = (2 * index) as f64 / head_dim as f64;
                *value = (theta.powf(-exponent)) as f32;
            }
            for value in inv_freq.iter_mut().skip(rope_angles) {
                *value = 0.0;
            }
        }
    }
    Ok(inv_freq)
}

/// Cosine/sine tables for the given positions, shaped `[tokens, head_dim]`.
///
/// The upstream implementation concatenates the frequencies twice, so the
/// rotation pairs are `(i, i + head_dim / 2)`.
pub fn rope_tables(
    rope_type: RopeType,
    positions: &[u32],
    head_dim: usize,
    theta: f64,
    partial_factor: f64,
) -> Result<(Vec<f32>, Vec<f32>), OpError> {
    let inv_freq = rope_inv_freq(rope_type, head_dim, theta, partial_factor)?;
    let mut cos = Vec::with_capacity(positions.len() * head_dim);
    let mut sin = Vec::with_capacity(positions.len() * head_dim);
    for position in positions {
        for frequency in &inv_freq {
            let angle = *position as f32 * *frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
        for frequency in &inv_freq {
            let angle = *position as f32 * *frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((cos, sin))
}

/// Dense linear projection without bias: `out[t, o] = sum_i x[t, i] * w[o, i]`.
pub fn linear(device: &dyn Device, input: &Tensor, weight: &Tensor) -> Result<Tensor, OpError> {
    let input_shape = input.layout().shape();
    let weight_shape = weight.layout().shape();
    if input_shape.len() != 2 || weight_shape.len() != 2 {
        return Err(OpError::ShapeMismatch);
    }
    let (tokens, in_features) = (input_shape[0], input_shape[1]);
    let (out_features, weight_in) = (weight_shape[0], weight_shape[1]);
    if in_features != weight_in {
        return Err(OpError::ShapeMismatch);
    }
    let source = read_numeric(device, input)?;
    let weights = read_numeric(device, weight)?;
    if source.dtype() != weights.dtype() {
        return Err(OpError::MixedDType);
    }
    let dtype = source.dtype();
    let x = source.to_f32();
    let w = weights.to_f32();
    let mut output = vec![
        0.0f32;
        tokens
            .checked_mul(out_features)
            .ok_or(OpError::ShapeMismatch)?
    ];
    for token in 0..tokens {
        for out in 0..out_features {
            let mut accumulator = 0.0f32;
            for inner in 0..in_features {
                accumulator = x[token * in_features + inner]
                    .mul_add(w[out * in_features + inner], accumulator);
            }
            output[token * out_features + out] = accumulator;
        }
    }
    write_numeric(device, &output, vec![tokens, out_features], dtype)
}

/// Apply rotary embeddings to `[tokens, heads, head_dim]` activations.
///
/// `cos`/`sin` are `[tokens, head_dim]` tables with the same dtype as the
/// input. The rotation is the upstream `rotate_half` convention:
/// `out = x * cos + rotate_half(x) * sin`.
pub fn rotary_embed(
    device: &dyn Device,
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor, OpError> {
    let shape = input.layout().shape();
    if shape.len() != 3 {
        return Err(OpError::ShapeMismatch);
    }
    let (tokens, heads, head_dim) = (shape[0], shape[1], shape[2]);
    if head_dim % 2 != 0 {
        return Err(OpError::ShapeMismatch);
    }
    let table_shape = cos.layout().shape();
    if table_shape != [tokens, head_dim] || sin.layout().shape() != [tokens, head_dim] {
        return Err(OpError::ShapeMismatch);
    }
    let source = read_numeric(device, input)?;
    let cosines = read_numeric(device, cos)?;
    let sines = read_numeric(device, sin)?;
    if source.dtype() != cosines.dtype() || source.dtype() != sines.dtype() {
        return Err(OpError::MixedDType);
    }
    let dtype = source.dtype();
    let x = source.to_f32();
    let cos_table = cosines.to_f32();
    let sin_table = sines.to_f32();
    let half = head_dim / 2;
    let mut output = vec![0.0f32; x.len()];
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for index in 0..half {
                let first = x[base + index];
                let second = x[base + half + index];
                let cosine = cos_table[token * head_dim + index];
                let sine = sin_table[token * head_dim + index];
                let cosine_second = cos_table[token * head_dim + half + index];
                let sine_second = sin_table[token * head_dim + half + index];
                output[base + index] = first * cosine - second * sine;
                output[base + half + index] = second * cosine_second + first * sine_second;
            }
        }
    }
    write_numeric(device, &output, shape.to_vec(), dtype)
}

/// Grouped-query multi-head attention with an optional additive mask.
///
/// * `q` is `[query_tokens, heads, head_dim]`;
/// * `k`/`v` are `[key_tokens, kv_heads, head_dim]` with `heads % kv_heads == 0`;
/// * `mask` is `[query_tokens, key_tokens]` and is added to the scores.
///
/// Scores, the softmax and the value reduction are evaluated in FP32. As in the
/// pinned reference, the probabilities are rounded to the input dtype before the
/// value reduction. A fully masked query row produces a zero output row instead
/// of NaN; mask builders reject such masks.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    device: &dyn Device,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor, OpError> {
    if !scale.is_finite() || scale == 0.0 {
        return Err(OpError::InvalidAttentionScale);
    }
    let query_shape = query.layout().shape();
    let key_shape = key.layout().shape();
    let value_shape = value.layout().shape();
    if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
        return Err(OpError::ShapeMismatch);
    }
    if key_shape != value_shape {
        return Err(OpError::ShapeMismatch);
    }
    let (query_tokens, heads, head_dim) = (query_shape[0], query_shape[1], query_shape[2]);
    let (key_tokens, kv_heads, kv_dim) = (key_shape[0], key_shape[1], key_shape[2]);
    if kv_dim != head_dim {
        return Err(OpError::ShapeMismatch);
    }
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
        return Err(OpError::InvalidHeadConfiguration(format!(
            "heads {heads} must be a non-zero multiple of kv_heads {kv_heads}"
        )));
    }
    if let Some(mask) = mask {
        if mask.layout().shape() != [query_tokens, key_tokens] {
            return Err(OpError::ShapeMismatch);
        }
    }
    let query_values = read_numeric(device, query)?;
    let key_values = read_numeric(device, key)?;
    let value_values = read_numeric(device, value)?;
    if query_values.dtype() != key_values.dtype() || query_values.dtype() != value_values.dtype() {
        return Err(OpError::MixedDType);
    }
    let dtype = query_values.dtype();
    let mask_values = match mask {
        Some(mask) => {
            let values = read_numeric(device, mask)?;
            if values.dtype() != dtype {
                return Err(OpError::MixedDType);
            }
            Some(values.to_f32())
        }
        None => None,
    };

    let q = query_values.to_f32();
    let k = key_values.to_f32();
    let v = value_values.to_f32();
    let groups = heads / kv_heads;
    let mut output = vec![0.0f32; query_tokens * heads * head_dim];
    let mut scores = vec![0.0f32; key_tokens];
    let mut probabilities = vec![0.0f32; key_tokens];
    for token in 0..query_tokens {
        for head in 0..heads {
            let kv_head = head / groups;
            let q_base = (token * heads + head) * head_dim;
            for key_index in 0..key_tokens {
                let k_base = (key_index * kv_heads + kv_head) * head_dim;
                let mut accumulator = 0.0f32;
                for dim in 0..head_dim {
                    accumulator = q[q_base + dim].mul_add(k[k_base + dim], accumulator);
                }
                let mut score = accumulator * scale;
                if let Some(mask_values) = &mask_values {
                    score += mask_values[token * key_tokens + key_index];
                }
                scores[key_index] = score;
            }
            softmax_row(&scores, &mut probabilities);
            let rounded = round_to_dtype(&probabilities, dtype);
            let out_row = &mut output[q_base..q_base + head_dim];
            for (key_index, probability) in rounded.iter().enumerate() {
                let v_base = (key_index * kv_heads + kv_head) * head_dim;
                for dim in 0..head_dim {
                    out_row[dim] = probability.mul_add(v[v_base + dim], out_row[dim]);
                }
            }
        }
    }
    write_numeric(device, &output, query_shape.to_vec(), dtype)
}

/// Stable softmax over one row, evaluated in FP32.
///
/// A fully masked row (every score is `-inf`) yields an all-zero probability
/// row instead of NaN. NaN or infinite scores propagate as NaN, exactly like
/// the reference implementation: nothing is silently sanitised.
fn softmax_row(scores: &[f32], output: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    let mut has_nan = false;
    for score in scores {
        if score.is_nan() {
            has_nan = true;
            break;
        }
        if *score > max {
            max = *score;
        }
    }
    if has_nan || max == f32::INFINITY {
        output.fill(f32::NAN);
        return;
    }
    if max == f32::NEG_INFINITY {
        output.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for (index, score) in scores.iter().enumerate() {
        let value = (*score - max).exp();
        output[index] = value;
        sum += value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        output.fill(f32::NAN);
        return;
    }
    let reciprocal = sum.recip();
    for value in output.iter_mut() {
        *value *= reciprocal;
    }
}

/// Round FP32 values to the storage precision of `dtype`.
fn round_to_dtype(values: &[f32], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::BF16 => values
            .iter()
            .map(|value| crate::Bf16::from_f32(*value).to_f32())
            .collect(),
        _ => values.to_vec(),
    }
}

/// Logit soft-capping: `out = cap * tanh(x / cap)`.
///
/// The pinned checkpoint uses `cap = 30` on the final logits.
pub fn soft_cap(device: &dyn Device, input: &Tensor, cap: f32) -> Result<Tensor, OpError> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(OpError::InvalidSoftCap);
    }
    let source = read_numeric(device, input)?;
    let dtype = source.dtype();
    let values = source.to_f32();
    let output: Vec<f32> = values
        .iter()
        .map(|value| cap * (*value / cap).tanh())
        .collect();
    write_numeric(device, &output, input.layout().shape().to_vec(), dtype)
}

/// Multiply a tensor by an explicit factor tensor.
///
/// The pinned checkpoint scales every decoder layer output by a learned
/// per-layer scalar (`layer_scalar`, BF16 shape `[1]`). The factor is always an
/// explicit tensor: a factor of shape `[1]` is broadcast over the whole input,
/// a factor of shape `[width]` is broadcast over the last dimension. Products
/// are evaluated in FP32 and rounded once to the input dtype.
pub fn scale(device: &dyn Device, input: &Tensor, factor: &Tensor) -> Result<Tensor, OpError> {
    let shape = input.layout().shape();
    if shape.is_empty() {
        return Err(OpError::InvalidRank);
    }
    let width = shape[shape.len() - 1];
    let factor_shape = factor.layout().shape();
    if factor_shape != [1] && factor_shape != [width] {
        return Err(OpError::ScaleMismatch);
    }
    let source = read_numeric(device, input)?;
    let factors = read_numeric(device, factor)?;
    if source.dtype() != factors.dtype() {
        return Err(OpError::MixedDType);
    }
    let dtype = source.dtype();
    let values = source.to_f32();
    let factor_values = factors.to_f32();
    let factor_width = factor_values.len();
    let mut output = vec![0.0f32; values.len()];
    if factor_width == 1 {
        for (index, value) in values.iter().enumerate() {
            output[index] = value * factor_values[0];
        }
    } else {
        let rows = values.len() / factor_width;
        for row in 0..rows {
            for (column, factor_value) in factor_values.iter().enumerate() {
                let index = row * factor_width + column;
                output[index] = values[index] * factor_value;
            }
        }
    }
    write_numeric(device, &output, shape.to_vec(), dtype)
}

/// Additive mask where every query attends to every key.
pub fn bidirectional_mask(query_tokens: usize, key_tokens: usize) -> Vec<f32> {
    vec![0.0; query_tokens * key_tokens]
}

/// Additive causal mask: query `q` (absolute position `query_offset + q`)
/// attends to keys `0..=query_offset + q`.
pub fn causal_mask(query_tokens: usize, key_tokens: usize, query_offset: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; query_tokens * key_tokens];
    for query in 0..query_tokens {
        let absolute = query_offset + query;
        for key in 0..key_tokens {
            if key > absolute {
                mask[query * key_tokens + key] = MASK_BLOCKED;
            }
        }
    }
    mask
}

/// Additive sliding-window mask: query `q` attends to the `window` most recent
/// keys, including itself.
pub fn sliding_window_mask(
    query_tokens: usize,
    key_tokens: usize,
    window: usize,
    query_offset: usize,
) -> Vec<f32> {
    let mut mask = vec![0.0f32; query_tokens * key_tokens];
    for query in 0..query_tokens {
        let absolute = query_offset + query;
        let oldest = absolute.saturating_sub(window.saturating_sub(1));
        for key in 0..key_tokens {
            if key > absolute || key < oldest {
                mask[query * key_tokens + key] = MASK_BLOCKED;
            }
        }
    }
    mask
}

/// Additive mask for a decoder canvas attending to an encoder prefix and to
/// the whole canvas (bidirectional inside the canvas).
///
/// `prefix_tokens` is the number of encoder keys; the remaining keys are the
/// canvas positions.
pub fn canvas_prefix_mask(
    query_tokens: usize,
    prefix_tokens: usize,
    canvas_tokens: usize,
) -> Vec<f32> {
    let key_tokens = prefix_tokens + canvas_tokens;
    let mut mask = vec![0.0f32; query_tokens * key_tokens];
    for query in 0..query_tokens {
        for key in 0..key_tokens {
            let visible = if key < prefix_tokens {
                true
            } else {
                // Canvas keys are bidirectional but never visible before the
                // canvas itself exists: the canvas block is always complete.
                true
            };
            if !visible {
                mask[query * key_tokens + key] = MASK_BLOCKED;
            }
        }
    }
    mask
}

/// Mask that blocks the trailing `key_tokens - valid_keys` positions.
pub fn padding_mask(query_tokens: usize, key_tokens: usize, valid_keys: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; query_tokens * key_tokens];
    if valid_keys >= key_tokens {
        return mask;
    }
    for query in 0..query_tokens {
        for key in valid_keys..key_tokens {
            mask[query * key_tokens + key] = MASK_BLOCKED;
        }
    }
    mask
}

/// Reject masks that would leave a query row without any visible key.
pub fn validate_mask(mask: &[f32], query_tokens: usize, key_tokens: usize) -> Result<(), OpError> {
    if mask.len() != query_tokens * key_tokens {
        return Err(OpError::ShapeMismatch);
    }
    for query in 0..query_tokens {
        let row = &mask[query * key_tokens..(query + 1) * key_tokens];
        if row.iter().all(|value| !value.is_finite()) {
            return Err(OpError::ShapeMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::Bf16;
    use claymore_core::MemoryTier;
    use claymore_device::{CpuDevice, Device};

    fn tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut buffer = device
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        device.write(buffer.as_mut(), 0, &bytes).unwrap();
        Tensor::new(buffer, shape, DType::F32).unwrap()
    }

    fn bf16_tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| Bf16::from_f32(*value).to_ne_bytes())
            .collect();
        let mut buffer = device
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        device.write(buffer.as_mut(), 0, &bytes).unwrap();
        Tensor::new(buffer, shape, DType::BF16).unwrap()
    }

    fn values(device: &CpuDevice, tensor: &Tensor) -> Vec<f32> {
        match read_numeric(device, tensor).unwrap() {
            crate::ops::Numeric::F32(values) => values,
            crate::ops::Numeric::BF16(values) => {
                values.iter().map(|value| value.to_f32()).collect()
            }
        }
    }

    #[test]
    fn default_rope_matches_the_upstream_formula() {
        let inv_freq = rope_inv_freq(RopeType::Default, 8, 10_000.0, 1.0).unwrap();
        assert_eq!(inv_freq.len(), 4);
        for (index, value) in inv_freq.iter().enumerate() {
            let expected = 10_000f64.powf(-((2 * index) as f64) / 8.0) as f32;
            assert!((value - expected).abs() < 1e-6, "{value} vs {expected}");
        }
    }

    #[test]
    fn proportional_rope_keeps_the_full_frequency_vector_and_zeroes_the_tail() {
        let inv_freq = rope_inv_freq(RopeType::Proportional, 8, 1_000_000.0, 0.25).unwrap();
        assert_eq!(inv_freq.len(), 4);
        // `rope_angles = int(0.25 * 8 // 2) = 1`: a single rotated frequency.
        assert_eq!(inv_freq[0], 1.0);
        assert_eq!(inv_freq[1], 0.0);
        assert_eq!(inv_freq[2], 0.0);
        assert_eq!(inv_freq[3], 0.0);

        let inv_freq = rope_inv_freq(RopeType::Proportional, 512, 1_000_000.0, 0.25).unwrap();
        assert_eq!(inv_freq.len(), 256);
        assert_eq!(inv_freq[0], 1.0);
        assert!((inv_freq[1] - 1_000_000f64.powf(-2.0 / 512.0) as f32).abs() < 1e-6);
        assert_eq!(inv_freq[64], 0.0);
        assert_eq!(inv_freq[255], 0.0);
    }

    #[test]
    fn rope_parameters_are_validated() {
        assert!(rope_inv_freq(RopeType::Default, 0, 10_000.0, 1.0).is_err());
        assert!(rope_inv_freq(RopeType::Default, 7, 10_000.0, 1.0).is_err());
        assert!(rope_inv_freq(RopeType::Default, 8, 0.0, 1.0).is_err());
        assert!(rope_inv_freq(RopeType::Default, 8, 10_000.0, 0.0).is_err());
        assert!(rope_inv_freq(RopeType::Default, 8, 10_000.0, 1.5).is_err());
    }

    #[test]
    fn rope_tables_repeat_the_frequencies_and_are_identity_at_position_zero() {
        let (cos, sin) = rope_tables(RopeType::Default, &[0, 1], 4, 10_000.0, 1.0).unwrap();
        assert_eq!(cos.len(), 8);
        assert_eq!(sin.len(), 8);
        assert!(cos[0..4].iter().all(|value| (*value - 1.0).abs() < 1e-6));
        assert!(sin[0..4].iter().all(|value| value.abs() < 1e-6));
        // The table concatenates the frequency vector twice.
        assert_eq!(cos[0], cos[2]);
        assert_eq!(sin[0], sin[2]);
    }

    #[test]
    fn rotary_embedding_is_identity_at_position_zero_and_rotates_pairs() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let cos = tensor(&device, vec![1, 4], &[1.0, 1.0, 1.0, 1.0]);
        let sin = tensor(&device, vec![1, 4], &[0.0, 0.0, 0.0, 0.0]);
        let out = values(&device, &rotary_embed(&device, &input, &cos, &sin).unwrap());
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);

        // A 90 degree rotation on the first pair only.
        let cos = tensor(&device, vec![1, 4], &[0.0, 1.0, 1.0, 1.0]);
        let sin = tensor(&device, vec![1, 4], &[1.0, 0.0, 0.0, 0.0]);
        let out = values(&device, &rotary_embed(&device, &input, &cos, &sin).unwrap());
        assert!((out[0] + 3.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn rotary_embedding_rejects_mismatched_tables() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let cos = tensor(&device, vec![1, 2], &[1.0, 1.0]);
        let sin = tensor(&device, vec![1, 4], &[0.0; 4]);
        assert_eq!(
            rotary_embed(&device, &input, &cos, &sin).unwrap_err(),
            OpError::ShapeMismatch
        );
    }

    #[test]
    fn linear_matches_the_explicit_projection() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let weight = tensor(&device, vec![2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
        let out = values(&device, &linear(&device, &input, &weight).unwrap());
        assert_eq!(out, [1.0, 5.0, 4.0, 11.0]);
    }

    #[test]
    fn linear_rejects_shape_and_dtype_mismatches() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![2, 3], &[1.0; 6]);
        let weight = tensor(&device, vec![2, 2], &[1.0; 4]);
        assert_eq!(
            linear(&device, &input, &weight).unwrap_err(),
            OpError::ShapeMismatch
        );
        let bf16_weight = bf16_tensor(&device, vec![2, 3], &[1.0; 6]);
        assert_eq!(
            linear(&device, &input, &bf16_weight).unwrap_err(),
            OpError::MixedDType
        );
    }

    #[test]
    fn attention_matches_a_hand_computed_single_head_case() {
        let device = CpuDevice::new();
        let q = tensor(&device, vec![1, 1, 2], &[1.0, 0.0]);
        let k = tensor(&device, vec![2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = tensor(&device, vec![2, 1, 2], &[2.0, 0.0, 0.0, 4.0]);
        let out = values(&device, &attention(&device, &q, &k, &v, None, 1.0).unwrap());
        // scores = [1, 0] -> softmax = [e/(e+1), 1/(e+1)].
        let first = 1.0f32 / (1.0 + (-1.0f32).exp());
        let second = 1.0 - first;
        assert!((out[0] - 2.0 * first).abs() < 1e-6);
        assert!((out[1] - 4.0 * second).abs() < 1e-6);
    }

    #[test]
    fn attention_applies_the_additive_mask() {
        let device = CpuDevice::new();
        let q = tensor(&device, vec![1, 1, 2], &[1.0, 0.0]);
        let k = tensor(&device, vec![2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = tensor(&device, vec![2, 1, 2], &[2.0, 0.0, 0.0, 4.0]);
        let mask = tensor(&device, vec![1, 2], &[0.0, MASK_BLOCKED]);
        let out = values(
            &device,
            &attention(&device, &q, &k, &v, Some(&mask), 1.0).unwrap(),
        );
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
    }

    #[test]
    fn attention_maps_grouped_query_heads_to_the_right_kv_head() {
        let device = CpuDevice::new();
        // Two query heads, one KV head: both heads read the same key/value.
        let q = tensor(&device, vec![1, 2, 1], &[1.0, 3.0]);
        let k = tensor(&device, vec![1, 1, 1], &[1.0]);
        let v = tensor(&device, vec![1, 1, 1], &[5.0]);
        let out = values(&device, &attention(&device, &q, &k, &v, None, 1.0).unwrap());
        assert_eq!(out, [5.0, 5.0]);
    }

    #[test]
    fn attention_rejects_invalid_shapes_scales_and_head_counts() {
        let device = CpuDevice::new();
        let q = tensor(&device, vec![1, 3, 2], &[1.0; 6]);
        let k = tensor(&device, vec![1, 2, 2], &[1.0; 4]);
        let v = tensor(&device, vec![1, 2, 2], &[1.0; 4]);
        assert!(matches!(
            attention(&device, &q, &k, &v, None, 1.0).unwrap_err(),
            OpError::InvalidHeadConfiguration(_)
        ));

        let q = tensor(&device, vec![1, 2, 2], &[1.0; 4]);
        let k = tensor(&device, vec![1, 1, 2], &[1.0; 2]);
        let v = tensor(&device, vec![1, 1, 2], &[1.0; 2]);
        assert_eq!(
            attention(&device, &q, &k, &v, None, 0.0).unwrap_err(),
            OpError::InvalidAttentionScale
        );
        let mask = tensor(&device, vec![2, 2], &[0.0; 4]);
        assert_eq!(
            attention(&device, &q, &k, &v, Some(&mask), 1.0).unwrap_err(),
            OpError::ShapeMismatch
        );
    }

    #[test]
    fn fully_masked_rows_produce_zeros_instead_of_nan() {
        let device = CpuDevice::new();
        let q = tensor(&device, vec![1, 1, 2], &[1.0, 0.0]);
        let k = tensor(&device, vec![2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = tensor(&device, vec![2, 1, 2], &[2.0, 0.0, 0.0, 4.0]);
        let mask = tensor(&device, vec![1, 2], &[MASK_BLOCKED, MASK_BLOCKED]);
        let out = values(
            &device,
            &attention(&device, &q, &k, &v, Some(&mask), 1.0).unwrap(),
        );
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn attention_propagates_nan_without_sanitising_it() {
        let device = CpuDevice::new();
        let q = tensor(&device, vec![1, 1, 2], &[f32::NAN, 0.0]);
        let k = tensor(&device, vec![1, 1, 2], &[1.0, 0.0]);
        let v = tensor(&device, vec![1, 1, 2], &[2.0, 0.0]);
        let out = values(&device, &attention(&device, &q, &k, &v, None, 1.0).unwrap());
        assert!(out.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn soft_cap_saturates_at_the_cap() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![3], &[0.0, 1000.0, -1000.0]);
        let out = values(&device, &soft_cap(&device, &input, 30.0).unwrap());
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 30.0).abs() < 1e-3);
        assert!((out[2] + 30.0).abs() < 1e-3);
        assert!(soft_cap(&device, &input, 0.0).is_err());
        assert!(soft_cap(&device, &input, f32::NAN).is_err());
    }

    #[test]
    fn scale_broadcasts_an_explicit_factor() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scalar = tensor(&device, vec![1], &[0.5]);
        let out = values(&device, &scale(&device, &input, &scalar).unwrap());
        assert_eq!(out, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);

        let per_column = tensor(&device, vec![3], &[1.0, 2.0, 3.0]);
        let out = values(&device, &scale(&device, &input, &per_column).unwrap());
        assert_eq!(out, [1.0, 4.0, 9.0, 4.0, 10.0, 18.0]);

        // A factor that matches neither `[1]` nor the last dimension is refused.
        let bad = tensor(&device, vec![2], &[1.0, 2.0]);
        assert_eq!(
            scale(&device, &input, &bad).unwrap_err(),
            OpError::ScaleMismatch
        );
    }

    #[test]
    fn masks_describe_causal_sliding_and_padding_visibility() {
        let causal = causal_mask(3, 3, 0);
        assert_eq!(causal[0], 0.0);
        assert_eq!(causal[1], MASK_BLOCKED);
        assert_eq!(causal[4], 0.0);
        assert_eq!(causal[5], MASK_BLOCKED);

        let sliding = sliding_window_mask(3, 3, 2, 0);
        assert_eq!(sliding[0], 0.0);
        assert_eq!(sliding[1], MASK_BLOCKED);
        assert_eq!(sliding[2], MASK_BLOCKED);
        assert_eq!(sliding[3], 0.0);
        assert_eq!(sliding[4], 0.0);
        assert_eq!(sliding[5], MASK_BLOCKED);
        assert_eq!(sliding[6], MASK_BLOCKED);
        assert_eq!(sliding[7], 0.0);
        assert_eq!(sliding[8], 0.0);

        let padded = padding_mask(2, 3, 2);
        assert_eq!(padded[0], 0.0);
        assert_eq!(padded[2], MASK_BLOCKED);
        assert_eq!(padded[5], MASK_BLOCKED);

        let bidirectional = bidirectional_mask(2, 2);
        assert!(bidirectional.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn mask_validation_rejects_fully_masked_rows() {
        let mask = vec![0.0, MASK_BLOCKED, MASK_BLOCKED, MASK_BLOCKED];
        assert!(validate_mask(&mask, 2, 2).is_err());
        let mask = vec![0.0, MASK_BLOCKED, 0.0, 0.0];
        assert!(validate_mask(&mask, 2, 2).is_ok());
        assert!(validate_mask(&mask, 1, 2).is_err());
    }

    #[test]
    fn canvas_prefix_mask_keeps_encoder_and_canvas_visible() {
        let mask = canvas_prefix_mask(2, 3, 2);
        assert_eq!(mask.len(), 10);
        assert!(mask.iter().all(|value| *value == 0.0));
    }
}
