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
///
/// On BF16 the upstream tensor operations round each product to BF16 before
/// the sum, and round the sum again; that boundary is preserved here so the
/// CUDA kernel and the CPU reference agree bit for bit. F32 keeps its single
/// final rounding.
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
    // The pinned upstream evaluates `(x * cos) + (rotate_half(x) * sin)` on
    // BF16 tensors: each product is rounded to BF16 before the sum, and the
    // sum is rounded again. F32 keeps its single final rounding.
    let bf16 = dtype == DType::BF16;
    let round = |value: f32| {
        if bf16 {
            crate::Bf16::from_f32(value).to_f32()
        } else {
            value
        }
    };
    let half = head_dim / 2;
    let mut output = vec![0.0f32; x.len()];
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for index in 0..half {
                let first = x[base + index];
                let second = x[base + half + index];
                // The upstream `rotate_half` convention pairs `(i, i + half)`
                // and reads the table entry of each half independently. The
                // generated tables repeat the frequencies, so both entries
                // carry the same value.
                let cosine = cos_table[token * head_dim + index];
                let sine = sin_table[token * head_dim + index];
                let cosine_second = cos_table[token * head_dim + half + index];
                let sine_second = sin_table[token * head_dim + half + index];
                output[base + index] = round(round(first * cosine) - round(second * sine));
                output[base + half + index] =
                    round(round(second * cosine_second) + round(first * sine_second));
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
/// Scores, the softmax and the value reduction are evaluated in FP32. The
/// eager pinned reference materializes BF16 scores, then rounds the softmax
/// probabilities to BF16 before the value reduction. A fully masked query row
/// produces a zero output row instead of NaN; mask builders reject such masks.
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
            // The additive mask is the contract's F32-typed slot: blocked keys
            // are encoded as -Infinity in FP32 precision, independent of the
            // activation dtype.
            let values = read_numeric(device, mask)?;
            if values.dtype() != DType::F32 {
                return Err(OpError::IndexDTypeMismatch);
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
                // The eager reference first materializes the BF16 score, then
                // adds the attention mask, and only then upcasts for softmax.
                let mut score = round_to_dtype(&[accumulator * scale], dtype)[0];
                if let Some(mask_values) = &mask_values {
                    score = round_to_dtype(
                        &[score + mask_values[token * key_tokens + key_index]],
                        dtype,
                    )[0];
                }
                scores[key_index] = score;
            }
            softmax_row(&scores, &mut probabilities);
            // The eager reference casts the probabilities back to the activation
            // dtype before the value reduction.
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
    for value in output.iter_mut() {
        *value /= sum;
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

/// Multiply a tensor by an FP32 scalar, rounding the product once to the
/// input dtype.
///
/// The pinned upstream multiplies a BF16 tensor by a plain Python float
/// (`hidden_states * scalar_root_size`). PyTorch evaluates that product in
/// FP32 and rounds the result to BF16; the scalar itself is never rounded to
/// BF16. Reproducing that boundary matters: rounding the scalar first shifts
/// the router logits and changes the BF16 tie structure of the checkpoint.
pub fn scale_scalar(device: &dyn Device, input: &Tensor, factor: f32) -> Result<Tensor, OpError> {
    if !factor.is_finite() {
        return Err(OpError::InvalidScaleFactor);
    }
    let shape = input.layout().shape();
    if shape.is_empty() {
        return Err(OpError::InvalidRank);
    }
    let source = read_numeric(device, input)?;
    let dtype = source.dtype();
    let values = source.to_f32();
    let output: Vec<f32> = values.iter().map(|value| value * factor).collect();
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

/// Bit-exact BF16 widening to an F32 tensor (the contract's one explicit
/// cross-dtype operation).
pub fn widen(device: &dyn Device, input: &Tensor) -> Result<Tensor, OpError> {
    let source = read_numeric(device, input)?;
    if source.dtype() != DType::BF16 {
        return Err(OpError::UnsupportedDType);
    }
    write_numeric(
        device,
        &source.to_f32(),
        input.layout().shape().to_vec(),
        DType::F32,
    )
}

/// Router scores -> stable softmax, deterministic top-k, weight normalization
/// and per-expert scale, emitted as `[tokens, 2 * top_k]` F32: the first `k`
/// columns are weights, the last `k` columns are expert indices.
///
/// The selection reproduces the pinned upstream's `torch.topk` call exactly,
/// including the order inside a group of experts with identical probabilities
/// (see [`crate::aten_topk`]): PyTorch's CPU selection is unstable there, and
/// the pinned checkpoint contains such tie groups at the top-k boundary.
pub fn router_top_k(
    device: &dyn Device,
    scores: &Tensor,
    per_expert_scale: &Tensor,
    top_k: usize,
) -> Result<Tensor, OpError> {
    let shape = scores.layout().shape();
    if shape.len() != 2 {
        return Err(OpError::ShapeMismatch);
    }
    let (tokens, experts) = (shape[0], shape[1]);
    if top_k == 0 || top_k > experts {
        return Err(OpError::InvalidTopK(top_k, experts));
    }
    let scores = read_numeric(device, scores)?;
    if scores.dtype() != DType::F32 {
        return Err(OpError::UnsupportedDType);
    }
    let per_expert = read_numeric(device, per_expert_scale)?;
    if per_expert.dtype() != DType::F32 {
        return Err(OpError::UnsupportedDType);
    }
    if per_expert.to_f32().len() != experts {
        return Err(OpError::ScaleMismatch);
    }
    let per_expert = per_expert.to_f32();
    let scores = scores.to_f32();
    let mut out = vec![0.0f32; tokens * 2 * top_k];
    for token in 0..tokens {
        let row = &scores[token * experts..token * experts + experts];
        // Stable softmax in FP32; NaN scores propagate as NaN.
        let max_value = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; experts];
        for (index, score) in row.iter().enumerate() {
            let value = (score - max_value).exp();
            probs[index] = value;
            sum += value;
        }
        for prob in probs.iter_mut() {
            *prob /= sum;
        }
        // ATen-compatible top-k: the pinned upstream calls `torch.topk` on the
        // FP32 probabilities, and PyTorch's CPU selection is unstable inside a
        // tie group (the pinned checkpoint has exact BF16 ties, including at
        // the top-k boundary where the choice changes the model's output).
        // `top_k_largest` reproduces ATen's algorithm exactly.
        let selected = crate::aten_topk::top_k_largest(&probs, top_k);
        let mut weight_sum = 0.0f32;
        for &index in &selected {
            weight_sum += probs[index];
        }
        for (slot, &index) in selected.iter().enumerate() {
            let weight = probs[index] / weight_sum * per_expert[index];
            out[token * 2 * top_k + slot] = weight;
            out[token * 2 * top_k + top_k + slot] = index as f32;
        }
    }
    write_numeric(device, &out, vec![tokens, 2 * top_k], DType::F32)
}

/// `gelu_pytorch_tanh(gate) * up`.
///
/// The activation is rounded to the tensor dtype before it is multiplied by
/// `up`, matching the pinned upstream BF16 execution boundary. The product is
/// rounded once more at the output.
pub fn gelu_mul_up(device: &dyn Device, gate: &Tensor, up: &Tensor) -> Result<Tensor, OpError> {
    if gate.layout().shape() != up.layout().shape() {
        return Err(OpError::ShapeMismatch);
    }
    let shape = gate.layout().shape().to_vec();
    let gate = read_numeric(device, gate)?;
    let up = read_numeric(device, up)?;
    if gate.dtype() != up.dtype() {
        return Err(OpError::MixedDType);
    }
    let dtype = gate.dtype();
    let g = gate.to_f32();
    let u = up.to_f32();
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut activated = vec![0.0f32; g.len()];
    for (index, gate_value) in g.iter().enumerate() {
        let inner = k * (gate_value + 0.044715 * gate_value * gate_value * gate_value);
        activated[index] = 0.5 * gate_value * (1.0 + inner.tanh());
    }
    let activated = round_to_dtype(&activated, dtype);
    let out: Vec<f32> = activated
        .iter()
        .zip(u.iter())
        .map(|(activation, up_value)| activation * up_value)
        .collect();
    write_numeric(device, &out, shape, dtype)
}

/// Row gather with F32-encoded integer indices.
pub fn gather_rows(
    device: &dyn Device,
    input: &Tensor,
    indices: &Tensor,
) -> Result<Tensor, OpError> {
    let shape = input.layout().shape();
    if shape.len() != 2 {
        return Err(OpError::ShapeMismatch);
    }
    let (rows, width) = (shape[0], shape[1]);
    let source = read_numeric(device, input)?;
    let idx = read_numeric(device, indices)?;
    if idx.dtype() != DType::F32 {
        return Err(OpError::IndexDTypeMismatch);
    }
    let idx = idx.to_f32();
    let values = source.to_f32();
    let mut out = Vec::with_capacity(idx.len() * width);
    for index in &idx {
        let row = index.to_int_index(rows)?;
        out.extend_from_slice(&values[row * width..row * width + width]);
    }
    write_numeric(device, &out, vec![idx.len(), width], source.dtype())
}

/// Weighted scatter-add into a fresh zeroed `[out_rows, width]` tensor.
/// Deterministic accumulation walks the pairs in index order; the CUDA kernel
/// uses atomics, so its differential budget is a model-derived bound, not ULP.
pub fn scatter_add(
    device: &dyn Device,
    source: &Tensor,
    weights: &Tensor,
    rows: &Tensor,
    out_rows: usize,
) -> Result<Tensor, OpError> {
    let shape = source.layout().shape();
    if shape.len() != 2 {
        return Err(OpError::ShapeMismatch);
    }
    let (pairs, width) = (shape[0], shape[1]);
    if out_rows == 0 {
        return Err(OpError::InvalidOutputRows);
    }
    let source = read_numeric(device, source)?;
    let weights = read_numeric(device, weights)?;
    let rows = read_numeric(device, rows)?;
    if weights.dtype() != DType::F32 || rows.dtype() != DType::F32 {
        return Err(OpError::IndexDTypeMismatch);
    }
    let weights = weights.to_f32();
    let rows = rows.to_f32();
    if weights.len() != pairs || rows.len() != pairs {
        return Err(OpError::ScaleMismatch);
    }
    let dtype = source.dtype();
    let source = source.to_f32();
    let mut out = vec![0.0f32; out_rows * width];
    for pair in 0..pairs {
        let row = rows[pair].to_int_index(out_rows)?;
        let weight = weights[pair];
        let src_row = &source[pair * width..pair * width + width];
        let dst = &mut out[row * width..row * width + width];
        for (cell, value) in src_row.iter().enumerate() {
            dst[cell] += weight * value;
        }
    }
    write_numeric(device, &out, vec![out_rows, width], dtype)
}

/// Embedding gather with F32-encoded token ids and an explicit scale.
pub fn embed_gather(
    device: &dyn Device,
    ids: &Tensor,
    table: &Tensor,
    scale: f32,
) -> Result<Tensor, OpError> {
    let shape = table.layout().shape();
    if shape.len() != 2 {
        return Err(OpError::ShapeMismatch);
    }
    let (vocab, width) = (shape[0], shape[1]);
    let ids = read_numeric(device, ids)?;
    if ids.dtype() != DType::F32 {
        return Err(OpError::IndexDTypeMismatch);
    }
    let ids = ids.to_f32();
    let table_values = read_numeric(device, table)?;
    let dtype = table_values.dtype();
    let table = table_values.to_f32();
    let mut out = Vec::with_capacity(ids.len() * width);
    for id in &ids {
        let row = id.to_int_index(vocab)?;
        for column in 0..width {
            out.push(table[row * width + column] * scale);
        }
    }
    write_numeric(device, &out, vec![ids.len(), width], dtype)
}

/// Zero rows flagged as padding: `flags[row] == 1.0` yields zeros.
pub fn row_mask(device: &dyn Device, input: &Tensor, flags: &Tensor) -> Result<Tensor, OpError> {
    let shape = input.layout().shape();
    if shape.len() != 2 {
        return Err(OpError::ShapeMismatch);
    }
    let (rows, width) = (shape[0], shape[1]);
    let source = read_numeric(device, input)?;
    let flags = read_numeric(device, flags)?;
    if flags.dtype() != DType::F32 {
        return Err(OpError::IndexDTypeMismatch);
    }
    let flags = flags.to_f32();
    if flags.len() != rows {
        return Err(OpError::ScaleMismatch);
    }
    let dtype = source.dtype();
    let source = source.to_f32();
    let mut out = vec![0.0f32; rows * width];
    for row in 0..rows {
        if flags[row] != 1.0 {
            out[row * width..row * width + width]
                .copy_from_slice(&source[row * width..row * width + width]);
        }
    }
    write_numeric(device, &out, shape.to_vec(), dtype)
}

/// `(x * scale) - bias` over the last dimension, FP32, rounded once.
pub fn scale_bias(
    device: &dyn Device,
    input: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
) -> Result<Tensor, OpError> {
    let shape = input.layout().shape();
    if shape.len() != 2 {
        return Err(OpError::ShapeMismatch);
    }
    let (rows, width) = (shape[0], shape[1]);
    let source = read_numeric(device, input)?;
    let scale = read_numeric(device, scale)?;
    let bias = read_numeric(device, bias)?;
    if source.dtype() != scale.dtype() || source.dtype() != bias.dtype() {
        return Err(OpError::MixedDType);
    }
    let dtype = source.dtype();
    let scale = scale.to_f32();
    let bias = bias.to_f32();
    if scale.len() != width || bias.len() != width {
        return Err(OpError::ScaleMismatch);
    }
    let source = source.to_f32();
    let mut out = vec![0.0f32; rows * width];
    for row in 0..rows {
        for column in 0..width {
            out[row * width + column] =
                source[row * width + column].mul_add(scale[column], -bias[column]);
        }
    }
    write_numeric(device, &out, shape.to_vec(), dtype)
}

/// F32 index values must encode exact integers within the tensor's row count.
trait IntIndex {
    fn to_int_index(&self, rows: usize) -> Result<usize, OpError>;
}

impl IntIndex for f32 {
    fn to_int_index(&self, rows: usize) -> Result<usize, OpError> {
        if !self.is_finite() || *self < 0.0 || *self >= rows as f32 {
            return Err(OpError::InvalidIndex);
        }
        let index = *self as usize;
        if (index as f32) != *self {
            return Err(OpError::InvalidIndex);
        }
        Ok(index)
    }
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
    fn scale_scalar_rounds_once_and_keeps_the_scalar_in_fp32() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = values(&device, &scale_scalar(&device, &input, 0.5).unwrap());
        assert_eq!(out, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);

        // The scalar is an FP32 value: it is never rounded to BF16 before the
        // product. `2816 ** -0.5` is not BF16-representable, and rounding it
        // first shifts the result by up to 0.2 %.
        let factor = (2816f64).powf(-0.5) as f32;
        let bf16_input = bf16_tensor(&device, vec![1], &[3.0]);
        let scaled = values(
            &device,
            &scale_scalar(&device, &bf16_input, factor).unwrap(),
        );
        let expected = Bf16::from_f32(3.0f32 * factor).to_f32();
        assert_eq!(scaled[0], expected);
        let rounded_scalar = Bf16::from_f32(factor).to_f32();
        assert_ne!(
            scaled[0],
            Bf16::from_f32(3.0f32 * rounded_scalar).to_f32(),
            "the scalar must not be rounded to BF16"
        );

        assert_eq!(
            scale_scalar(&device, &input, f32::NAN).unwrap_err(),
            OpError::InvalidScaleFactor
        );
        assert_eq!(
            scale_scalar(&device, &input, f32::INFINITY).unwrap_err(),
            OpError::InvalidScaleFactor
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

    #[test]
    fn widening_is_the_only_cross_dtype_operation_and_is_exact() {
        let device = CpuDevice::new();
        let input = bf16_tensor(&device, vec![2, 2], &[1.0, 2.0, 3.0, 256.0]);
        let out = widen(&device, &input).unwrap();
        assert_eq!(out.layout().dtype(), DType::F32);
        assert_eq!(values(&device, &out), [1.0, 2.0, 3.0, 256.0]);
        let f32_input = tensor(&device, vec![2, 2], &[1.0; 4]);
        assert_eq!(
            widen(&device, &f32_input).unwrap_err(),
            OpError::UnsupportedDType
        );
    }

    #[test]
    fn router_top_k_selects_normalized_top_weights() {
        let device = CpuDevice::new();
        // One token, three experts: probabilities [0.6, 0.3, 0.1] are produced
        // by scores whose softmax is exactly these values.
        let scores = tensor(&device, vec![1, 3], &[0.5, -0.2, -1.1]);
        let per_expert = tensor(&device, vec![3], &[1.0, 1.0, 1.0]);
        let out = values(
            &device,
            &router_top_k(&device, &scores, &per_expert, 2).unwrap(),
        );
        let e: Vec<f32> = [0.5f32, -0.2, -1.1]
            .iter()
            .map(|v| (v - 0.5f32).exp())
            .collect();
        let s: f32 = e.iter().sum();
        let probs: Vec<f32> = e.iter().map(|v| v / s).collect();
        let top_sum = probs[0] + probs[1];
        assert_eq!(out[0], probs[0] / top_sum);
        assert_eq!(out[1], probs[1] / top_sum);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn router_top_k_applies_per_expert_scale_and_rejects_bad_top_k() {
        let device = CpuDevice::new();
        let scores = tensor(&device, vec![1, 3], &[1.0, 0.0, -1.0]);
        let per_expert = tensor(&device, vec![3], &[2.0, 1.0, 0.5]);
        let out = values(
            &device,
            &router_top_k(&device, &scores, &per_expert, 2).unwrap(),
        );
        // probabilities: p0 > p1 > p2; weights = p_i/(p0+p1) * per_expert_i.
        let e1 = 1.0f32;
        let e2 = (0.0f32 - 1.0).exp();
        let e3 = (-1.0f32 - 1.0).exp();
        let s = e1 + e2 + e3;
        let p0 = e1 / s;
        let p1 = e2 / s;
        let top = p0 + p1;
        assert!((out[0] - p0 / top * 2.0).abs() < 1e-6);
        assert!((out[1] - p1 / top * 1.0).abs() < 1e-6);
        assert_eq!(
            router_top_k(&device, &scores, &per_expert, 4).unwrap_err(),
            OpError::InvalidTopK(4, 3)
        );
    }

    #[test]
    fn gelu_mul_up_matches_the_pinned_activation() {
        let device = CpuDevice::new();
        let gate = tensor(&device, vec![1, 4], &[0.0, 1.0, -1.0, 2.0]);
        let up = tensor(&device, vec![1, 4], &[7.0, 1.0, 1.0, 1.0]);
        let out = values(&device, &gelu_mul_up(&device, &gate, &up).unwrap());
        assert_eq!(out[0], 0.0);
        let k = (2.0f32 / std::f32::consts::PI).sqrt();
        let gelu = |x: f32| 0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh());
        assert!((out[1] - gelu(1.0)).abs() < 1e-6);
        assert!((out[2] - gelu(-1.0)).abs() < 1e-6);
        assert!((out[3] - gelu(2.0) * 1.0).abs() < 1e-6);
    }

    #[test]
    fn gather_rows_and_scatter_add_round_trip_indices() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![3, 2], &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        let indices = tensor(&device, vec![4], &[2.0, 0.0, 0.0, 1.0]);
        let gathered = values(&device, &gather_rows(&device, &input, &indices).unwrap());
        assert_eq!(gathered, [30.0, 31.0, 10.0, 11.0, 10.0, 11.0, 20.0, 21.0]);

        // Scatter the gathered rows back with weights [1,1,2,1] into 3 rows.
        let gathered_tensor = tensor(&device, vec![4, 2], &gathered);
        let weights = tensor(&device, vec![4], &[1.0, 1.0, 2.0, 1.0]);
        let rows = tensor(&device, vec![4], &[2.0, 0.0, 0.0, 1.0]);
        let out = values(
            &device,
            &scatter_add(&device, &gathered_tensor, &weights, &rows, 3).unwrap(),
        );
        assert_eq!(out, [30.0, 33.0, 20.0, 21.0, 30.0, 31.0]);
    }

    #[test]
    fn gather_rows_rejects_index_dtype_and_fractional_indices() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![3, 2], &[1.0; 6]);
        let bad = bf16_tensor(&device, vec![2], &[1.0, 2.0]);
        assert_eq!(
            gather_rows(&device, &input, &bad).unwrap_err(),
            OpError::IndexDTypeMismatch
        );
        let fractional = tensor(&device, vec![2], &[1.5, 0.0]);
        assert_eq!(
            gather_rows(&device, &input, &fractional).unwrap_err(),
            OpError::InvalidIndex
        );
        let out_of_range = tensor(&device, vec![1], &[3.0]);
        assert_eq!(
            gather_rows(&device, &input, &out_of_range).unwrap_err(),
            OpError::InvalidIndex
        );
    }

    #[test]
    fn embed_gather_scales_rows_by_the_explicit_factor() {
        let device = CpuDevice::new();
        let table = tensor(&device, vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let ids = tensor(&device, vec![2], &[2.0, 0.0]);
        let out = values(&device, &embed_gather(&device, &ids, &table, 2.0).unwrap());
        assert_eq!(out, [10.0, 12.0, 2.0, 4.0]);
    }

    #[test]
    fn row_mask_zeroes_flagged_rows() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let flags = tensor(&device, vec![3], &[0.0, 1.0, 0.0]);
        let out = values(&device, &row_mask(&device, &input, &flags).unwrap());
        assert_eq!(out, [1.0, 2.0, 0.0, 0.0, 5.0, 6.0]);
    }

    #[test]
    fn scale_bias_applies_explicit_vectors() {
        let device = CpuDevice::new();
        let input = tensor(&device, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let scale = tensor(&device, vec![2], &[2.0, 0.5]);
        let bias = tensor(&device, vec![2], &[1.0, 1.0]);
        let out = values(
            &device,
            &scale_bias(&device, &input, &scale, &bias).unwrap(),
        );
        assert_eq!(out, [1.0, 0.0, 5.0, 1.0]);
    }
}
