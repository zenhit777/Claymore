//! CPU reference graph for the full DiffusionGemma model (M10.3).
//!
//! Reproduces the pinned upstream composition op by op: encoder causal pass
//! with KV cache, decoder canvas pass with self-conditioning, MoE top-8
//! routing, dense MLP, vision tower with axial 2D rope, tied lm head with
//! final logit softcapping. It is the deterministic ground truth for CUDA
//! differential campaigns. Batch size is always 1; tensors are
//! `[tokens, features]` for projections and `[tokens, heads, head_dim]` for
//! attention heads.

use claymore_core::MemoryTier;
use claymore_device::{CpuDevice, DType, Device};
use claymore_tensor::diffusion as dfn;
use claymore_tensor::ops;
use claymore_tensor::{Bf16, Tensor};

pub const MASK_BLOCKED_CPU: f32 = f32::NEG_INFINITY;

/// Per-layer KV cache (batch 1): keys/values `[seq, kv_heads, head_dim]`.
pub struct CacheLayer {
    pub keys: Tensor,
    pub values: Tensor,
}

pub struct KvCache {
    pub layers: Vec<CacheLayer>,
}

impl KvCache {
    pub fn new(layers: usize) -> Self {
        Self {
            layers: Vec::with_capacity(layers),
        }
    }

    pub fn append(&mut self, host: &Host, layer: usize, keys: Tensor, values: Tensor) {
        let (heads, dim) = (keys.layout().shape()[1], keys.layout().shape()[2]);
        if self.layers.len() <= layer {
            self.layers.push(CacheLayer { keys, values });
        } else {
            let rows = self.layers[layer].keys.layout().shape()[0];
            let mut kb = host.to_bits(&self.layers[layer].keys);
            kb.extend(host.to_bits(&keys));
            let mut vb = host.to_bits(&self.layers[layer].values);
            vb.extend(host.to_bits(&values));
            self.layers[layer] = CacheLayer {
                keys: bf16_tensor(host, vec![rows + keys.layout().shape()[0], heads, dim], kb),
                values: bf16_tensor(
                    host,
                    vec![rows + values.layout().shape()[0], heads, dim],
                    vb,
                ),
            };
        }
    }

    pub fn seq_len(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.keys.layout().shape()[0])
            .unwrap_or(0)
    }
}

/// Host tensor factory bound to one CPU device instance.
pub struct Host {
    dev: CpuDevice,
}

impl Default for Host {
    fn default() -> Self {
        Self::new()
    }
}

impl Host {
    pub fn new() -> Self {
        Self {
            dev: CpuDevice::new(),
        }
    }

    pub fn device(&self) -> &CpuDevice {
        &self.dev
    }

    pub fn bf16(&self, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| Bf16::from_f32(*v).to_ne_bytes())
            .collect();
        let mut buffer = self
            .dev
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        self.dev.write(buffer.as_mut(), 0, &bytes).unwrap();
        Tensor::new(buffer, shape, DType::BF16).unwrap()
    }

    pub fn f32(&self, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut buffer = self
            .dev
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        self.dev.write(buffer.as_mut(), 0, &bytes).unwrap();
        Tensor::new(buffer, shape, DType::F32).unwrap()
    }

    /// Reads a tensor and widens BF16 to F32 bit-exactly.
    pub fn to_f32(&self, tensor: &Tensor) -> Vec<f32> {
        let bytes = tensor.layout().span_bytes();
        let raw = self.dev.read(tensor.storage(), 0, bytes).unwrap();
        match tensor.layout().dtype() {
            DType::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect(),
            DType::BF16 => raw
                .chunks_exact(2)
                .map(|b| Bf16::from_ne_bytes(b.try_into().unwrap()).to_f32())
                .collect(),
            other => panic!("unsupported dtype {other:?}"),
        }
    }

    pub fn to_bits(&self, tensor: &Tensor) -> Vec<Bf16> {
        let bytes = tensor.layout().span_bytes();
        let raw = self.dev.read(tensor.storage(), 0, bytes).unwrap();
        raw.chunks_exact(2)
            .map(|b| Bf16::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }
}

pub fn bf16_tensor(host: &Host, shape: Vec<usize>, bits: Vec<Bf16>) -> Tensor {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_ne_bytes()).collect();
    let mut buffer = host
        .dev
        .allocate(MemoryTier::Host, bytes.len() as u64)
        .unwrap();
    host.dev.write(buffer.as_mut(), 0, &bytes).unwrap();
    Tensor::new(buffer, shape, DType::BF16).unwrap()
}

/// Row concat of `[a, heads, dim]` + `[b, heads, dim]` (BF16).
pub fn concat_head_rows(host: &Host, a: &Tensor, b: &Tensor) -> Tensor {
    let (heads, dim) = (a.layout().shape()[1], a.layout().shape()[2]);
    let mut bits = host.to_bits(a);
    bits.extend(host.to_bits(b));
    bf16_tensor(
        host,
        vec![a.layout().shape()[0] + b.layout().shape()[0], heads, dim],
        bits,
    )
}

pub fn scale_bias(host: &Host, x: &Tensor, scale: &Tensor, bias: &Tensor) -> Tensor {
    claymore_tensor::diffusion::scale_bias(host.device(), x, scale, bias).unwrap()
}

pub fn clone_tensor(host: &Host, t: &Tensor) -> Tensor {
    let shape = t.layout().shape().to_vec();
    bf16_tensor(host, shape, host.to_bits(t))
}

/// Row concat of `[a, feat]` + `[b, feat]` (BF16).
pub fn concat_rows(host: &Host, a: &Tensor, b: &Tensor) -> Tensor {
    let width = a.layout().shape()[1];
    let mut bits = host.to_bits(a);
    bits.extend(host.to_bits(b));
    bf16_tensor(
        host,
        vec![a.layout().shape()[0] + b.layout().shape()[0], width],
        bits,
    )
}

/// Weight `[out, in]` transposed to `[in, out]` for `ops::matmul`.
pub fn transpose_weight(host: &Host, w: &Tensor) -> Tensor {
    let (out, inner) = (w.layout().shape()[0], w.layout().shape()[1]);
    let bits = host.to_bits(w);
    let mut out_bits = Vec::with_capacity(bits.len());
    for col in 0..inner {
        for row in 0..out {
            out_bits.push(bits[row * inner + col]);
        }
    }
    bf16_tensor(host, vec![inner, out], out_bits)
}

pub fn linear(host: &Host, x: &Tensor, weight: &Tensor) -> Tensor {
    ops::matmul(host.device(), x, &transpose_weight(host, weight)).unwrap()
}

pub fn rms_norm(host: &Host, x: &Tensor, scale: &Tensor, eps: f64) -> Tensor {
    ops::rms_norm(host.device(), x, scale, eps as f32).unwrap()
}

pub fn rms_norm_unscaled(host: &Host, x: &Tensor, eps: f64) -> Tensor {
    let width = x.layout().shape().last().copied().unwrap();
    let ones = host.bf16(vec![width], &vec![1.0f32; width]);
    ops::rms_norm(host.device(), x, &ones, eps as f32).unwrap()
}

pub fn add(host: &Host, a: &Tensor, b: &Tensor) -> Tensor {
    ops::add(host.device(), a, b).unwrap()
}

pub fn gelu_mul_up(host: &Host, gate: &Tensor, up: &Tensor) -> Tensor {
    dfn::gelu_mul_up(host.device(), gate, up).unwrap()
}

pub fn dense_mlp(host: &Host, x: &Tensor, gate: &Tensor, up: &Tensor, down: &Tensor) -> Tensor {
    let g = linear(host, x, gate);
    let u = linear(host, x, up);
    let a = gelu_mul_up(host, &g, &u);
    linear(host, &a, down)
}

/// Router: RMSNorm(no scale) -> *scale*scalar_root -> proj -> softmax ->
/// deterministic top-k -> normalized weights * per-expert scale.
/// Returns `(probabilities [tokens, experts], raw_scores [tokens, experts],
/// weights [tokens*top_k], indices [tokens*top_k])` as F32 vectors: the FP32
/// softmax probabilities, the FP32 projection scores before softmax, the
/// normalized weights scaled per expert, and the selected expert indices.
#[allow(clippy::too_many_arguments)]
pub fn router_forward(
    host: &Host,
    x: &Tensor,
    proj: &Tensor,
    scale: &Tensor,
    per_expert_scale: &Tensor,
    eps: f64,
    root_scale: f64,
    top_k: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let normed = rms_norm_unscaled(host, x, eps);
    let scale_v = host.to_f32(scale);
    let width = normed.layout().shape()[1];
    // The pinned upstream performs two BF16 tensor operations:
    // `hidden_states * self.scale`, then `* scalar_root_size`. Preserve both
    // rounding boundaries; folding the factors into one FP32 multiply changes
    // the selected experts for close router scores.
    let scaled_values: Vec<f32> = host
        .to_f32(&normed)
        .iter()
        .enumerate()
        .map(|(i, v)| v * scale_v[i % width])
        .collect();
    let scaled_by_expert = host.bf16(normed.layout().shape().to_vec(), &scaled_values);
    // The upstream multiplies by `scalar_root_size`, a plain Python float
    // (`config.hidden_size ** -0.5`). PyTorch keeps the scalar in the
    // operation's math type (FP32 for a BF16 tensor) and rounds only the
    // *result* to BF16: rounding the scalar itself to BF16 first shifts the
    // router logits by up to 0.2 %, which changes the BF16 tie structure of
    // the pinned checkpoint and therefore the selected experts.
    let root_scale = root_scale as f32;
    let scaled_values: Vec<f32> = host
        .to_f32(&scaled_by_expert)
        .iter()
        .map(|value| value * root_scale)
        .collect();
    let scaled = host.bf16(normed.layout().shape().to_vec(), &scaled_values);
    let scores = linear(host, &scaled, proj);
    let scores = host.to_f32(&dfn::widen(host.device(), &scores).unwrap());
    let (tokens, experts) = (
        scaled.layout().shape()[0],
        scores.len() / scaled.layout().shape()[0],
    );
    let per_expert = host.to_f32(per_expert_scale);
    let mut probabilities = vec![0.0f32; tokens * experts];
    let mut weights = vec![0.0f32; tokens * top_k];
    let mut indices = vec![0.0f32; tokens * top_k];
    for t in 0..tokens {
        let row = &scores[t * experts..(t + 1) * experts];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        probabilities[t * experts..(t + 1) * experts].copy_from_slice(&probs);
        // ATen-compatible top-k: the pinned upstream calls `torch.topk` on the
        // FP32 probabilities. PyTorch's CPU selection is unstable inside a tie
        // group, and the pinned checkpoint contains exact BF16 ties at the
        // top-k boundary, where the choice changes the model's output. The
        // shared helper reproduces ATen's algorithm exactly.
        let selected = claymore_tensor::top_k_largest(&probs, top_k);
        let mut wsum = 0.0f32;
        for &i in &selected {
            wsum += probs[i];
        }
        for (slot, &i) in selected.iter().enumerate() {
            weights[t * top_k + slot] = probs[i] / wsum * per_expert[i];
            indices[t * top_k + slot] = i as f32;
        }
    }
    (probabilities, scores, weights, indices)
}

/// Experts: for each (token, slot) pair run expert e = index; gate/up from
/// `gate_up_proj [E, 2*inter, hidden]`, gelu(gate)*up, down `[E, hidden, inter]`,
/// weight-scaled accumulation per token.
pub fn experts_forward(
    host: &Host,
    x: &Tensor,
    weights: &[f32],
    indices: &[f32],
    gate_up: &Tensor,
    down: &Tensor,
) -> Tensor {
    let tokens = x.layout().shape()[0];
    let width = x.layout().shape()[1];
    let top_k = indices.len() / tokens;
    let pairs = tokens * top_k;
    let gu = host.to_bits(gate_up);
    let dp = host.to_bits(down);
    let inter = gate_up.layout().shape()[1] / 2;
    assert_eq!(
        gate_up.layout().shape()[1],
        2 * inter,
        "expert gate/up tensor has an invalid second dimension"
    );
    assert_eq!(
        down.layout().shape()[2],
        inter,
        "expert down tensor does not match the expert intermediate size"
    );
    let xv = host.to_f32(x);
    let mut out = vec![0.0f32; tokens * width];
    let mut contributions = Vec::with_capacity(pairs);
    for p in 0..pairs {
        let e = indices[p] as usize;
        let token = p / top_k;
        let row = &xv[token * width..(token + 1) * width];
        let mut acc_row = vec![0.0f32; width];
        for i in 0..inter {
            let mut acc_g = 0.0f32;
            let mut acc_u = 0.0f32;
            for h in 0..width {
                acc_g += row[h] * gu[(e * 2 * inter + i) * width + h].to_f32();
                acc_u += row[h] * gu[(e * 2 * inter + inter + i) * width + h].to_f32();
            }
            // Eager expert boundaries: BF16 gate/up, BF16 activation, BF16
            // gated product, BF16 down projection.
            let acc_g = Bf16::from_f32(acc_g).to_f32();
            let acc_u = Bf16::from_f32(acc_u).to_f32();
            let act = 0.5
                * acc_g
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt()
                        * (acc_g + 0.044715 * acc_g * acc_g * acc_g))
                        .tanh());
            let act = Bf16::from_f32(act).to_f32();
            let v = Bf16::from_f32(act * acc_u).to_f32();
            for (col, dst) in acc_row.iter_mut().enumerate().take(width) {
                *dst += v * dp[(e * width + col) * inter + i].to_f32();
            }
        }
        let w = weights[p];
        let contribution: Vec<f32> = acc_row
            .iter()
            .map(|value| Bf16::from_f32(w * Bf16::from_f32(*value).to_f32()).to_f32())
            .collect();
        contributions.push((e, p, contribution));
    }
    // Eager `index_add_` consumes the contributions in ascending expert order.
    contributions.sort_by_key(|(expert, pair, _row)| (*expert, *pair));
    for (_expert, pair, contribution) in contributions {
        let token = pair / top_k;
        for (col, dst) in out[token * width..(token + 1) * width]
            .iter_mut()
            .enumerate()
        {
            *dst = Bf16::from_f32(*dst + contribution[col]).to_f32();
        }
    }
    host.bf16(vec![tokens, width], &out)
}

/// Axial 2D vision RoPE: `inv_freq` over `head_dim/2`, recomposed as
/// `[h, h, w, w]` per patch, exactly as the pinned
/// `Gemma4VisionRotaryEmbedding.recomposition_frequencies`.
pub fn vision_rope_tables(
    head_dim: usize,
    theta: f32,
    positions: &[(u32, u32)],
) -> (Vec<f32>, Vec<f32>) {
    let spatial = head_dim / 2;
    let inv: Vec<f32> = (0..spatial / 2)
        .map(|i| theta.powf((2 * i) as f32 / spatial as f32).recip())
        .collect();
    let mut cos_out = Vec::with_capacity(positions.len() * 2 * head_dim);
    let mut sin_out = Vec::with_capacity(positions.len() * 2 * head_dim);
    for (x, y) in positions {
        let ch: Vec<f32> = inv
            .iter()
            .map(|f| Bf16::from_f32((*x as f32 * f).cos()).to_f32())
            .collect();
        let sh: Vec<f32> = inv
            .iter()
            .map(|f| Bf16::from_f32((*x as f32 * f).sin()).to_f32())
            .collect();
        let cw: Vec<f32> = inv
            .iter()
            .map(|f| Bf16::from_f32((*y as f32 * f).cos()).to_f32())
            .collect();
        let sw: Vec<f32> = inv
            .iter()
            .map(|f| Bf16::from_f32((*y as f32 * f).sin()).to_f32())
            .collect();
        cos_out.extend(ch.iter().chain(ch.iter()).chain(cw.iter()).chain(cw.iter()));
        sin_out.extend(sh.iter().chain(sh.iter()).chain(sw.iter()).chain(sw.iter()));
    }
    (cos_out, sin_out)
}

/// Multidimensional RoPE on `[tokens, heads, head_dim]` with two axes: the
/// head dimension splits into H and W parts of `head_dim/2`; each part gets
/// `rotate_half` rotation with its recomposed cos/sin.
pub fn apply_multidim_rope(
    host: &Host,
    x: &Tensor,
    cos: &[f32],
    sin: &[f32],
    heads: usize,
) -> Tensor {
    let tokens = x.layout().shape()[0];
    let dim = x.layout().shape()[2];
    let part = dim / 2;
    let half = part / 2;
    let xv = host.to_f32(x);
    let mut out = vec![0.0f32; xv.len()];
    let round = |value: f32| Bf16::from_f32(value).to_f32();
    for t in 0..tokens {
        for h in 0..heads {
            let base = (t * heads + h) * dim;
            for axis in 0..2 {
                let pbase = base + axis * part;
                let cbase = t * dim + axis * part;
                for i in 0..half {
                    let c0 = cos[cbase + i];
                    let s0 = sin[cbase + i];
                    let c1 = cos[cbase + i + half];
                    let s1 = sin[cbase + i + half];
                    let x0 = xv[pbase + i];
                    let x1 = xv[pbase + i + half];
                    out[pbase + i] = round(round(x0 * c0) - round(x1 * s0));
                    out[pbase + i + half] = round(round(x1 * c1) + round(x0 * s1));
                }
            }
        }
    }
    host.bf16(x.layout().shape().to_vec(), &out)
}

pub fn attention_op(
    host: &Host,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Tensor {
    match dfn::attention(host.device(), q, k, v, mask, scale) {
        Ok(t) => t,
        Err(e) => panic!(
            "attention q{:?} k{:?} v{:?} mask{:?}: {e}",
            q.layout().shape(),
            k.layout().shape(),
            v.layout().shape(),
            mask.map(|m| m.layout().shape())
        ),
    }
}

pub fn softmax_last_dim(host: &Host, x: &Tensor) -> Tensor {
    ops::softmax_last_dim(host.device(), x).unwrap()
}

pub fn softcap(host: &Host, x: &Tensor, cap: f32) -> Tensor {
    dfn::soft_cap(host.device(), x, cap).unwrap()
}

pub fn embed_gather_scaled(host: &Host, ids: &[u32], table: &Tensor, scale: f32) -> Tensor {
    let f = ids.iter().map(|i| *i as f32).collect::<Vec<f32>>();
    let ids_t = host.f32(vec![ids.len()], &f);
    dfn::embed_gather(host.device(), &ids_t, table, scale).unwrap()
}

/// Text embedding scale used by the pinned upstream model.
///
/// `DiffusionGemmaTextScaledWordEmbedding` stores `sqrt(hidden_size)` in a
/// float32 buffer and converts that buffer to the embedding dtype (BF16)
/// before multiplying. Preserve that dtype boundary here.
pub fn text_embedding_scale(hidden: usize) -> f32 {
    Bf16::from_f32((hidden as f32).sqrt()).to_f32()
}

/// Matmul `[rows, inner] x [inner, cols]` (F32 result when inputs BF16).
pub fn matmul(host: &Host, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    ops::matmul(host.device(), lhs, rhs).unwrap()
}
