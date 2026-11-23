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
/// Returns `[tokens, top_k]` F32 weights and `[tokens, top_k]` F32 indices.
#[allow(clippy::too_many_arguments)]
pub fn router_forward(
    host: &Host,
    x: &Tensor,
    proj: &Tensor,
    scale: &Tensor,
    per_expert_scale: &Tensor,
    eps: f64,
    root_scale: f32,
    top_k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let normed = rms_norm_unscaled(host, x, eps);
    let scale_v = host.to_f32(scale);
    let width = normed.layout().shape()[1];
    let scaled_values: Vec<f32> = host
        .to_f32(&normed)
        .iter()
        .enumerate()
        .map(|(i, v)| v * scale_v[i % width] * root_scale)
        .collect();
    let scaled = host.bf16(normed.layout().shape().to_vec(), &scaled_values);
    let scores = linear(host, &scaled, proj);
    let scores = host.to_f32(&dfn::widen(host.device(), &scores).unwrap());
    let (tokens, experts) = (
        scaled.layout().shape()[0],
        scores.len() / scaled.layout().shape()[0],
    );
    let per_expert = host.to_f32(per_expert_scale);
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
        let mut order: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut wsum = 0.0f32;
        for (_i, p) in order.iter().take(top_k) {
            wsum += p;
        }
        for (slot, (i, p)) in order.into_iter().take(top_k).enumerate() {
            weights[t * top_k + slot] = p / wsum * per_expert[i];
            indices[t * top_k + slot] = i as f32;
        }
    }
    (weights, indices)
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
    let inter = gate_up.layout().shape()[0] / 2;
    let xv = host.to_f32(x);
    let mut out = vec![0.0f32; tokens * width];
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
            let act = 0.5
                * acc_g
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt()
                        * (acc_g + 0.044715 * acc_g * acc_g * acc_g))
                        .tanh());
            let v = act * acc_u;
            for (col, dst) in acc_row.iter_mut().enumerate().take(width) {
                *dst += v * dp[(e * width + col) * inter + i].to_f32();
            }
        }
        let w = weights[p];
        for (col, dst) in out[token * width..(token + 1) * width]
            .iter_mut()
            .enumerate()
        {
            *dst += w * acc_row[col];
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
        let ch: Vec<f32> = inv.iter().map(|f| (*x as f32 * f).cos()).collect();
        let sh: Vec<f32> = inv.iter().map(|f| (*x as f32 * f).sin()).collect();
        let cw: Vec<f32> = inv.iter().map(|f| (*y as f32 * f).cos()).collect();
        let sw: Vec<f32> = inv.iter().map(|f| (*y as f32 * f).sin()).collect();
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
                    out[pbase + i] = x0 * c0 - x1 * s0;
                    out[pbase + i + half] = x1 * c1 + x0 * s1;
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

/// Matmul `[rows, inner] x [inner, cols]` (F32 result when inputs BF16).
pub fn matmul(host: &Host, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    ops::matmul(host.device(), lhs, rhs).unwrap()
}
