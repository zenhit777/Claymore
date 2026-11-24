//! Full CPU reference forward for DiffusionGemma (M10.3).
//!
//! Mirrors the pinned upstream composition exactly. All weights are read from
//! the pinned checkpoint on demand; activations stay in host tensors. Batch
//! size 1. This is the reference the CUDA campaign must match.

use claymore_tensor::Tensor;

use crate::checkpoint::Checkpoint;
use crate::config::GemmaModelConfig;
use crate::graph::{
    add, attention_op, clone_tensor, concat_head_rows, concat_rows, dense_mlp, embed_gather_scaled,
    experts_forward, linear, rms_norm, rms_norm_unscaled, router_forward, text_embedding_scale,
    transpose_weight, Host,
};

fn bits(host: &Host, ckpt: &Checkpoint, name: &str, shape: Vec<usize>) -> Tensor {
    let bits = ckpt
        .read_bf16_bits(name)
        .unwrap_or_else(|e| panic!("missing tensor {name}: {e}"));
    use claymore_core::MemoryTier;
    use claymore_device::{DType, Device};
    let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_ne_bytes()).collect();
    let mut buffer = host
        .device()
        .allocate(MemoryTier::Host, raw.len() as u64)
        .unwrap();
    host.device().write(buffer.as_mut(), 0, &raw).unwrap();
    Tensor::new(buffer, shape, DType::BF16).unwrap()
}

fn tensor_of(host: &Host, ckpt: &Checkpoint, name: &str) -> Tensor {
    let shape = ckpt
        .tensor(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"))
        .entry
        .shape()
        .to_vec();
    bits(host, ckpt, name, shape)
}

/// Text embedding table used by both the encoder and decoder.
pub fn text_embedding_weight(host: &Host, ckpt: &Checkpoint) -> Tensor {
    tensor_of(host, ckpt, "model.decoder.embed_tokens.weight")
}

fn rope_tables_text(
    head_dim: usize,
    theta: f32,
    positions: &[usize],
    partial: Option<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let frequencies = head_dim / 2;
    let rope_angles = partial.map_or(frequencies, |factor| {
        (head_dim as f32 * factor / 2.0) as usize
    });
    let inv: Vec<f32> = (0..frequencies)
        .map(|i| {
            if i < rope_angles {
                theta.powf((2 * i) as f32 / head_dim as f32).recip()
            } else {
                0.0
            }
        })
        .collect();
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for p in positions {
        let mut row_c = Vec::with_capacity(head_dim);
        let mut row_s = Vec::with_capacity(head_dim);
        for frequency in &inv {
            let angle = *p as f32 * frequency;
            row_c.push(claymore_tensor::Bf16::from_f32(angle.cos()).to_f32());
            row_s.push(claymore_tensor::Bf16::from_f32(angle.sin()).to_f32());
        }
        row_c.extend_from_slice(&row_c.clone());
        row_s.extend_from_slice(&row_s.clone());
        cos.extend(row_c);
        sin.extend(row_s);
    }
    (cos, sin)
}

/// RoPE on `[tokens, heads, head_dim]` with per-token cos/sin tables
/// `[tokens, head_dim]` (standard rotary, pairs `i` with `i + head_dim/2`).
fn apply_rope(host: &Host, x: &Tensor, cos: &[f32], sin: &[f32], heads: usize) -> Tensor {
    let tokens = x.layout().shape()[0];
    let dim = x.layout().shape()[2];
    let half = dim / 2;
    let xv = host.to_f32(x);
    let mut out = xv.clone();
    let round = |value: f32| claymore_tensor::Bf16::from_f32(value).to_f32();
    for t in 0..tokens {
        for h in 0..heads {
            let base = (t * heads + h) * dim;
            for i in 0..half {
                let c = cos[t * dim + i];
                let s = sin[t * dim + i];
                let a = xv[base + i];
                let b = xv[base + i + half];
                out[base + i] = round(round(a * c) - round(b * s));
                out[base + i + half] = round(round(b * c) + round(a * s));
            }
        }
    }
    host.bf16(x.layout().shape().to_vec(), &out)
}

/// One attention block: projections, head norms, rope, cache append, masked
/// attention, output projection. `positions` carries the absolute position
/// ids of the current pass.
#[allow(clippy::too_many_arguments)]
fn attention_block(
    host: &Host,
    ckpt: &Checkpoint,
    prefix: &str,
    layer: usize,
    x: &Tensor,
    cache: &mut crate::graph::KvCache,
    positions: &[usize],
    head_dim: usize,
    heads: usize,
    kv_heads: usize,
    rope_theta: f32,
    partial: Option<f32>,
    sliding: Option<usize>,
    is_causal: bool,
    eps: f64,
    append_cache: bool,
) -> Tensor {
    let p = |n: &str| format!("model.{prefix}.layers.{layer}.self_attn.{n}");
    let q_w = tensor_of(host, ckpt, &p("q_proj.weight"));
    let k_w = tensor_of(host, ckpt, &p("k_proj.weight"));
    let has_v = ckpt.tensor(&p("v_proj.weight")).is_some();
    let q = linear(host, x, &q_w);
    let q_shape = q.layout().shape().to_vec();
    let q3 = host.bf16(vec![q_shape[0], heads, head_dim], &host.to_f32(&q));
    let qn = rms_norm(host, &q3, &tensor_of(host, ckpt, &p("q_norm.weight")), eps);
    let (cos, sin) = rope_tables_text(head_dim, rope_theta, positions, partial);
    let qr = apply_rope(host, &qn, &cos, &sin, heads);

    let k = linear(host, x, &k_w);
    let k3 = host.bf16(vec![q_shape[0], kv_heads, head_dim], &host.to_f32(&k));
    let kn = rms_norm(host, &k3, &tensor_of(host, ckpt, &p("k_norm.weight")), eps);
    let kr = apply_rope(host, &kn, &cos, &sin, kv_heads);

    let v3 = if has_v {
        let v_w = tensor_of(host, ckpt, &p("v_proj.weight"));
        let v = linear(host, x, &v_w);
        host.bf16(vec![q_shape[0], kv_heads, head_dim], &host.to_f32(&v))
    } else {
        k3
    };
    let vn = rms_norm_unscaled(host, &v3, eps);
    let (keys, values) = if append_cache {
        cache.append(host, layer, kr, vn);
        (
            clone_tensor(host, &cache.layers[layer].keys),
            clone_tensor(host, &cache.layers[layer].values),
        )
    } else {
        // The decoder reads the encoder cache read-only: the canvas K/V are
        // concatenated for this pass only, exactly like the pinned
        // `append_to_cache`, which never mutates the cache object.
        (
            concat_head_rows(host, &cache.layers[layer].keys, &kr),
            concat_head_rows(host, &cache.layers[layer].values, &vn),
        )
    };
    let kv_len = keys.layout().shape()[0];
    let q_len = qr.layout().shape()[0];
    let mask = build_mask(host, positions, kv_len, sliding, is_causal);
    let attn = attention_op(host, &qr, &keys, &values, mask.as_ref(), 1.0);
    let flat = host.bf16(vec![q_len, heads * head_dim], &host.to_f32(&attn));
    linear(host, &flat, &tensor_of(host, ckpt, &p("o_proj.weight")))
}

fn build_mask(
    host: &Host,
    positions: &[usize],
    kv_len: usize,
    sliding: Option<usize>,
    is_causal: bool,
) -> Option<Tensor> {
    let q_len = positions.len();
    let mut values = Vec::with_capacity(q_len * kv_len);
    let mut any_masked = false;
    for &q_abs in positions {
        for k_abs in 0..kv_len {
            let visible = if is_causal && k_abs > q_abs {
                false
            } else if let Some(w) = sliding {
                (q_abs as i64 - k_abs as i64).abs() < w as i64
            } else {
                true
            };
            values.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
            any_masked |= !visible;
        }
    }
    if !any_masked {
        return None;
    }
    Some(host.f32(vec![q_len, kv_len], &values))
}

/// Per-layer encoder diagnostics captured by [`encoder_forward_traced`].
pub struct EncoderLayerTrace {
    pub hidden: Tensor,
    pub input_normed: Tensor,
    pub attention_output: Tensor,
    pub routing_residual: Tensor,
    pub expert_input: Tensor,
    pub mlp_output: Tensor,
    pub router_scores: Vec<f32>,
    pub router_raw_scores: Vec<f32>,
    pub router_weights: Vec<f32>,
    pub router_indices: Vec<f32>,
    pub expert_output: Tensor,
}

/// Encoder text forward: ids -> last hidden (post final norm). Fills the KV
/// cache layer by layer.
pub fn encoder_forward(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    ids: &[u32],
    cache: &mut crate::graph::KvCache,
) -> Tensor {
    let mut layer_trace = Vec::new();
    encoder_forward_traced(host, ckpt, cfg, ids, cache, &mut layer_trace)
}

/// Encoder forward with one BF16 snapshot after each layer. The trace is
/// intended for campaign diagnostics; the returned value and cache semantics
/// are identical to [`encoder_forward`].
pub fn encoder_forward_traced(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    ids: &[u32],
    cache: &mut crate::graph::KvCache,
    layer_trace: &mut Vec<EncoderLayerTrace>,
) -> Tensor {
    encoder_forward_traced_limited(host, ckpt, cfg, ids, cache, layer_trace, usize::MAX)
}

/// Encoder forward truncated to `layer_count` layers, for campaign diagnostics.
pub fn encoder_forward_traced_limited(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    ids: &[u32],
    cache: &mut crate::graph::KvCache,
    layer_trace: &mut Vec<EncoderLayerTrace>,
    layer_count: usize,
) -> Tensor {
    let text = cfg.text_config.as_ref().unwrap();
    let hidden = text.hidden_size.unwrap();
    let eps = text.rms_norm_eps.unwrap_or(1e-6);
    let scale = text_embedding_scale(hidden);
    let embed = embed_gather_scaled(
        host,
        ids,
        &tensor_of(host, ckpt, "model.decoder.embed_tokens.weight"),
        scale,
    );
    let mut x = embed;
    let positions: Vec<usize> = (0..ids.len()).collect();
    let layer_count = layer_count.min(text.num_hidden_layers.unwrap());
    for layer in 0..layer_count {
        let global = text
            .layer_types
            .get(layer)
            .map(|t| t == "full_attention")
            .unwrap_or(false);
        let head_dim = if global {
            text.global_head_dim.unwrap()
        } else {
            text.head_dim.unwrap()
        };
        let kv_heads = if global {
            text.num_global_key_value_heads.unwrap()
        } else {
            text.num_key_value_heads.unwrap()
        };
        let heads = text.num_attention_heads.unwrap();
        let rope = if global { 1_000_000.0f32 } else { 10_000.0f32 };
        let partial = if global { Some(0.25f32) } else { None };
        let sliding = if global {
            None
        } else {
            Some(text.sliding_window.unwrap_or(1024))
        };
        let normed = rms_norm(
            host,
            &x,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.input_layernorm.weight"),
            ),
            eps,
        );
        let attn_out = attention_block(
            host, ckpt, "decoder", layer, &normed, cache, &positions, head_dim, heads, kv_heads,
            rope, partial, sliding, true, eps, true,
        );
        let post = rms_norm(
            host,
            &attn_out,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_attention_layernorm.weight"),
            ),
            eps,
        );
        x = add(host, &x, &post);
        // Encoder FFN is dense only in the pinned code path? No: encoder layers
        // include MoE too; the dense MLP + MoE composition follows the decoder
        // composition with encoder tensor names.
        let mlp_out = dense_mlp(
            host,
            &rms_norm(
                host,
                &x,
                &tensor_of(
                    host,
                    ckpt,
                    &format!("model.decoder.layers.{layer}.pre_feedforward_layernorm.weight"),
                ),
                eps,
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.mlp.gate_proj.weight"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.mlp.up_proj.weight"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.mlp.down_proj.weight"),
            ),
        );
        let pff1 = rms_norm(
            host,
            &mlp_out,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_feedforward_layernorm_1.weight"),
            ),
            eps,
        );
        let residual = clone_tensor(host, &x);
        let pre2 = rms_norm(
            host,
            &residual,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.pre_feedforward_layernorm_2.weight"),
            ),
            eps,
        );
        let (scores, raw_scores, weights, indices) = router_forward(
            host,
            &residual,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.router.proj.weight"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.router.scale"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.router.per_expert_scale"),
            ),
            eps,
            (hidden as f32).powf(-0.5),
            text.top_k_experts.unwrap(),
        );
        let exp = experts_forward(
            host,
            &pre2,
            &weights,
            &indices,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.experts.gate_up_proj"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.experts.down_proj"),
            ),
        );
        let pff2 = rms_norm(
            host,
            &exp,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_feedforward_layernorm_2.weight"),
            ),
            eps,
        );
        let comb = add(host, &pff1, &pff2);
        let pff = rms_norm(
            host,
            &comb,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_feedforward_layernorm.weight"),
            ),
            eps,
        );
        x = add(host, &residual, &pff);
        x = layer_scalar_mul(
            host,
            &x,
            &tensor_of(
                host,
                ckpt,
                &format!("model.encoder.language_model.layers.{layer}.layer_scalar"),
            ),
        );
        layer_trace.push(EncoderLayerTrace {
            hidden: clone_tensor(host, &x),
            input_normed: clone_tensor(host, &normed),
            attention_output: clone_tensor(host, &attn_out),
            routing_residual: clone_tensor(host, &residual),
            expert_input: clone_tensor(host, &pre2),
            mlp_output: clone_tensor(host, &mlp_out),
            router_scores: scores,
            router_raw_scores: raw_scores,
            router_weights: weights,
            router_indices: indices,
            expert_output: clone_tensor(host, &exp),
        });
    }
    rms_norm(
        host,
        &x,
        &tensor_of(host, ckpt, "model.decoder.norm.weight"),
        eps,
    )
}

fn layer_scalar_mul(host: &Host, x: &Tensor, scalar: &Tensor) -> Tensor {
    let width = x.layout().shape().last().copied().unwrap();
    let sv = host.to_f32(scalar);
    let scale = host.bf16(vec![width], &vec![sv[0]; width]);
    let zero = host.bf16(vec![width], &vec![0.0f32; width]);
    crate::graph::scale_bias(host, x, &scale, &zero)
}

pub fn scale_bias(host: &Host, x: &Tensor, scale: &Tensor, bias: &Tensor) -> Tensor {
    claymore_tensor::diffusion::scale_bias(host.device(), x, scale, bias).unwrap()
}

/// Decoder canvas forward with self-conditioning. `self_cond` is F32 logits
/// `[canvas_len, vocab]` from the previous step, None for the first step.
/// Traced decoder pass: `trace` receives `(scores, weights, indices)` for
/// every layer, in layer order, as F32 vectors.
pub fn decoder_forward(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    canvas: &[u32],
    cache: &mut crate::graph::KvCache,
    self_cond: Option<&Vec<f32>>,
    trace: &mut Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
) -> Tensor {
    let mut layer_trace = Vec::new();
    let mut stages = Vec::new();
    decoder_forward_traced(
        host,
        ckpt,
        cfg,
        canvas,
        cache,
        self_cond,
        trace,
        &mut layer_trace,
        &mut stages,
    )
}

/// Number of intermediate snapshots recorded per decoder layer by
/// [`decoder_forward_traced`], in order: attention output (after `o_proj`),
/// `post_attention_layernorm`, residual after the attention add, dense MLP
/// output, `post_feedforward_layernorm_1`, `pre_feedforward_layernorm_2`, raw
/// expert output, `post_feedforward_layernorm_2`, MLP+MoE sum,
/// `post_feedforward_layernorm`, and the layer output.
pub const DECODER_STAGES_PER_LAYER: usize = 11;

/// Decoder forward with per-layer BF16 snapshots in addition to the MoE
/// router trace.
#[allow(clippy::too_many_arguments)]
pub fn decoder_forward_traced(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    canvas: &[u32],
    cache: &mut crate::graph::KvCache,
    self_cond: Option<&Vec<f32>>,
    trace: &mut Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    layer_trace: &mut Vec<Tensor>,
    stages: &mut Vec<Tensor>,
) -> Tensor {
    let text = cfg.text_config.as_ref().unwrap();
    let hidden = text.hidden_size.unwrap();
    let vocab = text.vocab_size.unwrap();
    let eps = text.rms_norm_eps.unwrap_or(1e-6);
    let scale = text_embedding_scale(hidden);
    let embed_table = tensor_of(host, ckpt, "model.decoder.embed_tokens.weight");
    let embed = embed_gather_scaled(host, canvas, &embed_table, scale);
    let canvas_len = canvas.len();
    let mut inputs = embed;
    if let Some(logits) = self_cond {
        // softmax in F32 -> bf16 -> matmul with embed table -> scale
        let probs = softmax_rows(host, logits, canvas_len, vocab);
        let probs_t = host.bf16(vec![canvas_len, vocab], &probs);
        let soft = claymore_tensor::ops::matmul(
            host.device(),
            &probs_t,
            &transpose_weight(host, &embed_table),
        )
        .unwrap();
        let soft_scaled = host.bf16(
            vec![canvas_len, hidden],
            &host
                .to_f32(&soft)
                .iter()
                .map(|v| v * scale)
                .collect::<Vec<f32>>(),
        );
        // self_conditioning: pre_norm(soft) -> gate/up gelu -> down -> add
        let pre_norm = rms_norm(
            host,
            &soft_scaled,
            &tensor_of(
                host,
                ckpt,
                "model.decoder.self_conditioning.pre_norm.weight",
            ),
            eps,
        );
        let sc = dense_mlp(
            host,
            &pre_norm,
            &tensor_of(
                host,
                ckpt,
                "model.decoder.self_conditioning.gate_proj.weight",
            ),
            &tensor_of(host, ckpt, "model.decoder.self_conditioning.up_proj.weight"),
            &tensor_of(
                host,
                ckpt,
                "model.decoder.self_conditioning.down_proj.weight",
            ),
        );
        inputs = add(host, &inputs, &sc);
    }
    // The upstream self-conditioning module always applies its unscaled
    // post-norm. With no previous logits, the MLP signal is exactly zero
    // but this final normalization is still part of the decoder input.
    inputs = rms_norm_unscaled(host, &inputs, eps);
    let prefix_len = cache.seq_len();
    let positions: Vec<usize> = (prefix_len..prefix_len + canvas_len).collect();
    for layer in 0..text.num_hidden_layers.unwrap() {
        let global = text
            .layer_types
            .get(layer)
            .map(|t| t == "full_attention")
            .unwrap_or(false);
        let head_dim = if global {
            text.global_head_dim.unwrap()
        } else {
            text.head_dim.unwrap()
        };
        let kv_heads = if global {
            text.num_global_key_value_heads.unwrap()
        } else {
            text.num_key_value_heads.unwrap()
        };
        let heads = text.num_attention_heads.unwrap();
        let rope = if global { 1_000_000.0f32 } else { 10_000.0f32 };
        let partial = if global { Some(0.25f32) } else { None };
        // Decoder attention is fully bidirectional over both the encoder
        // prefix and the canvas; the default upstream decoder mask does not
        // impose the encoder's sliding window.
        let sliding = None;
        let normed = rms_norm(
            host,
            &inputs,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.input_layernorm.weight"),
            ),
            eps,
        );
        let attn_out = attention_block(
            host, ckpt, "decoder", layer, &normed, cache, &positions, head_dim, heads, kv_heads,
            rope, partial, sliding, false, eps, false,
        );
        let post = rms_norm(
            host,
            &attn_out,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_attention_layernorm.weight"),
            ),
            eps,
        );
        let x = add(host, &inputs, &post);
        stages.push(clone_tensor(host, &attn_out));
        stages.push(clone_tensor(host, &post));
        stages.push(clone_tensor(host, &x));
        let pre_ff = rms_norm(
            host,
            &x,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.pre_feedforward_layernorm.weight"),
            ),
            eps,
        );
        let mlp_out = dense_mlp(
            host,
            &pre_ff,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.mlp.gate_proj.weight"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.mlp.up_proj.weight"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.mlp.down_proj.weight"),
            ),
        );
        stages.push(clone_tensor(host, &mlp_out));
        let pff1 = rms_norm(
            host,
            &mlp_out,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_feedforward_layernorm_1.weight"),
            ),
            eps,
        );
        stages.push(clone_tensor(host, &pff1));
        let pre2 = rms_norm(
            host,
            &x,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.pre_feedforward_layernorm_2.weight"),
            ),
            eps,
        );
        stages.push(clone_tensor(host, &pre2));
        let (scores, _raw_scores, weights, indices) = router_forward(
            host,
            &x,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.router.proj.weight"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.router.scale"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.router.per_expert_scale"),
            ),
            eps,
            (hidden as f32).powf(-0.5),
            text.top_k_experts.unwrap(),
        );
        trace.push((scores, weights.to_vec(), indices.to_vec()));
        let exp = experts_forward(
            host,
            &pre2,
            &weights,
            &indices,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.experts.gate_up_proj"),
            ),
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.experts.down_proj"),
            ),
        );
        stages.push(clone_tensor(host, &exp));
        let pff2 = rms_norm(
            host,
            &exp,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_feedforward_layernorm_2.weight"),
            ),
            eps,
        );
        stages.push(clone_tensor(host, &pff2));
        let comb = add(host, &pff1, &pff2);
        stages.push(clone_tensor(host, &comb));
        let pff = rms_norm(
            host,
            &comb,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.post_feedforward_layernorm.weight"),
            ),
            eps,
        );
        stages.push(clone_tensor(host, &pff));
        let mut y = add(host, &x, &pff);
        y = layer_scalar_mul(
            host,
            &y,
            &tensor_of(
                host,
                ckpt,
                &format!("model.decoder.layers.{layer}.layer_scalar"),
            ),
        );
        inputs = y;
        stages.push(clone_tensor(host, &inputs));
        layer_trace.push(clone_tensor(host, &inputs));
    }
    rms_norm(
        host,
        &inputs,
        &tensor_of(host, ckpt, "model.decoder.norm.weight"),
        eps,
    )
}

fn softmax_rows(host: &Host, logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; logits.len()];
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..cols {
            let p = (row[i] - max).exp();
            out[r * cols + i] = p;
            sum += p;
        }
        for v in out[r * cols..(r + 1) * cols].iter_mut() {
            *v /= sum;
        }
    }
    let _ = host;
    out
}

/// Tied lm head with final logit softcapping: F32, /cap, tanh, *cap.
pub fn lm_logits(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    hidden: &Tensor,
) -> Tensor {
    let cap = cfg
        .text_config
        .as_ref()
        .unwrap()
        .final_logit_softcapping
        .unwrap() as f32;
    let w = tensor_of(host, ckpt, "model.decoder.embed_tokens.weight");
    let logits =
        claymore_tensor::ops::matmul(host.device(), hidden, &transpose_weight(host, &w)).unwrap();
    let widened = claymore_tensor::diffusion::widen(host.device(), &logits).unwrap();
    claymore_tensor::diffusion::soft_cap(host.device(), &widened, cap).unwrap()
}

pub fn layer_scalar_mul_public(host: &Host, x: &Tensor, scalar: &Tensor) -> Tensor {
    layer_scalar_mul(host, x, scalar)
}

pub fn concat_rows_public(host: &Host, a: &Tensor, b: &Tensor) -> Tensor {
    concat_rows(host, a, b)
}

pub fn transpose_public(host: &Host, w: &Tensor) -> Tensor {
    transpose_weight(host, w)
}

/// Full vision tower forward for one image (batch 1): patch embedder, 27
/// layers with axial 2D rope, pooling, standardization. Returns
/// `(post_encoder [patches, hidden], pooled_standardized [out_len, hidden])`
/// matching `Gemma4VisionModel` `last_hidden_state` and `pooler_output`.
pub fn vision_forward(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    pixels: &[f32],
    positions: &[(u32, u32)],
    pooling_kernel: usize,
) -> (Tensor, Tensor) {
    let mut trace = Vec::new();
    vision_forward_traced(
        host,
        ckpt,
        cfg,
        pixels,
        positions,
        pooling_kernel,
        &mut trace,
    )
}

/// Number of intermediate snapshots recorded per vision layer by
/// [`vision_forward_stages`].
pub const VISION_STAGES_PER_LAYER: usize = 7;

/// Vision tower forward with one BF16 snapshot per layer, for campaign
/// diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn vision_forward_traced(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    pixels: &[f32],
    positions: &[(u32, u32)],
    pooling_kernel: usize,
    trace: &mut Vec<Tensor>,
) -> (Tensor, Tensor) {
    let mut stages = Vec::new();
    let (post_encoder, pooled) = vision_forward_stages(
        host,
        ckpt,
        cfg,
        pixels,
        positions,
        pooling_kernel,
        &mut stages,
    );
    for layer in stages.chunks_exact(VISION_STAGES_PER_LAYER) {
        trace.push(clone_tensor(host, &layer[VISION_STAGES_PER_LAYER - 1]));
    }
    (post_encoder, pooled)
}

/// Vision tower forward recording the upstream per-layer hook outputs, in
/// order: input layernorm, attention output (after `o_proj`), post-attention
/// layernorm, pre-feedforward layernorm, MLP output, post-feedforward
/// layernorm, and the residual layer output.
#[allow(clippy::too_many_arguments)]
pub fn vision_forward_stages(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    pixels: &[f32],
    positions: &[(u32, u32)],
    pooling_kernel: usize,
    stages: &mut Vec<Tensor>,
) -> (Tensor, Tensor) {
    let vision = cfg.vision_config.as_ref().unwrap();
    let hidden = vision.hidden_size.unwrap();
    let heads = vision.num_attention_heads.unwrap();
    let kv_heads = vision.num_key_value_heads.unwrap();
    let head_dim = vision.head_dim.unwrap();
    let eps = vision.rms_norm_eps.unwrap_or(1e-6);
    let patches = positions.len();
    let per_patch = pixels.len() / patches;
    let scaled: Vec<f32> = pixels.iter().map(|v| 2.0 * (v - 0.5)).collect();
    let mut x = linear(
        host,
        &host.bf16(vec![patches, per_patch], &scaled),
        &tensor_of(
            host,
            ckpt,
            "model.encoder.vision_tower.patch_embedder.input_proj.weight",
        ),
    );
    // Position embeddings: table [2, 10240, hidden].
    let table = tensor_of(
        host,
        ckpt,
        "model.encoder.vision_tower.patch_embedder.position_embedding_table",
    );
    let tv = host.to_f32(&table);
    let psize = vision.position_embedding_size.unwrap();
    let mut pos_emb = Vec::with_capacity(patches * hidden);
    for (xpos, ypos) in positions {
        let xbase = (*xpos as usize) * hidden;
        let ybase = psize * hidden + (*ypos as usize) * hidden;
        for h in 0..hidden {
            pos_emb.push(tv[xbase + h] + tv[ybase + h]);
        }
    }
    x = add(host, &x, &host.bf16(vec![patches, hidden], &pos_emb));
    let (cos, sin) = crate::graph::vision_rope_tables(head_dim, 100.0f32, positions);
    for layer in 0..vision.num_hidden_layers.unwrap() {
        let p = |n: &str| format!("model.encoder.vision_tower.encoder.layers.{layer}.{n}");
        let normed = rms_norm(
            host,
            &x,
            &tensor_of(host, ckpt, &p("input_layernorm.weight")),
            eps,
        );
        stages.push(clone_tensor(host, &normed));
        let q = linear(
            host,
            &normed,
            &tensor_of(host, ckpt, &p("self_attn.q_proj.linear.weight")),
        );
        let q3 = host.bf16(vec![patches, heads, head_dim], &host.to_f32(&q));
        let qn = rms_norm(
            host,
            &q3,
            &tensor_of(host, ckpt, &p("self_attn.q_norm.weight")),
            eps,
        );
        let qr = crate::graph::apply_multidim_rope(host, &qn, &cos, &sin, heads);
        let k = linear(
            host,
            &normed,
            &tensor_of(host, ckpt, &p("self_attn.k_proj.linear.weight")),
        );
        let k3 = host.bf16(vec![patches, kv_heads, head_dim], &host.to_f32(&k));
        let kn = rms_norm(
            host,
            &k3,
            &tensor_of(host, ckpt, &p("self_attn.k_norm.weight")),
            eps,
        );
        let kr = crate::graph::apply_multidim_rope(host, &kn, &cos, &sin, kv_heads);
        let v_w = tensor_of(host, ckpt, &p("self_attn.v_proj.linear.weight"));
        let v3 = host.bf16(
            vec![patches, kv_heads, head_dim],
            &host.to_f32(&linear(host, &normed, &v_w)),
        );
        let vn = rms_norm_unscaled(host, &v3, eps);
        let attn = attention_op(host, &qr, &kr, &vn, None, 1.0);
        let flat = host.bf16(vec![patches, heads * head_dim], &host.to_f32(&attn));
        let o = linear(
            host,
            &flat,
            &tensor_of(host, ckpt, &p("self_attn.o_proj.linear.weight")),
        );
        stages.push(clone_tensor(host, &o));
        let post = rms_norm(
            host,
            &o,
            &tensor_of(host, ckpt, &p("post_attention_layernorm.weight")),
            eps,
        );
        stages.push(clone_tensor(host, &post));
        x = add(host, &x, &post);
        let pre_ff = rms_norm(
            host,
            &x,
            &tensor_of(host, ckpt, &p("pre_feedforward_layernorm.weight")),
            eps,
        );
        stages.push(clone_tensor(host, &pre_ff));
        let mlp = dense_mlp(
            host,
            &pre_ff,
            &tensor_of(host, ckpt, &p("mlp.gate_proj.linear.weight")),
            &tensor_of(host, ckpt, &p("mlp.up_proj.linear.weight")),
            &tensor_of(host, ckpt, &p("mlp.down_proj.linear.weight")),
        );
        stages.push(clone_tensor(host, &mlp));
        let pff = rms_norm(
            host,
            &mlp,
            &tensor_of(host, ckpt, &p("post_feedforward_layernorm.weight")),
            eps,
        );
        stages.push(clone_tensor(host, &pff));
        x = add(host, &x, &pff);
        stages.push(clone_tensor(host, &x));
    }
    // Pooling: k^2 groups by position grid, mean over each group.
    let out_len = patches / (pooling_kernel * pooling_kernel);
    let max_x = positions.iter().map(|p| p.0).max().unwrap() + 1;
    let xv = host.to_f32(&x);
    let mut pooled = vec![0.0f32; out_len * hidden];
    let inv_k2 = 1.0f32 / (pooling_kernel * pooling_kernel) as f32;
    for (i, (xpos, ypos)) in positions.iter().enumerate() {
        let gx = (*xpos as usize) / pooling_kernel;
        let gy = (*ypos as usize) / pooling_kernel;
        let group = gx + (max_x as usize / pooling_kernel) * gy;
        for h in 0..hidden {
            pooled[group * hidden + h] += xv[i * hidden + h] * inv_k2;
        }
    }
    let root = (hidden as f32).sqrt();
    let pooled_bf16: Vec<f32> = pooled
        .iter()
        .map(|value| claymore_tensor::Bf16::from_f32(*value).to_f32())
        .collect();
    let pooled_scaled: Vec<f32> = pooled_bf16.iter().map(|v| v * root).collect();
    let std_bias = tensor_of(host, ckpt, "model.encoder.vision_tower.std_bias");
    let std_scale = tensor_of(host, ckpt, "model.encoder.vision_tower.std_scale");
    let bv = host.to_f32(&std_bias);
    let sv = host.to_f32(&std_scale);
    let mut out = vec![0.0f32; out_len * hidden];
    for g in 0..out_len {
        for h in 0..hidden {
            out[g * hidden + h] = (pooled_scaled[g * hidden + h] - bv[h]) * sv[h];
        }
    }
    (x, host.bf16(vec![out_len, hidden], &out))
}

/// `DiffusionGemmaMultimodalEmbedder`: unscaled RMSNorm + projection.
pub fn embed_vision_forward(
    host: &Host,
    ckpt: &Checkpoint,
    cfg: &GemmaModelConfig,
    vision: &Tensor,
) -> Tensor {
    let vision_cfg = cfg.vision_config.as_ref().unwrap();
    let eps = vision_cfg.rms_norm_eps.unwrap_or(1e-6);
    let normed = rms_norm_unscaled(host, vision, eps);
    linear(
        host,
        &normed,
        &tensor_of(
            host,
            ckpt,
            "model.encoder.embed_vision.embedding_projection.weight",
        ),
    )
}
