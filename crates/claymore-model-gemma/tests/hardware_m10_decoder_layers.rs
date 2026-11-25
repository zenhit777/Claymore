//! Hardware campaign: CUDA ↔ CPU differential of **all 30 decoder layers**
//! and of the tied lm head (logits), on the pinned checkpoint and the
//! versioned M10 layer fixtures.
//!
//! This is the measurement that remains open in the architect decision
//! register (`docs/audits/m10-architect-decisions.md` §2, reserve 1): the
//! grouped campaign of 2026-11-25 replayed the real decoder **subgraph of
//! layer 0** (`hw_m10_09`, bit-exact) but the per-layer CUDA ↔ CPU table of
//! the 30 layers and the logits table had never been produced.
//!
//! Method (mirrors `tests/decoder_layer_bisect.rs` on the CPU side and
//! `tests/hardware_m10_generation.rs::encoder_first_layer_matches_cpu_step_by_step`
//! on the CUDA side):
//!
//! * every layer runs from its **own golden entry state** `dec_in_L` and its
//!   **own reference encoder cache** `enc_kv_L_*` (never a Claymore-derived
//!   chain state, never an empty cache);
//! * the CPU side is the production shared layer body
//!   (`fullgraph::decoder_forward_traced_from_hidden`); the CUDA side is
//!   composed from the `CudaGraph` primitives in exactly the order
//!   `text_layer`/`attention_layer` run them, so every intermediate is
//!   downloadable. The composition is then **verified** against the
//!   unmodified production paths: the official `text_layer` must return the
//!   replica's layer output **bit for bit** (this also proves the official
//!   scope release never frees a tensor that a later op still reads — a
//!   premature release would corrupt one path but not the other), and the
//!   CPU attention replica must reach the production `stages[0]` bit for
//!   bit;
//! * every stage is compared CUDA ↔ CPU with the D1 metric (worst BF16 row
//!   step relative to the row peak, plus `max_abs`/`rms`), and the layer
//!   output is additionally tracked against the golden (D2: tracking, never
//!   a golden gate here);
//! * the router is compared as raw F32 scores (both from the same pinned
//!   weights), as the post-softmax weights, and as the discrete **expert
//!   index sets** against the reference set, with the same boundary-swap
//!   policy the CPU bisect contract fixed;
//! * after the full 30-layer table, a second CUDA pass re-runs every layer
//!   and every layer output must come back **bit-identical** (in-session
//!   reproducibility, arena/LRU cross-layer contamination check).
//!
//! Numerical gates are **fixed before the run** (never loosened from an
//! observed value; the logits inherit the bounds already coded in
//! `hardware_m10_generation.rs`):
//!
//! | metric | bound | provenance |
//! |---|---|---|
//! | layer-stage CUDA↔CPU worst-row drift | `20.0` BF16 steps | D1 rule applied to the decoder layer drift already measured CPU↔reference (10.0, layer 26, bound 20.0): the CUDA path must not drift from the reference path more than the reference path itself is bound to |
//! | router weights / raw scores CUDA↔CPU | `max_abs ≤ 1e-3` | the tie-precision bound `MAX_SCORE_DRIFT` the decoder bisect contract fixed from the reference's own 8-vs-9 gaps |
//! | router expert sets | exact, else one reference-tied boundary swap, zero structural | decoder bisect contract, `docs/audits/m10-cpu-campaign-2026-11-25.md` §7 |
//! | logits CUDA↔CPU | `max_abs ≤ 2.0`, `rms ≤ 0.1` | the bounds already coded in `hardware_m10_generation.rs`; the isolated head cannot legitimately be looser |
//! | all comparisons | finite values, peak VRAM ≤ 28 GiB | M10 hard ceiling |
//!
//! The ranking divergence of the logits (argmax / top-8 flips CUDA vs CPU)
//! is **reported**, per D2; turning it into a contract is a register
//! decision to take once the numbers exist (tightening only, never
//! loosening).
//!
//! ```text
//! RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib \
//! CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
//! cargo test --release -p claymore-model-gemma --features cuda-streaming \
//!   --test hardware_m10_decoder_layers --locked -- \
//!   --ignored --test-threads=1 --nocapture
//! ```

use std::path::Path;

use claymore_memory::{CudaMemoryManager, CudaTensorMemory, EvictionPolicy, MemoryBudgets};
use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::cuda_graph::{CudaGraph, KvLayerEntry};
use claymore_model_gemma::fullgraph::{
    checkpoint_tensor, decoder_forward_traced_from_hidden, lm_logits,
};
use claymore_model_gemma::graph::{
    self, add, attention_op, clone_tensor, concat_head_rows, linear, rms_norm, rms_norm_unscaled,
    Host, KvCache,
};
use claymore_tensor::{Bf16, Tensor};

const CANVAS: usize = 16;
const HIDDEN: usize = 2816;
const LAYER_COUNT: usize = 30;
const PREFIX: usize = 7;
const TOP_K: usize = 8;
const EXPERTS: usize = 128;
const GOLDEN: &str = "/tmp/opencode/m10/golden_eager";

/// Gates fixed **before** the run — see the module table. Never loosen these
/// after observation; tightening is a register decision with the campaign
/// evidence attached.
const MAX_ROW_DRIFT_STEPS: f32 = 20.0;
const MAX_SCORE_DRIFT: f32 = 1.0e-3;
const LOGITS_MAX_ABS: f32 = 2.0;
const LOGITS_RMS: f32 = 0.1;

fn fixture_dir() -> std::path::PathBuf {
    match std::env::var_os("CLAYMORE_M10_LAYERS_DIR") {
        Some(dir) => Path::new(&dir).to_path_buf(),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("data")
            .join("m10_layers"),
    }
}

fn checkpoint_root() -> std::path::PathBuf {
    match std::env::var_os("CLAYMORE_M10_CHECKPOINT") {
        Some(root) => Path::new(&root).to_path_buf(),
        None => Path::new("/tmp/opencode/m10/checkpoint").to_path_buf(),
    }
}

fn load_f32(path: &Path, name: &str, shape: &[usize]) -> Vec<f32> {
    let bytes = std::fs::read(path.join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("fixture {name}: {e}"));
    let out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        out.len(),
        shape.iter().product::<usize>(),
        "fixture {name} size"
    );
    out
}

fn stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut max_abs = 0.0f32;
    let mut sq = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite value under comparison"
        );
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sq += d * d;
    }
    (max_abs, (sq / a.len() as f32).sqrt())
}

/// Width of one BF16 step at magnitude `x`: 8 significand bits (D1 metric).
fn bf16_step(x: f32) -> f32 {
    let x = x.abs().max(f32::MIN_POSITIVE);
    2f32.powi(((x.to_bits() >> 23) as i32 - 127) - 7)
}

/// D1 metric over a `[rows, width]` tensor: worst row, `|a-b|` measured in
/// BF16 steps of that row's peak, the reference side being `b` (CPU).
fn worst_row_steps(a: &[f32], b: &[f32], rows: usize, width: usize) -> f32 {
    let mut worst = 0.0f32;
    for row in 0..rows {
        let lo = row * width;
        let hi = lo + width;
        let max_abs = a[lo..hi]
            .iter()
            .zip(&b[lo..hi])
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let peak = b[lo..hi].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        worst = worst.max(max_abs / bf16_step(peak));
    }
    worst
}

/// Permute the reference cache `[batch, kv_heads, seq, head_dim]` (batch 1)
/// into the local `[seq, kv_heads, head_dim]` layout.
fn permute_kv_to_local(flat: &[f32], heads: usize, seq: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; flat.len()];
    for head in 0..heads {
        for row in 0..seq {
            for d in 0..dim {
                out[(row * heads + head) * dim + d] = flat[(head * seq + row) * dim + d];
            }
        }
    }
    out
}

fn to_bits(values: &[f32]) -> Vec<Bf16> {
    values.iter().map(|v| Bf16::from_f32(*v)).collect()
}

fn budgets() -> MemoryBudgets {
    MemoryBudgets {
        vram_soft_bytes: 26 * 1024 * 1024 * 1024,
        vram_hard_bytes: 28 * 1024 * 1024 * 1024,
        host_bytes: 32 * 1024 * 1024 * 1024,
        pinned_host_bytes: 8 * 1024 * 1024 * 1024,
        expert_cache_target_bytes: 4 * 1024 * 1024 * 1024,
        expert_cache_max_bytes: 8 * 1024 * 1024 * 1024,
    }
}

fn top_k_set(indices: &[u32], row: usize) -> Vec<u32> {
    let mut s: Vec<u32> = indices[row * TOP_K..(row + 1) * TOP_K].to_vec();
    s.sort_unstable();
    s
}

fn per_layer_params(
    text: &claymore_model_gemma::config::TextConfig,
    layer: usize,
) -> (bool, usize, usize, f32, Option<f32>) {
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
    let rope = if global { 1_000_000.0f32 } else { 10_000.0f32 };
    let partial = if global { Some(0.25f32) } else { None };
    (global, head_dim, kv_heads, rope, partial)
}

// ---------------------------------------------------------------------------
// CPU attention replica (mirrors fullgraph::attention_block read-only).
// ---------------------------------------------------------------------------

struct CpuAttentionStages {
    q: Vec<f32>,
    qn: Vec<f32>,
    qr: Vec<f32>,
    k: Vec<f32>,
    kn: Vec<f32>,
    kr: Vec<f32>,
    vn: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    attn: Vec<f32>,
    attn_out: Vec<f32>,
}

/// Copies of the private `fullgraph` rope helpers: the replica is only sound
/// if it is the same math, and the join below proves it — the replica's
/// `attn_out` must equal the production stage bit for bit.
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

/// Same tensors, same pinned weights, same mask semantics as the CPU
/// production `attention_block` for the decoder (`is_causal=false`,
/// `sliding=None`, `append=false`), but every internal is returned.
#[allow(clippy::too_many_arguments)]
fn cpu_attention_stages(
    host: &Host,
    ckpt: &Checkpoint,
    layer: usize,
    normed: &Tensor,
    cache: &KvCache,
    positions: &[usize],
    head_dim: usize,
    heads: usize,
    kv_heads: usize,
    rope_theta: f32,
    partial: Option<f32>,
    eps: f64,
) -> CpuAttentionStages {
    let p = |n: &str| format!("model.decoder.layers.{layer}.self_attn.{n}");
    let tokens = normed.layout().shape()[0];
    let q = linear(
        host,
        normed,
        &checkpoint_tensor(host, ckpt, &p("q_proj.weight")),
    );
    let q_flat = host.to_f32(&q);
    let q3 = host.bf16(vec![tokens, heads, head_dim], &q_flat);
    let qn_t = rms_norm(
        host,
        &q3,
        &checkpoint_tensor(host, ckpt, &p("q_norm.weight")),
        eps,
    );
    let qn = host.to_f32(&qn_t);
    let (cos, sin) = rope_tables_text(head_dim, rope_theta, positions, partial);
    let qr = apply_rope(host, &qn_t, &cos, &sin, heads);

    let k = linear(
        host,
        normed,
        &checkpoint_tensor(host, ckpt, &p("k_proj.weight")),
    );
    let k_flat = host.to_f32(&k);
    let k3 = host.bf16(vec![tokens, kv_heads, head_dim], &k_flat);
    let kn_t = rms_norm(
        host,
        &k3,
        &checkpoint_tensor(host, ckpt, &p("k_norm.weight")),
        eps,
    );
    let kn = host.to_f32(&kn_t);
    let kr = apply_rope(host, &kn_t, &cos, &sin, kv_heads);

    let has_v = ckpt.tensor(&p("v_proj.weight")).is_some();
    let v3 = if has_v {
        let v = linear(
            host,
            normed,
            &checkpoint_tensor(host, ckpt, &p("v_proj.weight")),
        );
        host.bf16(vec![tokens, kv_heads, head_dim], &host.to_f32(&v))
    } else {
        k3
    };
    let vn_t = rms_norm_unscaled(host, &v3, eps);

    // Decoder pass reads the cache read-only and concatenates for this pass.
    let keys = concat_head_rows(host, &cache.layers[layer].keys, &kr);
    let values = concat_head_rows(host, &cache.layers[layer].values, &vn_t);
    // The pinned decoder mask is fully bidirectional; over positions 7..23
    // the sliding window (even if it were applied) masks nothing — asserted
    // by the caller on both sides.
    let attn = attention_op(host, &qr, &keys, &values, None, 1.0);
    let attn_flat = host.to_f32(&attn);
    let flat = host.bf16(vec![tokens, heads * head_dim], &attn_flat);
    let attn_out = linear(
        host,
        &flat,
        &checkpoint_tensor(host, ckpt, &p("o_proj.weight")),
    );

    CpuAttentionStages {
        q: q_flat,
        qn,
        qr: host.to_f32(&qr),
        k: k_flat,
        kn,
        kr: host.to_f32(&kr),
        vn: host.to_f32(&vn_t),
        keys: host.to_f32(&keys),
        values: host.to_f32(&values),
        attn: attn_flat,
        attn_out: host.to_f32(&attn_out),
    }
}

// ---------------------------------------------------------------------------
// CUDA replica (mirrors CudaGraph::text_layer + attention_layer read-only).
// ---------------------------------------------------------------------------

/// Indices of every intermediate of one CUDA decoder layer, in the exact
/// order `text_layer`/`attention_layer` compose them. `aux` collects the
/// auxiliary tensors (rope tables, ones/zeros vectors, reshapes) so the
/// caller can release the whole layer after its comparisons.
struct CudaStages {
    normed: usize,
    q: usize,
    qn: usize,
    qr: usize,
    k: usize,
    kn: usize,
    kr: usize,
    vn: usize,
    keys: usize,
    values: usize,
    attn: usize,
    attn_out: usize,
    post: usize,
    residual: usize,
    pre1: usize,
    mlp: usize,
    pff1: usize,
    pre2: usize,
    router_scaled: usize,
    router_scaled_scalar: usize,
    router_proj: usize,
    router_scores: usize,
    topk: usize,
    exp: usize,
    pff2: usize,
    comb: usize,
    pff: usize,
    pre_scalar: usize,
    out: usize,
    aux: Vec<usize>,
}

/// Every intermediate stays alive until the caller releases it, once the
/// layer's comparisons are done: a premature release would corrupt a
/// still-referenced tensor and show up as a comparison or join failure.
#[allow(clippy::too_many_arguments)]
fn cuda_layer_stages(
    graph: &mut CudaGraph,
    layer: usize,
    x: usize,
    cache: &[KvLayerEntry],
    positions: &[usize],
    head_dim: usize,
    heads: usize,
    kv_heads: usize,
    rope_theta: f32,
    partial: Option<f32>,
    sliding: Option<usize>,
    eps: f32,
    hidden: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
) -> Result<CudaStages, Box<dyn std::error::Error>> {
    let mut aux = Vec::new();
    let p = |n: String| format!("model.decoder.layers.{layer}.self_attn.{n}");
    let lp = |n: &str| format!("model.decoder.layers.{layer}.{n}");
    let tokens = CANVAS;

    let input_norm_w = graph.weight(&lp("input_layernorm.weight"))?;
    let normed = graph.rms_norm(x, input_norm_w, eps)?;

    // attention_layer, op by op, keeping every internal.
    let (cos, sin) = CudaGraph::rope_tables(head_dim, rope_theta, positions, partial);
    let cos_idx = graph.upload_bits(vec![positions.len(), head_dim], cos)?;
    let sin_idx = graph.upload_bits(vec![positions.len(), head_dim], sin)?;
    aux.extend([cos_idx, sin_idx]);
    let q_w = graph.weight(&p("q_proj.weight".to_owned()))?;
    let q = graph.linear(normed, q_w)?;
    let q_bits = graph.host_bits(q)?;
    let q3 = graph.upload_bits(vec![tokens, heads, head_dim], q_bits)?;
    aux.push(q3);
    let q_norm_w = graph.weight(&p("q_norm.weight".to_owned()))?;
    let qn = graph.rms_norm(q3, q_norm_w, eps)?;
    let qr = graph.rope(qn, cos_idx, sin_idx)?;
    let k_w = graph.weight(&p("k_proj.weight".to_owned()))?;
    let k = graph.linear(normed, k_w)?;
    let k_bits = graph.host_bits(k)?;
    let k3 = graph.upload_bits(vec![tokens, kv_heads, head_dim], k_bits)?;
    let k_norm_w = graph.weight(&p("k_norm.weight".to_owned()))?;
    let kn = graph.rms_norm(k3, k_norm_w, eps)?;
    let kr = graph.rope(kn, cos_idx, sin_idx)?;
    let has_v = graph
        .checkpoint()
        .tensor(&p("v_proj.weight".to_owned()))
        .is_some();
    let v3 = if has_v {
        let v_w = graph.weight(&p("v_proj.weight".to_owned()))?;
        let v = graph.linear(normed, v_w)?;
        let v_bits = graph.host_bits(v)?;
        graph.upload_bits(vec![tokens, kv_heads, head_dim], v_bits)?
    } else {
        k3
    };
    let ones_hd = graph.upload_bits(
        vec![head_dim],
        vec![claymore_tensor::Bf16::from_f32(1.0f32); head_dim],
    )?;
    aux.push(ones_hd);
    // `v_norm` is an unscaled RMSNorm in the pinned code; the contract
    // applies the scale vector, so the ones vector is the scale.
    let vn = graph.rms_norm(v3, ones_hd, eps)?;
    let kr_bits = graph.host_bits(kr)?;
    let vn_bits = graph.host_bits(vn)?;
    // append=false: the reference encoder cache is read-only for the canvas.
    let entry = &cache[layer];
    assert_eq!(entry.2, PREFIX, "reference cache rows (cuda)");
    let kv_len = entry.2 + tokens;
    let mut keys_bits = entry.3.clone();
    keys_bits.extend(kr_bits);
    let mut values_bits = entry.4.clone();
    values_bits.extend(vn_bits);
    let keys_idx = graph.upload_bits(vec![kv_len, kv_heads, head_dim], keys_bits)?;
    let values_idx = graph.upload_bits(vec![kv_len, kv_heads, head_dim], values_bits)?;
    let mask = CudaGraph::build_mask(positions, kv_len, sliding, false);
    assert!(
        mask.is_none(),
        "the pinned bidirectional mask must hide nothing over positions 7..23"
    );
    let attn = graph.attention(qr, keys_idx, values_idx, None, 1.0)?;
    let attn_bits = graph.host_bits(attn)?;
    let flat = graph.upload_bits(vec![tokens, heads * head_dim], attn_bits)?;
    aux.push(flat);
    let o_w = graph.weight(&p("o_proj.weight".to_owned()))?;
    let attn_out = graph.linear(flat, o_w)?;

    let post_w = graph.weight(&lp("post_attention_layernorm.weight"))?;
    let post = graph.rms_norm(attn_out, post_w, eps)?;
    let residual = graph.add(x, post)?;
    let pre1_w = graph.weight(&lp("pre_feedforward_layernorm.weight"))?;
    let pre1 = graph.rms_norm(residual, pre1_w, eps)?;
    let mlp = graph.dense_mlp(
        pre1,
        &lp("mlp.gate_proj.weight"),
        &lp("mlp.up_proj.weight"),
        &lp("mlp.down_proj.weight"),
    )?;
    let pff1_w = graph.weight(&lp("post_feedforward_layernorm_1.weight"))?;
    let pff1 = graph.rms_norm(mlp, pff1_w, eps)?;
    let pre2_w = graph.weight(&lp("pre_feedforward_layernorm_2.weight"))?;
    let pre2 = graph.rms_norm(residual, pre2_w, eps)?;

    // router_layer, op by op, exposing the raw scores.
    let ones_w = graph.upload_bits(
        vec![hidden],
        vec![claymore_tensor::Bf16::from_f32(1.0f32); hidden],
    )?;
    let zero_bias = graph.upload_bits(
        vec![hidden],
        vec![claymore_tensor::Bf16::from_f32(0.0f32); hidden],
    )?;
    aux.extend([ones_w, zero_bias]);
    let normed_router = graph.rms_norm(residual, ones_w, eps)?;
    let scale_w = graph.weight(&lp("router.scale"))?;
    let scaled = graph.scale_bias(normed_router, scale_w, zero_bias)?;
    let root_scale = (hidden as f64).powf(-0.5) as f32;
    let scaled_scalar = graph.scale_scalar(scaled, root_scale)?;
    let proj_w = graph.weight(&lp("router.proj.weight"))?;
    let proj = graph.linear(scaled_scalar, proj_w)?;
    let scores = graph.widen(proj)?;
    let per_expert_w = graph.weight(&lp("router.per_expert_scale"))?;
    let per_expert_bits = graph.host_bits(per_expert_w)?;
    let per_expert = graph.upload_f32(vec![EXPERTS], &CudaGraph::f32_of(&per_expert_bits))?;
    aux.push(per_expert);
    let topk = graph.router_top_k(scores, per_expert, top_k)?;

    // experts_layer verbatim (production composition; its internals are not
    // stage-compared, its output is).
    let exp = graph.experts_layer(
        pre2,
        topk,
        &lp("experts.gate_up_proj"),
        &lp("experts.down_proj"),
        experts,
        inter,
        top_k,
    )?;
    let pff2_w = graph.weight(&lp("post_feedforward_layernorm_2.weight"))?;
    let pff2 = graph.rms_norm(exp, pff2_w, eps)?;
    let comb = graph.add(pff1, pff2)?;
    let pff_w = graph.weight(&lp("post_feedforward_layernorm.weight"))?;
    let pff = graph.rms_norm(comb, pff_w, eps)?;
    let pre_scalar = graph.add(residual, pff)?;
    let scalar_w = graph.weight(&lp("layer_scalar"))?;
    let scalar_bits = graph.host_bits(scalar_w)?;
    let scalar = CudaGraph::f32_of(&scalar_bits)[0];
    let scale_vec = graph.upload_bits(
        vec![hidden],
        vec![claymore_tensor::Bf16::from_f32(scalar); hidden],
    )?;
    let zero_vec = graph.upload_bits(
        vec![hidden],
        vec![claymore_tensor::Bf16::from_f32(0.0f32); hidden],
    )?;
    aux.extend([scale_vec, zero_vec]);
    let out = graph.scale_bias(pre_scalar, scale_vec, zero_vec)?;

    Ok(CudaStages {
        normed,
        q,
        qn,
        qr,
        k,
        kn,
        kr,
        vn,
        keys: keys_idx,
        values: values_idx,
        attn,
        attn_out,
        post,
        residual,
        pre1,
        mlp,
        pff1,
        pre2,
        router_scaled: scaled,
        router_scaled_scalar: scaled_scalar,
        router_proj: proj,
        router_scores: scores,
        topk,
        exp,
        pff2,
        comb,
        pff,
        pre_scalar,
        out,
        aux,
    })
}

fn stage_indices(s: &CudaStages) -> Vec<usize> {
    let mut v = vec![
        s.normed,
        s.q,
        s.qn,
        s.qr,
        s.k,
        s.kn,
        s.kr,
        s.vn,
        s.keys,
        s.values,
        s.attn,
        s.attn_out,
        s.post,
        s.residual,
        s.pre1,
        s.mlp,
        s.pff1,
        s.pre2,
        s.router_scaled,
        s.router_scaled_scalar,
        s.router_proj,
        s.router_scores,
        s.topk,
        s.exp,
        s.pff2,
        s.comb,
        s.pff,
        s.pre_scalar,
        s.out,
    ];
    v.extend(s.aux.iter().copied());
    // The global layers have no `v_proj`: their `v_norm` reads the `k_proj`
    // tensor, so the same arena slot appears twice. Releasing it twice would
    // fail on an already-freed slot.
    let mut seen = std::collections::BTreeSet::new();
    v.retain(|idx| seen.insert(*idx));
    v
}

// ---------------------------------------------------------------------------
// Test 1 — the 30-layer differential.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a CUDA driver, the pinned checkpoint, the layer fixtures, and the release profile"]
fn cuda_decoder_layers_match_cpu_step_by_step() {
    let dir = fixture_dir();
    let root = checkpoint_root();
    if !root.is_dir() {
        eprintln!("skip: checkpoint unavailable — set CLAYMORE_M10_CHECKPOINT");
        return;
    }
    for l in 0..LAYER_COUNT {
        for name in [
            format!("dec_in_{l}"),
            format!("decoder_layer_{l}"),
            format!("enc_kv_{l}_keys"),
            format!("enc_kv_{l}_values"),
            format!("moe{l}_router_scores"),
            format!("moe{l}_router_indices"),
        ] {
            assert!(
                dir.join(format!("{name}.bin")).is_file(),
                "missing fixture {name}"
            );
        }
    }
    // Entry fidelity (same contract as the CPU bisect): the fixture chain is
    // continuous, so each layer is judged on its own, not on a chain state.
    for l in 1..LAYER_COUNT {
        let input = std::fs::read(dir.join(format!("dec_in_{l}.bin"))).expect("fixture");
        let previous =
            std::fs::read(dir.join(format!("decoder_layer_{}.bin", l - 1))).expect("fixture");
        assert_eq!(
            input,
            previous,
            "entry fidelity broken: dec_in_{l} != decoder_layer_{}",
            l - 1
        );
    }

    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(&root).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let text = cfg.text_config.as_ref().unwrap();
    let heads = text.num_attention_heads.unwrap();
    let eps = text.rms_norm_eps.unwrap_or(1e-6) as f32;
    let inter = text.moe_intermediate_size.unwrap_or(704);
    let experts = text.num_experts.unwrap_or(128);
    let top_k = text.top_k_experts.unwrap();
    assert_eq!(top_k, TOP_K);
    assert_eq!(experts, EXPERTS);
    assert_eq!(text.hidden_size.unwrap(), HIDDEN);

    let manager = CudaMemoryManager::new(0, budgets(), EvictionPolicy::Lru)
        .expect("CUDA memory manager within the M10 budgets");
    let memory = CudaTensorMemory::from_manager(manager).expect("tensor memory owner");
    let cuda_ckpt = Checkpoint::open_pinned(&root).expect("pinned checkpoint (cuda)");
    let mut graph = CudaGraph::new(memory, cuda_ckpt);

    eprintln!("couche|etape|max_abs|rms|pire_ligne_pas");
    let mut worst_overall = (0.0f32, 0usize, String::new());
    let mut worst_score_drift = (0.0f32, 0usize);
    let mut drift_failures: Vec<String> = Vec::new();
    let mut score_drift_failures: Vec<String> = Vec::new();
    let mut vram_failures: Vec<String> = Vec::new();
    let mut layer_outputs_pass1: Vec<Vec<Bf16>> = Vec::new();
    let mut router_faithful = 0usize;
    let mut router_swaps = 0usize;
    let mut router_structural: Vec<String> = Vec::new();
    let mut cuda_eq_cpu_sets = 0usize;
    let mut rows = 0usize;

    // Pass 1: the full differential table.
    for layer in 0..LAYER_COUNT {
        let (_global, head_dim, kv_heads, rope, partial) = per_layer_params(text, layer);
        let positions: Vec<usize> = (PREFIX..PREFIX + CANVAS).collect();
        // CPU production decoder semantics: fully bidirectional, no window.
        let sliding_cpu: Option<usize> = None;
        // CUDA production decoder call as written in `CudaGraph::decoder_forward`.
        let sliding_cuda: Option<usize> = if _global {
            None
        } else {
            Some(text.sliding_window.unwrap_or(1024))
        };
        // Over positions 7..23 (kv rows 0..22) neither interpretation masks
        // anything: the two sliding semantics coincide for this scenario
        // (recorded explicitly; they differ only beyond 1024 rows).
        assert!(CudaGraph::build_mask(&positions, PREFIX + CANVAS, sliding_cuda, false).is_none());
        assert!(CudaGraph::build_mask(&positions, PREFIX + CANVAS, sliding_cpu, false).is_none());

        let start = load_f32(&dir, &format!("dec_in_{layer}"), &[1, CANVAS, HIDDEN]);
        let start_bits = to_bits(&start);
        let reference_keys = load_f32(
            &dir,
            &format!("enc_kv_{layer}_keys"),
            &[1, kv_heads, PREFIX, head_dim],
        );
        let reference_values = load_f32(
            &dir,
            &format!("enc_kv_{layer}_values"),
            &[1, kv_heads, PREFIX, head_dim],
        );

        // ----- CPU: production shared layer body -----
        let mut cache = KvCache::new(LAYER_COUNT);
        let zero_rows = host.bf16(
            vec![PREFIX, kv_heads, head_dim],
            &vec![0.0f32; PREFIX * kv_heads * head_dim],
        );
        for slot in 0..layer {
            cache.append(
                &host,
                slot,
                clone_tensor(&host, &zero_rows),
                clone_tensor(&host, &zero_rows),
            );
        }
        cache.append(
            &host,
            layer,
            host.bf16(
                vec![PREFIX, kv_heads, head_dim],
                &permute_kv_to_local(&reference_keys, kv_heads, PREFIX, head_dim),
            ),
            host.bf16(
                vec![PREFIX, kv_heads, head_dim],
                &permute_kv_to_local(&reference_values, kv_heads, PREFIX, head_dim),
            ),
        );
        let start_t = host.bf16(vec![CANVAS, HIDDEN], &start);
        let mut trace = Vec::new();
        let mut layer_trace = Vec::new();
        let mut stages = Vec::new();
        let _output = decoder_forward_traced_from_hidden(
            &host,
            &ckpt,
            &cfg,
            &start_t,
            &mut cache,
            &positions,
            layer,
            &mut trace,
            &mut layer_trace,
            &mut stages,
        );
        assert_eq!(stages.len(), 11, "layer {layer} stage count");
        let cpu_layer_out = host.to_f32(&layer_trace[0]);

        // CPU attention replica + join against the production stage.
        let normed_t = rms_norm(
            &host,
            &start_t,
            &checkpoint_tensor(
                &host,
                &ckpt,
                &format!("model.decoder.layers.{layer}.input_layernorm.weight"),
            ),
            eps as f64,
        );
        let cpu_attn = cpu_attention_stages(
            &host, &ckpt, layer, &normed_t, &cache, &positions, head_dim, heads, kv_heads, rope,
            partial, eps as f64,
        );
        // Join: replica attention output == production stage[0], bit for bit
        // as BF16 — the replica is the same math as the production body.
        let replica_out = to_bits(&cpu_attn.attn_out);
        assert_eq!(
            replica_out,
            host.to_bits(&stages[0]),
            "CPU replica attention output diverges from the production stage (layer {layer})"
        );

        // CPU router on the production residual stage, exactly the tensors
        // the production body feeds it.
        let residual_cpu = &stages[2];
        let (_cpu_probs, cpu_raw, cpu_weights, cpu_indices) = graph::router_forward(
            &host,
            residual_cpu,
            &checkpoint_tensor(
                &host,
                &ckpt,
                &format!("model.decoder.layers.{layer}.router.proj.weight"),
            ),
            &checkpoint_tensor(
                &host,
                &ckpt,
                &format!("model.decoder.layers.{layer}.router.scale"),
            ),
            &checkpoint_tensor(
                &host,
                &ckpt,
                &format!("model.decoder.layers.{layer}.router.per_expert_scale"),
            ),
            eps as f64,
            (HIDDEN as f64).powf(-0.5),
            TOP_K,
        );
        // Join: the router helper on the same tensors reproduces the
        // production trace weights.
        assert!(
            trace[0].1.iter().zip(&cpu_weights).all(|(a, b)| a == b),
            "CPU router replica diverges from the production trace (layer {layer})"
        );

        // ----- CUDA -----
        let mut cuda_cache: Vec<KvLayerEntry> = Vec::new();
        for slot in 0..layer {
            CudaGraph::append_cache(
                &mut cuda_cache,
                slot,
                kv_heads,
                head_dim,
                vec![Bf16::from_f32(0.0f32); PREFIX * kv_heads * head_dim],
                vec![Bf16::from_f32(0.0f32); PREFIX * kv_heads * head_dim],
            );
        }
        CudaGraph::append_cache(
            &mut cuda_cache,
            layer,
            kv_heads,
            head_dim,
            to_bits(&permute_kv_to_local(
                &reference_keys,
                kv_heads,
                PREFIX,
                head_dim,
            )),
            to_bits(&permute_kv_to_local(
                &reference_values,
                kv_heads,
                PREFIX,
                head_dim,
            )),
        );
        let x_idx = graph
            .upload_bits(vec![CANVAS, HIDDEN], start_bits.clone())
            .expect("upload dec_in");
        let s = cuda_layer_stages(
            &mut graph,
            layer,
            x_idx,
            &cuda_cache,
            &positions,
            head_dim,
            heads,
            kv_heads,
            rope,
            partial,
            sliding_cuda,
            eps,
            HIDDEN,
            inter,
            experts,
            TOP_K,
        )
        .expect("cuda layer stages");

        // Join: the official production text_layer (internals released by
        // its scope) returns the replica output bit for bit — no premature
        // release and no composition drift in the official path.
        let official = graph
            .text_layer(
                "decoder",
                layer,
                &format!("model.decoder.layers.{layer}"),
                x_idx,
                &mut cuda_cache,
                &positions,
                head_dim,
                heads,
                kv_heads,
                rope,
                partial,
                sliding_cuda,
                false,
                eps,
                false,
                HIDDEN,
                inter,
                experts,
                TOP_K,
            )
            .expect("official text_layer");
        let official_bits = graph.download_bf16(official).expect("download official");
        let replica_bits = graph.download_bf16(s.out).expect("download replica out");
        assert_eq!(
            official_bits,
            replica_bits,
            "official text_layer diverges from the replica composition (layer {layer}) — premature release or composition drift"
        );

        // ----- stage-by-stage comparison -----
        let kv_w = kv_heads * head_dim;
        // The q width is `heads * head_dim` (4096 local / 8192 global), not the
        // hidden size: the attention output is flattened to that width before
        // `o_proj` maps it back to `hidden`.
        let q_w = heads * head_dim;
        let kv_len = PREFIX + CANVAS;
        let mut layer_worst_steps = 0.0f32;
        let mut layer_worst_label = String::new();
        {
            let mut one = |label: &str, idx: usize, rows: usize, width: usize, cpu: &[f32]| -> () {
                let cuda = CudaGraph::f32_of(&graph.download_bf16(idx).expect("download stage"));
                let (max_abs, rms) = stats(&cuda, cpu);
                let steps = worst_row_steps(&cuda, cpu, rows, width);
                eprintln!("layer {layer:>2} | {label:<16} | {max_abs:.6} | {rms:.6} | {steps:.1}");
                if steps > layer_worst_steps {
                    layer_worst_steps = steps;
                    layer_worst_label = label.to_owned();
                }
            };
            one(
                "input_norm",
                s.normed,
                CANVAS,
                HIDDEN,
                &host.to_f32(&normed_t),
            );
            one("q_proj", s.q, CANVAS, q_w, &cpu_attn.q);
            one("q_norm", s.qn, CANVAS, q_w, &cpu_attn.qn);
            one("rope_q", s.qr, CANVAS, q_w, &cpu_attn.qr);
            one("k_proj", s.k, CANVAS, kv_w, &cpu_attn.k);
            one("k_norm", s.kn, CANVAS, kv_w, &cpu_attn.kn);
            one("rope_k", s.kr, CANVAS, kv_w, &cpu_attn.kr);
            one("v_norm", s.vn, CANVAS, kv_w, &cpu_attn.vn);
            one("cache_keys", s.keys, kv_len, kv_w, &cpu_attn.keys);
            one("cache_values", s.values, kv_len, kv_w, &cpu_attn.values);
            one("attention", s.attn, CANVAS, q_w, &cpu_attn.attn);
            one("o_proj", s.attn_out, CANVAS, HIDDEN, &cpu_attn.attn_out);
            one(
                "post_attn_norm",
                s.post,
                CANVAS,
                HIDDEN,
                &host.to_f32(&stages[1]),
            );
            one(
                "residual",
                s.residual,
                CANVAS,
                HIDDEN,
                &host.to_f32(&stages[2]),
            );
            let pre1_cpu = host.to_f32(&rms_norm(
                &host,
                &stages[2],
                &checkpoint_tensor(
                    &host,
                    &ckpt,
                    &format!("model.decoder.layers.{layer}.pre_feedforward_layernorm.weight"),
                ),
                eps as f64,
            ));
            one("pre_ff1_norm", s.pre1, CANVAS, HIDDEN, &pre1_cpu);
            one("mlp", s.mlp, CANVAS, HIDDEN, &host.to_f32(&stages[3]));
            one(
                "pff_norm_1",
                s.pff1,
                CANVAS,
                HIDDEN,
                &host.to_f32(&stages[4]),
            );
            one(
                "pre_ff2_norm",
                s.pre2,
                CANVAS,
                HIDDEN,
                &host.to_f32(&stages[5]),
            );
            let router_scaled_cpu = host.to_f32(&{
                let normed = rms_norm_unscaled(&host, residual_cpu, eps as f64);
                let sv = host.to_f32(&checkpoint_tensor(
                    &host,
                    &ckpt,
                    &format!("model.decoder.layers.{layer}.router.scale"),
                ));
                let vals: Vec<f32> = host
                    .to_f32(&normed)
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * sv[i % HIDDEN])
                    .collect();
                host.bf16(vec![CANVAS, HIDDEN], &vals)
            });
            one(
                "router_scale",
                s.router_scaled,
                CANVAS,
                HIDDEN,
                &router_scaled_cpu,
            );
            let router_scaled_scalar_cpu = host.to_f32(&{
                let normed = rms_norm_unscaled(&host, residual_cpu, eps as f64);
                let sv = host.to_f32(&checkpoint_tensor(
                    &host,
                    &ckpt,
                    &format!("model.decoder.layers.{layer}.router.scale"),
                ));
                let vals: Vec<f32> = host
                    .to_f32(&normed)
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * sv[i % HIDDEN])
                    .collect();
                let by_expert = host.bf16(vec![CANVAS, HIDDEN], &vals);
                let root_scale = (HIDDEN as f64).powf(-0.5) as f32;
                let vals: Vec<f32> = host
                    .to_f32(&by_expert)
                    .iter()
                    .map(|v| v * root_scale)
                    .collect();
                host.bf16(vec![CANVAS, HIDDEN], &vals)
            });
            one(
                "router_scaled",
                s.router_scaled_scalar,
                CANVAS,
                HIDDEN,
                &router_scaled_scalar_cpu,
            );
            let router_proj_cpu = host.to_f32(&{
                let scaled_normed = rms_norm_unscaled(&host, residual_cpu, eps as f64);
                let sv = host.to_f32(&checkpoint_tensor(
                    &host,
                    &ckpt,
                    &format!("model.decoder.layers.{layer}.router.scale"),
                ));
                let vals: Vec<f32> = host
                    .to_f32(&scaled_normed)
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * sv[i % HIDDEN])
                    .collect();
                let by_expert = host.bf16(vec![CANVAS, HIDDEN], &vals);
                let root_scale = (HIDDEN as f64).powf(-0.5) as f32;
                let vals: Vec<f32> = host
                    .to_f32(&by_expert)
                    .iter()
                    .map(|v| v * root_scale)
                    .collect();
                let scaled = host.bf16(vec![CANVAS, HIDDEN], &vals);
                let proj = checkpoint_tensor(
                    &host,
                    &ckpt,
                    &format!("model.decoder.layers.{layer}.router.proj.weight"),
                );
                linear(&host, &scaled, &proj)
            });
            one(
                "router_proj",
                s.router_proj,
                CANVAS,
                EXPERTS,
                &router_proj_cpu,
            );
            one("experts", s.exp, CANVAS, HIDDEN, &host.to_f32(&stages[6]));
            one(
                "pff_norm_2",
                s.pff2,
                CANVAS,
                HIDDEN,
                &host.to_f32(&stages[7]),
            );
            one(
                "moe_dense_sum",
                s.comb,
                CANVAS,
                HIDDEN,
                &host.to_f32(&stages[8]),
            );
            one("pff_norm", s.pff, CANVAS, HIDDEN, &host.to_f32(&stages[9]));
            let pre_scalar_cpu = host.to_f32(&add(&host, &stages[2], &stages[9]));
            one("pre_scalar", s.pre_scalar, CANVAS, HIDDEN, &pre_scalar_cpu);
            one("layer_output", s.out, CANVAS, HIDDEN, &cpu_layer_out);
        }

        // Raw widened scores: F32 comparison with the tie bound. The drift is
        // collected, not asserted per layer: the table must be complete even
        // when a bound is exceeded (the bound is checked once, at the end).
        {
            let cuda_scores = graph
                .download_f32(s.router_scores)
                .expect("download scores");
            let (max_abs, rms) = stats(&cuda_scores, &cpu_raw);
            eprintln!(
                "layer {layer:>2} | router_scores    | {max_abs:.6} | {rms:.6} | (f32, bound {MAX_SCORE_DRIFT})"
            );
            if max_abs > MAX_SCORE_DRIFT {
                score_drift_failures.push(format!(
                    "couche {layer}: scores routeur CUDA↔CPU {max_abs} > {MAX_SCORE_DRIFT}"
                ));
            }
            if max_abs > worst_score_drift.0 {
                worst_score_drift = (max_abs, layer);
            }
        }

        // Router weights (F32) CUDA vs CPU + discrete expert sets vs the
        // reference (same policy as the CPU bisect contract).
        let topk_values = graph.download_f32(s.topk).expect("download topk");
        let cuda_weights: Vec<f32> = topk_values
            .chunks_exact(2 * TOP_K)
            .flat_map(|row| row[..TOP_K].to_vec())
            .collect();
        let (max_abs, rms) = stats(&cuda_weights, &cpu_weights);
        eprintln!(
            "layer {layer:>2} | router_weights   | {max_abs:.6} | {rms:.6} | (f32, bound {MAX_SCORE_DRIFT})"
        );
        if max_abs > MAX_SCORE_DRIFT {
            score_drift_failures.push(format!(
                "couche {layer}: poids routeur CUDA↔CPU {max_abs} > {MAX_SCORE_DRIFT}"
            ));
        }
        let cuda_indices: Vec<u32> = topk_values
            .chunks_exact(2 * TOP_K)
            .flat_map(|row| row[TOP_K..].iter().map(|v| *v as u32))
            .collect();
        let cpu_indices32: Vec<u32> = cpu_indices.iter().map(|v| *v as u32).collect();
        let golden_scores = load_f32(
            &dir,
            &format!("moe{layer}_router_scores"),
            &[CANVAS, EXPERTS],
        );
        let golden_indices: Vec<u32> = load_f32(
            &dir,
            &format!("moe{layer}_router_indices"),
            &[CANVAS, TOP_K],
        )
        .iter()
        .map(|v| *v as u32)
        .collect();
        for row in 0..CANVAS {
            rows += 1;
            let cuda_set = top_k_set(&cuda_indices, row);
            if cuda_set == top_k_set(&cpu_indices32, row) {
                cuda_eq_cpu_sets += 1;
            }
            let golden_set = top_k_set(&golden_indices, row);
            if cuda_set == golden_set {
                router_faithful += 1;
                continue;
            }
            let extra: Vec<u32> = cuda_set
                .iter()
                .copied()
                .filter(|e| !golden_set.contains(e))
                .collect();
            let missing: Vec<u32> = golden_set
                .iter()
                .copied()
                .filter(|e| !cuda_set.contains(e))
                .collect();
            let gap = (golden_scores[row * EXPERTS + extra[0] as usize]
                - golden_scores[row * EXPERTS + missing[0] as usize])
                .abs();
            if extra.len() == 1 && missing.len() == 1 && gap <= MAX_SCORE_DRIFT {
                router_swaps += 1;
            } else {
                router_structural.push(format!(
                    "couche {layer} ligne {row}: cuda+{extra:?} référence+{missing:?} écart={gap:.6}"
                ));
            }
        }

        // Golden tracking of the CUDA layer output (D2: tracked, not gated).
        {
            let cuda_out = CudaGraph::f32_of(&replica_bits);
            let golden_layer = load_f32(
                &dir,
                &format!("decoder_layer_{layer}"),
                &[1, CANVAS, HIDDEN],
            );
            let (gmax, grms) = stats(&cuda_out, &golden_layer);
            let gsteps = worst_row_steps(&cuda_out, &golden_layer, CANVAS, HIDDEN);
            eprintln!(
                "layer {layer:>2} | SUIVI vs dorée   | {gmax:.6} | {grms:.6} | {gsteps:.1} (cuda-vs-golden, D2)"
            );
        }

        if layer_worst_steps > worst_overall.0 {
            worst_overall = (layer_worst_steps, layer, layer_worst_label.clone());
        }
        if layer_worst_steps > MAX_ROW_DRIFT_STEPS {
            drift_failures.push(format!(
                "couche {layer} étape {}: {layer_worst_steps:.1} pas BF16 > {MAX_ROW_DRIFT_STEPS}",
                layer_worst_label
            ));
        }

        layer_outputs_pass1.push(replica_bits.clone());

        // The layer is fully compared: release every arena tensor it created
        // (the input last). Everything after this point is host memory.
        for idx in stage_indices(&s) {
            graph.release(idx).expect("release stage after comparison");
        }
        graph.release(official).expect("release official");
        graph.release(x_idx).expect("release layer input");
        let (peak, uploaded, downloaded, launches) = graph.metrics();
        eprintln!(
            "layer {layer:>2} metrics: peak={} MiB uploaded={} MiB downloaded={} MiB launches={launches}",
            peak / (1 << 20),
            uploaded / (1 << 20),
            downloaded / (1 << 20)
        );
        if peak > 28 * (1 << 30) {
            vram_failures.push(format!("couche {layer}: pic VRAM {peak} > 28 GiB"));
        }
    }

    // Pass 2 (reproducibility): re-run every CUDA layer; outputs must come
    // back bit-identical — no cross-layer arena/LRU contamination, no
    // accumulation-order instability.
    for (layer, expected_bits) in layer_outputs_pass1.iter().enumerate() {
        let (global, head_dim, kv_heads, rope, partial) = per_layer_params(text, layer);
        let sliding_cuda: Option<usize> = if global {
            None
        } else {
            Some(text.sliding_window.unwrap_or(1024))
        };
        let reference_keys = load_f32(
            &dir,
            &format!("enc_kv_{layer}_keys"),
            &[1, kv_heads, PREFIX, head_dim],
        );
        let reference_values = load_f32(
            &dir,
            &format!("enc_kv_{layer}_values"),
            &[1, kv_heads, PREFIX, head_dim],
        );
        let mut cuda_cache: Vec<KvLayerEntry> = Vec::new();
        for slot in 0..layer {
            CudaGraph::append_cache(
                &mut cuda_cache,
                slot,
                kv_heads,
                head_dim,
                vec![Bf16::from_f32(0.0f32); PREFIX * kv_heads * head_dim],
                vec![Bf16::from_f32(0.0f32); PREFIX * kv_heads * head_dim],
            );
        }
        CudaGraph::append_cache(
            &mut cuda_cache,
            layer,
            kv_heads,
            head_dim,
            to_bits(&permute_kv_to_local(
                &reference_keys,
                kv_heads,
                PREFIX,
                head_dim,
            )),
            to_bits(&permute_kv_to_local(
                &reference_values,
                kv_heads,
                PREFIX,
                head_dim,
            )),
        );
        let start = load_f32(&dir, &format!("dec_in_{layer}"), &[1, CANVAS, HIDDEN]);
        let positions: Vec<usize> = (PREFIX..PREFIX + CANVAS).collect();
        let x_idx = graph
            .upload_bits(vec![CANVAS, HIDDEN], to_bits(&start))
            .expect("upload");
        let s = cuda_layer_stages(
            &mut graph,
            layer,
            x_idx,
            &cuda_cache,
            &positions,
            head_dim,
            heads,
            kv_heads,
            rope,
            partial,
            sliding_cuda,
            eps,
            HIDDEN,
            inter,
            experts,
            TOP_K,
        )
        .expect("cuda layer stages pass 2");
        let pass2 = graph.download_bf16(s.out).expect("download pass 2");
        assert_eq!(
            pass2, *expected_bits,
            "layer {layer} CUDA output is not reproducible bit for bit between the two passes"
        );
        for idx in stage_indices(&s) {
            graph.release(idx).expect("release pass 2");
        }
        graph.release(x_idx).expect("release pass 2 input");
    }

    eprintln!(
        "router: ensembles fideles={router_faithful}/{rows} echanges de frontiere={router_swaps} structurels={}",
        router_structural.len()
    );
    eprintln!("router: ensembles CUDA==CPU: {cuda_eq_cpu_sets}/{rows} (rapporté)");
    for line in &router_structural {
        eprintln!("STRUCTUREL {line}");
    }
    assert!(
        router_structural.is_empty(),
        "{} CUDA router rows diverged from the reference top-{TOP_K} set beyond boundary ties",
        router_structural.len()
    );
    let (peak, uploaded, downloaded, launches) = graph.metrics();
    eprintln!(
        "DECODER_DIFF_DONE pire_ligne={:.1} pas (couche {} etape {}) pire_scores={:.3e} (couche {}) peak={} MiB uploaded={} MiB downloaded={} MiB launches={}",
        worst_overall.0,
        worst_overall.1,
        worst_overall.2,
        worst_score_drift.0,
        worst_score_drift.1,
        peak / (1 << 20),
        uploaded / (1 << 20),
        downloaded / (1 << 20),
        launches
    );
    if peak > 28 * (1 << 30) {
        vram_failures.push(format!("pic VRAM final {peak} > 28 GiB"));
    }
    // The bounds are checked once, on the collected violations, so the table
    // above is complete even when a bound is exceeded.
    assert!(
        drift_failures.is_empty() && score_drift_failures.is_empty() && vram_failures.is_empty(),
        "porte(s) dépassée(s):\n  {}\n  {}\n  {}",
        drift_failures.join("\n  "),
        score_drift_failures.join("\n  "),
        vram_failures.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// Test 2 — the isolated tied lm head (logits).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a CUDA driver, the pinned checkpoint, golden files, and the release profile"]
fn cuda_lm_logits_match_cpu_on_the_reference_hidden() {
    let root = checkpoint_root();
    let hidden_path = Path::new(GOLDEN).join("decoder_last_hidden0_direct.bin");
    let logits_path = Path::new(GOLDEN).join("decoder_logits0_direct.bin");
    if !root.is_dir() || !hidden_path.is_file() || !logits_path.is_file() {
        eprintln!("skip: checkpoint or golden files unavailable");
        return;
    }
    const VOCAB: usize = 262_144;

    let golden_hidden = load_f32(
        Path::new(GOLDEN),
        "decoder_last_hidden0_direct",
        &[1, CANVAS, HIDDEN],
    );
    let golden_logits = load_f32(
        Path::new(GOLDEN),
        "decoder_logits0_direct",
        &[1, CANVAS, VOCAB],
    );
    // The eager golden hidden is an F32 dump: it is not exactly BF16
    // representable. Both heads therefore read the **same** BF16
    // re-quantization of it (the quantization error is reported, never
    // gated).
    let bits = to_bits(&golden_hidden);
    let quant_err = bits
        .iter()
        .zip(&golden_hidden)
        .map(|(b, g)| (b.to_f32() - *g).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "logits entrée: golden hidden re-quantifié en BF16, erreur max_abs={quant_err:.6} (même entrée des deux têtes)"
    );

    let host = Host::new();
    let cpu_ckpt = Checkpoint::open_pinned(&root).expect("pinned checkpoint (cpu)");
    let cfg = cpu_ckpt.config().clone();
    let hidden_t = host.bf16(vec![CANVAS, HIDDEN], &golden_hidden);
    let cpu_logits = host.to_f32(&lm_logits(&host, &cpu_ckpt, &cfg, &hidden_t));

    let manager = CudaMemoryManager::new(0, budgets(), EvictionPolicy::Lru)
        .expect("CUDA memory manager within the M10 budgets");
    let memory = CudaTensorMemory::from_manager(manager).expect("tensor memory owner");
    let cuda_ckpt = Checkpoint::open_pinned(&root).expect("pinned checkpoint (cuda)");
    let mut graph = CudaGraph::new(memory, cuda_ckpt);
    let x_idx = graph
        .upload_bits(vec![CANVAS, HIDDEN], bits.clone())
        .expect("upload hidden");
    let logits_idx = graph.lm_logits(x_idx).expect("cuda lm head");
    let cuda_logits = graph.download_f32(logits_idx).expect("download logits");

    let (max_abs, rms) = stats(&cuda_logits, &cpu_logits);
    let steps = worst_row_steps(&cuda_logits, &cpu_logits, CANVAS, VOCAB);
    eprintln!(
        "logits CUDA vs CPU (input identique): max_abs={max_abs:.6} rms={rms:.6} pire_ligne={steps:.1} pas (relatif au pic de ligne CPU)"
    );
    let (gmax, grms) = stats(&cuda_logits, &golden_logits);
    let gsteps = worst_row_steps(&cuda_logits, &golden_logits, CANVAS, VOCAB);
    let (cgmax, cgrms) = stats(&cpu_logits, &golden_logits);
    let cgsteps = worst_row_steps(&cpu_logits, &golden_logits, CANVAS, VOCAB);
    eprintln!(
        "logits CUDA vs dorée (SUIVI): max_abs={gmax:.6} rms={grms:.6} pire_ligne={gsteps:.1} pas"
    );
    eprintln!("logits CPU vs dorée (SUIVI): max_abs={cgmax:.6} rms={cgrms:.6} pire_ligne={cgsteps:.1} pas");

    // Ranking divergence (D2: reported). argmax and top-8 set per row.
    let mut argmax_same = 0usize;
    let mut top8_same = 0usize;
    for row in 0..CANVAS {
        let lo = row * VOCAB;
        let hi = lo + VOCAB;
        let amax = |v: &[f32]| {
            v[lo..hi]
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, x)| {
                    if *x > acc.1 {
                        (lo + i, *x)
                    } else {
                        acc
                    }
                })
                .0
        };
        if amax(&cuda_logits) == amax(&cpu_logits) {
            argmax_same += 1;
        }
        let top8 = |v: &[f32]| {
            let mut idx: Vec<usize> = (0..8).collect();
            let mut val: Vec<f32> = (0..8).map(|i| v[lo + i]).collect();
            for i in 8..VOCAB {
                let mut worst = 0usize;
                for slot in 1..8 {
                    if val[slot] < val[worst] {
                        worst = slot;
                    }
                }
                if v[lo + i] > val[worst] {
                    val[worst] = v[lo + i];
                    idx[worst] = i;
                }
            }
            idx.sort_unstable();
            idx
        };
        if top8(&cuda_logits) == top8(&cpu_logits) {
            top8_same += 1;
        }
    }
    eprintln!(
        "logits classement: argmax identiques {argmax_same}/{CANVAS} lignes, top-8 identiques {top8_same}/{CANVAS} lignes (rapportés, D2)"
    );

    assert!(
        cuda_logits.iter().all(|v| v.is_finite()),
        "non-finite CUDA logits"
    );
    assert!(
        cpu_logits.iter().all(|v| v.is_finite()),
        "non-finite CPU logits"
    );
    assert!(
        max_abs <= LOGITS_MAX_ABS,
        "logits CUDA↔CPU drift {max_abs} above {LOGITS_MAX_ABS}"
    );
    assert!(
        rms <= LOGITS_RMS,
        "logits CUDA↔CPU rms {rms} above {LOGITS_RMS}"
    );
    let (peak, uploaded, downloaded, launches) = graph.metrics();
    eprintln!(
        "LOGITS_DIFF_DONE peak={} MiB uploaded={} MiB downloaded={} MiB launches={}",
        peak / (1 << 20),
        uploaded / (1 << 20),
        downloaded / (1 << 20),
        launches
    );
    assert!(
        peak <= 28 * (1 << 30),
        "peak VRAM {peak} exceeds the hard budget"
    );
}
