//! CPU per-layer bisect of the decoder, fed with the reference's own layer
//! entry state and the reference encoder KV cache.
//!
//! `tests/data/m10_layers/dec_in_{L}.bin` is the BF16 `[16, 2816]` state the
//! pinned upstream reference handed to decoder layer `L`, and
//! `enc_kv_{L}_{keys,values}.bin` are the reference encoder-prefix caches in
//! the upstream `[batch, kv_heads, seq, head_dim]` layout. A chain-level
//! differential cannot tell a wrong layer apart from a layer amplifying its
//! predecessors' drift; here each layer runs with bit-exact reference inputs
//! and bit-exact reference caches — never an empty cache, never a cache
//! derived from a Claymore chain.
//!
//! The reference scenario is denoising step 0 with a 16-token canvas and a
//! 7-token encoder prefix; the manifest records that the cache rows stay 7
//! through the whole decoder pass (the upstream `append_to_cache` semantics
//! never mutate the cache object), so no decoder-prefix K/V fixture exists or
//! is needed. The cache layout, positions and prefix length are asserted from
//! the fixture manifest before any comparison.
//!
//! Entry fidelity: for every `L > 0`, `dec_in_{L}` must equal
//! `decoder_layer_{L-1}` bit for bit before any Rust comparison (enforced).
//! Conformity follows decision D1. The router requires the reference top-8
//! expert set per row, up to a single **boundary swap** — the reference's own
//! 8-vs-9 selection is at tie precision (measured gaps: mostly 0.0, max
//! ~5e-4), so an exact set equality is only reproducible on a bit-identical
//! reduction; a swap across a real score gap, or two swaps, is structural.
//! The numeric drift bound is fixed once from the full 30-layer measurement
//! table.
//!
//! Requires the pinned checkpoint (env `CLAYMORE_M10_CHECKPOINT`). Run:
//! ```text
//! CLAYMORE_M10_CHECKPOINT=/path/to/checkpoint \
//!   cargo test --release -p claymore-model-gemma --test decoder_layer_bisect \
//!   --locked -- --ignored --test-threads=1 --nocapture
//! ```

use std::path::Path;

use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::fullgraph::decoder_forward_traced_from_hidden;
use claymore_model_gemma::graph::{Host, KvCache};

const CANVAS: usize = 16;
const HIDDEN: usize = 2816;
const LAYER_COUNT: usize = 30;
const PREFIX: usize = 7;
const TOP_K: usize = 8;

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

fn load(dir: &Path, name: &str, shape: &[usize]) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(format!("{name}.bin"))).expect("fixture file");
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
    let mut max_abs = 0.0f32;
    let mut sq = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sq += d * d;
    }
    (max_abs, (sq / a.len() as f32).sqrt())
}

/// Width of one BF16 step at magnitude `x`: 8 significand bits.
fn bf16_step(x: f32) -> f32 {
    let x = x.abs().max(f32::MIN_POSITIVE);
    2f32.powi(((x.to_bits() >> 23) as i32 - 127) - 7)
}

/// Permute the reference cache `[batch, kv_heads, seq, head_dim]` (batch 1)
/// into the local `KvCache` layout `[seq, kv_heads, head_dim]`.
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

/// `(layer, row, experts selected only by Rust, experts selected only by the
/// reference, reference score gap between the two boundary experts)`.
type BoundaryReport = (usize, usize, Vec<u32>, Vec<u32>, f32);

fn top_k_set(indices: &[u32], row: usize) -> Vec<u32> {
    let mut s: Vec<u32> = indices[row * TOP_K..(row + 1) * TOP_K].to_vec();
    s.sort_unstable();
    s
}

#[test]
#[ignore = "requires the pinned checkpoint (release profile)"]
fn decoder_layers_reproduce_the_reference_when_fed_reference_inputs_and_caches() {
    let dir = fixture_dir();
    let root = checkpoint_root();
    if !root.is_dir() {
        eprintln!(
            "skip: checkpoint unavailable — set CLAYMORE_M10_CHECKPOINT \
             (see docs/m10-diffusiongemma.md for the pinned revision)"
        );
        return;
    }
    for l in 0..LAYER_COUNT {
        for name in [
            format!("dec_in_{l}"),
            format!("decoder_layer_{l}"),
            format!("enc_kv_{l}_keys"),
            format!("enc_kv_{l}_values"),
            format!("moe{l}_router_indices"),
        ] {
            assert!(
                dir.join(format!("{name}.bin")).is_file(),
                "missing fixture {name}"
            );
        }
    }

    // Entry fidelity: the fixture chain must be continuous before any Rust
    // comparison is meaningful — for every L > 0, dec_in_{L} must equal
    // decoder_layer_{L-1} bit for bit.
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

    // The reference encoder-prefix caches must exist for every layer with
    // exactly 7 rows (the pinned scenario has no decoder-prefix K/V).
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest");
    let manifest: serde_json::Value = manifest.parse().expect("manifest parses");
    assert_eq!(
        manifest["artifacts"]["enc_kv_0_keys"]["layout"]
            .as_str()
            .unwrap(),
        "[batch, kv_heads, seq, head_dim] post-rope, reference layout"
    );
    let mut cache_dims = Vec::new();
    for l in 0..LAYER_COUNT {
        let meta = &manifest["artifacts"][&format!("enc_kv_{l}_keys")];
        let shape = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        assert_eq!(shape[0], 1, "enc_kv_{l}_keys batch");
        assert_eq!(
            shape[2], PREFIX,
            "enc_kv_{l}_keys must hold the 7-row encoder prefix"
        );
        cache_dims.push((shape[1], shape[3]));
    }

    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(&root).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();

    let mut rows = 0usize;
    let mut faithful_rows = 0usize;
    let mut swap_rows = 0usize;
    let mut structural: Vec<BoundaryReport> = Vec::new();
    let mut worst_row_layer = 0usize;
    let mut worst_row_steps = 0.0f32;

    eprintln!("couche  sortie_max  sortie_rms  scores_max  scores_rms  fideles  frontieres  pire_ligne(pas)");
    for (layer, &(kv_heads, head_dim)) in cache_dims.iter().enumerate() {
        // Reference cache for this layer, permuted into the local KvCache
        // layout. An empty or Claymore-derived cache would not be a reference
        // input; these come straight from the pinned upstream run.
        // `KvCache::append` appends in order when growing, so the slots below
        // `layer` get same-shaped placeholders first — attention only ever
        // reads the slot of its own layer, these are never used.
        let mut cache = KvCache::new(LAYER_COUNT);
        for slot in 0..layer {
            cache.append(
                &host,
                slot,
                host.bf16(
                    vec![PREFIX, kv_heads, head_dim],
                    &vec![0.0f32; PREFIX * kv_heads * head_dim],
                ),
                host.bf16(
                    vec![PREFIX, kv_heads, head_dim],
                    &vec![0.0f32; PREFIX * kv_heads * head_dim],
                ),
            );
        }
        let reference_keys = load(
            &dir,
            &format!("enc_kv_{layer}_keys"),
            &[1, kv_heads, PREFIX, head_dim],
        );
        let reference_values = load(
            &dir,
            &format!("enc_kv_{layer}_values"),
            &[1, kv_heads, PREFIX, head_dim],
        );
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
        assert_eq!(cache.seq_len(), PREFIX, "reference cache rows");

        let start = load(&dir, &format!("dec_in_{layer}"), &[1, CANVAS, HIDDEN]);
        let start_t = host.bf16(vec![CANVAS, HIDDEN], &start);
        let positions: Vec<usize> = (PREFIX..PREFIX + CANVAS).collect();
        let mut trace = Vec::new();
        let mut layer_trace = Vec::new();
        let mut stages = Vec::new();
        let output = decoder_forward_traced_from_hidden(
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
        assert_eq!(layer_trace.len(), 1, "layer {layer} trace");
        assert_eq!(
            output.layout().shape(),
            vec![CANVAS, HIDDEN].as_slice(),
            "layer {layer} runner output shape"
        );

        let rust_layer = host.to_f32(&layer_trace[0]);
        let golden_layer = load(
            &dir,
            &format!("decoder_layer_{layer}"),
            &[1, CANVAS, HIDDEN],
        );
        let (lmax, lrms) = stats(&rust_layer, &golden_layer);

        let (rust_scores, _, rust_indices_vec) = &trace[0];
        let golden_scores = load(&dir, &format!("moe{layer}_router_scores"), &[CANVAS, 128]);
        let (smax, srms) = stats(rust_scores, &golden_scores);
        let rust_indices: Vec<u32> = rust_indices_vec.iter().map(|v| *v as u32).collect();
        let golden_indices: Vec<u32> = load(
            &dir,
            &format!("moe{layer}_router_indices"),
            &[CANVAS, TOP_K],
        )
        .iter()
        .map(|v| *v as u32)
        .collect();

        let mut faithful = 0usize;
        let mut swaps = 0usize;
        let mut worst_row = 0.0f32;
        for row in 0..CANVAS {
            rows += 1;
            let max_abs = rust_layer[row * HIDDEN..(row + 1) * HIDDEN]
                .iter()
                .zip(&golden_layer[row * HIDDEN..(row + 1) * HIDDEN])
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            let peak = golden_layer[row * HIDDEN..(row + 1) * HIDDEN]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            let steps = max_abs / bf16_step(peak);
            worst_row = worst_row.max(steps);
            let rust_set = top_k_set(&rust_indices, row);
            let golden_set = top_k_set(&golden_indices, row);
            if rust_set == golden_set {
                faithful += 1;
                faithful_rows += 1;
            } else {
                // Exactly one boundary swap whose two experts are a tie in
                // the reference's own scores (gap within the measured score
                // drift) is drift; anything else is structural.
                let extra: Vec<u32> = rust_set
                    .iter()
                    .copied()
                    .filter(|e| !golden_set.contains(e))
                    .collect();
                let missing: Vec<u32> = golden_set
                    .iter()
                    .copied()
                    .filter(|e| !rust_set.contains(e))
                    .collect();
                let gap = (golden_scores[row * 128 + extra[0] as usize]
                    - golden_scores[row * 128 + missing[0] as usize])
                    .abs();
                if extra.len() == 1 && missing.len() == 1 && gap <= MAX_SCORE_DRIFT {
                    swaps += 1;
                    swap_rows += 1;
                } else {
                    structural.push((layer, row, extra, missing, gap));
                }
            }
        }
        if worst_row > worst_row_steps {
            worst_row_steps = worst_row;
            worst_row_layer = layer;
        }
        eprintln!(
            "{layer:>5}  {lmax:>9.4}  {lrms:>10.6}  {smax:>10.6}  {srms:>10.7}  \
             {faithful:>7}  {swaps:>6}  {worst_row:>14.1}"
        );
    }

    for (layer, row, extra, missing, gap) in &structural {
        eprintln!(
            "STRUCTUREL couche {layer} ligne {row}: rust+{extra:?} référence+{missing:?} écart={gap:.6} > tol {MAX_SCORE_DRIFT}"
        );
    }
    eprintln!(
        "lignes fideles={faithful_rows}/{rows}  echanges de frontiere={swap_rows}  structurels={}  pire ligne={worst_row_steps:.1} pas (couche {worst_row_layer})",
        structural.len()
    );

    // Numeric gates, fixed once from the 30-layer measurement table
    // (`docs/audits/m10-cpu-campaign-2026-11-25.md`):
    // * Expert selection: the reference's own 8-vs-9 top-k boundary is at tie
    //   precision — measured gaps are 0.0 in most rows and never above ~5e-4
    //   — so an exact top-{TOP_K} set equality is only reproducible on a
    //   bit-identical reduction. A row passes when the set matches the
    //   reference's exactly (470/480) or differs by exactly one boundary swap
    //   whose experts are tied in the reference within the observed score
    //   drift (10/480). Anything else — a swap across a real gap, or two or
    //   more swaps — is structural and fails.
    // * Drift: observed worst 10.0 BF16 steps of the row peak (layer 26);
    //   the bound is the D1 rule `max(2 × measured, 4.0)` = 20.0 — the same
    //   rule as the encoder (2 × 2.0 = 4.0) and the vision bisect (floor 4.0).
    const MAX_SCORE_DRIFT: f32 = 1.0e-3;
    const MAX_ROW_DRIFT_STEPS: f32 = 20.0;

    assert!(
        structural.is_empty(),
        "{} (layer, row) pairs diverged from the reference top-{TOP_K} set beyond boundary ties",
        structural.len()
    );
    assert!(
        worst_row_steps <= MAX_ROW_DRIFT_STEPS,
        "worst row drifted {worst_row_steps:.1} BF16 steps (layer {worst_row_layer}), above the {MAX_ROW_DRIFT_STEPS} bound"
    );
}
