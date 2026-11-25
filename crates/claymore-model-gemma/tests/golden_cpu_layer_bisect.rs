//! CPU per-layer bisect of the encoder, fed with the reference's own layer
//! inputs.
//!
//! The chain-level differential (`golden_cpu.rs`) compares a Claymore layer
//! whose input already drifted from the reference, so it cannot tell a wrong
//! layer apart from a layer that faithfully amplifies its predecessors. This
//! test removes that ambiguity: `tests/data/m10_layers/enc_in_{L}.bin`
//! is the `hidden_states` tensor the pinned upstream reference actually handed to
//! encoder layer `L`, so each layer runs with bit-exact reference input.
//!
//! The reference is bit-reproducible on CPU (verified across thread counts), so
//! a wrong layer stands out instead of hiding behind the chain drift.
//!
//! The fixture set is committed and read-only (`m10_reference.rs` verifies it
//! against its pinned manifest); regenerate it only through
//! `tools/m10/gen_reference.py` into a fresh directory — never repin in place.
//! Only the pinned checkpoint (env `CLAYMORE_M10_CHECKPOINT`) is external.
//!
//! Run:
//! ```text
//! CLAYMORE_M10_CHECKPOINT=/path/to/checkpoint \
//!   cargo test --release -p claymore-model-gemma --test golden_cpu_layer_bisect \
//!   --locked -- --ignored --test-threads=1 --nocapture
//! ```

use std::path::Path;

use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::fullgraph::encoder_forward_traced_from_hidden;
use claymore_model_gemma::graph::{Host, KvCache};

const ROWS: usize = 7;
const HIDDEN: usize = 2816;
const LAYER_COUNT: usize = 30;
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

fn load(root: &Path, name: &str, shape: &[usize]) -> Vec<f32> {
    let bytes = std::fs::read(root.join(format!("{name}.bin"))).expect("golden file");
    let out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        out.len(),
        shape.iter().product::<usize>(),
        "golden {name} size"
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

fn row_diff(rust: &[f32], golden: &[f32], row: usize) -> (f32, f32) {
    let a = &rust[row * HIDDEN..(row + 1) * HIDDEN];
    let b = &golden[row * HIDDEN..(row + 1) * HIDDEN];
    stats(a, b)
}

fn row_peak(v: &[f32], row: usize) -> f32 {
    v[row * HIDDEN..(row + 1) * HIDDEN]
        .iter()
        .fold(0.0f32, |m, x| m.max(x.abs()))
}

fn top_k_set(indices: &[u32], row: usize) -> Vec<u32> {
    let mut s: Vec<u32> = indices[row * TOP_K..(row + 1) * TOP_K].to_vec();
    s.sort_unstable();
    s
}

#[test]
#[ignore = "requires the pinned checkpoint (release profile)"]
fn encoder_layers_reproduce_the_reference_when_fed_reference_inputs() {
    let dir = fixture_dir();
    if !checkpoint_root().is_dir()
        || !dir.join("encoder_layer_0.bin").is_file()
        || !dir.join("enc_in_0.bin").is_file()
    {
        eprintln!(
            "skip: checkpoint or committed fixture set unavailable — set CLAYMORE_M10_CHECKPOINT \
             (see docs/m10-diffusiongemma.md for the pinned revision)"
        );
        return;
    }

    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(checkpoint_root()).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let mut cache = KvCache::new(LAYER_COUNT);

    let mut rows = 0usize;
    let mut faithful_rows = 0usize;
    let mut worst_faithful_steps = 0.0f32;
    let mut worst_faithful_layer = 0usize;
    let mut worst_row_layer = 0usize;
    let mut worst_row_steps = 0.0f32;

    eprintln!("couche  sortie_max  sortie_rms  expert_max  expert_rms  scores_max  scores_rms  fideles  pire_fidele(ULP)  pire_global(ULP)");
    for layer in 0..LAYER_COUNT {
        let start = load(&dir, &format!("enc_in_{layer}"), &[1, ROWS, HIDDEN]);
        let start_t = host.bf16(vec![ROWS, HIDDEN], &start);
        let mut trace = Vec::new();
        let output = encoder_forward_traced_from_hidden(
            &host,
            &ckpt,
            &cfg,
            &start_t,
            &mut cache,
            &mut trace,
            layer..layer + 1,
        );
        assert_eq!(trace.len(), 1, "layer {layer} trace");
        assert_eq!(
            output.layout().shape(),
            start_t.layout().shape(),
            "layer {layer} output shape"
        );

        let rust_layer = host.to_f32(&trace[0].hidden);
        let golden_layer = load(&dir, &format!("encoder_layer_{layer}"), &[1, ROWS, HIDDEN]);
        let (lmax, lrms) = stats(&rust_layer, &golden_layer);

        let rust_expert = host.to_f32(&trace[0].expert_output);
        let golden_expert = load(
            &dir,
            &format!("encoder{layer}_expert_output"),
            &[ROWS, HIDDEN],
        );
        let (emax, erms) = stats(&rust_expert, &golden_expert);

        let rust_scores: &[f32] = &trace[0].router_scores;
        let golden_scores = load(&dir, &format!("encoder{layer}_router_scores"), &[ROWS, 128]);
        let (smax, srms) = stats(rust_scores, &golden_scores);

        let rust_indices: Vec<u32> = trace[0].router_indices.iter().map(|v| *v as u32).collect();
        let golden_indices: Vec<u32> = load(
            &dir,
            &format!("encoder{layer}_router_indices"),
            &[ROWS, TOP_K],
        )
        .iter()
        .map(|v| *v as u32)
        .collect();

        let mut faithful = 0usize;
        let mut worst_row = 0.0f32;
        let mut worst_faithful_row = 0.0f32;
        for row in 0..ROWS {
            rows += 1;
            let (rmax, _) = row_diff(&rust_layer, &golden_layer, row);
            let steps = rmax / bf16_step(row_peak(&golden_layer, row));
            worst_row = worst_row.max(steps);
            if top_k_set(&rust_indices, row) == top_k_set(&golden_indices, row) {
                faithful += 1;
                faithful_rows += 1;
                worst_faithful_row = worst_faithful_row.max(steps);
            }
        }
        if worst_faithful_row > worst_faithful_steps {
            worst_faithful_steps = worst_faithful_row;
            worst_faithful_layer = layer;
        }
        if worst_row > worst_row_steps {
            worst_row_steps = worst_row;
            worst_row_layer = layer;
        }

        eprintln!(
            "{layer:>5}  {lmax:>9.4}  {lrms:>10.5}  {emax:>10.5}  {erms:>10.5}  \
             {smax:>10.6}  {srms:>10.6}  {faithful:>7}  {worst_faithful_row:>14.1}  {worst_row:>13.1}"
        );
    }

    eprintln!(
        "lignes fideles={faithful_rows}/{rows}  pire ligne fidele={worst_faithful_steps:.1} ULP (couche {worst_faithful_layer})  \
         pire ligne globale={worst_row_steps:.1} ULP (couche {worst_row_layer})"
    );

    // Numeric gate, fixed once from the measurement table printed above
    // (30 layers x 7 rows fed with reference input, 2026-11-25): every row
    // reproduced the reference expert set, and the worst row drifted by 2.0
    // BF16 steps of that row's reference peak (layers 2, 20 and 26). The bound
    // is twice that observation, so it absorbs single-step differences of
    // reduction order while failing anything structural -- a wrong formula, a
    // wrong rounding point, or an expert set that does not match (that last
    // case is caught by the set-equality count below, independently of size).
    const MAX_ROW_DRIFT_STEPS: f32 = 4.0;

    assert_eq!(
        faithful_rows, rows,
        "some (layer, row) pair did not reproduce the reference top-{TOP_K} expert set"
    );
    assert!(
        worst_faithful_steps <= MAX_ROW_DRIFT_STEPS,
        "faithful rows drifted {worst_faithful_steps:.1} BF16 steps (layer {worst_faithful_layer}), above the {MAX_ROW_DRIFT_STEPS} bound"
    );
    assert!(
        worst_row_steps <= MAX_ROW_DRIFT_STEPS,
        "worst row drifted {worst_row_steps:.1} BF16 steps (layer {worst_row_layer}), above the {MAX_ROW_DRIFT_STEPS} bound"
    );
}
