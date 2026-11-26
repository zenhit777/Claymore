//! CPU per-layer bisect of the 27-layer vision tower, fed with the
//! reference's own layer inputs.
//!
//! `tests/data/m10_layers/vision_in_{L}.bin` is the BF16-to-F32 widened
//! `[1, 9, 1152]` state the pinned upstream reference handed to vision layer
//! `L` (after patch embedding and position-embedding addition), captured by
//! `tools/m10/gen_reference.py` and gated bit-exact against the campaign
//! golden set. Feeding each layer the reference's own input judges that layer
//! on its own instead of on the drift accumulated by its predecessors — the
//! chain-level vision differential in `golden_cpu.rs` cannot do that.
//!
//! Entry fidelity: for every `L > 0` the fixture must satisfy
//! `vision_in_{L} == vision_layer_{L-1}` bit for bit before any Rust layer is
//! compared (enforced below); `vision_in_0` is the reference's own first-layer
//! entry, so the injection point is validated by the fixture itself.
//!
//! Conformity follows architect decision D1: drift in BF16 steps of the
//! reference row peak, never an absolute `max_abs`. The numeric bound is
//! fixed once from the full 27-layer measurement table printed by this test.
//!
//! Requires the pinned checkpoint (env `CLAYMORE_M10_CHECKPOINT`). Run:
//! ```text
//! CLAYMORE_M10_CHECKPOINT=/path/to/checkpoint \
//!   cargo test --release -p claymore-model-gemma --test vision_layer_bisect \
//!   --locked -- --ignored --test-threads=1 --nocapture
//! ```

use std::path::Path;

use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::fullgraph::vision_forward_layer_from_hidden;
use claymore_model_gemma::graph::Host;

const PATCHES: usize = 9;
const HIDDEN: usize = 1152;
const LAYER_COUNT: usize = 27;

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

/// Decision D1: worst drift over the `patches` rows, in BF16 steps of each
/// row's reference peak.
fn worst_row_drift_steps(rust: &[f32], golden: &[f32], rows: usize, row_len: usize) -> f32 {
    let mut worst = 0.0f32;
    for (r, g) in rust.chunks(row_len).zip(golden.chunks(row_len)).take(rows) {
        let peak = g.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let max_abs = r
            .iter()
            .zip(g)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        worst = worst.max(max_abs / bf16_step(peak));
    }
    worst
}

#[test]
#[ignore = "requires the pinned checkpoint (release profile)"]
fn vision_layers_reproduce_the_reference_when_fed_reference_inputs() {
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
        assert!(
            dir.join(format!("vision_in_{l}.bin")).is_file()
                && dir.join(format!("vision_layer_{l}.bin")).is_file(),
            "fixture set incomplete for layer {l}"
        );
    }

    // Entry fidelity: the fixture chain must be continuous before any Rust
    // comparison is meaningful — for every L > 0, vision_in_{L} must equal
    // vision_layer_{L-1} bit for bit (both read as raw little-endian words).
    for l in 1..LAYER_COUNT {
        let input = std::fs::read(dir.join(format!("vision_in_{l}.bin"))).expect("fixture");
        let previous =
            std::fs::read(dir.join(format!("vision_layer_{}.bin", l - 1))).expect("fixture");
        assert_eq!(
            input,
            previous,
            "entry fidelity broken: vision_in_{l} != vision_layer_{}",
            l - 1
        );
    }

    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(&root).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();

    // The pixel fixtures are recorded inputs; the injected state per layer is
    // `vision_in_{L}`. Load the position ids only (the patch embedder output
    // is exactly what the fixture chain starts from).
    let pos = load(&dir, "vision_pixel_position_ids", &[1, PATCHES, 2]);
    let positions: Vec<(u32, u32)> = pos
        .chunks_exact(2)
        .map(|c| (c[0] as u32, c[1] as u32))
        .collect();
    assert_eq!(positions.len(), PATCHES);

    let mut worst_layer = 0usize;
    let mut worst_steps = 0.0f32;
    for layer in 0..LAYER_COUNT {
        let start = load(&dir, &format!("vision_in_{layer}"), &[1, PATCHES, HIDDEN]);
        let start_t = host.bf16(vec![PATCHES, HIDDEN], &start);
        let mut stages = Vec::new();
        let output = vision_forward_layer_from_hidden(
            &host,
            &ckpt,
            &cfg,
            &start_t,
            &positions,
            layer,
            &mut stages,
        );
        assert_eq!(stages.len(), 7, "layer {layer} stage count");
        let rust = host.to_f32(&output);
        let golden = load(
            &dir,
            &format!("vision_layer_{layer}"),
            &[1, PATCHES, HIDDEN],
        );
        let (lmax, lrms) = stats(&rust, &golden);
        let steps = worst_row_drift_steps(&rust, &golden, PATCHES, HIDDEN);
        if steps > worst_steps {
            worst_steps = steps;
            worst_layer = layer;
        }
        eprintln!(
            "vision layer {layer:>2}  max_abs={lmax:.6}  rms={lrms:.9}  pire_ligne={steps:>5.1} pas BF16"
        );
    }
    eprintln!(
        "pire dérive vision {worst_steps:.1} pas BF16 (couche {worst_layer}) sur {LAYER_COUNT} couches"
    );

    // Numeric gate, fixed from the full 27-layer measurement of 2026-11-25
    // (`docs/audits/m10-cpu-campaign-2026-11-25.md`): the worst row over all
    // 27 layers drifted **0.2** BF16 steps of its reference peak (layer 7).
    // The bound is the D1 rule `max(2 × measured, 4.0)` = 4.0 — the floor
    // guarantees a single one-step reduction-order difference on another host
    // (one step = 1.0 at the row peak) cannot fail the contract, while any
    // structural error (wrong tensor, wrong layer) measures orders of
    // magnitude above it. Same rule as the encoder (2 × 2.0 = 4.0) and the
    // decoder (2 × 10.0 = 20.0) bisections.
    const MAX_ROW_DRIFT_STEPS: f32 = 4.0;
    assert!(
        worst_steps <= MAX_ROW_DRIFT_STEPS,
        "a vision layer drifted {worst_steps:.1} BF16 steps (layer {worst_layer}), above the {MAX_ROW_DRIFT_STEPS} bound"
    );
}
