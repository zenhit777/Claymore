//! CPU differential test against pinned-upstream golden tensors.
//!
//! Requires the pinned checkpoint at `/tmp/opencode/m10/checkpoint` and the
//! golden files produced by `gen_golden.py` in `/tmp/opencode/m10/golden`.
//! It skips cleanly when either is absent.

use std::path::Path;

use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::config::GemmaModelConfig;
use claymore_model_gemma::fullgraph::encoder_forward;
use claymore_model_gemma::graph::{Host, KvCache};

const ROOT: &str = "/tmp/opencode/m10/checkpoint";
const GOLDEN: &str = "/tmp/opencode/m10/golden";

fn load(name: &str, shape: &[usize]) -> Vec<f32> {
    let path = format!("{GOLDEN}/{name}.bin");
    let bytes = std::fs::read(&path).expect("golden file");
    let out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let numel: usize = shape.iter().product();
    assert_eq!(out.len(), numel, "golden {name} size");
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

#[test]
fn encoder_matches_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir() || !Path::new(&format!("{GOLDEN}/encoder_last_hidden.bin")).is_file() {
        eprintln!("skip: checkpoint/golden unavailable");
        return;
    }
    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let ids = load("prompt_ids", &[1, 7])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    assert_eq!(ids.len(), 7);
    let mut cache = KvCache::new(30);
    let hidden = encoder_forward(&host, &ckpt, &cfg, &ids, &mut cache);
    let golden_hidden = load("encoder_last_hidden", &[1, 7, 2816]);
    let rust_hidden = host.to_f32(&hidden);
    let (max_abs, rms) = stats(&rust_hidden, &golden_hidden);
    eprintln!("encoder_last_hidden max_abs={max_abs:.6} rms={rms:.6}");
    // BF16 reference drift bound documented in the M10 audit campaign.
    assert!(max_abs <= 0.03, "encoder hidden drift {max_abs}");
    assert!(rms <= 2e-3, "encoder hidden rms {rms}");
    // Layer 0 keys/values after the encoder prefill.
    let keys = host.to_f32(&cache.layers[0].keys);
    let golden_keys = load("enc_kv_0_keys", &[1, 8, 7, 256]);
    let (kmax, krms) = stats(&golden_keys, &keys);
    eprintln!("enc_kv_0 keys max_abs={kmax:.6} rms={krms:.6}");
    assert!(kmax <= 0.05, "kv keys drift {kmax}");
    assert!(krms <= 2e-3, "kv keys rms {krms}");
}
