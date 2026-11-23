//! CPU differential test against pinned-upstream golden tensors.
//!
//! Requires the pinned checkpoint at `/tmp/opencode/m10/checkpoint` and the
//! golden files produced by `gen_golden.py` in `/tmp/opencode/m10/golden`.
//! It skips cleanly when either is absent.

use std::path::Path;

use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::fullgraph::{
    decoder_forward, embed_vision_forward, encoder_forward, lm_logits, vision_forward,
};
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
#[ignore = "requires the pinned checkpoint and venv-pinned golden files (release profile)"]
fn encoder_matches_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{GOLDEN}/encoder_last_hidden.bin")).is_file()
    {
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

#[test]
#[ignore = "requires the pinned checkpoint and venv-pinned golden files (release profile)"]
fn vision_tower_matches_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{GOLDEN}/vision_last_hidden_direct.bin")).is_file()
    {
        eprintln!("skip: vision golden unavailable");
        return;
    }
    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let pixels = load("vision_pixel_values", &[1, 9, 768]);
    let pos = load("vision_pixel_position_ids", &[1, 9, 2]);
    let positions: Vec<(u32, u32)> = pos
        .chunks_exact(2)
        .map(|c| (c[0] as u32, c[1] as u32))
        .collect();
    assert_eq!(positions.len(), 9);
    let (post_encoder, pooled) = vision_forward(&host, &ckpt, &cfg, &pixels, &positions, 3);
    let rust_vision = host.to_f32(&post_encoder);
    let golden_vision = load("vision_last_hidden_direct", &[1, 9, 1152]);
    let (vmax, vrms) = stats(&rust_vision, &golden_vision);
    eprintln!("vision_last_hidden max_abs={vmax:.6} rms={vrms:.6}");
    assert!(vmax <= 0.06, "vision hidden drift {vmax}");
    assert!(vrms <= 2e-3, "vision hidden rms {vrms}");
    let embedded = embed_vision_forward(&host, &ckpt, &cfg, &pooled);
    let rust_embed = host.to_f32(&embedded);
    let golden_embed = load("embed_vision_output_direct", &[1, 1, 2816]);
    let (emax, erms) = stats(&rust_embed, &golden_embed);
    eprintln!("embed_vision max_abs={emax:.6} rms={erms:.6}");
    assert!(emax <= 0.1, "embed vision drift {emax}");
    assert!(erms <= 4e-3, "embed vision rms {erms}");
}

#[test]
#[ignore = "requires the pinned checkpoint and venv-pinned golden files (release profile)"]
fn decoder_first_step_matches_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{GOLDEN}/decoder_last_hidden0_direct.bin")).is_file()
    {
        eprintln!("skip: decoder golden unavailable");
        return;
    }
    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let ids = load("prompt_ids", &[1, 7])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    let mut cache = KvCache::new(30);
    encoder_forward(&host, &ckpt, &cfg, &ids, &mut cache);
    let canvas = load("canvas0_direct", &[1, 16])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    assert_eq!(canvas.len(), 16);
    let mut trace = Vec::new();
    let hidden = decoder_forward(&host, &ckpt, &cfg, &canvas, &mut cache, None, &mut trace);
    let rust_hidden = host.to_f32(&hidden);
    let golden_hidden = load("decoder_last_hidden0_direct", &[1, 16, 2816]);
    let (hmax, hrms) = stats(&rust_hidden, &golden_hidden);
    eprintln!("decoder_last_hidden0 max_abs={hmax:.6} rms={hrms:.6}");
    assert!(hmax <= 0.12, "decoder hidden drift {hmax}");
    assert!(hrms <= 4e-3, "decoder hidden rms {hrms}");
    let logits = lm_logits(&host, &ckpt, &cfg, &hidden);
    let rust_logits = host.to_f32(&logits);
    let golden_logits = load("decoder_logits0_direct", &[1, 16, 262144]);
    let (lmax, lrms) = stats(&rust_logits, &golden_logits);
    eprintln!("decoder_logits0 max_abs={lmax:.6} rms={lrms:.6}");
    // Logits carry the full 30-layer drift; the campaign documents this bound.
    assert!(lmax <= 2.0, "decoder logits drift {lmax}");
    assert!(lrms <= 0.1, "decoder logits rms {lrms}");

    // Layer-0 MoE router: scores, top-k weights and indices.
    let (scores, weights, indices) = &trace[0];
    let golden_scores = load("moe0_router_logits", &[16, 128]);
    let (smax, srms) = stats(scores, &golden_scores);
    eprintln!("moe0 router scores max_abs={smax:.6} rms={srms:.6}");
    assert!(smax <= 0.02, "router scores drift {smax}");
    let golden_weights = load("moe0_topk_weights", &[16, 8]);
    let (wmax, wrms) = stats(weights, &golden_weights);
    eprintln!("moe0 topk weights max_abs={wmax:.6} rms={wrms:.6}");
    assert!(wmax <= 0.01, "router weights drift {wmax}");
    let golden_indices = load("moe0_topk_indices", &[16, 8]);
    assert_eq!(
        indices.iter().map(|v| *v as u32).collect::<Vec<u32>>(),
        golden_indices
            .iter()
            .map(|v| *v as u32)
            .collect::<Vec<u32>>(),
        "router top-k indices must match exactly"
    );
    println!("moe0 router: indices exact, weights rms={wrms:.6}");
}
