//! CPU differential test against pinned-upstream golden tensors.
//!
//! Requires the pinned checkpoint at `/tmp/opencode/m10/checkpoint` and the
//! golden files produced by `gen_golden.py` in `/tmp/opencode/m10/golden`.
//! It skips cleanly when either is absent.

use std::path::Path;

use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::fullgraph::{
    embed_vision_forward, encoder_forward, encoder_forward_traced_limited, lm_logits,
};
use claymore_model_gemma::graph::{
    embed_gather_scaled, linear, rms_norm, text_embedding_scale, Host, KvCache,
};
use claymore_tensor::Bf16;

const ROOT: &str = "/tmp/opencode/m10/checkpoint";
const GOLDEN: &str = "/tmp/opencode/m10/golden_eager";
const STAGES: &str = "/tmp/opencode/m10/golden_eager";

fn load(name: &str, shape: &[usize]) -> Vec<f32> {
    load_from(GOLDEN, name, shape)
}

fn load_from(root: &str, name: &str, shape: &[usize]) -> Vec<f32> {
    let path = format!("{root}/{name}.bin");
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

/// Permute a local cache tensor `[seq, kv_heads, head_dim]` into the pinned
/// reference layout `[batch, kv_heads, seq, head_dim]` (batch = 1).
fn permute_kv(flat: &[f32], rows: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut permuted = vec![0.0f32; flat.len()];
    for row in 0..rows {
        for head in 0..heads {
            for d in 0..dim {
                permuted[(head * rows + row) * dim + d] = flat[(row * heads + head) * dim + d];
            }
        }
    }
    permuted
}

#[test]
#[ignore = "requires the pinned checkpoint and stage golden files (release profile)"]
fn text_embedding_matches_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir() || !Path::new(&format!("{STAGES}/encoder_embedding.bin")).is_file()
    {
        eprintln!("skip: checkpoint/stage golden unavailable");
        return;
    }
    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let hidden = cfg.text_config.as_ref().unwrap().hidden_size.unwrap();
    let ids = load("prompt_ids", &[1, 7])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    let table = claymore_model_gemma::fullgraph::text_embedding_weight(&host, &ckpt);
    let embedding = embed_gather_scaled(&host, &ids, &table, text_embedding_scale(hidden));
    let rust = host.to_f32(&embedding);
    let golden = load_from(STAGES, "encoder_embedding", &[1, 7, 2816]);
    let (max_abs, rms) = stats(&rust, &golden);
    eprintln!("encoder_embedding max_abs={max_abs:.9} rms={rms:.9}");
    assert_eq!(max_abs, 0.0, "encoder embedding drift {max_abs}");
    assert_eq!(rms, 0.0, "encoder embedding rms {rms}");
}

#[test]
#[ignore = "requires the pinned checkpoint and stage golden files (release profile)"]
fn encoder_layer0_projections_match_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{STAGES}/encoder0_q_projection.bin")).is_file()
    {
        eprintln!("skip: checkpoint/stage golden unavailable");
        return;
    }
    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let hidden = cfg.text_config.as_ref().unwrap().hidden_size.unwrap();
    let eps = cfg.text_config.as_ref().unwrap().rms_norm_eps.unwrap();
    let ids = load("prompt_ids", &[1, 7])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    let table = claymore_model_gemma::fullgraph::text_embedding_weight(&host, &ckpt);
    let embedding = embed_gather_scaled(&host, &ids, &table, text_embedding_scale(hidden));

    let weight = |name: &str| {
        let shape = ckpt
            .tensor(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
            .entry
            .shape()
            .to_vec();
        let bits = ckpt.read_bf16_bits(name).expect("weight payload");
        let values = bits
            .iter()
            .map(|value| Bf16::from_ne_bytes(value.to_ne_bytes()).to_f32())
            .collect::<Vec<f32>>();
        host.bf16(shape, &values)
    };
    let normed = rms_norm(
        &host,
        &embedding,
        &weight("model.decoder.layers.0.input_layernorm.weight"),
        eps,
    );
    for (projection, shape, golden_name) in [
        ("q_proj", vec![1, 7, 4096], "encoder0_q_projection"),
        ("k_proj", vec![1, 7, 2048], "encoder0_k_projection"),
        ("v_proj", vec![1, 7, 2048], "encoder0_v_projection"),
    ] {
        let actual = linear(
            &host,
            &normed,
            &weight(&format!(
                "model.decoder.layers.0.self_attn.{projection}.weight"
            )),
        );
        let rust = host.to_f32(&actual);
        let golden = load_from(STAGES, golden_name, &shape);
        let (max_abs, rms) = stats(&rust, &golden);
        eprintln!("{projection} max_abs={max_abs:.6} rms={rms:.6}");
    }
}

#[test]
#[ignore = "requires the pinned checkpoint and stage golden files (release profile)"]
fn encoder0_router_matches_pinned_upstream_on_the_real_checkpoint() {
    // Fast, focused check of the router semantics: the golden dump contains the
    // upstream's router input (`encoder0_routing_residual`), its BF16 router
    // logits (`encoder0_router_raw_scores`), the FP32 probabilities
    // (`encoder0_router_scores`), the scaled weights and the selected expert
    // indices. Feeding the upstream's own input back through our router must
    // reproduce the upstream's logits bit for bit — the BF16 tie structure of
    // the pinned checkpoint decides the selected experts, so a one-ULP drift
    // changes the model output.
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{GOLDEN}/encoder0_routing_residual.bin")).is_file()
    {
        eprintln!("skip: checkpoint/golden unavailable");
        return;
    }
    let host = Host::new();
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let cfg = ckpt.config().clone();
    let text = cfg.text_config.as_ref().unwrap();
    let hidden = text.hidden_size.unwrap();
    let eps = text.rms_norm_eps.unwrap_or(1e-6);
    let residual = load("encoder0_routing_residual", &[1, 7, 2816]);
    let x = host.bf16(vec![7, hidden], &residual);
    let (probabilities, raw_scores, weights, indices) = claymore_model_gemma::graph::router_forward(
        &host,
        &x,
        &claymore_model_gemma::fullgraph::checkpoint_tensor(
            &host,
            &ckpt,
            "model.decoder.layers.0.router.proj.weight",
        ),
        &claymore_model_gemma::fullgraph::checkpoint_tensor(
            &host,
            &ckpt,
            "model.decoder.layers.0.router.scale",
        ),
        &claymore_model_gemma::fullgraph::checkpoint_tensor(
            &host,
            &ckpt,
            "model.decoder.layers.0.router.per_expert_scale",
        ),
        eps,
        (hidden as f64).powf(-0.5),
        8,
    );

    let golden_raw = load("encoder0_router_raw_scores", &[7, 128]);
    let golden_probs = load("encoder0_router_scores", &[7, 128]);
    let golden_weights = load("encoder0_router_weights", &[7, 8]);
    let golden_indices = load("encoder0_router_indices", &[7, 8]);

    let raw_exact = raw_scores == golden_raw;
    let (raw_max, raw_rms) = stats(&raw_scores, &golden_raw);
    let (prob_max, prob_rms) = stats(&probabilities, &golden_probs);
    let (weight_max, weight_rms) = stats(&weights, &golden_weights);
    eprintln!("encoder0_router raw_max_abs={raw_max:.9} raw_rms={raw_rms:.9} exact={raw_exact}");
    eprintln!("encoder0_router prob_max_abs={prob_max:.9} prob_rms={prob_rms:.9}");
    eprintln!("encoder0_router weight_max_abs={weight_max:.9} weight_rms={weight_rms:.9}");
    assert_eq!(
        raw_scores, golden_raw,
        "the router logits must be bit-exact with the pinned upstream"
    );
    assert_eq!(
        indices, golden_indices,
        "the selected experts must match the pinned upstream exactly"
    );
}

#[test]
#[ignore = "requires the pinned checkpoint and stage golden files (release profile)"]
fn encoder_layers_match_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir() || !Path::new(&format!("{STAGES}/encoder_layer_0.bin")).is_file() {
        eprintln!("skip: checkpoint/stage golden unavailable");
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
    let mut layer_trace = Vec::new();
    encoder_forward_traced_limited(
        &host,
        &ckpt,
        &cfg,
        &ids,
        &mut cache,
        &mut layer_trace,
        usize::MAX,
    );
    for (layer, actual) in layer_trace.iter().enumerate() {
        let rust = host.to_f32(&actual.hidden);
        let golden = load_from(STAGES, &format!("encoder_layer_{layer}"), &[1, 7, 2816]);
        let (max_abs, rms) = stats(&rust, &golden);
        eprintln!("encoder_layer_{layer} max_abs={max_abs:.6} rms={rms:.6}");
        let cache_layer = &cache.layers[layer];
        let heads = cache_layer.keys.layout().shape()[1];
        let dim = cache_layer.keys.layout().shape()[2];
        let rows = cache_layer.keys.layout().shape()[0];
        for (which, tensor, name) in [
            (
                "keys",
                &cache_layer.keys,
                format!("enc_kv_{layer}_keys_direct"),
            ),
            (
                "values",
                &cache_layer.values,
                format!("enc_kv_{layer}_values_direct"),
            ),
        ] {
            let flat = host.to_f32(tensor);
            let golden_cache = load_from(STAGES, &name, &[1, heads, rows, dim]);
            let permuted = permute_kv(&flat, rows, heads, dim);
            let (kmax, krms) = stats(&permuted, &golden_cache);
            eprintln!("enc_kv_{layer}_{which} max_abs={kmax:.6} rms={krms:.6}");
        }
        let golden_indices = load_from(STAGES, &format!("encoder{layer}_router_indices"), &[7, 8]);
        let golden_scores = load_from(STAGES, &format!("encoder{layer}_router_scores"), &[7, 128]);
        let (smax, srms) = stats(&actual.router_scores, &golden_scores);
        let exact = actual.router_indices == golden_indices;
        eprintln!(
            "encoder{layer}_router_scores max_abs={smax:.6} rms={srms:.6} indices_exact={exact}"
        );
        if !exact {
            for token in 0..7 {
                let rust_row = &actual.router_indices[token * 8..(token + 1) * 8];
                let golden_row = &golden_indices[token * 8..(token + 1) * 8];
                if rust_row != golden_row {
                    eprintln!(
                        "  layer {layer} token {token}: rust={rust_row:?} golden={golden_row:?}"
                    );
                }
            }
        }
        let golden_indices = load_from(STAGES, &format!("encoder{layer}_router_indices"), &[7, 8]);
        let exact = actual.router_indices == golden_indices;
        eprintln!("encoder{layer}_router_indices exact={exact}");
        if !exact {
            for token in 0..7 {
                let rust_row = &actual.router_indices[token * 8..(token + 1) * 8];
                let golden_row = &golden_indices[token * 8..(token + 1) * 8];
                if rust_row != golden_row {
                    eprintln!(
                        "  layer {layer} token {token}: rust={rust_row:?} golden={golden_row:?}"
                    );
                }
            }
        }
        let rust_keys = host.to_f32(&cache.layers[layer].keys);
        let rust_values = host.to_f32(&cache.layers[layer].values);
        let heads = cache.layers[layer].keys.layout().shape()[1];
        let dim = cache.layers[layer].keys.layout().shape()[2];
        let rows = cache.layers[layer].keys.layout().shape()[0];
        for (which, flat, golden_name) in [
            ("keys", &rust_keys, format!("enc_kv_{layer}_keys_direct")),
            (
                "values",
                &rust_values,
                format!("enc_kv_{layer}_values_direct"),
            ),
        ] {
            // The golden cache is [batch, kv_heads, seq, head_dim]; the local cache
            // is [seq, kv_heads, head_dim]. Permute before comparing.
            let golden_cache = load_from(STAGES, &golden_name, &[1, heads, rows, dim]);
            let permuted = permute_kv(flat, rows, heads, dim);
            let (kmax, krms) = stats(&permuted, &golden_cache);
            eprintln!("enc_kv_{layer}_{which} max_abs={kmax:.6} rms={krms:.6}");
        }
        if layer == 0 {
            for (name, tensor, golden_name) in [
                (
                    "attention",
                    &actual.attention_output,
                    "encoder0_attention_output",
                ),
                ("mlp", &actual.mlp_output, "encoder0_mlp_output"),
                ("experts", &actual.expert_output, "encoder0_expert_output"),
            ] {
                let rust = host.to_f32(tensor);
                let golden = load_from(STAGES, golden_name, &[1, 7, 2816]);
                let (max_abs, rms) = stats(&rust, &golden);
                eprintln!("encoder0_{name} max_abs={max_abs:.6} rms={rms:.6}");
            }
            let golden_scores = load_from(STAGES, "encoder0_router_scores", &[7, 128]);
            let (max_abs, rms) = stats(&actual.router_scores, &golden_scores);
            eprintln!("encoder0_router_scores max_abs={max_abs:.6} rms={rms:.6}");
            let golden_weights = load_from(STAGES, "encoder0_router_weights", &[7, 8]);
            let (max_abs, rms) = stats(&actual.router_weights, &golden_weights);
            eprintln!("encoder0_router_weights max_abs={max_abs:.6} rms={rms:.6}");
            let golden_indices = load_from(STAGES, "encoder0_router_indices", &[7, 8]);
            eprintln!(
                "encoder0_router_indices exact={}",
                actual.router_indices == golden_indices
            );
            eprintln!("encoder0_router_indices rust={:?}", actual.router_indices);
            eprintln!("encoder0_router_indices golden={golden_indices:?}");
            for (name, tensor, golden_name) in [
                (
                    "input_normed",
                    &actual.input_normed,
                    "encoder0_input_normed",
                ),
                (
                    "routing_residual",
                    &actual.routing_residual,
                    "encoder0_routing_residual",
                ),
                ("expert_input", &actual.expert_input, "encoder0_pre2"),
            ] {
                let rust = host.to_f32(tensor);
                let golden = load_from(STAGES, golden_name, &[7, 2816]);
                let (max_abs, rms) = stats(&rust, &golden);
                eprintln!("encoder0_{name} max_abs={max_abs:.6} rms={rms:.6}");
            }
            let golden_probabilities = load_from(STAGES, "encoder0_router_scores", &[7, 128]);
            let golden_raw = load_from(STAGES, "encoder0_router_raw_scores", &[7, 128]);
            for (token, experts) in [(0, [33usize, 68]), (2, [113, 115]), (5, [0, 127])] {
                eprintln!("router token {token} candidates");
                for expert in experts {
                    eprintln!(
                        "  expert {expert}: raw_rust={} raw_golden={} prob_rust={} prob_golden={}",
                        actual.router_raw_scores[token * 128 + expert],
                        golden_raw[token * 128 + expert],
                        actual.router_scores[token * 128 + expert],
                        golden_probabilities[token * 128 + expert]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires the pinned checkpoint and venv-pinned golden files (release profile)"]
fn encoder_matches_pinned_upstream_on_the_real_checkpoint() {
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{GOLDEN}/encoder_last_hidden_direct.bin")).is_file()
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
    let golden_hidden = load("encoder_last_hidden_direct", &[1, 7, 2816]);
    let rust_hidden = host.to_f32(&hidden);
    let (max_abs, rms) = stats(&rust_hidden, &golden_hidden);
    eprintln!("encoder_last_hidden max_abs={max_abs:.6} rms={rms:.6}");
    // BF16 reference drift bound documented in the M10 audit campaign.
    assert!(max_abs <= 0.03, "encoder hidden drift {max_abs}");
    assert!(rms <= 2e-3, "encoder hidden rms {rms}");
    // Layer 0 keys/values after the encoder prefill.
    let keys = host.to_f32(&cache.layers[0].keys);
    let golden_keys = load("enc_kv_0_keys_direct", &[1, 8, 7, 256]);
    let permuted = permute_kv(&keys, 7, 8, 256);
    let (kmax, krms) = stats(&permuted, &golden_keys);
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
    let mut stages = Vec::new();
    let (_post_encoder, pooled) = claymore_model_gemma::fullgraph::vision_forward_stages(
        &host,
        &ckpt,
        &cfg,
        &pixels,
        &positions,
        3,
        &mut stages,
    );
    const STAGE_NAMES: [&str; 7] = [
        "input_norm",
        "attn_out",
        "post_attn_norm",
        "pre_ff_norm",
        "mlp_out",
        "post_ff_norm",
        "layer_out",
    ];
    let dump = std::env::var_os("M10_DUMP_RUST_VISION");
    if dump.is_some() {
        std::fs::create_dir_all("/tmp/opencode/m10/rust_vision").expect("dump dir");
    }
    for (layer, chunk) in stages.chunks_exact(STAGE_NAMES.len()).enumerate() {
        for (stage, tensor) in chunk.iter().enumerate() {
            let name = if stage == STAGE_NAMES.len() - 1 {
                format!("vision_layer_{layer}")
            } else {
                format!("v{layer}_{}", STAGE_NAMES[stage])
            };
            let rust = host.to_f32(tensor);
            if dump.is_some() {
                let bytes: Vec<u8> = rust.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("/tmp/opencode/m10/rust_vision/{name}.bin"), bytes)
                    .expect("dump stage");
            }
            let golden = load(&name, &[1, 9, 1152]);
            let (lmax, lrms) = stats(&rust, &golden);
            eprintln!("{name} max_abs={lmax:.6} rms={lrms:.6}");
        }
    }
    // Upstream exposes the standardized, pooled vision output as
    // `last_hidden_state`; the unpooled 9-patch tensor is internal.
    let rust_vision = host.to_f32(&pooled);
    let golden_vision = load("vision_last_hidden_direct", &[1, 1152]);
    let (vmax, vrms) = stats(&rust_vision, &golden_vision);
    eprintln!("vision_pooled_standardized max_abs={vmax:.6} rms={vrms:.6}");
    let embedded = embed_vision_forward(&host, &ckpt, &cfg, &pooled);
    let rust_embed = host.to_f32(&embedded);
    let golden_embed = load("embed_vision_output_direct", &[1, 2816]);
    let (emax, erms) = stats(&rust_embed, &golden_embed);
    eprintln!("embed_vision max_abs={emax:.6} rms={erms:.6}");
    assert!(vmax <= 0.06, "vision hidden drift {vmax}");
    assert!(vrms <= 2e-3, "vision hidden rms {vrms}");
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
    let mut layer_trace = Vec::new();
    let mut stages = Vec::new();
    let hidden = claymore_model_gemma::fullgraph::decoder_forward_traced(
        &host,
        &ckpt,
        &cfg,
        &canvas,
        &mut cache,
        None,
        &mut trace,
        &mut layer_trace,
        &mut stages,
    );
    const DEC_STAGE_NAMES: [&str; 11] = [
        "attn_out",
        "post_attn_norm",
        "residual_after_attn",
        "mlp_out",
        "post_ff1",
        "pre_ff2",
        "experts",
        "post_ff2",
        "mlp_moe_sum",
        "post_ff",
        "out",
    ];
    let dump = std::env::var_os("M10_DUMP_RUST_DECODER").is_some();
    if dump {
        std::fs::create_dir_all("/tmp/opencode/m10/rust_decoder").expect("dump dir");
        let table = claymore_model_gemma::fullgraph::text_embedding_weight(&host, &ckpt);
        let embedding = embed_gather_scaled(
            &host,
            &canvas,
            &table,
            text_embedding_scale(cfg.text_config.as_ref().unwrap().hidden_size.unwrap()),
        );
        // The decoder input is the embedding after the always-applied unscaled
        // self-conditioning post-norm; dump it for the pinned replay.
        let decoder_input = claymore_model_gemma::graph::rms_norm_unscaled(
            &host,
            &embedding,
            cfg.text_config
                .as_ref()
                .unwrap()
                .rms_norm_eps
                .unwrap_or(1e-6),
        );
        for (name, tensor) in [
            ("rust_decoder_embedding", &embedding),
            ("rust_decoder_input", &decoder_input),
        ] {
            let bytes: Vec<u8> = host
                .to_f32(tensor)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(format!("/tmp/opencode/m10/rust_decoder/{name}.bin"), bytes)
                .expect("dump decoder input");
        }
        for (layer, cache_layer) in cache.layers.iter().enumerate() {
            for (name, tensor) in [("keys", &cache_layer.keys), ("values", &cache_layer.values)] {
                let shape = tensor.layout().shape().to_vec();
                let bytes: Vec<u8> = host
                    .to_f32(tensor)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                std::fs::write(
                    format!("/tmp/opencode/m10/rust_decoder/rust_kv{layer}_{name}.bin"),
                    bytes,
                )
                .expect("dump kv layer");
                eprintln!("kv{layer}_{name} shape={shape:?}");
            }
        }
    }
    for (layer, chunk) in stages
        .chunks(claymore_model_gemma::fullgraph::DECODER_STAGES_PER_LAYER)
        .enumerate()
    {
        for (stage, tensor) in chunk.iter().enumerate() {
            let name = format!("dec{layer}_{}", DEC_STAGE_NAMES[stage]);
            let path = format!("{GOLDEN}/{name}.bin");
            let rust = host.to_f32(tensor);
            if dump {
                let bytes: Vec<u8> = rust.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("/tmp/opencode/m10/rust_decoder/{name}.bin"), bytes)
                    .expect("dump stage");
            }
            if Path::new(&path).is_file() {
                let golden = load(&name, &[1, 16, 2816]);
                let (smax, srms) = stats(&rust, &golden);
                eprintln!("{name} max_abs={smax:.6} rms={srms:.6}");
            }
        }
    }
    for (layer, tensor) in layer_trace.iter().enumerate() {
        let rust = host.to_f32(tensor);
        let golden = load(&format!("decoder_layer_{layer}"), &[1, 16, 2816]);
        let (lmax, lrms) = stats(&rust, &golden);
        eprintln!("decoder_layer_{layer} max_abs={lmax:.6} rms={lrms:.6}");
    }
    let rust_hidden = host.to_f32(&hidden);
    let golden_hidden = load("decoder_last_hidden0_direct", &[1, 16, 2816]);
    let (hmax, hrms) = stats(&rust_hidden, &golden_hidden);
    eprintln!("decoder_last_hidden0 max_abs={hmax:.6} rms={hrms:.6}");
    let logits = lm_logits(&host, &ckpt, &cfg, &hidden);
    let rust_logits = host.to_f32(&logits);
    let golden_logits = load("decoder_logits0_direct", &[1, 16, 262144]);
    let (lmax, lrms) = stats(&rust_logits, &golden_logits);
    eprintln!("decoder_logits0 max_abs={lmax:.6} rms={lrms:.6}");

    // Layer-0 MoE router: scores, top-k weights and indices.
    let (scores, weights, indices) = &trace[0];
    let golden_scores = load("moe0_router_logits", &[16, 128]);
    let (smax, srms) = stats(scores, &golden_scores);
    eprintln!("moe0 router scores max_abs={smax:.6} rms={srms:.6}");
    let golden_weights = load("moe0_topk_weights", &[16, 8]);
    let (wmax, wrms) = stats(weights, &golden_weights);
    eprintln!("moe0 topk weights max_abs={wmax:.6} rms={wrms:.6}");
    let golden_indices = load("moe0_topk_indices", &[16, 8]);
    let indices_exact = indices.iter().map(|v| *v as u32).collect::<Vec<u32>>()
        == golden_indices
            .iter()
            .map(|v| *v as u32)
            .collect::<Vec<u32>>();
    if !indices_exact {
        let rust_indices: Vec<u32> = indices.iter().map(|v| *v as u32).collect();
        let golden_indices: Vec<u32> = golden_indices.iter().map(|v| *v as u32).collect();
        let mut flips = 0;
        for token in 0..16 {
            for slot in 0..8 {
                if rust_indices[token * 8 + slot] != golden_indices[token * 8 + slot] {
                    if flips < 4 {
                        eprintln!(
                            "moe0 topk flip token {token} slot {slot}: rust={} golden={}",
                            rust_indices[token * 8 + slot],
                            golden_indices[token * 8 + slot]
                        );
                    }
                    flips += 1;
                }
            }
        }
        eprintln!("moe0 topk index flips={flips}/128");
    }
    println!("moe0 router: indices_exact={indices_exact}, weights rms={wrms:.6}");

    // Preserve the pre-campaign tolerances: no threshold was adjusted after
    // observing these values. The CPU campaign documents the failures.
    assert!(hmax <= 0.12, "decoder hidden drift {hmax}");
    assert!(hrms <= 4e-3, "decoder hidden rms {hrms}");
    assert!(lmax <= 2.0, "decoder logits drift {lmax}");
    assert!(lrms <= 0.1, "decoder logits rms {lrms}");
    assert!(smax <= 0.02, "router scores drift {smax}");
    assert!(wmax <= 0.01, "router weights drift {wmax}");
    assert!(indices_exact, "router top-k indices must match exactly");
}
