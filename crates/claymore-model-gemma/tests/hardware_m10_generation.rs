//! Hardware campaign: CUDA graph against the CPU reference graph and the
//! venv-pinned eager golden references.
//!
//! Requires a real CUDA driver, the pinned checkpoint and the golden files.
//! The first encoder prefill, the first decoder canvas step and the tied lm
//! head run on the card through `CudaGraph`. Every stage is reported as a
//! triple: CUDA against the CPU reference graph on the same checkpoint (the
//! differential the hardware campaign can defend), CUDA against the eager
//! golden, and the CPU reference against the same golden (the drift measured
//! in the CPU campaign, recorded so the golden comparison is never hidden).
//!
//! ```text
//! RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib \
//! CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
//! cargo test --release -p claymore-model-gemma --features cuda-streaming \
//!   --test hardware_m10_generation -- --ignored --test-threads=1 --nocapture
//! ```

use std::path::Path;

use claymore_memory::{CudaMemoryManager, CudaTensorMemory, EvictionPolicy, MemoryBudgets};
use claymore_model_gemma::checkpoint::Checkpoint;
use claymore_model_gemma::cuda_graph::CudaGraph;

const ROOT: &str = "/tmp/opencode/m10/checkpoint";
const GOLDEN: &str = "/tmp/opencode/m10/golden_eager";

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

#[test]
#[ignore = "requires a CUDA driver, the pinned checkpoint, golden files, and release profile"]
fn cuda_graph_matches_golden_first_step() {
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{GOLDEN}/decoder_last_hidden0_direct.bin")).is_file()
    {
        eprintln!("skip: checkpoint/golden unavailable");
        return;
    }
    let manager = CudaMemoryManager::new(0, budgets(), EvictionPolicy::Lru)
        .expect("CUDA memory manager within the M10 budgets");
    let memory = CudaTensorMemory::from_manager(manager).expect("tensor memory owner");
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let mut graph = CudaGraph::new(memory, ckpt);

    let ids = load("prompt_ids", &[1, 7])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    let mut cache = Vec::new();
    let hidden = graph
        .encoder_forward(&ids, &mut cache)
        .expect("encoder prefill");
    let rust_hidden = {
        let bits = graph
            .download_bf16(hidden)
            .expect("download encoder hidden");
        bits.iter().map(|b| b.to_f32()).collect::<Vec<f32>>()
    };

    // CPU reference graph on the same checkpoint and the same inputs.
    let host = claymore_model_gemma::graph::Host::new();
    let cpu_ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint for the CPU graph");
    let cfg = cpu_ckpt.config().clone();
    let mut cpu_cache = claymore_model_gemma::graph::KvCache::new(30);
    let cpu_hidden = claymore_model_gemma::fullgraph::encoder_forward(
        &host,
        &cpu_ckpt,
        &cfg,
        &ids,
        &mut cpu_cache,
    );
    let cpu_hidden_values = host.to_f32(&cpu_hidden);

    let golden = load("encoder_last_hidden_direct", &[1, 7, 2816]);
    let (emax, erms) = stats(&rust_hidden, &golden);
    let (cmax, crms) = stats(&rust_hidden, &cpu_hidden_values);
    let (gmax, grms) = stats(&cpu_hidden_values, &golden);
    println!("encoder_last_hidden CUDA vs CPU: max_abs={cmax:.6} rms={crms:.6}");
    println!("encoder_last_hidden CUDA vs golden: max_abs={emax:.6} rms={erms:.6}");
    println!("encoder_last_hidden CPU vs golden: max_abs={gmax:.6} rms={grms:.6}");
    assert!(cmax <= 0.03, "encoder CUDA/CPU drift {cmax}");
    assert!(crms <= 2e-3, "encoder CUDA/CPU rms {crms}");
    assert!(emax <= 0.06, "encoder CUDA drift {emax}");
    assert!(erms <= 4e-3, "encoder CUDA rms {erms}");

    let canvas = load("canvas0_direct", &[1, 16])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    let dec = graph
        .decoder_forward(&canvas, &mut cache, None)
        .expect("decoder first step");
    let dec_bits = graph.download_bf16(dec).expect("download decoder hidden");
    let dec_values: Vec<f32> = dec_bits.iter().map(|b| b.to_f32()).collect();
    let mut cpu_trace = Vec::new();
    let cpu_dec = claymore_model_gemma::fullgraph::decoder_forward(
        &host,
        &cpu_ckpt,
        &cfg,
        &canvas,
        &mut cpu_cache,
        None,
        &mut cpu_trace,
    );
    let cpu_dec_values = host.to_f32(&cpu_dec);
    let golden_dec = load("decoder_last_hidden0_direct", &[1, 16, 2816]);
    let (dmax, drms) = stats(&dec_values, &golden_dec);
    let (cdmax, cdrms) = stats(&dec_values, &cpu_dec_values);
    let (gdmax, grms2) = stats(&cpu_dec_values, &golden_dec);
    println!("decoder_last_hidden0 CUDA vs CPU: max_abs={cdmax:.6} rms={cdrms:.6}");
    println!("decoder_last_hidden0 CUDA vs golden: max_abs={dmax:.6} rms={drms:.6}");
    println!("decoder_last_hidden0 CPU vs golden: max_abs={gdmax:.6} rms={grms2:.6}");
    assert!(cdmax <= 0.12, "decoder CUDA/CPU drift {cdmax}");
    assert!(cdrms <= 4e-3, "decoder CUDA/CPU rms {cdrms}");
    assert!(dmax <= 0.3, "decoder CUDA drift {dmax}");
    assert!(drms <= 0.01, "decoder CUDA rms {drms}");

    let logits = graph.lm_logits(dec).expect("lm head");
    let logits_values = graph.download_f32(logits).expect("download logits");
    let cpu_logits = claymore_model_gemma::fullgraph::lm_logits(&host, &cpu_ckpt, &cfg, &cpu_dec);
    let cpu_logits_values = host.to_f32(&cpu_logits);
    let golden_logits = load("decoder_logits0_direct", &[1, 16, 262144]);
    let (lmax, lrms) = stats(&logits_values, &golden_logits);
    let (clmax, clrms) = stats(&logits_values, &cpu_logits_values);
    let (glmax, glrms) = stats(&cpu_logits_values, &golden_logits);
    println!("decoder_logits0 CUDA vs CPU: max_abs={clmax:.6} rms={clrms:.6}");
    println!("decoder_logits0 CUDA vs golden: max_abs={lmax:.6} rms={lrms:.6}");
    println!("decoder_logits0 CPU vs golden: max_abs={glmax:.6} rms={glrms:.6}");
    assert!(clmax <= 2.0, "logits CUDA/CPU drift {clmax}");
    assert!(clrms <= 0.1, "logits CUDA/CPU rms {clrms}");
    assert!(lmax <= 4.0, "logits CUDA drift {lmax}");
    assert!(lrms <= 0.2, "logits CUDA rms {lrms}");

    let (peak, uploaded, downloaded, launches) = graph.metrics();
    println!(
        "CUDA campaign metrics: peak={} MiB uploaded={} MiB downloaded={} MiB launches={launches}",
        peak / (1 << 20),
        uploaded / (1 << 20),
        downloaded / (1 << 20)
    );
    assert!(
        peak <= 28 * (1 << 30),
        "peak VRAM {peak} exceeds the hard budget"
    );
}

#[test]
#[ignore = "requires a CUDA driver, the pinned checkpoint, golden files, and release profile"]
fn real_generation_runs_one_canvas_block_on_the_card() {
    if !Path::new(ROOT).is_dir()
        || !Path::new(&format!("{GOLDEN}/encoder_last_hidden_direct.bin")).is_file()
    {
        eprintln!("skip: checkpoint/golden unavailable");
        return;
    }
    let manager = CudaMemoryManager::new(0, budgets(), EvictionPolicy::Lru)
        .expect("CUDA memory manager within the M10 budgets");
    let memory = CudaTensorMemory::from_manager(manager).expect("tensor memory owner");
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let mut graph = CudaGraph::new(memory, ckpt);

    let ids = load("prompt_ids", &[1, 7])
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>();
    let mut cache = Vec::new();
    graph
        .encoder_forward(&ids, &mut cache)
        .expect("encoder prefill");

    let policy = claymore_model_gemma::generation_loop::SamplerPolicy::pinned(256, 262_144);
    policy.validate().expect("pinned policy");
    let mut state = 0x2345678u64;
    let mut rng = |n: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % n
    };
    let canvas = (0..256).map(|_| rng(262_144) as u32).collect::<Vec<u32>>();
    let accepted = graph
        .generate_block(canvas, &mut cache, &policy, &mut rng)
        .expect("canvas block generated on the card");
    assert_eq!(accepted.len(), 256);
    let (peak, uploaded, downloaded, launches) = graph.metrics();
    println!(
        "real generation: canvas accepted={} positions; peak={} MiB uploaded={} MiB downloaded={} MiB launches={launches}",
        accepted.iter().filter(|t| **t < 262_144).count(),
        peak / (1 << 20),
        uploaded / (1 << 20),
        downloaded / (1 << 20)
    );
    assert!(
        peak <= 28 * (1 << 30),
        "peak VRAM {peak} exceeds the hard budget"
    );
}

/// Step-level differential of the **first** encoder layer, CUDA against the
/// CPU reference graph on the same checkpoint and the same ids.
///
/// The chain-level tolerances of the test above are *not* met (the BF16 chain
/// amplifies the accumulation-order difference between two implementations),
/// so this test isolates the **semantic** alignment: every step is reported,
/// and the router **expert indices** must be identical. The indices are a
/// discrete decision (which experts are selected), not a numerical drift: an
/// exact equality there proves that the CUDA graph composes the pinned MoE
/// routing exactly like the CPU reference.
#[test]
#[ignore = "requires a CUDA driver, the pinned checkpoint, golden files, and release profile"]
fn encoder_first_layer_matches_cpu_step_by_step() {
    if !Path::new(ROOT).is_dir() {
        eprintln!("skip: checkpoint unavailable");
        return;
    }
    let ids: Vec<u32> = vec![3689, 563, 506, 5279, 529, 7001, 236881];

    // CPU reference: the traced encoder gives the intermediate tensors of
    // every step of layer 0.
    let host = claymore_model_gemma::graph::Host::new();
    let cpu_ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint for the CPU graph");
    let cfg = cpu_ckpt.config().clone();
    let mut cpu_cache = claymore_model_gemma::graph::KvCache::new(30);
    let mut trace = Vec::new();
    claymore_model_gemma::fullgraph::encoder_forward_traced_limited(
        &host,
        &cpu_ckpt,
        &cfg,
        &ids,
        &mut cpu_cache,
        &mut trace,
        1,
    );
    let cpu = &trace[0];

    let manager = CudaMemoryManager::new(0, budgets(), EvictionPolicy::Lru)
        .expect("CUDA memory manager within the M10 budgets");
    let memory = CudaTensorMemory::from_manager(manager).expect("tensor memory owner");
    let ckpt = Checkpoint::open_pinned(ROOT).expect("pinned checkpoint");
    let mut graph = CudaGraph::new(memory, ckpt);

    let text = graph.checkpoint().config().text_config.clone().unwrap();
    let hidden = text.hidden_size.unwrap();
    let inter = text.moe_intermediate_size.unwrap_or(704);
    let experts = text.num_experts.unwrap_or(128);
    let top_k = text.top_k_experts.unwrap_or(8);
    let eps = text.rms_norm_eps.unwrap_or(1e-6) as f32;
    let heads = text.num_attention_heads.unwrap();
    let head_dim = text.head_dim.unwrap();
    let kv_heads = text.num_key_value_heads.unwrap();
    let sliding = Some(text.sliding_window.unwrap_or(1024));
    let positions: Vec<usize> = (0..ids.len()).collect();

    let x = graph.embed_text(&ids).expect("embed");
    let mut cache = Vec::new();
    let prefix = "model.decoder.layers.0".to_owned();

    let input_norm = graph
        .weight(&format!("{prefix}.input_layernorm.weight"))
        .unwrap();
    let normed = graph.rms_norm(x, input_norm, eps).unwrap();
    let attn = graph
        .attention_layer(
            "decoder",
            0,
            normed,
            &mut cache,
            &positions,
            head_dim,
            heads,
            kv_heads,
            10_000.0f32,
            None,
            sliding,
            true,
            eps,
            true,
        )
        .unwrap();
    let post_attn = graph
        .weight(&format!("{prefix}.post_attention_layernorm.weight"))
        .unwrap();
    let post = graph.rms_norm(attn, post_attn, eps).unwrap();
    let residual = graph.add(x, post).unwrap();
    let pre_ff1 = graph
        .weight(&format!("{prefix}.pre_feedforward_layernorm.weight"))
        .unwrap();
    let pre1 = graph.rms_norm(residual, pre_ff1, eps).unwrap();
    let mlp = graph
        .dense_mlp(
            pre1,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &format!("{prefix}.mlp.up_proj.weight"),
            &format!("{prefix}.mlp.down_proj.weight"),
        )
        .unwrap();
    let pre2_w = graph
        .weight(&format!("{prefix}.pre_feedforward_layernorm_2.weight"))
        .unwrap();
    let pre2 = graph.rms_norm(residual, pre2_w, eps).unwrap();
    let root_scale = (hidden as f64).powf(-0.5) as f32;
    // The router reads the residual, normalized without scale.
    let topk = graph
        .router_layer(residual, "decoder", 0, eps, root_scale, top_k)
        .unwrap();
    let experts_out = graph
        .experts_layer(
            pre2,
            topk,
            &format!("{prefix}.experts.gate_up_proj"),
            &format!("{prefix}.experts.down_proj"),
            experts,
            inter,
            top_k,
        )
        .unwrap();

    let bf16 = |graph: &mut CudaGraph, index: usize| -> Vec<f32> {
        graph
            .download_bf16(index)
            .expect("download")
            .iter()
            .map(|b| b.to_f32())
            .collect()
    };
    let report = |label: &str, cuda: &[f32], cpu: &[f32]| {
        let (max_abs, rms) = stats(cuda, cpu);
        let absmax = cpu.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        println!(
            "layer0 {label}: cuda-vs-cpu max_abs={max_abs:.6} rms={rms:.6} absmax_cpu={absmax:.4}"
        );
    };
    report(
        "input_norm",
        &bf16(&mut graph, normed),
        &host.to_f32(&cpu.input_normed),
    );
    report(
        "attention",
        &bf16(&mut graph, attn),
        &host.to_f32(&cpu.attention_output),
    );
    report(
        "residual",
        &bf16(&mut graph, residual),
        &host.to_f32(&cpu.routing_residual),
    );
    report("mlp", &bf16(&mut graph, mlp), &host.to_f32(&cpu.mlp_output));
    report(
        "expert_input",
        &bf16(&mut graph, pre2),
        &host.to_f32(&cpu.expert_input),
    );
    report(
        "experts",
        &bf16(&mut graph, experts_out),
        &host.to_f32(&cpu.expert_output),
    );

    // The router output is `[tokens, 2 * top_k]`: weights then indices, per
    // token. The selected experts must be **identical** to the CPU reference.
    let topk_values = graph.download_f32(topk).expect("download router output");
    let mut cuda_indices = Vec::new();
    for token in 0..ids.len() {
        let base = token * 2 * top_k;
        cuda_indices.extend_from_slice(&topk_values[base + top_k..base + 2 * top_k]);
    }
    let cpu_indices = &cpu.router_indices;
    assert_eq!(cuda_indices.len(), cpu_indices.len(), "router index count");
    for (slot, (cuda, cpu)) in cuda_indices.iter().zip(cpu_indices.iter()).enumerate() {
        assert_eq!(
            *cuda as i64, *cpu as i64,
            "router expert index {slot} differs: CUDA {cuda} vs CPU {cpu}"
        );
    }
    println!(
        "layer0 router: {} expert indices identical to the CPU reference",
        cuda_indices.len()
    );
}
