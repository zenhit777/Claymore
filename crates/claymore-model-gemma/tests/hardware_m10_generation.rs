//! Hardware campaign: CUDA graph against the venv-pinned golden references.
//!
//! Requires a real CUDA driver, the pinned checkpoint and the golden files.
//! The first encoder prefill, the first decoder canvas step and the tied lm
//! head run on the card through `CudaGraph`; the results are compared
//! against the golden tensors captured with the venv-pinned upstream code.
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
    let golden = load("encoder_last_hidden_direct", &[1, 7, 2816]);
    let (emax, erms) = stats(&rust_hidden, &golden);
    println!("CUDA encoder_last_hidden: max_abs={emax:.6} rms={erms:.6}");
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
    let golden_dec = load("decoder_last_hidden0_direct", &[1, 16, 2816]);
    let (dmax, drms) = stats(&dec_values, &golden_dec);
    println!("CUDA decoder_last_hidden0: max_abs={dmax:.6} rms={drms:.6}");
    assert!(dmax <= 0.3, "decoder CUDA drift {dmax}");
    assert!(drms <= 0.01, "decoder CUDA rms {drms}");

    let logits = graph.lm_logits(dec).expect("lm head");
    let logits_values = graph.download_f32(logits).expect("download logits");
    let golden_logits = load("decoder_logits0_direct", &[1, 16, 262144]);
    let (lmax, lrms) = stats(&logits_values, &golden_logits);
    println!("CUDA decoder_logits0: max_abs={lmax:.6} rms={lrms:.6}");
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
