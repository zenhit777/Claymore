//! M11.4 end-to-end LowConfidence generation on the concrete CPU backend.
//!
//! The scenario drives the real `CloverDiffusionExecutor` against a synthetic
//! pinned-topology checkpoint so the full M9 scheduler loop, block
//! LowConfidence acceptance, causal replay, GDN/KV commit, MoE residency and
//! EOS handling all run together. Greedy generation over the deterministic
//! synthetic weights is fully reproducible, so the expected trajectories are
//! anchored on a reference run rather than re-derived arithmetic.

use std::fs;
use std::path::Path;

use claymore_model_clover::synth_checkpoint::SynthCloverCheckpoint;
use claymore_model_clover::{
    CloverCheckpointInventory, CloverConfig, CloverCpuBackend, CloverCpuBackendOptions,
    CloverDiffusionExecutor, CloverGenerationOptions, CloverGraph, CloverGraphBackend,
    CloverInvalidTokenPolicy,
};
use claymore_runtime::diffusion::{
    DiffusionOrchestrator, DiffusionRunOutcome, DiffusionSessionState,
};
use claymore_runtime::CancellationToken;
use claymore_scheduler::{SchedulerConfig, TaskPriority};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("claymore-m114-{name}"));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_checkpoint(config: &CloverConfig, root: &Path) {
    SynthCloverCheckpoint::new(config).write(root);
}

fn open_backend(config: &CloverConfig, root: &Path, budget_experts: usize) -> CloverCpuBackend {
    let inventory = CloverCheckpointInventory::inspect(root).unwrap();
    let expert_bytes = (config.moe_intermediate_size * 2 * config.hidden_size
        + config.hidden_size * config.moe_intermediate_size)
        * 2;
    let shared_bytes =
        (3 * config.shared_expert_intermediate_size * config.hidden_size + config.hidden_size) * 2;
    let hard = expert_bytes as u64 * budget_experts as u64 + shared_bytes as u64;
    CloverCpuBackend::new(
        config,
        inventory,
        CloverCpuBackendOptions {
            expert_budgets: claymore_memory::MemoryBudgets {
                vram_soft_bytes: hard,
                vram_hard_bytes: hard,
                host_bytes: hard,
                pinned_host_bytes: hard,
                expert_cache_target_bytes: hard,
                expert_cache_max_bytes: hard,
            },
            eviction: claymore_memory::EvictionPolicy::Lru,
            cancellation: Some(CancellationToken::new()),
        },
    )
    .unwrap()
}

fn options(
    max_new_tokens: usize,
    threshold: f32,
    eos: Option<Vec<u32>>,
) -> CloverGenerationOptions {
    CloverGenerationOptions {
        max_new_tokens,
        block_size: 2,
        threshold,
        tokenizer_vocab_size: 8,
        // Mask keeps the synthetic model's out-of-tokenizer output rows out of
        // the argmax; the hard-error Reject path is covered by executor tests.
        invalid_token_policy: Some(CloverInvalidTokenPolicy::Mask),
        eos_token_ids: eos,
        ..CloverGenerationOptions::greedy(max_new_tokens, 8)
    }
}

fn run_inner(
    name: &str,
    config: &CloverConfig,
    prompt: Vec<u32>,
    opts: CloverGenerationOptions,
    stop_early: bool,
) -> (
    std::path::PathBuf,
    CloverDiffusionExecutor<CloverCpuBackend>,
) {
    let root = temp_dir(name);
    write_checkpoint(config, &root);
    let graph = CloverGraph::new(config.clone()).unwrap();
    let backend = open_backend(config, &root, 8);
    let mut executor = CloverDiffusionExecutor::new(graph, backend, prompt, opts).unwrap();
    let plan = executor.plan().diffusion_plan().clone();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let session = orchestrator
        .submit(plan, CancellationToken::new(), TaskPriority::Normal)
        .unwrap();
    loop {
        match orchestrator.run_one(&mut executor).unwrap() {
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            DiffusionRunOutcome::StepCompleted { .. } => {
                if stop_early {
                    return (root, executor);
                }
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }
    let state = orchestrator.session(session).unwrap().state;
    assert!(
        matches!(state, DiffusionSessionState::Completed),
        "session did not complete: {state:?}"
    );
    (root, executor)
}

fn run(
    name: &str,
    config: &CloverConfig,
    prompt: Vec<u32>,
    opts: CloverGenerationOptions,
) -> (
    std::path::PathBuf,
    CloverDiffusionExecutor<CloverCpuBackend>,
) {
    run_inner(name, config, prompt, opts, false)
}

fn tiny_config() -> CloverConfig {
    claymore_model_clover::synth_checkpoint::tiny_clover_config()
}

#[test]
fn low_confidence_accepts_the_first_block_and_eos_truncates_it() {
    let config = tiny_config();
    // Reference trajectory: threshold 0 accepts block 0 in one iteration.
    let (_root, reference) = run("ref", &config, vec![3, 4], options(2, 0.0, Some(vec![6])));
    let trajectory = reference.generated_tokens().to_vec();
    assert_eq!(trajectory.len(), 2);

    // The first generated token, declared as the only terminal id, must stop
    // the run exactly at that token.
    let stop_token = trajectory[0];
    let (_root, stopped) = run(
        "eos",
        &config,
        vec![3, 4],
        options(8, 0.0, Some(vec![stop_token])),
    );
    assert_eq!(stopped.generated_tokens(), &[stop_token]);
    assert_eq!(stopped.output().sequence(), &[3, 4, stop_token]);
    assert!(stopped.backend().cache_metrics().cache_misses >= 1);
}

#[test]
fn strict_threshold_falls_back_through_every_iteration() {
    let config = tiny_config();
    // Terminal id 6 is a valid token the deterministic synthetic trajectory
    // never emits, so these runs always span the full budget.
    let (_root, strict) = run(
        "strict",
        &config,
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
    );
    assert_eq!(strict.generated_tokens().len(), 4);
    assert!(strict
        .generated_tokens()
        .iter()
        .all(|&token| (token as usize) < 8));

    // A lenient run over the same budget completes the same block count;
    // acceptance order differs but the session still commits every slot.
    let (_root, lenient) = run(
        "lenient",
        &config,
        vec![3, 4],
        options(4, 0.0, Some(vec![6])),
    );
    assert_eq!(lenient.generated_tokens().len(), 4);
}

#[test]
fn greedy_trajectory_is_reproducible_across_fresh_backends() {
    let config = tiny_config();
    let (_root_a, first) = run("det-a", &config, vec![3, 4], options(5, 1.0, Some(vec![6])));
    let (_root_b, second) = run("det-b", &config, vec![3, 4], options(5, 1.0, Some(vec![6])));
    assert_eq!(first.generated_tokens(), second.generated_tokens());
    assert_eq!(first.generated_tokens().len(), 5);
}

#[test]
fn prefill_commits_state_and_previews_leave_it_untouched() {
    let config = tiny_config();
    // Stop after the first denoise step: prefill has committed, the block
    // preview must not have touched any clean state.
    let (_root, mut executor) = run_inner(
        "partial",
        &config,
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
        true,
    );
    let prompt_len = 2;
    assert_eq!(executor.backend().gdn_committed_tokens(0), prompt_len);
    assert_eq!(executor.backend().cached_kv_tokens(1), prompt_len);
    let (keys, values) = executor.backend().kv_cache(1);
    let kv_width = config.num_key_value_heads * config.head_dim;
    assert_eq!(keys.len(), prompt_len * kv_width);
    assert_eq!(values.len(), keys.len());
    let gdn_before = executor.backend().gdn_state(0).unwrap();
    // A further preview-only forward leaves every clean state component
    // bitwise identical.
    let backend = executor.backend_mut();
    let activations = backend.embed_tokens(&[5, 6]).unwrap();
    let _ = backend
        .forward_gdn(0, &activations, &[true, false], true)
        .unwrap();
    assert_eq!(backend.gdn_state(0).unwrap(), gdn_before);
    assert_eq!(backend.gdn_committed_tokens(0), prompt_len);
    assert_eq!(backend.cached_kv_tokens(1), prompt_len);
}

#[test]
fn session_completion_clears_backend_state() {
    let config = tiny_config();
    let (_root, executor) = run("reset", &config, vec![3, 4], options(4, 1.0, Some(vec![6])));
    // The executor resets the backend when the session completes so the next
    // session starts from clean state and freed expert residency.
    assert_eq!(executor.backend().gdn_committed_tokens(0), 0);
    assert_eq!(executor.backend().cached_kv_tokens(1), 0);
    assert_eq!(executor.backend().cache_metrics().resident_experts, 0);
}
