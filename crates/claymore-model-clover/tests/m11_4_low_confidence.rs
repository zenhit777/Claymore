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
    CloverDiffusionExecutor, CloverExecutorError, CloverGenerationError, CloverGenerationOptions,
    CloverGraph, CloverGraphBackend, CloverInvalidTokenPolicy,
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
    let executor_key = orchestrator.register_executor();
    let session = orchestrator
        .submit(
            plan,
            CancellationToken::new(),
            TaskPriority::Normal,
            executor_key,
        )
        .unwrap();
    loop {
        match orchestrator.run_one(executor_key, &mut executor).unwrap() {
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

/// P10 (red-first): an adapter that reaches EOS mid-plan short-circuits its
/// remaining nominal steps. The orchestrator must count as committed only the
/// blocks actually committed — never the phantom ones.
#[test]
fn p10_blocks_completed_counts_only_effective_commits() {
    let config = tiny_config();
    let (_root, reference) = run(
        "p10-ref",
        &config,
        vec![3, 4],
        options(2, 0.0, Some(vec![6])),
    );
    let stop_token = reference.generated_tokens()[0];

    let root = temp_dir("p10-eos");
    write_checkpoint(&config, &root);
    let graph = CloverGraph::new(config.clone()).unwrap();
    let backend = open_backend(&config, &root, 8);
    let mut executor = CloverDiffusionExecutor::new(
        graph,
        backend,
        vec![3, 4],
        options(8, 0.0, Some(vec![stop_token])),
    )
    .unwrap();
    let plan = executor.plan().diffusion_plan().clone();
    assert_eq!(plan.block_count(), 4);

    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let executor_key = orchestrator.register_executor();
    let session = orchestrator
        .submit(
            plan,
            CancellationToken::new(),
            TaskPriority::Normal,
            executor_key,
        )
        .unwrap();
    loop {
        match orchestrator.run_one(executor_key, &mut executor).unwrap() {
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            DiffusionRunOutcome::StepCompleted { .. } => {}
            other => panic!("unexpected outcome: {other:?}"),
        }
    }
    assert_eq!(executor.generated_tokens(), &[stop_token]);
    assert_eq!(
        orchestrator.session(session).unwrap().state,
        DiffusionSessionState::Completed
    );
    let metrics = orchestrator.metrics();
    assert_eq!(metrics.blocks_submitted, 4);
    assert_eq!(
        metrics.blocks_completed, 1,
        "a stopped adapter must not turn nominal CommitBlock steps into committed blocks"
    );
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

/// Backend with an explicit cancellation token so a test can cancel the
/// expert streaming path while a session is running.
fn open_backend_with_token(
    config: &CloverConfig,
    root: &Path,
    budget_experts: usize,
    cancellation: CancellationToken,
) -> CloverCpuBackend {
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
            cancellation: Some(cancellation),
        },
    )
    .unwrap()
}

#[test]
fn stop_token_ids_truncate_the_block_on_the_real_backend() {
    let config = tiny_config();
    let (_root, reference) = run(
        "stop-ref",
        &config,
        vec![3, 4],
        options(3, 0.0, Some(vec![6])),
    );
    let trajectory = reference.generated_tokens().to_vec();
    assert_eq!(trajectory.len(), 3);

    // A stop id that is not an EOS id must terminate the run just like EOS.
    let stop = trajectory[0];
    let mut opts = options(8, 0.0, Some(vec![6]));
    opts.stop_token_ids = vec![stop];
    let (_root, stopped) = run("stop-run", &config, vec![3, 4], opts);
    assert_eq!(stopped.generated_tokens(), &[stop]);
    assert_eq!(stopped.output().sequence(), [3, 4, stop]);

    // A stop id the request never emits must not truncate anything.
    let mut never = options(3, 0.0, Some(vec![6]));
    never.stop_token_ids = vec![5];
    let (_root, complete) = run("stop-never", &config, vec![3, 4], never);
    assert_eq!(complete.generated_tokens(), trajectory);
}

#[test]
fn pad_token_id_is_explicit_and_the_mask_row_is_reserved() {
    let config = tiny_config();
    let mut opts = options(2, 0.0, Some(vec![6]));
    opts.pad_token_id = Some(5);
    let (_root, explicit) = run("pad-explicit", &config, vec![3, 4], opts);
    assert_eq!(explicit.output().pad_token_id, 5);

    // Without an explicit PAD the documented fallback chain resolves to the
    // first terminal id, never to an arbitrary row.
    let (_root, fallback) = run(
        "pad-fallback",
        &config,
        vec![3, 4],
        options(2, 0.0, Some(vec![6])),
    );
    assert_eq!(fallback.output().pad_token_id, 6);

    // PAD must stay inside the model vocabulary; it is request metadata and is
    // never invented from an arbitrary row.
    let mut outside = options(2, 0.0, Some(vec![6]));
    outside.pad_token_id = Some(config.vocab_size as u32);
    let outside_root = temp_dir("pad-outside");
    write_checkpoint(&config, &outside_root);
    let error = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &outside_root, 8),
        vec![3, 4],
        outside,
    )
    .err()
    .expect("PAD outside the model vocabulary must be refused");
    assert!(
        matches!(
            error,
            CloverExecutorError::Generation(CloverGenerationError::InvalidTokenId(id))
                if id == config.vocab_size as u32
        ),
        "unexpected PAD error: {error:?}"
    );

    // Terminal ids may not steal the reserved mask row.
    let mut mask_stop = options(2, 0.0, Some(vec![6]));
    mask_stop.stop_token_ids = vec![config.mask_token_id];
    let mask_root = temp_dir("mask-stop");
    write_checkpoint(&config, &mask_root);
    let error = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &mask_root, 8),
        vec![3, 4],
        mask_stop,
    )
    .err()
    .expect("the mask row is reserved and must not be a stop id");
    assert!(
        matches!(
            error,
            CloverExecutorError::Generation(CloverGenerationError::ReservedMaskToken)
        ),
        "unexpected mask error: {error:?}"
    );
}

#[test]
fn unmapped_head_rows_follow_the_explicit_policy_on_the_real_backend() {
    let config = tiny_config();
    // Sampling with a fixed seed is deterministic, so the search below always
    // lands on the same seed: the first one whose resolved block selects a
    // model-head row that the tokenizer cannot address.
    // The mask row must stay addressable, so the widest boundary that still
    // leaves an unmapped row is one row below the model vocabulary.
    let boundary = config.vocab_size - 1;
    let mut sampled = options(2, 0.0, Some(vec![6]));
    sampled.tokenizer_vocab_size = boundary;
    sampled.do_sample = true;
    sampled.temperature = 1.0;
    let mut rejected_seed = None;
    for seed in 0..64_u64 {
        let mut reject_opts = sampled.clone();
        reject_opts.seed = seed;
        reject_opts.invalid_token_policy = Some(CloverInvalidTokenPolicy::Reject);
        if let Some(error) = run_failure("policy-reject", &config, vec![3, 4], reject_opts) {
            assert!(
                error.contains("outside tokenizer-addressable vocabulary"),
                "seed {seed} failed for an unrelated reason: {error}"
            );
            rejected_seed = Some(seed);
            break;
        }
    }
    let seed = rejected_seed.expect("a sampled run must eventually select an unmapped head row");

    // The identical request under Mask completes and stays addressable.
    let mut mask_opts = sampled.clone();
    mask_opts.seed = seed;
    mask_opts.invalid_token_policy = Some(CloverInvalidTokenPolicy::Mask);
    let (_root, masked) = run("policy-mask", &config, vec![3, 4], mask_opts);
    assert_eq!(masked.generated_tokens().len(), 2);
    assert!(
        masked
            .generated_tokens()
            .iter()
            .all(|&token| (token as usize) < boundary),
        "Mask must keep every emitted row inside the tokenizer"
    );
}

/// Drive a session to its first failure and return the step error.
fn run_failure(
    name: &str,
    config: &CloverConfig,
    prompt: Vec<u32>,
    opts: CloverGenerationOptions,
) -> Option<String> {
    let root = temp_dir(name);
    write_checkpoint(config, &root);
    let graph = CloverGraph::new(config.clone()).unwrap();
    let mut executor =
        CloverDiffusionExecutor::new(graph, open_backend(config, &root, 8), prompt, opts).unwrap();
    let plan = executor.plan().diffusion_plan().clone();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let executor_key = orchestrator.register_executor();
    let session = orchestrator
        .submit(
            plan,
            CancellationToken::new(),
            TaskPriority::Normal,
            executor_key,
        )
        .unwrap();
    for _ in 0..64 {
        match orchestrator.run_one(executor_key, &mut executor).unwrap() {
            DiffusionRunOutcome::StepFailed { error, .. } => {
                // A failed session must still release request-local state.
                assert_eq!(
                    orchestrator.session(session).unwrap().state,
                    DiffusionSessionState::Failed
                );
                assert_eq!(executor.backend().gdn_committed_tokens(0), 0);
                assert_eq!(executor.backend().cached_kv_tokens(1), 0);
                assert_eq!(executor.backend().cache_metrics().resident_experts, 0);
                return Some(error);
            }
            DiffusionRunOutcome::SessionCompleted { .. } => return None,
            DiffusionRunOutcome::StepCompleted { .. } => {}
            other => panic!("unexpected outcome: {other:?}"),
        }
    }
    panic!("session did not terminate");
}

#[test]
fn cancellation_during_expert_streaming_is_terminal_and_clears_state() {
    let config = tiny_config();
    let root = temp_dir("cancel-stream");
    write_checkpoint(&config, &root);
    let token = CancellationToken::new();
    let mut executor = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend_with_token(&config, &root, 8, token.clone()),
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
    )
    .unwrap();
    let plan = executor.plan().diffusion_plan().clone();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let executor_key = orchestrator.register_executor();
    let session = orchestrator
        .submit(
            plan,
            CancellationToken::new(),
            TaskPriority::Normal,
            executor_key,
        )
        .unwrap();

    // Let the prefill commit, then cancel the request-local expert stream.
    let outcome = orchestrator.run_one(executor_key, &mut executor).unwrap();
    assert!(matches!(outcome, DiffusionRunOutcome::StepCompleted { .. }));
    assert!(executor.backend().gdn_committed_tokens(0) > 0);
    token.cancel();

    let mut failed = None;
    for _ in 0..64 {
        match orchestrator.run_one(executor_key, &mut executor).unwrap() {
            DiffusionRunOutcome::StepFailed { error, .. } => {
                failed = Some(error);
                break;
            }
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            DiffusionRunOutcome::StepCompleted { .. } => {}
            other => panic!("unexpected outcome: {other:?}"),
        }
    }
    let error = failed.expect("cancelling the expert stream must fail the step");
    assert!(error.contains("cancelled"), "unexpected error: {error}");
    assert_eq!(
        orchestrator.session(session).unwrap().state,
        DiffusionSessionState::Failed
    );
    // The terminal notification still resets the request-local backend.
    assert_eq!(executor.backend().gdn_committed_tokens(0), 0);
    assert_eq!(executor.backend().cached_kv_tokens(1), 0);
    assert_eq!(executor.backend().cache_metrics().resident_experts, 0);
}

#[test]
fn a_session_cancelled_before_its_first_step_runs_no_forward() {
    let config = tiny_config();
    let root = temp_dir("cancel-before");
    write_checkpoint(&config, &root);
    let executor = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &root, 8),
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
    )
    .unwrap();
    let plan = executor.plan().diffusion_plan().clone();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let executor_key = orchestrator.register_executor();
    let token = CancellationToken::new();
    token.cancel();
    let session = orchestrator
        .submit(plan, token, TaskPriority::Normal, executor_key)
        .unwrap();
    assert_eq!(
        orchestrator.session(session).unwrap().state,
        DiffusionSessionState::Cancelled
    );
    assert_eq!(executor.backend().gdn_committed_tokens(0), 0);
    assert_eq!(executor.backend().cached_kv_tokens(1), 0);
    assert!(executor.generated_tokens().is_empty());
}

#[test]
fn expert_stream_io_failure_is_terminal_and_clears_state() {
    let config = tiny_config();
    let root = temp_dir("expert-io");
    write_checkpoint(&config, &root);
    // One routed expert of budget: the prefill routes to more experts than the
    // cache can hold, so at least one routed expert must be re-read later.
    let mut executor = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &root, 1),
        vec![3, 4],
        options(4, 0.0, Some(vec![6])),
    )
    .unwrap();
    let plan = executor.plan().diffusion_plan().clone();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let executor_key = orchestrator.register_executor();
    let session = orchestrator
        .submit(
            plan,
            CancellationToken::new(),
            TaskPriority::Normal,
            executor_key,
        )
        .unwrap();

    // Prefill commits and the routed set already exceeds the one-expert cache,
    // so later blocks cannot be served entirely from residency.
    let outcome = orchestrator.run_one(executor_key, &mut executor).unwrap();
    assert!(matches!(outcome, DiffusionRunOutcome::StepCompleted { .. }));
    assert!(executor.backend().gdn_committed_tokens(0) > 0);
    let misses_before = executor.backend().cache_metrics().cache_misses;
    assert!(misses_before > 0);

    // Destroy the shard the pending expert read must stream from.
    let shard = root.join(claymore_model_clover::synth_checkpoint::SYNTH_MODEL_SHARD);
    fs::write(&shard, []).expect("truncate shard");

    let mut failed = None;
    for _ in 0..64 {
        let misses_before = executor.backend().cache_metrics().cache_misses;
        match orchestrator.run_one(executor_key, &mut executor).unwrap() {
            DiffusionRunOutcome::StepFailed { error, .. } => {
                failed = Some(error);
                break;
            }
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            DiffusionRunOutcome::StepCompleted { .. } => {
                let after = executor.backend().cache_metrics().cache_misses;
                assert_eq!(
                    after, misses_before,
                    "a cold expert read returned success from a destroyed shard"
                );
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }
    let error = failed.expect("a cold expert read from a destroyed shard must fail the step");
    assert!(error.contains("expert"), "unexpected error: {error}");
    assert_eq!(
        orchestrator.session(session).unwrap().state,
        DiffusionSessionState::Failed
    );
    // Terminal notification still resets the request-local backend.
    assert_eq!(executor.backend().gdn_committed_tokens(0), 0);
    assert_eq!(executor.backend().cached_kv_tokens(1), 0);
    assert_eq!(executor.backend().cache_metrics().resident_experts, 0);
}

#[test]
fn complete_preview_blocks_beyond_the_context_are_refused_before_any_forward() {
    let config = tiny_config();
    let root = temp_dir("context");
    write_checkpoint(&config, &root);
    // max_position_embeddings is 32; block size 4 needs whole preview blocks.
    let mut opts = options(32, 1.0, Some(vec![6]));
    opts.block_size = 4;
    let error = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &root, 8),
        vec![3, 4],
        opts,
    )
    .err()
    .expect("prompt plus full preview blocks must exceed the pinned context");
    assert!(
        matches!(error, CloverExecutorError::ContextExceeded),
        "unexpected context error: {error:?}"
    );
}

#[test]
fn concurrent_sessions_stay_isolated_with_independent_resource_budgets() {
    let config = tiny_config();
    let root_a = temp_dir("conc-a");
    let root_b = temp_dir("conc-b");
    write_checkpoint(&config, &root_a);
    write_checkpoint(&config, &root_b);

    let mut first = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &root_a, 8),
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
    )
    .unwrap();
    // The second request runs with the minimum routed budget: eviction is
    // forced on every block, which must not change what it generates.
    let mut second = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &root_b, 1),
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
    )
    .unwrap();
    let mut first_runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let mut second_runtime = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let first_key = first_runtime.register_executor();
    let second_key = second_runtime.register_executor();
    first_runtime
        .submit(
            first.plan().diffusion_plan().clone(),
            CancellationToken::new(),
            TaskPriority::Normal,
            first_key,
        )
        .unwrap();
    second_runtime
        .submit(
            second.plan().diffusion_plan().clone(),
            CancellationToken::new(),
            TaskPriority::Normal,
            second_key,
        )
        .unwrap();

    let mut first_done = false;
    let mut second_done = false;
    while !(first_done && second_done) {
        if !first_done {
            match first_runtime.run_one(first_key, &mut first).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => first_done = true,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected first-session outcome: {other:?}"),
            }
        }
        if !second_done {
            match second_runtime.run_one(second_key, &mut second).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => second_done = true,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected second-session outcome: {other:?}"),
            }
        }
    }

    // Solo reference for the same request.
    let (_root, solo) = run(
        "conc-solo",
        &config,
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
    );
    assert_eq!(first.generated_tokens(), solo.generated_tokens());
    assert_eq!(second.generated_tokens(), solo.generated_tokens());

    // Interleaving must not have disturbed either request's streaming: the
    // pressure-bound request re-reads experts the generous request keeps hot.
    let first_metrics = first.backend().cache_metrics();
    let second_metrics = second.backend().cache_metrics();
    assert_eq!(first_metrics.admission_failures, 0);
    assert_eq!(second_metrics.admission_failures, 0);
    assert!(
        second_metrics.evictions > first_metrics.evictions,
        "the tight-budget request must have evicted more: {first_metrics:?} vs {second_metrics:?}"
    );
    // Completion released each request's residency independently.
    assert_eq!(first_metrics.resident_experts, 0);
    assert_eq!(second_metrics.resident_experts, 0);
    assert_eq!(first.backend().gdn_committed_tokens(0), 0);
    assert_eq!(second.backend().gdn_committed_tokens(0), 0);
}

#[test]
fn incremental_causal_replay_matches_a_single_shot_prefill() {
    let config = tiny_config();
    let root = temp_dir("replay");
    write_checkpoint(&config, &root);
    let prompt = [3_u32, 4];
    let published = [1_u32, 2];

    // Incremental path: causal prefill of the prompt, then a causal replay of
    // the published prefix, exactly as CommitBlock performs it.
    let graph = CloverGraph::new(config.clone()).unwrap();
    let mut incremental = open_backend(&config, &root, 8);
    graph
        .forward_causal(&mut incremental, &prompt, &[true; 2], &[], &[0, 1])
        .unwrap();
    graph
        .forward_causal(
            &mut incremental,
            &published,
            &[true; 2],
            &[true; 2],
            &[2, 3],
        )
        .unwrap();
    let replayed_state = incremental.gdn_state(0).unwrap();
    let (replayed_keys, replayed_values) = incremental.kv_cache(1);
    let replayed_keys = replayed_keys.to_vec();
    let replayed_values = replayed_values.to_vec();
    let replayed_preview = graph
        .forward_preview(
            &mut incremental,
            &[config.mask_token_id; 2],
            &[true; 2],
            &[true; 4],
            &[4, 5],
        )
        .unwrap();

    // Reference path: one causal prefill that already contains the prefix.
    let mut single = open_backend(&config, &root, 8);
    graph
        .forward_causal(&mut single, &[3, 4, 1, 2], &[true; 4], &[], &[0, 1, 2, 3])
        .unwrap();
    let single_state = single.gdn_state(0).unwrap();
    let (single_keys, single_values) = single.kv_cache(1);

    assert_eq!(incremental.gdn_committed_tokens(0), 4);
    assert_eq!(single.gdn_committed_tokens(0), 4);
    assert_eq!(incremental.cached_kv_tokens(1), 4);
    // BF16 rounding of the same FP32 accumulation order is bitwise here; the
    // bound keeps the assertion honest if the kernels ever reorder.
    let bound = 2f32.powi(-7);
    for (name, reference, actual) in [
        ("conv", &replayed_state.0, &single_state.0),
        ("ssm", &replayed_state.1, &single_state.1),
        ("keys", &replayed_keys, &single_keys.to_vec()),
        ("values", &replayed_values, &single_values.to_vec()),
    ] {
        assert_eq!(reference.len(), actual.len(), "{name} length");
        let scale = reference
            .iter()
            .fold(0f32, |acc, &v| acc.max(v.abs()))
            .max(1.0);
        for (index, (&left, &right)) in reference.iter().zip(actual).enumerate() {
            assert!(
                (left - right).abs() <= bound * scale,
                "{name}[{index}]: replay {left} != single-shot {right}"
            );
        }
    }
    // And the preview over the replayed prefix is the same computation as the
    // preview over the single-shot prefix.
    let single_preview = graph
        .forward_preview(
            &mut single,
            &[config.mask_token_id; 2],
            &[true; 2],
            &[true; 4],
            &[4, 5],
        )
        .unwrap();
    let replayed_logits = replayed_preview.logits.unwrap();
    let single_logits = single_preview.logits.unwrap();
    assert_eq!(replayed_logits.len(), single_logits.len());
    let scale = replayed_logits
        .iter()
        .fold(0f32, |acc, &v| acc.max(v.abs()))
        .max(1.0);
    for (index, (&left, &right)) in replayed_logits.iter().zip(&single_logits).enumerate() {
        assert!(
            (left - right).abs() <= bound * scale,
            "preview logit {index}: replay {left} != single-shot {right}"
        );
    }
}

#[test]
fn active_penalties_are_validated_and_executed_end_to_end() {
    let config = tiny_config();
    let mut opts = options(4, 1.0, Some(vec![6]));
    opts.penalties.presence_penalty = 0.5;
    opts.penalties.frequency_penalty = 0.25;
    opts.penalties.repetition_penalty = 1.6;
    let (_root, penalized) = run("pen-active", &config, vec![3, 4], opts.clone());
    assert_eq!(penalized.generated_tokens().len(), 4);

    // Out-of-range penalties are refused before any forward.
    let mut invalid = opts.clone();
    invalid.penalties.presence_penalty = 2.5;
    let root = temp_dir("pen-invalid");
    write_checkpoint(&config, &root);
    let error = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        open_backend(&config, &root, 8),
        vec![3, 4],
        invalid,
    )
    .err()
    .expect("penalties must be validated");
    assert!(
        matches!(
            error,
            CloverExecutorError::Generation(CloverGenerationError::InvalidPenalty(name))
                if name == "presence_penalty"
        ),
        "unexpected penalty error: {error:?}"
    );

    // The same request without penalties still completes, and the active
    // penalties change the resolved trajectory: the penalty path reaches the
    // resolution loop, not only validation.
    let (_root, baseline) = run(
        "pen-baseline",
        &config,
        vec![3, 4],
        options(4, 1.0, Some(vec![6])),
    );
    assert_eq!(baseline.generated_tokens().len(), 4);
    assert_ne!(
        penalized.generated_tokens(),
        baseline.generated_tokens(),
        "active penalties must reach the resolved block on the real backend"
    );
}
