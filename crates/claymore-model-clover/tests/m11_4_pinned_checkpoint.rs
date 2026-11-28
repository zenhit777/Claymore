//! M11.4 integration against the pinned Clover-35B-A3B-Preview checkpoint.
//!
//! Requires the authentic release artifacts. Point `CLAYMORE_M11_CHECKPOINT`
//! at a directory holding `config.json`, `model.safetensors.index.json`, the
//! 26 model shards, and the two GDN sidecars. Nothing in this test downloads
//! data; it never writes to the checkpoint directory.
//!
//! Run locally with:
//!   CLAYMORE_M11_CHECKPOINT=/path/to/Clover-35B-A3B-Preview-4b0402b \
//!       cargo test -p claymore-model-clover --test m11_4_pinned_checkpoint -- --ignored --nocapture
//! Set `CLAYMORE_M11_VERIFY_SHA=1` to additionally re-hash all 28 artifacts
//! (a full pass takes about 55 minutes on a warm cache).

use std::env;
use std::path::PathBuf;

use claymore_model_clover::{
    CloverCheckpointInventory, CloverConfig, CloverCpuBackend, CloverCpuBackendOptions,
    CloverDiffusionExecutor, CloverGenerationOptions, CloverGraph, CloverInvalidTokenPolicy,
    CONFIG_SHA256, INDEX_SHA256,
};
use claymore_runtime::diffusion::{
    DiffusionOrchestrator, DiffusionRunOutcome, DiffusionSessionState,
};
use claymore_runtime::CancellationToken;
use claymore_scheduler::{SchedulerConfig, TaskPriority};

const PINNED_CONFIG_JSON: &str = include_str!("../../../docs/audits/fixtures/clover-config.json");
const MAX_CHUNK_BYTES: usize = 8 * 1024 * 1024;

fn checkpoint_root() -> Option<PathBuf> {
    env::var_os("CLAYMORE_M11_CHECKPOINT").map(PathBuf::from)
}

#[test]
#[ignore = "requires the authentic pinned Clover checkpoint (set CLAYMORE_M11_CHECKPOINT)"]
fn pinned_checkpoint_preview_and_causal_forward_preserve_state_contracts() {
    let root = checkpoint_root().expect("CLAYMORE_M11_CHECKPOINT must point at the release dir");
    let config = CloverConfig::from_json(PINNED_CONFIG_JSON).unwrap();
    config.validate_pinned().unwrap();
    assert_eq!(config.vocab_size, 248_320);

    let inventory = CloverCheckpointInventory::inspect_pinned(&root, &config)
        .expect("pinned inventory inspection failed");
    inventory.validate_pinned_tensor_schema().unwrap();
    let identity = inventory.identity();
    assert_eq!(identity.index_sha256, INDEX_SHA256);
    assert_eq!(identity.shard_count, 26);
    assert_eq!(identity.sidecar_count, 2);
    assert_eq!(identity.tensor_count, 813);
    assert_eq!(identity.parameter_count, 34_661_350_848);
    assert_eq!(identity.payload_bytes, 69_322_701_696);
    assert_eq!(inventory.root(), root.as_path());

    let plan = inventory.load_plan().unwrap();
    assert_eq!(plan.shards.len(), 28);
    assert_eq!(plan.total_payload_bytes, identity.payload_bytes);
    assert!(plan.max_chunk_bytes <= MAX_CHUNK_BYTES);
    let mut chunk_count = 0usize;
    for shard in &plan.shards {
        for tensor in &shard.tensors {
            for chunk in &tensor.chunks {
                assert!(chunk.length_bytes <= MAX_CHUNK_BYTES);
                chunk_count += 1;
            }
        }
    }
    assert_eq!(chunk_count, 8_845);

    // The config digest on disk must match the pinned release digest.
    let config_bytes = std::fs::read(root.join("config.json")).unwrap();
    let digest = {
        use claymore_checkpoint::sha256::{hex, sha256};
        hex(&sha256(&config_bytes))
    };
    assert_eq!(digest, CONFIG_SHA256);

    // Backend construction must succeed with a bounded expert budget: only
    // the routed experts of the preview tokens are ever streamed.
    let expert_bytes = (3 * config.moe_intermediate_size * config.hidden_size) as u64 * 2;
    let hard = expert_bytes * 32;
    let budgets = claymore_memory::MemoryBudgets {
        vram_soft_bytes: hard,
        vram_hard_bytes: hard,
        host_bytes: hard,
        pinned_host_bytes: hard,
        expert_cache_target_bytes: hard,
        expert_cache_max_bytes: hard,
    };
    let options = CloverCpuBackendOptions {
        expert_budgets: budgets,
        eviction: claymore_memory::EvictionPolicy::Lru,
        cancellation: Some(CancellationToken::new()),
    };
    let mut backend = CloverCpuBackend::new(&config, inventory, options).unwrap();

    let first_linear = config
        .layer_types
        .iter()
        .position(|kind| *kind == claymore_model_clover::LayerType::LinearAttention)
        .unwrap();
    let first_attention = config
        .layer_types
        .iter()
        .position(|kind| *kind == claymore_model_clover::LayerType::FullAttention)
        .unwrap();
    let gdn_before = backend.gdn_state(first_linear).unwrap();
    let kv_tokens_before = backend.cached_kv_tokens(first_attention);
    assert_eq!(kv_tokens_before, 0);

    let graph = CloverGraph::new(config.clone()).unwrap();
    let prompt: Vec<u32> = vec![151_644, 872, 198, 9_707];
    let output = graph
        .forward_preview(&mut backend, &prompt, &[true; 4], &[], &[0, 1, 2, 3])
        .unwrap();
    let logits = output.logits.expect("preview mode must emit logits");
    assert_eq!(logits.len(), 4 * config.vocab_size);
    assert!(
        logits.iter().all(|value| value.is_finite()),
        "preview logits are not all finite"
    );

    // The preview must leave every clean state component bitwise identical.
    assert_eq!(backend.gdn_state(first_linear).unwrap(), gdn_before);
    assert_eq!(backend.gdn_committed_tokens(first_linear), 0);
    assert_eq!(backend.cached_kv_tokens(first_attention), 0);
    let (keys, values) = backend.kv_cache(first_attention);
    assert!(keys.is_empty() && values.is_empty());

    // A real-weight causal prefill must advance every recurrent/KV layer
    // exactly once. This complements the read-only preview assertion above:
    // both paths now traverse the complete pinned graph and real MoE weights.
    let causal = graph
        .forward_causal(&mut backend, &prompt, &[true; 4], &[], &[0, 1, 2, 3])
        .unwrap();
    assert!(causal.logits.is_none(), "causal mode must skip the LM head");
    assert_eq!(causal.hidden.tokens(), prompt.len());
    assert_eq!(causal.hidden.hidden_size(), config.hidden_size);
    for (layer, kind) in config.layer_types.iter().enumerate() {
        match kind {
            claymore_model_clover::LayerType::LinearAttention => {
                assert_eq!(backend.gdn_committed_tokens(layer), prompt.len());
                let (conv, recurrent) = backend.gdn_state(layer).unwrap();
                assert!(conv.iter().any(|value| *value != 0.0));
                assert!(recurrent.iter().any(|value| *value != 0.0));
            }
            claymore_model_clover::LayerType::FullAttention => {
                assert_eq!(backend.cached_kv_tokens(layer), prompt.len());
                let (keys, values) = backend.kv_cache(layer);
                assert_eq!(
                    keys.len(),
                    prompt.len() * config.num_key_value_heads * config.head_dim
                );
                assert_eq!(values.len(), keys.len());
            }
        }
    }

    if env::var("CLAYMORE_M11_VERIFY_SHA").as_deref() == Ok("1") {
        // Full-artifact SHA-256 verification (~55 min); off by default.
        let inventory = CloverCheckpointInventory::inspect_pinned(&root, &config).unwrap();
        inventory.verify_pinned_asset_hashes().unwrap();
    } else {
        eprintln!("artifact_sha256=not_requested (set CLAYMORE_M11_VERIFY_SHA=1)");
    }
}

#[test]
#[ignore = "requires the authentic pinned Clover checkpoint (set CLAYMORE_M11_CHECKPOINT)"]
fn pinned_checkpoint_runs_multiblock_lowconfidence_generation_on_real_weights() {
    let root = checkpoint_root().expect("CLAYMORE_M11_CHECKPOINT must point at the release dir");
    let config = CloverConfig::from_json(PINNED_CONFIG_JSON).unwrap();
    config.validate_pinned().unwrap();
    let inventory = CloverCheckpointInventory::inspect_pinned(&root, &config)
        .expect("pinned inventory inspection failed");
    inventory.validate_pinned_tensor_schema().unwrap();

    let expert_bytes = (3 * config.moe_intermediate_size * config.hidden_size) as u64 * 2;
    let hard = expert_bytes * 32;
    let backend = CloverCpuBackend::new(
        &config,
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
    .unwrap();

    // The pinned tokenizer exposes 248,077 IDs; the remaining 243 model-head
    // rows have no tokenizer mapping and are masked explicitly for this run.
    let tokenizer_vocab_size = 248_077;
    let mut generation = CloverGenerationOptions::greedy(3, tokenizer_vocab_size);
    generation.block_size = 2;
    generation.threshold = 0.0;
    generation.invalid_token_policy = Some(CloverInvalidTokenPolicy::Mask);
    generation.eos_token_ids = Some(Vec::new());

    let prompt = vec![151_644, 872, 198, 9_707];
    let mut executor = CloverDiffusionExecutor::new(
        CloverGraph::new(config.clone()).unwrap(),
        backend,
        prompt.clone(),
        generation,
    )
    .unwrap();
    let plan = executor.plan().diffusion_plan().clone();
    assert_eq!(plan.block_count(), 2);

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
    let mut observed_causal_replay = false;

    loop {
        match orchestrator.run_one(executor_key, &mut executor).unwrap() {
            DiffusionRunOutcome::StepCompleted { .. } => {
                if !observed_causal_replay && executor.generated_tokens().len() >= 2 {
                    for (layer, kind) in config.layer_types.iter().enumerate() {
                        match kind {
                            claymore_model_clover::LayerType::LinearAttention => {
                                assert_eq!(
                                    executor.backend().gdn_committed_tokens(layer),
                                    prompt.len() + 2
                                );
                            }
                            claymore_model_clover::LayerType::FullAttention => {
                                assert_eq!(
                                    executor.backend().cached_kv_tokens(layer),
                                    prompt.len() + 2
                                );
                            }
                        }
                    }
                    observed_causal_replay = true;
                }
            }
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            other => panic!("unexpected diffusion outcome: {other:?}"),
        }
    }

    assert!(
        observed_causal_replay,
        "the first resolved block was not replayed causally"
    );
    assert_eq!(
        orchestrator.session(session).unwrap().state,
        DiffusionSessionState::Completed
    );
    let generated = executor.generated_tokens();
    eprintln!("pinned_checkpoint_generated_tokens={generated:?}");
    assert_eq!(generated.len(), 3);
    assert!(generated
        .iter()
        .all(|token| (*token as usize) < tokenizer_vocab_size));
    let output = executor.output();
    assert_eq!(output.prompt_tokens, prompt);
    assert_eq!(output.sequence().len(), output.prompt_tokens.len() + 3);
}
