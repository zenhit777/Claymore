//! Hardware validation of the bounded weight-streaming executor (M10.3).
//!
//! Requires a real CUDA driver and the pinned checkpoint. Every device
//! allocation must go through `CudaMemoryManager`; the test verifies
//! admission, residency, cache hits, LRU eviction under pressure, release
//! safety after completion, and the RAM/VRAM instrumentation.
//!
//! ```text
//! RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib \
//! CLAYMORE_M10_CHECKPOINT=/tmp/opencode/m10/checkpoint \
//! cargo test --release -p claymore-model-gemma --features cuda-streaming \
//!   --test hardware_m10_streaming -- --ignored --test-threads=1 --nocapture
//! ```

use claymore_core::MemoryTier;
use claymore_memory::{EvictionPolicy, MemoryBudgets};
use claymore_model_gemma::config::GemmaModelConfig;
use claymore_model_gemma::stream::{decoder_layer_names, text_prefix_names, WeightStream};

fn checkpoint_root() -> Option<std::path::PathBuf> {
    let root = std::env::var("CLAYMORE_M10_CHECKPOINT")
        .map(std::path::PathBuf::from)
        .ok()?;
    root.is_dir().then_some(root)
}

fn budgets(vram_soft: u64, vram_hard: u64) -> MemoryBudgets {
    MemoryBudgets {
        vram_soft_bytes: vram_soft,
        vram_hard_bytes: vram_hard,
        host_bytes: 32 * 1024 * 1024 * 1024,
        pinned_host_bytes: 4 * 1024 * 1024 * 1024,
        expert_cache_target_bytes: vram_hard / 2,
        expert_cache_max_bytes: vram_hard,
    }
}

#[test]
#[ignore = "requires a CUDA driver, the pinned checkpoint, and release profile"]
fn streaming_layer_plan_streams_through_the_manager() {
    let root = match checkpoint_root() {
        Some(root) => root,
        None => {
            eprintln!("skip: CLAYMORE_M10_CHECKPOINT unavailable");
            return;
        }
    };
    let mut stream = WeightStream::open_pinned(
        root,
        budgets(26 * 1024 * 1024 * 1024, 28 * 1024 * 1024 * 1024),
        EvictionPolicy::Lru,
    )
    .expect("pinned checkpoint open through the streaming executor");
    let cfg = stream.checkpoint().config().clone();
    let names = decoder_layer_names(&cfg, 0);
    let ids = stream
        .ensure_plan(&names, 1)
        .expect("layer 0 plan streamed");
    assert_eq!(ids.len(), names.len());

    let metrics = stream.metrics();
    let total: u64 = names
        .iter()
        .map(|name| {
            stream
                .checkpoint()
                .tensor(name)
                .expect("tensor present")
                .entry
                .byte_len()
        })
        .sum();
    assert_eq!(metrics.transferred_bytes, total, "every byte streamed once");
    assert_eq!(metrics.disk_read_bytes, total, "every payload read once");
    assert_eq!(metrics.loads, names.len() as u64, "one load per tensor");
    assert_eq!(metrics.cache_hits, 0);
    assert!(
        metrics.peak_vram_bytes >= total,
        "peak VRAM must cover the resident plan"
    );
    assert_eq!(
        stream.manager().usage_bytes(MemoryTier::Vram),
        total,
        "residency equals the streamed bytes"
    );
    // Pinned staging buffers are released after the transfer event completes.
    assert_eq!(
        stream.manager().usage_bytes(MemoryTier::PinnedHost),
        0,
        "staging must not stay resident"
    );

    // Cache hits: re-requesting the same plan must not touch disk.
    let again = stream.ensure_plan(&names, 2).expect("cache hits");
    assert_eq!(again.len(), names.len());
    let metrics = stream.metrics();
    assert_eq!(metrics.cache_hits, names.len() as u64);
    assert_eq!(metrics.disk_read_bytes, total, "no reread on cache hits");
    assert_eq!(
        metrics.transferred_bytes, total,
        "no re-transfer on cache hits"
    );
}

#[test]
#[ignore = "requires a CUDA driver, the pinned checkpoint, and release profile"]
fn streaming_evicts_lru_residents_before_admission() {
    let root = match checkpoint_root() {
        Some(root) => root,
        None => {
            eprintln!("skip: CLAYMORE_M10_CHECKPOINT unavailable");
            return;
        }
    };
    // Soft ceiling just above one text prefix (embedding table 1.4 GiB),
    // hard ceiling below two prefixes: the second prefix forces LRU eviction.
    let embedding_bytes = {
        let probe = WeightStream::open_pinned(
            &root,
            budgets(2 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024),
            EvictionPolicy::Lru,
        )
        .expect("probe open");
        let names = text_prefix_names(probe.checkpoint().config());
        let bytes = probe
            .checkpoint()
            .tensor(&names[0])
            .expect("embedding tensor present")
            .entry
            .byte_len();
        drop(probe);
        bytes
    };
    let soft = embedding_bytes + 64 * 1024 * 1024;
    let hard = embedding_bytes * 3 / 2;
    let mut stream = WeightStream::open_pinned(&root, budgets(soft, hard), EvictionPolicy::Lru)
        .expect("tight budgets open");

    let names = text_prefix_names(stream.checkpoint().config());
    let first = names[0].clone();
    stream
        .ensure_tensor(&first, 1)
        .expect("first prefix resident");
    assert!(
        stream.metrics().evictions == 0,
        "the first admission must not evict"
    );

    // A tensor larger than the remaining headroom evicts the oldest resident.
    let big = names
        .iter()
        .find(|name| {
            stream
                .checkpoint()
                .tensor(name)
                .map(|loc| loc.entry.byte_len() > embedding_bytes - 128 * 1024 * 1024)
                .unwrap_or(false)
        })
        .cloned()
        .expect("a prefix tensor larger than the headroom");
    stream
        .ensure_tensor(&big, 2)
        .expect("eviction admits the second tensor");
    assert!(
        stream.metrics().evictions >= 1,
        "LRU eviction must have released the oldest resident"
    );
    assert!(
        stream.manager().usage_bytes(MemoryTier::Vram) <= hard,
        "admission never exceeds the hard ceiling"
    );

    // Release after completion, and the stream closes safely.
    stream.release(&big, 3).expect("release after completion");
    stream.cancel(4).expect("cancel drains residents");
    assert!(stream.is_closed());
    assert!(
        stream.ensure_tensor(&first, 5).is_err(),
        "a closed stream refuses loads"
    );
}
