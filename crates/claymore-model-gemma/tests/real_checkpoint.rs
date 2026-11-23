//! Real-checkpoint validation for the pinned DiffusionGemma revision.
//!
//! These tests are opt-in because the audited checkpoint occupies 51.6 GB on
//! disk. Point `CLAYMORE_M10_CHECKPOINT` at a directory holding the pinned
//! revision and run:
//!
//! ```text
//! CLAYMORE_M10_CHECKPOINT=/path/to/checkpoint \
//!   cargo test -p claymore-model-gemma --test real_checkpoint -- --nocapture
//! ```
//!
//! The tests never claim that the full generation graph ran: they validate
//! provenance, the complete tensor inventory and bounded payload reads only.

use claymore_model_gemma::checkpoint::{Checkpoint, CheckpointError};
use claymore_model_gemma::config::PINNED_TENSOR_COUNT;

fn checkpoint_root() -> Option<std::path::PathBuf> {
    std::env::var_os("CLAYMORE_M10_CHECKPOINT").map(std::path::PathBuf::from)
}

#[test]
fn pinned_checkpoint_matches_the_audited_inventory() {
    let Some(root) = checkpoint_root() else {
        eprintln!("skipping: CLAYMORE_M10_CHECKPOINT is not set");
        return;
    };
    let checkpoint = Checkpoint::open_pinned(&root).unwrap_or_else(|error| {
        panic!(
            "failed to open pinned checkpoint at {}: {error}",
            root.display()
        )
    });
    let identity = checkpoint.identity();
    assert_eq!(identity.model_id, "google/diffusiongemma-26B-A4B-it");
    assert_eq!(
        identity.revision,
        "cd40195554d3cfcb56f34b9adfb66efc790ed5b6"
    );
    assert_eq!(identity.shard_count, 11);
    assert_eq!(identity.tensor_count, PINNED_TENSOR_COUNT);
    assert_eq!(identity.total_bytes, 51_647_562_456);
    assert_eq!(
        identity.dtype,
        claymore_model_gemma::safetensors::WeightDType::BF16
    );
    assert_eq!(
        identity.config_sha256,
        "13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506"
    );
    assert_eq!(
        identity.index_sha256,
        "6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a"
    );
    assert_eq!(
        identity.generation_config_sha256,
        "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200"
    );
}

#[test]
fn pinned_checkpoint_rejects_a_missing_shard() {
    let Some(root) = checkpoint_root() else {
        eprintln!("skipping: CLAYMORE_M10_CHECKPOINT is not set");
        return;
    };
    // The index is copied without any shard payload: the missing shard must be
    // detected from the index before a single weight byte is read.
    let temporary =
        std::env::temp_dir().join(format!("claymore-m10-missing-shard-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temporary);
    std::fs::create_dir_all(&temporary).unwrap();
    for file in [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ] {
        std::fs::copy(root.join(file), temporary.join(file)).unwrap();
    }
    let error = Checkpoint::open_pinned(&temporary).unwrap_err();
    assert!(
        matches!(error, CheckpointError::MissingShard { .. }),
        "unexpected error: {error}"
    );
    let _ = std::fs::remove_dir_all(&temporary);
}

#[test]
fn pinned_checkpoint_reads_bounded_real_weight_payloads() {
    let Some(root) = checkpoint_root() else {
        eprintln!("skipping: CLAYMORE_M10_CHECKPOINT is not set");
        return;
    };
    let checkpoint = Checkpoint::open_pinned(&root).unwrap();

    // A small real weight: RMS norm vector (2816 BF16 values).
    let norm = checkpoint
        .read_bf16_bits("model.decoder.layers.0.input_layernorm.weight")
        .unwrap();
    assert_eq!(norm.len(), 2816);

    // A medium real weight: sliding-window Q projection (4096 x 2816 BF16).
    let q = checkpoint
        .read_bf16_bits("model.decoder.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q.len(), 4096 * 2816);
    assert!(q
        .iter()
        .all(|bits| !f32::from_bits(u32::from(*bits) << 16).is_nan()));

    // A large real weight: the embedding matrix (262144 x 2816 BF16).
    let embedding = checkpoint
        .read_bf16_bits("model.decoder.embed_tokens.weight")
        .unwrap();
    assert_eq!(embedding.len(), 262_144 * 2816);
    assert!(embedding
        .iter()
        .all(|bits| !f32::from_bits(u32::from(*bits) << 16).is_nan()));
}
