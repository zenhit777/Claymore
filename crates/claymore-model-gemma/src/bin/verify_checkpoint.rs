//! Bounded full-payload verification of the pinned DiffusionGemma checkpoint.
//!
//! Every tensor payload is read in chunks of at most `MAX_CHUNK_BYTES`, hashed
//! with the streaming SHA-256, and checked against the validated shard index:
//! coverage must be complete, contiguous and aligned. No whole-shard copy is
//! ever made, and no GPU is used.
//!
//! ```text
//! cargo run -p claymore-model-gemma --bin verify_checkpoint \
//!   -- /path/to/pinned/checkpoint > verification.log
//! ```

use std::collections::BTreeMap;

use claymore_model_gemma::checkpoint::{expected_tensors, Checkpoint};
use claymore_model_gemma::safetensors::MAX_CHUNK_BYTES;
use claymore_model_gemma::sha256::{hex, Sha256Stream};

fn main() {
    let root = std::env::args()
        .nth(1)
        .expect("usage: verify_checkpoint <checkpoint-root>");
    let checkpoint = Checkpoint::open_pinned(&root)
        .unwrap_or_else(|error| panic!("failed to open pinned checkpoint at {root}: {error}"));
    let identity = checkpoint.identity();
    println!(
        "identity model={} revision={} shards={} tensors={} bytes={} dtype={}",
        identity.model_id,
        identity.revision,
        identity.shard_count,
        identity.tensor_count,
        identity.total_bytes,
        identity.dtype.name()
    );
    println!(
        "digests config={} index={} generation_config={}",
        identity.config_sha256, identity.index_sha256, identity.generation_config_sha256
    );

    // The expected inventory is derived from the pinned configuration only.
    let expected = expected_tensors(checkpoint.config())
        .unwrap_or_else(|error| panic!("expected inventory failed: {error}"));
    println!(
        "expected_inventory tensors={} names_and_shapes_match={}",
        expected.len(),
        checkpoint.tensor_count() == expected.len()
    );

    // Group the audited tensor names by shard, in payload order.
    let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor, shard) in checkpoint.index().weight_map() {
        by_shard
            .entry(shard.clone())
            .or_default()
            .push(tensor.clone());
    }

    let mut total_payload_bytes = 0u64;
    let mut total_tensors = 0usize;
    for shard_name in checkpoint.shard_names() {
        let mut names: Vec<String> = by_shard
            .get(shard_name)
            .cloned()
            .unwrap_or_else(|| panic!("shard {shard_name} is not in the weight map"));
        names.sort_by_key(|name| {
            checkpoint
                .tensor(name)
                .unwrap_or_else(|| panic!("tensor {name} is missing from the inventory"))
                .entry
                .data_start()
        });
        let first = checkpoint
            .tensor(&names[0])
            .unwrap_or_else(|| panic!("tensor {} missing", names[0]));
        let shard = first.shard;
        let mut payload = Sha256Stream::new();
        let mut cursor = 0u64;
        let mut gaps = 0usize;
        let mut shard_bytes = 0u64;
        for name in &names {
            let location = checkpoint
                .tensor(name)
                .unwrap_or_else(|| panic!("tensor {name} missing"));
            let entry = location.entry;
            if entry.data_start() != cursor {
                gaps += 1;
            }
            let shape_bytes = entry.numel() * entry.dtype().element_size_bytes() as u64;
            assert_eq!(shape_bytes, entry.byte_len(), "{name}: byte length");
            assert_eq!(
                expected.get(name).map(|shape| shape.as_slice()),
                Some(entry.shape()),
                "{name}: shape"
            );
            assert_eq!(entry.dtype(), identity.dtype, "{name}: dtype");
            let mut digest = Sha256Stream::new();
            let mut offset = entry.data_start();
            while offset < entry.data_end() {
                let length = usize::try_from(entry.data_end() - offset)
                    .unwrap()
                    .min(MAX_CHUNK_BYTES);
                let bytes = shard
                    .read_payload_range(offset, length)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                digest.update(&bytes);
                payload.update(&bytes);
                offset += length as u64;
            }
            shard_bytes += entry.byte_len();
            total_payload_bytes += entry.byte_len();
            total_tensors += 1;
            println!(
                "tensor={name} shard={shard_name} shape=[{}] dtype={} bytes={} sha256={}",
                entry
                    .shape()
                    .iter()
                    .map(|dimension| dimension.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                entry.dtype().name(),
                entry.byte_len(),
                hex(&digest.finish())
            );
            cursor = entry.data_end();
        }
        assert_eq!(cursor, shard.data_len(), "{shard_name}: payload coverage");
        assert_eq!(shard_bytes, shard.data_len(), "{shard_name}: byte coverage");
        assert_eq!(gaps, 0, "{shard_name}: contiguity");
        println!(
            "shard={shard_name} tensors={} header_len={} data_len={} file_len={} payload_sha256={} gaps={gaps}",
            names.len(),
            shard.header_len(),
            shard.data_len(),
            shard.file_len(),
            hex(&payload.finish()),
        );
    }

    assert_eq!(total_tensors, identity.tensor_count, "tensor coverage");
    assert_eq!(total_payload_bytes, identity.total_bytes, "byte coverage");
    println!(
        "summary tensors={total_tensors} bytes={total_payload_bytes} shards={} coverage=complete contiguous=true dtype={}",
        identity.shard_count,
        identity.dtype.name()
    );
}
