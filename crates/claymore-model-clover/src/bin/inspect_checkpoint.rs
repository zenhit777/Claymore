//! CPU-only audit of a local Clover checkpoint directory.
//!
//! Usage:
//! cargo run -p claymore-model-clover --bin inspect_checkpoint -- /path/to/Clover
//!
//! This command reads config/index files and Safetensors headers only. It does
//! not read model payloads, allocate tensors, initialize CUDA, or download data.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::str;

use claymore_checkpoint::sha256::{hex, sha256};
use claymore_model_clover::CloverCheckpointInventory;
use claymore_model_clover::CloverConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: inspect_checkpoint <checkpoint-directory>")?;

    let config_path = root.join("config.json");
    let config_bytes = fs::read(&config_path)?;
    let config_text = str::from_utf8(&config_bytes)?;
    let config = CloverConfig::from_json(config_text)?;
    config.validate_pinned()?;

    let inventory = CloverCheckpointInventory::inspect_pinned(&root, &config)?;
    let identity = inventory.identity();

    println!("model_id={}", claymore_model_clover::MODEL_ID);
    println!("revision={}", claymore_model_clover::CONFIG_REVISION);
    println!("config_sha256={}", hex(&sha256(&config_bytes)));
    println!("index_sha256={}", identity.index_sha256);
    println!("model_shards={}", identity.shard_count);
    println!("gdn_sidecars={}", identity.sidecar_count);
    println!("tensor_count={}", identity.tensor_count);
    println!("parameter_count={}", identity.parameter_count);
    println!("payload_bytes={}", identity.payload_bytes);

    for shard in inventory.shard_summaries() {
        println!(
            "file={} bytes={} header_bytes={} payload_bytes={} tensors={}",
            shard.name,
            shard.file_bytes,
            shard.header_bytes,
            shard.payload_bytes,
            shard.tensor_count
        );
    }

    println!("result=PINNED_CONFIG_AND_HEADER_INVENTORY_VALID");
    println!("payloads_read=false");
    println!("cuda_initialized=false");
    Ok(())
}
