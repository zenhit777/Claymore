//! CPU-only audit of a local Clover checkpoint directory.
//!
//! Usage:
//! cargo run -p claymore-model-clover --bin inspect_checkpoint -- /path/to/Clover [--verify-sha256]
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
use claymore_model_clover::{CONFIG_SHA256, INDEX_SHA256};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: inspect_checkpoint <checkpoint-directory> [--verify-sha256]")?;
    let verify_hashes = match args.next() {
        None => false,
        Some(flag) if flag == "--verify-sha256" => true,
        Some(_) => return Err("unknown inspect_checkpoint option".into()),
    };
    if args.next().is_some() {
        return Err("too many inspect_checkpoint arguments".into());
    }

    let config_path = root.join("config.json");
    let config_bytes = fs::read(&config_path)?;
    let config_digest = hex(&sha256(&config_bytes));
    if config_digest != CONFIG_SHA256 {
        return Err(format!(
            "config.json digest mismatch: expected {CONFIG_SHA256}, got {config_digest}"
        )
        .into());
    }
    let config_text = str::from_utf8(&config_bytes)?;
    let config = CloverConfig::from_json(config_text)?;
    config.validate_pinned()?;

    let inventory = CloverCheckpointInventory::inspect_pinned(&root, &config)?;
    let identity = inventory.identity();
    if identity.index_sha256 != INDEX_SHA256 {
        return Err(format!(
            "checkpoint index digest mismatch: expected {INDEX_SHA256}, got {}",
            identity.index_sha256
        )
        .into());
    }

    println!("model_id={}", claymore_model_clover::MODEL_ID);
    println!("revision={}", claymore_model_clover::CONFIG_REVISION);
    println!("config_sha256={config_digest}");
    println!("index_sha256={}", identity.index_sha256);
    println!("model_shards={}", identity.shard_count);
    println!("gdn_sidecars={}", identity.sidecar_count);
    println!("tensor_count={}", identity.tensor_count);
    println!("parameter_count={}", identity.parameter_count);
    println!("payload_bytes={}", identity.payload_bytes);
    if verify_hashes {
        inventory.verify_pinned_asset_hashes()?;
        println!("artifact_sha256=verified");
    } else {
        println!("artifact_sha256=not_requested");
    }

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
