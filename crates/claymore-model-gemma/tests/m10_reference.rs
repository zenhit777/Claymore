//! Contract test: the committed M10 per-layer reference fixture set is
//! self-consistent and identical to the pinned set.
//!
//! This test never needs the 51 GB checkpoint and therefore runs in the
//! standard CI suite. It checks:
//!
//! * `manifest.sha256` authenticates `manifest.json`, and `manifest.json`
//!   hashes to the pinned digest (the fixture set identity);
//! * every listed artifact exists, has exactly `prod(shape)` FP32 values, is
//!   all-finite, and hashes to its recorded SHA-256;
//! * the recorded generation gates are intact: 455 artifacts, 85 internal
//!   chain-consistency checks passed (including
//!   `vision_in_{L+1} == vision_layer_{L}` and `dec_in_{L+1} == decoder_layer_{L}`,
//!   the entry-fidelity property the bisections rely on), 311 bit-exactness
//!   gates against the historical golden set, zero gate failures;
//! * the environment fingerprint records the pinned upstream commit and the
//!   pinned checkpoint metadata hashes.
//!
//! Regenerating this set is a CPU-only, documented procedure:
//! `tools/m10/README.md`. Never overwrite an approved set — generate a fresh
//! directory and validate it with `tools/m10/validate_reference.py`.

use std::path::Path;

/// Identity of the pinned fixture set (SHA-256 of `manifest.json`). Changing
/// it requires regenerating the artifacts through `tools/m10/gen_reference.py`
/// with every gate green, and an explicit architect decision.
const PINNED_MANIFEST_SHA256: &str =
    "660da49e8a12e24a793d17e4a5911ba6667d2f68c43561aa566f9bc9c80c74e3";

fn fixture_dir() -> std::path::PathBuf {
    match std::env::var_os("CLAYMORE_M10_LAYERS_DIR") {
        Some(dir) => Path::new(&dir).to_path_buf(),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("data")
            .join("m10_layers"),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    claymore_model_gemma::sha256::hex(&claymore_model_gemma::sha256::sha256(bytes))
}

fn artifact_names(prefix: &str, count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("{prefix}{i}"))
        .collect::<Vec<_>>()
}

#[test]
fn committed_reference_set_matches_the_pinned_manifest() {
    let dir = fixture_dir();
    let manifest_bytes = std::fs::read(dir.join("manifest.json")).expect("manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).expect("manifest.json parses");

    // The sidecar authenticates the manifest, and the manifest is pinned.
    let sidecar = std::fs::read_to_string(dir.join("manifest.sha256")).expect("manifest.sha256");
    let sidecar_sha = sidecar
        .split_whitespace()
        .next()
        .expect("sidecar digest")
        .to_string();
    assert_eq!(
        sidecar_sha,
        sha256_hex(&manifest_bytes),
        "manifest.sha256 must authenticate manifest.json"
    );
    assert_eq!(
        sha256_hex(&manifest_bytes),
        PINNED_MANIFEST_SHA256,
        "fixture set identity changed: regenerate through tools/m10/gen_reference.py, gates green"
    );

    assert_eq!(manifest["schema"], "claymore-m10-reference/1");

    // Gates recorded at generation time.
    let artifacts = manifest["artifacts"]
        .as_object()
        .expect("artifacts object");
    assert_eq!(artifacts.len(), 455, "artifact count");
    assert!(
        manifest["gate_failures"]
            .as_array()
            .is_some_and(|f| f.is_empty()),
        "generation gates recorded as failed"
    );
    let internal = manifest["internal_checks"].as_array().expect("internal");
    assert_eq!(internal.len(), 85);
    assert!(
        internal
            .iter()
            .all(|c| c["bit_identical"] == serde_json::Value::Bool(true)),
        "internal chain-consistency check recorded as failed"
    );
    let gates = manifest["golden_gates"].as_array().expect("gates");
    assert_eq!(gates.len(), 311);
    assert!(
        gates
            .iter()
            .all(|c| c["bit_identical"] == serde_json::Value::Bool(true)),
        "bit-exactness gate against the historical golden set recorded as failed"
    );

    // Environment fingerprint: pinned upstream and checkpoint metadata.
    let env = &manifest["environment"];
    assert_eq!(
        env["transformers_commit_pinned"],
        "671aa77576266f41f33f9c28e2d824bfd1b1b63c"
    );
    assert_eq!(env["torch"], "2.15.0+cpu");
    assert_eq!(env["cuda_visible_devices"], "");
    assert_eq!(
        manifest["checkpoint"]["revision_pinned"],
        "cd40195554d3cfcb56f34b9adfb66efc790ed5b6"
    );
    assert_eq!(
        manifest["checkpoint"]["config_sha256"],
        "13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506"
    );
    assert_eq!(
        manifest["checkpoint"]["index_sha256"],
        "6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a"
    );
    // The decoder pass never mutates the encoder KV rows: the manifest records
    // 7 rows at every decoder layer entry (no decoder-prefix fixture needed).
    let rows = manifest["scenario"]["cache_rows_at_decoder_entry"]
        .as_object()
        .expect("cache rows");
    assert_eq!(rows.len(), 30);
    assert!(rows.values().all(|v| *v == serde_json::json!(7)));

    // Every artifact: present, right size, all-finite, SHA-256 matches.
    for (name, meta) in artifacts {
        let path = dir.join(meta["file"].as_str().expect("file name"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let shape = meta["shape"]
            .as_array()
            .expect("shape array")
            .iter()
            .map(|v| v.as_u64().expect("dim") as usize)
            .collect::<Vec<_>>();
        let numel: usize = shape.iter().product();
        assert_eq!(bytes.len(), numel * 4, "{name}: byte size");
        assert_eq!(meta["dtype"], "float32-le", "{name}: dtype");
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for w in &words {
            // A non-finite F32 has an all-ones exponent field.
            assert!((w >> 23) & 0xff != 0xff, "{name}: non-finite value {w:#010x}");
        }
        assert_eq!(
            sha256_hex(&bytes),
            meta["sha256"].as_str().expect("sha256"),
            "{name}: sha256"
        );
    }
}

#[test]
fn every_bisect_fixture_family_is_present() {
    let dir = fixture_dir();
    let mut names: Vec<String> = Vec::new();
    names.extend(artifact_names("enc_in_", 30));
    names.extend(artifact_names("encoder_layer_", 30));
    for l in 0..30 {
        for suffix in ["router_scores", "router_weights", "router_indices"] {
            names.push(format!("encoder{l}_{suffix}"));
        }
        names.push(format!("encoder{l}_expert_output"));
    }
    names.extend(artifact_names("vision_in_", 27));
    names.extend(artifact_names("vision_layer_", 27));
    names.extend(artifact_names("dec_in_", 30));
    names.extend(artifact_names("decoder_layer_", 30));
    for l in 0..30 {
        for suffix in ["router_scores", "router_weights", "router_indices"] {
            names.push(format!("moe{l}_{suffix}"));
        }
        names.push(format!("enc_kv_{l}_keys"));
        names.push(format!("enc_kv_{l}_values"));
    }
    names.extend([
        "prompt_ids".to_string(),
        "canvas0_direct".to_string(),
        "vision_pixel_values".to_string(),
        "vision_pixel_position_ids".to_string(),
        "encoder_embedding".to_string(),
        "decoder_embedding".to_string(),
    ]);
    for name in &names {
        assert!(
            dir.join(format!("{name}.bin")).is_file(),
            "missing fixture {name}"
        );
    }
}
