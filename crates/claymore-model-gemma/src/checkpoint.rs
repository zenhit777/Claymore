//! Pinned checkpoint ingestion: shard index validation, full tensor inventory
//! validation and on-demand bounded reads.
//!
//! `Checkpoint::open_pinned` accepts exactly the audited revision metadata and
//! weight geometry. It never materialises all 51.6 GB: shard headers are
//! validated eagerly and tensor payloads are read one bounded tensor at a time.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use crate::config::{
    GemmaConfigError, GemmaModelConfig, CHECKPOINT_REVISION, MODEL_ID, PINNED_CONFIG_SHA256,
    PINNED_GENERATION_CONFIG_SHA256, PINNED_INDEX_SHA256, PINNED_SHARD_COUNT, PINNED_TENSOR_COUNT,
    PINNED_TOTAL_BYTES, PINNED_TOTAL_PARAMETERS,
};
use crate::safetensors::{SafetensorsError, SafetensorsFile, TensorEntry, WeightDType};
use crate::sha256::{hex, sha256};
use crate::strict_json::Json;

/// Maximum accepted shard index size.
pub const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum accepted config/generation JSON size.
pub const MAX_METADATA_BYTES: u64 = 1024 * 1024;

const INDEX_FILE: &str = "model.safetensors.index.json";
const CONFIG_FILE: &str = "config.json";
const GENERATION_CONFIG_FILE: &str = "generation_config.json";

/// Validated view of `model.safetensors.index.json`.
#[derive(Debug, Clone)]
pub struct ShardIndex {
    total_size: u64,
    total_parameters: Option<u64>,
    weight_map: BTreeMap<String, String>,
    shards: BTreeSet<String>,
}

impl ShardIndex {
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Parameter count declared by the index, when the upstream writer emits
    /// it. The pinned revision declares 25,823,778,864 BF16 parameters.
    pub fn total_parameters(&self) -> Option<u64> {
        self.total_parameters
    }

    pub fn weight_map(&self) -> &BTreeMap<String, String> {
        &self.weight_map
    }

    pub fn shards(&self) -> &BTreeSet<String> {
        &self.shards
    }

    pub fn shard_for(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }

    fn load(root: &Path, verify_pin: bool) -> Result<Self, CheckpointError> {
        let path = root.join(INDEX_FILE);
        let bytes = read_bounded(&path, MAX_INDEX_BYTES)?;
        if verify_pin {
            let digest = hex(&sha256(&bytes));
            if digest != PINNED_INDEX_SHA256 {
                return Err(CheckpointError::PinnedDigest {
                    file: INDEX_FILE,
                    expected: PINNED_INDEX_SHA256,
                    actual: digest,
                });
            }
        }
        let root_json = Json::parse(&bytes).map_err(|message| CheckpointError::InvalidIndex {
            reason: format!("invalid JSON: {message}"),
        })?;
        let fields = root_json
            .as_object()
            .ok_or_else(|| CheckpointError::InvalidIndex {
                reason: format!("root is {}", root_json.type_name()),
            })?;
        require_only_fields(fields, &["metadata", "weight_map"], "index root")?;

        let metadata = root_json
            .get("metadata")
            .and_then(Json::as_object)
            .ok_or_else(|| CheckpointError::InvalidIndex {
                reason: "metadata must be an object".to_owned(),
            })?;
        require_only_fields(
            metadata,
            &["total_size", "total_parameters"],
            "index metadata",
        )?;
        let total_size = root_json
            .get("metadata")
            .and_then(|value| value.get("total_size"))
            .and_then(Json::as_u64)
            .ok_or_else(|| CheckpointError::InvalidIndex {
                reason: "metadata.total_size must be an unsigned integer".to_owned(),
            })?;
        if total_size == 0 {
            return Err(CheckpointError::InvalidIndex {
                reason: "metadata.total_size must be non-zero".to_owned(),
            });
        }
        let total_parameters = match root_json
            .get("metadata")
            .and_then(|value| value.get("total_parameters"))
        {
            None => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .ok_or_else(|| CheckpointError::InvalidIndex {
                        reason: "metadata.total_parameters must be an unsigned integer".to_owned(),
                    })?,
            ),
        };
        if total_parameters == Some(0) {
            return Err(CheckpointError::InvalidIndex {
                reason: "metadata.total_parameters must be non-zero".to_owned(),
            });
        }

        let weight_map_json = root_json
            .get("weight_map")
            .and_then(Json::as_object)
            .ok_or_else(|| CheckpointError::InvalidIndex {
                reason: "weight_map must be an object".to_owned(),
            })?;
        if weight_map_json.is_empty() {
            return Err(CheckpointError::InvalidIndex {
                reason: "weight_map must not be empty".to_owned(),
            });
        }
        let mut weight_map = BTreeMap::new();
        let mut shards = BTreeSet::new();
        for (tensor, shard) in weight_map_json {
            if tensor.is_empty() {
                return Err(CheckpointError::InvalidIndex {
                    reason: "weight_map contains an empty tensor name".to_owned(),
                });
            }
            let shard = shard
                .as_str()
                .ok_or_else(|| CheckpointError::InvalidIndex {
                    reason: format!("weight_map entry {tensor:?} is not a string"),
                })?;
            validate_shard_name(shard)?;
            shards.insert(shard.to_owned());
            weight_map.insert(tensor.clone(), shard.to_owned());
        }
        for shard in &shards {
            let shard_path = root.join(shard);
            if !shard_path.is_file() {
                return Err(CheckpointError::MissingShard {
                    shard: shard.clone(),
                    path: shard_path,
                });
            }
        }
        Ok(Self {
            total_size,
            total_parameters,
            weight_map,
            shards,
        })
    }
}

/// Location of a tensor inside one validated shard.
#[derive(Debug, Clone, Copy)]
pub struct TensorLocation<'a> {
    pub shard: &'a SafetensorsFile,
    pub entry: &'a TensorEntry,
}

/// Fully validated pinned checkpoint. Payloads remain on disk.
pub struct Checkpoint {
    root: PathBuf,
    config: GemmaModelConfig,
    index: ShardIndex,
    files: BTreeMap<String, SafetensorsFile>,
    locations: BTreeMap<String, String>,
    identity: CheckpointIdentity,
}

impl std::fmt::Debug for Checkpoint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Checkpoint")
            .field("root", &self.root)
            .field("identity", &self.identity)
            .field("tensor_count", &self.locations.len())
            .finish()
    }
}

/// Provenance captured while opening the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointIdentity {
    pub model_id: &'static str,
    pub revision: &'static str,
    pub config_sha256: String,
    pub index_sha256: String,
    pub generation_config_sha256: String,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub dtype: WeightDType,
}

impl Checkpoint {
    /// Open exactly the audited checkpoint revision. This checks file digests,
    /// the pinned configuration, shard/tensor counts, total size, BF16 dtype
    /// and the complete expected tensor inventory.
    pub fn open_pinned(root: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        Self::open(root.as_ref(), true)
    }

    fn open(root: &Path, verify_pin: bool) -> Result<Self, CheckpointError> {
        let root = root.to_path_buf();
        if !root.is_dir() {
            return Err(CheckpointError::MissingDirectory { path: root });
        }

        let config_bytes = read_bounded(&root.join(CONFIG_FILE), MAX_METADATA_BYTES)?;
        let config_json =
            std::str::from_utf8(&config_bytes).map_err(|error| CheckpointError::InvalidConfig {
                reason: format!("config.json is not UTF-8: {error}"),
            })?;
        let config = GemmaModelConfig::from_json(config_json).map_err(CheckpointError::Config)?;
        if verify_pin {
            config.validate_pinned().map_err(CheckpointError::Config)?;
            let digest = hex(&sha256(&config_bytes));
            if digest != PINNED_CONFIG_SHA256 {
                return Err(CheckpointError::PinnedDigest {
                    file: CONFIG_FILE,
                    expected: PINNED_CONFIG_SHA256,
                    actual: digest,
                });
            }
        }

        let generation_bytes =
            read_bounded(&root.join(GENERATION_CONFIG_FILE), MAX_METADATA_BYTES)?;
        let generation_digest = hex(&sha256(&generation_bytes));
        if verify_pin {
            if generation_digest != PINNED_GENERATION_CONFIG_SHA256 {
                return Err(CheckpointError::PinnedDigest {
                    file: GENERATION_CONFIG_FILE,
                    expected: PINNED_GENERATION_CONFIG_SHA256,
                    actual: generation_digest.clone(),
                });
            }
            crate::generation::DiffusionGenerationConfig::from_json(
                std::str::from_utf8(&generation_bytes).unwrap_or(""),
            )
            .and_then(|config| config.validate_pinned())
            .map_err(CheckpointError::Generation)?;
        }
        let generation_json = Json::parse(&generation_bytes)
            .map_err(|message| CheckpointError::InvalidGeneration { reason: message })?;
        if generation_json.as_object().is_none() {
            return Err(CheckpointError::InvalidGeneration {
                reason: "generation_config.json root must be an object".to_owned(),
            });
        }

        let index = ShardIndex::load(&root, verify_pin)?;
        if verify_pin {
            let expected_shards: BTreeSet<String> = (1..=PINNED_SHARD_COUNT)
                .map(|index| format!("model-{index:05}-of-{PINNED_SHARD_COUNT:05}.safetensors"))
                .collect();
            if index.shards != expected_shards {
                return Err(CheckpointError::PinnedShardSet {
                    expected: expected_shards.into_iter().collect(),
                    actual: index.shards.iter().cloned().collect(),
                });
            }
        }

        let mut files = BTreeMap::new();
        let mut locations = BTreeMap::new();
        let mut actual_total = 0u64;
        let mut dtype: Option<WeightDType> = None;
        for shard_name in &index.shards {
            let shard = SafetensorsFile::open(root.join(shard_name)).map_err(|source| {
                CheckpointError::Safetensors {
                    shard: shard_name.clone(),
                    source,
                }
            })?;
            for entry in shard.tensors() {
                let mapped = index.weight_map.get(entry.name()).ok_or_else(|| {
                    CheckpointError::UnexpectedTensor {
                        tensor: entry.name().to_owned(),
                        shard: shard_name.clone(),
                    }
                })?;
                if mapped != shard_name {
                    return Err(CheckpointError::TensorInWrongShard {
                        tensor: entry.name().to_owned(),
                        indexed: mapped.clone(),
                        found: shard_name.clone(),
                    });
                }
                locations.insert(entry.name().to_owned(), shard_name.clone());
                actual_total = actual_total
                    .checked_add(entry.byte_len())
                    .ok_or(CheckpointError::SizeOverflow)?;
                match dtype {
                    None => dtype = Some(entry.dtype()),
                    Some(expected) if expected != entry.dtype() => {
                        return Err(CheckpointError::MixedDtype {
                            tensor: entry.name().to_owned(),
                            expected,
                            actual: entry.dtype(),
                        });
                    }
                    Some(_) => {}
                }
            }
            files.insert(shard_name.clone(), shard);
        }
        for tensor in index.weight_map.keys() {
            if !locations.contains_key(tensor) {
                return Err(CheckpointError::MissingTensor {
                    tensor: tensor.clone(),
                    shard: index.weight_map[tensor].clone(),
                });
            }
        }
        if actual_total != index.total_size {
            return Err(CheckpointError::TotalSizeMismatch {
                indexed: index.total_size,
                actual: actual_total,
            });
        }
        let dtype = dtype.ok_or(CheckpointError::EmptyCheckpoint)?;

        if verify_pin {
            if dtype != WeightDType::BF16 {
                return Err(CheckpointError::PinnedDtype { actual: dtype });
            }
            if locations.len() != PINNED_TENSOR_COUNT {
                return Err(CheckpointError::PinnedTensorCount {
                    expected: PINNED_TENSOR_COUNT,
                    actual: locations.len(),
                });
            }
            if actual_total != PINNED_TOTAL_BYTES {
                return Err(CheckpointError::PinnedTotalBytes {
                    expected: PINNED_TOTAL_BYTES,
                    actual: actual_total,
                });
            }
            if index.total_parameters != Some(PINNED_TOTAL_PARAMETERS) {
                return Err(CheckpointError::PinnedParameterCount {
                    expected: PINNED_TOTAL_PARAMETERS,
                    actual: index.total_parameters,
                });
            }
            let expected = expected_tensors(&config)?;
            let mut expected_names = BTreeSet::new();
            for (name, shape) in &expected {
                expected_names.insert(name.clone());
                let shard_name =
                    locations
                        .get(name)
                        .ok_or_else(|| CheckpointError::MissingWeight {
                            tensor: name.clone(),
                        })?;
                let entry = files[shard_name].tensor(name).ok_or_else(|| {
                    CheckpointError::MissingWeight {
                        tensor: name.clone(),
                    }
                })?;
                if entry.dtype() != WeightDType::BF16 {
                    return Err(CheckpointError::UnexpectedDType {
                        tensor: name.clone(),
                        expected: WeightDType::BF16,
                        actual: entry.dtype(),
                    });
                }
                if entry.shape() != shape.as_slice() {
                    return Err(CheckpointError::WeightShape {
                        tensor: name.clone(),
                        expected: shape.clone(),
                        actual: entry.shape().to_vec(),
                    });
                }
            }
            let actual_names: BTreeSet<&String> = locations.keys().collect();
            if let Some(unexpected) = actual_names
                .iter()
                .find(|name| !expected_names.contains(**name))
            {
                return Err(CheckpointError::UnexpectedWeight {
                    tensor: (*unexpected).clone(),
                });
            }
        }

        let identity = CheckpointIdentity {
            model_id: MODEL_ID,
            revision: CHECKPOINT_REVISION,
            config_sha256: hex(&sha256(&config_bytes)),
            index_sha256: if verify_pin {
                PINNED_INDEX_SHA256.to_owned()
            } else {
                let bytes =
                    std::fs::read(root.join(INDEX_FILE)).map_err(|source| CheckpointError::Io {
                        path: root.join(INDEX_FILE),
                        message: source.to_string(),
                    })?;
                hex(&sha256(&bytes))
            },
            generation_config_sha256: generation_digest,
            shard_count: files.len(),
            tensor_count: locations.len(),
            total_bytes: actual_total,
            dtype,
        };

        Ok(Self {
            root,
            config,
            index,
            files,
            locations,
            identity,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &GemmaModelConfig {
        &self.config
    }

    pub fn index(&self) -> &ShardIndex {
        &self.index
    }

    pub fn identity(&self) -> &CheckpointIdentity {
        &self.identity
    }

    pub fn tensor_count(&self) -> usize {
        self.locations.len()
    }

    pub fn total_bytes(&self) -> u64 {
        self.identity.total_bytes
    }

    pub fn shard_names(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<TensorLocation<'_>> {
        let shard_name = self.locations.get(name)?;
        let shard = self.files.get(shard_name)?;
        let entry = shard.tensor(name)?;
        Some(TensorLocation { shard, entry })
    }

    /// Read exact stored bytes for one tensor.
    pub fn read_bytes(&self, name: &str) -> Result<Vec<u8>, CheckpointError> {
        let location = self
            .tensor(name)
            .ok_or_else(|| CheckpointError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        location
            .shard
            .read_bytes(name)
            .map_err(|source| CheckpointError::Safetensors {
                shard: location.shard.path().display().to_string(),
                source,
            })
    }

    /// Read a byte range inside one tensor's payload.
    ///
    /// `offset` is relative to the start of the tensor payload. The shard
    /// header offset is applied by the shard reader, so the returned bytes are
    /// exactly the stored bytes of the requested range.
    pub fn read_tensor_range(
        &self,
        name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, CheckpointError> {
        let location = self
            .tensor(name)
            .ok_or_else(|| CheckpointError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        location
            .shard
            .read_tensor_range(name, offset, length)
            .map_err(|source| CheckpointError::Safetensors {
                shard: location.shard.path().display().to_string(),
                source,
            })
    }

    /// Read one tensor as explicit FP32 values (BF16 widened exactly).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, CheckpointError> {
        let location = self
            .tensor(name)
            .ok_or_else(|| CheckpointError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        location
            .shard
            .read_f32(name)
            .map_err(|source| CheckpointError::Safetensors {
                shard: location.shard.path().display().to_string(),
                source,
            })
    }

    /// Read one BF16 tensor without narrowing or implicit conversion.
    pub fn read_bf16_bits(&self, name: &str) -> Result<Vec<u16>, CheckpointError> {
        let location = self
            .tensor(name)
            .ok_or_else(|| CheckpointError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        location
            .shard
            .read_bf16_bits(name)
            .map_err(|source| CheckpointError::Safetensors {
                shard: location.shard.path().display().to_string(),
                source,
            })
    }
}

/// Expected tensor inventory derived from the pinned configuration.
pub fn expected_tensors(
    config: &GemmaModelConfig,
) -> Result<BTreeMap<String, Vec<usize>>, CheckpointError> {
    let text = config
        .text_config
        .as_ref()
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "text_config is required to derive the expected tensor inventory".to_owned(),
        })?;
    let vision = config
        .vision_config
        .as_ref()
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "vision_config is required to derive the expected tensor inventory".to_owned(),
        })?;
    let hidden = text
        .hidden_size
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "text_config.hidden_size is required".to_owned(),
        })?;
    let layers = text
        .num_hidden_layers
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "text_config.num_hidden_layers is required".to_owned(),
        })?;
    let intermediate = text
        .intermediate_size
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "text_config.intermediate_size is required".to_owned(),
        })?;
    let moe = text
        .moe_intermediate_size
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "text_config.moe_intermediate_size is required".to_owned(),
        })?;
    let experts = text
        .num_experts
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "text_config.num_experts is required".to_owned(),
        })?;
    let vocab = text
        .vocab_size
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "text_config.vocab_size is required".to_owned(),
        })?;
    let vision_hidden = vision
        .hidden_size
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "vision_config.hidden_size is required".to_owned(),
        })?;
    let vision_layers = vision
        .num_hidden_layers
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "vision_config.num_hidden_layers is required".to_owned(),
        })?;
    let vision_intermediate =
        vision
            .intermediate_size
            .ok_or_else(|| CheckpointError::InvalidConfig {
                reason: "vision_config.intermediate_size is required".to_owned(),
            })?;
    let patch_size = vision
        .patch_size
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "vision_config.patch_size is required".to_owned(),
        })?;
    let position_embedding_size =
        vision
            .position_embedding_size
            .ok_or_else(|| CheckpointError::InvalidConfig {
                reason: "vision_config.position_embedding_size is required".to_owned(),
            })?;
    let vision_heads =
        vision
            .num_attention_heads
            .ok_or_else(|| CheckpointError::InvalidConfig {
                reason: "vision_config.num_attention_heads is required".to_owned(),
            })?;
    let vision_head_dim = vision
        .head_dim
        .ok_or_else(|| CheckpointError::InvalidConfig {
            reason: "vision_config.head_dim is required".to_owned(),
        })?;
    let vision_q = checked_product("vision q projection", &[vision_heads, vision_head_dim])?;

    let mut expected = BTreeMap::new();
    let mut insert = |name: String, shape: Vec<usize>| -> Result<(), CheckpointError> {
        if expected.insert(name.clone(), shape).is_some() {
            return Err(CheckpointError::InvalidConfig {
                reason: format!("expected inventory contains duplicate tensor {name:?}"),
            });
        }
        Ok(())
    };

    for layer in 0..layers {
        let attention =
            text.layer_attention(layer)
                .ok_or_else(|| CheckpointError::InvalidConfig {
                    reason: format!("missing attention geometry for layer {layer}"),
                })?;
        let q = checked_product(
            "attention q projection",
            &[attention.heads, attention.head_dim],
        )?;
        let kv = checked_product(
            "attention kv projection",
            &[attention.kv_heads, attention.head_dim],
        )?;
        let prefix = format!("model.decoder.layers.{layer}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            insert(format!("{prefix}.{norm}.weight"), vec![hidden])?;
        }
        insert(format!("{prefix}.layer_scalar"), vec![1])?;
        insert(format!("{prefix}.self_attn.q_proj.weight"), vec![q, hidden])?;
        insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv, hidden],
        )?;
        if !attention.global {
            insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv, hidden],
            )?;
        }
        insert(format!("{prefix}.self_attn.o_proj.weight"), vec![hidden, q])?;
        insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![attention.head_dim],
        )?;
        insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![attention.head_dim],
        )?;
        insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        )?;
        insert(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        )?;
        insert(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        )?;
        insert(format!("{prefix}.router.per_expert_scale"), vec![experts])?;
        insert(
            format!("{prefix}.router.proj.weight"),
            vec![experts, hidden],
        )?;
        insert(format!("{prefix}.router.scale"), vec![hidden])?;
        insert(
            format!("{prefix}.experts.down_proj"),
            vec![experts, hidden, moe],
        )?;
        insert(
            format!("{prefix}.experts.gate_up_proj"),
            vec![experts, 2 * moe, hidden],
        )?;
        insert(
            format!("model.encoder.language_model.layers.{layer}.layer_scalar"),
            vec![1],
        )?;
    }

    for layer in 0..vision_layers {
        let prefix = format!("model.encoder.vision_tower.encoder.layers.{layer}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            insert(format!("{prefix}.{norm}.weight"), vec![vision_hidden])?;
        }
        insert(
            format!("{prefix}.mlp.gate_proj.linear.weight"),
            vec![vision_intermediate, vision_hidden],
        )?;
        insert(
            format!("{prefix}.mlp.up_proj.linear.weight"),
            vec![vision_intermediate, vision_hidden],
        )?;
        insert(
            format!("{prefix}.mlp.down_proj.linear.weight"),
            vec![vision_hidden, vision_intermediate],
        )?;
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            insert(
                format!("{prefix}.self_attn.{projection}.linear.weight"),
                vec![vision_q, vision_hidden],
            )?;
        }
        insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![vision_head_dim],
        )?;
        insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![vision_head_dim],
        )?;
    }

    insert(
        "model.decoder.embed_tokens.weight".to_owned(),
        vec![vocab, hidden],
    )?;
    insert("model.decoder.norm.weight".to_owned(), vec![hidden])?;
    insert(
        "model.decoder.self_conditioning.pre_norm.weight".to_owned(),
        vec![hidden],
    )?;
    insert(
        "model.decoder.self_conditioning.down_proj.weight".to_owned(),
        vec![hidden, intermediate],
    )?;
    insert(
        "model.decoder.self_conditioning.gate_proj.weight".to_owned(),
        vec![intermediate, hidden],
    )?;
    insert(
        "model.decoder.self_conditioning.up_proj.weight".to_owned(),
        vec![intermediate, hidden],
    )?;
    insert(
        "model.encoder.embed_vision.embedding_projection.weight".to_owned(),
        vec![hidden, vision_hidden],
    )?;
    insert(
        "model.encoder.vision_tower.patch_embedder.input_proj.weight".to_owned(),
        vec![
            vision_hidden,
            checked_product("vision patch", &[patch_size, patch_size, 3])?,
        ],
    )?;
    insert(
        "model.encoder.vision_tower.patch_embedder.position_embedding_table".to_owned(),
        vec![2, position_embedding_size, vision_hidden],
    )?;
    insert(
        "model.encoder.vision_tower.std_bias".to_owned(),
        vec![vision_hidden],
    )?;
    insert(
        "model.encoder.vision_tower.std_scale".to_owned(),
        vec![vision_hidden],
    )?;

    Ok(expected)
}

fn checked_product(field: &str, dimensions: &[usize]) -> Result<usize, CheckpointError> {
    dimensions.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| CheckpointError::InvalidConfig {
                reason: format!("{field} overflows usize"),
            })
    })
}

fn require_only_fields(
    entries: &[(String, Json)],
    allowed: &[&str],
    context: &str,
) -> Result<(), CheckpointError> {
    for (name, _) in entries {
        if !allowed.contains(&name.as_str()) {
            return Err(CheckpointError::InvalidIndex {
                reason: format!("unexpected {context} field {name:?}"),
            });
        }
    }
    Ok(())
}

fn validate_shard_name(shard: &str) -> Result<(), CheckpointError> {
    let invalid = shard.is_empty()
        || shard.contains('\0')
        || shard.contains('\\')
        || !shard.ends_with(".safetensors")
        || shard.starts_with('.');
    if invalid {
        return Err(CheckpointError::InvalidIndex {
            reason: format!("unsafe or unsupported shard name {shard:?}"),
        });
    }
    let path = Path::new(shard);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(name)), None) if name == std::ffi::OsStr::new(shard) => Ok(()),
        _ => Err(CheckpointError::InvalidIndex {
            reason: format!("shard name {shard:?} must be a single relative file name"),
        }),
    }
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, CheckpointError> {
    let metadata = std::fs::metadata(path).map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    if metadata.len() > maximum {
        return Err(CheckpointError::FileTooLarge {
            path: path.to_path_buf(),
            len: metadata.len(),
            maximum,
        });
    }
    std::fs::read(path).map_err(|source| CheckpointError::Io {
        path: path.to_path_buf(),
        message: source.to_string(),
    })
}

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("checkpoint directory {path} does not exist")]
    MissingDirectory { path: PathBuf },
    #[error("I/O error for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("{path} is {len} bytes, above the {maximum}-byte limit")]
    FileTooLarge {
        path: PathBuf,
        len: u64,
        maximum: u64,
    },
    #[error("{file} digest mismatch: expected {expected}, got {actual}")]
    PinnedDigest {
        file: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("invalid shard index: {reason}")]
    InvalidIndex { reason: String },
    #[error("invalid config.json: {reason}")]
    InvalidConfig { reason: String },
    #[error("invalid generation_config.json: {reason}")]
    InvalidGeneration { reason: String },
    #[error(transparent)]
    Config(#[from] GemmaConfigError),
    #[error(transparent)]
    Generation(#[from] crate::generation::GenerationConfigError),
    #[error("shard {shard} is missing at {path}")]
    MissingShard { shard: String, path: PathBuf },
    #[error("Safetensors validation failed for shard {shard}: {source}")]
    Safetensors {
        shard: String,
        #[source]
        source: SafetensorsError,
    },
    #[error("tensor {tensor:?} appears in shard {shard:?} but not in the weight map")]
    UnexpectedTensor { tensor: String, shard: String },
    #[error("tensor {tensor:?} is indexed in shard {indexed:?} but stored in {found:?}")]
    TensorInWrongShard {
        tensor: String,
        indexed: String,
        found: String,
    },
    #[error("tensor {tensor:?} is indexed in {shard:?} but absent from the shard")]
    MissingTensor { tensor: String, shard: String },
    #[error("tensor {tensor:?} is missing from the checkpoint")]
    MissingWeight { tensor: String },
    #[error("unexpected weight {tensor:?} is not part of the pinned graph")]
    UnexpectedWeight { tensor: String },
    #[error("tensor {tensor:?} has shape {actual:?}, expected {expected:?}")]
    WeightShape {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor {tensor:?} has dtype {actual:?}, expected {expected:?}")]
    UnexpectedDType {
        tensor: String,
        expected: WeightDType,
        actual: WeightDType,
    },
    #[error(
        "checkpoint mixes dtypes: {tensor:?} is {actual:?}, previous tensors were {expected:?}"
    )]
    MixedDtype {
        tensor: String,
        expected: WeightDType,
        actual: WeightDType,
    },
    #[error("index declares {indexed} bytes but shards contain {actual}")]
    TotalSizeMismatch { indexed: u64, actual: u64 },
    #[error("pinned checkpoint must be entirely BF16, found {actual:?}")]
    PinnedDtype { actual: WeightDType },
    #[error("pinned checkpoint has {actual} tensors, expected {expected}")]
    PinnedTensorCount { expected: usize, actual: usize },
    #[error("pinned checkpoint has {actual} tensor bytes, expected {expected}")]
    PinnedTotalBytes { expected: u64, actual: u64 },
    #[error("pinned checkpoint declares {actual:?} parameters, expected {expected}")]
    PinnedParameterCount { expected: u64, actual: Option<u64> },
    #[error("pinned checkpoint shard set mismatch: expected {expected:?}, got {actual:?}")]
    PinnedShardSet {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    #[error("pinned checkpoint contains no tensors")]
    EmptyCheckpoint,
    #[error("checkpoint size arithmetic overflow")]
    SizeOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "claymore-m10-checkpoint-{}-{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct TensorSpec {
        name: String,
        dtype: &'static str,
        shape: Vec<u64>,
        bytes: usize,
    }

    fn bf16(name: &str, shape: &[u64], payload: usize) -> TensorSpec {
        TensorSpec {
            name: name.to_owned(),
            dtype: "BF16",
            shape: shape.to_vec(),
            bytes: payload,
        }
    }

    fn write_shard(path: &Path, tensors: &[TensorSpec]) {
        let mut header = String::from("{");
        let mut payload = Vec::new();
        for (index, tensor) in tensors.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let start = payload.len();
            payload.extend(std::iter::repeat_n(0x3fu8, tensor.bytes));
            let end = payload.len();
            header.push_str(&format!(
                r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{start},{end}]}}"#,
                tensor.name,
                tensor.dtype,
                tensor
                    .shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        header.push('}');
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&payload);
        std::fs::write(path, file).unwrap();
    }

    fn write_index(root: &Path, total: u64, map: &[(&str, &str)]) -> std::io::Result<()> {
        let entries = map
            .iter()
            .map(|(tensor, shard)| format!(r#""{tensor}":"{shard}""#))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            root.join(INDEX_FILE),
            format!(r#"{{"metadata":{{"total_size":{total}}},"weight_map":{{{entries}}}}}"#),
        )
    }

    fn write_minimal_checkpoint(root: &Path, tensor: &TensorSpec, shard: &str) {
        std::fs::write(root.join(CONFIG_FILE), r#"{"model_type":"synthetic"}"#).unwrap();
        std::fs::write(root.join(GENERATION_CONFIG_FILE), "{}").unwrap();
        write_shard(
            &root.join(shard),
            &[bf16(&tensor.name, &tensor.shape, tensor.bytes)],
        );
        write_index(root, tensor.bytes as u64, &[(tensor.name.as_str(), shard)]).unwrap();
    }

    #[test]
    fn opens_a_consistent_synthetic_checkpoint_and_reads_tensors() {
        let dir = TestDir::new();
        let spec = bf16("a.weight", &[2], 4);
        write_minimal_checkpoint(dir.path(), &spec, "shard.safetensors");
        let checkpoint = Checkpoint::open(dir.path(), false).unwrap();
        assert_eq!(checkpoint.tensor_count(), 1);
        assert_eq!(checkpoint.total_bytes(), 4);
        assert_eq!(checkpoint.read_bf16_bits("a.weight").unwrap().len(), 2);
        assert_eq!(
            checkpoint.read_f32("a.weight").unwrap(),
            [0.74609375, 0.74609375]
        );
        assert!(checkpoint.tensor("missing").is_none());
        assert!(matches!(
            checkpoint.read_bytes("missing"),
            Err(CheckpointError::MissingWeight { .. })
        ));
        assert_eq!(checkpoint.identity().dtype, WeightDType::BF16);
    }

    #[test]
    fn rejects_missing_directory_config_and_generation_config() {
        let dir = TestDir::new();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::Io { .. })
        ));
        std::fs::write(dir.path().join(CONFIG_FILE), "{}").unwrap();
        std::fs::write(dir.path().join(GENERATION_CONFIG_FILE), "{}").unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::Io { .. })
        ));
    }

    #[test]
    fn rejects_malformed_indexes_and_unsafe_shard_names() {
        let dir = TestDir::new();
        std::fs::write(dir.path().join(CONFIG_FILE), "{}").unwrap();
        std::fs::write(dir.path().join(GENERATION_CONFIG_FILE), "{}").unwrap();
        let index = dir.path().join(INDEX_FILE);

        for (name, contents) in [
            ("json", "{".to_owned()),
            (
                "missing metadata",
                r#"{"weight_map":{"a":"s.safetensors"}}"#.to_owned(),
            ),
            (
                "missing weight map",
                r#"{"metadata":{"total_size":1}}"#.to_owned(),
            ),
            (
                "unknown field",
                r#"{"metadata":{"total_size":1},"weight_map":{"a":"s.safetensors"},"x":1}"#.to_owned(),
            ),
            (
                "zero total",
                r#"{"metadata":{"total_size":0},"weight_map":{"a":"s.safetensors"}}"#.to_owned(),
            ),
            (
                "duplicate tensor",
                r#"{"metadata":{"total_size":1},"weight_map":{"a":"s.safetensors","a":"s.safetensors"}}"#.to_owned(),
            ),
            (
                "traversal",
                r#"{"metadata":{"total_size":1},"weight_map":{"a":"../s.safetensors"}}"#.to_owned(),
            ),
            (
                "absolute",
                r#"{"metadata":{"total_size":1},"weight_map":{"a":"/tmp/s.safetensors"}}"#.to_owned(),
            ),
            (
                "wrong suffix",
                r#"{"metadata":{"total_size":1},"weight_map":{"a":"s.bin"}}"#.to_owned(),
            ),
            (
                "hidden shard",
                r#"{"metadata":{"total_size":1},"weight_map":{"a":".hidden.safetensors"}}"#.to_owned(),
            ),
            (
                "empty tensor map",
                r#"{"metadata":{"total_size":1},"weight_map":{}}"#.to_owned(),
            ),
        ] {
            std::fs::write(&index, contents).unwrap();
            let error = Checkpoint::open(dir.path(), false).unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckpointError::InvalidIndex { .. }
                        | CheckpointError::MissingShard { .. }
                        | CheckpointError::Io { .. }
                ),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn rejects_missing_shard_unexpected_tensor_wrong_shard_and_missing_tensor() {
        let dir = TestDir::new();
        std::fs::write(dir.path().join(CONFIG_FILE), "{}").unwrap();
        std::fs::write(dir.path().join(GENERATION_CONFIG_FILE), "{}").unwrap();
        write_index(dir.path(), 4, &[("a", "one.safetensors")]).unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::MissingShard { .. })
        ));

        write_shard(
            &dir.path().join("one.safetensors"),
            &[bf16("a", &[2], 4), bf16("extra", &[1], 2)],
        );
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::UnexpectedTensor { .. })
        ));

        write_shard(&dir.path().join("one.safetensors"), &[bf16("a", &[2], 4)]);
        write_shard(&dir.path().join("two.safetensors"), &[bf16("b", &[2], 4)]);
        write_index(
            dir.path(),
            8,
            &[("a", "one.safetensors"), ("b", "two.safetensors")],
        )
        .unwrap();
        std::fs::remove_file(dir.path().join("two.safetensors")).unwrap();
        // A directory is not a file, so it is reported as a missing shard.
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::MissingShard { .. })
        ));
        std::fs::create_dir(dir.path().join("two.safetensors")).unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::MissingShard { .. })
        ));
        std::fs::remove_dir(dir.path().join("two.safetensors")).unwrap();

        write_shard(&dir.path().join("two.safetensors"), &[bf16("b", &[2], 4)]);
        write_index(dir.path(), 8, &[("a", "two.safetensors")]).unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::TensorInWrongShard { .. })
                | Err(CheckpointError::UnexpectedTensor { .. })
        ));

        write_index(
            dir.path(),
            8,
            &[
                ("a", "one.safetensors"),
                ("b", "two.safetensors"),
                ("c", "two.safetensors"),
            ],
        )
        .unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::MissingTensor { .. })
        ));
    }

    #[test]
    fn rejects_total_size_mismatch_and_mixed_dtypes() {
        let dir = TestDir::new();
        std::fs::write(dir.path().join(CONFIG_FILE), "{}").unwrap();
        std::fs::write(dir.path().join(GENERATION_CONFIG_FILE), "{}").unwrap();
        write_shard(&dir.path().join("s.safetensors"), &[bf16("a", &[2], 4)]);
        write_index(dir.path(), 5, &[("a", "s.safetensors")]).unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::TotalSizeMismatch { .. })
        ));

        write_shard(
            &dir.path().join("s.safetensors"),
            &[
                bf16("a", &[2], 4),
                TensorSpec {
                    name: "b".to_owned(),
                    dtype: "F32",
                    shape: vec![1],
                    bytes: 4,
                },
            ],
        );
        write_index(
            dir.path(),
            8,
            &[("a", "s.safetensors"), ("b", "s.safetensors")],
        )
        .unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::MixedDtype { .. })
        ));
    }

    #[test]
    fn expected_inventory_for_the_pinned_configuration_has_1047_entries() {
        let config = GemmaModelConfig::from_json(include_str!(
            "../../../docs/audits/fixtures/diffusiongemma-config.json"
        ))
        .unwrap();
        let expected = expected_tensors(&config).unwrap();
        assert_eq!(expected.len(), PINNED_TENSOR_COUNT);
        assert_eq!(
            expected.get("model.decoder.layers.0.self_attn.v_proj.weight"),
            Some(&vec![2048, 2816])
        );
        assert_eq!(
            expected.get("model.decoder.layers.5.self_attn.q_proj.weight"),
            Some(&vec![8192, 2816])
        );
        assert_eq!(
            expected.get("model.decoder.layers.5.self_attn.v_proj.weight"),
            None
        );
        assert_eq!(
            expected.get("model.decoder.embed_tokens.weight"),
            Some(&vec![262_144, 2816])
        );
        assert_eq!(
            expected.get("model.encoder.vision_tower.patch_embedder.position_embedding_table"),
            Some(&vec![2, 10_240, 1152])
        );
        let bytes: u64 = expected
            .values()
            .map(|shape| shape.iter().map(|&value| value as u64).product::<u64>() * 2)
            .sum();
        assert_eq!(bytes, PINNED_TOTAL_BYTES);
    }

    #[test]
    fn structural_open_does_not_claim_pin_compliance() {
        let dir = TestDir::new();
        let spec = bf16("a.weight", &[2], 4);
        write_minimal_checkpoint(dir.path(), &spec, "shard.safetensors");
        assert!(Checkpoint::open_pinned(dir.path()).is_err());
    }

    #[test]
    fn rejects_duplicate_tensor_across_shards() {
        let dir = TestDir::new();
        std::fs::write(dir.path().join(CONFIG_FILE), "{}").unwrap();
        std::fs::write(dir.path().join(GENERATION_CONFIG_FILE), "{}").unwrap();
        write_shard(&dir.path().join("one.safetensors"), &[bf16("a", &[2], 4)]);
        write_shard(
            &dir.path().join("two.safetensors"),
            &[bf16("a", &[2], 4), bf16("b", &[2], 4)],
        );
        write_index(
            dir.path(),
            8,
            &[("a", "one.safetensors"), ("b", "two.safetensors")],
        )
        .unwrap();
        assert!(matches!(
            Checkpoint::open(dir.path(), false),
            Err(CheckpointError::TensorInWrongShard { .. })
        ));
    }
}
