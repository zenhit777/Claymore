//! Strict Clover checkpoint index and shard-header inventory.
//!
//! This module validates metadata and Safetensors headers only. It never reads
//! tensor payloads or allocates host/device tensor storage. Raw artifact hashes
//! are recorded for later pin verification; expected upstream digests remain
//! an explicit input and are not inferred from repository revision IDs.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use claymore_checkpoint::safetensors::{SafetensorsError, SafetensorsFile, WeightDType};
use claymore_checkpoint::sha256::{hex, sha256, sha256_file};
use claymore_checkpoint::strict_json::Json;
use thiserror::Error;

use crate::config::{
    CloverConfig, CloverConfigError, GDN_CONV_KERNEL_DIM, GDN_KEY_HEADS, GDN_KEY_HEAD_DIM,
    GDN_VALUE_HEADS, GDN_VALUE_HEAD_DIM, LAYER_COUNT,
};

pub const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
pub const PINNED_SHARD_COUNT: usize = 26;
pub const PINNED_TOTAL_SIZE_BYTES: u64 = 69_322_701_696;
const INDEX_FILE: &str = "model.safetensors.index.json";
const GDN_ASSETS: [&str; 2] = [
    "bd_gdn_conv_right_weight.safetensors",
    "bd_gdn_softmax_scalars.safetensors",
];

/// Upstream Xet SHA-256 values for all 28 indexed Safetensors artifacts.
pub const PINNED_ASSET_SHA256: [(&str, &str); 28] = [
    (
        "bd_gdn_conv_right_weight.safetensors",
        "6b27471aa645628c403ce1cba40105528f60aa9bd5b4e873b5ffaa2074ad71d4",
    ),
    (
        "bd_gdn_softmax_scalars.safetensors",
        "a8165aee2433e9b795bcc572a80c53fdfa1585a1b84534af4c20bbdb1332037a",
    ),
    (
        "model-00001-of-00026.safetensors",
        "c340eef575e465c061812a527d427f25b1cd09e850b453a936083b3d769f1751",
    ),
    (
        "model-00002-of-00026.safetensors",
        "2a24ff37d42bf33c8cefd58d96d8f502f98a3da0889d086f5eaa1648ced03610",
    ),
    (
        "model-00003-of-00026.safetensors",
        "f7d0ec384ebe97ad56b6f3468cb15bb4d55281327ddbfdbc7c359c58e2038bcd",
    ),
    (
        "model-00004-of-00026.safetensors",
        "0b88421141e09e9cec2316914bf3e5e73d62a0c03a2d4486c7c09932649ff6a0",
    ),
    (
        "model-00005-of-00026.safetensors",
        "a065d9424ab410e2ddc5e8d0d4f417282e3993d5ea612898ea64c04632c30d67",
    ),
    (
        "model-00006-of-00026.safetensors",
        "e61750b3be8c7934608df1bf4b9a9e15fdb5ae0c624899321ce1d7887323b23e",
    ),
    (
        "model-00007-of-00026.safetensors",
        "92a452f10d75a41d2e5eaaba46858dcecfec400ba8f853122460e31c1f825085",
    ),
    (
        "model-00008-of-00026.safetensors",
        "226e7fb60fbdfe85f3671ebf471f503930b721e6ff06b62c50d460ff2f5e5ca2",
    ),
    (
        "model-00009-of-00026.safetensors",
        "73ff108e69d42fc9d2353067209546b7b25833796b7e3e1d1290a995d98e4807",
    ),
    (
        "model-00010-of-00026.safetensors",
        "71b8223e3f135d9f0215f07ba2ffa376d8177ac0913d119e531728813f6e2d21",
    ),
    (
        "model-00011-of-00026.safetensors",
        "e2a6b92e994b28b7b9125f443dce256668b140dcd71f3c65cb03ac1e31552d44",
    ),
    (
        "model-00012-of-00026.safetensors",
        "221c34e566494b044813aa4b4de82350fa8ee1601323d2d52224a7db73979180",
    ),
    (
        "model-00013-of-00026.safetensors",
        "752a860ad1593e9d273d215ab7aa194ce95c35cea75f5d60498142d07a725ccf",
    ),
    (
        "model-00014-of-00026.safetensors",
        "4351cbf05ee65e8c0b8a4316324423e2bb9874fd98b062f9ad02bd1cd34c5319",
    ),
    (
        "model-00015-of-00026.safetensors",
        "b6a1f0d82bb6145632e369a1de85adde3446cd721b5cced28b4a037f41d34fe6",
    ),
    (
        "model-00016-of-00026.safetensors",
        "10ada7f1814b1ef6da75e913e946af05b135bef1cc5c8e874c11e03f238f76e7",
    ),
    (
        "model-00017-of-00026.safetensors",
        "e598e5845299541f4c38fa0bb80e20e2c39c525e9676880e3b2477e9740df939",
    ),
    (
        "model-00018-of-00026.safetensors",
        "cdaa56291ae23785a489f9e9fbd063fe25e9b1a7e228d787b910a2e3559ba3eb",
    ),
    (
        "model-00019-of-00026.safetensors",
        "66550744d97ffa0b9f8f0cad1b66a7a1426ed0b99404ae0731e51e622ca705d1",
    ),
    (
        "model-00020-of-00026.safetensors",
        "4d9a696b1efc80ef80554cbfb09629e9355e8082bf7eaae7053b67a9eb9556d1",
    ),
    (
        "model-00021-of-00026.safetensors",
        "96cf4f331f51936dac903f9e8726c2d4125b7fda97bebd2bee34eefa319a2ceb",
    ),
    (
        "model-00022-of-00026.safetensors",
        "d5e11497960fe856c241662ed6e1bb0bda234baf42b8b7f09421c6bfe35a63e3",
    ),
    (
        "model-00023-of-00026.safetensors",
        "d9cf73995d153079fdec0e87f83c711914457331139f11f06fa312b15f439308",
    ),
    (
        "model-00024-of-00026.safetensors",
        "adcf9389c87d8e95d76cad82d35100317d164331cce2fe9ddecf3d5c68f372f9",
    ),
    (
        "model-00025-of-00026.safetensors",
        "52990ce8792a4cda271a9aad4135ff4f5707c9b91c3051afe1d7ffae071be593",
    ),
    (
        "model-00026-of-00026.safetensors",
        "7fb786f9db4227edd863e0cc00f777f454bff460671cda3dd63f6faec2f79592",
    ),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloverShardIndex {
    total_size: u64,
    total_parameters: Option<u64>,
    weight_map: BTreeMap<String, String>,
    shards: BTreeSet<String>,
    model_shards: BTreeSet<String>,
    sidecars: BTreeSet<String>,
    shard_count: usize,
}

impl CloverShardIndex {
    pub fn parse(bytes: &[u8]) -> Result<Self, CloverCheckpointError> {
        let root = Json::parse(bytes).map_err(|message| CloverCheckpointError::InvalidIndex {
            reason: format!("invalid JSON: {message}"),
        })?;
        let fields = root
            .as_object()
            .ok_or_else(|| CloverCheckpointError::InvalidIndex {
                reason: format!("root must be an object, got {}", root.type_name()),
            })?;
        require_only_fields(fields, &["metadata", "weight_map"], "index root")?;

        let metadata = root
            .get("metadata")
            .and_then(Json::as_object)
            .ok_or_else(|| CloverCheckpointError::InvalidIndex {
                reason: "metadata must be an object".to_owned(),
            })?;
        require_only_fields(
            metadata,
            &["total_size", "total_parameters"],
            "index metadata",
        )?;

        let total_size = root
            .get("metadata")
            .and_then(|value| value.get("total_size"))
            .and_then(Json::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| CloverCheckpointError::InvalidIndex {
                reason: "metadata.total_size must be a non-zero unsigned integer".to_owned(),
            })?;
        let total_parameters = match root
            .get("metadata")
            .and_then(|value| value.get("total_parameters"))
        {
            None => None,
            Some(value) => Some(value.as_u64().filter(|value| *value > 0).ok_or_else(|| {
                CloverCheckpointError::InvalidIndex {
                    reason: "metadata.total_parameters must be a non-zero unsigned integer"
                        .to_owned(),
                }
            })?),
        };

        let entries = root
            .get("weight_map")
            .and_then(Json::as_object)
            .ok_or_else(|| CloverCheckpointError::InvalidIndex {
                reason: "weight_map must be an object".to_owned(),
            })?;
        if entries.is_empty() {
            return Err(CloverCheckpointError::InvalidIndex {
                reason: "weight_map must not be empty".to_owned(),
            });
        }

        let mut weight_map = BTreeMap::new();
        let mut shards = BTreeSet::new();
        let mut model_shards = BTreeSet::new();
        let mut sidecars = BTreeSet::new();
        let mut declared_shard_count = None;
        for (tensor, value) in entries {
            if tensor.is_empty() || tensor.contains('\0') {
                return Err(CloverCheckpointError::InvalidIndex {
                    reason: "weight_map contains an empty or invalid tensor name".to_owned(),
                });
            }
            let shard = value
                .as_str()
                .ok_or_else(|| CloverCheckpointError::InvalidIndex {
                    reason: format!("weight_map entry {tensor:?} must be a string"),
                })?;
            if GDN_ASSETS.contains(&shard) {
                sidecars.insert(shard.to_owned());
            } else {
                let (_, declared_count) = parse_shard_name(shard)
                    .ok_or_else(|| CloverCheckpointError::UnsafeShardName(shard.to_owned()))?;
                if declared_shard_count
                    .replace(declared_count)
                    .is_some_and(|old| old != declared_count)
                {
                    return Err(CloverCheckpointError::InvalidIndex {
                        reason: "model shard filenames declare inconsistent shard counts"
                            .to_owned(),
                    });
                }
                model_shards.insert(shard.to_owned());
            }
            shards.insert(shard.to_owned());
            weight_map.insert(tensor.clone(), shard.to_owned());
        }

        let shard_count =
            declared_shard_count.ok_or_else(|| CloverCheckpointError::InvalidIndex {
                reason: "weight_map has no model shard entries".to_owned(),
            })?;
        if model_shards.len() != shard_count {
            return Err(CloverCheckpointError::InvalidIndex {
                reason: format!(
                    "index references {} model shards but filenames declare {shard_count}",
                    model_shards.len()
                ),
            });
        }
        for index in 1..=shard_count {
            let name = format!("model-{index:05}-of-{shard_count:05}.safetensors");
            if !model_shards.contains(&name) {
                return Err(CloverCheckpointError::InvalidIndex {
                    reason: format!("model shard sequence is missing {name}"),
                });
            }
        }
        let expected_sidecars: BTreeSet<String> =
            GDN_ASSETS.iter().map(|name| (*name).to_owned()).collect();
        if sidecars != expected_sidecars {
            return Err(CloverCheckpointError::InvalidIndex {
                reason: format!(
                    "GDN sidecar set mismatch: expected {expected_sidecars:?}, got {sidecars:?}"
                ),
            });
        }

        Ok(Self {
            total_size,
            total_parameters,
            weight_map,
            shards,
            model_shards,
            sidecars,
            shard_count,
        })
    }

    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    pub fn total_parameters(&self) -> Option<u64> {
        self.total_parameters
    }

    pub fn weight_map(&self) -> &BTreeMap<String, String> {
        &self.weight_map
    }

    pub fn shards(&self) -> &BTreeSet<String> {
        &self.shards
    }

    pub fn shard_count(&self) -> usize {
        self.shard_count
    }

    pub fn model_shards(&self) -> &BTreeSet<String> {
        &self.model_shards
    }

    pub fn sidecars(&self) -> &BTreeSet<String> {
        &self.sidecars
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSummary {
    pub name: String,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSummary {
    pub shard: String,
    pub shape: Vec<usize>,
    pub dtype: WeightDType,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloverInventoryIdentity {
    pub index_sha256: String,
    pub shard_count: usize,
    pub sidecar_count: usize,
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub parameter_count: u64,
}

pub struct CloverCheckpointInventory {
    root: PathBuf,
    index: CloverShardIndex,
    identity: CloverInventoryIdentity,
    shards: BTreeMap<String, SafetensorsFile>,
    summaries: BTreeMap<String, TensorSummary>,
    shard_summaries: Vec<ShardSummary>,
}

impl std::fmt::Debug for CloverCheckpointInventory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CloverCheckpointInventory")
            .field("root", &self.root)
            .field("identity", &self.identity)
            .finish()
    }
}

impl CloverCheckpointInventory {
    /// Open a pinned Clover inventory only after validating its model config,
    /// all indexed Safetensors headers, release topology and exact GDN sidecars.
    pub fn inspect_pinned(
        root: impl AsRef<Path>,
        config: &CloverConfig,
    ) -> Result<Self, CloverCheckpointError> {
        config.validate_pinned()?;
        let inventory = Self::inspect(root)?;
        inventory.validate_pinned_geometry()?;
        inventory.validate_pinned_tensor_schema()?;
        Ok(inventory)
    }

    /// Inspect the index and every shard header without reading tensor payloads.
    pub fn inspect(root: impl AsRef<Path>) -> Result<Self, CloverCheckpointError> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(CloverCheckpointError::MissingDirectory(root));
        }
        let index_path = root.join(INDEX_FILE);
        let metadata = fs::metadata(&index_path).map_err(|source| io_error(&index_path, source))?;
        if metadata.len() == 0 || metadata.len() > MAX_INDEX_BYTES {
            return Err(CloverCheckpointError::IndexSize {
                actual: metadata.len(),
                maximum: MAX_INDEX_BYTES,
            });
        }
        let bytes = fs::read(&index_path).map_err(|source| io_error(&index_path, source))?;
        let index_sha256 = hex(&sha256(&bytes));
        let index = CloverShardIndex::parse(&bytes)?;

        let mut shards = BTreeMap::new();
        let mut summaries = BTreeMap::new();
        let mut shard_summaries = Vec::new();
        let mut payload_bytes = 0u64;
        let mut parameter_count = 0u64;

        for shard_name in index.shards() {
            let path = root.join(shard_name);
            if !path.is_file() {
                return Err(CloverCheckpointError::MissingShard {
                    name: shard_name.clone(),
                    path,
                });
            }
            let shard = SafetensorsFile::open(&path)?;
            let expected: BTreeSet<&str> = index
                .weight_map()
                .iter()
                .filter_map(|(tensor, mapped_shard)| {
                    (mapped_shard == shard_name).then_some(tensor.as_str())
                })
                .collect();
            let actual: BTreeSet<&str> = shard.tensors().map(|entry| entry.name()).collect();
            if expected != actual {
                let missing: Vec<_> = expected
                    .difference(&actual)
                    .map(|name| (*name).to_owned())
                    .collect();
                let unexpected: Vec<_> = actual
                    .difference(&expected)
                    .map(|name| (*name).to_owned())
                    .collect();
                return Err(CloverCheckpointError::ShardTensorMismatch {
                    shard: shard_name.clone(),
                    missing,
                    unexpected,
                });
            }

            for entry in shard.tensors() {
                let is_sidecar = GDN_ASSETS.contains(&shard_name.as_str());
                if !is_sidecar && entry.dtype() != WeightDType::BF16 {
                    return Err(CloverCheckpointError::UnsupportedWeightDType {
                        tensor: entry.name().to_owned(),
                        actual: entry.dtype().name(),
                    });
                }
                let expected_bytes = entry
                    .numel()
                    .checked_mul(entry.dtype().element_size_bytes() as u64)
                    .ok_or(CloverCheckpointError::SizeOverflow)?;
                if expected_bytes != entry.byte_len() {
                    return Err(CloverCheckpointError::TensorByteLength {
                        tensor: entry.name().to_owned(),
                        expected: expected_bytes,
                        actual: entry.byte_len(),
                    });
                }
                if summaries
                    .insert(
                        entry.name().to_owned(),
                        TensorSummary {
                            shard: shard_name.clone(),
                            shape: entry.shape().to_vec(),
                            dtype: entry.dtype(),
                            byte_len: entry.byte_len(),
                        },
                    )
                    .is_some()
                {
                    return Err(CloverCheckpointError::DuplicateTensor(
                        entry.name().to_owned(),
                    ));
                }
                payload_bytes = payload_bytes
                    .checked_add(entry.byte_len())
                    .ok_or(CloverCheckpointError::SizeOverflow)?;
                parameter_count = parameter_count
                    .checked_add(entry.numel())
                    .ok_or(CloverCheckpointError::SizeOverflow)?;
            }

            shard_summaries.push(ShardSummary {
                name: shard_name.clone(),
                file_bytes: shard.file_len(),
                header_bytes: shard.header_len(),
                payload_bytes: shard.data_len(),
                tensor_count: shard.tensor_count(),
            });
            shards.insert(shard_name.clone(), shard);
        }

        if payload_bytes != index.total_size() {
            return Err(CloverCheckpointError::TotalSizeMismatch {
                index: index.total_size(),
                actual: payload_bytes,
            });
        }
        if let Some(declared) = index.total_parameters() {
            if declared != parameter_count {
                return Err(CloverCheckpointError::ParameterCountMismatch {
                    index: declared,
                    actual: parameter_count,
                });
            }
        }

        let identity = CloverInventoryIdentity {
            index_sha256,
            shard_count: index.shard_count(),
            sidecar_count: index.sidecars().len(),
            tensor_count: summaries.len(),
            payload_bytes,
            parameter_count,
        };

        Ok(Self {
            root,
            index,
            identity,
            shards,
            summaries,
            shard_summaries,
        })
    }

    /// Apply the initial pinned Clover release geometry after a full inspection.
    pub fn validate_pinned_geometry(&self) -> Result<(), CloverCheckpointError> {
        if self.identity.shard_count != PINNED_SHARD_COUNT {
            return Err(CloverCheckpointError::PinnedShardCount {
                expected: PINNED_SHARD_COUNT,
                actual: self.identity.shard_count,
            });
        }
        if self.identity.sidecar_count != GDN_ASSETS.len() {
            return Err(CloverCheckpointError::PinnedSidecarCount {
                expected: GDN_ASSETS.len(),
                actual: self.identity.sidecar_count,
            });
        }
        if self.identity.payload_bytes != PINNED_TOTAL_SIZE_BYTES {
            return Err(CloverCheckpointError::PinnedTotalSize {
                expected: PINNED_TOTAL_SIZE_BYTES,
                actual: self.identity.payload_bytes,
            });
        }

        let channels = 2 * GDN_KEY_HEADS * GDN_KEY_HEAD_DIM + GDN_VALUE_HEADS * GDN_VALUE_HEAD_DIM;
        let conv_shape = vec![channels, 1, GDN_CONV_KERNEL_DIM - 1];
        let scalar_shape = vec![GDN_VALUE_HEADS];
        let mut expected_sidecars = BTreeMap::new();
        for layer in 0..LAYER_COUNT {
            // Clover uses three GDN layers followed by one full-attention layer.
            if layer % 4 == 3 {
                continue;
            }
            expected_sidecars.insert(
                format!("model.layers.{layer}.linear_attn.conv_right_weight"),
                (GDN_ASSETS[0], conv_shape.clone()),
            );
            for scalar in [
                "bd_softmax_logit_scale",
                "bd_softmax_history_bias",
                "bd_softmax_history_c",
            ] {
                expected_sidecars.insert(
                    format!("model.layers.{layer}.linear_attn.{scalar}"),
                    (GDN_ASSETS[1], scalar_shape.clone()),
                );
            }
        }

        let actual_sidecars: BTreeSet<String> = self
            .summaries
            .iter()
            .filter_map(|(name, tensor)| {
                GDN_ASSETS
                    .contains(&tensor.shard.as_str())
                    .then_some(name.clone())
            })
            .collect();
        let expected_names: BTreeSet<String> = expected_sidecars.keys().cloned().collect();
        if actual_sidecars != expected_names {
            return Err(CloverCheckpointError::PinnedSidecarTensorSetMismatch {
                missing: expected_names
                    .difference(&actual_sidecars)
                    .cloned()
                    .collect(),
                unexpected: actual_sidecars
                    .difference(&expected_names)
                    .cloned()
                    .collect(),
            });
        }
        for (name, (expected_file, expected_shape)) in expected_sidecars {
            let actual = self
                .summaries
                .get(&name)
                .expect("sidecar tensor set validated above");
            if actual.shard != expected_file || actual.shape != expected_shape {
                return Err(CloverCheckpointError::PinnedSidecarTensorMismatch {
                    tensor: name,
                    expected_file: expected_file.to_owned(),
                    actual_file: actual.shard.clone(),
                    expected_shape,
                    actual_shape: actual.shape.clone(),
                });
            }
        }
        Ok(())
    }


    /// Validate every model tensor against the pinned Python module structure.
    ///
    /// The expected names and shapes are derived from modeling_clover_moe.py at
    /// CONFIG_REVISION. This is a metadata-only check; no tensor payload is read.
    pub fn validate_pinned_tensor_schema(&self) -> Result<(), CloverCheckpointError> {
        let hidden = 2048usize;
        let vocab = 248_320usize;
        let experts = 256usize;
        let top_k = 8usize;
        let moe_intermediate = 512usize;
        let shared_intermediate = 512usize;
        let q_heads = 16usize;
        let kv_heads = 2usize;
        let head_dim = 256usize;
        let gdn_channels = 2 * GDN_KEY_HEADS * GDN_KEY_HEAD_DIM
            + GDN_VALUE_HEADS * GDN_VALUE_HEAD_DIM;
        let gdn_value_width = GDN_VALUE_HEADS * GDN_VALUE_HEAD_DIM;
        let mut expected: BTreeMap<String, (Vec<usize>, Option<&'static str>)> =
            BTreeMap::new();

        let mut add = |name: String, shape: Vec<usize>, sidecar: Option<&'static str>| {
            expected.insert(name, (shape, sidecar));
        };
        add("model.embed_tokens.weight".into(), vec![vocab, hidden], None);
        add("model.norm.weight".into(), vec![hidden], None);
        add("lm_head.weight".into(), vec![vocab, hidden], None);

        for layer in 0..LAYER_COUNT {
            let prefix = format!("model.layers.{layer}");
            add(format!("{prefix}.input_layernorm.weight"), vec![hidden], None);
            add(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
                None,
            );

            if layer % 4 == 3 {
                let attn = format!("{prefix}.self_attn");
                add(
                    format!("{attn}.q_proj.weight"),
                    vec![2 * q_heads * head_dim, hidden],
                    None,
                );
                add(
                    format!("{attn}.k_proj.weight"),
                    vec![kv_heads * head_dim, hidden],
                    None,
                );
                add(
                    format!("{attn}.v_proj.weight"),
                    vec![kv_heads * head_dim, hidden],
                    None,
                );
                add(
                    format!("{attn}.o_proj.weight"),
                    vec![hidden, q_heads * head_dim],
                    None,
                );
                add(format!("{attn}.q_norm.weight"), vec![head_dim], None);
                add(format!("{attn}.k_norm.weight"), vec![head_dim], None);
            } else {
                let gdn = format!("{prefix}.linear_attn");
                add(
                    format!("{gdn}.conv1d.weight"),
                    vec![gdn_channels, 1, GDN_CONV_KERNEL_DIM],
                    None,
                );
                add(
                    format!("{gdn}.in_proj_qkv.weight"),
                    vec![gdn_channels, hidden],
                    None,
                );
                add(
                    format!("{gdn}.in_proj_z.weight"),
                    vec![gdn_value_width, hidden],
                    None,
                );
                add(
                    format!("{gdn}.in_proj_a.weight"),
                    vec![GDN_VALUE_HEADS, hidden],
                    None,
                );
                add(
                    format!("{gdn}.in_proj_b.weight"),
                    vec![GDN_VALUE_HEADS, hidden],
                    None,
                );
                add(format!("{gdn}.A_log"), vec![GDN_VALUE_HEADS], None);
                add(format!("{gdn}.dt_bias"), vec![GDN_VALUE_HEADS], None);
                add(
                    format!("{gdn}.norm.weight"),
                    vec![GDN_VALUE_HEAD_DIM],
                    None,
                );
                add(
                    format!("{gdn}.out_proj.weight"),
                    vec![hidden, gdn_value_width],
                    None,
                );
                add(
                    format!("{gdn}.conv_right_weight"),
                    vec![gdn_channels, 1, GDN_CONV_KERNEL_DIM - 1],
                    Some(GDN_ASSETS[0]),
                );
                for scalar in [
                    "bd_softmax_logit_scale",
                    "bd_softmax_history_bias",
                    "bd_softmax_history_c",
                ] {
                    add(
                        format!("{gdn}.{scalar}"),
                        vec![GDN_VALUE_HEADS],
                        Some(GDN_ASSETS[1]),
                    );
                }
            }

            let mlp = format!("{prefix}.mlp");
            add(
                format!("{mlp}.gate.weight"),
                vec![experts, hidden],
                None,
            );
            add(
                format!("{mlp}.experts.gate_up_proj"),
                vec![experts, 2 * moe_intermediate, hidden],
                None,
            );
            add(
                format!("{mlp}.experts.down_proj"),
                vec![experts, hidden, moe_intermediate],
                None,
            );
            add(
                format!("{mlp}.shared_expert.gate_proj.weight"),
                vec![shared_intermediate, hidden],
                None,
            );
            add(
                format!("{mlp}.shared_expert.up_proj.weight"),
                vec![shared_intermediate, hidden],
                None,
            );
            add(
                format!("{mlp}.shared_expert.down_proj.weight"),
                vec![hidden, shared_intermediate],
                None,
            );
            add(
                format!("{mlp}.shared_expert_gate.weight"),
                vec![1, hidden],
                None,
            );
        }

        let actual_names: BTreeSet<&str> = self.summaries.keys().map(String::as_str).collect();
        let expected_names: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
        if actual_names != expected_names {
            return Err(CloverCheckpointError::PinnedTensorSetMismatch {
                missing: expected_names
                    .difference(&actual_names)
                    .map(|name| (*name).to_owned())
                    .collect(),
                unexpected: actual_names
                    .difference(&expected_names)
                    .map(|name| (*name).to_owned())
                    .collect(),
            });
        }

        for (name, (shape, expected_sidecar)) in expected {
            let actual = self.summaries.get(&name).expect("tensor set checked above");
            if actual.shape != shape {
                return Err(CloverCheckpointError::PinnedTensorShapeMismatch {
                    tensor: name,
                    expected: shape,
                    actual: actual.shape.clone(),
                });
            }
            if let Some(sidecar) = expected_sidecar {
                if actual.shard != sidecar {
                    return Err(CloverCheckpointError::PinnedTensorShardMismatch {
                        tensor: name,
                        expected: sidecar.to_owned(),
                        actual: actual.shard.clone(),
                    });
                }
            } else if GDN_ASSETS.contains(&actual.shard.as_str()) {
                return Err(CloverCheckpointError::PinnedTensorShardMismatch {
                    tensor: name,
                    expected: "one of the 26 model shards".to_owned(),
                    actual: actual.shard.clone(),
                });
            } else if actual.dtype != WeightDType::BF16 {
                return Err(CloverCheckpointError::UnsupportedWeightDType {
                    tensor: name,
                    actual: actual.dtype.name(),
                });
            }
        }

        let _ = top_k; // Top-k is a routing semantic, not a stored tensor dimension.
        Ok(())
    }

    /// Stream and verify the SHA-256 of all 26 model shards and both GDN sidecars.
    ///
    /// This reads the complete 69.3 GB checkpoint from local storage in bounded
    /// chunks. It does not allocate model tensors or initialize a device.
    pub fn verify_pinned_asset_hashes(&self) -> Result<(), CloverCheckpointError> {
        for (name, expected) in PINNED_ASSET_SHA256 {
            let path = self.root.join(name);
            let digest = sha256_file(&path).map_err(|source| io_error(&path, source))?;
            let actual = hex(&digest);
            if actual != expected {
                return Err(CloverCheckpointError::HashMismatch {
                    file: name.to_owned(),
                    expected: expected.to_owned(),
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Validate the two GDN sidecar files separately from the indexed 26 shards.
    pub fn inspect_gdn_sidecars(
        root: impl AsRef<Path>,
    ) -> Result<Vec<ShardSummary>, CloverCheckpointError> {
        let root = root.as_ref();
        let mut result = Vec::with_capacity(GDN_ASSETS.len());
        for name in GDN_ASSETS {
            let path = root.join(name);
            if !path.is_file() {
                return Err(CloverCheckpointError::MissingShard {
                    name: name.to_owned(),
                    path,
                });
            }
            let file = SafetensorsFile::open(&path)?;
            for entry in file.tensors() {
                if entry.numel() == 0 {
                    return Err(CloverCheckpointError::InvalidSidecarTensor {
                        file: name.to_owned(),
                        tensor: entry.name().to_owned(),
                    });
                }
            }
            result.push(ShardSummary {
                name: name.to_owned(),
                file_bytes: file.file_len(),
                header_bytes: file.header_len(),
                payload_bytes: file.data_len(),
                tensor_count: file.tensor_count(),
            });
        }
        Ok(result)
    }

    pub fn index(&self) -> &CloverShardIndex {
        &self.index
    }

    pub fn identity(&self) -> &CloverInventoryIdentity {
        &self.identity
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorSummary> {
        self.summaries.get(name)
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorSummary> {
        &self.summaries
    }

    pub fn shard_summaries(&self) -> &[ShardSummary] {
        &self.shard_summaries
    }

    pub fn shard(&self, name: &str) -> Option<&SafetensorsFile> {
        self.shards.get(name)
    }
}

fn require_only_fields(
    fields: &[(String, Json)],
    allowed: &[&str],
    scope: &str,
) -> Result<(), CloverCheckpointError> {
    for (name, _) in fields {
        if !allowed.contains(&name.as_str()) {
            return Err(CloverCheckpointError::InvalidIndex {
                reason: format!("unknown field {name:?} in {scope}"),
            });
        }
    }
    Ok(())
}

fn parse_shard_name(name: &str) -> Option<(usize, usize)> {
    let bytes = name.as_bytes();
    if bytes.len() != "model-00001-of-00026.safetensors".len()
        || !name.starts_with("model-")
        || !name.ends_with(".safetensors")
    {
        return None;
    }
    let index = name.get(6..11)?.parse::<usize>().ok()?;
    let count = name.get(15..20)?.parse::<usize>().ok()?;
    if index == 0 || count == 0 || index > count {
        return None;
    }
    if !bytes[6..11].iter().all(u8::is_ascii_digit)
        || !bytes[15..20].iter().all(u8::is_ascii_digit)
        || &name[11..15] != "-of-"
    {
        return None;
    }
    Some((index, count))
}

fn io_error(path: &Path, source: std::io::Error) -> CloverCheckpointError {
    CloverCheckpointError::Io {
        path: path.to_path_buf(),
        message: source.to_string(),
    }
}

#[derive(Debug, Error)]
pub enum CloverCheckpointError {
    #[error("invalid Clover checkpoint index: {reason}")]
    InvalidIndex { reason: String },
    #[error("checkpoint directory does not exist: {0}")]
    MissingDirectory(PathBuf),
    #[error("checkpoint index size {actual} is invalid (maximum {maximum})")]
    IndexSize { actual: u64, maximum: u64 },
    #[error("unsafe or unsupported shard name: {0}")]
    UnsafeShardName(String),
    #[error("missing checkpoint shard {name}: {path}")]
    MissingShard { name: String, path: PathBuf },
    #[error("I/O error for {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error(transparent)]
    Safetensors(#[from] SafetensorsError),
    #[error(transparent)]
    Config(#[from] CloverConfigError),
    #[error("tensor {0} is listed more than once across shards")]
    DuplicateTensor(String),
    #[error("shard {shard} tensor inventory differs from the index (missing={missing:?}, unexpected={unexpected:?})")]
    ShardTensorMismatch {
        shard: String,
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    #[error("tensor {tensor} has unsupported dtype {actual}; BF16 is required for model shards")]
    UnsupportedWeightDType {
        tensor: String,
        actual: &'static str,
    },
    #[error("tensor {tensor} byte length mismatch: expected {expected}, got {actual}")]
    TensorByteLength {
        tensor: String,
        expected: u64,
        actual: u64,
    },
    #[error("checkpoint payload size mismatch: index declares {index}, shards contain {actual}")]
    TotalSizeMismatch { index: u64, actual: u64 },
    #[error(
        "checkpoint parameter count mismatch: index declares {index}, shards contain {actual}"
    )]
    ParameterCountMismatch { index: u64, actual: u64 },
    #[error("integer overflow while inspecting checkpoint")]
    SizeOverflow,
    #[error("pinned Clover requires {expected} model shards, found {actual}")]
    PinnedShardCount { expected: usize, actual: usize },
    #[error("pinned Clover requires {expected} GDN sidecars, found {actual}")]
    PinnedSidecarCount { expected: usize, actual: usize },
    #[error("pinned Clover requires {expected} payload bytes, found {actual}")]
    PinnedTotalSize { expected: u64, actual: u64 },
    #[error(
        "pinned GDN sidecar tensor set mismatch (missing={missing:?}, unexpected={unexpected:?})"
    )]
    PinnedSidecarTensorSetMismatch {
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    #[error("pinned GDN tensor {tensor} mismatch: expected {expected_file} {expected_shape:?}, got {actual_file} {actual_shape:?}")]
    PinnedSidecarTensorMismatch {
        tensor: String,
        expected_file: String,
        actual_file: String,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },
    #[error("invalid GDN sidecar tensor {tensor} in {file}")]
    InvalidSidecarTensor { file: String, tensor: String },
    #[error("pinned model tensor set mismatch (missing={missing:?}, unexpected={unexpected:?})")]
    PinnedTensorSetMismatch { missing: Vec<String>, unexpected: Vec<String> },
    #[error("pinned tensor {tensor} shape mismatch: expected {expected:?}, got {actual:?}")]
    PinnedTensorShapeMismatch { tensor: String, expected: Vec<usize>, actual: Vec<usize> },
    #[error("pinned tensor {tensor} is in the wrong shard: expected {expected}, got {actual}")]
    PinnedTensorShardMismatch { tensor: String, expected: String, actual: String },
    #[error("SHA-256 mismatch for {file}: expected {expected}, got {actual}")]
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn pinned_digest_manifest_covers_all_weight_artifacts() {
        let mut names = BTreeSet::new();
        let mut digests = BTreeSet::new();
        for (name, digest) in PINNED_ASSET_SHA256 {
            assert!(names.insert(name));
            assert!(digests.insert(digest));
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }

        for name in GDN_ASSETS {
            assert!(names.contains(name));
        }
        for index in 1..=PINNED_SHARD_COUNT {
            let name = format!("model-{index:05}-of-{PINNED_SHARD_COUNT:05}.safetensors");
            assert!(names.contains(name.as_str()));
        }
        assert_eq!(names.len(), 28);
    }

    #[test]
    fn index_parser_rejects_path_traversal_and_duplicate_json_keys() {
        let traversal = br#"{"metadata":{"total_size":2},"weight_map":{"x":"../model-00001-of-00001.safetensors"}}"#;
        assert!(matches!(
            CloverShardIndex::parse(traversal),
            Err(CloverCheckpointError::UnsafeShardName(_))
        ));

        let duplicate = br#"{"metadata":{"total_size":2,"total_size":4},"weight_map":{"x":"model-00001-of-00001.safetensors"}}"#;
        assert!(matches!(
            CloverShardIndex::parse(duplicate),
            Err(CloverCheckpointError::InvalidIndex { .. })
        ));
    }

    fn write_synthetic_shard(root: &Path, filename: &str, tensor: &str) {
        let header_json =
            format!(r#"{{"{tensor}":{{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}}}"#);
        let mut header = header_json.into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        fs::write(root.join(filename), bytes).unwrap();
    }

    #[test]
    fn inspect_validates_model_shard_and_indexed_gdn_sidecars() {
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-index-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();

        write_synthetic_shard(&root, "model-00001-of-00001.safetensors", "weight");
        write_synthetic_shard(
            &root,
            "bd_gdn_conv_right_weight.safetensors",
            "model.layers.0.linear_attn.conv_right_weight",
        );
        write_synthetic_shard(
            &root,
            "bd_gdn_softmax_scalars.safetensors",
            "model.layers.0.linear_attn.bd_softmax_logit_scale",
        );
        fs::write(
            root.join(INDEX_FILE),
            br#"{"metadata":{"total_size":12,"total_parameters":6},"weight_map":{"weight":"model-00001-of-00001.safetensors","model.layers.0.linear_attn.conv_right_weight":"bd_gdn_conv_right_weight.safetensors","model.layers.0.linear_attn.bd_softmax_logit_scale":"bd_gdn_softmax_scalars.safetensors"}}"#,
        )
        .unwrap();

        let inventory = CloverCheckpointInventory::inspect(&root).unwrap();
        assert_eq!(inventory.identity().shard_count, 1);
        assert_eq!(inventory.identity().sidecar_count, 2);
        assert_eq!(inventory.identity().tensor_count, 3);
        assert_eq!(inventory.identity().payload_bytes, 12);
        assert_eq!(inventory.identity().parameter_count, 6);
        assert_eq!(inventory.tensor("weight").unwrap().shape, vec![2]);
        assert_eq!(inventory.shard_summaries().len(), 3);
        assert_eq!(
            CloverCheckpointInventory::inspect_gdn_sidecars(&root)
                .unwrap()
                .len(),
            2
        );
        assert!(matches!(
            inventory.validate_pinned_geometry(),
            Err(CloverCheckpointError::PinnedShardCount {
                expected: 26,
                actual: 1
            })
        ));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shard_name_parser_requires_a_complete_safe_sequence() {
        assert_eq!(
            parse_shard_name("model-00001-of-00026.safetensors"),
            Some((1, 26))
        );
        assert_eq!(
            parse_shard_name("model-00026-of-00026.safetensors"),
            Some((26, 26))
        );
        assert_eq!(
            parse_shard_name("../model-00001-of-00026.safetensors"),
            None
        );
        assert_eq!(parse_shard_name("model-00000-of-00026.safetensors"), None);
    }
}
