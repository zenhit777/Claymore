//! Strict Clover checkpoint index and shard-header inventory.
//!
//! This module validates metadata and Safetensors headers only. It never reads
//! tensor payloads or allocates host/device tensor storage. Raw artifact hashes
//! are recorded for later pin verification; expected upstream digests remain
//! an explicit input and are not inferred from repository revision IDs.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use claymore_checkpoint::safetensors::{SafetensorsError, SafetensorsFile, WeightDType};
use claymore_checkpoint::sha256::{hex, sha256};
use claymore_checkpoint::strict_json::Json;
use thiserror::Error;

pub const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
pub const PINNED_SHARD_COUNT: usize = 26;
const INDEX_FILE: &str = "model.safetensors.index.json";
const GDN_ASSETS: [&str; 2] = [
    "bd_gdn_conv_right_weight.safetensors",
    "bd_gdn_softmax_scalars.safetensors",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloverShardIndex {
    total_size: u64,
    total_parameters: Option<u64>,
    weight_map: BTreeMap<String, String>,
    shards: BTreeSet<String>,
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
            let (_, shard_count) = parse_shard_name(shard)
                .ok_or_else(|| CloverCheckpointError::UnsafeShardName(shard.to_owned()))?;
            if declared_shard_count
                .replace(shard_count)
                .is_some_and(|old| old != shard_count)
            {
                return Err(CloverCheckpointError::InvalidIndex {
                    reason: "shard filenames declare inconsistent shard counts".to_owned(),
                });
            }
            shards.insert(shard.to_owned());
            weight_map.insert(tensor.clone(), shard.to_owned());
        }

        let shard_count =
            declared_shard_count.ok_or_else(|| CloverCheckpointError::InvalidIndex {
                reason: "weight_map has no shard entries".to_owned(),
            })? as usize;
        if shards.len() != shard_count {
            return Err(CloverCheckpointError::InvalidIndex {
                reason: format!(
                    "index references {} shards but filenames declare {shard_count}",
                    shards.len()
                ),
            });
        }
        for index in 1..=shard_count {
            let name = format!("model-{index:05}-of-{shard_count:05}.safetensors");
            if !shards.contains(&name) {
                return Err(CloverCheckpointError::InvalidIndex {
                    reason: format!("shard sequence is missing {name}"),
                });
            }
        }

        Ok(Self {
            total_size,
            total_parameters,
            weight_map,
            shards,
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
                let missing: Vec<_> = expected.difference(&actual).copied().collect();
                let unexpected: Vec<_> = actual.difference(&expected).copied().collect();
                return Err(CloverCheckpointError::ShardTensorMismatch {
                    shard: shard_name.clone(),
                    missing,
                    unexpected,
                });
            }

            for entry in shard.tensors() {
                if entry.dtype() != WeightDType::BF16 {
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
            shard_count: shards.len(),
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
    #[error("invalid GDN sidecar tensor {tensor} in {file}")]
    InvalidSidecarTensor { file: String, tensor: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

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

    #[test]
    fn inspect_validates_synthetic_shard_header_and_payload_accounting() {
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-index-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();

        let mut header =
            br#"{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.to_vec();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut shard_bytes = (header.len() as u64).to_le_bytes().to_vec();
        shard_bytes.extend_from_slice(&header);
        shard_bytes.extend_from_slice(&[0, 0, 0, 0]);
        fs::write(root.join("model-00001-of-00001.safetensors"), shard_bytes).unwrap();
        fs::write(
            root.join(INDEX_FILE),
            br#"{"metadata":{"total_size":4,"total_parameters":2},"weight_map":{"weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let inventory = CloverCheckpointInventory::inspect(&root).unwrap();
        assert_eq!(inventory.identity().shard_count, 1);
        assert_eq!(inventory.identity().tensor_count, 1);
        assert_eq!(inventory.identity().payload_bytes, 4);
        assert_eq!(inventory.identity().parameter_count, 2);
        assert_eq!(inventory.tensor("weight").unwrap().shape, vec![2]);
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
