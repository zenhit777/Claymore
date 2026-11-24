//! Bounded, fail-closed Safetensors reader for the pinned DiffusionGemma
//! checkpoint.
//!
//! The reader validates the whole header before returning a handle: duplicate
//! names, unsupported dtypes, malformed shapes, arithmetic overflow, offsets
//! outside the data region, misalignment, overlapping tensors and gaps are all
//! rejected. Tensor payloads are read on demand with a bounded byte count and
//! never mapped or copied implicitly.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::strict_json::Json;

/// Maximum accepted Safetensors header. The pinned shards use headers below
/// 64 KiB; this bound prevents a corrupt length from allocating arbitrarily.
pub const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Largest payload range read in one call. Full-shard verification streams the
/// payload in chunks of at most this size, so no whole-shard copy is made.
pub const MAX_CHUNK_BYTES: usize = 8 * 1024 * 1024;

const LENGTH_PREFIX_BYTES: u64 = 8;

/// Weight dtypes accepted by the M10 execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDType {
    F32,
    BF16,
}

impl WeightDType {
    pub const fn element_size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::BF16 => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::BF16 => "BF16",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(Self::F32),
            "BF16" => Some(Self::BF16),
            _ => None,
        }
    }
}

/// One validated tensor entry, with offsets relative to the tensor data region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    name: String,
    dtype: WeightDType,
    shape: Vec<usize>,
    data_start: u64,
    data_end: u64,
}

impl TensorEntry {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> WeightDType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of logical elements, already checked for overflow at open time.
    pub fn numel(&self) -> u64 {
        // Validated in `parse_tensor`.
        self.shape
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim as u64))
            .expect("validated tensor shape")
    }

    pub fn byte_len(&self) -> u64 {
        self.data_end - self.data_start
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn data_end(&self) -> u64 {
        self.data_end
    }
}

/// Open, validated Safetensors shard.
pub struct SafetensorsFile {
    path: PathBuf,
    file: File,
    file_len: u64,
    header_len: u64,
    data_offset: u64,
    data_len: u64,
    tensors: BTreeMap<String, TensorEntry>,
    metadata: BTreeMap<String, String>,
}

impl std::fmt::Debug for SafetensorsFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SafetensorsFile")
            .field("path", &self.path)
            .field("file_len", &self.file_len)
            .field("header_len", &self.header_len)
            .field("data_len", &self.data_len)
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| io_error(&path, "open", source))?;
        let file_len = file
            .metadata()
            .map_err(|source| io_error(&path, "metadata", source))?
            .len();
        if file_len < LENGTH_PREFIX_BYTES {
            return Err(SafetensorsError::FileTooShort {
                path,
                file_len,
                minimum: LENGTH_PREFIX_BYTES,
            });
        }

        let mut prefix = [0u8; 8];
        read_exact_at(&file, &mut prefix, 0)
            .map_err(|source| io_error(&path, "read length prefix", source))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 {
            return Err(SafetensorsError::EmptyHeader { path });
        }
        if header_len > MAX_HEADER_BYTES {
            return Err(SafetensorsError::HeaderTooLarge {
                path,
                header_len,
                maximum: MAX_HEADER_BYTES,
            });
        }
        let data_offset = LENGTH_PREFIX_BYTES
            .checked_add(header_len)
            .ok_or(SafetensorsError::SizeOverflow)?;
        if data_offset > file_len {
            return Err(SafetensorsError::TruncatedHeader {
                path,
                header_len,
                file_len,
            });
        }
        let data_len = file_len - data_offset;
        let header_len_usize =
            usize::try_from(header_len).map_err(|_| SafetensorsError::SizeOverflow)?;
        let mut header = vec![0u8; header_len_usize];
        read_exact_at(&file, &mut header, LENGTH_PREFIX_BYTES)
            .map_err(|source| io_error(&path, "read header", source))?;

        let root = Json::parse(&header).map_err(|message| SafetensorsError::InvalidJson {
            path: path.clone(),
            message,
        })?;
        let entries = root
            .as_object()
            .ok_or_else(|| SafetensorsError::InvalidRoot {
                path: path.clone(),
                actual: root.type_name(),
            })?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, value) in entries {
            if name == "__metadata__" {
                metadata = parse_metadata(&path, value)?;
                continue;
            }
            if name.is_empty() {
                return Err(SafetensorsError::EmptyTensorName { path });
            }
            if name.contains('\0') {
                return Err(SafetensorsError::InvalidTensor {
                    path,
                    name: name.clone(),
                    reason: "tensor name contains a NUL byte".to_owned(),
                });
            }
            let entry = parse_tensor(&path, name, value)?;
            if tensors.insert(name.clone(), entry).is_some() {
                return Err(SafetensorsError::DuplicateTensor {
                    path,
                    name: name.clone(),
                });
            }
        }
        if tensors.is_empty() {
            return Err(SafetensorsError::NoTensors { path });
        }
        validate_coverage(&path, &tensors, data_len)?;

        Ok(Self {
            path,
            file,
            file_len,
            header_len,
            data_offset,
            data_len,
            tensors,
            metadata,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn header_len(&self) -> u64 {
        self.header_len
    }

    pub fn data_len(&self) -> u64 {
        self.data_len
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors.values()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Read a bounded range of the shard payload. `offset` is relative to the
    /// start of the data region; `length` must not exceed `MAX_CHUNK_BYTES`, and
    /// the range must lie inside the validated data region.
    pub fn read_payload_range(
        &self,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, SafetensorsError> {
        if length == 0 {
            return Err(SafetensorsError::ChunkLength {
                path: self.path.clone(),
                reason: "chunk length must be non-zero".to_owned(),
            });
        }
        if length > MAX_CHUNK_BYTES {
            return Err(SafetensorsError::ChunkLength {
                path: self.path.clone(),
                reason: format!(
                    "chunk length {length} exceeds the {}-byte limit",
                    MAX_CHUNK_BYTES
                ),
            });
        }
        let absolute = self
            .data_offset
            .checked_add(offset)
            .ok_or(SafetensorsError::SizeOverflow)?;
        let end = absolute
            .checked_add(length as u64)
            .ok_or(SafetensorsError::SizeOverflow)?;
        if end > self.file_len {
            return Err(SafetensorsError::ChunkOutOfBounds {
                path: self.path.clone(),
                end,
                file_len: self.file_len,
            });
        }
        let mut bytes = vec![0u8; length];
        read_exact_at(&self.file, &mut bytes, absolute)
            .map_err(|source| io_error(&self.path, "read payload range", source))?;
        Ok(bytes)
    }

    /// Read a byte range inside one tensor's payload.
    ///
    /// `offset` is relative to the **start of the tensor payload** (not to
    /// the start of the shard file): the shard header offset is applied here,
    /// so callers never have to know where the data region begins. The range
    /// is validated against the tensor's validated byte length and read in
    /// bounded chunks of at most `MAX_CHUNK_BYTES`.
    pub fn read_tensor_range(
        &self,
        name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, SafetensorsError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound {
                path: self.path.clone(),
                name: name.to_owned(),
            })?;
        let end = offset
            .checked_add(length as u64)
            .ok_or(SafetensorsError::SizeOverflow)?;
        if end > entry.byte_len() {
            return Err(SafetensorsError::TensorRangeOutOfBounds {
                path: self.path.clone(),
                name: name.to_owned(),
                end,
                byte_len: entry.byte_len(),
            });
        }
        let base = entry
            .data_start()
            .checked_add(offset)
            .ok_or(SafetensorsError::SizeOverflow)?;
        let mut bytes = Vec::with_capacity(length);
        let mut read = 0u64;
        while read < length as u64 {
            let chunk = usize::try_from(length as u64 - read)
                .unwrap_or(usize::MAX)
                .min(MAX_CHUNK_BYTES);
            bytes.extend_from_slice(
                &self.read_payload_range(
                    base.checked_add(read)
                        .ok_or(SafetensorsError::SizeOverflow)?,
                    chunk,
                )?,
            );
            read += chunk as u64;
        }
        Ok(bytes)
    }

    /// Read one tensor's exact stored bytes. The allocation is bounded by the
    /// tensor's validated byte length; no copy of the whole shard is made.
    pub fn read_bytes(&self, name: &str) -> Result<Vec<u8>, SafetensorsError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound {
                path: self.path.clone(),
                name: name.to_owned(),
            })?;
        let length =
            usize::try_from(entry.byte_len()).map_err(|_| SafetensorsError::SizeOverflow)?;
        let mut bytes = vec![0u8; length];
        let offset = self
            .data_offset
            .checked_add(entry.data_start)
            .ok_or(SafetensorsError::SizeOverflow)?;
        read_exact_at(&self.file, &mut bytes, offset)
            .map_err(|source| io_error(&self.path, format!("read tensor {name:?}"), source))?;
        Ok(bytes)
    }

    /// Read a tensor and widen it explicitly to FP32. BF16 widening is exact;
    /// no rounding or dtype conversion other than the documented BF16->FP32
    /// bit expansion is performed.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, SafetensorsError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound {
                path: self.path.clone(),
                name: name.to_owned(),
            })?;
        let bytes = self.read_bytes(name)?;
        let values = match entry.dtype {
            WeightDType::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
            WeightDType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("two-byte chunk"));
                    f32::from_bits(u32::from(bits) << 16)
                })
                .collect(),
        };
        Ok(values)
    }

    /// Read BF16 storage exactly as little-endian 16-bit patterns. F32 tensors
    /// are rejected; callers must not silently narrow weights.
    pub fn read_bf16_bits(&self, name: &str) -> Result<Vec<u16>, SafetensorsError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound {
                path: self.path.clone(),
                name: name.to_owned(),
            })?;
        if entry.dtype != WeightDType::BF16 {
            return Err(SafetensorsError::UnexpectedDType {
                path: self.path.clone(),
                name: name.to_owned(),
                expected: "BF16",
                actual: entry.dtype.name(),
            });
        }
        let bytes = self.read_bytes(name)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("two-byte chunk")))
            .collect())
    }
}

fn parse_tensor(path: &Path, name: &str, value: &Json) -> Result<TensorEntry, SafetensorsError> {
    let fields = value
        .as_object()
        .ok_or_else(|| invalid(path, name, format!("entry is {}", value.type_name())))?;
    if fields.len() != 3 {
        return Err(invalid(
            path,
            name,
            "entry must contain exactly dtype, shape and data_offsets".to_owned(),
        ));
    }
    let dtype_value = value
        .get("dtype")
        .ok_or_else(|| invalid(path, name, "missing dtype".to_owned()))?;
    let dtype_name = dtype_value
        .as_str()
        .ok_or_else(|| invalid(path, name, "dtype is not a string".to_owned()))?;
    let dtype =
        WeightDType::parse(dtype_name).ok_or_else(|| SafetensorsError::UnsupportedDType {
            path: path.to_path_buf(),
            name: name.to_owned(),
            dtype: dtype_name.to_owned(),
        })?;

    let shape_value = value
        .get("shape")
        .ok_or_else(|| invalid(path, name, "missing shape".to_owned()))?;
    let dimensions = shape_value
        .as_array()
        .ok_or_else(|| invalid(path, name, "shape is not an array".to_owned()))?;
    if dimensions.is_empty() {
        return Err(invalid(
            path,
            name,
            "rank-zero tensors are not supported".to_owned(),
        ));
    }
    let mut shape = Vec::with_capacity(dimensions.len());
    let mut numel = 1u64;
    for dimension in dimensions {
        let raw = dimension.as_u64().ok_or_else(|| {
            invalid(
                path,
                name,
                "shape dimension is not an unsigned integer".to_owned(),
            )
        })?;
        if raw == 0 {
            return Err(invalid(
                path,
                name,
                "shape contains a zero dimension".to_owned(),
            ));
        }
        let dimension = usize::try_from(raw).map_err(|_| SafetensorsError::SizeOverflow)?;
        numel = numel
            .checked_mul(raw)
            .ok_or_else(|| invalid(path, name, "shape element count overflows u64".to_owned()))?;
        shape.push(dimension);
    }
    let byte_len = numel
        .checked_mul(dtype.element_size_bytes() as u64)
        .ok_or_else(|| invalid(path, name, "tensor byte length overflows u64".to_owned()))?;

    let offsets_value = value
        .get("data_offsets")
        .ok_or_else(|| invalid(path, name, "missing data_offsets".to_owned()))?;
    let offsets = offsets_value
        .as_array()
        .ok_or_else(|| invalid(path, name, "data_offsets is not an array".to_owned()))?;
    if offsets.len() != 2 {
        return Err(invalid(
            path,
            name,
            "data_offsets must contain two entries".to_owned(),
        ));
    }
    let start = offsets[0].as_u64().ok_or_else(|| {
        invalid(
            path,
            name,
            "data_offsets[0] is not an unsigned integer".to_owned(),
        )
    })?;
    let end = offsets[1].as_u64().ok_or_else(|| {
        invalid(
            path,
            name,
            "data_offsets[1] is not an unsigned integer".to_owned(),
        )
    })?;
    if start > end {
        return Err(invalid(
            path,
            name,
            format!("data_offsets start {start} exceeds end {end}"),
        ));
    }
    let actual = end - start;
    if actual != byte_len {
        return Err(SafetensorsError::SizeMismatch {
            path: path.to_path_buf(),
            name: name.to_owned(),
            shape_bytes: byte_len,
            offset_bytes: actual,
        });
    }
    let alignment = dtype.element_size_bytes() as u64;
    if start % alignment != 0 || end % alignment != 0 {
        return Err(SafetensorsError::MisalignedTensor {
            path: path.to_path_buf(),
            name: name.to_owned(),
            start,
            end,
            alignment,
        });
    }
    Ok(TensorEntry {
        name: name.to_owned(),
        dtype,
        shape,
        data_start: start,
        data_end: end,
    })
}

fn parse_metadata(path: &Path, value: &Json) -> Result<BTreeMap<String, String>, SafetensorsError> {
    let entries = value
        .as_object()
        .ok_or_else(|| SafetensorsError::InvalidMetadata {
            path: path.to_path_buf(),
            reason: format!("__metadata__ is {}", value.type_name()),
        })?;
    let mut metadata = BTreeMap::new();
    for (key, value) in entries {
        let text = value
            .as_str()
            .ok_or_else(|| SafetensorsError::InvalidMetadata {
                path: path.to_path_buf(),
                reason: format!("metadata value for {key:?} is not a string"),
            })?;
        metadata.insert(key.clone(), text.to_owned());
    }
    Ok(metadata)
}

fn validate_coverage(
    path: &Path,
    tensors: &BTreeMap<String, TensorEntry>,
    data_len: u64,
) -> Result<(), SafetensorsError> {
    let mut ranges: Vec<(&str, u64, u64)> = tensors
        .values()
        .map(|entry| (entry.name(), entry.data_start, entry.data_end))
        .collect();
    ranges.sort_by_key(|&(_, start, end)| (start, end));
    let mut expected_start = 0u64;
    for (name, start, end) in ranges {
        if end > data_len {
            return Err(SafetensorsError::OffsetOutOfBounds {
                path: path.to_path_buf(),
                name: name.to_owned(),
                end,
                data_len,
            });
        }
        if start < expected_start {
            return Err(SafetensorsError::OverlappingTensors {
                path: path.to_path_buf(),
                name: name.to_owned(),
                offset: start,
                previous_end: expected_start,
            });
        }
        if start != expected_start {
            return Err(SafetensorsError::DataLayout {
                path: path.to_path_buf(),
                name: name.to_owned(),
                reason: format!(
                    "gap before tensor: expected data offset {expected_start}, found {start}"
                ),
            });
        }
        expected_start = end;
    }
    if expected_start != data_len {
        return Err(SafetensorsError::DataLayout {
            path: path.to_path_buf(),
            name: String::new(),
            reason: format!(
                "tensor payloads cover {expected_start} bytes but the data region has {data_len}"
            ),
        });
    }
    Ok(())
}

fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buffer, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut clone = file.try_clone()?;
        clone.seek(SeekFrom::Start(offset))?;
        clone.read_exact(buffer)
    }
}

fn invalid(path: &Path, name: &str, reason: String) -> SafetensorsError {
    SafetensorsError::InvalidTensor {
        path: path.to_path_buf(),
        name: name.to_owned(),
        reason,
    }
}

fn io_error(path: &Path, operation: impl Into<String>, source: io::Error) -> SafetensorsError {
    SafetensorsError::Io {
        path: path.to_path_buf(),
        operation: operation.into(),
        message: source.to_string(),
    }
}

#[derive(Debug, Error)]
pub enum SafetensorsError {
    #[error("{operation} failed for {path}: {message}")]
    Io {
        path: PathBuf,
        operation: String,
        message: String,
    },
    #[error("{path}: file is {file_len} bytes, smaller than the {minimum}-byte length prefix")]
    FileTooShort {
        path: PathBuf,
        file_len: u64,
        minimum: u64,
    },
    #[error("{path}: declared header length {header_len} exceeds file length {file_len}")]
    TruncatedHeader {
        path: PathBuf,
        header_len: u64,
        file_len: u64,
    },
    #[error("{path}: header length {header_len} exceeds the {maximum}-byte limit")]
    HeaderTooLarge {
        path: PathBuf,
        header_len: u64,
        maximum: u64,
    },
    #[error("{path}: empty Safetensors header")]
    EmptyHeader { path: PathBuf },
    #[error("{path}: invalid header JSON: {message}")]
    InvalidJson { path: PathBuf, message: String },
    #[error("{path}: header root is {actual}, expected an object")]
    InvalidRoot { path: PathBuf, actual: &'static str },
    #[error("{path}: invalid __metadata__: {reason}")]
    InvalidMetadata { path: PathBuf, reason: String },
    #[error("{path}: invalid tensor {name:?}: {reason}")]
    InvalidTensor {
        path: PathBuf,
        name: String,
        reason: String,
    },
    #[error("{path}: empty tensor name in header")]
    EmptyTensorName { path: PathBuf },
    #[error("{path}: duplicate tensor name {name:?}")]
    DuplicateTensor { path: PathBuf, name: String },
    #[error("{path}: shard contains no tensors")]
    NoTensors { path: PathBuf },
    #[error("{path}: tensor {name:?} uses unsupported dtype {dtype}; only BF16/F32 are accepted")]
    UnsupportedDType {
        path: PathBuf,
        name: String,
        dtype: String,
    },
    #[error("{path}: tensor {name:?} has dtype {actual}, expected {expected}")]
    UnexpectedDType {
        path: PathBuf,
        name: String,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("{path}: tensor {name:?} shape requires {shape_bytes} bytes but offsets declare {offset_bytes}")]
    SizeMismatch {
        path: PathBuf,
        name: String,
        shape_bytes: u64,
        offset_bytes: u64,
    },
    #[error("{path}: tensor {name:?} offsets {start}..{end} are not aligned to {alignment} bytes")]
    MisalignedTensor {
        path: PathBuf,
        name: String,
        start: u64,
        end: u64,
        alignment: u64,
    },
    #[error("{path}: tensor {name:?} ends at {end}, beyond the {data_len}-byte data region")]
    OffsetOutOfBounds {
        path: PathBuf,
        name: String,
        end: u64,
        data_len: u64,
    },
    #[error("{path}: tensor {name:?} starts at {offset}, inside the previous tensor ending at {previous_end}")]
    OverlappingTensors {
        path: PathBuf,
        name: String,
        offset: u64,
        previous_end: u64,
    },
    #[error("{path}: invalid tensor data layout ({name:?}): {reason}")]
    DataLayout {
        path: PathBuf,
        name: String,
        reason: String,
    },
    #[error("{path}: tensor {name:?} does not exist")]
    TensorNotFound { path: PathBuf, name: String },
    #[error("checkpoint size arithmetic overflow")]
    SizeOverflow,
    #[error("{path}: invalid payload chunk: {reason}")]
    ChunkLength { path: PathBuf, reason: String },
    #[error("{path}: payload chunk ends at {end}, beyond the {file_len}-byte file")]
    ChunkOutOfBounds {
        path: PathBuf,
        end: u64,
        file_len: u64,
    },
    #[error("{path}: tensor {name:?} range ends at {end}, beyond its {byte_len}-byte payload")]
    TensorRangeOutOfBounds {
        path: PathBuf,
        name: String,
        end: u64,
        byte_len: u64,
    },
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
                "claymore-m10-safetensors-{}-{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tensor_json(dtype: &str, shape: &[u64], start: u64, end: u64) -> String {
        format!(
            r#""dtype":"{dtype}","shape":[{}],"data_offsets":[{start},{end}]"#,
            shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn write_shard(path: &Path, header: &str, data: &[u8]) -> std::io::Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(path, bytes)
    }

    #[test]
    fn reads_valid_bf16_and_f32_tensors_without_whole_shard_copy() {
        let dir = TestDir::new();
        let path = dir.file("valid.safetensors");
        let bf16: [u16; 3] = [0x3f80, 0xc000, 0x0001];
        let f32_values: [f32; 2] = [1.5, -2.25];
        let mut data = Vec::new();
        for value in f32_values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        for bits in bf16 {
            data.extend_from_slice(&bits.to_le_bytes());
        }
        let header = format!(
            r#"{{"__metadata__":{{"format":"pt"}},"a":{{{}}},"b":{{{}}}}}"#,
            tensor_json("BF16", &[3], 8, 14),
            tensor_json("F32", &[2], 0, 8)
        );
        write_shard(&path, &header, &data).unwrap();

        let shard = SafetensorsFile::open(&path).unwrap();
        assert_eq!(shard.tensor_count(), 2);
        assert_eq!(shard.data_len(), 14);
        assert_eq!(
            shard.metadata().get("format").map(String::as_str),
            Some("pt")
        );
        assert_eq!(shard.read_bf16_bits("a").unwrap(), bf16);
        assert_eq!(
            shard.read_f32("a").unwrap(),
            [1.0, -2.0, f32::from_bits(0x0001_0000)]
        );
        assert_eq!(shard.read_f32("b").unwrap(), f32_values);
        assert_eq!(shard.tensor("a").unwrap().shape(), &[3]);
        assert_eq!(shard.tensor("a").unwrap().numel(), 3);
        assert!(matches!(
            shard.read_bf16_bits("b"),
            Err(SafetensorsError::UnexpectedDType { .. })
        ));
        assert!(matches!(
            shard.read_bytes("missing"),
            Err(SafetensorsError::TensorNotFound { .. })
        ));
    }

    #[test]
    fn reads_a_tensor_range_relative_to_the_payload_not_to_the_file() {
        let dir = TestDir::new();
        let path = dir.file("range.safetensors");
        let first: [u16; 3] = [0x3f80, 0xc000, 0x0001];
        let second: [u16; 2] = [0x4000, 0x4040];
        let mut data = Vec::new();
        for bits in first {
            data.extend_from_slice(&bits.to_le_bytes());
        }
        for bits in second {
            data.extend_from_slice(&bits.to_le_bytes());
        }
        let header = format!(
            r#"{{"a":{{{}}},"b":{{{}}}}}"#,
            tensor_json("BF16", &[3], 0, 6),
            tensor_json("BF16", &[2], 6, 10)
        );
        write_shard(&path, &header, &data).unwrap();

        let shard = SafetensorsFile::open(&path).unwrap();
        // The whole payload of the second tensor, read relatively to its own
        // start: the shard header offset is applied by the reader.
        assert_eq!(
            shard.read_tensor_range("b", 0, 4).unwrap(),
            [0x4000u16.to_le_bytes(), 0x4040u16.to_le_bytes()].concat()
        );
        // A range inside one tensor, in bytes: the payload of `a` is
        // `80 3f 00 c0 01 00`, so bytes 2..4 are `00 c0`.
        assert_eq!(shard.read_tensor_range("a", 2, 2).unwrap(), [0x00u8, 0xc0]);
        // A range that leaves the tensor payload is refused.
        assert!(matches!(
            shard.read_tensor_range("a", 4, 4),
            Err(SafetensorsError::TensorRangeOutOfBounds { .. })
        ));
        assert!(matches!(
            shard.read_tensor_range("missing", 0, 2),
            Err(SafetensorsError::TensorNotFound { .. })
        ));
    }

    #[test]
    fn rejects_truncated_and_oversized_headers() {
        let dir = TestDir::new();
        let short = dir.file("short.safetensors");
        std::fs::write(&short, b"1234").unwrap();
        assert!(matches!(
            SafetensorsFile::open(&short),
            Err(SafetensorsError::FileTooShort { .. })
        ));

        let truncated = dir.file("truncated.safetensors");
        std::fs::write(&truncated, 100u64.to_le_bytes()).unwrap();
        assert!(matches!(
            SafetensorsFile::open(&truncated),
            Err(SafetensorsError::TruncatedHeader { .. })
        ));

        let oversized = dir.file("oversized.safetensors");
        let mut bytes = (MAX_HEADER_BYTES + 1).to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0; 8]);
        std::fs::write(&oversized, bytes).unwrap();
        assert!(matches!(
            SafetensorsFile::open(&oversized),
            Err(SafetensorsError::HeaderTooLarge { .. })
        ));

        let empty = dir.file("empty.safetensors");
        let mut bytes = 0u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&empty, bytes).unwrap();
        assert!(matches!(
            SafetensorsFile::open(&empty),
            Err(SafetensorsError::EmptyHeader { .. })
        ));
    }

    #[test]
    fn rejects_malformed_json_duplicate_names_and_empty_names() {
        let dir = TestDir::new();
        let bad = dir.file("bad.safetensors");
        write_shard(&bad, "{not json", b"").unwrap();
        assert!(matches!(
            SafetensorsFile::open(&bad),
            Err(SafetensorsError::InvalidJson { .. })
        ));

        let duplicate = dir.file("duplicate.safetensors");
        let header = format!(
            r#"{{"a":{{{}}},"a":{{{}}}}}"#,
            tensor_json("F32", &[1], 0, 4),
            tensor_json("F32", &[1], 0, 4)
        );
        write_shard(&duplicate, &header, &[0; 4]).unwrap();
        assert!(matches!(
            SafetensorsFile::open(&duplicate),
            Err(SafetensorsError::InvalidJson { .. })
        ));

        let unnamed = dir.file("unnamed.safetensors");
        let header = format!(r#"{{"":{{{}}}}}"#, tensor_json("F32", &[1], 0, 4));
        write_shard(&unnamed, &header, &[0; 4]).unwrap();
        assert!(matches!(
            SafetensorsFile::open(&unnamed),
            Err(SafetensorsError::EmptyTensorName { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_dtypes_and_bad_entries() {
        let dir = TestDir::new();
        for (dtype, expected) in [("F16", 0u8), ("I8", 0)] {
            let path = dir.file(&format!("{dtype}.safetensors"));
            let header = format!(r#"{{"a":{{{}}}}}"#, tensor_json(dtype, &[1], 0, 2));
            write_shard(&path, &header, &[0; 2]).unwrap();
            match SafetensorsFile::open(&path) {
                Err(SafetensorsError::UnsupportedDType { dtype: found, .. }) => {
                    assert_eq!(found, dtype);
                }
                other => panic!("unexpected result for {dtype}: {other:?}"),
            }
            let _ = expected;
        }

        let zero = dir.file("zero.safetensors");
        let header = format!(r#"{{"a":{{{}}}}}"#, tensor_json("BF16", &[0], 0, 0));
        write_shard(&zero, &header, b"").unwrap();
        assert!(matches!(
            SafetensorsFile::open(&zero),
            Err(SafetensorsError::InvalidTensor { .. })
        ));

        let missing = dir.file("missing.safetensors");
        write_shard(&missing, r#"{"a":{"dtype":"BF16","shape":[1]}}"#, &[0; 2]).unwrap();
        assert!(matches!(
            SafetensorsFile::open(&missing),
            Err(SafetensorsError::InvalidTensor { .. })
        ));

        let overflow = dir.file("overflow.safetensors");
        let header = format!(
            r#"{{"a":{{{}}}}}"#,
            tensor_json("BF16", &[u64::MAX, 2], 0, 0)
        );
        write_shard(&overflow, &header, b"").unwrap();
        assert!(matches!(
            SafetensorsFile::open(&overflow),
            Err(SafetensorsError::InvalidTensor { .. })
        ));
    }

    #[test]
    fn rejects_shape_mismatch_alignment_overlap_gap_and_bounds() {
        let dir = TestDir::new();
        let cases: [(&str, String, usize); 5] = [
            (
                "size",
                format!(r#"{{"a":{{{}}}}}"#, tensor_json("BF16", &[2], 0, 2)),
                2,
            ),
            (
                "align",
                format!(r#"{{"a":{{{}}}}}"#, tensor_json("BF16", &[1], 1, 3)),
                4,
            ),
            (
                "overlap",
                format!(
                    r#"{{"a":{{{}}},"b":{{{}}}}}"#,
                    tensor_json("BF16", &[2], 0, 4),
                    tensor_json("BF16", &[2], 2, 6)
                ),
                6,
            ),
            (
                "gap",
                format!(
                    r#"{{"a":{{{}}},"b":{{{}}}}}"#,
                    tensor_json("BF16", &[2], 0, 4),
                    tensor_json("BF16", &[1], 6, 8)
                ),
                8,
            ),
            (
                "bounds",
                format!(r#"{{"a":{{{}}}}}"#, tensor_json("BF16", &[2], 0, 4)),
                2,
            ),
        ];
        for (name, header, payload) in cases {
            let path = dir.file(&format!("{name}.safetensors"));
            write_shard(&path, &header, &vec![0u8; payload]).unwrap();
            let result = SafetensorsFile::open(&path);
            let expected = match name {
                "size" => matches!(result, Err(SafetensorsError::SizeMismatch { .. })),
                "align" => matches!(result, Err(SafetensorsError::MisalignedTensor { .. })),
                "overlap" => matches!(result, Err(SafetensorsError::OverlappingTensors { .. })),
                "gap" => matches!(result, Err(SafetensorsError::DataLayout { .. })),
                "bounds" => matches!(result, Err(SafetensorsError::OffsetOutOfBounds { .. })),
                _ => unreachable!(),
            };
            assert!(expected, "{name}: {result:?}");
        }
    }

    #[test]
    fn detects_a_file_truncated_after_open() {
        let dir = TestDir::new();
        let path = dir.file("shrink.safetensors");
        let header = format!(r#"{{"a":{{{}}}}}"#, tensor_json("BF16", &[4], 0, 8));
        write_shard(&path, &header, &[1u8; 8]).unwrap();
        let shard = SafetensorsFile::open(&path).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(8 + header.len() as u64 + 2).unwrap();
        drop(file);
        assert!(matches!(
            shard.read_bf16_bits("a"),
            Err(SafetensorsError::Io { .. })
        ));
    }

    #[test]
    fn rejects_metadata_that_is_not_a_string_map() {
        let dir = TestDir::new();
        let path = dir.file("metadata.safetensors");
        let header = format!(
            r#"{{"__metadata__":{{"x":1}},"a":{{{}}}}}"#,
            tensor_json("BF16", &[1], 0, 2)
        );
        write_shard(&path, &header, &[0; 2]).unwrap();
        assert!(matches!(
            SafetensorsFile::open(&path),
            Err(SafetensorsError::InvalidMetadata { .. })
        ));
    }
}
