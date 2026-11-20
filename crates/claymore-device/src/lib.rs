//! Backend-neutral device, buffer, transfer, and tensor contracts.
//!
//! No Send/Sync bounds are imposed: a concrete backend may be thread-affine
//! (as the current CUDA primary-context owner is). Callers must respect the
//! ownership model reported by the backend rather than assuming thread safety.
use std::any::Any;

use claymore_core::MemoryTier;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub kind: DeviceKind,
    pub total_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Other,
}

/// Capabilities are reported facts, not assumptions made by callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub device_memory: bool,
    pub host_memory: bool,
    pub pinned_host_memory: bool,
    pub asynchronous_transfers: bool,
    pub tensor_compute: bool,
}

/// Owned allocation. No Send/Sync bounds: backends may have thread/context affinity.
pub trait DeviceBuffer: Any {
    fn device_id(&self) -> &str;
    fn tier(&self) -> MemoryTier;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Synchronous operations: write/read/copy are complete when they return.
/// Async work must later use an explicit completion token, not an implicit tick.
pub trait Device {
    fn info(&self) -> &DeviceInfo;
    fn capabilities(&self) -> DeviceCapabilities;
    fn allocate(&self, tier: MemoryTier, bytes: u64) -> Result<Box<dyn DeviceBuffer>, DeviceError>;
    fn write(
        &self,
        buffer: &mut dyn DeviceBuffer,
        offset: usize,
        data: &[u8],
    ) -> Result<(), DeviceError>;
    fn read(
        &self,
        buffer: &dyn DeviceBuffer,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>, DeviceError>;
    fn copy(
        &self,
        source: &dyn DeviceBuffer,
        destination: &mut dyn DeviceBuffer,
        bytes: usize,
    ) -> Result<(), DeviceError>;
    fn synchronize(&self) -> Result<(), DeviceError>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeviceError {
    #[error("zero-sized allocations are not supported")]
    ZeroSizedAllocation,
    #[error("allocation size does not fit this platform")]
    SizeOverflow,
    #[error("memory tier {0:?} is not supported by this device")]
    UnsupportedMemoryTier(MemoryTier),
    #[error("buffer belongs to another device")]
    DeviceMismatch,
    #[error("buffer implementation is incompatible with this device")]
    BufferTypeMismatch,
    #[error("buffer range {offset}..{end} exceeds capacity {capacity}")]
    BufferRangeOutOfBounds {
        offset: usize,
        end: usize,
        capacity: usize,
    },
    #[error("copy size {requested} exceeds source or destination capacity")]
    BufferTooSmall { requested: usize },
    #[error("tensor shape contains a zero-sized dimension")]
    EmptyTensorDimension,
    #[error("tensor shape or byte size overflows usize")]
    TensorSizeOverflow,
    #[error("tensor requires {required} bytes but its buffer has {capacity}")]
    TensorBufferTooSmall { required: usize, capacity: usize },
}

/// Deterministic in-memory backend. It simulates device and host tiers with
/// ordinary Vec storage; it does not claim to provide physically pinned memory.
#[derive(Debug, Clone)]
pub struct CpuDevice {
    info: DeviceInfo,
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            info: DeviceInfo {
                id: "cpu:0".to_owned(),
                name: "Claymore CPU simulator".to_owned(),
                kind: DeviceKind::Cpu,
                total_memory_bytes: None,
            },
        }
    }

    fn checked<'a>(&self, buffer: &'a dyn DeviceBuffer) -> Result<&'a CpuBuffer, DeviceError> {
        if buffer.device_id() != self.info.id {
            return Err(DeviceError::DeviceMismatch);
        }
        buffer
            .as_any()
            .downcast_ref::<CpuBuffer>()
            .ok_or(DeviceError::BufferTypeMismatch)
    }

    fn checked_mut<'a>(
        &self,
        buffer: &'a mut dyn DeviceBuffer,
    ) -> Result<&'a mut CpuBuffer, DeviceError> {
        if buffer.device_id() != self.info.id {
            return Err(DeviceError::DeviceMismatch);
        }
        buffer
            .as_any_mut()
            .downcast_mut::<CpuBuffer>()
            .ok_or(DeviceError::BufferTypeMismatch)
    }
}

struct CpuBuffer {
    device_id: String,
    tier: MemoryTier,
    bytes: Vec<u8>,
}

impl DeviceBuffer for CpuBuffer {
    fn device_id(&self) -> &str {
        &self.device_id
    }
    fn tier(&self) -> MemoryTier {
        self.tier
    }
    fn len(&self) -> usize {
        self.bytes.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Device for CpuDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            device_memory: true,
            host_memory: true,
            pinned_host_memory: false,
            asynchronous_transfers: false,
            tensor_compute: false,
        }
    }

    fn allocate(&self, tier: MemoryTier, bytes: u64) -> Result<Box<dyn DeviceBuffer>, DeviceError> {
        if bytes == 0 {
            return Err(DeviceError::ZeroSizedAllocation);
        }
        if tier == MemoryTier::PinnedHost {
            return Err(DeviceError::UnsupportedMemoryTier(tier));
        }
        let len = usize::try_from(bytes).map_err(|_| DeviceError::SizeOverflow)?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(len)
            .map_err(|_| DeviceError::SizeOverflow)?;
        storage.resize(len, 0);
        Ok(Box::new(CpuBuffer {
            device_id: self.info.id.clone(),
            tier,
            bytes: storage,
        }))
    }

    fn write(
        &self,
        buffer: &mut dyn DeviceBuffer,
        offset: usize,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        let buffer = self.checked_mut(buffer)?;
        let end = offset
            .checked_add(data.len())
            .ok_or(DeviceError::BufferRangeOutOfBounds {
                offset,
                end: usize::MAX,
                capacity: buffer.bytes.len(),
            })?;
        if end > buffer.bytes.len() {
            return Err(DeviceError::BufferRangeOutOfBounds {
                offset,
                end,
                capacity: buffer.bytes.len(),
            });
        }
        buffer.bytes[offset..end].copy_from_slice(data);
        Ok(())
    }

    fn read(
        &self,
        buffer: &dyn DeviceBuffer,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>, DeviceError> {
        let buffer = self.checked(buffer)?;
        let end = offset
            .checked_add(len)
            .ok_or(DeviceError::BufferRangeOutOfBounds {
                offset,
                end: usize::MAX,
                capacity: buffer.bytes.len(),
            })?;
        if end > buffer.bytes.len() {
            return Err(DeviceError::BufferRangeOutOfBounds {
                offset,
                end,
                capacity: buffer.bytes.len(),
            });
        }
        Ok(buffer.bytes[offset..end].to_vec())
    }

    fn copy(
        &self,
        source: &dyn DeviceBuffer,
        destination: &mut dyn DeviceBuffer,
        bytes: usize,
    ) -> Result<(), DeviceError> {
        let source = self.checked(source)?;
        let destination = self.checked_mut(destination)?;
        if bytes > source.bytes.len() || bytes > destination.bytes.len() {
            return Err(DeviceError::BufferTooSmall { requested: bytes });
        }
        destination.bytes[..bytes].copy_from_slice(&source.bytes[..bytes]);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), DeviceError> {
        Ok(())
    }
}

/// Element metadata only; this enum does not imply arithmetic kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    U8,
    I32,
    I64,
    F16,
    BF16,
    F32,
}

impl DType {
    pub const fn element_size_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I32 | Self::F32 => 4,
            Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Owned contiguous tensor metadata and storage. No operators, autograd, or kernels.
pub struct Tensor {
    buffer: Box<dyn DeviceBuffer>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
    numel: usize,
}

impl Tensor {
    pub fn new(
        buffer: Box<dyn DeviceBuffer>,
        shape: impl Into<Vec<usize>>,
        dtype: DType,
    ) -> Result<Self, DeviceError> {
        let shape = shape.into();
        let mut numel = 1_usize;
        for dimension in &shape {
            if *dimension == 0 {
                return Err(DeviceError::EmptyTensorDimension);
            }
            numel = numel
                .checked_mul(*dimension)
                .ok_or(DeviceError::TensorSizeOverflow)?;
        }
        let required = numel
            .checked_mul(dtype.element_size_bytes())
            .ok_or(DeviceError::TensorSizeOverflow)?;
        if required > buffer.len() {
            return Err(DeviceError::TensorBufferTooSmall {
                required,
                capacity: buffer.len(),
            });
        }
        let mut strides = vec![1; shape.len()];
        let mut stride = 1_usize;
        for index in (0..shape.len()).rev() {
            strides[index] = stride;
            stride = stride
                .checked_mul(shape[index])
                .ok_or(DeviceError::TensorSizeOverflow)?;
        }
        Ok(Self {
            buffer,
            shape,
            strides,
            dtype,
            numel,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn numel(&self) -> usize {
        self.numel
    }
    pub fn required_bytes(&self) -> usize {
        self.numel * self.dtype.element_size_bytes()
    }
    pub fn buffer(&self) -> &dyn DeviceBuffer {
        self.buffer.as_ref()
    }
    pub fn buffer_mut(&mut self) -> &mut dyn DeviceBuffer {
        self.buffer.as_mut()
    }
    pub fn into_buffer(self) -> Box<dyn DeviceBuffer> {
        self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_allocates_and_roundtrips_bytes() {
        let device = CpuDevice::new();
        let mut buffer = device.allocate(MemoryTier::Host, 8).unwrap();
        device.write(buffer.as_mut(), 2, &[1, 2, 3]).unwrap();
        assert_eq!(device.read(buffer.as_ref(), 2, 3).unwrap(), [1, 2, 3]);
    }

    #[test]
    fn cpu_device_copies_between_owned_buffers() {
        let device = CpuDevice::new();
        let mut source = device.allocate(MemoryTier::Vram, 4).unwrap();
        let mut destination = device.allocate(MemoryTier::Host, 4).unwrap();
        device.write(source.as_mut(), 0, &[4, 3, 2, 1]).unwrap();
        device
            .copy(source.as_ref(), destination.as_mut(), 4)
            .unwrap();
        assert_eq!(
            device.read(destination.as_ref(), 0, 4).unwrap(),
            [4, 3, 2, 1]
        );
    }

    #[test]
    fn cpu_device_rejects_unsupported_pinned_memory() {
        let device = CpuDevice::new();
        assert_eq!(
            device.allocate(MemoryTier::PinnedHost, 4).err(),
            Some(DeviceError::UnsupportedMemoryTier(MemoryTier::PinnedHost))
        );
    }

    #[test]
    fn cpu_device_rejects_out_of_bounds_access() {
        let device = CpuDevice::new();
        let mut buffer = device.allocate(MemoryTier::Host, 4).unwrap();
        assert!(matches!(
            device.write(buffer.as_mut(), 3, &[1, 2]),
            Err(DeviceError::BufferRangeOutOfBounds { .. })
        ));
    }

    #[test]
    fn tensor_computes_contiguous_shape_and_strides() {
        let device = CpuDevice::new();
        let buffer = device.allocate(MemoryTier::Host, 24).unwrap();
        let tensor = Tensor::new(buffer, vec![2, 3], DType::F32).unwrap();
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(tensor.strides(), [3, 1]);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.required_bytes(), 24);
    }

    #[test]
    fn tensor_rejects_insufficient_storage_and_empty_dimensions() {
        let device = CpuDevice::new();
        let small = device.allocate(MemoryTier::Host, 8).unwrap();
        assert!(matches!(
            Tensor::new(small, vec![3], DType::F32),
            Err(DeviceError::TensorBufferTooSmall { .. })
        ));
        let buffer = device.allocate(MemoryTier::Host, 4).unwrap();
        assert_eq!(
            Tensor::new(buffer, vec![0], DType::F32).err(),
            Some(DeviceError::EmptyTensorDimension)
        );
    }
}
