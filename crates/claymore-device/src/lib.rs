//! Backend-neutral device, buffer, transfer, and tensor contracts.
//!
//! No Send/Sync bounds are imposed: a concrete backend may be thread-affine
//! (as the current CUDA primary-context owner is). Callers must respect the
//! ownership model reported by the backend rather than assuming thread safety.
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

use claymore_core::MemoryTier;
use thiserror::Error;

static NEXT_DEVICE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Opaque process-local identity for one device instance.
///
/// This is deliberately separate from `DeviceInfo::id`, which is a stable
/// logical identifier suitable for display/configuration. Cloning a device
/// handle preserves this identity; constructing a new device allocates a new
/// identity even when its logical id and name are identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceInstanceId(u64);

impl DeviceInstanceId {
    fn next() -> Self {
        let id = NEXT_DEVICE_INSTANCE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("device instance identity space exhausted");
        Self(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Stable logical identifier, not an ownership/security boundary.
    pub id: String,
    /// Unique process-local identity of this concrete device instance.
    pub instance_id: DeviceInstanceId,
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
    /// Stable logical device identifier, for diagnostics only.
    fn device_id(&self) -> &str;
    /// Identity of the exact device instance that owns this buffer.
    fn device_instance_id(&self) -> DeviceInstanceId;
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
    #[error("allocation of {requested_bytes} bytes could not be satisfied: {reason}")]
    OutOfMemory { requested_bytes: usize, reason: String },
    #[error("memory tier {0:?} is not supported by this device")]
    UnsupportedMemoryTier(MemoryTier),
    #[error("buffer belongs to another device instance")]
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

fn checked_allocation_len(bytes: u64, platform_max: usize) -> Result<usize, DeviceError> {
    let len = usize::try_from(bytes).map_err(|_| DeviceError::SizeOverflow)?;
    if len > platform_max {
        return Err(DeviceError::SizeOverflow);
    }
    Ok(len)
}

fn allocate_zeroed_with<E: std::fmt::Display>(
    len: usize,
    reserve: impl FnOnce(&mut Vec<u8>, usize) -> Result<(), E>,
) -> Result<Vec<u8>, DeviceError> {
    let mut storage = Vec::new();
    reserve(&mut storage, len).map_err(|error| DeviceError::OutOfMemory {
        requested_bytes: len,
        reason: error.to_string(),
    })?;
    storage.resize(len, 0);
    Ok(storage)
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
                instance_id: DeviceInstanceId::next(),
                name: "Claymore CPU simulator".to_owned(),
                kind: DeviceKind::Cpu,
                total_memory_bytes: None,
            },
        }
    }

    fn checked<'a>(&self, buffer: &'a dyn DeviceBuffer) -> Result<&'a CpuBuffer, DeviceError> {
        if buffer.device_instance_id() != self.info.instance_id {
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
        if buffer.device_instance_id() != self.info.instance_id {
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
    device_instance_id: DeviceInstanceId,
    tier: MemoryTier,
    bytes: Vec<u8>,
}

impl DeviceBuffer for CpuBuffer {
    fn device_id(&self) -> &str {
        &self.device_id
    }
    fn device_instance_id(&self) -> DeviceInstanceId {
        self.device_instance_id
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
        let len = checked_allocation_len(bytes, isize::MAX as usize)?;
        let storage = allocate_zeroed_with(len, |storage, len| storage.try_reserve_exact(len))?;
        Ok(Box::new(CpuBuffer {
            device_id: self.info.id.clone(),
            device_instance_id: self.info.instance_id,
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
    fn independent_devices_reject_each_others_buffers_for_all_operations() {
        let first = CpuDevice::new();
        let second = CpuDevice::new();
        assert_eq!(first.info().id, second.info().id);
        assert_ne!(first.info().instance_id, second.info().instance_id);

        let mut first_buffer = first.allocate(MemoryTier::Host, 4).unwrap();
        let mut second_buffer = second.allocate(MemoryTier::Host, 4).unwrap();
        assert_eq!(
            second.read(first_buffer.as_ref(), 0, 1),
            Err(DeviceError::DeviceMismatch)
        );
        assert_eq!(
            second.write(first_buffer.as_mut(), 0, &[1]),
            Err(DeviceError::DeviceMismatch)
        );
        assert_eq!(
            second.copy(first_buffer.as_ref(), second_buffer.as_mut(), 1),
            Err(DeviceError::DeviceMismatch)
        );
        assert_eq!(
            first.copy(first_buffer.as_ref(), second_buffer.as_mut(), 1),
            Err(DeviceError::DeviceMismatch)
        );
        assert_eq!(second.read(second_buffer.as_ref(), 0, 4).unwrap(), [0; 4]);
        assert_eq!(first.read(first_buffer.as_ref(), 0, 4).unwrap(), [0; 4]);
    }

    #[test]
    fn cloned_device_handle_shares_instance_identity() {
        let device = CpuDevice::new();
        let clone = device.clone();
        assert_eq!(device.info().instance_id, clone.info().instance_id);
        let buffer = device.allocate(MemoryTier::Host, 4).unwrap();
        assert_eq!(clone.read(buffer.as_ref(), 0, 4).unwrap(), [0; 4]);
    }

    #[test]
    fn cpu_capabilities_match_supported_operations_and_memory_tiers() {
        let device = CpuDevice::new();
        let capabilities = device.capabilities();
        assert!(capabilities.device_memory);
        assert!(capabilities.host_memory);
        assert!(!capabilities.pinned_host_memory);
        assert!(!capabilities.asynchronous_transfers);
        assert!(!capabilities.tensor_compute);

        assert!(device.allocate(MemoryTier::Vram, 1).is_ok());
        assert!(device.allocate(MemoryTier::Host, 1).is_ok());
        assert!(matches!(
            device.allocate(MemoryTier::PinnedHost, 1),
            Err(DeviceError::UnsupportedMemoryTier(MemoryTier::PinnedHost))
        ));
    }

    #[test]
    fn zero_sized_allocation_is_rejected() {
        let device = CpuDevice::new();
        assert_eq!(
            device.allocate(MemoryTier::Host, 0).err(),
            Some(DeviceError::ZeroSizedAllocation)
        );
    }

    #[test]
    fn size_overflow_is_distinct_from_allocation_failure() {
        assert_eq!(
            checked_allocation_len(1025, 1024),
            Err(DeviceError::SizeOverflow)
        );
        let oom = allocate_zeroed_with(1025, |_storage: &mut Vec<u8>, _len| {
            Err::<(), _>("injected allocator exhaustion")
        })
        .unwrap_err();
        assert!(matches!(oom, DeviceError::OutOfMemory { .. }));
    }

    #[test]
    fn allocation_failure_is_reported_as_out_of_memory() {
        let error = allocate_zeroed_with(4096, |_storage: &mut Vec<u8>, _len| {
            Err::<(), _>("injected allocation failure")
        })
        .unwrap_err();
        assert_eq!(
            error,
            DeviceError::OutOfMemory {
                requested_bytes: 4096,
                reason: "injected allocation failure".to_owned(),
            }
        );
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
