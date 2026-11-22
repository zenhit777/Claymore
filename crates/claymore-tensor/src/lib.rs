//! Claymore-owned tensor metadata and storage lifetime contracts.
//!
//! This crate is intentionally below model execution semantics. It owns no CUDA
//! handles and performs no arithmetic; device buffers remain opaque and are
//! released only through the backend's explicit, fallible contract.
use claymore_device::{DType, DeviceBuffer, DeviceError};
use thiserror::Error;

/// A validated tensor layout. Strides are measured in elements, not bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
    numel: usize,
    span_bytes: usize,
}

impl TensorLayout {
    /// Construct a contiguous layout with checked shape and byte arithmetic.
    pub fn contiguous(shape: impl Into<Vec<usize>>, dtype: DType) -> Result<Self, TensorError> {
        let shape = shape.into();
        let mut strides = vec![1; shape.len()];
        let mut stride = 1usize;
        for index in (0..shape.len()).rev() {
            strides[index] = stride;
            stride = stride
                .checked_mul(shape[index])
                .ok_or(TensorError::SizeOverflow)?;
        }
        Self::strided(shape, strides, dtype)
    }

    /// Construct a strided layout. Zero-sized dimensions are rejected in this
    /// initial contract; empty tensors require a separately specified storage
    /// and operation policy rather than an accidental zero-byte allocation.
    pub fn strided(
        shape: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
        dtype: DType,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let strides = strides.into();
        if shape.len() != strides.len() {
            return Err(TensorError::RankMismatch {
                shape_rank: shape.len(),
                stride_rank: strides.len(),
            });
        }
        let mut numel = 1usize;
        let mut max_offset = 0usize;
        for (&dimension, &stride) in shape.iter().zip(&strides) {
            if dimension == 0 {
                return Err(TensorError::EmptyDimension);
            }
            numel = numel
                .checked_mul(dimension)
                .ok_or(TensorError::SizeOverflow)?;
            let axis = (dimension - 1)
                .checked_mul(stride)
                .ok_or(TensorError::SizeOverflow)?;
            max_offset = max_offset
                .checked_add(axis)
                .ok_or(TensorError::SizeOverflow)?;
        }
        let span_bytes = max_offset
            .checked_add(1)
            .and_then(|n| n.checked_mul(dtype.element_size_bytes()))
            .ok_or(TensorError::SizeOverflow)?;
        Ok(Self {
            shape,
            strides,
            dtype,
            numel,
            span_bytes,
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

    pub fn span_bytes(&self) -> usize {
        self.span_bytes
    }

    pub fn is_contiguous(&self) -> bool {
        Self::contiguous(self.shape.clone(), self.dtype)
            .is_ok_and(|layout| layout.strides == self.strides)
    }
}

/// Tensor storage owns exactly one backend allocation. Views borrow this
/// owner, so Rust prevents releasing or mutating the allocation while a view
/// remains alive. In-flight asynchronous work must additionally retain the
/// backend completion token/guard required by the Device contract.
pub struct Tensor {
    storage: Box<dyn DeviceBuffer>,
    layout: TensorLayout,
    offset_bytes: usize,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("layout", &self.layout)
            .field("offset_bytes", &self.offset_bytes)
            .field("storage_len", &self.storage.len())
            .finish()
    }
}

impl Tensor {
    pub fn new(
        storage: Box<dyn DeviceBuffer>,
        shape: impl Into<Vec<usize>>,
        dtype: DType,
    ) -> Result<Self, TensorError> {
        let layout = TensorLayout::contiguous(shape, dtype)?;
        Self::with_layout(storage, layout, 0)
    }

    pub fn with_layout(
        storage: Box<dyn DeviceBuffer>,
        layout: TensorLayout,
        offset_bytes: usize,
    ) -> Result<Self, TensorError> {
        let end = offset_bytes
            .checked_add(layout.span_bytes())
            .ok_or(TensorError::SizeOverflow)?;
        if end > storage.len() {
            return Err(TensorError::StorageTooSmall {
                required: end,
                capacity: storage.len(),
            });
        }
        Ok(Self {
            storage,
            layout,
            offset_bytes,
        })
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub fn storage(&self) -> &dyn DeviceBuffer {
        self.storage.as_ref()
    }

    pub fn storage_mut(&mut self) -> &mut dyn DeviceBuffer {
        self.storage.as_mut()
    }

    pub fn into_storage(self) -> Box<dyn DeviceBuffer> {
        self.storage
    }

    pub fn view(
        &self,
        shape: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
        relative_offset_bytes: usize,
    ) -> Result<TensorView<'_>, TensorError> {
        let layout = TensorLayout::strided(shape, strides, self.layout.dtype())?;
        let offset_bytes = self
            .offset_bytes
            .checked_add(relative_offset_bytes)
            .ok_or(TensorError::SizeOverflow)?;
        let end = offset_bytes
            .checked_add(layout.span_bytes())
            .ok_or(TensorError::SizeOverflow)?;
        let tensor_end = self
            .offset_bytes
            .checked_add(self.layout.span_bytes())
            .ok_or(TensorError::SizeOverflow)?;
        if end > tensor_end {
            return Err(TensorError::StorageTooSmall {
                required: end,
                capacity: tensor_end,
            });
        }
        Ok(TensorView {
            storage: self.storage.as_ref(),
            layout,
            offset_bytes,
        })
    }
}

/// Borrowed view. It never owns or independently releases backend storage.
pub struct TensorView<'a> {
    storage: &'a dyn DeviceBuffer,
    layout: TensorLayout,
    offset_bytes: usize,
}

impl TensorView<'_> {
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub fn storage(&self) -> &dyn DeviceBuffer {
        self.storage
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    #[error("tensor shape contains a zero-sized dimension")]
    EmptyDimension,
    #[error("tensor shape and byte span overflow usize")]
    SizeOverflow,
    #[error("shape rank {shape_rank} differs from stride rank {stride_rank}")]
    RankMismatch {
        shape_rank: usize,
        stride_rank: usize,
    },
    #[error("tensor requires {required} bytes but storage capacity is {capacity}")]
    StorageTooSmall { required: usize, capacity: usize },
    #[error(transparent)]
    Device(#[from] DeviceError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_core::MemoryTier;
    use claymore_device::{CpuDevice, Device};

    #[test]
    fn contiguous_layout_uses_checked_element_strides() {
        let layout = TensorLayout::contiguous(vec![2, 3, 4], DType::F32).unwrap();
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert_eq!(layout.numel(), 24);
        assert_eq!(layout.span_bytes(), 96);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn strided_view_checks_byte_span_and_borrows_storage() {
        let device = CpuDevice::new();
        let storage = device.allocate(MemoryTier::Host, 64).unwrap();
        let tensor = Tensor::new(storage, vec![4, 4], DType::F32).unwrap();
        let view = tensor.view(vec![2, 2], vec![4, 1], 0).unwrap();
        assert_eq!(view.layout().strides(), &[4, 1]);
        assert_eq!(view.layout().span_bytes(), 24);
        assert_eq!(view.offset_bytes(), 0);
    }

    #[test]
    fn view_cannot_escape_parent_tensor_span() {
        let device = CpuDevice::new();
        let storage = device.allocate(MemoryTier::Host, 64).unwrap();
        let tensor = Tensor::new(storage, vec![2, 2], DType::F32).unwrap();
        assert!(matches!(
            tensor.view(vec![2, 2], vec![4, 1], 0),
            Err(TensorError::StorageTooSmall {
                required: 24,
                capacity: 16
            })
        ));
    }

    #[test]
    fn rejects_rank_mismatch_zero_dimensions_and_overflow() {
        assert!(matches!(
            TensorLayout::strided(vec![2, 3], vec![1], DType::F32),
            Err(TensorError::RankMismatch { .. })
        ));
        assert_eq!(
            TensorLayout::contiguous(vec![2, 0], DType::F32).unwrap_err(),
            TensorError::EmptyDimension
        );
        assert_eq!(
            TensorLayout::contiguous(vec![usize::MAX, 2], DType::F32).unwrap_err(),
            TensorError::SizeOverflow
        );
    }

    #[test]
    fn storage_capacity_is_checked_including_offset() {
        let device = CpuDevice::new();
        let storage = device.allocate(MemoryTier::Host, 32).unwrap();
        let layout = TensorLayout::contiguous(vec![4], DType::F32).unwrap();
        assert_eq!(
            Tensor::with_layout(storage, layout, 17).unwrap_err(),
            TensorError::StorageTooSmall {
                required: 33,
                capacity: 32
            }
        );
    }

    #[test]
    fn non_contiguous_layout_is_not_misreported_as_contiguous() {
        let layout = TensorLayout::strided(vec![2, 2], vec![3, 1], DType::F32).unwrap();
        assert!(!layout.is_contiguous());
        assert_eq!(layout.span_bytes(), 20);
    }
}
