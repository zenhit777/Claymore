//! Deterministic CPU reference operations for the native tensor contract.
use claymore_core::MemoryTier;
use claymore_device::{DType, Device, DeviceError, DeviceKind};
use thiserror::Error;

use crate::{Tensor, TensorError, TensorLayout};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum OpError {
    #[error("operation currently supports F32 tensors only")]
    UnsupportedDType,
    #[error("operation inputs must use a single dtype")]
    MixedDType,
    #[error("operation inputs must belong to the same device instance")]
    MixedDevices,
    #[error("operation requires contiguous tensor layouts")]
    NonContiguous,
    #[error("tensor shapes are incompatible for this operation")]
    ShapeMismatch,
    #[error("softmax and RMSNorm require rank >= 1")]
    InvalidRank,
    #[error("RMSNorm scale length must match the final tensor dimension")]
    ScaleMismatch,
    #[error("CPU reference operations require the CPU device and host-resident tensors")]
    CpuReferenceRequiresHostBuffers,
    #[error("epsilon must be finite and non-negative")]
    InvalidEpsilon,
    #[error("backend {backend} refused execution: {reason}")]
    Backend {
        backend: &'static str,
        reason: String,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Device(#[from] DeviceError),
}

fn read_f32(device: &dyn Device, tensor: &Tensor) -> Result<Vec<f32>, OpError> {
    if device.info().kind != DeviceKind::Cpu || tensor.storage().tier() != MemoryTier::Host {
        return Err(OpError::CpuReferenceRequiresHostBuffers);
    }
    if tensor.layout().dtype() != DType::F32 {
        return Err(OpError::UnsupportedDType);
    }
    if !tensor.layout().is_contiguous() {
        return Err(OpError::NonContiguous);
    }
    let bytes = device.read(
        tensor.storage(),
        tensor.offset_bytes(),
        tensor.layout().span_bytes(),
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn write_f32(device: &dyn Device, values: &[f32], shape: Vec<usize>) -> Result<Tensor, OpError> {
    let layout = TensorLayout::contiguous(shape, DType::F32)?;
    let byte_len = layout.span_bytes();
    let mut storage = device.allocate(
        MemoryTier::Host,
        u64::try_from(byte_len).map_err(|_| DeviceError::SizeOverflow)?,
    )?;
    let mut bytes = Vec::with_capacity(byte_len);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    device.write(storage.as_mut(), 0, &bytes)?;
    Ok(Tensor::with_layout(storage, layout, 0)?)
}

/// Elementwise addition. Broadcasting and implicit dtype conversion are deliberately unsupported.
pub fn add(device: &dyn Device, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, OpError> {
    if lhs.layout().shape() != rhs.layout().shape() {
        return Err(OpError::ShapeMismatch);
    }
    let a = read_f32(device, lhs)?;
    let b = read_f32(device, rhs)?;
    let values: Vec<_> = a.into_iter().zip(b).map(|(x, y)| x + y).collect();
    write_f32(device, &values, lhs.layout().shape().to_vec())
}

/// Row-major matrix multiplication for rank-2 F32 tensors.
pub fn matmul(device: &dyn Device, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, OpError> {
    let a_shape = lhs.layout().shape();
    let b_shape = rhs.layout().shape();
    if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
        return Err(OpError::ShapeMismatch);
    }
    let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
    let a = read_f32(device, lhs)?;
    let b = read_f32(device, rhs)?;
    let mut out = vec![0.0f32; m.checked_mul(n).ok_or(DeviceError::SizeOverflow)?];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc = a[row * k + inner].mul_add(b[inner * n + col], acc);
            }
            out[row * n + col] = acc;
        }
    }
    write_f32(device, &out, vec![m, n])
}

/// Numerically stable softmax over the final contiguous dimension.
pub fn softmax_last_dim(device: &dyn Device, input: &Tensor) -> Result<Tensor, OpError> {
    let shape = input.layout().shape();
    let width = *shape.last().ok_or(OpError::InvalidRank)?;
    let values = read_f32(device, input)?;
    let mut output = vec![0.0f32; values.len()];
    for (src, dst) in values
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = src.iter().map(|value| (*value - max).exp()).sum();
        for (out, value) in dst.iter_mut().zip(src) {
            *out = (*value - max).exp() / sum;
        }
    }
    write_f32(device, &output, shape.to_vec())
}

/// RMS normalization on the final dimension with an explicit F32 scale vector.
pub fn rms_norm(
    device: &dyn Device,
    input: &Tensor,
    scale: &Tensor,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(OpError::InvalidEpsilon);
    }
    let shape = input.layout().shape();
    let width = *shape.last().ok_or(OpError::InvalidRank)?;
    if scale.layout().shape() != [width] {
        return Err(OpError::ScaleMismatch);
    }
    let values = read_f32(device, input)?;
    let weights = read_f32(device, scale)?;
    let mut output = vec![0.0f32; values.len()];
    for (src, dst) in values
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let mean_square = src.iter().map(|x| x * x).sum::<f32>() / width as f32;
        let inv_rms = (mean_square + epsilon).sqrt().recip();
        for index in 0..width {
            dst[index] = src[index] * inv_rms * weights[index];
        }
    }
    write_f32(device, &output, shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_device::{CpuDevice, Device};

    fn tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut buffer = device
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        device.write(buffer.as_mut(), 0, &bytes).unwrap();
        Tensor::new(buffer, shape, DType::F32).unwrap()
    }

    fn values(device: &CpuDevice, tensor: &Tensor) -> Vec<f32> {
        device
            .read(
                tensor.storage(),
                tensor.offset_bytes(),
                tensor.layout().span_bytes(),
            )
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn add_is_elementwise_and_rejects_implicit_broadcast() {
        let device = CpuDevice::new();
        let a = tensor(&device, vec![2], &[1.0, 2.0]);
        let b = tensor(&device, vec![2], &[3.0, 4.0]);
        assert_eq!(values(&device, &add(&device, &a, &b).unwrap()), [4.0, 6.0]);
        let c = tensor(&device, vec![1], &[1.0]);
        assert_eq!(add(&device, &a, &c).unwrap_err(), OpError::ShapeMismatch);
    }

    #[test]
    fn matmul_matches_reference_result() {
        let device = CpuDevice::new();
        let a = tensor(&device, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tensor(&device, vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(
            values(&device, &matmul(&device, &a, &b).unwrap()),
            [58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let device = CpuDevice::new();
        let a = tensor(&device, vec![1, 3], &[1000.0, 1001.0, 1002.0]);
        let out = values(&device, &softmax_last_dim(&device, &a).unwrap());
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rms_norm_applies_explicit_scale() {
        let device = CpuDevice::new();
        let a = tensor(&device, vec![1, 2], &[3.0, 4.0]);
        let scale = tensor(&device, vec![2], &[1.0, 2.0]);
        let out = values(&device, &rms_norm(&device, &a, &scale, 0.0).unwrap());
        assert!((out[0] - 3.0 / 12.5f32.sqrt()).abs() < 1e-6);
        assert!((out[1] - 8.0 / 12.5f32.sqrt()).abs() < 1e-6);
    }
}
