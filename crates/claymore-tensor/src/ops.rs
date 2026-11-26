//! Deterministic CPU reference operations for the native tensor contract.
//!
//! Both supported dtypes share one numerical policy: values are read as their
//! exact stored representation, interpreted as FP32, and every intermediate
//! (including all reductions and the whole matmul accumulation) is evaluated
//! in FP32. Only the final result is converted back to the output dtype with
//! `Bf16::from_f32`, i.e. round-to-nearest-even for BF16. No operation ever
//! casts implicitly, broadcasts or falls back to another backend.
#![allow(clippy::too_many_arguments)]

use claymore_core::MemoryTier;
use claymore_device::{DType, Device, DeviceError, DeviceKind};
use thiserror::Error;

use crate::{Bf16, Tensor, TensorError, TensorLayout};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum OpError {
    #[error("operation supports F32 and BF16 tensors only")]
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
    #[error("attention scale must be finite and non-zero")]
    InvalidAttentionScale,
    #[error("soft cap must be finite and strictly positive")]
    InvalidSoftCap,
    #[error("index tensors must encode exact integer rows")]
    InvalidIndex,
    #[error("top-k selection is invalid: top_k {0} for {1} experts")]
    InvalidTopK(usize, usize),
    #[error("scatter-add requires a non-zero output row count")]
    InvalidOutputRows,
    #[error("index/weight/flag tensors must be F32")]
    IndexDTypeMismatch,
    #[error("embedding scale must be finite")]
    InvalidEmbedScale,
    #[error("scale factor must be finite")]
    InvalidScaleFactor,
    #[error("RoPE parameters are invalid: {0}")]
    InvalidRopeParameter(String),
    #[error("attention head configuration is invalid: {0}")]
    InvalidHeadConfiguration(String),
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

/// Host-side scalar buffer with the exact stored representation preserved.
pub(crate) enum Numeric {
    F32(Vec<f32>),
    BF16(Vec<Bf16>),
}

impl Numeric {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Numeric::F32(_) => DType::F32,
            Numeric::BF16(_) => DType::BF16,
        }
    }

    /// Exact widening to FP32. BF16 -> FP32 is bit-exact for every pattern.
    pub(crate) fn to_f32(&self) -> Vec<f32> {
        match self {
            Numeric::F32(values) => values.clone(),
            Numeric::BF16(values) => values.iter().map(|value| value.to_f32()).collect(),
        }
    }
}

pub(crate) fn supported_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
}

pub(crate) fn read_numeric(device: &dyn Device, tensor: &Tensor) -> Result<Numeric, OpError> {
    if device.info().kind != DeviceKind::Cpu || tensor.storage().tier() != MemoryTier::Host {
        return Err(OpError::CpuReferenceRequiresHostBuffers);
    }
    let dtype = tensor.layout().dtype();
    if !supported_dtype(dtype) {
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
    match dtype {
        DType::F32 => Ok(Numeric::F32(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect(),
        )),
        DType::BF16 => Ok(Numeric::BF16(
            bytes
                .chunks_exact(2)
                .map(|chunk| Bf16::from_ne_bytes(chunk.try_into().expect("two-byte chunk")))
                .collect(),
        )),
        _ => Err(OpError::UnsupportedDType),
    }
}

pub(crate) fn write_numeric(
    device: &dyn Device,
    values: &[f32],
    shape: Vec<usize>,
    dtype: DType,
) -> Result<Tensor, OpError> {
    if !supported_dtype(dtype) {
        return Err(OpError::UnsupportedDType);
    }
    let layout = TensorLayout::contiguous(shape, dtype)?;
    if layout.numel() != values.len() {
        return Err(OpError::ShapeMismatch);
    }
    let byte_len = layout.span_bytes();
    let mut storage = device.allocate(
        MemoryTier::Host,
        u64::try_from(byte_len).map_err(|_| DeviceError::SizeOverflow)?,
    )?;
    let bytes: Vec<u8> = match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .map(|value| Bf16::from_f32(*value).to_ne_bytes())
            .flat_map(|bytes| bytes.into_iter())
            .collect(),
        _ => return Err(OpError::UnsupportedDType),
    };
    device.write(storage.as_mut(), 0, &bytes)?;
    Ok(Tensor::with_layout(storage, layout, 0)?)
}

/// Elementwise addition. Broadcasting and implicit dtype conversion are deliberately unsupported.
pub fn add(device: &dyn Device, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, OpError> {
    if lhs.layout().shape() != rhs.layout().shape() {
        return Err(OpError::ShapeMismatch);
    }
    let a = read_numeric(device, lhs)?;
    let b = read_numeric(device, rhs)?;
    if a.dtype() != b.dtype() {
        return Err(OpError::MixedDType);
    }
    let values: Vec<f32> = a
        .to_f32()
        .into_iter()
        .zip(b.to_f32())
        .map(|(x, y)| x + y)
        .collect();
    write_numeric(device, &values, lhs.layout().shape().to_vec(), a.dtype())
}

/// Row-major matrix multiplication for rank-2 F32/BF16 tensors.
///
/// Every product is accumulated in FP32 with `mul_add`, in the same index order
/// as the CUDA kernel. Only the final FP32 value is rounded to BF16 when the
/// output dtype is BF16.
pub fn matmul(device: &dyn Device, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, OpError> {
    let a_shape = lhs.layout().shape();
    let b_shape = rhs.layout().shape();
    if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
        return Err(OpError::ShapeMismatch);
    }
    let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
    let a = read_numeric(device, lhs)?;
    let b = read_numeric(device, rhs)?;
    if a.dtype() != b.dtype() {
        return Err(OpError::MixedDType);
    }
    let a = a.to_f32();
    let b = b.to_f32();
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
    write_numeric(device, &out, vec![m, n], lhs.layout().dtype())
}

/// Numerically stable softmax over the final contiguous dimension.
///
/// The maximum subtraction, exponentials, sum and reciprocal all run in FP32;
/// BF16 input values are widened bit-exactly first and the probabilities are
/// rounded to BF16 only once, at output.
pub fn softmax_last_dim(device: &dyn Device, input: &Tensor) -> Result<Tensor, OpError> {
    let shape = input.layout().shape();
    let width = *shape.last().ok_or(OpError::InvalidRank)?;
    let source = read_numeric(device, input)?;
    let dtype = source.dtype();
    let values = source.to_f32();
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
    write_numeric(device, &output, shape.to_vec(), dtype)
}

/// RMS normalization on the final dimension with an explicit scale vector of
/// the same dtype as the input.
///
/// `sum(x²)`, the mean square, `sqrt` and the reciprocal are evaluated in FP32.
/// BF16 inputs and scale are widened bit-exactly; the product is rounded to
/// BF16 once, at output.
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
    let source = read_numeric(device, input)?;
    let weights = read_numeric(device, scale)?;
    if source.dtype() != weights.dtype() {
        return Err(OpError::MixedDType);
    }
    let dtype = source.dtype();
    let values = source.to_f32();
    let weights = weights.to_f32();
    let mut output = vec![0.0f32; values.len()];
    for (src, dst) in values
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let mean_square = src.iter().map(|x| x * x).sum::<f32>() / width as f32;
        let inv_rms = 1.0f32 / (mean_square + epsilon).sqrt();
        for index in 0..width {
            dst[index] = src[index] * inv_rms * weights[index];
        }
    }
    write_numeric(device, &output, shape.to_vec(), dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_device::CpuDevice;

    fn tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut buffer = device
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        device.write(buffer.as_mut(), 0, &bytes).unwrap();
        Tensor::new(buffer, shape, DType::F32).unwrap()
    }

    fn bf16_tensor(device: &CpuDevice, shape: Vec<usize>, values: &[Bf16]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut buffer = device
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        device.write(buffer.as_mut(), 0, &bytes).unwrap();
        Tensor::new(buffer, shape, DType::BF16).unwrap()
    }

    fn values(device: &CpuDevice, tensor: &Tensor) -> Vec<f32> {
        match read_numeric(device, tensor).unwrap() {
            Numeric::F32(values) => values,
            Numeric::BF16(values) => values.iter().map(|value| value.to_f32()).collect(),
        }
    }

    fn stored_bf16(device: &CpuDevice, tensor: &Tensor) -> Vec<Bf16> {
        match read_numeric(device, tensor).unwrap() {
            Numeric::BF16(values) => values,
            Numeric::F32(_) => panic!("expected a BF16 tensor"),
        }
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

    fn bf(values: &[f32]) -> Vec<Bf16> {
        values.iter().copied().map(Bf16::from_f32).collect()
    }

    #[test]
    fn bf16_add_rounds_only_the_output() {
        let device = CpuDevice::new();
        let a = bf16_tensor(&device, vec![3], &bf(&[1.0, -2.5, 0.125]));
        let b = bf16_tensor(&device, vec![3], &bf(&[2.0, 1.25, 0.375]));
        let out = stored_bf16(&device, &add(&device, &a, &b).unwrap());
        assert_eq!(out, bf(&[3.0, -1.25, 0.5]));
    }

    #[test]
    fn bf16_matmul_accumulates_in_fp32() {
        let device = CpuDevice::new();
        // Odd inner dimension (5) so a product is not exact in BF16.
        let a = bf16_tensor(
            &device,
            vec![3, 5],
            &bf(&[
                1.0, 0.5, -2.0, 3.0, 0.25, 0.125, -4.0, 2.0, 1.0, 0.75, -1.0, 2.0, 0.5, -0.25, 4.0,
            ]),
        );
        let b = bf16_tensor(
            &device,
            vec![5, 2],
            &bf(&[2.0, 1.0, -1.0, 0.5, 3.0, -2.0, 0.25, 1.5, 4.0, -0.75]),
        );
        let out = stored_bf16(&device, &matmul(&device, &a, &b).unwrap());
        let expected: Vec<f32> = [
            1.0f32, 0.5, -2.0, 3.0, 0.25, 0.125, -4.0, 2.0, 1.0, 0.75, -1.0, 2.0, 0.5, -0.25, 4.0,
        ]
        .chunks(5)
        .flat_map(|row| {
            let weights = [2.0f32, 1.0, -1.0, 0.5, 3.0, -2.0, 0.25, 1.5, 4.0, -0.75];
            (0..2).map(move |col| {
                row.iter()
                    .enumerate()
                    .map(|(k, value)| value * weights[k * 2 + col])
                    .sum::<f32>()
            })
        })
        .collect();
        let expected = bf(&expected);
        for (index, (got, want)) in out.iter().zip(&expected).enumerate() {
            let difference = (got.to_f32() - want.to_f32()).abs();
            assert!(
                difference <= 2f32.powi(-7) * want.to_f32().abs().max(1.0),
                "matmul[{index}]: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn bf16_softmax_is_stable_and_normalized_after_rounding() {
        let device = CpuDevice::new();
        let a = bf16_tensor(
            &device,
            vec![2, 3],
            &bf(&[10_000.0, -10_000.0, 0.0, 1.0, 2.0, 3.0]),
        );
        let out = stored_bf16(&device, &softmax_last_dim(&device, &a).unwrap());
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|value| value.is_finite()));
        for row in out.chunks(3) {
            let sum: f32 = row.iter().map(|value| value.to_f32()).sum();
            assert!((sum - 1.0).abs() < 4e-3, "row sum {sum}");
        }
        assert_eq!(out[0], Bf16::from_bits(0x3F80));
        assert_eq!(out[1], Bf16::from_bits(Bf16::ZERO_BITS));
    }

    #[test]
    fn bf16_rms_norm_uses_bf16_scale_and_fp32_reduction() {
        let device = CpuDevice::new();
        let a = bf16_tensor(&device, vec![2, 3], &bf(&[3.0, 4.0, 0.0, 1.0, -2.0, 2.0]));
        let scale = bf16_tensor(&device, vec![3], &bf(&[1.0, 2.0, 0.5]));
        let out = stored_bf16(&device, &rms_norm(&device, &a, &scale, 1e-5).unwrap());
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|value| value.is_finite()));
        for row in out.chunks(3) {
            let mean_square: f32 =
                row.iter().map(|value| value.to_f32().powi(2)).sum::<f32>() / 3.0;
            assert!(mean_square < 1e6, "unexpected magnitude {mean_square}");
        }
    }

    #[test]
    fn bf16_ops_reject_mixed_dtype_and_unsupported_dtype() {
        let device = CpuDevice::new();
        let a = bf16_tensor(&device, vec![2], &bf(&[1.0, 2.0]));
        let b = tensor(&device, vec![2], &[1.0, 2.0]);
        assert_eq!(add(&device, &a, &b).unwrap_err(), OpError::MixedDType);

        let bytes: Vec<u8> = vec![0; 4];
        let mut storage = device.allocate(MemoryTier::Host, 4).unwrap();
        device.write(storage.as_mut(), 0, &bytes).unwrap();
        let f16 = Tensor::new(storage, vec![2], DType::F16).unwrap();
        assert_eq!(
            softmax_last_dim(&device, &f16).unwrap_err(),
            OpError::UnsupportedDType
        );
    }

    #[test]
    fn bf16_shape_scale_layout_and_epsilon_errors_are_typed() {
        let device = CpuDevice::new();
        let a = bf16_tensor(&device, vec![2, 3], &bf(&[1.0; 6]));
        let bad_inner = bf16_tensor(&device, vec![2, 3], &bf(&[1.0; 6]));
        assert_eq!(
            matmul(&device, &a, &bad_inner).unwrap_err(),
            OpError::ShapeMismatch
        );

        let short_scale = bf16_tensor(&device, vec![2], &bf(&[1.0, 1.0]));
        assert_eq!(
            rms_norm(&device, &a, &short_scale, 0.0).unwrap_err(),
            OpError::ScaleMismatch
        );
        let good_scale = bf16_tensor(&device, vec![3], &bf(&[1.0, 1.0, 1.0]));
        assert_eq!(
            rms_norm(&device, &a, &good_scale, f32::NAN).unwrap_err(),
            OpError::InvalidEpsilon
        );

        let storage = device.allocate(MemoryTier::Host, 20).unwrap();
        let strided = Tensor::with_layout(
            storage,
            TensorLayout::strided(vec![2, 2], vec![3, 1], DType::BF16).unwrap(),
            0,
        )
        .unwrap();
        assert_eq!(
            softmax_last_dim(&device, &strided).unwrap_err(),
            OpError::NonContiguous
        );
    }
}
