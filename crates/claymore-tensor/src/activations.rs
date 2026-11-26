//! Numerically explicit activation references shared by model adapters.
//!
//! These functions are backend-independent CPU references. They do not add
//! model graph semantics to the tensor crate and never perform an implicit
//! dtype conversion.

use claymore_device::Device;

use crate::ops::{read_numeric, write_numeric, OpError};
use crate::Tensor;

/// SiLU activation, evaluated in FP32 and rounded once to the input dtype.
///
/// The input must be a contiguous F32 or BF16 tensor resident on the CPU.
/// NaNs propagate, positive infinity remains positive infinity, and negative
/// infinity maps to negative zero (the limiting value of SiLU).
pub fn silu(device: &dyn Device, input: &Tensor) -> Result<Tensor, OpError> {
    let source = read_numeric(device, input)?;
    let dtype = source.dtype();
    let values = source.to_f32();
    let output: Vec<f32> = values.into_iter().map(silu_scalar).collect();
    write_numeric(device, &output, input.layout().shape().to_vec(), dtype)
}

#[inline]
fn silu_scalar(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == f32::NEG_INFINITY {
        return -0.0;
    }

    // Avoid exp overflow for large negative inputs while preserving the
    // ordinary sigmoid expression for non-negative inputs.
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    value * sigmoid
}

/// Logistic sigmoid, evaluated in FP32 and rounded once to the input dtype.
pub fn sigmoid(device: &dyn Device, input: &Tensor) -> Result<Tensor, OpError> {
    let source = read_numeric(device, input)?;
    let dtype = source.dtype();
    let values = source.to_f32();
    let output: Vec<f32> = values.into_iter().map(sigmoid_scalar).collect();
    write_numeric(device, &output, input.layout().shape().to_vec(), dtype)
}

#[inline]
pub(crate) fn sigmoid_scalar(value: f32) -> f32 {
    if value.is_nan() {
        return value;
    }
    if value == f32::INFINITY {
        return 1.0;
    }
    if value == f32::NEG_INFINITY {
        return 0.0;
    }
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_core::MemoryTier;
    use claymore_device::{CpuDevice, DType, Device};

    fn f32_tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let mut storage = device.allocate(MemoryTier::Host, bytes.len() as u64).unwrap();
        device.write(storage.as_mut(), 0, &bytes).unwrap();
        Tensor::new(storage, shape, DType::F32).unwrap()
    }

    fn bf16_tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values
            .iter()
            .map(|value| crate::Bf16::from_f32(*value).to_ne_bytes())
            .flat_map(|bytes| bytes.into_iter())
            .collect();
        let mut storage = device.allocate(MemoryTier::Host, bytes.len() as u64).unwrap();
        device.write(storage.as_mut(), 0, &bytes).unwrap();
        Tensor::new(storage, shape, DType::BF16).unwrap()
    }

    fn read_f32(device: &CpuDevice, tensor: &Tensor) -> Vec<f32> {
        read_numeric(device, tensor).unwrap().to_f32()
    }

    #[test]
    fn silu_matches_reference_values_for_f32() {
        let device = CpuDevice::new();
        let input = f32_tensor(
            &device,
            vec![7],
            &[-10.0, -2.0, -0.0, 0.0, 1.0, 2.0, 10.0],
        );
        let output = read_f32(&device, &silu(&device, &input).unwrap());
        for (actual, source) in output.iter().zip([-10.0f32, -2.0, -0.0, 0.0, 1.0, 2.0, 10.0]) {
            let expected = silu_scalar(source);
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn silu_preserves_bf16_output_dtype_and_rounds_once() {
        let device = CpuDevice::new();
        let input = bf16_tensor(&device, vec![4], &[-4.0, -1.0, 1.0, 4.0]);
        let output = silu(&device, &input).unwrap();
        assert_eq!(output.layout().dtype(), DType::BF16);
        let actual = read_f32(&device, &output);
        for (value, source) in actual.iter().zip([-4.0f32, -1.0, 1.0, 4.0]) {
            let expected = crate::Bf16::from_f32(silu_scalar(source)).to_f32();
            assert_eq!(value.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn silu_has_explicit_nan_and_infinity_semantics() {
        let device = CpuDevice::new();
        let input = f32_tensor(
            &device,
            vec![4],
            &[f32::NEG_INFINITY, -100.0, f32::INFINITY, f32::NAN],
        );
        let output = read_f32(&device, &silu(&device, &input).unwrap());
        assert_eq!(output[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(output[1], silu_scalar(-100.0));
        assert_eq!(output[2], f32::INFINITY);
        assert!(output[3].is_nan());
    }
    #[test]
    fn sigmoid_has_stable_finite_and_infinite_semantics() {
        let device = CpuDevice::new();
        let input = f32_tensor(
            &device,
            vec![5],
            &[f32::NEG_INFINITY, -100.0, 0.0, 100.0, f32::INFINITY],
        );
        let output = read_f32(&device, &sigmoid(&device, &input).unwrap());
        assert_eq!(output[0], 0.0);
        assert!(output[1] >= 0.0 && output[1] < 1e-30);
        assert_eq!(output[2], 0.5);
        assert_eq!(output[3], 1.0);
        assert_eq!(output[4], 1.0);
    }
}
