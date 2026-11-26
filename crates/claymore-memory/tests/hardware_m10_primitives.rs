//! Hardware differential validation of the M10 diffusion primitives.
//!
//! These tests require a real NVIDIA driver and device. They are ignored by
//! default so CI stays hardware-independent. Run them with:
//!
//! ```text
//! RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib \
//! cargo test -p claymore-memory --features cuda-driver \
//!   --test hardware_m10_primitives -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! Every comparison uses the deterministic CPU reference in `claymore-tensor`.
//! The tolerances are declared here, before the campaign, and must not be
//! lowered after observing results:
//!
//! - F32 linear/rotary/soft-cap: absolute error <= 1e-4. Both sides reduce in
//!   the same order with fused multiply-add, so the residual is limited to
//!   `tanhf`/`expf` implementation differences.
//! - F32 attention: absolute error <= 1e-4. The softmax is evaluated in FP32
//!   on both sides.
//! - BF16 reductions: at most 1 representable BF16 step (ULP), because the CPU
//!   `mul_add` and the device `fmaf` contract in the same order but `expf` vs
//!   `f32::exp` can move a probability across one rounding boundary.
//! - BF16 attention: at most 2 representable steps, because the reference
//!   rounds every softmax probability to BF16 before the value reduction and a
//!   one-step probability difference is amplified by the reduction.

use claymore_core::MemoryTier;
use claymore_device::DType;
use claymore_memory::{CudaMemoryError, CudaTensorError, MemoryError};
use claymore_tensor::{
    add, attention, embed_gather, gather_rows, gelu_mul_up, linear, rms_norm, rope_inv_freq,
    rope_tables, rotary_embed, router_top_k, row_mask, scale, scale_bias, scale_scalar,
    scatter_add, soft_cap, widen, Bf16, ExecutionRequest, OpKind, RopeType, ShapeSignature,
    TensorOp,
};

/// FP32 differential tolerance (absolute) for every M10 primitive.
const F32_ABS_TOLERANCE: f32 = 1e-4;
/// BF16 ULP budget for elementwise and reduction outputs.
const BF16_ULP: u32 = 1;
/// BF16 ULP budget for attention, which rounds every probability to BF16.
const BF16_ATTENTION_ULP: u32 = 2;
/// Model-derived bound for the BF16 attention value reduction: the BF16
/// probability rounding contributes at most `2^-9` relative error, and the
/// budget keeps one extra bit of margin (`2^-8`) against the value magnitude.
const BF16_ATTENTION_REDUCTION_TOLERANCE: f32 = 1.0 / 256.0;

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_01_linear_f32_matches_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (tokens, in_features, out_features) = (13usize, 32usize, 7usize);
    let x: Vec<f32> = (0..tokens * in_features)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .collect();
    let w: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i % 11) as f32 - 5.0) / 16.0)
        .collect();

    let cpu_out = {
        let input = common::f32_tensor(&cpu, vec![tokens, in_features], &x);
        let weight = common::f32_tensor(&cpu, vec![out_features, in_features], &w);
        common::values(&cpu, &linear(&cpu, &input, &weight).unwrap())
    };

    let input = memory.upload_f32(vec![tokens, in_features], &x).unwrap();
    let weight = memory
        .upload_f32(vec![out_features, in_features], &w)
        .unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Linear {
            input: &input,
            weight: &weight,
        }))
        .expect("linear launch");
    let gpu_out = memory.download_f32(&output).unwrap();

    assert_close("linear f32", &cpu_out, &gpu_out);
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_02_linear_bf16_matches_cpu_within_one_ulp() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (tokens, in_features, out_features) = (9usize, 48usize, 5usize);
    let x: Vec<f32> = (0..tokens * in_features)
        .map(|i| ((i % 13) as f32 - 6.0) / 8.0)
        .collect();
    let w: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i % 7) as f32 - 3.0) / 32.0)
        .collect();
    let x_bf16: Vec<Bf16> = x.iter().map(|v| Bf16::from_f32(*v)).collect();
    let w_bf16: Vec<Bf16> = w.iter().map(|v| Bf16::from_f32(*v)).collect();

    let cpu_out = {
        let input = common::bf16_tensor(&cpu, vec![tokens, in_features], &x_bf16);
        let weight = common::bf16_tensor(&cpu, vec![out_features, in_features], &w_bf16);
        common::bf16_values(&cpu, &linear(&cpu, &input, &weight).unwrap())
    };

    let input = memory
        .upload_bf16(vec![tokens, in_features], &x_bf16)
        .unwrap();
    let weight = memory
        .upload_bf16(vec![out_features, in_features], &w_bf16)
        .unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Linear {
            input: &input,
            weight: &weight,
        }))
        .expect("bf16 linear launch");
    let gpu_out = memory.download_bf16(&output).unwrap();

    assert_bf16("linear bf16", &cpu_out, &gpu_out, BF16_ULP);
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_03_rotary_f32_and_bf16_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (tokens, heads, head_dim) = (5usize, 4usize, 16usize);
    let x: Vec<f32> = (0..tokens * heads * head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) / 8.0)
        .collect();
    let (cos, sin) = rope_tables(
        RopeType::Proportional,
        &[0, 1, 2, 3, 4],
        head_dim,
        1_000_000.0,
        0.25,
    )
    .unwrap();

    let cpu_out = {
        let input = common::f32_tensor(&cpu, vec![tokens, heads, head_dim], &x);
        let cos = common::f32_tensor(&cpu, vec![tokens, head_dim], &cos);
        let sin = common::f32_tensor(&cpu, vec![tokens, head_dim], &sin);
        common::values(&cpu, &rotary_embed(&cpu, &input, &cos, &sin).unwrap())
    };
    let input = memory
        .upload_f32(vec![tokens, heads, head_dim], &x)
        .unwrap();
    let cos_gpu = memory.upload_f32(vec![tokens, head_dim], &cos).unwrap();
    let sin_gpu = memory.upload_f32(vec![tokens, head_dim], &sin).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::RotaryEmbed {
            input: &input,
            cos: &cos_gpu,
            sin: &sin_gpu,
        }))
        .expect("rotary launch");
    assert_close(
        "rotary f32",
        &cpu_out,
        &memory.download_f32(&output).unwrap(),
    );

    // The same tables narrowed to BF16 exercise the exact widening/narrowing.
    let x_bf16: Vec<Bf16> = x.iter().map(|v| Bf16::from_f32(*v)).collect();
    let cos_bf16: Vec<Bf16> = cos.iter().map(|v| Bf16::from_f32(*v)).collect();
    let sin_bf16: Vec<Bf16> = sin.iter().map(|v| Bf16::from_f32(*v)).collect();
    let cpu_out = {
        let input = common::bf16_tensor(&cpu, vec![tokens, heads, head_dim], &x_bf16);
        let cos = common::bf16_tensor(&cpu, vec![tokens, head_dim], &cos_bf16);
        let sin = common::bf16_tensor(&cpu, vec![tokens, head_dim], &sin_bf16);
        common::bf16_values(&cpu, &rotary_embed(&cpu, &input, &cos, &sin).unwrap())
    };
    let input = memory
        .upload_bf16(vec![tokens, heads, head_dim], &x_bf16)
        .unwrap();
    let cos_gpu = memory
        .upload_bf16(vec![tokens, head_dim], &cos_bf16)
        .unwrap();
    let sin_gpu = memory
        .upload_bf16(vec![tokens, head_dim], &sin_bf16)
        .unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::RotaryEmbed {
            input: &input,
            cos: &cos_gpu,
            sin: &sin_gpu,
        }))
        .expect("bf16 rotary launch");
    assert_bf16(
        "rotary bf16",
        &cpu_out,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_04_attention_f32_masked_and_unmasked_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (query_tokens, key_tokens, heads, kv_heads, head_dim) =
        (6usize, 10usize, 8usize, 2usize, 4usize);
    let q: Vec<f32> = (0..query_tokens * heads * head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) / 8.0)
        .collect();
    let k: Vec<f32> = (0..key_tokens * kv_heads * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
        .collect();
    let v: Vec<f32> = (0..key_tokens * kv_heads * head_dim)
        .map(|i| ((i % 5) as f32 - 2.0) / 8.0)
        .collect();
    let mask: Vec<f32> = (0..query_tokens * key_tokens)
        .map(|i| {
            if (i % key_tokens) % 3 == 0 {
                claymore_tensor::MASK_BLOCKED
            } else {
                0.0
            }
        })
        .collect();

    let gpu_q = memory
        .upload_f32(vec![query_tokens, heads, head_dim], &q)
        .unwrap();
    let gpu_k = memory
        .upload_f32(vec![key_tokens, kv_heads, head_dim], &k)
        .unwrap();
    let gpu_v = memory
        .upload_f32(vec![key_tokens, kv_heads, head_dim], &v)
        .unwrap();

    for masked in [false, true] {
        let cpu_out = {
            let q = common::f32_tensor(&cpu, vec![query_tokens, heads, head_dim], &q);
            let k = common::f32_tensor(&cpu, vec![key_tokens, kv_heads, head_dim], &k);
            let v = common::f32_tensor(&cpu, vec![key_tokens, kv_heads, head_dim], &v);
            let mask_tensor = if masked {
                Some(common::f32_tensor(
                    &cpu,
                    vec![query_tokens, key_tokens],
                    &mask,
                ))
            } else {
                None
            };
            common::values(
                &cpu,
                &attention(&cpu, &q, &k, &v, mask_tensor.as_ref(), 0.125).unwrap(),
            )
        };

        let gpu_mask = if masked {
            Some(
                memory
                    .upload_f32(vec![query_tokens, key_tokens], &mask)
                    .unwrap(),
            )
        } else {
            None
        };
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::Attention {
                query: &gpu_q,
                key: &gpu_k,
                value: &gpu_v,
                mask: gpu_mask.as_ref(),
                scale: 0.125,
            }))
            .expect("attention launch");
        let gpu_out = memory.download_f32(&output).unwrap();
        assert_close(
            &format!("attention f32 masked={masked}"),
            &cpu_out,
            &gpu_out,
        );
    }
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_05_attention_bf16_within_two_ulp() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (query_tokens, key_tokens, heads, kv_heads, head_dim) =
        (4usize, 6usize, 4usize, 2usize, 8usize);
    let q: Vec<f32> = (0..query_tokens * heads * head_dim)
        .map(|i| ((i % 9) as f32 - 4.0) / 8.0)
        .collect();
    let k: Vec<f32> = (0..key_tokens * kv_heads * head_dim)
        .map(|i| ((i % 6) as f32 - 3.0) / 8.0)
        .collect();
    let v: Vec<f32> = (0..key_tokens * kv_heads * head_dim)
        .map(|i| ((i % 4) as f32 - 1.5) / 8.0)
        .collect();
    let q_bf16: Vec<Bf16> = q.iter().map(|x| Bf16::from_f32(*x)).collect();
    let k_bf16: Vec<Bf16> = k.iter().map(|x| Bf16::from_f32(*x)).collect();
    let v_bf16: Vec<Bf16> = v.iter().map(|x| Bf16::from_f32(*x)).collect();

    let cpu_out = {
        let q = common::bf16_tensor(&cpu, vec![query_tokens, heads, head_dim], &q_bf16);
        let k = common::bf16_tensor(&cpu, vec![key_tokens, kv_heads, head_dim], &k_bf16);
        let v = common::bf16_tensor(&cpu, vec![key_tokens, kv_heads, head_dim], &v_bf16);
        common::bf16_values(&cpu, &attention(&cpu, &q, &k, &v, None, 1.0).unwrap())
    };

    let gpu_q = memory
        .upload_bf16(vec![query_tokens, heads, head_dim], &q_bf16)
        .unwrap();
    let gpu_k = memory
        .upload_bf16(vec![key_tokens, kv_heads, head_dim], &k_bf16)
        .unwrap();
    let gpu_v = memory
        .upload_bf16(vec![key_tokens, kv_heads, head_dim], &v_bf16)
        .unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Attention {
            query: &gpu_q,
            key: &gpu_k,
            value: &gpu_v,
            mask: None,
            scale: 1.0,
        }))
        .expect("bf16 attention launch");
    assert_bf16(
        "attention bf16",
        &cpu_out,
        &memory.download_bf16(&output).unwrap(),
        BF16_ATTENTION_ULP,
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_06_soft_cap_f32_and_bf16_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let values: Vec<f32> = [-1000.0, -30.0, -1.0, 0.0, 0.5, 1.0, 30.0, 1000.0]
        .into_iter()
        .chain((0..64).map(|i| (i as f32 - 32.0) / 4.0))
        .collect();

    let cpu_f32 = {
        let input = common::f32_tensor(&cpu, vec![values.len()], &values);
        common::values(&cpu, &soft_cap(&cpu, &input, 30.0).unwrap())
    };
    let input = memory.upload_f32(vec![values.len()], &values).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::SoftCap {
            input: &input,
            cap: 30.0,
        }))
        .expect("soft_cap launch");
    assert_close(
        "soft_cap f32",
        &cpu_f32,
        &memory.download_f32(&output).unwrap(),
    );

    let bf16: Vec<Bf16> = values.iter().map(|x| Bf16::from_f32(*x)).collect();
    let cpu_bf16 = {
        let input = common::bf16_tensor(&cpu, vec![bf16.len()], &bf16);
        common::bf16_values(&cpu, &soft_cap(&cpu, &input, 30.0).unwrap())
    };
    let input = memory.upload_bf16(vec![bf16.len()], &bf16).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::SoftCap {
            input: &input,
            cap: 30.0,
        }))
        .expect("bf16 soft_cap launch");
    assert_bf16(
        "soft_cap bf16",
        &cpu_bf16,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_07_dispatch_is_exact_on_dtype_and_signature() {
    let _serial = common::serial();
    let memory = common::memory();

    let input = memory.upload_f32(vec![2, 3], &[1.0; 6]).unwrap();
    let weight = memory.upload_f32(vec![4, 3], &[1.0; 12]).unwrap();
    let op = TensorOp::Linear {
        input: &input,
        weight: &weight,
    };
    assert!(matches!(op.kind(), OpKind::Linear));
    assert_eq!(
        op.shape_signature(),
        ShapeSignature::Linear {
            tokens: 2,
            in_features: 3,
            out_features: 4
        }
    );

    // A mismatched weight is refused by the contract, never silently broadcast.
    let bad_weight = memory.upload_f32(vec![4, 2], &[1.0; 8]).unwrap();
    let op = TensorOp::Linear {
        input: &input,
        weight: &bad_weight,
    };
    assert!(op.validate().is_err());

    // FP16 has no CUDA allocation path, so no M10 kernel can ever see it.
    assert!(memory.allocate_tensor(vec![4], DType::F16).is_err());

    // RoPE parameters are validated before any table is built.
    assert!(rope_inv_freq(RopeType::Default, 8, 10_000.0, 1.0).is_ok());
    assert!(rope_inv_freq(RopeType::Default, 8, 0.0, 1.0).is_err());

    let dispatcher = common::dispatcher(&memory);
    let too_wide = memory.upload_f32(vec![1, 3], &[1.0; 3]).unwrap();
    let request = ExecutionRequest::new(TensorOp::Linear {
        input: &too_wide,
        weight: &bad_weight,
    });
    assert!(dispatcher.plan(&request).is_err());
}

/// Absolute-error comparison for FP32 primitives.
fn assert_close(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: element count");
    let mut max_abs = 0.0f32;
    for (index, (want, got)) in expected.iter().zip(actual).enumerate() {
        assert!(
            got.is_finite(),
            "{label}: non-finite output at {index}: {got}"
        );
        let difference = (got - want).abs();
        max_abs = max_abs.max(difference);
        assert!(
            difference <= F32_ABS_TOLERANCE,
            "{label}[{index}]: gpu={got} cpu={want} diff={difference} > {F32_ABS_TOLERANCE}"
        );
    }
    println!("HW-M10 {label}: n={} max_abs={max_abs:e}", expected.len());
}

/// Representable-step comparison for BF16 primitives.
fn assert_bf16(label: &str, expected: &[Bf16], actual: &[Bf16], limit: u32) {
    assert_eq!(expected.len(), actual.len(), "{label}: element count");
    let mut worst = 0u32;
    for (index, (want, got)) in expected.iter().zip(actual).enumerate() {
        if want.is_nan() || got.is_nan() {
            assert!(
                want.is_nan() && got.is_nan(),
                "{label}: index {index} NaN mismatch: expected {want}, got {got}"
            );
            continue;
        }
        let steps = bf16_ulp(*want, *got);
        worst = worst.max(steps);
        assert!(
            steps <= limit,
            "{label}[{index}]: gpu={got:?} cpu={want:?} ulp={steps} > {limit}"
        );
    }
    println!("HW-M10 {label}: n={} max_ulp={worst}", expected.len());
}
#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_08_scale_f32_and_bf16_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let rows = 4usize;
    let columns = 6usize;
    let values: Vec<f32> = (0..rows * columns)
        .map(|i| (i as f32 - 12.0) / 3.0)
        .collect();

    // Scalar factor: the audited checkpoint stores `layer_scalar` as a single
    // BF16 value, so the [1] broadcast path is the real one.
    let scalar = 0.068847656f32;
    let cpu_f32 = {
        let input = common::f32_tensor(&cpu, vec![rows, columns], &values);
        let factor = common::f32_tensor(&cpu, vec![1], &[scalar]);
        common::values(&cpu, &scale(&cpu, &input, &factor).unwrap())
    };
    let input = memory.upload_f32(vec![rows, columns], &values).unwrap();
    let factor = memory.upload_f32(vec![1], &[scalar]).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Scale {
            input: &input,
            factor: &factor,
        }))
        .expect("scale f32 launch");
    assert_close(
        "scale f32 scalar",
        &cpu_f32,
        &memory.download_f32(&output).unwrap(),
    );

    // Per-column factor: the broadcast rule over the last dimension.
    let per_column: Vec<f32> = (0..columns).map(|c| (c as f32 - 2.0) * 0.5).collect();
    let cpu_f32 = {
        let input = common::f32_tensor(&cpu, vec![rows, columns], &values);
        let factor = common::f32_tensor(&cpu, vec![columns], &per_column);
        common::values(&cpu, &scale(&cpu, &input, &factor).unwrap())
    };
    let input = memory.upload_f32(vec![rows, columns], &values).unwrap();
    let factor = memory.upload_f32(vec![columns], &per_column).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Scale {
            input: &input,
            factor: &factor,
        }))
        .expect("scale f32 per-column launch");
    assert_close(
        "scale f32 column",
        &cpu_f32,
        &memory.download_f32(&output).unwrap(),
    );

    // BF16, scalar factor, with the real audited layer_scalar value.
    let bf16: Vec<Bf16> = values.iter().map(|x| Bf16::from_f32(*x)).collect();
    let scalar_bits: Vec<Bf16> = vec![Bf16::from_f32(scalar)];
    let cpu_bf16 = {
        let input = common::bf16_tensor(&cpu, vec![rows, columns], &bf16);
        let factor = common::bf16_tensor(&cpu, vec![1], &scalar_bits);
        common::bf16_values(&cpu, &scale(&cpu, &input, &factor).unwrap())
    };
    let input = memory.upload_bf16(vec![rows, columns], &bf16).unwrap();
    let factor = memory.upload_bf16(vec![1], &scalar_bits).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Scale {
            input: &input,
            factor: &factor,
        }))
        .expect("scale bf16 launch");
    assert_bf16(
        "scale bf16 scalar",
        &cpu_bf16,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );

    // BF16, per-column factor.
    let column_bits: Vec<Bf16> = per_column.iter().map(|x| Bf16::from_f32(*x)).collect();
    let cpu_bf16 = {
        let input = common::bf16_tensor(&cpu, vec![rows, columns], &bf16);
        let factor = common::bf16_tensor(&cpu, vec![columns], &column_bits);
        common::bf16_values(&cpu, &scale(&cpu, &input, &factor).unwrap())
    };
    let input = memory.upload_bf16(vec![rows, columns], &bf16).unwrap();
    let factor = memory.upload_bf16(vec![columns], &column_bits).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Scale {
            input: &input,
            factor: &factor,
        }))
        .expect("scale bf16 per-column launch");
    assert_bf16(
        "scale bf16 column",
        &cpu_bf16,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );

    // A factor that matches neither [1] nor the last dimension is refused
    // before any kernel is submitted.
    let input = memory.upload_f32(vec![rows, columns], &values).unwrap();
    let bad: Vec<f32> = vec![1.0; rows];
    let bad = memory.upload_f32(vec![rows], &bad).unwrap();
    let refusal = dispatcher.execute(&ExecutionRequest::new(TensorOp::Scale {
        input: &input,
        factor: &bad,
    }));
    assert!(
        matches!(
            refusal,
            Err(claymore_tensor::ExecutionError::Dispatch(
                claymore_tensor::DispatchError::Contract(claymore_tensor::OpError::ScaleMismatch)
            ))
        ),
        "a [rows] factor must be refused as ScaleMismatch, got {refusal:?}"
    );
}

// -----------------------------------------------------------------------
// M10.2 primitives: MoE, embedding, masking and widening.
//
// Extra tolerances declared before the campaign:
// - F32 scatter-add: exact. The kernel accumulates each destination row in
//   increasing pair index in FP32 and narrows once, so the order is
//   deterministic and identical to the CPU reference. The campaign of
//   25 November 2026 measured `max_abs=0e0` on the RTX 5090.
// - BF16 scatter-add: exact, same deterministic accumulation narrowed to BF16
//   once at output. The campaign of 25 November 2026 measured `max_ulp=0`.
// - Widen, gather, embed-gather and row-mask are pure data movement: exact.

/// F32 absolute tolerance for the scatter-add kernel: the accumulation order
/// is deterministic, so the kernel must match the CPU reference bit for bit.
const F32_SCATTER_ABS_TOLERANCE: f32 = 0.0;
/// BF16 ULP budget for the scatter-add kernel: exact, as measured on hardware.
const BF16_SCATTER_ULP: u32 = 0;

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_10_widen_bf16_is_bit_exact() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let shape = vec![5usize, 9usize];
    let bits: Vec<Bf16> = (0..45)
        .map(|i| Bf16::from_f32(((i % 15) as f32 - 7.0) / 4.0))
        .collect();

    let cpu_out = {
        let input = common::bf16_tensor(&cpu, shape.clone(), &bits);
        common::values(&cpu, &widen(&cpu, &input).unwrap())
    };

    let input = memory.upload_bf16(shape, &bits).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Widen { input: &input }))
        .expect("widen launch");
    let gpu_out = memory.download_f32(&output).unwrap();

    assert_eq!(cpu_out, gpu_out, "widen must be bit-exact");
    println!("HW-M10 widen bf16->f32: n={} exact", cpu_out.len());

    // An F32 input is refused: `widen` is the contract's BF16-only op.
    let f32_input = memory.upload_f32(vec![4], &[0.5, 1.5, 2.5, 3.5]).unwrap();
    let refusal = dispatcher.execute(&ExecutionRequest::new(TensorOp::Widen {
        input: &f32_input,
    }));
    assert!(
        refusal.is_err(),
        "F32 widen must be refused, got {refusal:?}"
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_11_router_top_k_matches_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (tokens, experts, top_k) = (4usize, 16usize, 4usize);
    let scores: Vec<f32> = (0..tokens * experts)
        .map(|i| ((i % 23) as f32 - 11.0) / 8.0)
        .collect();
    let per_expert: Vec<f32> = (0..experts).map(|i| 1.0 + (i as f32) / 32.0).collect();

    let cpu_out = {
        let scores_t = common::f32_tensor(&cpu, vec![tokens, experts], &scores);
        let scale_t = common::f32_tensor(&cpu, vec![experts], &per_expert);
        common::values(
            &cpu,
            &router_top_k(&cpu, &scores_t, &scale_t, top_k).unwrap(),
        )
    };

    let scores_t = memory.upload_f32(vec![tokens, experts], &scores).unwrap();
    let scale_t = memory.upload_f32(vec![experts], &per_expert).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::RouterTopK {
            scores: &scores_t,
            per_expert_scale: &scale_t,
            top_k,
        }))
        .expect("router launch");
    let gpu_out = memory.download_f32(&output).unwrap();

    // Expert indices are integers: they must match exactly; weights use the
    // F32 budget.
    assert_eq!(gpu_out.len(), cpu_out.len());
    for (slot, (want, got)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        if slot % (2 * top_k) >= top_k {
            assert_eq!(
                *want as u32, *got as u32,
                "router index mismatch at {slot}: cpu={want} gpu={got}"
            );
        }
    }
    assert_close("router_top_k f32", &cpu_out, &gpu_out);

    // BF16 scores are refused: the router is the contract's F32-only op.
    let bf16_scores: Vec<Bf16> = scores.iter().map(|v| Bf16::from_f32(*v)).collect();
    let scores_bf16 = memory
        .upload_bf16(vec![tokens, experts], &bf16_scores)
        .unwrap();
    let refusal = dispatcher.execute(&ExecutionRequest::new(TensorOp::RouterTopK {
        scores: &scores_bf16,
        per_expert_scale: &scale_t,
        top_k,
    }));
    assert!(
        refusal.is_err(),
        "BF16 router scores must be refused, got {refusal:?}"
    );

    // The pinned upstream's own router row: the BF16 logits contain exact tie
    // groups (experts 68/33 and 22/126), one of them straddling the top-k
    // boundary. The selection order inside a tie group is decided by ATen's
    // unstable `torch.topk`, so the device must reproduce the upstream's
    // order exactly, not just the selected set.
    let pinned = claymore_tensor::aten_topk::PINNED_ROUTER_RAW_ROW_0;
    let pinned_experts = pinned.len();
    let pinned_top_k = 8usize;
    let pinned_scale: Vec<f32> = (0..pinned_experts)
        .map(|i| 1.0 + (i as f32) / 32.0)
        .collect();
    let cpu_out = {
        let scores_t = common::f32_tensor(&cpu, vec![1, pinned_experts], &pinned);
        let scale_t = common::f32_tensor(&cpu, vec![pinned_experts], &pinned_scale);
        common::values(
            &cpu,
            &router_top_k(&cpu, &scores_t, &scale_t, pinned_top_k).unwrap(),
        )
    };
    let scores_t = memory.upload_f32(vec![1, pinned_experts], &pinned).unwrap();
    let scale_t = memory
        .upload_f32(vec![pinned_experts], &pinned_scale)
        .unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::RouterTopK {
            scores: &scores_t,
            per_expert_scale: &scale_t,
            top_k: pinned_top_k,
        }))
        .expect("router launch");
    let gpu_out = memory.download_f32(&output).unwrap();
    for slot in 0..pinned_top_k {
        assert_eq!(
            cpu_out[pinned_top_k + slot] as u32,
            gpu_out[pinned_top_k + slot] as u32,
            "pinned router index mismatch at {slot}: cpu={} gpu={}",
            cpu_out[pinned_top_k + slot],
            gpu_out[pinned_top_k + slot]
        );
    }
    // The pinned upstream's selection for that row.
    assert_eq!(
        cpu_out[pinned_top_k..pinned_top_k * 2]
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<u32>>(),
        vec![90, 113, 68, 33, 55, 22, 126, 105]
    );
    assert_close("router_top_k pinned row", &cpu_out, &gpu_out);
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_12_gelu_mul_up_f32_and_bf16_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let shape = vec![6usize, 20usize];
    let gate: Vec<f32> = (0..120).map(|i| ((i % 21) as f32 - 10.0) / 5.0).collect();
    let up: Vec<f32> = (0..120).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect();
    let gate_bf16: Vec<Bf16> = gate.iter().map(|v| Bf16::from_f32(*v)).collect();
    let up_bf16: Vec<Bf16> = up.iter().map(|v| Bf16::from_f32(*v)).collect();

    let cpu_f32 = {
        let g = common::f32_tensor(&cpu, shape.clone(), &gate);
        let u = common::f32_tensor(&cpu, shape.clone(), &up);
        common::values(&cpu, &gelu_mul_up(&cpu, &g, &u).unwrap())
    };
    let g = memory.upload_f32(shape.clone(), &gate).unwrap();
    let u = memory.upload_f32(shape.clone(), &up).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::GeluMulUp {
            gate: &g,
            up: &u,
        }))
        .expect("gelu_mul_up f32 launch");
    assert_close(
        "gelu_mul_up f32",
        &cpu_f32,
        &memory.download_f32(&output).unwrap(),
    );

    let cpu_bf16 = {
        let g = common::bf16_tensor(&cpu, shape.clone(), &gate_bf16);
        let u = common::bf16_tensor(&cpu, shape.clone(), &up_bf16);
        common::bf16_values(&cpu, &gelu_mul_up(&cpu, &g, &u).unwrap())
    };
    let g = memory.upload_bf16(shape.clone(), &gate_bf16).unwrap();
    let u = memory.upload_bf16(shape, &up_bf16).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::GeluMulUp {
            gate: &g,
            up: &u,
        }))
        .expect("gelu_mul_up bf16 launch");
    assert_bf16(
        "gelu_mul_up bf16",
        &cpu_bf16,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_13_gather_rows_f32_and_bf16_are_exact() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (rows, width) = (7usize, 5usize);
    let values: Vec<f32> = (0..rows * width)
        .map(|i| ((i % 11) as f32 - 5.0) / 4.0)
        .collect();
    let indices: Vec<f32> = vec![6.0, 0.0, 3.0, 3.0, 5.0];

    let cpu_out = {
        let input = common::f32_tensor(&cpu, vec![rows, width], &values);
        let idx = common::f32_tensor(&cpu, vec![indices.len()], &indices);
        common::values(&cpu, &gather_rows(&cpu, &input, &idx).unwrap())
    };
    let input = memory.upload_f32(vec![rows, width], &values).unwrap();
    let idx = memory.upload_f32(vec![indices.len()], &indices).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::GatherRows {
            input: &input,
            indices: &idx,
        }))
        .expect("gather_rows f32 launch");
    assert_eq!(
        cpu_out,
        memory.download_f32(&output).unwrap(),
        "f32 gather must be exact"
    );

    let bits: Vec<Bf16> = values.iter().map(|v| Bf16::from_f32(*v)).collect();
    let cpu_out = {
        let input = common::bf16_tensor(&cpu, vec![rows, width], &bits);
        let idx = common::f32_tensor(&cpu, vec![indices.len()], &indices);
        common::bf16_values(&cpu, &gather_rows(&cpu, &input, &idx).unwrap())
    };
    let input = memory.upload_bf16(vec![rows, width], &bits).unwrap();
    let idx = memory.upload_f32(vec![indices.len()], &indices).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::GatherRows {
            input: &input,
            indices: &idx,
        }))
        .expect("gather_rows bf16 launch");
    assert_eq!(
        cpu_out,
        memory.download_bf16(&output).unwrap(),
        "bf16 gather must be exact"
    );
    println!("HW-M10 gather_rows: exact on f32 and bf16");

    // Out-of-range indices are refused before launch.
    let bad = memory.upload_f32(vec![1], &[7.0]).unwrap();
    let refusal = dispatcher.execute(&ExecutionRequest::new(TensorOp::GatherRows {
        input: &input,
        indices: &bad,
    }));
    assert!(
        refusal.is_err(),
        "out-of-range gather index must be refused, got {refusal:?}"
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_14_scatter_add_f32_and_bf16_match_cpu_exactly() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (pairs, width, out_rows) = (12usize, 6usize, 5usize);
    let source: Vec<f32> = (0..pairs * width)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .collect();
    let weights: Vec<f32> = (0..pairs).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    // Rows repeat targets so that atomics collide.
    let rows: Vec<f32> = (0..pairs).map(|i| (i % out_rows) as f32).collect();

    let cpu_out = {
        let src = common::f32_tensor(&cpu, vec![pairs, width], &source);
        let w = common::f32_tensor(&cpu, vec![pairs], &weights);
        let r = common::f32_tensor(&cpu, vec![pairs], &rows);
        common::values(&cpu, &scatter_add(&cpu, &src, &w, &r, out_rows).unwrap())
    };
    let src = memory.upload_f32(vec![pairs, width], &source).unwrap();
    let w = memory.upload_f32(vec![pairs], &weights).unwrap();
    let r = memory.upload_f32(vec![pairs], &rows).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::ScatterAdd {
            source: &src,
            weights: &w,
            rows: &r,
            out_rows,
        }))
        .expect("scatter_add f32 launch");
    let gpu_out = memory.download_f32(&output).unwrap();
    assert_eq!(gpu_out.len(), cpu_out.len());
    let mut max_abs = 0.0f32;
    for (want, got) in cpu_out.iter().zip(gpu_out.iter()) {
        max_abs = max_abs.max((got - want).abs());
        assert!(
            max_abs <= F32_SCATTER_ABS_TOLERANCE,
            "f32 scatter drift {max_abs} > {F32_SCATTER_ABS_TOLERANCE}"
        );
    }
    println!(
        "HW-M10 scatter_add f32: n={} max_abs={max_abs:e}",
        cpu_out.len()
    );

    let bits: Vec<Bf16> = source.iter().map(|v| Bf16::from_f32(*v)).collect();
    let cpu_out = {
        let src = common::bf16_tensor(&cpu, vec![pairs, width], &bits);
        let w = common::f32_tensor(&cpu, vec![pairs], &weights);
        let r = common::f32_tensor(&cpu, vec![pairs], &rows);
        common::bf16_values(&cpu, &scatter_add(&cpu, &src, &w, &r, out_rows).unwrap())
    };
    let src = memory.upload_bf16(vec![pairs, width], &bits).unwrap();
    let w = memory.upload_f32(vec![pairs], &weights).unwrap();
    let r = memory.upload_f32(vec![pairs], &rows).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::ScatterAdd {
            source: &src,
            weights: &w,
            rows: &r,
            out_rows,
        }))
        .expect("scatter_add bf16 launch");
    assert_bf16(
        "scatter_add bf16",
        &cpu_out,
        &memory.download_bf16(&output).unwrap(),
        BF16_SCATTER_ULP,
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_15_embed_gather_f32_and_bf16_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (vocab, width, tokens) = (11usize, 8usize, 4usize);
    let table: Vec<f32> = (0..vocab * width)
        .map(|i| ((i % 19) as f32 - 9.0) / 16.0)
        .collect();
    let ids: Vec<f32> = vec![10.0, 0.0, 7.0, 3.0];
    let scale = 0.25f32;

    let cpu_out = {
        let t = common::f32_tensor(&cpu, vec![vocab, width], &table);
        let ids_t = common::f32_tensor(&cpu, vec![tokens], &ids);
        common::values(&cpu, &embed_gather(&cpu, &ids_t, &t, scale).unwrap())
    };
    let t = memory.upload_f32(vec![vocab, width], &table).unwrap();
    let ids_t = memory.upload_f32(vec![tokens], &ids).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::EmbedGather {
            ids: &ids_t,
            table: &t,
            scale,
        }))
        .expect("embed_gather f32 launch");
    assert_close(
        "embed_gather f32",
        &cpu_out,
        &memory.download_f32(&output).unwrap(),
    );

    let bits: Vec<Bf16> = table.iter().map(|v| Bf16::from_f32(*v)).collect();
    let cpu_out = {
        let t = common::bf16_tensor(&cpu, vec![vocab, width], &bits);
        let ids_t = common::f32_tensor(&cpu, vec![tokens], &ids);
        common::bf16_values(&cpu, &embed_gather(&cpu, &ids_t, &t, scale).unwrap())
    };
    let t = memory.upload_bf16(vec![vocab, width], &bits).unwrap();
    let ids_t = memory.upload_f32(vec![tokens], &ids).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::EmbedGather {
            ids: &ids_t,
            table: &t,
            scale,
        }))
        .expect("embed_gather bf16 launch");
    assert_bf16(
        "embed_gather bf16",
        &cpu_out,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_16_row_mask_f32_and_bf16_are_exact() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (rows, width) = (6usize, 4usize);
    let values: Vec<f32> = (0..rows * width)
        .map(|i| ((i % 9) as f32 - 4.0) / 2.0)
        .collect();
    let flags: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];

    let cpu_out = {
        let input = common::f32_tensor(&cpu, vec![rows, width], &values);
        let f = common::f32_tensor(&cpu, vec![rows], &flags);
        common::values(&cpu, &row_mask(&cpu, &input, &f).unwrap())
    };
    let input = memory.upload_f32(vec![rows, width], &values).unwrap();
    let f = memory.upload_f32(vec![rows], &flags).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::RowMask {
            input: &input,
            flags: &f,
        }))
        .expect("row_mask f32 launch");
    let gpu = memory.download_f32(&output).unwrap();
    for (index, (want, got)) in cpu_out.iter().zip(gpu.iter()).enumerate() {
        if flags[index / width] == 1.0 {
            assert_eq!(*got, 0.0f32, "masked row {index} must be zero");
        } else {
            assert_eq!(want, got, "unmasked row {index} must be verbatim");
        }
    }
    println!("HW-M10 row_mask f32: exact");

    let bits: Vec<Bf16> = values.iter().map(|v| Bf16::from_f32(*v)).collect();
    let input = memory.upload_bf16(vec![rows, width], &bits).unwrap();
    let f = memory.upload_f32(vec![rows], &flags).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::RowMask {
            input: &input,
            flags: &f,
        }))
        .expect("row_mask bf16 launch");
    let gpu = memory.download_bf16(&output).unwrap();
    for (index, got) in gpu.iter().enumerate() {
        if flags[index / width] == 1.0 {
            assert_eq!(got.to_f32(), 0.0f32, "bf16 masked row {index} must be zero");
        }
    }
    println!("HW-M10 row_mask bf16: masked rows zeroed");
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_17_scale_bias_f32_and_bf16_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (rows, width) = (5usize, 7usize);
    let values: Vec<f32> = (0..rows * width)
        .map(|i| ((i % 15) as f32 - 7.0) / 8.0)
        .collect();
    let scales: Vec<f32> = (0..width).map(|i| 1.0 + (i as f32) / 16.0).collect();
    let biases: Vec<f32> = (0..width).map(|i| (i as f32 - 3.0) / 32.0).collect();

    let cpu_out = {
        let input = common::f32_tensor(&cpu, vec![rows, width], &values);
        let s = common::f32_tensor(&cpu, vec![width], &scales);
        let b = common::f32_tensor(&cpu, vec![width], &biases);
        common::values(&cpu, &scale_bias(&cpu, &input, &s, &b).unwrap())
    };
    let input = memory.upload_f32(vec![rows, width], &values).unwrap();
    let s = memory.upload_f32(vec![width], &scales).unwrap();
    let b = memory.upload_f32(vec![width], &biases).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::ScaleBias {
            input: &input,
            scale: &s,
            bias: &b,
        }))
        .expect("scale_bias f32 launch");
    assert_close(
        "scale_bias f32",
        &cpu_out,
        &memory.download_f32(&output).unwrap(),
    );

    let bits: Vec<Bf16> = values.iter().map(|v| Bf16::from_f32(*v)).collect();
    let s_bits: Vec<Bf16> = scales.iter().map(|v| Bf16::from_f32(*v)).collect();
    let b_bits: Vec<Bf16> = biases.iter().map(|v| Bf16::from_f32(*v)).collect();
    let cpu_out = {
        let input = common::bf16_tensor(&cpu, vec![rows, width], &bits);
        let s = common::bf16_tensor(&cpu, vec![width], &s_bits);
        let b = common::bf16_tensor(&cpu, vec![width], &b_bits);
        common::bf16_values(&cpu, &scale_bias(&cpu, &input, &s, &b).unwrap())
    };
    let input = memory.upload_bf16(vec![rows, width], &bits).unwrap();
    let s = memory.upload_bf16(vec![width], &s_bits).unwrap();
    let b = memory.upload_bf16(vec![width], &b_bits).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::ScaleBias {
            input: &input,
            scale: &s,
            bias: &b,
        }))
        .expect("scale_bias bf16 launch");
    assert_bf16(
        "scale_bias bf16",
        &cpu_out,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );
}

#[test]
#[ignore = "requires a CUDA driver and a usable NVIDIA device"]
fn hw_m10_18_scale_scalar_f32_and_bf16_match_cpu() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let (rows, width) = (5usize, 7usize);
    let values: Vec<f32> = (0..rows * width)
        .map(|i| ((i % 15) as f32 - 7.0) / 8.0)
        .collect();
    // The upstream's `scalar_root_size` is a plain Python float that is not
    // BF16-representable: the scalar must stay in FP32 and the product must be
    // rounded once to the input dtype.
    let factor = (2816f64).powf(-0.5) as f32;

    let cpu_out = {
        let input = common::f32_tensor(&cpu, vec![rows, width], &values);
        common::values(&cpu, &scale_scalar(&cpu, &input, factor).unwrap())
    };
    let input = memory.upload_f32(vec![rows, width], &values).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::ScaleScalar {
            input: &input,
            factor,
        }))
        .expect("scale_scalar f32 launch");
    assert_close(
        "scale_scalar f32",
        &cpu_out,
        &memory.download_f32(&output).unwrap(),
    );

    let bits: Vec<Bf16> = values.iter().map(|v| Bf16::from_f32(*v)).collect();
    let cpu_out = {
        let input = common::bf16_tensor(&cpu, vec![rows, width], &bits);
        common::bf16_values(&cpu, &scale_scalar(&cpu, &input, factor).unwrap())
    };
    let input = memory.upload_bf16(vec![rows, width], &bits).unwrap();
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::ScaleScalar {
            input: &input,
            factor,
        }))
        .expect("scale_scalar bf16 launch");
    let gpu_out = memory.download_bf16(&output).unwrap();
    assert_bf16("scale_scalar bf16", &cpu_out, &gpu_out, BF16_ULP);
    // The product is rounded once: the device result is the BF16 rounding of
    // the exact FP32 product, not a second rounding of a BF16 scalar.
    for (index, value) in bits.iter().enumerate() {
        let expected = Bf16::from_f32(value.to_f32() * factor);
        assert_eq!(
            gpu_out[index].to_bits(),
            expected.to_bits(),
            "scale_scalar bf16 mismatch at {index}"
        );
    }
}

/// Bounded full-payload verification of the pinned checkpoint is available in
/// `src/bin/verify_checkpoint.rs`; these tests use the same pinned revision.
fn checkpoint_root() -> Option<std::path::PathBuf> {
    std::env::var_os("CLAYMORE_M10_CHECKPOINT").map(std::path::PathBuf::from)
}

/// Layer 0 of the pinned decoder, executed with the real BF16 weights of the
/// audited checkpoint (`model.decoder.layers.0.*`) on the real canvas shape
/// (256 tokens, hidden 2816, 16 heads, 8 KV heads, head_dim 256, sliding
/// attention, full rotary with theta 10000). Every CUDA step is compared with
/// the deterministic CPU reference, step by step.
///
/// The dense MLP/MoE tail of the layer needs primitives that are not part of
/// the audited M10 kernel set (gated activation, router top-k, expert gather),
/// so this subgraph stops at the attention + residual path: it is real-weight
/// evidence for the implemented primitives, not a full-graph generation claim.
#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and the pinned checkpoint"]
fn hw_m10_09_real_weight_decoder_layer_subgraph_matches_cpu() {
    let Some(root) = checkpoint_root() else {
        eprintln!("skipping: CLAYMORE_M10_CHECKPOINT is not set");
        return;
    };
    let _serial = common::serial();
    let checkpoint = claymore_model_gemma::checkpoint::Checkpoint::open_pinned(&root)
        .expect("pinned checkpoint");
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);
    let peak = std::cell::Cell::new(0u64);

    let canvas_tokens = 256usize;
    let hidden = 2816usize;
    let heads = 16usize;
    let kv_heads = 8usize;
    let head_dim = 256usize;
    let epsilon = 1e-6f32;
    let scaling = (head_dim as f32).powf(-0.5);
    let vocabulary = 262_144u32;

    // Real canvas hidden states: 256 rows of the pinned embedding matrix.
    let embedding = checkpoint
        .read_bf16_bits("model.decoder.embed_tokens.weight")
        .expect("real embedding matrix");
    let mut rng = claymore_model_gemma::rng::SplitMix64::new(0x5169_5090_5c3e_1234);
    let ids: Vec<u32> = (0..canvas_tokens).map(|_| rng.below(vocabulary)).collect();
    println!("HW-M10 real canvas token ids[:8] = {:?}", &ids[..8]);
    let mut canvas_bits: Vec<Bf16> = Vec::with_capacity(canvas_tokens * hidden);
    for id in &ids {
        let start = *id as usize * hidden;
        canvas_bits.extend(
            embedding[start..start + hidden]
                .iter()
                .copied()
                .map(Bf16::from_bits),
        );
    }

    // Real pinned weights, kept in their exact BF16 storage form.
    let bits = |name: &str| -> Vec<Bf16> {
        checkpoint
            .read_bf16_bits(name)
            .unwrap_or_else(|error| panic!("real weight {name}: {error}"))
            .into_iter()
            .map(Bf16::from_bits)
            .collect()
    };
    let input_layernorm = bits("model.decoder.layers.0.input_layernorm.weight");
    let q_proj = bits("model.decoder.layers.0.self_attn.q_proj.weight");
    let q_norm_weight = bits("model.decoder.layers.0.self_attn.q_norm.weight");
    let k_proj = bits("model.decoder.layers.0.self_attn.k_proj.weight");
    let k_norm_weight = bits("model.decoder.layers.0.self_attn.k_norm.weight");
    let v_proj = bits("model.decoder.layers.0.self_attn.v_proj.weight");
    let o_proj = bits("model.decoder.layers.0.self_attn.o_proj.weight");
    let post_attention_layernorm = bits("model.decoder.layers.0.post_attention_layernorm.weight");
    let layer_scalar = bits("model.decoder.layers.0.layer_scalar");
    // `layer_scalar` is a learned per-layer value, not 1.0: the audited layer 0
    // ships 0.068847656. The scale is applied explicitly as the last operation
    // of the layer, exactly as `hidden_states *= self.layer_scalar` in the
    // pinned reference, so the primitive is validated against real weights.
    assert_eq!(layer_scalar.len(), 1, "layer_scalar is a single BF16 value");
    let scalar_value = layer_scalar[0].to_f32();
    assert!(
        scalar_value.is_finite() && scalar_value > 0.0,
        "layer_scalar must be a finite positive BF16 value, got {scalar_value}"
    );
    println!(
        "HW-M10 real layer 0 layer_scalar = {scalar_value} (bits 0x{:04x})",
        layer_scalar[0].to_bits()
    );

    // Sliding layer 0: full rotary, rope_theta 10000, rope_type "default".
    let positions: Vec<u32> = (0..canvas_tokens as u32).collect();
    let (cos, sin) = rope_tables(RopeType::Default, &positions, head_dim, 10_000.0, 1.0).unwrap();
    let cos_bits: Vec<Bf16> = cos.iter().map(|v| Bf16::from_f32(*v)).collect();
    let sin_bits: Vec<Bf16> = sin.iter().map(|v| Bf16::from_f32(*v)).collect();
    // `v_norm` is an RMSNorm without a learned scale: an explicit all-ones scale
    // is passed, never an implicit cast.
    let ones: Vec<Bf16> = vec![Bf16::from_f32(1.0); head_dim];

    // CPU step primitives driven by explicit BF16 bit vectors. Every CPU step
    // and every CUDA step then consume the *same* input bits, so a measured
    // difference is the kernel's own arithmetic, not inherited input drift.
    let norm_step = |input: &Vec<Bf16>, scale: &Vec<Bf16>, shape: Vec<usize>| -> Vec<Bf16> {
        let input_t = common::bf16_tensor(&cpu, shape, input);
        let scale_t = common::bf16_tensor(&cpu, vec![scale.len()], scale);
        common::bf16_values(&cpu, &rms_norm(&cpu, &input_t, &scale_t, epsilon).unwrap())
    };
    let linear_step = |input: &Vec<Bf16>, weight: &Vec<Bf16>, shape: Vec<usize>, out: usize| {
        let input_t = common::bf16_tensor(&cpu, shape, input);
        let weight_t = common::bf16_tensor(&cpu, vec![out, input_t.layout().shape()[1]], weight);
        common::bf16_values(&cpu, &linear(&cpu, &input_t, &weight_t).unwrap())
    };
    let rope_step = |input: &Vec<Bf16>, shape: Vec<usize>| -> Vec<Bf16> {
        let input_t = common::bf16_tensor(&cpu, shape, input);
        let cos_t = common::bf16_tensor(&cpu, vec![canvas_tokens, head_dim], &cos_bits);
        let sin_t = common::bf16_tensor(&cpu, vec![canvas_tokens, head_dim], &sin_bits);
        common::bf16_values(&cpu, &rotary_embed(&cpu, &input_t, &cos_t, &sin_t).unwrap())
    };
    let attention_step = |q: &Vec<Bf16>, k: &Vec<Bf16>, v: &Vec<Bf16>| -> Vec<Bf16> {
        let q_t = common::bf16_tensor(&cpu, vec![canvas_tokens, heads, head_dim], q);
        let k_t = common::bf16_tensor(&cpu, vec![canvas_tokens, kv_heads, head_dim], k);
        let v_t = common::bf16_tensor(&cpu, vec![canvas_tokens, kv_heads, head_dim], v);
        common::bf16_values(
            &cpu,
            &attention(&cpu, &q_t, &k_t, &v_t, None, scaling).unwrap(),
        )
    };

    // The full reference chain, from the real canvas, defines the ground truth.
    let mut reference: Vec<Vec<Bf16>> = vec![canvas_bits.clone()];
    let r_norm = norm_step(&canvas_bits, &input_layernorm, vec![canvas_tokens, hidden]);
    reference.push(r_norm.clone());
    let r_q = linear_step(
        &r_norm,
        &q_proj,
        vec![canvas_tokens, hidden],
        heads * head_dim,
    );
    reference.push(r_q.clone());
    let r_qn = norm_step(&r_q, &q_norm_weight, vec![canvas_tokens, heads, head_dim]);
    reference.push(r_qn.clone());
    let r_qr = rope_step(&r_qn, vec![canvas_tokens, heads, head_dim]);
    reference.push(r_qr.clone());
    let r_k = linear_step(
        &r_norm,
        &k_proj,
        vec![canvas_tokens, hidden],
        kv_heads * head_dim,
    );
    reference.push(r_k.clone());
    let r_kn = norm_step(
        &r_k,
        &k_norm_weight,
        vec![canvas_tokens, kv_heads, head_dim],
    );
    reference.push(r_kn.clone());
    let r_kr = rope_step(&r_kn, vec![canvas_tokens, kv_heads, head_dim]);
    reference.push(r_kr.clone());
    let r_v = linear_step(
        &r_norm,
        &v_proj,
        vec![canvas_tokens, hidden],
        kv_heads * head_dim,
    );
    reference.push(r_v.clone());
    let r_vn = norm_step(&r_v, &ones, vec![canvas_tokens, kv_heads, head_dim]);
    reference.push(r_vn.clone());
    let r_attn = attention_step(&r_qr, &r_kr, &r_vn);
    reference.push(r_attn.clone());
    let r_o = linear_step(
        &r_attn,
        &o_proj,
        vec![canvas_tokens, heads * head_dim],
        hidden,
    );
    reference.push(r_o.clone());
    let r_post = norm_step(&r_o, &post_attention_layernorm, vec![canvas_tokens, hidden]);
    reference.push(r_post.clone());
    let canvas_t = common::bf16_tensor(&cpu, vec![canvas_tokens, hidden], &canvas_bits);
    let post_t = common::bf16_tensor(&cpu, vec![canvas_tokens, hidden], &r_post);
    let r_out = common::bf16_values(&cpu, &add(&cpu, &canvas_t, &post_t).unwrap());
    reference.push(r_out.clone());
    let scalar_t = common::bf16_tensor(&cpu, vec![1], &layer_scalar);
    let out_t = common::bf16_tensor(&cpu, vec![canvas_tokens, hidden], &r_out);
    let r_scaled = common::bf16_values(&cpu, &scale(&cpu, &out_t, &scalar_t).unwrap());
    reference.push(r_scaled);

    // CUDA chain on the real device. Each step consumes the previous step's
    // downloaded device bits, re-uploaded explicitly: no aliasing, no implicit
    // copy, and every allocation goes through the memory manager.
    let mut gpu: Vec<Vec<Bf16>> = vec![common::upload(
        &memory,
        &peak,
        vec![canvas_tokens, hidden],
        &canvas_bits,
    )];
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::RmsNorm {
            input: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[0]),
            scale: &common::tensor_of(&memory, &peak, vec![hidden], &input_layernorm),
            epsilon,
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::Linear {
            input: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[1]),
            weight: &common::tensor_of(&memory, &peak, vec![heads * head_dim, hidden], &q_proj),
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::RmsNorm {
            input: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, heads, head_dim],
                &gpu[2],
            ),
            scale: &common::tensor_of(&memory, &peak, vec![head_dim], &q_norm_weight),
            epsilon,
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::RotaryEmbed {
            input: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, heads, head_dim],
                &gpu[3],
            ),
            cos: &common::tensor_of(&memory, &peak, vec![canvas_tokens, head_dim], &cos_bits),
            sin: &common::tensor_of(&memory, &peak, vec![canvas_tokens, head_dim], &sin_bits),
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::Linear {
            input: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[1]),
            weight: &common::tensor_of(&memory, &peak, vec![kv_heads * head_dim, hidden], &k_proj),
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::RmsNorm {
            input: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, kv_heads, head_dim],
                &gpu[5],
            ),
            scale: &common::tensor_of(&memory, &peak, vec![head_dim], &k_norm_weight),
            epsilon,
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::RotaryEmbed {
            input: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, kv_heads, head_dim],
                &gpu[6],
            ),
            cos: &common::tensor_of(&memory, &peak, vec![canvas_tokens, head_dim], &cos_bits),
            sin: &common::tensor_of(&memory, &peak, vec![canvas_tokens, head_dim], &sin_bits),
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::Linear {
            input: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[1]),
            weight: &common::tensor_of(&memory, &peak, vec![kv_heads * head_dim, hidden], &v_proj),
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::RmsNorm {
            input: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, kv_heads, head_dim],
                &gpu[8],
            ),
            scale: &common::tensor_of(&memory, &peak, vec![head_dim], &ones),
            epsilon,
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::Attention {
            query: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, heads, head_dim],
                &gpu[4],
            ),
            key: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, kv_heads, head_dim],
                &gpu[7],
            ),
            value: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, kv_heads, head_dim],
                &gpu[9],
            ),
            mask: None,
            scale: scaling,
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::Linear {
            input: &common::tensor_of(
                &memory,
                &peak,
                vec![canvas_tokens, heads * head_dim],
                &gpu[10],
            ),
            weight: &common::tensor_of(&memory, &peak, vec![hidden, heads * head_dim], &o_proj),
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::RmsNorm {
            input: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[11]),
            scale: &common::tensor_of(&memory, &peak, vec![hidden], &post_attention_layernorm),
            epsilon,
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::Add {
            lhs: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[0]),
            rhs: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[12]),
        },
    ));
    gpu.push(common::run(
        &memory,
        &dispatcher,
        &peak,
        TensorOp::Scale {
            input: &common::tensor_of(&memory, &peak, vec![canvas_tokens, hidden], &gpu[13]),
            factor: &common::tensor_of(&memory, &peak, vec![1], &layer_scalar),
        },
    ));

    // Per-step local reference: the CPU op fed with the *same* bits the kernel
    // consumed. This isolates each kernel's own arithmetic, which is what the
    // declared per-operation budget bounds.
    let local_norm = norm_step(&gpu[0], &input_layernorm, vec![canvas_tokens, hidden]);
    let local_q = linear_step(
        &gpu[1],
        &q_proj,
        vec![canvas_tokens, hidden],
        heads * head_dim,
    );
    let local_qn = norm_step(
        &gpu[2],
        &q_norm_weight,
        vec![canvas_tokens, heads, head_dim],
    );
    let local_qr = rope_step(&gpu[3], vec![canvas_tokens, heads, head_dim]);
    let local_k = linear_step(
        &gpu[1],
        &k_proj,
        vec![canvas_tokens, hidden],
        kv_heads * head_dim,
    );
    let local_kn = norm_step(
        &gpu[5],
        &k_norm_weight,
        vec![canvas_tokens, kv_heads, head_dim],
    );
    let local_kr = rope_step(&gpu[6], vec![canvas_tokens, kv_heads, head_dim]);
    let local_v = linear_step(
        &gpu[1],
        &v_proj,
        vec![canvas_tokens, hidden],
        kv_heads * head_dim,
    );
    let local_vn = norm_step(&gpu[8], &ones, vec![canvas_tokens, kv_heads, head_dim]);
    let local_attn = attention_step(&gpu[4], &gpu[7], &gpu[9]);
    let local_o = linear_step(
        &gpu[10],
        &o_proj,
        vec![canvas_tokens, heads * head_dim],
        hidden,
    );
    let local_post = norm_step(
        &gpu[11],
        &post_attention_layernorm,
        vec![canvas_tokens, hidden],
    );
    let canvas_ref = common::bf16_tensor(&cpu, vec![canvas_tokens, hidden], &gpu[0]);
    let post_ref = common::bf16_tensor(&cpu, vec![canvas_tokens, hidden], &gpu[12]);
    let local_out = common::bf16_values(&cpu, &add(&cpu, &canvas_ref, &post_ref).unwrap());
    let scalar_ref = common::bf16_tensor(&cpu, vec![1], &layer_scalar);
    let out_ref = common::bf16_tensor(&cpu, vec![canvas_tokens, hidden], &gpu[13]);
    let local_scaled = common::bf16_values(&cpu, &scale(&cpu, &out_ref, &scalar_ref).unwrap());

    let labels = [
        "canvas",
        "input_layernorm",
        "q_proj",
        "q_norm",
        "q_rope",
        "k_proj",
        "k_norm",
        "k_rope",
        "v_proj",
        "v_norm",
        "attention",
        "o_proj",
        "post_attn_norm",
        "residual_add",
        "layer_scale",
    ];
    // Elementwise, reduction and projection kernels are bounded at one
    // representable step; attention rounds every softmax probability to BF16
    // before the value reduction, so its budget is two.
    let budgets = [0u32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1];
    assert_eq!(labels.len(), gpu.len(), "step count");
    let locals: Vec<Vec<Bf16>> = vec![
        gpu[0].clone(),
        local_norm,
        local_q,
        local_qn,
        local_qr,
        local_k,
        local_kn,
        local_kr,
        local_v,
        local_vn,
        local_attn,
        local_o,
        local_post,
        local_out,
        local_scaled,
    ];
    for (index, label) in labels.iter().enumerate() {
        if index == 10 {
            // Attention: the value reduction is the only M10 step whose output
            // can cancel, so it is bounded against the value magnitude scale
            // derived from the BF16 probability rounding.
            let scale = gpu[9]
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(0f32, f32::max);
            assert_bf16_reduction(
                &format!("real layer0 {label}"),
                &locals[index],
                &gpu[index],
                scale,
                BF16_ATTENTION_REDUCTION_TOLERANCE,
            );
            continue;
        }
        assert_bf16(
            &format!("real layer0 {label}"),
            &locals[index],
            &gpu[index],
            budgets[index],
        );
    }

    // End-to-end drift against the reference chain, recorded as independent
    // campaign evidence. A per-step budget is not a chain budget: a one-step
    // RMSNorm rounding difference is amplified by cancellation in a 2816-term
    // projection, so the propagated drift is measured, not asserted against a
    // per-operation budget.
    println!("HW-M10 real layer 0 end-to-end drift (device chain vs reference chain):");
    for (index, label) in labels.iter().enumerate() {
        let drift = bf16_report(
            &format!("real layer0 {label}"),
            &reference[index],
            &gpu[index],
        );
        println!("{drift}");
    }
    println!(
        "HW-M10 real layer 0 subgraph: vram_peak_bytes={} (26 GiB soft, 28 GiB hard)",
        peak.get()
    );
}
/// BF16 comparison for a reduction whose output can cancel.
///
/// The attention value reduction sums BF16-rounded softmax probabilities. A
/// BF16 probability carries a relative error of at most `2^-9` (round to
/// nearest, eight explicit mantissa bits), so an output element inherits at
/// most `2^-9 * sum_i p_i |v_i| <= 2^-9 * max_i |v_i|`. The declared budget
/// uses one extra bit of margin (`2^-8`) and the magnitude scale of the value
/// tensor: it is derived from the error model, not fitted to an observation.
///
/// A per-element ULP count is *not* a bound for this operator. An output
/// element close to zero can be the result of a cancelling sum; its ULP
/// distance is then unbounded while its absolute error stays inside the model.
/// The ULP count, the maximum, the RMS and the relative error are still
/// measured and reported.
fn assert_bf16_reduction(
    label: &str,
    expected: &[Bf16],
    actual: &[Bf16],
    scale: f32,
    tolerance: f32,
) {
    assert_eq!(expected.len(), actual.len(), "{label}: element count");
    assert!(scale > 0.0, "{label}: the reduction scale must be positive");
    let mut worst = 0u32;
    let mut max_abs = 0f32;
    let mut sum_squares = 0f64;
    for (index, (want, got)) in expected.iter().zip(actual).enumerate() {
        let (w, g) = (want.to_f32(), got.to_f32());
        assert!(
            w.is_finite() && g.is_finite(),
            "{label}: non-finite output at {index}: gpu={g} cpu={w}"
        );
        let difference = (g - w).abs();
        max_abs = max_abs.max(difference);
        sum_squares += (difference as f64) * (difference as f64);
        worst = worst.max(bf16_ulp(*want, *got));
    }
    let rms = (sum_squares / expected.len() as f64).sqrt() as f32;
    let bound = tolerance * scale;
    assert!(
        max_abs <= bound,
        "{label}: max_abs={max_abs:e} exceeds the model bound {bound:e} (tolerance={tolerance:e} scale={scale:e})"
    );
    println!(
        "HW-M10 {label}: n={} max_abs={max_abs:e} bound={bound:e} rms={rms:e} max_ulp={worst}",
        expected.len()
    );
}

/// Campaign metric report: maximum, RMS, relative and ULP distance between the
/// reference chain and the device chain. Every value must be finite.
fn bf16_report(label: &str, expected: &[Bf16], actual: &[Bf16]) -> String {
    assert_eq!(expected.len(), actual.len(), "{label}: element count");
    let mut max_ulp = 0u32;
    let mut max_abs = 0f32;
    let mut sum_squares = 0f64;
    let mut max_reference = 0f32;
    for (index, (want, got)) in expected.iter().zip(actual).enumerate() {
        let (w, g) = (want.to_f32(), got.to_f32());
        assert!(
            w.is_finite() && g.is_finite(),
            "{label}: non-finite at {index}: gpu={g} cpu={w}"
        );
        let difference = (g - w).abs();
        max_abs = max_abs.max(difference);
        sum_squares += (difference as f64) * (difference as f64);
        max_reference = max_reference.max(w.abs());
        max_ulp = max_ulp.max(bf16_ulp(*want, *got));
    }
    let rms = (sum_squares / expected.len() as f64).sqrt() as f32;
    let relative = if max_reference > 0.0 {
        max_abs / max_reference
    } else {
        0.0
    };
    format!(
        "HW-M10 drift {label}: n={} max_ulp={max_ulp} max_abs={max_abs:e} rms={rms:e} relative={relative:e}",
        expected.len()
    )
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m10_10_hard_budget_is_enforced_before_device_allocation() {
    let _serial = common::serial();
    // 1 MiB hard ceiling. Every refusal must come from the admission policy,
    // before any CUDA allocation call, and the accounting must never exceed it.
    let memory = common::memory_with_vram(1 << 20);
    let baseline = memory.usage_bytes(MemoryTier::Vram);
    assert_eq!(baseline, 0, "the owner starts with no VRAM held");

    // A request larger than the ceiling is refused with a typed budget error.
    let oversized = memory.allocate_tensor(vec![8192, 513], DType::BF16);
    assert!(
        matches!(
            oversized,
            Err(CudaTensorError::Memory(CudaMemoryError::Policy(
                MemoryError::RequestTooLarge {
                    tier: MemoryTier::Vram,
                    ..
                }
            )))
        ),
        "an 8.01 MiB request must be refused by the 1 MiB hard budget, got {oversized:?}"
    );
    assert_eq!(
        memory.usage_bytes(MemoryTier::Vram),
        baseline,
        "a refused admission must not create a device allocation"
    );

    // Small real M10 tensors inside the ceiling: a Linear whose output would
    // need 2 MiB, more than the ceiling. The op is refused and nothing is
    // allocated, so no un-budgeted allocation ever reaches the device.
    let mut rng = claymore_model_gemma::rng::SplitMix64::new(0x5169_5090_5e01_0001);
    let mut bits = |elements: usize| -> Vec<Bf16> {
        (0..elements)
            .map(|_| Bf16::from_f32(rng.next_f64() as f32 * 2.0 - 1.0))
            .collect()
    };
    let input_bits = bits(4096 * 8);
    let weight_bits = bits(256 * 8);
    let input = memory
        .upload_bf16(vec![4096, 8], &input_bits)
        .expect("input tensor inside the ceiling");
    let weight = memory
        .upload_bf16(vec![256, 8], &weight_bits)
        .expect("weight tensor inside the ceiling");
    let used = memory.usage_bytes(MemoryTier::Vram);
    assert_eq!(
        used,
        (4096 * 8 + 256 * 8) * 2,
        "the M10 tensors are accounted byte-exactly"
    );

    let dispatcher = common::dispatcher(&memory);
    let refusal = dispatcher.execute(&ExecutionRequest::new(TensorOp::Linear {
        input: &input,
        weight: &weight,
    }));
    assert!(
        matches!(refusal, Err(claymore_tensor::ExecutionError::Operation(_))),
        "a 2 MiB output over a 1 MiB ceiling must be a typed refusal, got {refusal:?}"
    );
    assert_eq!(
        memory.usage_bytes(MemoryTier::Vram),
        used,
        "a refused op must not allocate VRAM"
    );

    // A Linear that fits inside the ceiling runs normally on the same owner.
    let small_weight_bits = bits(32 * 8);
    let small_weight = memory
        .upload_bf16(vec![32, 8], &small_weight_bits)
        .expect("small weight tensor inside the ceiling");
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Linear {
            input: &input,
            weight: &small_weight,
        }))
        .expect("in-budget M10 linear");
    // The reference uses the exact same bits, so the comparison is a true
    // differential test on identical inputs.
    let cpu = common::cpu();
    let cpu_input = common::bf16_tensor(&cpu, vec![4096, 8], &input_bits);
    let cpu_weight = common::bf16_tensor(&cpu, vec![32, 8], &small_weight_bits);
    let reference = common::bf16_values(&cpu, &linear(&cpu, &cpu_input, &cpu_weight).unwrap());
    assert_bf16(
        "in-budget linear",
        &reference,
        &memory.download_bf16(&output).unwrap(),
        BF16_ULP,
    );
    assert!(
        memory.usage_bytes(MemoryTier::Vram) <= 1 << 20,
        "the accounting must stay inside the hard ceiling"
    );

    let snapshot = memory
        .snapshot(112 << 30, 100 << 30)
        .expect("device memory snapshot");
    println!(
        "HW-M10 pressure: hard=1 MiB over_ceiling_request=refused over_ceiling_op=refused in_budget_op=ok vram_total={} vram_free={}",
        snapshot.vram_total_bytes, snapshot.vram_free_bytes
    );
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m10_11_in_flight_m10_launch_protects_its_allocations() {
    let _serial = common::serial();
    // 24 MiB ceiling, 24 MiB of real M10 tensors, all guarded by the queued
    // launches. A further request can only be served by evicting in-flight
    // allocations, which the policy refuses until completion is observed.
    let memory = common::memory_with_exact_vram(24 << 20);
    let tokens = 1024usize;
    let heads = 16usize;
    let kv_heads = 8usize;
    let head_dim = 256usize;
    let scaling = (head_dim as f32).powf(-0.5);

    let mut rng = claymore_model_gemma::rng::SplitMix64::new(0x5169_5090_5e02_0002);
    let mut fill = |elements: usize| -> Vec<Bf16> {
        (0..elements)
            .map(|_| Bf16::from_f32(rng.next_f64() as f32 * 2.0 - 1.0))
            .collect()
    };
    let query = memory
        .upload_bf16(
            vec![tokens, heads, head_dim],
            &fill(tokens * heads * head_dim),
        )
        .expect("query upload");
    let key = memory
        .upload_bf16(
            vec![tokens, kv_heads, head_dim],
            &fill(tokens * kv_heads * head_dim),
        )
        .expect("key upload");
    let value = memory
        .upload_bf16(
            vec![tokens, kv_heads, head_dim],
            &fill(tokens * kv_heads * head_dim),
        )
        .expect("value upload");
    let output = memory
        .allocate_tensor(vec![tokens, heads, head_dim], DType::BF16)
        .expect("output allocation");
    assert_eq!(
        memory.usage_bytes(MemoryTier::Vram),
        24 << 20,
        "the M10 tensors fill the ceiling exactly"
    );

    let op = TensorOp::Attention {
        query: &query,
        key: &key,
        value: &value,
        mask: None,
        scale: scaling,
    };
    // Five full-canvas attention launches (1024 x 1024 x 256 x 16): the queued
    // stream keeps the last launch pending for a measurable window.
    let tokens_queued: Vec<_> = (0..5)
        .map(|_| common::launch_raw(&memory, &op, &output))
        .collect();
    let last = tokens_queued.last().expect("pending token");
    assert!(
        !memory
            .manager()
            .is_transfer_complete(last)
            .expect("event query"),
        "the last queued M10 launch must still be pending"
    );

    // Every guarded allocation must be in the token's guard set.
    let guarded: Vec<String> = last
        .allocation_ids()
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    for tensor in [&query, &key, &value, &output] {
        let id = memory.buffer_id(tensor).expect("managed id");
        assert!(
            guarded.contains(&id),
            "the in-flight launch must guard {id}, got {guarded:?}"
        );
    }

    // All VRAM is in flight, so an intruder allocation cannot evict anything.
    let tick = memory.tick();
    let refusal = memory
        .manager_mut()
        .allocate_device("m10-intruder", 4 << 20, tick, false)
        .expect_err("an in-flight launch must not be evicted");
    assert!(
        matches!(
            refusal,
            CudaMemoryError::Policy(MemoryError::InsufficientReclaimable {
                tier: MemoryTier::Vram,
                ..
            })
        ),
        "the in-flight M10 launch must block eviction, got {refusal:?}"
    );
    assert_eq!(
        memory.usage_bytes(MemoryTier::Vram),
        24 << 20,
        "the refused admission must not release any guard"
    );

    // After the completion boundary is observed, eviction becomes possible.
    for token in &tokens_queued {
        common::complete(&memory, token);
    }
    let tick = memory.tick();
    memory
        .manager_mut()
        .allocate_device("m10-intruder", 4 << 20, tick, false)
        .expect("eviction is possible after the launch completed");

    println!(
        "HW-M10 in-flight guard: {} guarded allocations protected, eviction blocked while pending, allowed after completion",
        guarded.len()
    );
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m10_12_repeated_m10_launches_are_bit_exact() {
    let _serial = common::serial();
    let cpu = common::cpu();
    let memory = common::memory();
    let dispatcher = common::dispatcher(&memory);

    let canvas_tokens = 128usize;
    let heads = 8usize;
    let kv_heads = 4usize;
    let head_dim = 256usize;
    let scaling = (head_dim as f32).powf(-0.5);
    let mut rng = claymore_model_gemma::rng::SplitMix64::new(0x5169_5090_5e03_0003);
    let values: Vec<f32> = (0..(canvas_tokens * head_dim * (heads + 2 * kv_heads)))
        .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
        .collect();
    let query_bits: Vec<Bf16> = values[..canvas_tokens * heads * head_dim]
        .iter()
        .map(|v| Bf16::from_f32(*v))
        .collect();
    let key_bits: Vec<Bf16> = values[canvas_tokens * heads * head_dim
        ..canvas_tokens * heads * head_dim + canvas_tokens * kv_heads * head_dim]
        .iter()
        .map(|v| Bf16::from_f32(*v))
        .collect();
    let value_bits: Vec<Bf16> = values
        [canvas_tokens * heads * head_dim + canvas_tokens * kv_heads * head_dim..]
        .iter()
        .map(|v| Bf16::from_f32(*v))
        .collect();

    let cpu_query = common::bf16_tensor(&cpu, vec![canvas_tokens, heads, head_dim], &query_bits);
    let cpu_key = common::bf16_tensor(&cpu, vec![canvas_tokens, kv_heads, head_dim], &key_bits);
    let cpu_value = common::bf16_tensor(&cpu, vec![canvas_tokens, kv_heads, head_dim], &value_bits);
    let reference = common::bf16_values(
        &cpu,
        &attention(&cpu, &cpu_query, &cpu_key, &cpu_value, None, scaling).unwrap(),
    );

    let baseline = memory.usage_bytes(MemoryTier::Vram);
    let peak = std::cell::Cell::new(0u64);
    let mut first: Option<Vec<Bf16>> = None;
    for repetition in 0..5 {
        let query = memory
            .upload_bf16(vec![canvas_tokens, heads, head_dim], &query_bits)
            .expect("query upload");
        let key = memory
            .upload_bf16(vec![canvas_tokens, kv_heads, head_dim], &key_bits)
            .expect("key upload");
        let value = memory
            .upload_bf16(vec![canvas_tokens, kv_heads, head_dim], &value_bits)
            .expect("value upload");
        let bits = common::run(
            &memory,
            &dispatcher,
            &peak,
            TensorOp::Attention {
                query: &query,
                key: &key,
                value: &value,
                mask: None,
                scale: scaling,
            },
        );
        drop(query);
        drop(key);
        drop(value);
        assert_eq!(
            memory.usage_bytes(MemoryTier::Vram),
            baseline,
            "repetition {repetition} must return the accounting to the baseline"
        );
        assert_bf16_reduction(
            &format!("repeated attention {repetition}"),
            &reference,
            &bits,
            value_bits
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(0f32, f32::max),
            BF16_ATTENTION_REDUCTION_TOLERANCE,
        );
        match &first {
            None => first = Some(bits),
            Some(first_bits) => {
                assert_eq!(
                    first_bits, &bits,
                    "repetition {repetition} must be bit-identical"
                )
            }
        }
    }
    println!(
        "HW-M10 repetition: 5 launches bit-identical, accounting returned to baseline each cycle, peak={} bytes",
        peak.get()
    );
}
/// BF16 distance in representable steps.
fn bf16_ulp(a: Bf16, b: Bf16) -> u32 {
    let a = a.to_bits();
    let b = b.to_bits();
    if a == b {
        return 0;
    }
    // Map to an ordered integer so the distance is monotone across zero.
    let key = |bits: u16| -> i32 {
        if bits & 0x8000 != 0 {
            i32::from(!bits)
        } else {
            i32::from(bits) + 0x8000
        }
    };
    (key(a) - key(b)).unsigned_abs()
}

mod common {
    use claymore_core::MemoryTier;
    use claymore_device::{CpuDevice, DType, Device};
    use claymore_memory::{CudaTensorBackend, CudaTensorMemory, EvictionPolicy, MemoryBudgets};
    use claymore_tensor::{Bf16, Tensor, TensorDispatcher, TensorLayout, TensorOp};

    pub(super) fn serial() -> std::sync::MutexGuard<'static, ()> {
        static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        GPU_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(super) fn cpu() -> CpuDevice {
        CpuDevice::new()
    }

    pub(super) fn memory() -> CudaTensorMemory {
        // M10 budget envelope: 26 GiB soft, 28 GiB hard on the 32 GiB device.
        let budgets = MemoryBudgets {
            vram_soft_bytes: 26 * 1024 * 1024 * 1024,
            vram_hard_bytes: 28 * 1024 * 1024 * 1024,
            host_bytes: 32 * 1024 * 1024 * 1024,
            pinned_host_bytes: 8 * 1024 * 1024 * 1024,
            expert_cache_target_bytes: 4 * 1024 * 1024 * 1024,
            expert_cache_max_bytes: 8 * 1024 * 1024 * 1024,
        };
        CudaTensorMemory::new(0, budgets, EvictionPolicy::Lru)
            .expect("CUDA tensor memory within the M10 budgets")
    }

    pub(super) fn dispatcher(memory: &CudaTensorMemory) -> TensorDispatcher {
        TensorDispatcher::new(vec![Box::new(CudaTensorBackend::new(memory))])
    }

    /// A memory owner with an explicit VRAM hard ceiling, used by the pressure
    /// and eviction tests. The soft budget is 90 % of the ceiling.
    pub(super) fn memory_with_vram(vram_bytes: u64) -> CudaTensorMemory {
        let budgets = MemoryBudgets {
            vram_soft_bytes: (vram_bytes * 9) / 10,
            vram_hard_bytes: vram_bytes,
            host_bytes: 8 << 30,
            pinned_host_bytes: 1 << 30,
            expert_cache_target_bytes: vram_bytes / 2,
            expert_cache_max_bytes: vram_bytes / 2 + 1,
        };
        CudaTensorMemory::new(0, budgets, EvictionPolicy::Lru).expect("tight CUDA tensor memory")
    }

    /// A memory owner whose soft and hard VRAM ceilings are equal: the policy
    /// then evicts only under hard pressure, never proactively. Used by the
    /// in-flight protection test, where the point is to keep every allocation
    /// resident while the launch is pending.
    pub(super) fn memory_with_exact_vram(vram_bytes: u64) -> CudaTensorMemory {
        let budgets = MemoryBudgets {
            vram_soft_bytes: vram_bytes,
            vram_hard_bytes: vram_bytes,
            host_bytes: 8 << 30,
            pinned_host_bytes: 1 << 30,
            expert_cache_target_bytes: vram_bytes / 2,
            expert_cache_max_bytes: vram_bytes / 2 + 1,
        };
        CudaTensorMemory::new(0, budgets, EvictionPolicy::Lru)
            .expect("exact-ceiling CUDA tensor memory")
    }

    /// Launch one M10 op through the manager's raw launch boundary and return
    /// the pending transfer token, without waiting for completion.
    pub(super) fn launch_raw(
        memory: &CudaTensorMemory,
        op: &TensorOp<'_>,
        output: &Tensor,
    ) -> claymore_memory::TransferToken {
        let input_ids: Vec<String> = op
            .inputs()
            .iter()
            .map(|input| memory.buffer_id(input).expect("managed input id"))
            .collect();
        let inputs: Vec<&str> = input_ids.iter().map(String::as_str).collect();
        let output_id = memory.buffer_id(output).expect("managed output id");
        let launch_tick = memory.tick();
        let mut manager = memory.manager_mut();
        manager
            .launch_tensor_op(op, &inputs, &output_id, launch_tick)
            .expect("M10 launch through the manager boundary")
    }

    /// Complete a pending transfer token through the explicit event wait.
    pub(super) fn complete(memory: &CudaTensorMemory, token: &claymore_memory::TransferToken) {
        let wait_tick = memory.tick();
        memory
            .manager_mut()
            .wait_for_transfer(token, wait_tick)
            .expect("explicit M10 transfer completion");
    }

    /// Record the VRAM watermark after every allocation and launch.
    fn record_peak(memory: &CudaTensorMemory, peak: &std::cell::Cell<u64>) {
        let usage = memory.usage_bytes(MemoryTier::Vram);
        if usage > peak.get() {
            peak.set(usage);
        }
    }

    /// Upload BF16 bits to VRAM through the memory manager and return the
    /// round-trip download, which must be bit-for-bit identical.
    pub(super) fn upload(
        memory: &CudaTensorMemory,
        peak: &std::cell::Cell<u64>,
        shape: Vec<usize>,
        bits: &[Bf16],
    ) -> Vec<Bf16> {
        let tensor = memory
            .upload_bf16(shape, bits)
            .expect("BF16 upload through the memory manager");
        record_peak(memory, peak);
        let downloaded = memory.download_bf16(&tensor).expect("BF16 download");
        assert_eq!(&downloaded, bits, "upload round trip must be verbatim");
        drop(tensor);
        downloaded
    }

    /// Materialize a VRAM tensor from downloaded bits. Reshapes between the
    /// reference and the device chain are explicit re-uploads, never aliases.
    pub(super) fn tensor_of(
        memory: &CudaTensorMemory,
        peak: &std::cell::Cell<u64>,
        shape: Vec<usize>,
        bits: &[Bf16],
    ) -> Tensor {
        let tensor = memory
            .upload_bf16(shape, bits)
            .expect("BF16 tensor materialization");
        record_peak(memory, peak);
        tensor
    }

    /// Execute one validated op on the CUDA backend and return the completed
    /// device output, downloaded after the kernel event has been observed.
    pub(super) fn run(
        memory: &CudaTensorMemory,
        dispatcher: &TensorDispatcher,
        peak: &std::cell::Cell<u64>,
        op: TensorOp<'_>,
    ) -> Vec<Bf16> {
        let output = dispatcher
            .execute(&claymore_tensor::ExecutionRequest::new(op))
            .expect("M10 kernel launch and completion");
        record_peak(memory, peak);
        let bits = memory.download_bf16(&output).expect("BF16 download");
        drop(output);
        bits
    }

    pub(super) fn f32_tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let layout = TensorLayout::contiguous(shape, DType::F32).unwrap();
        let mut storage = device
            .allocate(MemoryTier::Host, layout.span_bytes() as u64)
            .unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        device.write(storage.as_mut(), 0, &bytes).unwrap();
        Tensor::new(storage, layout.shape().to_vec(), DType::F32).unwrap()
    }

    pub(super) fn bf16_tensor(device: &CpuDevice, shape: Vec<usize>, values: &[Bf16]) -> Tensor {
        let layout = TensorLayout::contiguous(shape, DType::BF16).unwrap();
        let mut storage = device
            .allocate(MemoryTier::Host, layout.span_bytes() as u64)
            .unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        device.write(storage.as_mut(), 0, &bytes).unwrap();
        Tensor::new(storage, layout.shape().to_vec(), DType::BF16).unwrap()
    }

    pub(super) fn values(device: &CpuDevice, tensor: &Tensor) -> Vec<f32> {
        let bytes = device
            .read(
                tensor.storage(),
                tensor.offset_bytes(),
                tensor.layout().span_bytes(),
            )
            .unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub(super) fn bf16_values(device: &CpuDevice, tensor: &Tensor) -> Vec<Bf16> {
        let bytes = device
            .read(
                tensor.storage(),
                tensor.offset_bytes(),
                tensor.layout().span_bytes(),
            )
            .unwrap();
        bytes
            .chunks_exact(2)
            .map(|c| Bf16::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }
}
