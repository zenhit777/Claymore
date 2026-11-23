//! Real RTX 5090 differential validation of the M8 CUDA tensor backend.
//!
//! These tests are ignored by default. They require the NVIDIA driver exposed
//! in WSL2 and are never substituted by a CPU simulation. Run them with:
//!
//! ```text
//! RUSTFLAGS="-L native=/usr/lib/wsl/lib" \
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib \
//! cargo test -p claymore-memory --features cuda-driver \
//!   --test hardware_cuda_tensor -- --ignored --test-threads=1 --nocapture
//! ```

use std::sync::{Mutex, MutexGuard};

use claymore_core::MemoryTier;
use claymore_device::{CpuDevice, DType, Device};
use claymore_memory::{
    CudaMemoryManager, CudaTensorBackend, CudaTensorComputeExecutor,
    CudaTensorError, CudaTensorMemory, CudaTensorTask, EvictionPolicy, MemoryBudgets,
};
use claymore_scheduler::{
    ExecutionOutcome, QueueRunner, Scheduler, SchedulerConfig, TaskKind, TaskSpec,
};
use claymore_tensor::{
    Bf16, CpuReferenceBackend, DispatchError, ExecutionRequest, OpError, Tensor, TensorDispatcher,
    TensorError, TensorLayout, TensorOp, UnsupportedReason,
};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn budgets(vram_bytes: u64) -> MemoryBudgets {
    MemoryBudgets {
        vram_soft_bytes: (vram_bytes * 9) / 10,
        vram_hard_bytes: vram_bytes,
        host_bytes: 8 << 30,
        pinned_host_bytes: 1 << 30,
        expert_cache_target_bytes: (vram_bytes / 2).min(14 << 30),
        expert_cache_max_bytes: (vram_bytes / 2 + 1).min(16 << 30),
    }
}

fn memory() -> CudaTensorMemory {
    CudaTensorMemory::new(0, budgets(24 << 30), EvictionPolicy::Lru).expect("CUDA memory owner")
}

fn values(seed: u32, len: usize, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let magnitude = ((state >> 16) & 0x7ff) as f32 / 2048.0;
            let sign = if (state >> 8) & 1 == 0 { 1.0 } else { -1.0 };
            // A few large amplitudes are mixed in to exercise F32 range.
            let boost = if index % 97 == 0 { 8.0 } else { 1.0 };
            sign * magnitude * scale * boost
        })
        .collect()
}

fn cpu_values(backend: &CpuReferenceBackend, tensor: &Tensor) -> Vec<f32> {
    backend
        .device()
        .read(
            tensor.storage(),
            tensor.offset_bytes(),
            tensor.layout().span_bytes(),
        )
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) {
    assert_eq!(expected.len(), actual.len(), "{label}: element count");
    let mut max_abs = 0.0f32;
    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (want, got)) in expected.iter().zip(actual).enumerate() {
        let difference = (want - got).abs();
        let tolerance = atol + rtol * want.abs();
        if difference > worst {
            worst = difference;
            worst_index = index;
        }
        max_abs = max_abs.max(difference);
        assert!(
            got.is_finite(),
            "{label}: non-finite output at {index}: {got}"
        );
        assert!(
            difference <= tolerance,
            "{label}: index {index} expected {want}, got {got}, |diff|={difference}, tol={tolerance}"
        );
    }
    println!(
        "HW-M8 {label}: n={} max_abs_diff={max_abs:e} worst_index={worst_index} worst_diff={worst:e}",
        expected.len()
    );
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_01_add_matches_cpu_for_multiple_shapes_and_signs() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, shape, scale) in [
        (1u32, vec![1usize], 1.0f32),
        (2, vec![7], 1000.0),
        (3, vec![2, 3], 0.01),
        (4, vec![64, 257], 25.0),
    ] {
        let len = shape.iter().product();
        let a = values(seed, len, scale);
        let b = values(seed + 100, len, scale);
        let cpu_a = cpu.tensor(shape.clone(), &a).unwrap();
        let cpu_b = cpu.tensor(shape.clone(), &b).unwrap();
        let gpu_a = memory.upload_f32(shape.clone(), &a).unwrap();
        let gpu_b = memory.upload_f32(shape, &b).unwrap();
        let expected = cpu_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::Add {
                        lhs: &cpu_a,
                        rhs: &cpu_b,
                    })
                    .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::Add {
                lhs: &gpu_a,
                rhs: &gpu_b,
            }))
            .unwrap();
        let actual = memory.download_f32(&output).unwrap();
        assert_close(
            &format!("add shape={:?}", output.layout().shape()),
            &expected,
            &actual,
            1e-5,
            1e-6,
        );
    }
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_02_matmul_matches_cpu_for_rank2_shapes() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, m, k, n) in [
        (11usize, 2usize, 3usize, 2usize),
        (12, 16, 64, 32),
        (13, 129, 17, 65),
    ] {
        let a = values(seed as u32, m * k, 3.0);
        let b = values(seed as u32 + 200, k * n, 3.0);
        let cpu_a = cpu.tensor(vec![m, k], &a).unwrap();
        let cpu_b = cpu.tensor(vec![k, n], &b).unwrap();
        let gpu_a = memory.upload_f32(vec![m, k], &a).unwrap();
        let gpu_b = memory.upload_f32(vec![k, n], &b).unwrap();
        let expected = cpu_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::MatMul {
                        lhs: &cpu_a,
                        rhs: &cpu_b,
                    })
                    .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::MatMul {
                lhs: &gpu_a,
                rhs: &gpu_b,
            }))
            .unwrap();
        let actual = memory.download_f32(&output).unwrap();
        assert_eq!(output.layout().shape(), &[m, n]);
        assert_close(
            &format!("matmul {m}x{k}x{n}"),
            &expected,
            &actual,
            1e-3,
            2e-5,
        );
    }
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_03_softmax_is_stable_for_large_logits() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, shape, scale) in [
        (21u32, vec![1usize, 3usize], 1.0f32),
        (22, vec![8, 257], 100.0),
        (23, vec![1, 1024], 1000.0),
        (24, vec![2, 3, 64], 500.0),
    ] {
        let len = shape.iter().product();
        let mut logits = values(seed, len, scale);
        logits[0] = 10_000.0;
        logits[len - 1] = -10_000.0;
        let cpu_input = cpu.tensor(shape.clone(), &logits).unwrap();
        let gpu_input = memory.upload_f32(shape.clone(), &logits).unwrap();
        let expected = cpu_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::SoftmaxLastDim { input: &cpu_input })
                        .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::SoftmaxLastDim {
                input: &gpu_input,
            }))
            .unwrap();
        let actual = memory.download_f32(&output).unwrap();
        assert_close(
            &format!("softmax shape={shape:?}"),
            &expected,
            &actual,
            1e-6,
            1e-5,
        );
        let width = *shape.last().unwrap();
        for row in actual.chunks(width) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "softmax row sum {sum}");
        }
    }
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_04_rms_norm_matches_cpu_with_explicit_scale() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, shape, epsilon, scale) in [
        (31u32, vec![4usize], 0.0f32, 1.5f32),
        (32, vec![6, 64], 1e-5, 0.25),
        (33, vec![2, 1, 129], 1e-3, 2.0),
    ] {
        let len: usize = shape.iter().product();
        let width = *shape.last().unwrap();
        let input = values(seed, len, scale);
        let weights: Vec<f32> = (0..width)
            .map(|index| 0.25 + (index % 7) as f32 * 0.125)
            .collect();
        let cpu_input = cpu.tensor(shape.clone(), &input).unwrap();
        let cpu_scale = cpu.tensor(vec![width], &weights).unwrap();
        let gpu_input = memory.upload_f32(shape.clone(), &input).unwrap();
        let gpu_scale = memory.upload_f32(vec![width], &weights).unwrap();
        let expected = cpu_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::RmsNorm {
                        input: &cpu_input,
                        scale: &cpu_scale,
                        epsilon,
                    })
                    .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::RmsNorm {
                input: &gpu_input,
                scale: &gpu_scale,
                epsilon,
            }))
            .unwrap();
        let actual = memory.download_f32(&output).unwrap();
        assert_close(
            &format!("rms_norm shape={shape:?}"),
            &expected,
            &actual,
            1e-5,
            2e-5,
        );
    }
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_05_dispatch_refusals_and_typed_errors() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);
    let gpu_a = memory.upload_f32(vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let gpu_b = memory.upload_f32(vec![4], &[4.0, 3.0, 2.0, 1.0]).unwrap();

    let cpu_forced = ExecutionRequest::new(TensorOp::Add {
        lhs: &gpu_a,
        rhs: &gpu_b,
    })
    .on("cpu-reference");
    assert!(matches!(
        dispatcher.plan(&cpu_forced),
        Err(DispatchError::NoCapableBackend { .. })
    ));

    let report = dispatcher
        .explain(&ExecutionRequest::new(TensorOp::Add {
            lhs: &gpu_a,
            rhs: &gpu_b,
        }))
        .unwrap();
    let text = report.to_string();
    assert!(text.contains("SELECT cuda-native"), "{text}");
    assert!(text.contains("REFUSE cpu-reference"), "{text}");

    // FP16 metadata exists in DType but no path is implemented for it.
    let f16 = memory.allocate_tensor(vec![4], DType::F16);
    assert!(matches!(f16, Err(CudaTensorError::UnsupportedDType)));

    let host = cpu.tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let short = cpu.tensor(vec![2], &[1.0, 2.0]).unwrap();
    let mismatch = ExecutionRequest::new(TensorOp::Add {
        lhs: &host,
        rhs: &short,
    });
    assert!(matches!(
        dispatcher.plan(&mismatch),
        Err(DispatchError::Contract(OpError::ShapeMismatch))
    ));

    let storage = cpu.device().allocate(MemoryTier::Host, 40).unwrap();
    let strided = Tensor::with_layout(
        storage,
        TensorLayout::strided(vec![2, 2], vec![3, 1], DType::F32).unwrap(),
        0,
    )
    .unwrap();
    let non_contiguous = ExecutionRequest::new(TensorOp::SoftmaxLastDim { input: &strided });
    assert!(matches!(
        dispatcher.plan(&non_contiguous),
        Err(DispatchError::Contract(OpError::NonContiguous))
    ));
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_06_allocation_accounting_lifetime_and_budget_guards() {
    let _serial = serial();
    let memory = memory();
    let before = memory.usage_bytes(MemoryTier::Vram);
    let a = memory
        .upload_f32(vec![4096], &values(41, 4096, 1.0))
        .unwrap();
    let b = memory
        .upload_f32(vec![4096], &values(42, 4096, 1.0))
        .unwrap();
    let after_inputs = memory.usage_bytes(MemoryTier::Vram);
    assert_eq!(after_inputs - before, 2 * 4096 * 4);

    let dispatcher = TensorDispatcher::new(vec![Box::new(CudaTensorBackend::new(&memory))]);
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Add { lhs: &a, rhs: &b }))
        .unwrap();
    assert_eq!(memory.usage_bytes(MemoryTier::Vram) - before, 3 * 4096 * 4);
    let actual = memory.download_f32(&output).unwrap();
    assert_eq!(actual.len(), 4096);

    drop(output);
    assert_eq!(memory.usage_bytes(MemoryTier::Vram) - before, 2 * 4096 * 4);
    drop(a);
    drop(b);
    assert_eq!(memory.usage_bytes(MemoryTier::Vram), before);

    let snapshot = memory.snapshot(8 << 30, 4 << 30).unwrap();
    assert!(snapshot.vram_total_bytes >= snapshot.vram_free_bytes);
    println!(
        "HW-M8 accounting: before={before} peak={} after={} vram_total={} vram_free={}",
        after_inputs + 4096 * 4,
        memory.usage_bytes(MemoryTier::Vram),
        snapshot.vram_total_bytes,
        snapshot.vram_free_bytes
    );

    let tight = CudaTensorMemory::new(0, budgets(8 << 10), EvictionPolicy::Lru).unwrap();
    assert!(matches!(
        tight.allocate_tensor(vec![100_000], DType::F32),
        Err(CudaTensorError::Memory(_))
    ));
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_07_kernel_event_guards_and_explicit_wait() {
    let _serial = serial();
    let memory = memory();
    let a = memory
        .upload_f32(vec![32, 32], &values(51, 1024, 2.0))
        .unwrap();
    let b = memory
        .upload_f32(vec![32, 32], &values(52, 1024, 2.0))
        .unwrap();
    let output = memory.allocate_tensor(vec![32, 32], DType::F32).unwrap();
    let a_id = memory.buffer_id(&a).unwrap();
    let b_id = memory.buffer_id(&b).unwrap();
    let out_id = memory.buffer_id(&output).unwrap();

    let inputs = [a_id.as_str(), b_id.as_str()];
    let launch_tick = memory.tick();
    let token = {
        let mut manager = memory.manager_mut();
        let op = TensorOp::Add { lhs: &a, rhs: &b };
        manager
            .launch_tensor_op(&op, &inputs, &out_id, launch_tick)
            .expect("kernel launch")
    };
    {
        let manager = memory.manager();
        assert!(manager.allocation(&out_id).is_some());
        let _ = manager.is_transfer_complete(&token).expect("event query");
    }
    let wait_tick = memory.tick();
    {
        let mut manager = memory.manager_mut();
        manager
            .wait_for_transfer(&token, wait_tick)
            .expect("explicit event wait");
    }
    let actual = memory.download_f32(&output).unwrap();
    let expected: Vec<f32> = values(51, 1024, 2.0)
        .iter()
        .zip(values(52, 1024, 2.0).iter())
        .map(|(left, right)| left + right)
        .collect();
    assert_close("event-backed add", &expected, &actual, 1e-5, 1e-6);
    println!(
        "HW-M8 event guard: token={} allocations={:?}",
        token.id(),
        token.allocation_ids()
    );
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_08_scheduler_driven_compute_keeps_tensors_out_of_scheduler() {
    let _serial = serial();
    let memory = memory();
    let a = memory
        .upload_f32(vec![128, 128], &values(61, 16_384, 1.5))
        .unwrap();
    let b = memory
        .upload_f32(vec![128, 128], &values(62, 16_384, 1.5))
        .unwrap();

    let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
    let compute_id = scheduler.submit(TaskSpec::new(TaskKind::Compute)).unwrap();
    let dependent_id = scheduler
        .submit(TaskSpec::new(TaskKind::Compute).depends_on(compute_id))
        .unwrap();
    let mut runner = QueueRunner::new(TaskKind::Compute, CudaTensorComputeExecutor::new(&memory));
    runner
        .executor_mut()
        .register(compute_id, CudaTensorTask::Add { lhs: a, rhs: b })
        .unwrap();
    runner
        .executor_mut()
        .register(
            dependent_id,
            CudaTensorTask::SoftmaxLastDim {
                input: memory.upload_f32(vec![4], &[0.0, 1.0, 2.0, 3.0]).unwrap(),
            },
        )
        .unwrap();

    assert_eq!(
        runner.run_one(&mut scheduler).unwrap(),
        ExecutionOutcome::Completed(compute_id)
    );
    assert_eq!(
        scheduler.task(dependent_id).unwrap().state,
        claymore_scheduler::TaskState::Ready
    );
    assert_eq!(
        runner.run_one(&mut scheduler).unwrap(),
        ExecutionOutcome::Completed(dependent_id)
    );

    let output = runner
        .executor_mut()
        .take_result(compute_id)
        .expect("scheduler executor returns the completed tensor");
    let actual = memory.download_f32(&output).unwrap();
    let expected: Vec<f32> = values(61, 16_384, 1.5)
        .iter()
        .zip(values(62, 16_384, 1.5).iter())
        .map(|(left, right)| left + right)
        .collect();
    assert_close("scheduler-driven add", &expected, &actual, 1e-5, 1e-6);
    assert_eq!(runner.executor().pending_results(), 1);
    assert_eq!(runner.executor().pending_operations(), 0);
    println!("HW-M8 scheduler: completed={compute_id:?} dependent={dependent_id:?}");
}

#[test]
#[ignore = "requires a CUDA driver and RTX 5090 to validate the real CUDA manager boundary"]
fn hw_m8_09_grid_y_limit_is_enforced_by_dispatch_and_launch_boundary() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    // A 16x16 matmul tile produces exactly the maximum supported grid.y.
    let boundary_lhs = memory
        .allocate_tensor(vec![1_048_560, 1], DType::F32)
        .unwrap();
    let rhs = memory.allocate_tensor(vec![1, 1], DType::F32).unwrap();
    let boundary = ExecutionRequest::new(TensorOp::MatMul {
        lhs: &boundary_lhs,
        rhs: &rhs,
    });
    assert!(
        dispatcher.plan(&boundary).is_ok(),
        "grid.y == 65,535 must remain dispatchable"
    );

    // One additional row requires grid.y == 65,536 and must be rejected by
    // dispatch before a backend is selected, even though the CPU backend
    // cannot consume VRAM tensors either.
    let oversized_lhs = memory
        .allocate_tensor(vec![1_048_577, 1], DType::F32)
        .unwrap();
    let oversized = ExecutionRequest::new(TensorOp::MatMul {
        lhs: &oversized_lhs,
        rhs: &rhs,
    });
    match dispatcher.plan(&oversized) {
        Err(DispatchError::NoCapableBackend { refusals, .. }) => {
            assert!(
                refusals.iter().any(|(backend, reason)| {
                    backend == "cuda-native"
                        && matches!(
                            reason,
                            UnsupportedReason::LaunchLimit {
                                axis: "y",
                                blocks: 65_536,
                                maximum: 65_535
                            }
                        )
                }),
                "CUDA must explain the exact grid limit: {refusals:?}"
            );
        }
        other => panic!("oversized CUDA grid was not refused: {other:?}"),
    }

    // Exercise the defensive check immediately before cuLaunchKernel through
    // the production manager. This must fail as a configuration error without
    // submitting a kernel or leaving guards/events behind.
    let output = memory
        .allocate_tensor(vec![1_048_577, 1], DType::F32)
        .unwrap();
    let lhs_id = memory.buffer_id(&oversized_lhs).unwrap();
    let rhs_id = memory.buffer_id(&rhs).unwrap();
    let output_id = memory.buffer_id(&output).unwrap();
    let input_ids = [lhs_id.as_str(), rhs_id.as_str()];
    let tick = memory.tick();
    let error = memory
        .manager_mut()
        .launch_tensor_op(
            &TensorOp::MatMul {
                lhs: &oversized_lhs,
                rhs: &rhs,
            },
            &input_ids,
            &output_id,
            tick,
        )
        .expect_err("the defensive CUDA launch check must reject grid.y > 65,535");
    assert!(
        error.to_string().contains("65,536") || error.to_string().contains("65536"),
        "unexpected grid-limit error: {error}"
    );
    assert!(
        memory.manager().allocation(&lhs_id).is_some()
            && memory.manager().allocation(&rhs_id).is_some()
            && memory.manager().allocation(&output_id).is_some(),
        "all allocations must remain accounted after the rejected launch"
    );
    println!(
        "HW-M8 grid limit: exact boundary accepted; overflow rejected before kernel submission"
    );
}

// ---------------------------------------------------------------------------
// BF16 differential validation
// ---------------------------------------------------------------------------
//
// Numerical policy (fixed before the campaign, never widened after a run):
//   - inputs are exact 16-bit patterns uploaded verbatim to VRAM;
//   - CPU and CUDA both widen to FP32, compute/accumulate in FP32 and round the
//     result once with round-to-nearest-even;
//   - add is exact at the BF16 bit level (0 representable steps);
//   - reductions (matmul, softmax, RMSNorm) allow at most 1 BF16 representable
//     step because the CPU and CUDA reduction orders differ (sequential vs
//     tree), which can move a value across a single rounding boundary;
//   - NaNs must appear on both sides; infinities are compared as ordered
//     representable values.

fn bf16_values(seed: u32, len: usize, scale: f32) -> Vec<Bf16> {
    values(seed, len, scale)
        .into_iter()
        .map(Bf16::from_f32)
        .collect()
}

fn cpu_bf16_values(backend: &CpuReferenceBackend, tensor: &Tensor) -> Vec<Bf16> {
    assert_eq!(tensor.layout().dtype(), DType::BF16);
    backend
        .device()
        .read(
            tensor.storage(),
            tensor.offset_bytes(),
            tensor.layout().span_bytes(),
        )
        .unwrap()
        .chunks_exact(2)
        .map(|chunk| Bf16::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Monotonic key over the BF16 bit pattern: adjacent representable values
/// (including the ±0 transition) map to adjacent integers.
fn bf16_ordered(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits ^ 0x8000
    }
}

fn assert_bf16_close(label: &str, expected: &[Bf16], actual: &[Bf16], max_steps: u16) {
    assert_eq!(expected.len(), actual.len(), "{label}: element count");
    let mut worst = 0u16;
    let mut worst_index = 0usize;
    for (index, (want, got)) in expected.iter().zip(actual).enumerate() {
        if want.is_nan() || got.is_nan() {
            assert!(
                want.is_nan() && got.is_nan(),
                "{label}: index {index} NaN mismatch: expected {want}, got {got}"
            );
            continue;
        }
        let steps = bf16_ordered(want.to_bits()).abs_diff(bf16_ordered(got.to_bits()));
        if steps > worst {
            worst = steps;
            worst_index = index;
        }
        assert!(
            steps <= max_steps,
            "{label}: index {index} expected {want} ({:#06x}), got {got} ({:#06x}), steps={steps} > {max_steps}",
            want.to_bits(),
            got.to_bits()
        );
    }
    println!(
        "HW-M8-BF16 {label}: n={} max_steps={worst} worst_index={worst_index} limit={max_steps}",
        expected.len()
    );
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_01_add_matches_cpu_for_multiple_shapes_and_special_values() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(claymore_memory::CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, shape, scale) in [
        (101u32, vec![1usize], 1.0f32),
        (102, vec![7], 1000.0),
        (103, vec![2, 3], 0.01),
        (104, vec![64, 257], 25.0),
        (105, vec![3, 5, 7], 1.5), // odd dimensions in three axes
    ] {
        let len: usize = shape.iter().product();
        let a = bf16_values(seed, len, scale);
        let b = bf16_values(seed + 100, len, scale);
        let cpu_a = cpu.tensor_bf16(shape.clone(), &a).unwrap();
        let cpu_b = cpu.tensor_bf16(shape.clone(), &b).unwrap();
        let gpu_a = memory.upload_bf16(shape.clone(), &a).unwrap();
        let gpu_b = memory.upload_bf16(shape.clone(), &b).unwrap();
        let expected = cpu_bf16_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::Add {
                        lhs: &cpu_a,
                        rhs: &cpu_b,
                    })
                    .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::Add {
                lhs: &gpu_a,
                rhs: &gpu_b,
            }))
            .unwrap();
        assert_eq!(output.layout().dtype(), DType::BF16);
        let actual = memory.download_bf16(&output).unwrap();
        assert_bf16_close(&format!("add shape={shape:?}"), &expected, &actual, 0);
    }

    // Signed zeros, subnormals, infinities and NaN are exercised explicitly.
    let a = vec![
        Bf16::ZERO_BITS,
        Bf16::NEG_ZERO_BITS,
        Bf16::MIN_SUBNORMAL_BITS,
        Bf16::MIN_NORMAL_BITS,
        Bf16::INFINITY_BITS,
        Bf16::NEG_INFINITY_BITS,
        Bf16::QUIET_NAN_BITS,
        Bf16::from_f32(1.0).to_bits(),
    ]
    .into_iter()
    .map(Bf16::from_bits)
    .collect::<Vec<_>>();
    let b = vec![
        Bf16::from_f32(-0.0),
        Bf16::from_f32(1.0),
        Bf16::from_f32(0.0),
        Bf16::from_f32(-1.0),
        Bf16::from_f32(1.0),
        Bf16::from_f32(1.0),
        Bf16::from_f32(1.0),
        Bf16::from_f32(2.0),
    ];
    let cpu_a = cpu.tensor_bf16(vec![8], &a).unwrap();
    let cpu_b = cpu.tensor_bf16(vec![8], &b).unwrap();
    let gpu_a = memory.upload_bf16(vec![8], &a).unwrap();
    let gpu_b = memory.upload_bf16(vec![8], &b).unwrap();
    let expected = cpu_bf16_values(
        &cpu,
        &dispatcher
            .execute(
                &ExecutionRequest::new(TensorOp::Add {
                    lhs: &cpu_a,
                    rhs: &cpu_b,
                })
                .on("cpu-reference"),
            )
            .unwrap(),
    );
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Add {
            lhs: &gpu_a,
            rhs: &gpu_b,
        }))
        .unwrap();
    let actual = memory.download_bf16(&output).unwrap();
    assert_bf16_close("add special values", &expected, &actual, 0);
    assert_eq!(actual[0], Bf16::from_bits(Bf16::ZERO_BITS)); // +0 + -0 = +0 in FP32
    assert_eq!(actual[1], Bf16::from_f32(1.0)); // -0 + 1 = 1
    assert_eq!(actual[4], Bf16::from_bits(Bf16::INFINITY_BITS));
    assert_eq!(actual[5], Bf16::from_bits(Bf16::NEG_INFINITY_BITS));
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_02_matmul_matches_cpu_for_rank2_shapes() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(claymore_memory::CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, m, k, n) in [
        (111usize, 2usize, 3usize, 2usize),
        (112, 16, 64, 32),
        (113, 129, 17, 65), // all odd / non-multiple of the 16x16 tile
        (114, 7, 9, 5),
    ] {
        let a = bf16_values(seed as u32, m * k, 3.0);
        let b = bf16_values(seed as u32 + 200, k * n, 3.0);
        let cpu_a = cpu.tensor_bf16(vec![m, k], &a).unwrap();
        let cpu_b = cpu.tensor_bf16(vec![k, n], &b).unwrap();
        let gpu_a = memory.upload_bf16(vec![m, k], &a).unwrap();
        let gpu_b = memory.upload_bf16(vec![k, n], &b).unwrap();
        let expected = cpu_bf16_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::MatMul {
                        lhs: &cpu_a,
                        rhs: &cpu_b,
                    })
                    .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::MatMul {
                lhs: &gpu_a,
                rhs: &gpu_b,
            }))
            .unwrap();
        assert_eq!(output.layout().shape(), &[m, n]);
        let actual = memory.download_bf16(&output).unwrap();
        assert_bf16_close(&format!("matmul {m}x{k}x{n}"), &expected, &actual, 1);
    }
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_03_softmax_is_stable_for_large_logits() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(claymore_memory::CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, shape, scale) in [
        (121u32, vec![1usize, 3usize], 1.0f32),
        (122, vec![8, 257], 100.0),
        (123, vec![1, 1024], 1000.0),
        (124, vec![2, 3, 64], 500.0),
        (125, vec![5, 7, 3], 2.0),
    ] {
        let len: usize = shape.iter().product();
        let mut logits = bf16_values(seed, len, scale);
        logits[0] = Bf16::from_f32(10_000.0);
        logits[len - 1] = Bf16::from_f32(-10_000.0);
        let cpu_input = cpu.tensor_bf16(shape.clone(), &logits).unwrap();
        let gpu_input = memory.upload_bf16(shape.clone(), &logits).unwrap();
        let expected = cpu_bf16_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::SoftmaxLastDim { input: &cpu_input })
                        .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::SoftmaxLastDim {
                input: &gpu_input,
            }))
            .unwrap();
        let actual = memory.download_bf16(&output).unwrap();
        assert_bf16_close(&format!("softmax shape={shape:?}"), &expected, &actual, 1);
        let width = *shape.last().unwrap();
        for row in actual.chunks(width) {
            let sum: f32 = row.iter().map(|value| value.to_f32()).sum();
            assert!(
                (sum - 1.0).abs() < 4e-3,
                "BF16 softmax row sum {sum} for shape {shape:?}"
            );
            assert!(row.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_04_rms_norm_matches_cpu_with_explicit_bf16_scale() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(claymore_memory::CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);

    for (seed, shape, epsilon, scale) in [
        (131u32, vec![4usize], 0.0f32, 1.5f32),
        (132, vec![6, 64], 1e-5, 0.25),
        (133, vec![2, 1, 129], 1e-3, 2.0),
        (134, vec![3, 5, 7], 1e-5, 0.5),
    ] {
        let len: usize = shape.iter().product();
        let width = *shape.last().unwrap();
        let input = bf16_values(seed, len, scale);
        let weights: Vec<Bf16> = (0..width)
            .map(|index| Bf16::from_f32(0.25 + (index % 7) as f32 * 0.125))
            .collect();
        let cpu_input = cpu.tensor_bf16(shape.clone(), &input).unwrap();
        let cpu_scale = cpu.tensor_bf16(vec![width], &weights).unwrap();
        let gpu_input = memory.upload_bf16(shape.clone(), &input).unwrap();
        let gpu_scale = memory.upload_bf16(vec![width], &weights).unwrap();
        let expected = cpu_bf16_values(
            &cpu,
            &dispatcher
                .execute(
                    &ExecutionRequest::new(TensorOp::RmsNorm {
                        input: &cpu_input,
                        scale: &cpu_scale,
                        epsilon,
                    })
                    .on("cpu-reference"),
                )
                .unwrap(),
        );
        let output = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::RmsNorm {
                input: &gpu_input,
                scale: &gpu_scale,
                epsilon,
            }))
            .unwrap();
        let actual = memory.download_bf16(&output).unwrap();
        assert_bf16_close(&format!("rms_norm shape={shape:?}"), &expected, &actual, 1);
        assert!(actual.iter().all(|value| value.is_finite()));
    }
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_05_dispatch_refusals_and_typed_errors() {
    let _serial = serial();
    let memory = memory();
    let cpu = CpuReferenceBackend::new();
    let dispatcher = TensorDispatcher::new(vec![
        Box::new(claymore_memory::CudaTensorBackend::new(&memory)),
        Box::new(CpuReferenceBackend::with_device(cpu.device().clone())),
    ]);
    let gpu_a = memory
        .upload_bf16(vec![4], &bf16_values(141, 4, 1.0))
        .unwrap();
    let gpu_b = memory
        .upload_bf16(vec![4], &bf16_values(142, 4, 1.0))
        .unwrap();

    let report = dispatcher
        .explain(&ExecutionRequest::new(TensorOp::Add {
            lhs: &gpu_a,
            rhs: &gpu_b,
        }))
        .unwrap();
    let text = report.to_string();
    assert!(text.contains("SELECT cuda-native"), "{text}");
    assert!(text.contains("BF16"), "{text}");
    assert!(text.contains("REFUSE cpu-reference"), "{text}");

    // FP16 is metadata only: no allocation, no dispatch, no kernel.
    assert!(matches!(
        memory.allocate_tensor(vec![4], DType::F16),
        Err(CudaTensorError::UnsupportedDType)
    ));
    let f16 = Tensor::new(
        cpu.device().allocate(MemoryTier::Host, 8).unwrap(),
        vec![4],
        DType::F16,
    )
    .unwrap();
    assert_eq!(
        dispatcher
            .plan(&ExecutionRequest::new(TensorOp::SoftmaxLastDim {
                input: &f16
            }))
            .unwrap_err(),
        DispatchError::Contract(OpError::UnsupportedDType)
    );

    // Mixed dtype and shape/layout errors remain typed and refuse before launch.
    let host_f32 = cpu.tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        dispatcher
            .plan(&ExecutionRequest::new(TensorOp::Add {
                lhs: &gpu_a,
                rhs: &host_f32,
            }))
            .unwrap_err(),
        DispatchError::Contract(OpError::MixedDType)
    );
    let short = memory
        .upload_bf16(vec![2], &bf16_values(143, 2, 1.0))
        .unwrap();
    assert_eq!(
        dispatcher
            .plan(&ExecutionRequest::new(TensorOp::Add {
                lhs: &gpu_a,
                rhs: &short,
            }))
            .unwrap_err(),
        DispatchError::Contract(OpError::ShapeMismatch)
    );
    let strided = Tensor::with_layout(
        cpu.device().allocate(MemoryTier::Host, 24).unwrap(),
        TensorLayout::strided(vec![2, 2], vec![3, 1], DType::BF16).unwrap(),
        0,
    )
    .unwrap();
    assert_eq!(
        dispatcher
            .plan(&ExecutionRequest::new(TensorOp::SoftmaxLastDim {
                input: &strided
            }))
            .unwrap_err(),
        DispatchError::Contract(OpError::NonContiguous)
    );
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_06_allocation_transfer_lifetime_and_budget_guards() {
    let _serial = serial();
    let memory = memory();
    let before = memory.usage_bytes(MemoryTier::Vram);
    let a = memory
        .upload_bf16(vec![4096], &bf16_values(151, 4096, 1.0))
        .unwrap();
    let b = memory
        .upload_bf16(vec![4096], &bf16_values(152, 4096, 1.0))
        .unwrap();
    assert_eq!(a.layout().span_bytes(), 8192);
    assert_eq!(memory.usage_bytes(MemoryTier::Vram) - before, 2 * 8192);

    let dispatcher = TensorDispatcher::new(vec![Box::new(
        claymore_memory::CudaTensorBackend::new(&memory),
    )]);
    let output = dispatcher
        .execute(&ExecutionRequest::new(TensorOp::Add { lhs: &a, rhs: &b }))
        .unwrap();
    assert_eq!(memory.usage_bytes(MemoryTier::Vram) - before, 3 * 8192);
    assert_eq!(memory.download_bf16(&output).unwrap().len(), 4096);
    drop(output);
    assert_eq!(memory.usage_bytes(MemoryTier::Vram) - before, 2 * 8192);
    drop(a);
    drop(b);
    assert_eq!(memory.usage_bytes(MemoryTier::Vram), before);

    let snapshot = memory.snapshot(8 << 30, 4 << 30).unwrap();
    assert!(snapshot.vram_total_bytes >= snapshot.vram_free_bytes);
    println!(
        "HW-M8-BF16 accounting: before={before} after={} vram_total={} vram_free={}",
        memory.usage_bytes(MemoryTier::Vram),
        snapshot.vram_total_bytes,
        snapshot.vram_free_bytes
    );

    // A BF16 tensor of 100_000 elements needs 200_000 bytes: refused by the
    // 8 KiB hard budget before any physical allocation.
    let tight = CudaTensorMemory::new(0, budgets(8 << 10), EvictionPolicy::Lru).unwrap();
    assert!(matches!(
        tight.allocate_tensor(vec![100_000], DType::BF16),
        Err(CudaTensorError::Memory(_))
    ));
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_07_kernel_event_guards_and_explicit_wait() {
    let _serial = serial();
    let memory = memory();
    let a = memory
        .upload_bf16(vec![32, 32], &bf16_values(161, 1024, 2.0))
        .unwrap();
    let b = memory
        .upload_bf16(vec![32, 32], &bf16_values(162, 1024, 2.0))
        .unwrap();
    let output = memory.allocate_tensor(vec![32, 32], DType::BF16).unwrap();
    let a_id = memory.buffer_id(&a).unwrap();
    let b_id = memory.buffer_id(&b).unwrap();
    let out_id = memory.buffer_id(&output).unwrap();

    let launch_tick = memory.tick();
    let token = {
        let mut manager = memory.manager_mut();
        let op = TensorOp::Add { lhs: &a, rhs: &b };
        manager
            .launch_tensor_op(&op, &[a_id.as_str(), b_id.as_str()], &out_id, launch_tick)
            .expect("BF16 kernel launch")
    };
    {
        let manager = memory.manager();
        assert!(manager.allocation(&out_id).is_some());
        let _ = manager.is_transfer_complete(&token).expect("event query");
    }
    memory
        .manager_mut()
        .wait_for_transfer(&token, memory.tick())
        .expect("explicit event wait");
    let actual = memory.download_bf16(&output).unwrap();
    let expected: Vec<Bf16> = bf16_values(161, 1024, 2.0)
        .iter()
        .zip(bf16_values(162, 1024, 2.0).iter())
        .map(|(left, right)| Bf16::from_f32(left.to_f32() + right.to_f32()))
        .collect();
    assert_bf16_close("event-backed BF16 add", &expected, &actual, 0);
    println!(
        "HW-M8-BF16 event guard: token={} allocations={:?}",
        token.id(),
        token.allocation_ids()
    );
}

#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real BF16 GPU kernels"]
fn hw_m8_bf16_08_scheduler_driven_compute_keeps_tensors_out_of_scheduler() {
    let _serial = serial();
    let memory = memory();
    let a = memory
        .upload_bf16(vec![128, 128], &bf16_values(171, 16_384, 1.5))
        .unwrap();
    let b = memory
        .upload_bf16(vec![128, 128], &bf16_values(172, 16_384, 1.5))
        .unwrap();

    let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
    let compute_id = scheduler.submit(TaskSpec::new(TaskKind::Compute)).unwrap();
    let dependent_id = scheduler
        .submit(TaskSpec::new(TaskKind::Compute).depends_on(compute_id))
        .unwrap();
    let mut runner = QueueRunner::new(TaskKind::Compute, CudaTensorComputeExecutor::new(&memory));
    runner
        .executor_mut()
        .register(compute_id, CudaTensorTask::Add { lhs: a, rhs: b })
        .unwrap();
    runner
        .executor_mut()
        .register(
            dependent_id,
            CudaTensorTask::SoftmaxLastDim {
                input: memory
                    .upload_bf16(vec![7], &bf16_values(173, 7, 1.0))
                    .unwrap(),
            },
        )
        .unwrap();

    assert_eq!(
        runner.run_one(&mut scheduler).unwrap(),
        ExecutionOutcome::Completed(compute_id)
    );
    assert_eq!(
        runner.run_one(&mut scheduler).unwrap(),
        ExecutionOutcome::Completed(dependent_id)
    );
    let output = runner
        .executor_mut()
        .take_result(compute_id)
        .expect("scheduler executor returns the completed BF16 tensor");
    let actual = memory.download_bf16(&output).unwrap();
    let expected: Vec<Bf16> = bf16_values(171, 16_384, 1.5)
        .iter()
        .zip(bf16_values(172, 16_384, 1.5).iter())
        .map(|(left, right)| Bf16::from_f32(left.to_f32() + right.to_f32()))
        .collect();
    assert_bf16_close("scheduler-driven BF16 add", &expected, &actual, 0);
    assert_eq!(runner.executor().pending_results(), 1);
    assert_eq!(runner.executor().pending_operations(), 0);
    println!("HW-M8-BF16 scheduler: completed={compute_id:?} dependent={dependent_id:?}");
}

/// B1: a non-zero, element-aligned `Tensor::offset_bytes` must be applied to
/// every pointer handed to the CUDA kernels, for both dtypes. Sentinels before
/// the tensor span detect a kernel that ignored the offset and wrote/read the
/// beginning of the allocation.
#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_b1_input_offsets_are_honored_for_f32_and_bf16() {
    let _serial = serial();
    let mut manager = CudaMemoryManager::new(0, budgets(1 << 30), EvictionPolicy::Lru).unwrap();
    let cpu = CpuDevice::new();
    let mut tick = 1u64;

    // ---- F32: 8-byte offset in a 24-byte allocation (two leading floats) ----
    let offset = 8usize;
    let total = 24usize;
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let sentinel = [0x5Au8; 24];

    manager.allocate_pinned("h_in", total as u64, tick).unwrap();
    tick += 1;
    manager
        .write_pinned("h_in", offset, &a_bytes, tick)
        .unwrap();
    tick += 1;
    manager
        .allocate_device("a", total as u64, tick, false)
        .unwrap();
    tick += 1;
    let token = manager.copy_h2d("h_in", "a", total, tick).unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    manager
        .write_pinned("h_in", offset, &b_bytes, tick)
        .unwrap();
    tick += 1;
    manager
        .allocate_device("b", total as u64, tick, false)
        .unwrap();
    tick += 1;
    let token = manager.copy_h2d("h_in", "b", total, tick).unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    manager.write_pinned("h_in", 0, &sentinel, tick).unwrap();
    tick += 1;
    manager
        .allocate_device("out_f32", total as u64, tick, false)
        .unwrap();
    tick += 1;
    let token = manager.copy_h2d("h_in", "out_f32", total, tick).unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    let layout = TensorLayout::contiguous(vec![4], DType::F32).unwrap();
    let ta = Tensor::with_layout(
        cpu.allocate(MemoryTier::Host, total as u64).unwrap(),
        layout.clone(),
        offset,
    )
    .unwrap();
    let tb = Tensor::with_layout(
        cpu.allocate(MemoryTier::Host, total as u64).unwrap(),
        layout,
        offset,
    )
    .unwrap();
    let op = TensorOp::Add { lhs: &ta, rhs: &tb };
    let launch_tick = tick;
    tick += 1;
    let token = manager
        .launch_tensor_op(&op, &["a", "b"], "out_f32", launch_tick)
        .expect("offset F32 launch");
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    manager
        .allocate_pinned("h_res", total as u64, tick)
        .unwrap();
    tick += 1;
    let token = manager.copy_d2h("out_f32", "h_res", total, tick).unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;
    let bytes = manager.read_pinned("h_res", 0, total, tick).unwrap();
    tick += 1;
    // `launch_tensor_op` has no output-offset argument: outputs are written at
    // the base of their allocation. The inputs must be read from their tensor
    // offsets; ignoring them would read the 0x5A sentinel bytes instead.
    let actual: Vec<f32> = bytes[..16]
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(&bytes[16..], &sentinel[16..], "F32 trailing sentinel");
    let expected: Vec<f32> = a.iter().zip(b).map(|(left, right)| left + right).collect();
    assert_eq!(actual, expected, "F32 add at offset {offset}");
    println!("HW-M8-B1 F32 offset={offset} actual={actual:?}");

    // ---- BF16: 2-byte offset (one element) in a 10-byte allocation --------
    let bf_offset = 2usize;
    let bf_total = 10usize;
    let bf_a: Vec<Bf16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .copied()
        .map(Bf16::from_f32)
        .collect();
    let bf_b: Vec<Bf16> = [0.5f32, 1.5, 2.5, 3.5]
        .iter()
        .copied()
        .map(Bf16::from_f32)
        .collect();
    let bf_a_bytes: Vec<u8> = bf_a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let bf_b_bytes: Vec<u8> = bf_b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let bf_sentinel = [0xDDu8; 10];
    manager
        .allocate_pinned("h_bf", bf_total as u64, tick)
        .unwrap();
    tick += 1;
    manager
        .write_pinned("h_bf", bf_offset, &bf_a_bytes, tick)
        .unwrap();
    tick += 1;
    manager
        .allocate_device("bf_a", bf_total as u64, tick, false)
        .unwrap();
    tick += 1;
    let token = manager.copy_h2d("h_bf", "bf_a", bf_total, tick).unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    manager
        .write_pinned("h_bf", bf_offset, &bf_b_bytes, tick)
        .unwrap();
    tick += 1;
    manager
        .allocate_device("bf_b", bf_total as u64, tick, false)
        .unwrap();
    tick += 1;
    let token = manager.copy_h2d("h_bf", "bf_b", bf_total, tick).unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    manager.write_pinned("h_bf", 0, &bf_sentinel, tick).unwrap();
    tick += 1;
    manager
        .allocate_device("out_bf16", bf_total as u64, tick, false)
        .unwrap();
    tick += 1;
    let token = manager
        .copy_h2d("h_bf", "out_bf16", bf_total, tick)
        .unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    let bf_layout = TensorLayout::contiguous(vec![4], DType::BF16).unwrap();
    let bfa = Tensor::with_layout(
        cpu.allocate(MemoryTier::Host, bf_total as u64).unwrap(),
        bf_layout.clone(),
        bf_offset,
    )
    .unwrap();
    let bfb = Tensor::with_layout(
        cpu.allocate(MemoryTier::Host, bf_total as u64).unwrap(),
        bf_layout,
        bf_offset,
    )
    .unwrap();
    let bf_op = TensorOp::Add {
        lhs: &bfa,
        rhs: &bfb,
    };
    let launch_tick = tick;
    tick += 1;
    let token = manager
        .launch_tensor_op(&bf_op, &["bf_a", "bf_b"], "out_bf16", launch_tick)
        .expect("offset BF16 launch");
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;

    manager
        .allocate_pinned("h_bf_res", bf_total as u64, tick)
        .unwrap();
    tick += 1;
    let token = manager
        .copy_d2h("out_bf16", "h_bf_res", bf_total, tick)
        .unwrap();
    tick += 1;
    manager.wait_for_transfer(&token, tick).unwrap();
    tick += 1;
    let bytes = manager.read_pinned("h_bf_res", 0, bf_total, tick).unwrap();
    tick += 1;
    let actual: Vec<Bf16> = bytes[..8]
        .chunks_exact(2)
        .map(|chunk| Bf16::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(&bytes[8..], &bf_sentinel[8..], "BF16 trailing sentinel");
    let expected: Vec<Bf16> = bf_a
        .iter()
        .zip(&bf_b)
        .map(|(left, right)| Bf16::from_f32(left.to_f32() + right.to_f32()))
        .collect();
    assert_eq!(actual, expected, "BF16 add at offset {bf_offset}");
    println!("HW-M8-B1 BF16 offset={bf_offset} actual={actual:?}");

    for id in [
        "a", "b", "out_f32", "bf_a", "bf_b", "out_bf16", "h_in", "h_out", "h_res", "h_bf",
        "h_bf_res",
    ] {
        let _ = manager.release(id, tick);
    }
}

/// B1: byte spans and capacities are checked before any launch, on both the
/// input and output side; misaligned offsets are rejected at tensor level.
#[test]
#[ignore = "requires a CUDA driver, an RTX 5090 and real GPU kernels"]
fn hw_m8_b1_byte_span_and_capacity_guards() {
    let _serial = serial();
    let mut manager = CudaMemoryManager::new(0, budgets(1 << 30), EvictionPolicy::Lru).unwrap();
    let cpu = CpuDevice::new();
    let mut tick = 1u64;

    manager
        .allocate_device("short_input", 8, tick, false)
        .unwrap();
    tick += 1;
    manager.allocate_device("out", 16, tick, false).unwrap();
    tick += 1;
    manager
        .allocate_device("short_output", 8, tick, false)
        .unwrap();
    tick += 1;

    let f32_layout = TensorLayout::contiguous(vec![4], DType::F32).unwrap();
    let input = Tensor::with_layout(
        cpu.allocate(MemoryTier::Host, 16).unwrap(),
        f32_layout.clone(),
        0,
    )
    .unwrap();
    let other = Tensor::with_layout(
        cpu.allocate(MemoryTier::Host, 16).unwrap(),
        f32_layout.clone(),
        0,
    )
    .unwrap();

    let input_span = TensorOp::SoftmaxLastDim { input: &input };
    assert!(matches!(
        manager.launch_tensor_op(&input_span, &["short_input"], "out", tick),
        Err(claymore_memory::CudaMemoryError::Cuda(
            claymore_cuda::CudaError::BufferTooSmall {
                requested: 16,
                capacity: 8
            }
        ))
    ));
    tick += 1;

    let add = TensorOp::Add {
        lhs: &input,
        rhs: &other,
    };
    assert!(matches!(
        manager.launch_tensor_op(&add, &["out", "out"], "short_output", tick),
        Err(claymore_memory::CudaMemoryError::Cuda(
            claymore_cuda::CudaError::BufferTooSmall {
                requested: 16,
                capacity: 8
            }
        ))
    ));
    tick += 1;

    let storage = cpu.allocate(MemoryTier::Host, 24).unwrap();
    assert_eq!(
        Tensor::with_layout(storage, f32_layout, 3).unwrap_err(),
        TensorError::MisalignedOffset {
            offset: 3,
            alignment: 4
        }
    );

    for id in ["short_input", "out", "short_output"] {
        manager.release(id, tick).unwrap();
    }
    println!("HW-M8-B1 span/capacity guards rejected short input and output allocations");
}
