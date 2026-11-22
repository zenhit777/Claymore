//! CUDA tensor execution integrated with the Claymore memory policy.
//!
//! This adapter is the only M8 path that allocates VRAM tensor outputs. Every
//! allocation goes through `CudaMemoryManager::allocate_device`, which applies
//! the configured admission/eviction policy and the driver's free-memory check.
//! Kernel work is tracked by real CUDA events through the same pending-work
//! tracker used for H2D/D2H transfers, so buffers stay protected until the
//! event has been waited on. Tensors remain owned by the caller; the scheduler
//! only ever sees task ids and descriptors, never tensor values.

use std::any::Any;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::collections::BTreeMap;
use std::rc::Rc;

use claymore_core::MemoryTier;
use claymore_cuda::CudaDeviceInfo;
use claymore_device::{DType, DeviceBuffer, DeviceError, DeviceInstanceId};
use claymore_scheduler::{TaskExecutor, TaskId, TaskKind, TaskSnapshot};
use claymore_tensor::{
    AliasRule, BackendCapabilities, LayoutRequirement, OpError, OpKind, OpSupport, Tensor,
    TensorComputeBackend, TensorLayout, TensorOp,
};
use thiserror::Error;

use crate::{CudaMemoryError, CudaMemoryManager, EvictionPolicy, MemoryBudgets, MemorySnapshot};

#[derive(Debug, Error)]
pub enum CudaTensorError {
    #[error(transparent)]
    Memory(#[from] CudaMemoryError),
    #[error(transparent)]
    Tensor(#[from] claymore_tensor::TensorError),
    #[error(transparent)]
    Op(#[from] OpError),
    #[error("only F32 tensors are implemented on the CUDA backend")]
    UnsupportedDType,
    #[error("tensor storage is not a CUDA-managed buffer")]
    ForeignBuffer,
    #[error("tensor inputs belong to a different CUDA memory owner")]
    ForeignOwner,
    #[error("float byte count does not match the tensor layout")]
    ValueCountMismatch,
}

struct CudaTensorInner {
    manager: RefCell<CudaMemoryManager>,
    next_tick: Cell<u64>,
    next_id: Cell<u64>,
    device_id: String,
    instance_id: DeviceInstanceId,
    device_info: CudaDeviceInfo,
}

impl CudaTensorInner {
    fn tick(&self) -> u64 {
        let tick = self.next_tick.get() + 1;
        self.next_tick.set(tick);
        tick
    }

    fn id(&self, purpose: &str) -> String {
        let sequence = self.next_id.get() + 1;
        self.next_id.set(sequence);
        format!("m8-{purpose}-{sequence}")
    }
}

/// Owner-thread handle to CUDA tensor allocations. It is deliberately
/// `!Send + !Sync`: CUDA primary-context resources must stay on the thread
/// that created them.
pub struct CudaTensorMemory {
    inner: Rc<CudaTensorInner>,
}

impl Clone for CudaTensorMemory {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl CudaTensorMemory {
    pub fn new(
        ordinal: u32,
        budgets: MemoryBudgets,
        eviction: EvictionPolicy,
    ) -> Result<Self, CudaTensorError> {
        let manager = CudaMemoryManager::new(ordinal, budgets, eviction)?;
        let device_info = manager.device_info().clone();
        let device_id = device_info.name.clone();
        Ok(Self {
            inner: Rc::new(CudaTensorInner {
                manager: RefCell::new(manager),
                next_tick: Cell::new(0),
                next_id: Cell::new(0),
                device_id,
                instance_id: DeviceInstanceId::new(),
                device_info,
            }),
        })
    }

    /// Next monotonic logical tick shared by this memory owner and every
    /// tensor buffer it manages. Callers driving low-level launch/wait APIs
    /// must use this clock instead of inventing ticks.
    pub fn tick(&self) -> u64 {
        self.inner.tick()
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.inner.device_info
    }

    /// Identity of the policy/backend allocation backing a managed tensor.
    pub fn buffer_id(&self, tensor: &Tensor) -> Result<String, CudaTensorError> {
        Ok(managed(tensor)?.id.clone())
    }

    pub fn usage_bytes(&self, tier: MemoryTier) -> u64 {
        self.inner.manager.borrow().usage_bytes(tier)
    }

    pub fn snapshot(
        &self,
        host_total_bytes: u64,
        host_available_bytes: u64,
    ) -> Result<MemorySnapshot, CudaTensorError> {
        Ok(self
            .inner
            .manager
            .borrow()
            .snapshot(host_total_bytes, host_available_bytes)?)
    }

    pub fn manager(&self) -> Ref<'_, CudaMemoryManager> {
        self.inner.manager.borrow()
    }

    pub fn manager_mut(&self) -> RefMut<'_, CudaMemoryManager> {
        self.inner.manager.borrow_mut()
    }

    /// Allocate a fresh VRAM-backed tensor through the memory policy.
    pub fn allocate_tensor(
        &self,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Tensor, CudaTensorError> {
        if dtype != DType::F32 {
            return Err(CudaTensorError::UnsupportedDType);
        }
        let layout = TensorLayout::contiguous(shape, dtype)?;
        let bytes =
            u64::try_from(layout.span_bytes()).map_err(|_| CudaMemoryError::ZeroSizedAllocation)?;
        let id = self.inner.id("vram");
        let mut manager = self.inner.manager.borrow_mut();
        manager.allocate_device(id.clone(), bytes, self.inner.tick(), false)?;
        let storage = ManagedCudaBuffer {
            inner: Rc::clone(&self.inner),
            id,
            len: layout.span_bytes(),
            device_id: self.inner.device_id.clone(),
            instance_id: self.inner.instance_id,
            released: false,
        };
        Ok(Tensor::with_layout(Box::new(storage), layout, 0)?)
    }

    /// Upload explicit F32 values through pinned staging and a real H2D event.
    pub fn upload_f32(&self, shape: Vec<usize>, values: &[f32]) -> Result<Tensor, CudaTensorError> {
        let tensor = self.allocate_tensor(shape, DType::F32)?;
        let expected = tensor.layout().numel();
        if values.len() != expected {
            return Err(CudaTensorError::ValueCountMismatch);
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let host_id = self.inner.id("h2d");
        let byte_len =
            u64::try_from(bytes.len()).map_err(|_| CudaMemoryError::ZeroSizedAllocation)?;
        {
            let mut manager = self.inner.manager.borrow_mut();
            manager.allocate_pinned(host_id.clone(), byte_len, self.inner.tick())?;
            manager.write_pinned(&host_id, 0, &bytes, self.inner.tick())?;
            let device_id = managed(&tensor)?.id.clone();
            let token = manager.copy_h2d(&host_id, &device_id, bytes.len(), self.inner.tick())?;
            manager.wait_for_transfer(&token, self.inner.tick())?;
            manager.release(&host_id, self.inner.tick())?;
        }
        Ok(tensor)
    }

    /// Read tensor values back through pinned staging. The D2H event is waited
    /// on before the bytes are returned.
    pub fn download_f32(&self, tensor: &Tensor) -> Result<Vec<f32>, CudaTensorError> {
        let buffer = managed(tensor)?;
        let byte_len = tensor.layout().span_bytes();
        let host_id = self.inner.id("d2h");
        let bytes = {
            let mut manager = self.inner.manager.borrow_mut();
            manager.allocate_pinned(
                host_id.clone(),
                u64::try_from(byte_len).map_err(|_| CudaMemoryError::ZeroSizedAllocation)?,
                self.inner.tick(),
            )?;
            let token = manager.copy_d2h(&buffer.id, &host_id, byte_len, self.inner.tick())?;
            manager.wait_for_transfer(&token, self.inner.tick())?;
            let bytes = manager.read_pinned(&host_id, 0, byte_len, self.inner.tick())?;
            manager.release(&host_id, self.inner.tick())?;
            bytes
        };
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
            .collect())
    }
}

/// VRAM allocation owned by `CudaTensorMemory`. Releasing the buffer asks the
/// manager to wait for outstanding events, free the physical allocation and
/// only then remove the policy accounting entry.
struct ManagedCudaBuffer {
    inner: Rc<CudaTensorInner>,
    id: String,
    len: usize,
    device_id: String,
    instance_id: DeviceInstanceId,
    released: bool,
}

impl DeviceBuffer for ManagedCudaBuffer {
    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn device_instance_id(&self) -> DeviceInstanceId {
        self.instance_id
    }

    fn tier(&self) -> MemoryTier {
        MemoryTier::Vram
    }

    fn len(&self) -> usize {
        self.len
    }

    fn release(&mut self) -> Result<(), DeviceError> {
        if self.released {
            return Err(DeviceError::BufferReleased);
        }
        let tick = self.inner.tick();
        self.inner
            .manager
            .borrow_mut()
            .release(&self.id, tick)
            .map_err(|error| DeviceError::ReleaseFailed(error.to_string()))?;
        self.released = true;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Drop for ManagedCudaBuffer {
    fn drop(&mut self) {
        // Best effort only. If the manager reports the allocation still busy,
        // it keeps ownership and the guard; the manager's own Drop path handles
        // final synchronization without ever freeing in-flight memory.
        let _ = self.release();
    }
}

fn managed(tensor: &Tensor) -> Result<&ManagedCudaBuffer, CudaTensorError> {
    tensor
        .storage()
        .as_any()
        .downcast_ref::<ManagedCudaBuffer>()
        .ok_or(CudaTensorError::ForeignBuffer)
}

const CUDA_F32_OPS: [OpSupport; 4] = [
    OpSupport {
        kind: OpKind::Add,
        dtype: DType::F32,
        input_tier: MemoryTier::Vram,
        output_tier: MemoryTier::Vram,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::MatMul,
        dtype: DType::F32,
        input_tier: MemoryTier::Vram,
        output_tier: MemoryTier::Vram,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::SoftmaxLastDim,
        dtype: DType::F32,
        input_tier: MemoryTier::Vram,
        output_tier: MemoryTier::Vram,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::RmsNorm,
        dtype: DType::F32,
        input_tier: MemoryTier::Vram,
        output_tier: MemoryTier::Vram,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
];

/// Native CUDA backend for the M8 operation set. It never reads host bytes
/// implicitly: inputs must already be VRAM tensors from the same memory owner.
pub struct CudaTensorBackend {
    memory: CudaTensorMemory,
    inner: Rc<CudaTensorInner>,
}

impl CudaTensorBackend {
    pub fn new(memory: &CudaTensorMemory) -> Self {
        Self {
            memory: memory.clone(),
            inner: Rc::clone(&memory.inner),
        }
    }
}

impl TensorComputeBackend for CudaTensorBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "cuda-native-f32",
            device_kind: claymore_device::DeviceKind::Cuda,
            supported_ops: &CUDA_F32_OPS,
        }
    }

    fn execute(&self, op: &TensorOp<'_>) -> Result<Tensor, OpError> {
        let validated = op.validate()?;
        let backend_error = |error: CudaTensorError| OpError::Backend {
            backend: "cuda-native-f32",
            reason: error.to_string(),
        };
        let mut input_ids = Vec::with_capacity(op.inputs().len());
        for input in op.inputs() {
            let buffer = managed(input).map_err(backend_error)?;
            if !Rc::ptr_eq(&buffer.inner, &self.inner) {
                return Err(OpError::MixedDevices);
            }
            input_ids.push(buffer.id.clone());
        }
        let output = self
            .memory
            .allocate_tensor(validated.shape, DType::F32)
            .map_err(backend_error)?;
        let output_id = managed(&output).map_err(backend_error)?.id.clone();
        let input_refs: Vec<&str> = input_ids.iter().map(String::as_str).collect();
        let result = (|| -> Result<(), CudaTensorError> {
            let mut manager = self.inner.manager.borrow_mut();
            let launch_tick = self.inner.tick();
            let token = manager.launch_tensor_op(op, &input_refs, &output_id, launch_tick)?;
            let wait_tick = self.inner.tick();
            manager.wait_for_transfer(&token, wait_tick)?;
            Ok(())
        })();
        if let Err(error) = result {
            // Dropping the output releases the allocation through the manager.
            return Err(backend_error(error));
        }
        Ok(output)
    }
}

/// Payload stored by the scheduler-facing executor. Tensors are owned here,
/// never by `Scheduler`, which continues to see only task ids and states.
pub enum CudaTensorTask {
    Add {
        lhs: Tensor,
        rhs: Tensor,
    },
    MatMul {
        lhs: Tensor,
        rhs: Tensor,
    },
    SoftmaxLastDim {
        input: Tensor,
    },
    RmsNorm {
        input: Tensor,
        scale: Tensor,
        epsilon: f32,
    },
}

impl CudaTensorTask {
    pub fn as_op(&self) -> TensorOp<'_> {
        match self {
            CudaTensorTask::Add { lhs, rhs } | CudaTensorTask::MatMul { lhs, rhs } => {
                if matches!(self, CudaTensorTask::Add { .. }) {
                    TensorOp::Add { lhs, rhs }
                } else {
                    TensorOp::MatMul { lhs, rhs }
                }
            }
            CudaTensorTask::SoftmaxLastDim { input } => TensorOp::SoftmaxLastDim { input },
            CudaTensorTask::RmsNorm {
                input,
                scale,
                epsilon,
            } => TensorOp::RmsNorm {
                input,
                scale,
                epsilon: *epsilon,
            },
        }
    }
}

/// Runs registered tensor operations from a `Compute` queue runner. A task is
/// reported complete only after the CUDA backend has waited on the real event
/// protecting every input and output allocation.
pub struct CudaTensorComputeExecutor {
    backend: CudaTensorBackend,
    operations: BTreeMap<TaskId, CudaTensorTask>,
    results: BTreeMap<TaskId, Tensor>,
}

impl CudaTensorComputeExecutor {
    pub fn new(memory: &CudaTensorMemory) -> Self {
        Self {
            backend: CudaTensorBackend::new(memory),
            operations: BTreeMap::new(),
            results: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, task_id: TaskId, task: CudaTensorTask) -> Result<(), String> {
        if self.operations.contains_key(&task_id) || self.results.contains_key(&task_id) {
            return Err(format!("duplicate tensor task {task_id:?}"));
        }
        self.operations.insert(task_id, task);
        Ok(())
    }

    pub fn take_result(&mut self, task_id: TaskId) -> Option<Tensor> {
        self.results.remove(&task_id)
    }

    pub fn pending_operations(&self) -> usize {
        self.operations.len()
    }

    pub fn pending_results(&self) -> usize {
        self.results.len()
    }
}

impl TaskExecutor for CudaTensorComputeExecutor {
    fn execute(&mut self, task: &TaskSnapshot) -> Result<(), String> {
        if task.kind != TaskKind::Compute {
            return Err(format!(
                "tensor task {:?} requires the Compute queue",
                task.id
            ));
        }
        let payload = self
            .operations
            .remove(&task.id)
            .ok_or_else(|| format!("no tensor payload for {:?}", task.id))?;
        let output = self
            .backend
            .execute(&payload.as_op())
            .map_err(|error| error.to_string())?;
        self.results.insert(task.id, output);
        Ok(())
    }

    fn discard(&mut self, task_id: TaskId) {
        self.operations.remove(&task_id);
        self.results.remove(&task_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_capabilities_declare_only_implemented_f32_vram_ops() {
        assert_eq!(CUDA_F32_OPS.len(), 4);
        for support in CUDA_F32_OPS {
            assert_eq!(support.dtype, DType::F32);
            assert_eq!(support.input_tier, MemoryTier::Vram);
            assert_eq!(support.output_tier, MemoryTier::Vram);
            assert_eq!(support.layout, LayoutRequirement::Contiguous);
            assert_eq!(support.alias_rule, AliasRule::ReadOnlyInputsFreshOutput);
        }
        let kinds: Vec<OpKind> = CUDA_F32_OPS.iter().map(|support| support.kind).collect();
        assert_eq!(
            kinds,
            vec![
                OpKind::Add,
                OpKind::MatMul,
                OpKind::SoftmaxLastDim,
                OpKind::RmsNorm
            ]
        );
    }
}
