//! CUDA event-backed task executor for the M7 Transfer/Prefetch queues.
//!
//! Construct and use this executor on the CUDA context-owner thread. Every
//! asynchronous H2D/D2H operation is waited to its real CUDA event before
//! returning success to QueueRunner, so scheduler dependents are not released
//! while DMA may still reference either allocation.
use std::collections::BTreeMap;

use claymore_core::MemoryTier;
use claymore_scheduler::{TaskExecutor, TaskId, TaskKind, TaskSnapshot};

use crate::{CudaMemoryError, CudaMemoryManager, EvictionPolicy, MemoryBudgets};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaMemoryTask {
    AllocateDevice { id: String, bytes: u64, expert: bool },
    AllocatePinned { id: String, bytes: u64 },
    WritePinned { id: String, offset: usize, data: Vec<u8> },
    CopyH2d { pinned_id: String, device_id: String, bytes: usize },
    CopyD2h { device_id: String, pinned_id: String, bytes: usize },
    ReadPinned { id: String, offset: usize, len: usize },
    Release { id: String },
}

#[derive(Debug, thiserror::Error)]
pub enum CudaTransferExecutorError {
    #[error("task {0:?} already has a registered CUDA memory operation")]
    DuplicateTask(TaskId),
    #[error("task {0:?} has no registered CUDA memory operation")]
    MissingTask(TaskId),
    #[error("CUDA memory task {task_id:?} must use Transfer or Prefetch queue, not {kind:?}")]
    WrongQueue { task_id: TaskId, kind: TaskKind },
    #[error("CUDA executor logical tick space exhausted")]
    TickExhausted,
    #[error(transparent)]
    Memory(#[from] CudaMemoryError),
}

/// Drives CUDA memory tasks synchronously on the thread that owns the CUDA
/// primary context. It intentionally has no Send/Sync implementation.
pub struct CudaTransferExecutor {
    manager: CudaMemoryManager,
    operations: BTreeMap<TaskId, CudaMemoryTask>,
    read_results: BTreeMap<TaskId, Vec<u8>>,
    logical_tick: u64,
}

impl CudaTransferExecutor {
    pub fn new(
        ordinal: u32,
        budgets: MemoryBudgets,
        eviction: EvictionPolicy,
    ) -> Result<Self, CudaTransferExecutorError> {
        Ok(Self {
            manager: CudaMemoryManager::new(ordinal, budgets, eviction)?,
            operations: BTreeMap::new(),
            read_results: BTreeMap::new(),
            logical_tick: 0,
        })
    }

    pub fn register(
        &mut self,
        task_id: TaskId,
        operation: CudaMemoryTask,
    ) -> Result<(), CudaTransferExecutorError> {
        if self.operations.contains_key(&task_id) {
            return Err(CudaTransferExecutorError::DuplicateTask(task_id));
        }
        self.operations.insert(task_id, operation);
        Ok(())
    }

    pub fn manager(&self) -> &CudaMemoryManager {
        &self.manager
    }

    pub fn take_read_result(&mut self, task_id: TaskId) -> Option<Vec<u8>> {
        self.read_results.remove(&task_id)
    }

    pub fn pending_operations(&self) -> usize {
        self.operations.len()
    }

    fn next_tick(&mut self) -> Result<u64, CudaTransferExecutorError> {
        self.logical_tick = self.logical_tick.checked_add(1)
            .ok_or(CudaTransferExecutorError::TickExhausted)?;
        Ok(self.logical_tick)
    }

    fn execute_operation(
        &mut self,
        task_id: TaskId,
        operation: CudaMemoryTask,
    ) -> Result<(), CudaTransferExecutorError> {
        match operation {
            CudaMemoryTask::AllocateDevice { id, bytes, expert } => {
                let tick = self.next_tick()?;
                self.manager.allocate_device(id, bytes, tick, expert)?;
            }
            CudaMemoryTask::AllocatePinned { id, bytes } => {
                let tick = self.next_tick()?;
                self.manager.allocate_pinned(id, bytes, tick)?;
            }
            CudaMemoryTask::WritePinned { id, offset, data } => {
                let tick = self.next_tick()?;
                self.manager.write_pinned(&id, offset, &data, tick)?;
            }
            CudaMemoryTask::CopyH2d { pinned_id, device_id, bytes } => {
                let enqueue_tick = self.next_tick()?;
                let token = self.manager.copy_h2d(&pinned_id, &device_id, bytes, enqueue_tick)?;
                let wait_tick = self.next_tick()?;
                self.manager.wait_for_transfer(&token, wait_tick)?;
            }
            CudaMemoryTask::CopyD2h { device_id, pinned_id, bytes } => {
                let enqueue_tick = self.next_tick()?;
                let token = self.manager.copy_d2h(&device_id, &pinned_id, bytes, enqueue_tick)?;
                let wait_tick = self.next_tick()?;
                self.manager.wait_for_transfer(&token, wait_tick)?;
            }
            CudaMemoryTask::ReadPinned { id, offset, len } => {
                let tick = self.next_tick()?;
                let bytes = self.manager.read_pinned(&id, offset, len, tick)?;
                self.read_results.insert(task_id, bytes);
            }
            CudaMemoryTask::Release { id } => {
                let tick = self.next_tick()?;
                self.manager.release(&id, tick)?;
            }
        }
        Ok(())
    }
}

impl TaskExecutor for CudaTransferExecutor {
    fn execute(&mut self, task: &TaskSnapshot) -> Result<(), String> {
        if !matches!(task.kind, TaskKind::Transfer | TaskKind::Prefetch) {
            return Err(CudaTransferExecutorError::WrongQueue {
                task_id: task.id,
                kind: task.kind,
            }.to_string());
        }
        let operation = self.operations.remove(&task.id)
            .ok_or(CudaTransferExecutorError::MissingTask(task.id))
            .map_err(|error| error.to_string())?;
        self.execute_operation(task.id, operation)
            .map_err(|error| error.to_string())
    }
}

// Keep the memory tier import in this feature module's public dependency
// surface explicit: CUDA device allocations are VRAM and pinned allocations
// are PinnedHost; no generic Device trait implementation is implied.
const _: Option<MemoryTier> = None;
