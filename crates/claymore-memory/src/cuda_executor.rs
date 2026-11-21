//! CUDA event-backed task executor for the M7 Transfer/Prefetch queues.
//!
//! Construct and use this executor on the CUDA context-owner thread. Every
//! asynchronous H2D/D2H operation is waited to its real CUDA event before
//! returning success to QueueRunner, so scheduler dependents are not released
//! while DMA may still reference either allocation.
use std::collections::BTreeMap;
use std::time::Instant;

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

/// CUDA transfer counters and timings, including time spent waiting for events.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaTransferMetrics {
    pub allocations: u64,
    pub releases: u64,
    pub h2d_transfers: u64,
    pub d2h_transfers: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub h2d_total_ns: u64,
    pub d2h_total_ns: u64,
    pub pinned_read_bytes: u64,
    pub event_wait_ns: u64,
    pub failed: u64,
    pub total_execution_ns: u64,
    pub max_execution_ns: u64,
}

/// Drives CUDA memory tasks synchronously on the thread that owns the CUDA
/// primary context. It intentionally has no Send/Sync implementation.
pub struct CudaTransferExecutor {
    manager: CudaMemoryManager,
    operations: BTreeMap<TaskId, CudaMemoryTask>,
    read_results: BTreeMap<TaskId, Vec<u8>>,
    logical_tick: u64,
    metrics: CudaTransferMetrics,
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
            metrics: CudaTransferMetrics::default(),
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

    pub fn metrics(&self) -> CudaTransferMetrics {
        self.metrics
    }

    fn next_tick(&mut self) -> Result<u64, CudaTransferExecutorError> {
        self.logical_tick = self
            .logical_tick
            .checked_add(1)
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
                self.metrics.allocations = self.metrics.allocations.saturating_add(1);
            }
            CudaMemoryTask::AllocatePinned { id, bytes } => {
                let tick = self.next_tick()?;
                self.manager.allocate_pinned(id, bytes, tick)?;
                self.metrics.allocations = self.metrics.allocations.saturating_add(1);
            }
            CudaMemoryTask::WritePinned { id, offset, data } => {
                let tick = self.next_tick()?;
                self.manager.write_pinned(&id, offset, &data, tick)?;
            }
            CudaMemoryTask::CopyH2d { pinned_id, device_id, bytes } => {
                let transfer_started = Instant::now();
                let enqueue_tick = self.next_tick()?;
                let token = self.manager.copy_h2d(&pinned_id, &device_id, bytes, enqueue_tick)?;
                let wait_tick = self.next_tick()?;
                let wait_started = Instant::now();
                self.manager.wait_for_transfer(&token, wait_tick)?;
                self.metrics.event_wait_ns = self.metrics.event_wait_ns.saturating_add(
                    wait_started.elapsed().as_nanos().min(u64::MAX as u128) as u64
                );
                self.metrics.h2d_transfers = self.metrics.h2d_transfers.saturating_add(1);
                self.metrics.h2d_bytes = self.metrics.h2d_bytes.saturating_add(bytes as u64);
                self.metrics.h2d_total_ns = self.metrics.h2d_total_ns.saturating_add(
                    transfer_started.elapsed().as_nanos().min(u64::MAX as u128) as u64
                );
            }
            CudaMemoryTask::CopyD2h { device_id, pinned_id, bytes } => {
                let transfer_started = Instant::now();
                let enqueue_tick = self.next_tick()?;
                let token = self.manager.copy_d2h(&device_id, &pinned_id, bytes, enqueue_tick)?;
                let wait_tick = self.next_tick()?;
                let wait_started = Instant::now();
                self.manager.wait_for_transfer(&token, wait_tick)?;
                self.metrics.event_wait_ns = self.metrics.event_wait_ns.saturating_add(
                    wait_started.elapsed().as_nanos().min(u64::MAX as u128) as u64
                );
                self.metrics.d2h_transfers = self.metrics.d2h_transfers.saturating_add(1);
                self.metrics.d2h_bytes = self.metrics.d2h_bytes.saturating_add(bytes as u64);
                self.metrics.d2h_total_ns = self.metrics.d2h_total_ns.saturating_add(
                    transfer_started.elapsed().as_nanos().min(u64::MAX as u128) as u64
                );
            }
            CudaMemoryTask::ReadPinned { id, offset, len } => {
                let tick = self.next_tick()?;
                let bytes = self.manager.read_pinned(&id, offset, len, tick)?;
                self.metrics.pinned_read_bytes = self
                    .metrics
                    .pinned_read_bytes
                    .saturating_add(bytes.len() as u64);
                self.read_results.insert(task_id, bytes);
            }
            CudaMemoryTask::Release { id } => {
                let tick = self.next_tick()?;
                self.manager.release(&id, tick)?;
                self.metrics.releases = self.metrics.releases.saturating_add(1);
            }
        }
        Ok(())
    }
}

impl TaskExecutor for CudaTransferExecutor {
    fn execute(&mut self, task: &TaskSnapshot) -> Result<(), String> {
        if !matches!(task.kind, TaskKind::Transfer | TaskKind::Prefetch) {
            self.metrics.failed = self.metrics.failed.saturating_add(1);
            return Err(CudaTransferExecutorError::WrongQueue {
                task_id: task.id,
                kind: task.kind,
            }.to_string());
        }
        let Some(operation) = self.operations.remove(&task.id) else {
            self.metrics.failed = self.metrics.failed.saturating_add(1);
            return Err(CudaTransferExecutorError::MissingTask(task.id).to_string());
        };
        let started_at = Instant::now();
        let result = self.execute_operation(task.id, operation);
        let elapsed_ns = started_at.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.metrics.total_execution_ns = self
            .metrics
            .total_execution_ns
            .saturating_add(elapsed_ns);
        self.metrics.max_execution_ns = self.metrics.max_execution_ns.max(elapsed_ns);
        if result.is_err() {
            self.metrics.failed = self.metrics.failed.saturating_add(1);
        }
        result.map_err(|error| error.to_string())
    }
}
