//! Synchronous memory operations driven by the M7 task scheduler.
//!
//! This adapter is backend-neutral and can be exercised with CpuDevice. It
//! deliberately completes each operation before returning to QueueRunner;
//! asynchronous CUDA operations require their own event-backed executor.
use std::collections::BTreeMap;

use claymore_core::MemoryTier;
use claymore_device::Device;
use claymore_scheduler::{TaskExecutor, TaskId, TaskKind, TaskSnapshot};

use crate::{DeviceMemoryError, DeviceMemoryManager};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryTask {
    Allocate {
        allocation_id: String,
        tier: MemoryTier,
        bytes: u64,
        expert: bool,
    },
    Write {
        allocation_id: String,
        offset: usize,
        data: Vec<u8>,
    },
    Copy {
        source_id: String,
        destination_id: String,
        bytes: usize,
    },
    Release {
        allocation_id: String,
    },
    RecordReloadCost {
        allocation_id: String,
        estimated_ns: u64,
    },
    Synchronize,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryTaskExecutorError {
    #[error("task {0:?} already has a registered memory operation")]
    DuplicateTask(TaskId),
    #[error("task {0:?} has no registered memory operation")]
    MissingTask(TaskId),
    #[error("memory operation task {task_id:?} must use Transfer or Prefetch queue, not {kind:?}")]
    WrongQueue { task_id: TaskId, kind: TaskKind },
    #[error("memory executor logical tick space exhausted")]
    TickExhausted,
    #[error(transparent)]
    Memory(#[from] DeviceMemoryError),
}

/// Maps scheduler task IDs to concrete backend-neutral memory operations.
pub struct MemoryTaskExecutor<D: Device> {
    manager: DeviceMemoryManager<D>,
    operations: BTreeMap<TaskId, MemoryTask>,
    logical_tick: u64,
}

impl<D: Device> MemoryTaskExecutor<D> {
    pub fn new(manager: DeviceMemoryManager<D>) -> Self {
        Self {
            manager,
            operations: BTreeMap::new(),
            logical_tick: 0,
        }
    }

    pub fn register(
        &mut self,
        task_id: TaskId,
        operation: MemoryTask,
    ) -> Result<(), MemoryTaskExecutorError> {
        if self.operations.contains_key(&task_id) {
            return Err(MemoryTaskExecutorError::DuplicateTask(task_id));
        }
        self.operations.insert(task_id, operation);
        Ok(())
    }

    pub fn manager(&self) -> &DeviceMemoryManager<D> {
        &self.manager
    }

    pub fn manager_mut(&mut self) -> &mut DeviceMemoryManager<D> {
        &mut self.manager
    }

    pub fn pending_operations(&self) -> usize {
        self.operations.len()
    }

    fn next_tick(&mut self) -> Result<u64, MemoryTaskExecutorError> {
        self.logical_tick = self.logical_tick.checked_add(1)
            .ok_or(MemoryTaskExecutorError::TickExhausted)?;
        Ok(self.logical_tick)
    }

    fn execute_operation(&mut self, operation: MemoryTask) -> Result<(), MemoryTaskExecutorError> {
        match operation {
            MemoryTask::Allocate { allocation_id, tier, bytes, expert } => {
                let tick = self.next_tick()?;
                self.manager.allocate(allocation_id, tier, bytes, tick, expert)?;
            }
            MemoryTask::Write { allocation_id, offset, data } => {
                let tick = self.next_tick()?;
                self.manager.write(&allocation_id, offset, &data, tick)?;
            }
            MemoryTask::Copy { source_id, destination_id, bytes } => {
                let tick = self.next_tick()?;
                self.manager.copy(&source_id, &destination_id, bytes, tick)?;
            }
            MemoryTask::Release { allocation_id } => {
                let tick = self.next_tick()?;
                self.manager.release(&allocation_id, tick)?;
            }
            MemoryTask::RecordReloadCost { allocation_id, estimated_ns } => {
                let tick = self.next_tick()?;
                self.manager.record_reload_cost(&allocation_id, estimated_ns, tick)?;
            }
            MemoryTask::Synchronize => self.manager.synchronize()?,
        }
        Ok(())
    }
}

impl<D: Device> TaskExecutor for MemoryTaskExecutor<D> {
    fn execute(&mut self, task: &TaskSnapshot) -> Result<(), String> {
        if !matches!(task.kind, TaskKind::Transfer | TaskKind::Prefetch) {
            return Err(MemoryTaskExecutorError::WrongQueue {
                task_id: task.id,
                kind: task.kind,
            }.to_string());
        }
        let operation = self.operations.remove(&task.id)
            .ok_or(MemoryTaskExecutorError::MissingTask(task.id))
            .map_err(|error| error.to_string())?;
        self.execute_operation(operation).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_device::CpuDevice;
    use claymore_scheduler::{ExecutionOutcome, QueueRunner, Scheduler, SchedulerConfig, TaskSpec};

    fn budgets() -> crate::MemoryBudgets {
        crate::MemoryBudgets {
            vram_soft_bytes: 64,
            vram_hard_bytes: 128,
            host_bytes: 256,
            pinned_host_bytes: 32,
            expert_cache_target_bytes: 32,
            expert_cache_max_bytes: 64,
        }
    }

    #[test]
    fn scheduler_drives_cpu_memory_operations_in_dependency_order() {
        let manager = DeviceMemoryManager::new(CpuDevice::new(), budgets(), crate::EvictionPolicy::Lru).unwrap();
        let mut executor = MemoryTaskExecutor::new(manager);
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();

        let allocate_source = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let allocate_destination = scheduler.submit(
            TaskSpec::new(TaskKind::Transfer).depends_on(allocate_source)
        ).unwrap();
        let write_source = scheduler.submit(
            TaskSpec::new(TaskKind::Transfer).depends_on(allocate_destination)
        ).unwrap();
        let copy = scheduler.submit(
            TaskSpec::new(TaskKind::Transfer).depends_on(write_source)
        ).unwrap();
        let release = scheduler.submit(
            TaskSpec::new(TaskKind::Transfer).depends_on(copy)
        ).unwrap();

        executor.register(allocate_source, MemoryTask::Allocate {
            allocation_id: "source".into(), tier: MemoryTier::Host, bytes: 8, expert: false,
        }).unwrap();
        executor.register(allocate_destination, MemoryTask::Allocate {
            allocation_id: "destination".into(), tier: MemoryTier::Vram, bytes: 8, expert: false,
        }).unwrap();
        executor.register(write_source, MemoryTask::Write {
            allocation_id: "source".into(), offset: 0, data: vec![9, 8, 7, 6],
        }).unwrap();
        executor.register(copy, MemoryTask::Copy {
            source_id: "source".into(), destination_id: "destination".into(), bytes: 4,
        }).unwrap();
        executor.register(release, MemoryTask::Release {
            allocation_id: "source".into(),
        }).unwrap();

        let mut runner = QueueRunner::new(TaskKind::Transfer, executor);
        for expected in [allocate_source, allocate_destination, write_source, copy, release] {
            assert_eq!(
                runner.run_one(&mut scheduler).unwrap(),
                ExecutionOutcome::Completed(expected)
            );
        }
        assert_eq!(scheduler.task(release).unwrap().state, claymore_scheduler::TaskState::Completed);
        assert_eq!(runner.executor().pending_operations(), 0);
        assert_eq!(
            runner.executor_mut().manager_mut().read("destination", 0, 4, 6).unwrap(),
            [9, 8, 7, 6]
        );
    }

    #[test]
    fn executor_rejects_compute_queue_without_consuming_operation() {
        let manager = DeviceMemoryManager::new(CpuDevice::new(), budgets(), crate::EvictionPolicy::Lru).unwrap();
        let mut executor = MemoryTaskExecutor::new(manager);
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let task = scheduler.submit(TaskSpec::new(TaskKind::Compute)).unwrap();
        executor.register(task, MemoryTask::Synchronize).unwrap();
        let mut runner = QueueRunner::new(TaskKind::Compute, executor);
        assert!(matches!(
            runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Failed { .. }
        ));
        assert_eq!(runner.executor().pending_operations(), 1);
    }
}
