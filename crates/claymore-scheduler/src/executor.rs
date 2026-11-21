//! Synchronous task execution adapters for the deterministic scheduler.
//!
//! An executor is invoked on the thread that calls `run_one`. No worker is
//! spawned and no `Send` / `Sync` bound is imposed. This is intentional:
//! CUDA-backed executors must be constructed and polled on their context-owner
//! thread. A backend must return success only after the operation has reached
//! the completion point required by its resource-lifetime contract (for CUDA,
//! normally after observing the associated event).
use crate::{Scheduler, SchedulerError, TaskId, TaskKind, TaskSnapshot};

/// Executes one scheduled task synchronously on the caller's thread.
pub trait TaskExecutor {
    /// Execute the task. Returning `Ok(())` confirms completion; returning an
    /// error marks the task failed and cancels its blocked descendants.
    fn execute(&mut self, task: &TaskSnapshot) -> Result<(), String>;
}

/// Result of one scheduler/executor polling step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionOutcome {
    /// No task was ready in the selected logical queue.
    Idle,
    /// The executor confirmed completion and the scheduler released dependents.
    Completed(TaskId),
    /// The executor failed; the scheduler marked the task failed.
    Failed { task_id: TaskId, reason: String },
}

/// Drives one logical queue without creating threads or moving its executor.
pub struct QueueRunner<E> {
    kind: TaskKind,
    executor: E,
}

impl<E> QueueRunner<E> {
    pub fn new(kind: TaskKind, executor: E) -> Self {
        Self { kind, executor }
    }

    pub fn kind(&self) -> TaskKind {
        self.kind
    }

    pub fn executor(&self) -> &E {
        &self.executor
    }

    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }

    pub fn into_executor(self) -> E {
        self.executor
    }
}

impl<E: TaskExecutor> QueueRunner<E> {
    /// Start and execute at most one ready task. The method is deliberately
    /// synchronous so the caller retains control over thread affinity.
    pub fn run_one(
        &mut self,
        scheduler: &mut Scheduler,
    ) -> Result<ExecutionOutcome, SchedulerError> {
        let Some(task_id) = scheduler.start_next(self.kind) else {
            return Ok(ExecutionOutcome::Idle);
        };
        let snapshot = scheduler.task(task_id)?;
        match self.executor.execute(&snapshot) {
            Ok(()) => {
                scheduler.complete(task_id)?;
                Ok(ExecutionOutcome::Completed(task_id))
            }
            Err(reason) => {
                scheduler.fail(task_id)?;
                Ok(ExecutionOutcome::Failed { task_id, reason })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SchedulerConfig, TaskPriority, TaskSpec, TaskState};
    use std::cell::RefCell;

    #[derive(Default)]
    struct RecordingExecutor {
        seen: RefCell<Vec<TaskId>>,
        fail: bool,
    }

    impl TaskExecutor for RecordingExecutor {
        fn execute(&mut self, task: &TaskSnapshot) -> Result<(), String> {
            self.seen.borrow_mut().push(task.id);
            if self.fail {
                Err("injected executor failure".to_owned())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn runner_executes_on_caller_and_releases_dependents_only_after_success() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let transfer = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let compute = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).depends_on(transfer))
            .unwrap();
        let mut runner = QueueRunner::new(TaskKind::Transfer, RecordingExecutor::default());

        assert_eq!(
            runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Completed(transfer)
        );
        assert_eq!(scheduler.task(transfer).unwrap().state, TaskState::Completed);
        assert_eq!(scheduler.task(compute).unwrap().state, TaskState::Ready);
        assert_eq!(runner.executor().seen.borrow().as_slice(), &[transfer]);
    }

    #[test]
    fn runner_reports_idle_without_invoking_executor() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let mut runner = QueueRunner::new(TaskKind::Compute, RecordingExecutor::default());
        assert_eq!(runner.run_one(&mut scheduler).unwrap(), ExecutionOutcome::Idle);
        assert!(runner.executor().seen.borrow().is_empty());
    }

    #[test]
    fn executor_failure_fails_task_and_cancels_dependents() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let transfer = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let compute = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).depends_on(transfer))
            .unwrap();
        let mut runner = QueueRunner::new(
            TaskKind::Transfer,
            RecordingExecutor { seen: RefCell::new(Vec::new()), fail: true },
        );

        assert_eq!(
            runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Failed {
                task_id: transfer,
                reason: "injected executor failure".to_owned(),
            }
        );
        assert_eq!(scheduler.task(transfer).unwrap().state, TaskState::Failed);
        assert_eq!(scheduler.task(compute).unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn each_runner_drives_only_its_own_queue() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let compute = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).with_priority(TaskPriority::High))
            .unwrap();
        let transfer = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let mut compute_runner =
            QueueRunner::new(TaskKind::Compute, RecordingExecutor::default());

        assert_eq!(
            compute_runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Completed(compute)
        );
        assert_eq!(scheduler.task(transfer).unwrap().state, TaskState::Ready);
    }
}
