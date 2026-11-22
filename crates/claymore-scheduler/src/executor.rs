//! Synchronous task execution adapters for the deterministic scheduler.
//!
//! An executor is invoked on the thread that calls `run_one`. No worker is
//! spawned and no `Send` / `Sync` bound is imposed. This is intentional:
//! CUDA-backed executors must be constructed and polled on their context-owner
//! thread. A backend must return success only after the operation has reached
//! the completion point required by its resource-lifetime contract (for CUDA,
//! normally after observing the associated event).
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use crate::{Scheduler, SchedulerError, TaskId, TaskKind, TaskSnapshot};

/// Executes one scheduled task synchronously on the caller's thread.
pub trait TaskExecutor {
    /// Execute the task. Returning `Ok(())` confirms completion; returning an
    /// error marks the task failed and cancels its blocked descendants.
    fn execute(&mut self, task: &TaskSnapshot) -> Result<(), String>;

    /// Release executor-owned payloads for work cancelled before execution.
    fn discard(&mut self, _task_id: TaskId) {}
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

/// Cumulative wall-clock measurements for one owner-thread queue runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRunnerMetrics {
    pub kind: TaskKind,
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
    pub idle_polls: u64,
    pub total_execution_ns: u64,
    pub max_execution_ns: u64,
}

/// Drives one logical queue without creating threads or moving its executor.
pub struct QueueRunner<E> {
    kind: TaskKind,
    executor: E,
    metrics: QueueRunnerMetrics,
    last_cancelled_generation: u64,
}

impl<E> QueueRunner<E> {
    pub fn new(kind: TaskKind, executor: E) -> Self {
        Self {
            kind,
            executor,
            metrics: QueueRunnerMetrics {
                kind,
                started: 0,
                completed: 0,
                failed: 0,
                idle_polls: 0,
                total_execution_ns: 0,
                max_execution_ns: 0,
            },
            last_cancelled_generation: 0,
        }
    }

    pub fn kind(&self) -> TaskKind {
        self.kind
    }

    pub fn metrics(&self) -> QueueRunnerMetrics {
        self.metrics
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
    fn prune_cancelled(&mut self, scheduler: &Scheduler) {
        let (generation, task_ids) =
            scheduler.cancelled_task_ids_since(self.last_cancelled_generation);
        self.last_cancelled_generation = generation;
        for task_id in task_ids {
            self.executor.discard(task_id);
        }
    }

    /// Start and execute at most one ready task. The method is deliberately
    /// synchronous so the caller retains control over thread affinity.
    pub fn run_one(
        &mut self,
        scheduler: &mut Scheduler,
    ) -> Result<ExecutionOutcome, SchedulerError> {
        self.prune_cancelled(scheduler);
        let Some(task_id) = scheduler.start_next(self.kind) else {
            self.metrics.idle_polls = self.metrics.idle_polls.saturating_add(1);
            return Ok(ExecutionOutcome::Idle);
        };
        self.metrics.started = self.metrics.started.saturating_add(1);
        let snapshot = scheduler.task(task_id)?;
        let started_at = Instant::now();
        // A panic must not strand the scheduler task in Running forever. The
        // executor is still invoked synchronously on its owner thread; unwind
        // is converted into the same failed-task path as an ordinary error.
        let result = catch_unwind(AssertUnwindSafe(|| self.executor.execute(&snapshot)))
            .unwrap_or_else(|payload| {
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("executor panicked with a non-string payload");
                Err(format!("executor panicked: {message}"))
            });
        let elapsed_ns = started_at.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.metrics.total_execution_ns =
            self.metrics.total_execution_ns.saturating_add(elapsed_ns);
        self.metrics.max_execution_ns = self.metrics.max_execution_ns.max(elapsed_ns);
        match result {
            Ok(()) => {
                scheduler.complete(task_id)?;
                self.metrics.completed = self.metrics.completed.saturating_add(1);
                Ok(ExecutionOutcome::Completed(task_id))
            }
            Err(reason) => {
                scheduler.fail(task_id)?;
                self.prune_cancelled(scheduler);
                self.metrics.failed = self.metrics.failed.saturating_add(1);
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
        assert_eq!(
            scheduler.task(transfer).unwrap().state,
            TaskState::Completed
        );
        assert_eq!(scheduler.task(compute).unwrap().state, TaskState::Ready);
        assert_eq!(runner.executor().seen.borrow().as_slice(), &[transfer]);
    }

    #[test]
    fn runner_reports_idle_without_invoking_executor() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let mut runner = QueueRunner::new(TaskKind::Compute, RecordingExecutor::default());
        assert_eq!(
            runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Idle
        );
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
            RecordingExecutor {
                seen: RefCell::new(Vec::new()),
                fail: true,
            },
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
    fn queue_runner_metrics_are_scoped_to_its_logical_queue() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let task = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let mut runner = QueueRunner::new(TaskKind::Transfer, RecordingExecutor::default());
        assert_eq!(
            runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Completed(task)
        );
        assert_eq!(
            runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Idle
        );
        let metrics = runner.metrics();
        assert_eq!(metrics.kind, TaskKind::Transfer);
        assert_eq!(metrics.started, 1);
        assert_eq!(metrics.completed, 1);
        assert_eq!(metrics.failed, 0);
        assert_eq!(metrics.idle_polls, 1);
    }

    #[test]
    fn each_runner_drives_only_its_own_queue() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let compute = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).with_priority(TaskPriority::High))
            .unwrap();
        let transfer = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let mut compute_runner = QueueRunner::new(TaskKind::Compute, RecordingExecutor::default());

        assert_eq!(
            compute_runner.run_one(&mut scheduler).unwrap(),
            ExecutionOutcome::Completed(compute)
        );
        assert_eq!(scheduler.task(transfer).unwrap().state, TaskState::Ready);
    }
}
