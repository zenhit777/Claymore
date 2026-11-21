//! Request-to-task orchestration for the M7 scheduler.
//!
//! This adapter keeps the existing synchronous GenerationStrategy contract,
//! while allowing requests to wait behind task dependencies and priority
//! ordering. It executes only Compute tasks; transfer and prefetch queues are
//! driven by their own owner-thread QueueRunner instances.
use std::collections::BTreeMap;

use claymore_scheduler::{
    Scheduler, SchedulerConfig, SchedulerError, TaskId, TaskKind, TaskPriority, TaskSnapshot,
    TaskSpec,
};
use thiserror::Error;

use crate::{
    CancellationToken, ExecutionReport, GenerationStrategy, RequestInput, RequestState, Runtime,
    RuntimeConfig, RuntimeError,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScheduledRuntimeError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error("scheduled request payload missing for task {0:?}")]
    MissingRequest(TaskId),
}

/// A request report paired with the scheduler task that drove it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledExecution {
    pub task_id: TaskId,
    pub report: ExecutionReport,
}

/// Bridges request lifecycle to deterministic Compute scheduling.
///
/// The scheduler and strategy are both owned by this value and are never
/// moved to worker threads. A request becomes a completed scheduler task only
/// after the runtime returns a Completed report. Failed and cancelled reports
/// propagate to dependent tasks through the scheduler state machine.
pub struct ScheduledRuntime<S> {
    runtime: Runtime<S>,
    scheduler: Scheduler,
    pending: BTreeMap<TaskId, (RequestInput, CancellationToken)>,
}

impl<S: GenerationStrategy> ScheduledRuntime<S> {
    pub fn new(
        runtime_config: RuntimeConfig,
        scheduler_config: SchedulerConfig,
        strategy: S,
    ) -> Result<Self, ScheduledRuntimeError> {
        Ok(Self {
            runtime: Runtime::new(runtime_config, strategy)?,
            scheduler: Scheduler::new(scheduler_config)?,
            pending: BTreeMap::new(),
        })
    }

    pub fn submit_request(
        &mut self,
        input: RequestInput,
        cancellation: CancellationToken,
        priority: TaskPriority,
        dependencies: impl IntoIterator<Item = TaskId>,
    ) -> Result<TaskId, ScheduledRuntimeError> {
        let mut spec = TaskSpec::new(TaskKind::Compute).with_priority(priority);
        for dependency in dependencies {
            spec = spec.depends_on(dependency);
        }
        let task_id = self.scheduler.submit(spec)?;
        self.pending.insert(task_id, (input, cancellation));
        Ok(task_id)
    }

    /// Execute at most one ready Compute request. Blocked requests remain
    /// queued until their scheduler dependencies have been confirmed complete.
    pub fn run_one(&mut self) -> Result<Option<ScheduledExecution>, ScheduledRuntimeError> {
        let Some(task_id) = self.scheduler.start_next(TaskKind::Compute) else {
            return Ok(None);
        };
        let (input, cancellation) = self
            .pending
            .remove(&task_id)
            .ok_or(ScheduledRuntimeError::MissingRequest(task_id))?;
        let report = match self.runtime.execute(input, cancellation) {
            Ok(report) => report,
            Err(error) => {
                self.scheduler.fail(task_id)?;
                return Err(error.into());
            }
        };
        match report.state {
            RequestState::Completed => self.scheduler.complete(task_id)?,
            RequestState::Failed => self.scheduler.fail(task_id)?,
            RequestState::Cancelled => self.scheduler.cancel_running(task_id)?,
            RequestState::Created | RequestState::Running => {
                // Runtime::execute is synchronous and must always return a
                // terminal report. Treat a contract violation as a task failure.
                self.scheduler.fail(task_id)?;
                return Err(ScheduledRuntimeError::Runtime(RuntimeError::StrategyFailed(
                    "runtime returned a non-terminal execution report".to_owned(),
                )));
            }
        }
        Ok(Some(ScheduledExecution { task_id, report }))
    }

    /// Cancel a request that has not started. Running requests must be
    /// cancelled through their shared CancellationToken.
    pub fn cancel_queued(&mut self, task_id: TaskId) -> Result<(), ScheduledRuntimeError> {
        self.scheduler.cancel(task_id)?;
        self.pending.remove(&task_id);
        Ok(())
    }

    pub fn task(&self, task_id: TaskId) -> Result<TaskSnapshot, SchedulerError> {
        self.scheduler.task(task_id)
    }

    pub fn ready_count(&self) -> usize {
        self.scheduler.ready_count(TaskKind::Compute)
    }

    pub fn runtime(&self) -> &Runtime<S> {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut Runtime<S> {
        &mut self.runtime
    }

    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionContext, RequestOutput};
    use claymore_scheduler::TaskState;
    use std::cell::Cell;

    struct Echo;
    impl GenerationStrategy for Echo {
        fn execute(
            &mut self,
            _context: &mut ExecutionContext<'_>,
            input: &RequestInput,
        ) -> Result<RequestOutput, String> {
            Ok(RequestOutput::new(input.payload.clone()))
        }
    }

    struct Fail;
    impl GenerationStrategy for Fail {
        fn execute(
            &mut self,
            _context: &mut ExecutionContext<'_>,
            _input: &RequestInput,
        ) -> Result<RequestOutput, String> {
            Err("injected failure".into())
        }
    }

    struct Count(Cell<usize>);
    impl GenerationStrategy for Count {
        fn execute(
            &mut self,
            _context: &mut ExecutionContext<'_>,
            input: &RequestInput,
        ) -> Result<RequestOutput, String> {
            self.0.set(self.0.get() + 1);
            Ok(RequestOutput::new(input.payload.clone()))
        }
    }

    fn runtime<S: GenerationStrategy>(strategy: S) -> ScheduledRuntime<S> {
        ScheduledRuntime::new(
            RuntimeConfig::default(),
            SchedulerConfig::default(),
            strategy,
        ).unwrap()
    }

    #[test]
    fn dependent_request_waits_for_predecessor_completion() {
        let mut runtime = runtime(Echo);
        let first = runtime.submit_request(
            RequestInput::new([1]),
            CancellationToken::new(),
            TaskPriority::Normal,
            [],
        ).unwrap();
        let second = runtime.submit_request(
            RequestInput::new([2]),
            CancellationToken::new(),
            TaskPriority::High,
            [first],
        ).unwrap();

        assert_eq!(runtime.ready_count(), 1);
        assert_eq!(runtime.run_one().unwrap().unwrap().task_id, first);
        assert_eq!(runtime.task(second).unwrap().state, TaskState::Ready);
        assert_eq!(runtime.run_one().unwrap().unwrap().task_id, second);
        assert_eq!(runtime.task(second).unwrap().state, TaskState::Completed);
    }

    #[test]
    fn failed_request_cancels_dependent_request() {
        let mut runtime = runtime(Fail);
        let first = runtime.submit_request(
            RequestInput::new([1]),
            CancellationToken::new(),
            TaskPriority::Normal,
            [],
        ).unwrap();
        let second = runtime.submit_request(
            RequestInput::new([2]),
            CancellationToken::new(),
            TaskPriority::Normal,
            [first],
        ).unwrap();

        let result = runtime.run_one().unwrap().unwrap();
        assert_eq!(result.report.state, RequestState::Failed);
        assert_eq!(runtime.task(second).unwrap().state, TaskState::Cancelled);
        assert!(runtime.run_one().unwrap().is_none());
    }

    #[test]
    fn pre_cancelled_request_cancels_scheduler_task() {
        let mut runtime = runtime(Echo);
        let token = CancellationToken::new();
        token.cancel();
        let id = runtime.submit_request(
            RequestInput::new([1]),
            token,
            TaskPriority::Normal,
            [],
        ).unwrap();

        let result = runtime.run_one().unwrap().unwrap();
        assert_eq!(result.report.state, RequestState::Cancelled);
        assert_eq!(runtime.task(id).unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn queued_cancellation_prevents_strategy_execution() {
        let mut runtime = runtime(Count(Cell::new(0)));
        let id = runtime.submit_request(
            RequestInput::new([1]),
            CancellationToken::new(),
            TaskPriority::Normal,
            [],
        ).unwrap();
        runtime.cancel_queued(id).unwrap();
        assert_eq!(runtime.runtime().strategy().0.get(), 0);
        assert_eq!(runtime.task(id).unwrap().state, TaskState::Cancelled);
    }
}
