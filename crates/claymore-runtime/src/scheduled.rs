//! Request-to-task orchestration for the M7 scheduler.
//!
//! This adapter keeps the existing synchronous GenerationStrategy contract,
//! while allowing requests to wait behind task dependencies and priority
//! ordering. It executes only Compute tasks; transfer and prefetch queues are
//! driven by their own owner-thread QueueRunner instances.
use std::collections::BTreeMap;
use std::time::Instant;

use claymore_scheduler::{
    ExecutionOutcome, QueueRunner, Scheduler, SchedulerConfig, SchedulerError, TaskExecutor,
    TaskId, TaskKind, TaskPriority, TaskSnapshot, TaskSpec,
};
use thiserror::Error;

use crate::{
    CancellationToken, ExecutionReport, GenerationStrategy, HardwareProfile, HardwareProfileError,
    RequestInput, RequestState, Runtime, RuntimeConfig, RuntimeError,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScheduledRuntimeError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Profile(#[from] HardwareProfileError),
    #[error("scheduled request payload missing for task {0:?}")]
    MissingRequest(TaskId),
    #[error("Compute tasks must be submitted through submit_request")]
    ComputeTaskRequiresRequest,
    #[error("the Compute queue is reserved for ScheduledRuntime::run_one")]
    ComputeQueueReserved,
}

/// A request report paired with the scheduler task that drove it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledExecution {
    pub task_id: TaskId,
    pub report: ExecutionReport,
}

/// Cumulative request/compute timing for one ScheduledRuntime instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScheduledRuntimeMetrics {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub requests_cancelled: u64,
    pub total_execution_ns: u64,
    pub max_execution_ns: u64,
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
    metrics: ScheduledRuntimeMetrics,
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
            metrics: ScheduledRuntimeMetrics::default(),
        })
    }

    /// Construct the scheduler from a validated hardware profile.
    pub fn from_profile(
        profile: &HardwareProfile,
        strategy: S,
    ) -> Result<Self, ScheduledRuntimeError> {
        Self::new(
            RuntimeConfig::default(),
            profile.scheduler_config()?,
            strategy,
        )
    }

    /// Submit a Transfer or Prefetch task into the same dependency graph
    /// used by scheduled Compute requests.
    pub fn submit_task(&mut self, spec: TaskSpec) -> Result<TaskId, ScheduledRuntimeError> {
        if spec.kind == TaskKind::Compute {
            return Err(ScheduledRuntimeError::ComputeTaskRequiresRequest);
        }
        let task_id = self.scheduler.submit(spec)?;
        self.prune_cancelled_requests();
        Ok(task_id)
    }

    /// Drive a non-Compute queue against this runtime's shared scheduler.
    /// This is the integration point for MemoryTaskExecutor and
    /// CudaTransferExecutor; their QueueRunner must stay on its owner thread.
    pub fn run_queue_one<E: TaskExecutor>(
        &mut self,
        runner: &mut QueueRunner<E>,
    ) -> Result<ExecutionOutcome, ScheduledRuntimeError> {
        if runner.kind() == TaskKind::Compute {
            return Err(ScheduledRuntimeError::ComputeQueueReserved);
        }
        let outcome = runner.run_one(&mut self.scheduler)?;
        self.prune_cancelled_requests();
        Ok(outcome)
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
        self.prune_cancelled_requests();
        Ok(task_id)
    }

    /// Execute at most one ready Compute request. Blocked requests remain
    /// queued until their scheduler dependencies have been confirmed complete.
    pub fn run_one(&mut self) -> Result<Option<ScheduledExecution>, ScheduledRuntimeError> {
        let Some(task_id) = self.scheduler.start_next(TaskKind::Compute) else {
            return Ok(None);
        };
        let Some((input, cancellation)) = self.pending.remove(&task_id) else {
            self.scheduler.fail(task_id)?;
            return Err(ScheduledRuntimeError::MissingRequest(task_id));
        };
        self.metrics.requests_started = self.metrics.requests_started.saturating_add(1);
        let started_at = Instant::now();
        let report = match self.runtime.execute(input, cancellation) {
            Ok(report) => report,
            Err(error) => {
                self.record_execution_time(started_at);
                self.metrics.requests_failed = self.metrics.requests_failed.saturating_add(1);
                self.scheduler.fail(task_id)?;
                self.prune_cancelled_requests();
                return Err(error.into());
            }
        };
        self.record_execution_time(started_at);
        match report.state {
            RequestState::Completed => {
                self.scheduler.complete(task_id)?;
                self.metrics.requests_completed = self.metrics.requests_completed.saturating_add(1);
            }
            RequestState::Failed => {
                self.scheduler.fail(task_id)?;
                self.prune_cancelled_requests();
                self.metrics.requests_failed = self.metrics.requests_failed.saturating_add(1);
            }
            RequestState::Cancelled => {
                self.scheduler.cancel_running(task_id)?;
                self.prune_cancelled_requests();
                self.metrics.requests_cancelled = self.metrics.requests_cancelled.saturating_add(1);
            }
            RequestState::Created | RequestState::Running => {
                // Runtime::execute is synchronous and must always return a
                // terminal report. Treat a contract violation as a task failure.
                self.scheduler.fail(task_id)?;
                self.prune_cancelled_requests();
                self.metrics.requests_failed = self.metrics.requests_failed.saturating_add(1);
                return Err(ScheduledRuntimeError::Runtime(
                    RuntimeError::StrategyFailed(
                        "runtime returned a non-terminal execution report".to_owned(),
                    ),
                ));
            }
        }
        Ok(Some(ScheduledExecution { task_id, report }))
    }

    /// Cancel a request that has not started. Running requests must be
    /// cancelled through their shared CancellationToken.
    pub fn cancel_queued(&mut self, task_id: TaskId) -> Result<(), ScheduledRuntimeError> {
        self.scheduler.cancel(task_id)?;
        self.prune_cancelled_requests();
        Ok(())
    }

    fn prune_cancelled_requests(&mut self) {
        for task_id in self.scheduler.cancelled_task_ids() {
            self.pending.remove(&task_id);
        }
    }

    pub fn pending_request_count(&self) -> usize {
        self.pending.len()
    }

    fn record_execution_time(&mut self, started_at: Instant) {
        let elapsed_ns = started_at.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.metrics.total_execution_ns =
            self.metrics.total_execution_ns.saturating_add(elapsed_ns);
        self.metrics.max_execution_ns = self.metrics.max_execution_ns.max(elapsed_ns);
    }

    pub fn metrics(&self) -> ScheduledRuntimeMetrics {
        self.metrics
    }

    pub fn task(&self, task_id: TaskId) -> Result<TaskSnapshot, SchedulerError> {
        self.scheduler.task(task_id)
    }

    pub fn ready_count(&self) -> usize {
        self.scheduler.ready_count(TaskKind::Compute)
    }

    pub fn ready_count_for(&self, kind: TaskKind) -> usize {
        self.scheduler.ready_count(kind)
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
        )
        .unwrap()
    }

    #[test]
    fn hardware_profile_supplies_scheduler_configuration() {
        let profile =
            HardwareProfile::from_json(include_str!("../../../config/rtx5090-7950x3d.json"))
                .unwrap();
        let runtime = ScheduledRuntime::from_profile(&profile, Echo).unwrap();
        assert_eq!(runtime.scheduler_config().cpu_workers, 10);
        assert_eq!(runtime.scheduler_config().max_prefetches, 2);
    }

    struct NoopTransfer;
    impl TaskExecutor for NoopTransfer {
        fn execute(&mut self, _task: &TaskSnapshot) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn scheduled_runtime_collects_compute_request_metrics() {
        let mut runtime = runtime(Echo);
        runtime
            .submit_request(
                RequestInput::new([1]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [],
            )
            .unwrap();
        runtime.run_one().unwrap().unwrap();
        let metrics = runtime.metrics();
        assert_eq!(metrics.requests_started, 1);
        assert_eq!(metrics.requests_completed, 1);
        assert_eq!(metrics.requests_failed, 0);
        assert_eq!(metrics.requests_cancelled, 0);
    }

    #[test]
    fn transfer_task_and_compute_request_share_one_dependency_graph() {
        let mut runtime = runtime(Echo);
        let transfer = runtime
            .submit_task(TaskSpec::new(TaskKind::Transfer))
            .unwrap();
        let request = runtime
            .submit_request(
                RequestInput::new([7, 8]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [transfer],
            )
            .unwrap();

        assert_eq!(runtime.ready_count_for(TaskKind::Transfer), 1);
        assert_eq!(runtime.ready_count(), 0);
        assert!(runtime.run_one().unwrap().is_none());

        let mut transfer_runner = QueueRunner::new(TaskKind::Transfer, NoopTransfer);
        assert_eq!(
            runtime.run_queue_one(&mut transfer_runner).unwrap(),
            ExecutionOutcome::Completed(transfer)
        );
        assert_eq!(runtime.task(request).unwrap().state, TaskState::Ready);
        assert_eq!(runtime.run_one().unwrap().unwrap().task_id, request);
        assert_eq!(runtime.task(request).unwrap().state, TaskState::Completed);
    }

    struct FailingTransfer;
    impl TaskExecutor for FailingTransfer {
        fn execute(&mut self, _task: &TaskSnapshot) -> Result<(), String> {
            Err("transfer failed".into())
        }
    }

    #[test]
    fn failed_transfer_discards_dependent_request_payload() {
        let mut runtime = runtime(Echo);
        let transfer = runtime
            .submit_task(TaskSpec::new(TaskKind::Transfer))
            .unwrap();
        runtime
            .submit_request(
                RequestInput::new([4, 5, 6]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [transfer],
            )
            .unwrap();
        assert_eq!(runtime.pending_request_count(), 1);

        let mut runner = QueueRunner::new(TaskKind::Transfer, FailingTransfer);
        assert!(matches!(
            runtime.run_queue_one(&mut runner).unwrap(),
            ExecutionOutcome::Failed { task_id, .. } if task_id == transfer
        ));
        assert_eq!(runtime.pending_request_count(), 0);
        assert!(runtime.run_one().unwrap().is_none());
    }

    #[test]
    fn request_submitted_after_failed_dependency_is_cancelled_and_payload_pruned() {
        let mut runtime = runtime(Echo);
        let failed = runtime
            .submit_task(TaskSpec::new(TaskKind::Transfer))
            .unwrap();
        let mut runner = QueueRunner::new(TaskKind::Transfer, FailingTransfer);
        assert!(matches!(
            runtime.run_queue_one(&mut runner).unwrap(),
            ExecutionOutcome::Failed { task_id, .. } if task_id == failed
        ));
        let request = runtime
            .submit_request(
                RequestInput::new([9, 8, 7]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [failed],
            )
            .unwrap();
        assert_eq!(runtime.task(request).unwrap().state, TaskState::Cancelled);
        assert_eq!(runtime.pending_request_count(), 0);
        assert!(runtime.run_one().unwrap().is_none());
    }

    #[test]
    fn request_submitted_after_cancelled_dependency_is_cancelled_and_payload_pruned() {
        let mut runtime = runtime(Echo);
        let cancelled = runtime
            .submit_task(TaskSpec::new(TaskKind::Transfer))
            .unwrap();
        runtime.cancel_queued(cancelled).unwrap();
        let request = runtime
            .submit_request(
                RequestInput::new([6, 5, 4]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [cancelled],
            )
            .unwrap();
        assert_eq!(runtime.task(request).unwrap().state, TaskState::Cancelled);
        assert_eq!(runtime.pending_request_count(), 0);
    }

    #[test]
    fn scheduled_runtime_reserves_compute_queue_for_request_payloads() {
        let mut runtime = runtime(Echo);
        assert_eq!(
            runtime.submit_task(TaskSpec::new(TaskKind::Compute)),
            Err(ScheduledRuntimeError::ComputeTaskRequiresRequest)
        );
        let mut compute_runner = QueueRunner::new(TaskKind::Compute, NoopTransfer);
        assert_eq!(
            runtime.run_queue_one(&mut compute_runner),
            Err(ScheduledRuntimeError::ComputeQueueReserved)
        );
    }

    #[test]
    fn dependent_request_waits_for_predecessor_completion() {
        let mut runtime = runtime(Echo);
        let first = runtime
            .submit_request(
                RequestInput::new([1]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [],
            )
            .unwrap();
        let second = runtime
            .submit_request(
                RequestInput::new([2]),
                CancellationToken::new(),
                TaskPriority::High,
                [first],
            )
            .unwrap();

        assert_eq!(runtime.ready_count(), 1);
        assert_eq!(runtime.run_one().unwrap().unwrap().task_id, first);
        assert_eq!(runtime.task(second).unwrap().state, TaskState::Ready);
        assert_eq!(runtime.run_one().unwrap().unwrap().task_id, second);
        assert_eq!(runtime.task(second).unwrap().state, TaskState::Completed);
    }

    #[test]
    fn failed_request_cancels_dependent_request() {
        let mut runtime = runtime(Fail);
        let first = runtime
            .submit_request(
                RequestInput::new([1]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [],
            )
            .unwrap();
        let second = runtime
            .submit_request(
                RequestInput::new([2]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [first],
            )
            .unwrap();

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
        let id = runtime
            .submit_request(RequestInput::new([1]), token, TaskPriority::Normal, [])
            .unwrap();

        let result = runtime.run_one().unwrap().unwrap();
        assert_eq!(result.report.state, RequestState::Cancelled);
        assert_eq!(runtime.task(id).unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn queued_cancellation_prevents_strategy_execution() {
        let mut runtime = runtime(Count(Cell::new(0)));
        let id = runtime
            .submit_request(
                RequestInput::new([1]),
                CancellationToken::new(),
                TaskPriority::Normal,
                [],
            )
            .unwrap();
        runtime.cancel_queued(id).unwrap();
        assert_eq!(runtime.runtime().strategy().0.get(), 0);
        assert_eq!(runtime.task(id).unwrap().state, TaskState::Cancelled);
    }
}
