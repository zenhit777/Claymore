//! Shared, model-neutral orchestration for block-based diffusion generation.
//!
//! The orchestrator owns lifecycle metadata and scheduler task IDs only. It
//! never owns tensors, device buffers, CUDA handles, or model weights. Model
//! adapters provide the work for each bounded step through `DiffusionExecutor`.
//! Cancellation is observed at step boundaries, so an in-flight backend call is
//! never reported cancelled before the executor returns.
use std::{collections::BTreeMap, time::Instant};

use claymore_scheduler::{
    Scheduler, SchedulerConfig, SchedulerError, TaskId, TaskKind, TaskPriority, TaskSpec, TaskState,
};
use thiserror::Error;

use crate::CancellationToken;

/// Stable identifier for one diffusion generation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiffusionSessionId(u64);

impl DiffusionSessionId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable identifier for a block within a generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

/// A bounded, model-provided unit of diffusion work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionStep {
    PrepareBlock { block: BlockId },
    Denoise { block: BlockId, iteration: u32 },
    ValidateBlock { block: BlockId },
    CommitBlock { block: BlockId },
}

/// Immutable ordered generation plan. Blocks are committed in plan order;
/// each block has its own denoising schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionPlan {
    steps: Vec<DiffusionStep>,
}

impl DiffusionPlan {
    pub fn new(steps: Vec<DiffusionStep>) -> Result<Self, DiffusionError> {
        if steps.is_empty() {
            return Err(DiffusionError::InvalidPlan(
                "plan must contain at least one step",
            ));
        }

        let mut expected_block = 0_u32;
        let mut phase = 0_u8;
        let mut next_iteration = 0_u32;
        let mut saw_denoise = false;

        for step in &steps {
            match *step {
                DiffusionStep::PrepareBlock { block } => {
                    if phase != 0 || block.0 != expected_block {
                        return Err(DiffusionError::InvalidPlan(
                            "each block must start with PrepareBlock in ascending block order",
                        ));
                    }
                    phase = 1;
                    next_iteration = 0;
                    saw_denoise = false;
                }
                DiffusionStep::Denoise { block, iteration } => {
                    if phase != 1 || block.0 != expected_block || iteration != next_iteration {
                        return Err(DiffusionError::InvalidPlan(
                            "denoising iterations must be contiguous and follow block preparation",
                        ));
                    }
                    next_iteration = next_iteration
                        .checked_add(1)
                        .ok_or(DiffusionError::InvalidPlan("denoising iteration overflow"))?;
                    saw_denoise = true;
                }
                DiffusionStep::ValidateBlock { block } => {
                    if phase != 1 || block.0 != expected_block || !saw_denoise {
                        return Err(DiffusionError::InvalidPlan(
                            "a block must be denoised before validation",
                        ));
                    }
                    phase = 2;
                }
                DiffusionStep::CommitBlock { block } => {
                    if phase != 2 || block.0 != expected_block {
                        return Err(DiffusionError::InvalidPlan(
                            "a block must be validated before commit",
                        ));
                    }
                    expected_block = expected_block
                        .checked_add(1)
                        .ok_or(DiffusionError::InvalidPlan("block identifier overflow"))?;
                    phase = 0;
                }
            }
        }

        if phase != 0 {
            return Err(DiffusionError::InvalidPlan(
                "the final block must be validated and committed",
            ));
        }
        Ok(Self { steps })
    }

    pub fn steps(&self) -> &[DiffusionStep] {
        &self.steps
    }

    pub fn block_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| matches!(step, DiffusionStep::PrepareBlock { .. }))
            .count()
    }
}

/// Executor boundary implemented by a model adapter. Implementations should
/// perform one bounded unit of work and return only after it is safe to advance.
pub trait DiffusionExecutor {
    fn execute_step(&mut self, step: DiffusionStep) -> Result<(), String>;
}

/// Observable lifecycle of a generation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionSessionState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl DiffusionSessionState {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Snapshot intentionally contains no tensor/buffer payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionSessionSnapshot {
    pub id: DiffusionSessionId,
    pub state: DiffusionSessionState,
    pub total_steps: usize,
    pub completed_steps: usize,
    pub current_step: Option<DiffusionStep>,
    pub error: Option<String>,
}

/// Monotonic counters for orchestration diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiffusionMetrics {
    pub sessions_submitted: u64,
    pub sessions_completed: u64,
    pub sessions_failed: u64,
    pub sessions_cancelled: u64,
    pub steps_started: u64,
    pub steps_completed: u64,
    pub steps_failed: u64,
    pub steps_cancelled: u64,
    /// Cumulative number of blocks admitted across all sessions.
    pub blocks_submitted: u64,
    /// Cumulative number of successfully committed blocks.
    pub blocks_completed: u64,
    /// Current blocks not yet committed in non-terminal sessions.
    pub blocks_pending: u64,
    /// Observed executor time, in nanoseconds, grouped by step phase.
    pub prepare_duration_ns: u64,
    pub denoise_duration_ns: u64,
    pub validate_duration_ns: u64,
    pub commit_duration_ns: u64,
    /// Sum of measured executor durations; excludes scheduler and queue wait.
    pub execution_duration_ns: u64,
}

/// Result of executing at most one scheduled diffusion step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffusionRunOutcome {
    Idle,
    StepCompleted {
        session_id: DiffusionSessionId,
        task_id: TaskId,
        step: DiffusionStep,
    },
    SessionCompleted {
        session_id: DiffusionSessionId,
        task_id: TaskId,
        step: DiffusionStep,
    },
    StepFailed {
        session_id: DiffusionSessionId,
        task_id: TaskId,
        step: DiffusionStep,
        error: String,
    },
    SessionCancelled {
        session_id: DiffusionSessionId,
        task_id: TaskId,
        step: DiffusionStep,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DiffusionError {
    #[error("invalid diffusion plan: {0}")]
    InvalidPlan(&'static str),
    #[error("diffusion session identifier space exhausted")]
    SessionIdExhausted,
    #[error("diffusion session {0:?} does not exist")]
    UnknownSession(DiffusionSessionId),
    #[error("diffusion session {0:?} is already terminal")]
    SessionTerminal(DiffusionSessionId),
    #[error("scheduler returned an unowned diffusion task {0:?}")]
    UnknownTask(TaskId),
    #[error("diffusion plan for session {0:?} is missing")]
    MissingPlan(DiffusionSessionId),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
}

struct SessionRecord {
    snapshot: DiffusionSessionSnapshot,
    cancellation: CancellationToken,
    tasks: Vec<TaskId>,
    completed_blocks: usize,
}

/// Deterministic block/step orchestrator backed by Claymore's shared scheduler.
///
/// All methods are synchronous and owner-thread confined by the scheduler.
/// The scheduler stores task IDs and dependency edges; model data remains with
/// the executor and memory manager.
pub struct DiffusionOrchestrator {
    scheduler: Scheduler,
    sessions: BTreeMap<DiffusionSessionId, SessionRecord>,
    plans: BTreeMap<DiffusionSessionId, DiffusionPlan>,
    task_index: BTreeMap<TaskId, (DiffusionSessionId, usize)>,
    next_session_id: u64,
    metrics: DiffusionMetrics,
}

impl DiffusionOrchestrator {
    pub fn new(config: SchedulerConfig) -> Result<Self, DiffusionError> {
        Ok(Self {
            scheduler: Scheduler::new(config)?,
            sessions: BTreeMap::new(),
            plans: BTreeMap::new(),
            task_index: BTreeMap::new(),
            next_session_id: 1,
            metrics: DiffusionMetrics::default(),
        })
    }

    /// Admit a plan as a serial dependency chain. Equal-priority sessions are
    /// FIFO by admission: an earlier session's full chain precedes a later
    /// session. Higher-priority ready work may run first.
    pub fn submit(
        &mut self,
        plan: DiffusionPlan,
        cancellation: CancellationToken,
        priority: TaskPriority,
    ) -> Result<DiffusionSessionId, DiffusionError> {
        if cancellation.is_cancelled() {
            // Preserve a visible terminal session without scheduling work.
            let id = self.allocate_session_id()?;
            self.sessions.insert(
                id,
                SessionRecord {
                    snapshot: DiffusionSessionSnapshot {
                        id,
                        state: DiffusionSessionState::Cancelled,
                        total_steps: plan.steps.len(),
                        completed_steps: 0,
                        current_step: None,
                        error: None,
                    },
                    cancellation,
                    tasks: Vec::new(),
                    completed_blocks: 0,
                },
            );
            self.metrics.sessions_submitted = self.metrics.sessions_submitted.saturating_add(1);
            self.metrics.blocks_submitted = self.metrics.blocks_submitted.saturating_add(
                u64::try_from(plan.block_count()).unwrap_or(u64::MAX),
            );
            self.metrics.sessions_cancelled = self.metrics.sessions_cancelled.saturating_add(1);
            return Ok(id);
        }

        let id = self.allocate_session_id()?;
        let mut tasks = Vec::with_capacity(plan.steps.len());
        let mut previous = None;
        for _ in &plan.steps {
            let mut spec = TaskSpec::new(TaskKind::Compute).with_priority(priority);
            if let Some(dependency) = previous {
                spec = spec.depends_on(dependency);
            }
            match self.scheduler.submit(spec) {
                Ok(task_id) => {
                    tasks.push(task_id);
                    previous = Some(task_id);
                }
                Err(error) => {
                    // Roll back all tasks admitted for this not-yet-visible session.
                    for task_id in tasks.iter().copied() {
                        let _ = self.scheduler.cancel(task_id);
                    }
                    return Err(error.into());
                }
            }
        }

        for (index, task_id) in tasks.iter().copied().enumerate() {
            self.task_index.insert(task_id, (id, index));
        }
        self.sessions.insert(
            id,
            SessionRecord {
                snapshot: DiffusionSessionSnapshot {
                    id,
                    state: DiffusionSessionState::Queued,
                    total_steps: plan.steps.len(),
                    completed_steps: 0,
                    current_step: None,
                    error: None,
                },
                cancellation,
                tasks,
                completed_blocks: 0,
            },
        );
        let block_count = plan.block_count();
        self.plans.insert(id, plan);
        self.metrics.sessions_submitted = self.metrics.sessions_submitted.saturating_add(1);
        self.metrics.blocks_submitted = self.metrics.blocks_submitted.saturating_add(
            u64::try_from(block_count).unwrap_or(u64::MAX),
        );
        Ok(id)
    }

    fn allocate_session_id(&mut self) -> Result<DiffusionSessionId, DiffusionError> {
        let id = DiffusionSessionId(self.next_session_id);
        self.next_session_id = self
            .next_session_id
            .checked_add(1)
            .ok_or(DiffusionError::SessionIdExhausted)?;
        Ok(id)
    }

    /// Run one ready step. Cancellation is honored before starting a step or
    /// after the executor returns; an in-flight step is never interrupted here.
    pub fn run_one<E: DiffusionExecutor>(
        &mut self,
        executor: &mut E,
    ) -> Result<DiffusionRunOutcome, DiffusionError> {
        let Some(task_id) = self.scheduler.start_next(TaskKind::Compute) else {
            return Ok(DiffusionRunOutcome::Idle);
        };
        let Some((session_id, index)) = self.task_index.get(&task_id).copied() else {
            self.scheduler.fail(task_id)?;
            return Err(DiffusionError::UnknownTask(task_id));
        };
        let Some(step) = self
            .plans
            .get(&session_id)
            .and_then(|p| p.steps.get(index))
            .copied()
        else {
            self.scheduler.fail(task_id)?;
            return Err(DiffusionError::MissingPlan(session_id));
        };
        let Some(session) = self.sessions.get(&session_id) else {
            self.scheduler.fail(task_id)?;
            return Err(DiffusionError::UnknownSession(session_id));
        };
        let cancelled_before_step = session.cancellation.is_cancelled();
        if let Some(session) = self.sessions.get_mut(&session_id) {
            session.snapshot.state = DiffusionSessionState::Running;
            session.snapshot.current_step = Some(step);
        }
        self.metrics.steps_started = self.metrics.steps_started.saturating_add(1);

        if cancelled_before_step {
            self.scheduler.cancel_running(task_id)?;
            self.metrics.steps_cancelled = self.metrics.steps_cancelled.saturating_add(1);
            self.cancel_pending_steps(session_id, Some(task_id))?;
            self.finish_cancelled(session_id);
            return Ok(DiffusionRunOutcome::SessionCancelled {
                session_id,
                task_id,
                step,
            });
        }

        let started_at = Instant::now();
        let execution = executor.execute_step(step);
        let elapsed_ns = u64::try_from(started_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.metrics.execution_duration_ns = self
            .metrics
            .execution_duration_ns
            .saturating_add(elapsed_ns);
        match step {
            DiffusionStep::PrepareBlock { .. } => {
                self.metrics.prepare_duration_ns =
                    self.metrics.prepare_duration_ns.saturating_add(elapsed_ns);
            }
            DiffusionStep::Denoise { .. } => {
                self.metrics.denoise_duration_ns =
                    self.metrics.denoise_duration_ns.saturating_add(elapsed_ns);
            }
            DiffusionStep::ValidateBlock { .. } => {
                self.metrics.validate_duration_ns =
                    self.metrics.validate_duration_ns.saturating_add(elapsed_ns);
            }
            DiffusionStep::CommitBlock { .. } => {
                self.metrics.commit_duration_ns =
                    self.metrics.commit_duration_ns.saturating_add(elapsed_ns);
            }
        }
        let cancelled_after_step = self.sessions[&session_id].cancellation.is_cancelled();
        match execution {
            Ok(()) => {
                // A successful bounded operation has taken effect, including CommitBlock.
                // Account for it before observing cancellation at the next boundary.
                self.scheduler.complete(task_id)?;
                self.metrics.steps_completed = self.metrics.steps_completed.saturating_add(1);
                let session = self
                    .sessions
                    .get_mut(&session_id)
                    .ok_or(DiffusionError::UnknownSession(session_id))?;
                session.snapshot.completed_steps += 1;
                if matches!(step, DiffusionStep::CommitBlock { .. }) {
                    session.completed_blocks = session.completed_blocks.saturating_add(1);
                    self.metrics.blocks_completed = self.metrics.blocks_completed.saturating_add(1);
                }
                session.snapshot.current_step = None;
                if session.snapshot.completed_steps == session.snapshot.total_steps {
                    session.snapshot.state = DiffusionSessionState::Completed;
                    self.metrics.sessions_completed =
                        self.metrics.sessions_completed.saturating_add(1);
                    Ok(DiffusionRunOutcome::SessionCompleted {
                        session_id,
                        task_id,
                        step,
                    })
                } else if cancelled_after_step {
                    self.cancel_pending_steps(session_id, None)?;
                    self.finish_cancelled(session_id);
                    Ok(DiffusionRunOutcome::SessionCancelled {
                        session_id,
                        task_id,
                        step,
                    })
                } else {
                    Ok(DiffusionRunOutcome::StepCompleted {
                        session_id,
                        task_id,
                        step,
                    })
                }
            }
            Err(error) => {
                self.scheduler.fail(task_id)?;
                self.metrics.steps_failed = self.metrics.steps_failed.saturating_add(1);
                let session_tasks = self.sessions[&session_id].tasks.clone();
                for dependent_id in session_tasks {
                    if self.scheduler.task(dependent_id)?.state == TaskState::Cancelled {
                        self.metrics.steps_cancelled =
                            self.metrics.steps_cancelled.saturating_add(1);
                    }
                }
                let session = self
                    .sessions
                    .get_mut(&session_id)
                    .ok_or(DiffusionError::UnknownSession(session_id))?;
                session.snapshot.state = DiffusionSessionState::Failed;
                session.snapshot.current_step = None;
                session.snapshot.error = Some(error.clone());
                self.metrics.sessions_failed = self.metrics.sessions_failed.saturating_add(1);
                Ok(DiffusionRunOutcome::StepFailed {
                    session_id,
                    task_id,
                    step,
                    error,
                })
            }
        }
    }

    /// Cancel a queued session. Running work is cancelled cooperatively through
    /// its token and is observed at the next safe step boundary.
    pub fn cancel(&mut self, id: DiffusionSessionId) -> Result<(), DiffusionError> {
        let session = self
            .sessions
            .get_mut(&id)
            .ok_or(DiffusionError::UnknownSession(id))?;
        if session.snapshot.state.is_terminal() {
            return Err(DiffusionError::SessionTerminal(id));
        }
        session.cancellation.cancel();
        let safe_boundary = session.snapshot.current_step.is_none();
        let queued = session.snapshot.state == DiffusionSessionState::Queued;
        self.cancel_pending_steps(id, None)?;
        if queued || safe_boundary {
            self.finish_cancelled(id);
        }
        Ok(())
    }

    fn cancel_pending_steps(
        &mut self,
        id: DiffusionSessionId,
        except: Option<TaskId>,
    ) -> Result<(), DiffusionError> {
        let tasks = self
            .sessions
            .get(&id)
            .ok_or(DiffusionError::UnknownSession(id))?
            .tasks
            .clone();
        let mut cancelled_before = 0_u64;
        for task_id in &tasks {
            if self.scheduler.task(*task_id)?.state == TaskState::Cancelled {
                cancelled_before = cancelled_before.saturating_add(1);
            }
        }
        for task_id in &tasks {
            if Some(*task_id) == except {
                continue;
            }
            if matches!(
                self.scheduler.task(*task_id)?.state,
                TaskState::Ready | TaskState::Blocked
            ) {
                self.scheduler.cancel(*task_id)?;
            }
        }
        let mut cancelled_after = 0_u64;
        for task_id in &tasks {
            if self.scheduler.task(*task_id)?.state == TaskState::Cancelled {
                cancelled_after = cancelled_after.saturating_add(1);
            }
        }
        self.metrics.steps_cancelled = self
            .metrics
            .steps_cancelled
            .saturating_add(cancelled_after.saturating_sub(cancelled_before));
        Ok(())
    }

    fn finish_cancelled(&mut self, id: DiffusionSessionId) {
        if let Some(session) = self.sessions.get_mut(&id) {
            if !session.snapshot.state.is_terminal() {
                session.snapshot.state = DiffusionSessionState::Cancelled;
                session.snapshot.current_step = None;
                self.metrics.sessions_cancelled = self.metrics.sessions_cancelled.saturating_add(1);
            }
        }
    }

    pub fn session(
        &self,
        id: DiffusionSessionId,
    ) -> Result<DiffusionSessionSnapshot, DiffusionError> {
        self.sessions
            .get(&id)
            .map(|session| session.snapshot.clone())
            .ok_or(DiffusionError::UnknownSession(id))
    }

    pub fn metrics(&self) -> DiffusionMetrics {
        let mut metrics = self.metrics;
        metrics.blocks_pending = self
            .sessions
            .values()
            .fold(0_u64, |pending, session| {
                if session.snapshot.state.is_terminal() {
                    pending
                } else {
                    let total = self
                        .plans
                        .get(&session.snapshot.id)
                        .map_or(0, DiffusionPlan::block_count);
                    pending.saturating_add(
                        u64::try_from(total.saturating_sub(session.completed_blocks))
                            .unwrap_or(u64::MAX),
                    )
                }
            });
        metrics
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Recorder(RefCell<Vec<DiffusionStep>>);
    impl DiffusionExecutor for Recorder {
        fn execute_step(&mut self, step: DiffusionStep) -> Result<(), String> {
            self.0.borrow_mut().push(step);
            Ok(())
        }
    }

    fn plan() -> DiffusionPlan {
        DiffusionPlan::new(vec![
            DiffusionStep::PrepareBlock { block: BlockId(0) },
            DiffusionStep::Denoise {
                block: BlockId(0),
                iteration: 0,
            },
            DiffusionStep::Denoise {
                block: BlockId(0),
                iteration: 1,
            },
            DiffusionStep::ValidateBlock { block: BlockId(0) },
            DiffusionStep::CommitBlock { block: BlockId(0) },
            DiffusionStep::PrepareBlock { block: BlockId(1) },
            DiffusionStep::Denoise {
                block: BlockId(1),
                iteration: 0,
            },
            DiffusionStep::ValidateBlock { block: BlockId(1) },
            DiffusionStep::CommitBlock { block: BlockId(1) },
        ])
        .unwrap()
    }

    #[test]
    fn plan_rejects_missing_validation_and_non_contiguous_denoising() {
        assert!(DiffusionPlan::new(vec![
            DiffusionStep::PrepareBlock { block: BlockId(0) },
            DiffusionStep::Denoise {
                block: BlockId(0),
                iteration: 1
            },
        ])
        .is_err());
    }

    #[test]
    fn executes_multiple_blocks_in_deterministic_order() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        loop {
            match orchestrator.run_one(&mut recorder).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Completed
        );
        assert_eq!(recorder.0.borrow().len(), 9);
        assert_eq!(orchestrator.metrics().steps_completed, 9);
    }

    #[test]
    fn failure_cancels_remaining_dependent_steps() {
        struct FailOnce(bool);
        impl DiffusionExecutor for FailOnce {
            fn execute_step(&mut self, _step: DiffusionStep) -> Result<(), String> {
                if !self.0 {
                    self.0 = true;
                    Err("synthetic failure".into())
                } else {
                    Ok(())
                }
            }
        }
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        assert!(matches!(
            orchestrator.run_one(&mut FailOnce(false)).unwrap(),
            DiffusionRunOutcome::StepFailed { .. }
        ));
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Failed
        );
        assert!(matches!(
            orchestrator.run_one(&mut FailOnce(true)).unwrap(),
            DiffusionRunOutcome::Idle
        ));
        for task_id in &orchestrator.sessions[&id].tasks {
            assert!(matches!(
                orchestrator.scheduler().task(*task_id).unwrap().state,
                TaskState::Failed | TaskState::Cancelled
            ));
        }
    }

    #[test]
    fn cancellation_during_successful_step_counts_effect_before_cancel() {
        let token = CancellationToken::new();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal)
            .unwrap();
        struct CancelAfterStep(CancellationToken);
        impl DiffusionExecutor for CancelAfterStep {
            fn execute_step(&mut self, _step: DiffusionStep) -> Result<(), String> {
                self.0.cancel();
                Ok(())
            }
        }
        assert!(matches!(
            orchestrator.run_one(&mut CancelAfterStep(token)).unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        let snapshot = orchestrator.session(id).unwrap();
        assert_eq!(snapshot.completed_steps, 1);
        assert_eq!(snapshot.state, DiffusionSessionState::Cancelled);
        assert_eq!(orchestrator.metrics().steps_completed, 1);
        assert_eq!(orchestrator.metrics().steps_cancelled, 8);
    }

    #[test]
    fn pre_cancelled_session_is_terminal_without_scheduler_tasks() {
        let token = CancellationToken::new();
        token.cancel();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan(), token, TaskPriority::Normal)
            .unwrap();
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
        assert_eq!(orchestrator.scheduler().ready_count(TaskKind::Compute), 0);
    }

    #[test]
    fn token_cancellation_after_admission_prevents_step_execution() {
        let token = CancellationToken::new();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal)
            .unwrap();
        token.cancel();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        assert!(matches!(
            orchestrator.run_one(&mut recorder).unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        assert!(recorder.0.borrow().is_empty());
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
    }

    #[test]
    fn queued_session_can_be_cancelled_before_execution() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        orchestrator.cancel(id).unwrap();
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
        assert_eq!(orchestrator.scheduler().ready_count(TaskKind::Compute), 0);
    }

    #[test]
    fn equal_priority_sessions_are_fifo_and_not_interleaved() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let first = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        let second = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal)
            .unwrap();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        while orchestrator.session(first).unwrap().state != DiffusionSessionState::Completed {
            orchestrator.run_one(&mut recorder).unwrap();
        }
        assert_eq!(
            orchestrator.session(second).unwrap().state,
            DiffusionSessionState::Queued
        );
        assert_eq!(recorder.0.borrow().len(), plan().steps().len());
    }

    #[test]
    fn invalid_plans_reject_empty_wrong_block_and_missing_validation() {
        assert!(DiffusionPlan::new(vec![]).is_err());
        assert!(DiffusionPlan::new(vec![
            DiffusionStep::PrepareBlock { block: BlockId(1) },
            DiffusionStep::Denoise {
                block: BlockId(1),
                iteration: 0
            },
            DiffusionStep::ValidateBlock { block: BlockId(1) },
            DiffusionStep::CommitBlock { block: BlockId(1) },
        ])
        .is_err());
        assert!(DiffusionPlan::new(vec![
            DiffusionStep::PrepareBlock { block: BlockId(0) },
            DiffusionStep::Denoise {
                block: BlockId(0),
                iteration: 0
            },
            DiffusionStep::CommitBlock { block: BlockId(0) },
        ])
        .is_err());
    }

    #[test]
    fn failure_does_not_cancel_an_independent_session() {
        struct FailFirst(bool);
        impl DiffusionExecutor for FailFirst {
            fn execute_step(&mut self, _step: DiffusionStep) -> Result<(), String> {
                if !self.0 {
                    self.0 = true;
                    Err("failure".into())
                } else {
                    Ok(())
                }
            }
        }
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let failed = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::High)
            .unwrap();
        let independent = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Low)
            .unwrap();
        assert!(matches!(
            orchestrator.run_one(&mut FailFirst(false)).unwrap(),
            DiffusionRunOutcome::StepFailed { .. }
        ));
        assert_eq!(
            orchestrator.session(failed).unwrap().state,
            DiffusionSessionState::Failed
        );
        assert_eq!(
            orchestrator.session(independent).unwrap().state,
            DiffusionSessionState::Queued
        );
        assert!(matches!(
            orchestrator.run_one(&mut FailFirst(true)).unwrap(),
            DiffusionRunOutcome::StepCompleted { .. }
        ));
    }

    #[test]
    fn cancellation_is_observed_at_step_boundary() {
        let token = CancellationToken::new();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let id = orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal)
            .unwrap();
        struct CancelAfterStep(CancellationToken);
        impl DiffusionExecutor for CancelAfterStep {
            fn execute_step(&mut self, _step: DiffusionStep) -> Result<(), String> {
                self.0.cancel();
                Ok(())
            }
        }
        assert!(matches!(
            orchestrator.run_one(&mut CancelAfterStep(token)).unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
    }
}
