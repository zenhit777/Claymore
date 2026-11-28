//! Shared, model-neutral orchestration for block-based diffusion generation.
//!
//! The orchestrator owns lifecycle metadata and scheduler task IDs only. It
//! never owns tensors, device buffers, CUDA handles, or model weights. Model
//! adapters provide the work for each bounded step through `DiffusionExecutor`.
//! Cancellation is observed at step boundaries, so an in-flight backend call is
//! never reported cancelled before the executor returns.
//!
//! Session↔executor binding (M11.4 review P1/P2): every session is bound at
//! submit time to a `DiffusionExecutorKey` issued by this orchestrator, and
//! `run_one` only executes steps of, and delivers terminal notifications for,
//! the sessions bound to the key it is given. A step of session A can therefore
//! never be executed by adapter B, and a terminal notification of A can never
//! reset the caches or residency owned by adapter B.
use std::{
    collections::{BTreeMap, BTreeSet},
    panic::{catch_unwind, AssertUnwindSafe},
    time::Instant,
};

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

/// Typed scheduler identity for one diffusion step task (M11.4 review P7).
///
/// `ScheduledRuntime` owns two schedulers (requests/transfer and diffusion)
/// whose opaque `claymore_scheduler::TaskId` counters collide by construction.
/// The diffusion side never hands its raw ids out: passing a diffusion task
/// identifier to a request-side API (`cancel_queued`) is a type error, and the
/// reverse is impossible because the diffusion API accepts no raw `TaskId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiffusionTaskId(TaskId);

/// Handle for one adapter instance paired with one orchestrator
/// (M11.4 review P1/P2).
///
/// Keys are issued by [`DiffusionOrchestrator::register_executor`], are valid
/// only for the issuing orchestrator, and identify exactly one live session at
/// a time. Because the adapter instance behind a key owns request-local state
/// (Clover request caches, Gemma canvas), binding a second open session to the
/// same key is rejected until the first is closed and reclaimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiffusionExecutorKey(u64);

impl DiffusionExecutorKey {
    pub const fn get(self) -> u64 {
        self.0
    }
}

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

/// Explicit result of one bounded step (M11.4 review P10).
///
/// An adapter that reaches its own terminal condition (for example the model
/// emitted EOS mid-plan) short-circuits the remaining nominal steps. The
/// orchestrator must not count such short-circuits as completed work or as
/// committed blocks: it only accounts for work the adapter reports `Executed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionStepOutcome {
    /// The adapter performed the unit of work requested by this step.
    Executed,
    /// The adapter is already in a terminal internal state and performed
    /// nothing. The scheduler task completes and the plan advances, but the
    /// step is counted as skipped, never as effective work.
    Skipped,
}

/// Executor boundary implemented by a model adapter. Implementations should
/// perform one bounded unit of work and return only after it is safe to advance.
pub trait DiffusionExecutor {
    fn execute_step(&mut self, step: DiffusionStep) -> Result<DiffusionStepOutcome, String>;

    /// Release adapter-owned request state after a terminal session. The
    /// callback runs at a safe step boundary on the adapter bound to the
    /// session, and is also flushed by `cancel_with_executor` for sessions
    /// cancelled before their first step.
    ///
    /// Returning `Err` does not change the session's terminal state, but the
    /// cleanup failure is recorded on the session snapshot and counted in
    /// `DiffusionMetrics::cleanups_failed`, so a failed release is never
    /// invisible to the host (M11.4 review P4).
    fn on_session_end(
        &mut self,
        _session_id: DiffusionSessionId,
        _state: DiffusionSessionState,
    ) -> Result<(), String> {
        Ok(())
    }
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
    /// Nominal steps the adapter short-circuited while already stopped.
    pub steps_skipped: u64,
    /// Terminal notifications whose adapter cleanup returned `Err`.
    pub cleanups_failed: u64,
    /// Cumulative number of blocks admitted across all sessions.
    pub blocks_submitted: u64,
    /// Cumulative number of successfully committed blocks. A block counts
    /// only when its `CommitBlock` step reported [`DiffusionStepOutcome::Executed`].
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
        task_id: DiffusionTaskId,
        step: DiffusionStep,
    },
    SessionCompleted {
        session_id: DiffusionSessionId,
        task_id: DiffusionTaskId,
        step: DiffusionStep,
    },
    StepFailed {
        session_id: DiffusionSessionId,
        task_id: DiffusionTaskId,
        step: DiffusionStep,
        error: String,
    },
    SessionCancelled {
        session_id: DiffusionSessionId,
        task_id: DiffusionTaskId,
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
    #[error("diffusion executor key {0:?} was not issued by this orchestrator")]
    UnknownExecutor(DiffusionExecutorKey),
    #[error(
        "diffusion executor key {executor:?} still owns open session {session:?}; \
         one adapter instance serves exactly one session at a time"
    )]
    ExecutorBusy {
        executor: DiffusionExecutorKey,
        session: DiffusionSessionId,
    },
    #[error("diffusion session {0:?} is not bound to executor key {1:?}")]
    ExecutorMismatch(DiffusionSessionId, DiffusionExecutorKey),
    #[error("scheduler returned an unowned diffusion task {0:?}")]
    UnknownTask(DiffusionTaskId),
    #[error("diffusion plan for session {0:?} is missing")]
    MissingPlan(DiffusionSessionId),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
}

/// Outcome of [`DiffusionOrchestrator::reclaim_closed_sessions`] (P8).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiffusionReclaimReport {
    /// Closed sessions whose state was removed atomically.
    pub reclaimed: Vec<DiffusionSessionId>,
    /// Closed sessions retained because their terminal notification has not
    /// been delivered to the owning adapter yet. Driving `run_one` with the
    /// owning key flushes them; only then can the session be reclaimed.
    pub deferred_terminal_notification: Vec<DiffusionSessionId>,
}

struct SessionRecord {
    snapshot: DiffusionSessionSnapshot,
    cancellation: CancellationToken,
    tasks: Vec<TaskId>,
    completed_blocks: usize,
    terminal_notification_pending: bool,
    executor: DiffusionExecutorKey,
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
    executors: BTreeSet<DiffusionExecutorKey>,
    key_open_session: BTreeMap<DiffusionExecutorKey, DiffusionSessionId>,
    next_session_id: u64,
    next_executor_key: u64,
    metrics: DiffusionMetrics,
}

impl DiffusionOrchestrator {
    pub fn new(config: SchedulerConfig) -> Result<Self, DiffusionError> {
        Ok(Self {
            scheduler: Scheduler::new(config)?,
            sessions: BTreeMap::new(),
            plans: BTreeMap::new(),
            task_index: BTreeMap::new(),
            executors: BTreeSet::new(),
            key_open_session: BTreeMap::new(),
            next_session_id: 1,
            next_executor_key: 1,
            metrics: DiffusionMetrics::default(),
        })
    }

    /// Declare one adapter instance that `submit`/`run_one` may be paired with.
    /// Issue exactly one key per adapter instance for the lifetime of that
    /// instance; keys are cheap, monotonic, and never reused.
    pub fn register_executor(&mut self) -> DiffusionExecutorKey {
        let key = DiffusionExecutorKey(self.next_executor_key);
        self.next_executor_key = self
            .next_executor_key
            .saturating_add(1)
            .max(self.next_executor_key);
        self.executors.insert(key);
        key
    }

    /// Admit a plan as a serial dependency chain bound to `executor`.
    /// Equal-priority sessions are FIFO by admission: an earlier session's full
    /// chain precedes a later session. Higher-priority ready work may run first.
    ///
    /// One key serves at most one open session: a second concurrent submit on
    /// the same key is rejected with [`DiffusionError::ExecutorBusy`] because
    /// the adapter instance behind the key holds request-local state.
    pub fn submit(
        &mut self,
        plan: DiffusionPlan,
        cancellation: CancellationToken,
        priority: TaskPriority,
        executor: DiffusionExecutorKey,
    ) -> Result<DiffusionSessionId, DiffusionError> {
        if !self.executors.contains(&executor) {
            return Err(DiffusionError::UnknownExecutor(executor));
        }
        if let Some(&busy) = self.key_open_session.get(&executor) {
            if self.sessions.contains_key(&busy) {
                return Err(DiffusionError::ExecutorBusy {
                    executor,
                    session: busy,
                });
            }
        }

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
                    terminal_notification_pending: true,
                    executor,
                },
            );
            self.key_open_session.insert(executor, id);
            self.metrics.sessions_submitted = self.metrics.sessions_submitted.saturating_add(1);
            self.metrics.blocks_submitted = self
                .metrics
                .blocks_submitted
                .saturating_add(u64::try_from(plan.block_count()).unwrap_or(u64::MAX));
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
                terminal_notification_pending: false,
                executor,
            },
        );
        self.key_open_session.insert(executor, id);
        let block_count = plan.block_count();
        self.plans.insert(id, plan);
        self.metrics.sessions_submitted = self.metrics.sessions_submitted.saturating_add(1);
        self.metrics.blocks_submitted = self
            .metrics
            .blocks_submitted
            .saturating_add(u64::try_from(block_count).unwrap_or(u64::MAX));
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

    /// Run one ready step of a session bound to `executor`, on the adapter
    /// paired with that key. Sessions owned by other keys are never selected,
    /// so `adapter` can only ever receive steps of its own session. Pending
    /// terminal notifications for the key's own closed sessions are flushed
    /// first, in deterministic session-id order (P2/P6).
    pub fn run_one<E: DiffusionExecutor>(
        &mut self,
        executor: DiffusionExecutorKey,
        adapter: &mut E,
    ) -> Result<DiffusionRunOutcome, DiffusionError> {
        if !self.executors.contains(&executor) {
            return Err(DiffusionError::UnknownExecutor(executor));
        }
        self.flush_terminal_notifications(executor, adapter);
        let started = {
            let Self {
                scheduler,
                task_index,
                sessions,
                ..
            } = self;
            scheduler.start_next_where(TaskKind::Compute, |task_id| {
                match task_index.get(&task_id) {
                    // An unowned task must stay consumable so an invariant
                    // violation fails loudly through the normal error path.
                    None => true,
                    Some((session_id, _)) => sessions
                        .get(session_id)
                        .is_some_and(|session| session.executor == executor),
                }
            })
        };
        let Some(raw_task_id) = started else {
            return Ok(DiffusionRunOutcome::Idle);
        };
        let task_id = DiffusionTaskId(raw_task_id);
        let Some((session_id, index)) = self.task_index.get(&raw_task_id).copied() else {
            self.scheduler.fail(raw_task_id)?;
            return Err(DiffusionError::UnknownTask(task_id));
        };
        let Some(step) = self
            .plans
            .get(&session_id)
            .and_then(|p| p.steps.get(index))
            .copied()
        else {
            self.scheduler.fail(raw_task_id)?;
            return Err(DiffusionError::MissingPlan(session_id));
        };
        let Some(session) = self.sessions.get(&session_id) else {
            self.scheduler.fail(raw_task_id)?;
            return Err(DiffusionError::UnknownSession(session_id));
        };
        let cancelled_before_step = session.cancellation.is_cancelled();
        if let Some(session) = self.sessions.get_mut(&session_id) {
            session.snapshot.state = DiffusionSessionState::Running;
            session.snapshot.current_step = Some(step);
        }
        self.metrics.steps_started = self.metrics.steps_started.saturating_add(1);

        if cancelled_before_step {
            // Cancellation is observed at a safe boundary before any work:
            // the running task and every still-scheduled dependent end
            // cancelled. Count with a before/after delta around the scheduler
            // transitions so no task is counted twice and none is missed (P5).
            let cancelled_before = self.cancelled_task_count(session_id);
            self.scheduler.cancel_running(raw_task_id)?;
            self.cancel_session_tasks(session_id)?;
            let cancelled_after = self.cancelled_task_count(session_id);
            self.metrics.steps_cancelled = self
                .metrics
                .steps_cancelled
                .saturating_add(cancelled_after.saturating_sub(cancelled_before));
            self.finish_cancelled(session_id);
            self.notify_session_end(session_id, executor, adapter);
            return Ok(DiffusionRunOutcome::SessionCancelled {
                session_id,
                task_id,
                step,
            });
        }

        let started_at = Instant::now();
        // A panic must not strand the session task in Running forever: unwind is
        // converted into the same failed-step path as an ordinary executor error,
        // so the session becomes terminal and `on_session_end` still runs
        // (residency and cache release depend on that notification).
        let execution = catch_unwind(AssertUnwindSafe(|| adapter.execute_step(step)))
            .unwrap_or_else(|payload| {
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("diffusion executor panicked with a non-string payload");
                Err(format!("executor panicked: {message}"))
            });
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
            Ok(outcome) => {
                // A successful bounded operation has taken effect: the scheduler
                // task completes whatever the adapter reported, and the plan
                // advances so a stopped adapter still closes its session.
                self.scheduler.complete(raw_task_id)?;
                // Account for effective work only (P10): a short-circuited step
                // never counts as a completed step nor commits its block.
                match outcome {
                    DiffusionStepOutcome::Executed => {
                        self.metrics.steps_completed =
                            self.metrics.steps_completed.saturating_add(1);
                    }
                    DiffusionStepOutcome::Skipped => {
                        self.metrics.steps_skipped = self.metrics.steps_skipped.saturating_add(1);
                    }
                }
                let session = self
                    .sessions
                    .get_mut(&session_id)
                    .ok_or(DiffusionError::UnknownSession(session_id))?;
                session.snapshot.completed_steps += 1;
                if matches!(step, DiffusionStep::CommitBlock { .. })
                    && matches!(outcome, DiffusionStepOutcome::Executed)
                {
                    session.completed_blocks = session.completed_blocks.saturating_add(1);
                    self.metrics.blocks_completed = self.metrics.blocks_completed.saturating_add(1);
                }
                session.snapshot.current_step = None;
                if session.snapshot.completed_steps == session.snapshot.total_steps {
                    session.snapshot.state = DiffusionSessionState::Completed;
                    session.terminal_notification_pending = true;
                    self.metrics.sessions_completed =
                        self.metrics.sessions_completed.saturating_add(1);
                    self.notify_session_end(session_id, executor, adapter);
                    Ok(DiffusionRunOutcome::SessionCompleted {
                        session_id,
                        task_id,
                        step,
                    })
                } else if cancelled_after_step {
                    self.cancel_and_count(session_id)?;
                    self.finish_cancelled(session_id);
                    self.notify_session_end(session_id, executor, adapter);
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
                // Dependants cascade to Cancelled through the scheduler; the
                // delta around `fail` counts exactly the tasks this failure
                // cancelled (P5).
                let cancelled_before = self.cancelled_task_count(session_id);
                self.scheduler.fail(raw_task_id)?;
                let cancelled_after = self.cancelled_task_count(session_id);
                self.metrics.steps_failed = self.metrics.steps_failed.saturating_add(1);
                self.metrics.steps_cancelled = self
                    .metrics
                    .steps_cancelled
                    .saturating_add(cancelled_after.saturating_sub(cancelled_before));
                let session = self
                    .sessions
                    .get_mut(&session_id)
                    .ok_or(DiffusionError::UnknownSession(session_id))?;
                session.snapshot.state = DiffusionSessionState::Failed;
                session.snapshot.current_step = None;
                session.snapshot.error = Some(error.clone());
                session.terminal_notification_pending = true;
                self.metrics.sessions_failed = self.metrics.sessions_failed.saturating_add(1);
                self.notify_session_end(session_id, executor, adapter);
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
        self.cancel_and_count(id)?;
        if queued || safe_boundary {
            self.finish_cancelled(id);
        }
        Ok(())
    }

    fn cancelled_task_count(&self, id: DiffusionSessionId) -> u64 {
        let Some(session) = self.sessions.get(&id) else {
            return 0;
        };
        let mut count = 0_u64;
        for task_id in &session.tasks {
            if let Ok(snapshot) = self.scheduler.task(*task_id) {
                if snapshot.state == TaskState::Cancelled {
                    count = count.saturating_add(1);
                }
            }
        }
        count
    }

    /// Cancel every still-schedulable task of the session and add the exact
    /// number of tasks that newly reached `Cancelled` (P5).
    fn cancel_and_count(&mut self, id: DiffusionSessionId) -> Result<(), DiffusionError> {
        let cancelled_before = self.cancelled_task_count(id);
        self.cancel_session_tasks(id)?;
        let cancelled_after = self.cancelled_task_count(id);
        self.metrics.steps_cancelled = self
            .metrics
            .steps_cancelled
            .saturating_add(cancelled_after.saturating_sub(cancelled_before));
        Ok(())
    }

    fn cancel_session_tasks(&mut self, id: DiffusionSessionId) -> Result<(), DiffusionError> {
        let tasks = self
            .sessions
            .get(&id)
            .ok_or(DiffusionError::UnknownSession(id))?
            .tasks
            .clone();
        for task_id in &tasks {
            if matches!(
                self.scheduler.task(*task_id)?.state,
                TaskState::Ready | TaskState::Blocked
            ) {
                self.scheduler.cancel(*task_id)?;
            }
        }
        Ok(())
    }

    fn finish_cancelled(&mut self, id: DiffusionSessionId) {
        if let Some(session) = self.sessions.get_mut(&id) {
            if !session.snapshot.state.is_terminal() {
                session.snapshot.state = DiffusionSessionState::Cancelled;
                session.snapshot.current_step = None;
                session.terminal_notification_pending = true;
                self.metrics.sessions_cancelled = self.metrics.sessions_cancelled.saturating_add(1);
            }
        }
    }

    /// Deliver the terminal notification of `id` — but only to the adapter that
    /// owns the session (P2). A cleanup `Err` never reopens or reinterprets the
    /// session; it is recorded on the snapshot and counted (P4).
    fn notify_session_end<E: DiffusionExecutor>(
        &mut self,
        id: DiffusionSessionId,
        executor: DiffusionExecutorKey,
        adapter: &mut E,
    ) {
        {
            let Some(session) = self.sessions.get_mut(&id) else {
                return;
            };
            if session.executor != executor || !session.terminal_notification_pending {
                return;
            }
            session.terminal_notification_pending = false;
        }
        let state = self.sessions[&id].snapshot.state;
        if let Err(error) = adapter.on_session_end(id, state) {
            self.metrics.cleanups_failed = self.metrics.cleanups_failed.saturating_add(1);
            if let Some(session) = self.sessions.get_mut(&id) {
                session.snapshot.error = Some(match session.snapshot.error.take() {
                    Some(existing) => format!("{existing}; session cleanup failed: {error}"),
                    None => format!("session cleanup failed: {error}"),
                });
            }
        }
    }

    /// Flush every pending terminal notification of sessions bound to
    /// `executor`, in deterministic session-id order (P6). Notifications of
    /// other keys are untouched and stay pending until their owner is driven.
    fn flush_terminal_notifications<E: DiffusionExecutor>(
        &mut self,
        executor: DiffusionExecutorKey,
        adapter: &mut E,
    ) {
        let pending: Vec<DiffusionSessionId> = self
            .sessions
            .iter()
            .filter(|(_, session)| {
                session.terminal_notification_pending && session.executor == executor
            })
            .map(|(id, _)| *id)
            .collect();
        for id in pending {
            self.notify_session_end(id, executor, adapter);
        }
    }

    /// Cancel a session and synchronously notify the owning adapter when
    /// cancellation reaches a terminal state immediately. Running work remains
    /// cooperative; its callback is issued by run_one after the in-flight step
    /// returns. `executor` must be the key bound to `id`.
    pub fn cancel_with_executor<E: DiffusionExecutor>(
        &mut self,
        id: DiffusionSessionId,
        executor: DiffusionExecutorKey,
        adapter: &mut E,
    ) -> Result<(), DiffusionError> {
        let bound = {
            let session = self
                .sessions
                .get(&id)
                .ok_or(DiffusionError::UnknownSession(id))?;
            session.executor
        };
        if !self.executors.contains(&executor) {
            return Err(DiffusionError::UnknownExecutor(executor));
        }
        if bound != executor {
            return Err(DiffusionError::ExecutorMismatch(id, executor));
        }
        self.cancel(id)?;
        self.notify_session_end(id, executor, adapter);
        Ok(())
    }

    /// Explicit bounded retention (P8): remove orchestration state for closed
    /// sessions — the session record, its plan, and its task-index entries,
    /// plus the scheduler's terminal task records — atomically per session.
    ///
    /// A closed session is never reclaimed before its terminal notification
    /// has been delivered to the owning adapter, so a host cannot lose a
    /// result or skip a residency release: deliver notifications by driving
    /// `run_one` with the owning key, then reclaim. Session snapshots stay
    /// readable through `session()` until they are reclaimed.
    pub fn reclaim_closed_sessions(&mut self) -> DiffusionReclaimReport {
        let closed: Vec<(DiffusionSessionId, DiffusionExecutorKey, Vec<TaskId>)> = self
            .sessions
            .iter()
            .filter(|(_, session)| session.snapshot.state.is_terminal())
            .map(|(id, session)| (*id, session.executor, session.tasks.clone()))
            .collect();
        let mut report = DiffusionReclaimReport::default();
        for (id, executor, tasks) in closed {
            if self
                .sessions
                .get(&id)
                .is_some_and(|session| session.terminal_notification_pending)
            {
                report.deferred_terminal_notification.push(id);
                continue;
            }
            self.scheduler.forget_terminal_tasks(&tasks);
            for task_id in &tasks {
                self.task_index.remove(task_id);
            }
            self.sessions.remove(&id);
            self.plans.remove(&id);
            if self.key_open_session.get(&executor) == Some(&id) {
                self.key_open_session.remove(&executor);
            }
            report.reclaimed.push(id);
        }
        report
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
        metrics.blocks_pending = self.sessions.values().fold(0_u64, |pending, session| {
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

    #[cfg(test)]
    pub(crate) fn retained_sessions_for_test(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Recorder(RefCell<Vec<DiffusionStep>>);
    impl DiffusionExecutor for Recorder {
        fn execute_step(&mut self, step: DiffusionStep) -> Result<DiffusionStepOutcome, String> {
            self.0.borrow_mut().push(step);
            Ok(DiffusionStepOutcome::Executed)
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
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        loop {
            match orchestrator.run_one(key, &mut recorder).unwrap() {
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
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                if !self.0 {
                    self.0 = true;
                    Err("synthetic failure".into())
                } else {
                    Ok(DiffusionStepOutcome::Executed)
                }
            }
        }
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        assert!(matches!(
            orchestrator.run_one(key, &mut FailOnce(false)).unwrap(),
            DiffusionRunOutcome::StepFailed { .. }
        ));
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Failed
        );
        assert!(matches!(
            orchestrator.run_one(key, &mut FailOnce(true)).unwrap(),
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
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal, key)
            .unwrap();
        struct CancelAfterStep(CancellationToken);
        impl DiffusionExecutor for CancelAfterStep {
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                self.0.cancel();
                Ok(DiffusionStepOutcome::Executed)
            }
        }
        assert!(matches!(
            orchestrator
                .run_one(key, &mut CancelAfterStep(token))
                .unwrap(),
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
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), token, TaskPriority::Normal, key)
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
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal, key)
            .unwrap();
        token.cancel();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        assert!(matches!(
            orchestrator.run_one(key, &mut recorder).unwrap(),
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
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
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
        let key_a = orchestrator.register_executor();
        let key_b = orchestrator.register_executor();
        let first = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_a,
            )
            .unwrap();
        let second = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_b,
            )
            .unwrap();
        let mut recorder_a = Recorder(RefCell::new(Vec::new()));
        let recorder_b = Recorder(RefCell::new(Vec::new()));
        while orchestrator.session(first).unwrap().state != DiffusionSessionState::Completed {
            orchestrator.run_one(key_a, &mut recorder_a).unwrap();
        }
        assert_eq!(
            orchestrator.session(second).unwrap().state,
            DiffusionSessionState::Queued
        );
        assert_eq!(recorder_a.0.borrow().len(), plan().steps().len());
        assert!(recorder_b.0.borrow().is_empty());
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
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                if !self.0 {
                    self.0 = true;
                    Err("failure".into())
                } else {
                    Ok(DiffusionStepOutcome::Executed)
                }
            }
        }
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key_failed = orchestrator.register_executor();
        let key_independent = orchestrator.register_executor();
        let failed = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::High,
                key_failed,
            )
            .unwrap();
        let independent = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Low,
                key_independent,
            )
            .unwrap();
        assert!(matches!(
            orchestrator
                .run_one(key_failed, &mut FailFirst(false))
                .unwrap(),
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
            orchestrator
                .run_one(key_independent, &mut FailFirst(true))
                .unwrap(),
            DiffusionRunOutcome::StepCompleted { .. }
        ));
    }

    #[test]
    fn cancellation_is_observed_at_step_boundary() {
        let token = CancellationToken::new();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal, key)
            .unwrap();
        struct CancelAfterStep(CancellationToken);
        impl DiffusionExecutor for CancelAfterStep {
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                self.0.cancel();
                Ok(DiffusionStepOutcome::Executed)
            }
        }
        assert!(matches!(
            orchestrator
                .run_one(key, &mut CancelAfterStep(token))
                .unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
    }

    #[derive(Default)]
    struct CleanupRecorder {
        ended: Vec<(DiffusionSessionId, DiffusionSessionState)>,
    }

    impl DiffusionExecutor for CleanupRecorder {
        fn execute_step(&mut self, _step: DiffusionStep) -> Result<DiffusionStepOutcome, String> {
            Ok(DiffusionStepOutcome::Executed)
        }

        fn on_session_end(
            &mut self,
            id: DiffusionSessionId,
            state: DiffusionSessionState,
        ) -> Result<(), String> {
            self.ended.push((id, state));
            Ok(())
        }
    }

    #[test]
    fn queued_cancel_flushes_adapter_cleanup_on_next_safe_runtime_call() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        orchestrator.cancel(id).unwrap();
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );

        let mut recorder = CleanupRecorder::default();
        assert_eq!(
            orchestrator.run_one(key, &mut recorder).unwrap(),
            DiffusionRunOutcome::Idle
        );
        assert_eq!(recorder.ended, vec![(id, DiffusionSessionState::Cancelled)]);
    }

    #[test]
    fn cancel_with_executor_flushes_cleanup_immediately_at_a_safe_boundary() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        let mut recorder = CleanupRecorder::default();
        orchestrator
            .cancel_with_executor(id, key, &mut recorder)
            .unwrap();
        assert_eq!(recorder.ended, vec![(id, DiffusionSessionState::Cancelled)]);
    }

    #[test]
    fn cancel_with_executor_rejects_a_foreign_key() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key_a = orchestrator.register_executor();
        let key_b = orchestrator.register_executor();
        let id = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_a,
            )
            .unwrap();
        let mut foreign = CleanupRecorder::default();
        assert_eq!(
            orchestrator.cancel_with_executor(id, key_b, &mut foreign),
            Err(DiffusionError::ExecutorMismatch(id, key_b))
        );
        // The session was NOT cancelled by the foreign adapter: the caller must
        // go through the owning key or the plain cancel path.
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Queued
        );
        assert!(foreign.ended.is_empty());
        // A plain cancel still works and defers the notification to the owner.
        orchestrator.cancel(id).unwrap();
        let mut owner = CleanupRecorder::default();
        orchestrator.run_one(key_a, &mut owner).unwrap();
        assert_eq!(owner.ended, vec![(id, DiffusionSessionState::Cancelled)]);
    }

    #[test]
    fn cancellation_during_a_step_notifies_adapter_after_the_step_returns() {
        struct CancelDuringStep {
            token: CancellationToken,
            ended: Vec<DiffusionSessionState>,
        }
        impl DiffusionExecutor for CancelDuringStep {
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                self.token.cancel();
                Ok(DiffusionStepOutcome::Executed)
            }
            fn on_session_end(
                &mut self,
                _id: DiffusionSessionId,
                state: DiffusionSessionState,
            ) -> Result<(), String> {
                self.ended.push(state);
                Ok(())
            }
        }

        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let token = CancellationToken::new();
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal, key)
            .unwrap();
        let mut executor = CancelDuringStep {
            token,
            ended: Vec::new(),
        };
        assert!(matches!(
            orchestrator.run_one(key, &mut executor).unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        assert_eq!(
            orchestrator.session(id).unwrap().state,
            DiffusionSessionState::Cancelled
        );
        assert_eq!(executor.ended, vec![DiffusionSessionState::Cancelled]);
    }

    #[derive(Default)]
    struct Panicking {
        ended: Vec<DiffusionSessionState>,
    }
    impl DiffusionExecutor for Panicking {
        fn execute_step(&mut self, _step: DiffusionStep) -> Result<DiffusionStepOutcome, String> {
            panic!("synthetic adapter panic");
        }
        fn on_session_end(
            &mut self,
            _id: DiffusionSessionId,
            state: DiffusionSessionState,
        ) -> Result<(), String> {
            self.ended.push(state);
            Ok(())
        }
    }

    /// A panic inside an adapter is a bug in that adapter, not a licence to lose
    /// the session: without containment the task stays `Running` forever, the
    /// session never joins the closed set, and `on_session_end` (residency and
    /// cache release) is never called.
    #[test]
    fn a_panicking_adapter_fails_the_session_instead_of_leaking_the_task() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let id = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        let mut adapter = Panicking::default();
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            orchestrator.run_one(key, &mut adapter)
        }));
        std::panic::set_hook(hook);

        let outcome = outcome
            .expect("run_one must not unwind out of the orchestrator")
            .unwrap();
        assert!(
            matches!(outcome, DiffusionRunOutcome::StepFailed { .. }),
            "unexpected outcome: {outcome:?}"
        );
        let session = orchestrator.session(id).unwrap();
        assert_eq!(session.state, DiffusionSessionState::Failed);
        let reason = session.error.clone().expect("a panic must be recorded");
        assert!(reason.contains("panicked"), "unhelpful reason: {reason}");
        // The failed session leaves no runnable task, and the terminal
        // notification still reaches the adapter.
        assert!(
            matches!(
                orchestrator.run_one(key, &mut adapter),
                Ok(DiffusionRunOutcome::Idle)
            ),
            "the failed session must not keep a task queued"
        );
        assert_eq!(adapter.ended, vec![DiffusionSessionState::Failed]);
    }

    // ---- M11.4 review corrections: green-phase regression tests ------------
    //
    // These are the new-API forms of the red tests captured in
    // `tests/m11_4_review_regressions.rs` (see docs/audits §9). Each failed on
    // the pre-fix code; here they pin the corrected guarantees.

    /// P1: with two sessions and two adapters, the runnable step of session A
    /// must never be executed by adapter B, even when B is the only adapter
    /// handed to `run_one` while only A has ready work.
    #[test]
    fn a_step_of_one_session_never_runs_on_another_sessions_adapter() {
        #[derive(Default)]
        struct Tagged {
            calls: Vec<DiffusionStep>,
        }
        impl DiffusionExecutor for Tagged {
            fn execute_step(
                &mut self,
                step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                self.calls.push(step);
                Ok(DiffusionStepOutcome::Executed)
            }
        }

        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key_a = orchestrator.register_executor();
        let key_b = orchestrator.register_executor();
        let session_a = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_a,
            )
            .unwrap();
        let session_b = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_b,
            )
            .unwrap();

        // Deliberately alternate adapters: A runs three steps, then only B is
        // driven. Adapter B must receive B's work and never A's work.
        let mut adapter_a = Tagged::default();
        let mut adapter_b = Tagged::default();
        for _ in 0..3 {
            orchestrator.run_one(key_a, &mut adapter_a).unwrap();
        }
        assert_eq!(orchestrator.session(session_a).unwrap().completed_steps, 3);
        assert_eq!(orchestrator.session(session_b).unwrap().completed_steps, 0);

        orchestrator.run_one(key_b, &mut adapter_b).unwrap();
        assert_eq!(adapter_b.calls, vec![plan().steps()[0]]);
        orchestrator.run_one(key_b, &mut adapter_b).unwrap();
        assert_eq!(adapter_b.calls, plan().steps()[0..2].to_vec());
        assert_eq!(orchestrator.session(session_a).unwrap().completed_steps, 3);
        assert_eq!(orchestrator.session(session_b).unwrap().completed_steps, 2);

        for _ in 0..3 {
            orchestrator.run_one(key_b, &mut adapter_b).unwrap();
        }
        assert_eq!(adapter_b.calls, plan().steps()[0..5].to_vec());
        assert_eq!(adapter_a.calls.len(), 3);
        assert_eq!(orchestrator.session(session_a).unwrap().completed_steps, 3);
        assert_eq!(orchestrator.session(session_b).unwrap().completed_steps, 5);
    }

    /// P2: a terminal notification for session A must never reach adapter B,
    /// even when it is delivered lazily from `run_one`.
    #[test]
    fn a_terminal_notification_never_resets_a_foreign_adapter() {
        #[derive(Default)]
        struct ForeignProbe {
            ended: Vec<DiffusionSessionId>,
            steps: usize,
        }
        impl DiffusionExecutor for ForeignProbe {
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                self.steps += 1;
                Ok(DiffusionStepOutcome::Executed)
            }
            fn on_session_end(
                &mut self,
                id: DiffusionSessionId,
                _state: DiffusionSessionState,
            ) -> Result<(), String> {
                self.ended.push(id);
                Ok(())
            }
        }

        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key_a = orchestrator.register_executor();
        let key_b = orchestrator.register_executor();
        let session_a = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_a,
            )
            .unwrap();
        let _session_b = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_b,
            )
            .unwrap();

        // Drive B through one step, then cancel A without its adapter present:
        // the notification is deferred.
        let mut adapter_b = ForeignProbe::default();
        orchestrator.run_one(key_b, &mut adapter_b).unwrap();
        orchestrator.cancel(session_a).unwrap();

        for _ in 0..2 {
            orchestrator.run_one(key_b, &mut adapter_b).unwrap();
        }
        // Adapter B is never the owner of A: its reset path is not invoked.
        assert!(
            adapter_b.ended.is_empty(),
            "adapter B received terminal notifications for {session_a:?}: {:?}",
            adapter_b.ended
        );
        assert_eq!(
            adapter_b.steps, 3,
            "adapter B must keep advancing its own session"
        );

        // The notification still reaches A's owner.
        let mut adapter_a = ForeignProbe::default();
        orchestrator.run_one(key_a, &mut adapter_a).unwrap();
        assert_eq!(adapter_a.ended, vec![session_a]);
    }

    /// P5: cancelling before the first step must account for every planned
    /// step that ends cancelled, not just the one task observed running.
    #[test]
    fn pre_step_cancellation_counts_every_cancelled_step() {
        let token = CancellationToken::new();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal, key)
            .unwrap();
        token.cancel();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        assert!(matches!(
            orchestrator.run_one(key, &mut recorder).unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        assert_eq!(orchestrator.metrics().steps_cancelled, 9);
        assert_eq!(orchestrator.metrics().steps_started, 1);
        assert_eq!(orchestrator.metrics().steps_completed, 0);
    }

    /// P5: cancellation observed at the boundary after a completed step must
    /// count all remaining steps once.
    #[test]
    fn boundary_cancellation_after_a_step_counts_all_remaining_steps() {
        let token = CancellationToken::new();
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        orchestrator
            .submit(plan(), token.clone(), TaskPriority::Normal, key)
            .unwrap();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        assert!(matches!(
            orchestrator.run_one(key, &mut recorder).unwrap(),
            DiffusionRunOutcome::StepCompleted { .. }
        ));
        token.cancel();
        assert!(matches!(
            orchestrator.run_one(key, &mut recorder).unwrap(),
            DiffusionRunOutcome::SessionCancelled { .. }
        ));
        assert_eq!(orchestrator.metrics().steps_completed, 1);
        assert_eq!(orchestrator.metrics().steps_cancelled, 8);
    }

    /// P5 guard: a queued session cancelled through `cancel` counts all of its
    /// steps exactly once, and failure-driven dependent cancellation stacks
    /// exactly on top of it.
    #[test]
    fn queued_cancellation_and_failure_dependants_count_exactly_once() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key_a = orchestrator.register_executor();
        let key_b = orchestrator.register_executor();
        let cancelled = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_a,
            )
            .unwrap();
        orchestrator.cancel(cancelled).unwrap();
        assert_eq!(orchestrator.metrics().steps_cancelled, 9);

        struct FailOnce(bool);
        impl DiffusionExecutor for FailOnce {
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                if !self.0 {
                    self.0 = true;
                    Err("synthetic failure".into())
                } else {
                    Ok(DiffusionStepOutcome::Executed)
                }
            }
        }
        let _failed = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_b,
            )
            .unwrap();
        assert!(matches!(
            orchestrator.run_one(key_b, &mut FailOnce(false)).unwrap(),
            DiffusionRunOutcome::StepFailed { .. }
        ));
        assert_eq!(orchestrator.metrics().steps_cancelled, 17);
        assert_eq!(orchestrator.metrics().steps_failed, 1);
    }

    /// P6: every pending terminal notification for the driven executor is
    /// flushed by one `run_one`, in deterministic session order.
    #[test]
    fn one_run_one_flushes_all_pending_terminal_notifications_in_order() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        // One key serves one open session, so three sessions need three keys;
        // determinism is asserted on session ids, not keys.
        let key_a = orchestrator.register_executor();
        let key_b = orchestrator.register_executor();
        let key_c = orchestrator.register_executor();
        let first = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_a,
            )
            .unwrap();
        let second = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_b,
            )
            .unwrap();
        let third = orchestrator
            .submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                key_c,
            )
            .unwrap();
        orchestrator.cancel(third).unwrap();
        orchestrator.cancel(second).unwrap();
        orchestrator.cancel(first).unwrap();

        let mut recorder_a = CleanupRecorder::default();
        let mut recorder_b = CleanupRecorder::default();
        let mut recorder_c = CleanupRecorder::default();
        assert!(matches!(
            orchestrator.run_one(key_a, &mut recorder_a).unwrap(),
            DiffusionRunOutcome::Idle
        ));
        assert!(matches!(
            orchestrator.run_one(key_b, &mut recorder_b).unwrap(),
            DiffusionRunOutcome::Idle
        ));
        assert!(matches!(
            orchestrator.run_one(key_c, &mut recorder_c).unwrap(),
            DiffusionRunOutcome::Idle
        ));
        assert_eq!(
            recorder_a.ended,
            vec![(first, DiffusionSessionState::Cancelled)]
        );
        assert_eq!(
            recorder_b.ended,
            vec![(second, DiffusionSessionState::Cancelled)]
        );
        assert_eq!(
            recorder_c.ended,
            vec![(third, DiffusionSessionState::Cancelled)]
        );
    }

    /// P10: a skipped step (adapter already stopped) must not count as a
    /// completed step, a committed block, or effective work.
    #[test]
    fn skipped_steps_never_count_as_completed_work_or_blocks() {
        struct SelfStopping {
            blocks: usize,
            stopped: bool,
        }
        impl DiffusionExecutor for SelfStopping {
            fn execute_step(
                &mut self,
                step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                if self.stopped {
                    return Ok(DiffusionStepOutcome::Skipped);
                }
                if let DiffusionStep::CommitBlock { block } = step {
                    self.blocks += 1;
                    if block.0 == 0 {
                        self.stopped = true; // model reached EOS mid-plan
                    }
                }
                Ok(DiffusionStepOutcome::Executed)
            }
        }

        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let session = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        let mut adapter = SelfStopping {
            blocks: 0,
            stopped: false,
        };
        loop {
            match orchestrator.run_one(key, &mut adapter).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        let metrics = orchestrator.metrics();
        assert_eq!(adapter.blocks, 1);
        assert_eq!(
            metrics.blocks_completed, 1,
            "the phantom second block must not be counted"
        );
        assert_eq!(metrics.steps_completed, 5);
        assert_eq!(metrics.steps_skipped, 4);
        assert_eq!(metrics.blocks_pending, 0);
        assert_eq!(orchestrator.session(session).unwrap().completed_steps, 9);
    }

    /// P4: an adapter cleanup error is surfaced on the session snapshot and
    /// counted, never swallowed.
    #[test]
    fn a_failed_session_cleanup_is_reported_on_the_session() {
        struct DirtyCleanup {
            failures: usize,
        }
        impl DiffusionExecutor for DirtyCleanup {
            fn execute_step(
                &mut self,
                _step: DiffusionStep,
            ) -> Result<DiffusionStepOutcome, String> {
                Ok(DiffusionStepOutcome::Executed)
            }
            fn on_session_end(
                &mut self,
                _id: DiffusionSessionId,
                _state: DiffusionSessionState,
            ) -> Result<(), String> {
                self.failures += 1;
                Err("expert release failed".to_owned())
            }
        }

        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let session = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        let mut adapter = DirtyCleanup { failures: 0 };
        loop {
            match orchestrator.run_one(key, &mut adapter).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        assert_eq!(adapter.failures, 1);
        let snapshot = orchestrator.session(session).unwrap();
        let error = snapshot.error.expect("cleanup failure must be surfaced");
        assert!(
            error.contains("expert release failed"),
            "unhelpful error: {error}"
        );
        assert_eq!(orchestrator.metrics().cleanups_failed, 1);
    }

    /// P8: orchestration state for closed sessions is bounded by an explicit
    /// reclaim that removes sessions, plans, and task indexes atomically — but
    /// never before the terminal notification has been delivered.
    #[test]
    fn closed_sessions_are_reclaimed_only_after_notification_delivery() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let session = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        let mut recorder = CleanupRecorder::default();
        loop {
            match orchestrator.run_one(key, &mut recorder).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        // The finished session keeps its snapshot until it is reclaimed.
        assert_eq!(
            orchestrator.session(session).unwrap().state,
            DiffusionSessionState::Completed
        );
        let cancelled_again = orchestrator.cancel(session);
        assert_eq!(
            cancelled_again,
            Err(DiffusionError::SessionTerminal(session))
        );
        let report = orchestrator.reclaim_closed_sessions();
        assert_eq!(report.reclaimed, vec![session]);
        assert!(report.deferred_terminal_notification.is_empty());
        assert!(matches!(
            orchestrator.session(session),
            Err(DiffusionError::UnknownSession(_))
        ));
        // Task index and plan maps must be gone with it (bounded retention).
        assert_eq!(orchestrator.retained_sessions_for_test(), 0);

        // A queued-cancelled session defers reclaim until its owner flushes.
        let late = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        orchestrator.cancel(late).unwrap();
        let deferred = orchestrator.reclaim_closed_sessions();
        assert!(deferred.reclaimed.is_empty());
        assert_eq!(deferred.deferred_terminal_notification, vec![late]);
        orchestrator.run_one(key, &mut recorder).unwrap();
        let after = orchestrator.reclaim_closed_sessions();
        assert_eq!(after.reclaimed, vec![late]);
        assert_eq!(orchestrator.retained_sessions_for_test(), 0);
    }

    #[test]
    fn a_step_never_routes_to_an_unregistered_executor_key() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let foreign = DiffusionOrchestrator::new(SchedulerConfig::default())
            .unwrap()
            .register_executor();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        assert!(matches!(
            orchestrator.run_one(foreign, &mut recorder),
            Err(DiffusionError::UnknownExecutor(_))
        ));
        assert!(matches!(
            orchestrator.submit(
                plan(),
                CancellationToken::new(),
                TaskPriority::Normal,
                foreign
            ),
            Err(DiffusionError::UnknownExecutor(_))
        ));
    }

    /// P1 guard: one adapter instance (one key) serves at most one open
    /// session; the key frees only when the closed session is reclaimed.
    #[test]
    fn a_key_may_drive_at_most_one_open_session_at_a_time() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        let first = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        assert!(matches!(
            orchestrator.submit(plan(), CancellationToken::new(), TaskPriority::Normal, key),
            Err(DiffusionError::ExecutorBusy { .. })
        ));
        // Once the first session is closed AND reclaimed, the key is free.
        let mut recorder = CleanupRecorder::default();
        loop {
            match orchestrator.run_one(key, &mut recorder).unwrap() {
                DiffusionRunOutcome::SessionCompleted { .. } => break,
                DiffusionRunOutcome::StepCompleted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        assert!(matches!(
            orchestrator.submit(plan(), CancellationToken::new(), TaskPriority::Normal, key),
            Err(DiffusionError::ExecutorBusy { .. })
        ));
        orchestrator.reclaim_closed_sessions();
        let second = orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        assert_ne!(first, second);
    }

    /// P7: the orchestrator's scheduler task identities are typed
    /// `DiffusionTaskId`; nothing in the public API yields a raw scheduler
    /// `TaskId` that could be routed into another scheduler.
    #[test]
    fn diffusion_task_ids_are_typed_and_never_leak_raw_scheduler_ids() {
        let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
        let key = orchestrator.register_executor();
        orchestrator
            .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
            .unwrap();
        let mut recorder = Recorder(RefCell::new(Vec::new()));
        let outcome = orchestrator.run_one(key, &mut recorder).unwrap();
        let task_id = match outcome {
            DiffusionRunOutcome::StepCompleted { task_id, .. } => task_id,
            other => panic!("unexpected outcome: {other:?}"),
        };
        // `task_id` is `DiffusionTaskId`: a distinct type with no `From`/`get`
        // bridge to `claymore_scheduler::TaskId`. The compile-time separation
        // is the guarantee; this assertion pins the constructor identity so a
        // future regression that returns raw ids fails.
        let again = match orchestrator.run_one(key, &mut recorder).unwrap() {
            DiffusionRunOutcome::StepCompleted { task_id, .. } => task_id,
            other => panic!("unexpected outcome: {other:?}"),
        };
        assert_ne!(task_id, again);
    }
}
