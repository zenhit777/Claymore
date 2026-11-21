//! Deterministic, thread-affine task scheduling primitives for Claymore.
//!
//! The scheduler owns task state and dependency ordering only. It does not
//! spawn workers, call CUDA, allocate memory, or execute task closures. The
//! thread that constructs it is the owner thread; the `Rc` marker deliberately
//! keeps the scheduler `!Send + !Sync`. A future CUDA executor must drain the
//! transfer queue on the CUDA context's owner thread and report completion only
//! after the backend confirms it.
//!
//! Compute, transfer, and prefetch tasks share one dependency graph but have
//! separate ready queues (selected through `start_next`). Ordering within a
//! queue is priority-descending and FIFO for equal priorities.

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::rc::Rc;
use thiserror::Error;

/// Opaque monotonically allocated identifier for one scheduled task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(u64);

impl TaskId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Queue to which a task is dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Compute,
    Transfer,
    Prefetch,
}

/// Higher-priority ready tasks are selected first; equal priorities are FIFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
}

/// Scheduler-owned lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Blocked,
    Ready,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TaskState {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Description submitted to the scheduler. Dependencies must already exist,
/// which makes the graph acyclic by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    pub kind: TaskKind,
    pub priority: TaskPriority,
    pub dependencies: Vec<TaskId>,
}

impl TaskSpec {
    pub fn new(kind: TaskKind) -> Self {
        Self {
            kind,
            priority: TaskPriority::Normal,
            dependencies: Vec::new(),
        }
    }

    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn depends_on(mut self, dependency: TaskId) -> Self {
        self.dependencies.push(dependency);
        self
    }
}

/// Read-only task state returned by inspection methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSnapshot {
    pub id: TaskId,
    pub kind: TaskKind,
    pub priority: TaskPriority,
    pub state: TaskState,
    pub dependencies: Vec<TaskId>,
}

/// Scheduler configuration. `max_prefetches` bounds concurrently running
/// prefetches; `lookahead` bounds additional outstanding prefetch work waiting
/// in the graph. Thus outstanding prefetches cannot exceed their sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub cpu_workers: usize,
    pub max_prefetches: usize,
    pub lookahead: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            cpu_workers: 10,
            max_prefetches: 2,
            lookahead: 1,
        }
    }
}

impl SchedulerConfig {
    pub fn validate(&self) -> bool {
        (1..=12).contains(&self.cpu_workers) && self.max_prefetches <= 2
    }

    const fn prefetch_window(self) -> usize {
        if self.max_prefetches == 0 {
            0
        } else {
            self.max_prefetches.saturating_add(self.lookahead)
        }
    }
}

/// Errors returned when a task cannot be admitted or transitioned.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("invalid scheduler configuration")]
    InvalidConfig,
    #[error("dependency {0:?} does not exist")]
    UnknownDependency(TaskId),
    #[error("dependency {0:?} is listed more than once")]
    DuplicateDependency(TaskId),
    #[error("prefetch window is full (limit {limit})")]
    PrefetchWindowFull { limit: usize },
    #[error("task {0:?} does not exist")]
    UnknownTask(TaskId),
    #[error("cannot {operation} task {task_id:?} while it is {state:?}")]
    InvalidTransition {
        task_id: TaskId,
        state: TaskState,
        operation: &'static str,
    },
    #[error("task identifier space exhausted")]
    TaskIdExhausted,
}

#[derive(Debug)]
struct TaskRecord {
    id: TaskId,
    sequence: u64,
    kind: TaskKind,
    priority: TaskPriority,
    state: TaskState,
    dependencies: Vec<TaskId>,
}

/// Single-owner scheduler. It is intentionally `!Send` and `!Sync`: a
/// transfer executor must remain on the thread that owns its CUDA context.
pub struct Scheduler {
    config: SchedulerConfig,
    tasks: BTreeMap<TaskId, TaskRecord>,
    next_task_id: u64,
    next_sequence: u64,
    running_prefetches: usize,
    _owner_thread: PhantomData<Rc<()>>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        if !config.validate() {
            return Err(SchedulerError::InvalidConfig);
        }
        Ok(Self {
            config,
            tasks: BTreeMap::new(),
            next_task_id: 1,
            next_sequence: 0,
            running_prefetches: 0,
            _owner_thread: PhantomData,
        })
    }

    pub fn config(&self) -> SchedulerConfig {
        self.config
    }

    /// Submit a task. Dependencies must refer to tasks already submitted.
    /// Prefetch admission is bounded by max_prefetches + lookahead, including
    /// blocked and ready prefetch tasks, so producers cannot grow it unbounded.
    pub fn submit(&mut self, spec: TaskSpec) -> Result<TaskId, SchedulerError> {
        for (index, dependency) in spec.dependencies.iter().enumerate() {
            if spec.dependencies[..index].contains(dependency) {
                return Err(SchedulerError::DuplicateDependency(*dependency));
            }
            if !self.tasks.contains_key(dependency) {
                return Err(SchedulerError::UnknownDependency(*dependency));
            }
        }

        if spec.kind == TaskKind::Prefetch {
            let outstanding = self
                .tasks
                .values()
                .filter(|task| task.kind == TaskKind::Prefetch && !task.state.is_terminal())
                .count();
            let limit = self.config.prefetch_window();
            if outstanding >= limit {
                return Err(SchedulerError::PrefetchWindowFull { limit });
            }
        }

        let id = TaskId(self.next_task_id);
        let next_task_id = self
            .next_task_id
            .checked_add(1)
            .ok_or(SchedulerError::TaskIdExhausted)?;
        let sequence = self.next_sequence;
        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SchedulerError::TaskIdExhausted)?;

        let dependency_states: Vec<_> = spec.dependencies.iter().filter_map(|dependency| self.tasks.get(dependency).map(|task| task.state)).collect();
        let state = if dependency_states.iter().any(|state| matches!(state, TaskState::Failed | TaskState::Cancelled)) {
            TaskState::Cancelled
        } else if dependency_states.iter().all(|state| *state == TaskState::Completed) {
            TaskState::Ready
        } else {
            TaskState::Blocked
        };

        self.tasks.insert(
            id,
            TaskRecord {
                id,
                sequence,
                kind: spec.kind,
                priority: spec.priority,
                state,
                dependencies: spec.dependencies,
            },
        );
        self.next_task_id = next_task_id;
        self.next_sequence = next_sequence;
        Ok(id)
    }

    /// Start the highest-priority ready task from one queue. Equal-priority
    /// tasks retain submission order. Returns None when no task is ready or
    /// when the concurrent prefetch limit has been reached.
    pub fn start_next(&mut self, kind: TaskKind) -> Option<TaskId> {
        if kind == TaskKind::Prefetch && self.running_prefetches >= self.config.max_prefetches {
            return None;
        }

        let selected = self
            .tasks
            .values()
            .filter(|task| task.kind == kind && task.state == TaskState::Ready)
            .min_by_key(|task| (Reverse(task.priority), task.sequence))
            .map(|task| task.id)?;

        let task = self.tasks.get_mut(&selected)?;
        task.state = TaskState::Running;
        if kind == TaskKind::Prefetch {
            self.running_prefetches += 1;
        }
        Some(selected)
    }

    /// Confirm completion after the executor/backend has actually finished.
    /// Dependents become ready only after every dependency is completed.
    pub fn complete(&mut self, id: TaskId) -> Result<(), SchedulerError> {
        let task = self.task_mut(id, "complete")?;
        if task.state != TaskState::Running {
            return Err(SchedulerError::InvalidTransition {
                task_id: id,
                state: task.state,
                operation: "complete",
            });
        }
        let was_prefetch = task.kind == TaskKind::Prefetch;
        task.state = TaskState::Completed;
        if was_prefetch {
            self.running_prefetches -= 1;
        }
        self.refresh_blocked_tasks();
        Ok(())
    }

    /// Mark a running task failed. Dependents that can no longer run are
    /// cancelled recursively; unrelated tasks remain schedulable.
    pub fn fail(&mut self, id: TaskId) -> Result<(), SchedulerError> {
        let task = self.task_mut(id, "fail")?;
        if task.state != TaskState::Running {
            return Err(SchedulerError::InvalidTransition {
                task_id: id,
                state: task.state,
                operation: "fail",
            });
        }
        let was_prefetch = task.kind == TaskKind::Prefetch;
        task.state = TaskState::Failed;
        if was_prefetch {
            self.running_prefetches -= 1;
        }
        self.refresh_blocked_tasks();
        Ok(())
    }

    /// Cancel a task that has not started. Running work must be stopped by its
    /// executor and then reported as failed/cancelled by a future executor API.
    pub fn cancel(&mut self, id: TaskId) -> Result<(), SchedulerError> {
        let task = self.task_mut(id, "cancel")?;
        if !matches!(task.state, TaskState::Blocked | TaskState::Ready) {
            return Err(SchedulerError::InvalidTransition {
                task_id: id,
                state: task.state,
                operation: "cancel",
            });
        }
        task.state = TaskState::Cancelled;
        self.refresh_blocked_tasks();
        Ok(())
    }

    /// Mark a running task cancelled after its executor has cooperatively
    /// stopped the underlying operation. This is distinct from cancelling
    /// queued work: callers must not use it while backend work is still active.
    pub fn cancel_running(&mut self, id: TaskId) -> Result<(), SchedulerError> {
        let task = self.task_mut(id, "cancel_running")?;
        if task.state != TaskState::Running {
            return Err(SchedulerError::InvalidTransition {
                task_id: id,
                state: task.state,
                operation: "cancel_running",
            });
        }
        let was_prefetch = task.kind == TaskKind::Prefetch;
        task.state = TaskState::Cancelled;
        if was_prefetch {
            self.running_prefetches -= 1;
        }
        self.refresh_blocked_tasks();
        Ok(())
    }

    /// Return cancelled task IDs in deterministic submission order.
    pub fn cancelled_task_ids(&self) -> Vec<TaskId> {
        self.tasks
            .values()
            .filter(|task| task.state == TaskState::Cancelled)
            .map(|task| task.id)
            .collect()
    }

    pub fn task(&self, id: TaskId) -> Result<TaskSnapshot, SchedulerError> {
        let task = self.tasks.get(&id).ok_or(SchedulerError::UnknownTask(id))?;
        Ok(TaskSnapshot {
            id: task.id,
            kind: task.kind,
            priority: task.priority,
            state: task.state,
            dependencies: task.dependencies.clone(),
        })
    }

    pub fn ready_count(&self, kind: TaskKind) -> usize {
        self.tasks
            .values()
            .filter(|task| task.kind == kind && task.state == TaskState::Ready)
            .count()
    }

    pub fn running_prefetches(&self) -> usize {
        self.running_prefetches
    }

    fn task_mut(
        &mut self,
        id: TaskId,
        operation: &'static str,
    ) -> Result<&mut TaskRecord, SchedulerError> {
        self.tasks
            .get_mut(&id)
            .ok_or(SchedulerError::UnknownTask(id))
            .and_then(|task| {
                if task.state.is_terminal() {
                    Err(SchedulerError::InvalidTransition {
                        task_id: id,
                        state: task.state,
                        operation,
                    })
                } else {
                    Ok(task)
                }
            })
    }

    fn refresh_blocked_tasks(&mut self) {
        loop {
            let states: BTreeMap<_, _> = self
                .tasks
                .iter()
                .map(|(id, task)| (*id, task.state))
                .collect();
            let mut changed = false;
            for task in self.tasks.values_mut() {
                if task.state != TaskState::Blocked {
                    continue;
                }
                let dependency_states: Vec<_> = task
                    .dependencies
                    .iter()
                    .filter_map(|id| states.get(id).copied())
                    .collect();
                if dependency_states
                    .iter()
                    .any(|state| matches!(state, TaskState::Failed | TaskState::Cancelled))
                {
                    task.state = TaskState::Cancelled;
                    changed = true;
                } else if dependency_states.len() == task.dependencies.len()
                    && dependency_states
                        .iter()
                        .all(|state| *state == TaskState::Completed)
                {
                    task.state = TaskState::Ready;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler() -> Scheduler {
        Scheduler::new(SchedulerConfig::default()).unwrap()
    }

    #[test]
    fn submission_cancels_tasks_with_already_failed_or_cancelled_dependencies() {
        let mut scheduler = scheduler();
        let failed = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        scheduler.start_next(TaskKind::Transfer).unwrap();
        scheduler.fail(failed).unwrap();
        let after_failure = scheduler.submit(TaskSpec::new(TaskKind::Compute).depends_on(failed)).unwrap();
        assert_eq!(scheduler.task(after_failure).unwrap().state, TaskState::Cancelled);
        let cancelled = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        scheduler.cancel(cancelled).unwrap();
        let after_cancellation = scheduler.submit(TaskSpec::new(TaskKind::Compute).depends_on(cancelled)).unwrap();
        assert_eq!(scheduler.task(after_cancellation).unwrap().state, TaskState::Cancelled);
        let descendant = scheduler.submit(TaskSpec::new(TaskKind::Compute).depends_on(after_failure)).unwrap();
        assert_eq!(scheduler.task(descendant).unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn dispatches_by_priority_then_fifo_within_each_queue() {
        let mut scheduler = scheduler();
        let normal = scheduler.submit(TaskSpec::new(TaskKind::Compute)).unwrap();
        let high_first = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).with_priority(TaskPriority::High))
            .unwrap();
        let high_second = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).with_priority(TaskPriority::High))
            .unwrap();

        assert_eq!(scheduler.start_next(TaskKind::Compute), Some(high_first));
        scheduler.complete(high_first).unwrap();
        assert_eq!(scheduler.start_next(TaskKind::Compute), Some(high_second));
        scheduler.complete(high_second).unwrap();
        assert_eq!(scheduler.start_next(TaskKind::Compute), Some(normal));
    }

    #[test]
    fn queues_are_independent() {
        let mut scheduler = scheduler();
        let compute = scheduler.submit(TaskSpec::new(TaskKind::Compute)).unwrap();
        let transfer = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();

        assert_eq!(scheduler.start_next(TaskKind::Transfer), Some(transfer));
        assert_eq!(scheduler.start_next(TaskKind::Compute), Some(compute));
    }

    #[test]
    fn dependent_task_becomes_ready_only_after_completion() {
        let mut scheduler = scheduler();
        let transfer = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let compute = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).depends_on(transfer))
            .unwrap();

        assert_eq!(scheduler.task(compute).unwrap().state, TaskState::Blocked);
        assert_eq!(scheduler.start_next(TaskKind::Compute), None);
        scheduler.start_next(TaskKind::Transfer).unwrap();
        scheduler.complete(transfer).unwrap();
        assert_eq!(scheduler.task(compute).unwrap().state, TaskState::Ready);
        assert_eq!(scheduler.start_next(TaskKind::Compute), Some(compute));
    }

    #[test]
    fn failed_dependency_cancels_descendants_but_not_unrelated_work() {
        let mut scheduler = scheduler();
        let transfer = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        let compute = scheduler
            .submit(TaskSpec::new(TaskKind::Compute).depends_on(transfer))
            .unwrap();
        let unrelated = scheduler.submit(TaskSpec::new(TaskKind::Compute)).unwrap();

        scheduler.start_next(TaskKind::Transfer).unwrap();
        scheduler.fail(transfer).unwrap();

        assert_eq!(scheduler.task(compute).unwrap().state, TaskState::Cancelled);
        assert_eq!(scheduler.task(unrelated).unwrap().state, TaskState::Ready);
    }

    #[test]
    fn prefetch_concurrency_and_lookahead_are_bounded() {
        let mut scheduler = scheduler();
        let first = scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).unwrap();
        let second = scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).unwrap();
        let third = scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).unwrap();
        assert_eq!(
            scheduler.submit(TaskSpec::new(TaskKind::Prefetch)),
            Err(SchedulerError::PrefetchWindowFull { limit: 3 })
        );

        assert_eq!(scheduler.start_next(TaskKind::Prefetch), Some(first));
        assert_eq!(scheduler.start_next(TaskKind::Prefetch), Some(second));
        assert_eq!(scheduler.start_next(TaskKind::Prefetch), None);
        assert_eq!(scheduler.running_prefetches(), 2);
        scheduler.complete(first).unwrap();
        assert_eq!(scheduler.start_next(TaskKind::Prefetch), Some(third));
    }

    #[test]
    fn cancelling_prefetch_releases_lookahead_capacity() {
        let mut scheduler = scheduler();
        let first = scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).unwrap();
        scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).unwrap();
        scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).unwrap();
        scheduler.cancel(first).unwrap();
        assert!(scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).is_ok());
    }

    #[test]
    fn duplicate_and_unknown_dependencies_are_rejected() {
        let mut scheduler = scheduler();
        let dependency = scheduler.submit(TaskSpec::new(TaskKind::Transfer)).unwrap();
        assert_eq!(
            scheduler.submit(
                TaskSpec::new(TaskKind::Compute)
                    .depends_on(dependency)
                    .depends_on(dependency)
            ),
            Err(SchedulerError::DuplicateDependency(dependency))
        );
        assert_eq!(
            scheduler.submit(TaskSpec::new(TaskKind::Compute).depends_on(TaskId(999))),
            Err(SchedulerError::UnknownDependency(TaskId(999)))
        );
    }

    #[test]
    fn running_tasks_cannot_be_cancelled_or_completed_twice() {
        let mut scheduler = scheduler();
        let task = scheduler.submit(TaskSpec::new(TaskKind::Compute)).unwrap();
        scheduler.start_next(TaskKind::Compute).unwrap();
        assert!(matches!(
            scheduler.cancel(task),
            Err(SchedulerError::InvalidTransition { .. })
        ));
        scheduler.complete(task).unwrap();
        assert!(matches!(
            scheduler.complete(task),
            Err(SchedulerError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn zero_prefetch_concurrency_disables_prefetch_admission() {
        let config = SchedulerConfig {
            max_prefetches: 0,
            lookahead: 3,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config).unwrap();
        assert_eq!(
            scheduler.submit(TaskSpec::new(TaskKind::Prefetch)),
            Err(SchedulerError::PrefetchWindowFull { limit: 0 })
        );
    }

    #[test]
    fn cancelling_running_prefetch_releases_concurrency_slot() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let id = scheduler.submit(TaskSpec::new(TaskKind::Prefetch)).unwrap();
        assert_eq!(scheduler.start_next(TaskKind::Prefetch), Some(id));
        assert_eq!(scheduler.running_prefetches(), 1);
        scheduler.cancel_running(id).unwrap();
        assert_eq!(scheduler.running_prefetches(), 0);
        assert_eq!(scheduler.task(id).unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let config = SchedulerConfig {
            cpu_workers: 0,
            ..SchedulerConfig::default()
        };
        assert_eq!(
            Scheduler::new(config).err(),
            Some(SchedulerError::InvalidConfig)
        );
    }
}

/// Synchronous, owner-thread task execution adapters.
pub mod executor;
pub use executor::{ExecutionOutcome, QueueRunner, QueueRunnerMetrics, TaskExecutor};
