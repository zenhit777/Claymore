//! Review-regression proofs for the M11.4 Tech-Lead review (2026-11-28).
//!
//! RED phase: these tests were first executed against the UNCORRECTED
//! orchestrator (pre-fix API) and each failed for its defect's behavioural
//! reason — adapter B executing session A's step, A's terminal notification
//! reaching adapter B, `steps_cancelled` counting 1 instead of 8/7, only one
//! deferred notification flushed, and a diffusion task id silently cancelling
//! an unrelated queued request. The captured failure log is
//! `docs/audits/logs/m11.4-p1p2p5p6p7-red-2026-11-28.txt`.
//!
//! GREEN phase (here): the same guarantees are asserted through the corrected
//! session↔executor-bound API. P10's red proof is
//! `p10_blocks_completed_counts_only_effective_commits` in
//! `claymore-model-clover/tests/m11_4_low_confidence.rs`; its green form is
//! `p10_blocks_completed_counts_only_effective_commits` in the same file
//! together with `skipped_steps_never_count_as_completed_work_or_blocks` in
//! `claymore-runtime/src/diffusion.rs`.

use std::cell::Cell;

use claymore_runtime::{
    BlockId, CancellationToken, DiffusionExecutor, DiffusionOrchestrator, DiffusionPlan,
    DiffusionRunOutcome, DiffusionSessionState, DiffusionStep, DiffusionStepOutcome,
    GenerationStrategy, RequestInput, RuntimeConfig, ScheduledRuntime,
};
use claymore_runtime::{ExecutionContext, RequestOutput};
use claymore_scheduler::{SchedulerConfig, TaskPriority, TaskState};

fn plan() -> DiffusionPlan {
    DiffusionPlan::new(vec![
        DiffusionStep::PrepareBlock { block: BlockId(0) },
        DiffusionStep::Denoise {
            block: BlockId(0),
            iteration: 0,
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

#[derive(Default)]
struct TaggedRecorder {
    calls: Vec<DiffusionStep>,
    ended: Vec<claymore_runtime::DiffusionSessionId>,
}

impl DiffusionExecutor for TaggedRecorder {
    fn execute_step(&mut self, step: DiffusionStep) -> Result<DiffusionStepOutcome, String> {
        self.calls.push(step);
        Ok(DiffusionStepOutcome::Executed)
    }
    fn on_session_end(
        &mut self,
        id: claymore_runtime::DiffusionSessionId,
        _state: DiffusionSessionState,
    ) -> Result<(), String> {
        self.ended.push(id);
        Ok(())
    }
}

/// P1 — two sessions and two adapters: driving only adapter B must never
/// advance (or even touch) session A.
#[test]
fn p1_a_step_of_session_a_must_not_run_on_adapter_b() {
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
    let mut adapter_b = TaggedRecorder::default();

    // Driving B may only run B's work: it must not consume A's ready step.
    orchestrator.run_one(key_b, &mut adapter_b).unwrap();
    orchestrator.run_one(key_b, &mut adapter_b).unwrap();

    assert_eq!(
        orchestrator.session(session_a).unwrap().completed_steps,
        0,
        "adapter B advanced session A: {:?}",
        adapter_b.calls
    );
    assert_eq!(
        adapter_b.calls,
        plan().steps()[0..2].to_vec(),
        "adapter B must have run its own session's first two steps"
    );
}

/// P2 — a deferred terminal notification for session A must not reach the
/// adapter that owns session B (which would reset B's caches/residency).
#[test]
fn p2_a_deferred_notification_must_not_reach_a_foreign_adapter() {
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
    orchestrator.cancel(session_a).unwrap();

    let mut adapter_b = TaggedRecorder::default();
    orchestrator.run_one(key_b, &mut adapter_b).unwrap();
    orchestrator.run_one(key_b, &mut adapter_b).unwrap();

    assert!(
        adapter_b.ended.is_empty(),
        "adapter B received terminal notification for session {session_a:?}: {:?}",
        adapter_b.ended
    );

    // Delivery still happens for the owner.
    let mut adapter_a = TaggedRecorder::default();
    orchestrator.run_one(key_a, &mut adapter_a).unwrap();
    assert_eq!(adapter_a.ended, vec![session_a]);
}

/// P5 — cancel observed before the first step: every planned step that ends
/// cancelled must be counted, not only the one task already running.
#[test]
fn p5_pre_step_cancellation_counts_every_cancelled_step() {
    let token = CancellationToken::new();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let key = orchestrator.register_executor();
    orchestrator
        .submit(plan(), token.clone(), TaskPriority::Normal, key)
        .unwrap();
    token.cancel();
    let mut recorder = TaggedRecorder::default();
    assert!(matches!(
        orchestrator.run_one(key, &mut recorder).unwrap(),
        DiffusionRunOutcome::SessionCancelled { .. }
    ));
    assert_eq!(orchestrator.metrics().steps_cancelled, 8);
    assert_eq!(orchestrator.metrics().steps_completed, 0);
}

/// P5 — cancellation at the boundary after a completed step: the remaining
/// steps are cancelled and counted exactly once.
#[test]
fn p5_boundary_cancellation_after_a_step_counts_all_remaining_steps() {
    let token = CancellationToken::new();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let key = orchestrator.register_executor();
    orchestrator
        .submit(plan(), token.clone(), TaskPriority::Normal, key)
        .unwrap();
    let mut recorder = TaggedRecorder::default();
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
    assert_eq!(orchestrator.metrics().steps_cancelled, 7);
}

/// P6 — one `run_one` flushes every pending terminal notification of the
/// driven key, in deterministic session-id order, and never duplicates.
#[test]
fn p6_one_run_flushes_all_pending_notifications_in_session_order() {
    // The one-session-per-key contract means several closed sessions for the
    // same *host* span several keys; flush routing is asserted per key and
    // exactly-once globally.
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let key_first = orchestrator.register_executor();
    let key_second = orchestrator.register_executor();
    let first = orchestrator
        .submit(
            plan(),
            CancellationToken::new(),
            TaskPriority::Normal,
            key_first,
        )
        .unwrap();
    let second = orchestrator
        .submit(
            plan(),
            CancellationToken::new(),
            TaskPriority::Normal,
            key_second,
        )
        .unwrap();
    orchestrator.cancel(second).unwrap();
    orchestrator.cancel(first).unwrap();

    let mut recorder_first = TaggedRecorder::default();
    let mut recorder_second = TaggedRecorder::default();
    assert!(matches!(
        orchestrator
            .run_one(key_first, &mut recorder_first)
            .unwrap(),
        DiffusionRunOutcome::Idle
    ));
    assert!(matches!(
        orchestrator
            .run_one(key_second, &mut recorder_second)
            .unwrap(),
        DiffusionRunOutcome::Idle
    ));
    assert_eq!(recorder_first.ended, vec![first]);
    assert_eq!(recorder_second.ended, vec![second]);

    // Driven again: no duplicate delivery.
    orchestrator
        .run_one(key_first, &mut recorder_first)
        .unwrap();
    orchestrator
        .run_one(key_second, &mut recorder_second)
        .unwrap();
    assert_eq!(recorder_first.ended, vec![first]);
    assert_eq!(recorder_second.ended, vec![second]);
}

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

/// P7 — `ScheduledRuntime` owns two schedulers whose opaque ids collide by
/// construction. The diffusion side reports `DiffusionTaskId`, which the
/// request-side API does not accept: red-phase code that extracted the raw
/// scheduler id from the outcome and cancelled through `cancel_queued` now
/// cannot even compile (see the red log). The runtime assertions below pin
/// that request-side operations cannot disturb diffusion sessions and the
/// diffusion step identity stays a `DiffusionTaskId`.
#[test]
fn p7_diffusion_task_ids_must_not_reach_the_request_scheduler() {
    let mut runtime =
        ScheduledRuntime::new(RuntimeConfig::default(), SchedulerConfig::default(), Echo).unwrap();

    let request = runtime
        .submit_request(
            RequestInput::new([7_u8, 7]),
            CancellationToken::new(),
            TaskPriority::Normal,
            [],
        )
        .unwrap();
    #[derive(Default)]
    struct Runner(Cell<usize>);
    impl DiffusionExecutor for Runner {
        fn execute_step(&mut self, _step: DiffusionStep) -> Result<DiffusionStepOutcome, String> {
            self.0.set(self.0.get() + 1);
            Ok(DiffusionStepOutcome::Executed)
        }
    }
    let key = runtime.register_diffusion_executor();
    runtime
        .submit_diffusion(plan(), CancellationToken::new(), TaskPriority::Normal, key)
        .unwrap();

    let mut runner = Runner::default();
    let outcome = runtime.run_diffusion_one(key, &mut runner).unwrap();
    let task_id = match outcome {
        DiffusionRunOutcome::StepCompleted { task_id, .. } => task_id,
        other => panic!("unexpected outcome: {other:?}"),
    };
    // Type-level isolation: `runtime.cancel_queued(task_id)` is a compile
    // error. This line pins the required identity:
    let _: claymore_runtime::DiffusionTaskId = task_id;

    // The queued request keeps its own state: nothing about driving diffusion
    // touched the request scheduler.
    assert_eq!(runtime.task(request).unwrap().state, TaskState::Ready);
    runtime.cancel_queued(request).unwrap();
    assert_eq!(runtime.task(request).unwrap().state, TaskState::Cancelled);
    // And diffusion keeps running untouched by request-side cancellation.
    assert!(matches!(
        runtime.run_diffusion_one(key, &mut runner).unwrap(),
        DiffusionRunOutcome::StepCompleted { .. }
    ));
}

/// P4 — a failed terminal cleanup remains pending, blocks reclamation, and is
/// retried on the next owner-driven run. The adapter callback must be
/// idempotent because a partial release can fail after releasing some entries.
#[derive(Default)]
struct RetryCleanupRecorder {
    cleanup_calls: usize,
}

impl DiffusionExecutor for RetryCleanupRecorder {
    fn execute_step(&mut self, _step: DiffusionStep) -> Result<DiffusionStepOutcome, String> {
        Ok(DiffusionStepOutcome::Executed)
    }

    fn on_session_end(
        &mut self,
        _id: claymore_runtime::DiffusionSessionId,
        _state: DiffusionSessionState,
    ) -> Result<(), String> {
        self.cleanup_calls += 1;
        if self.cleanup_calls == 1 {
            Err("transient release failure".to_owned())
        } else {
            Ok(())
        }
    }
}

#[test]
fn p4_failed_cleanup_stays_pending_until_owner_retry_succeeds() {
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let key = orchestrator.register_executor();
    let session = orchestrator
        .submit(plan(), CancellationToken::new(), TaskPriority::Normal, key)
        .unwrap();
    let mut adapter = RetryCleanupRecorder::default();

    loop {
        match orchestrator.run_one(key, &mut adapter).unwrap() {
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            DiffusionRunOutcome::StepCompleted { .. } => {}
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    assert_eq!(adapter.cleanup_calls, 1);
    assert_eq!(orchestrator.metrics().cleanups_failed, 1);
    let first_reclaim = orchestrator.reclaim_closed_sessions();
    assert!(
        first_reclaim.reclaimed.is_empty(),
        "a session with failed cleanup must not be reclaimed"
    );
    assert_eq!(first_reclaim.deferred_terminal_notification, vec![session]);

    assert!(matches!(
        orchestrator.run_one(key, &mut adapter).unwrap(),
        DiffusionRunOutcome::Idle
    ));
    assert_eq!(adapter.cleanup_calls, 2, "cleanup must be retried");
    assert_eq!(
        orchestrator.session(session).unwrap().error,
        None,
        "a successful retry must clear the outstanding cleanup error"
    );

    let second_reclaim = orchestrator.reclaim_closed_sessions();
    assert_eq!(second_reclaim.reclaimed, vec![session]);
    assert!(second_reclaim.deferred_terminal_notification.is_empty());
    orchestrator.run_one(key, &mut adapter).unwrap();
    assert_eq!(adapter.cleanup_calls, 2, "cleanup must not be duplicated");
}
