use claymore_runtime::{
    BlockId, CancellationToken, DiffusionExecutor, DiffusionOrchestrator, DiffusionPlan,
    DiffusionRunOutcome, DiffusionSessionState, DiffusionStep,
};
use claymore_scheduler::{SchedulerConfig, TaskPriority};

fn plan(blocks: u32, denoise_steps: u32) -> DiffusionPlan {
    let mut steps = Vec::new();
    for block in 0..blocks {
        steps.push(DiffusionStep::PrepareBlock { block: BlockId(block) });
        for iteration in 0..denoise_steps {
            steps.push(DiffusionStep::Denoise {
                block: BlockId(block),
                iteration,
            });
        }
        steps.push(DiffusionStep::ValidateBlock { block: BlockId(block) });
        steps.push(DiffusionStep::CommitBlock { block: BlockId(block) });
    }
    DiffusionPlan::new(steps).expect("synthetic plan is valid")
}

#[derive(Default)]
struct RecordingExecutor {
    calls: Vec<DiffusionStep>,
}

impl DiffusionExecutor for RecordingExecutor {
    fn execute_step(&mut self, step: DiffusionStep) -> Result<(), String> {
        self.calls.push(step);
        Ok(())
    }
}

#[test]
fn synthetic_multiblock_session_reports_progress_and_phase_metrics() {
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let id = orchestrator
        .submit(plan(3, 2), CancellationToken::new(), TaskPriority::Normal)
        .unwrap();
    let mut executor = RecordingExecutor::default();

    let initial = orchestrator.metrics();
    assert_eq!(initial.blocks_submitted, 3);
    assert_eq!(initial.blocks_completed, 0);
    assert_eq!(initial.blocks_pending, 3);

    // A block is complete only after its CommitBlock step succeeds.
    for _ in 0..5 {
        assert!(matches!(
            orchestrator.run_one(&mut executor).unwrap(),
            DiffusionRunOutcome::StepCompleted { .. }
        ));
    }
    let progress = orchestrator.metrics();
    assert_eq!(progress.blocks_completed, 1);
    assert_eq!(progress.blocks_pending, 2);
    assert_eq!(orchestrator.session(id).unwrap().completed_steps, 5);

    loop {
        match orchestrator.run_one(&mut executor).unwrap() {
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            DiffusionRunOutcome::StepCompleted { .. } => {}
            other => panic!("unexpected synthetic outcome: {other:?}"),
        }
    }

    let final_metrics = orchestrator.metrics();
    assert_eq!(orchestrator.session(id).unwrap().state, DiffusionSessionState::Completed);
    assert_eq!(final_metrics.blocks_completed, 3);
    assert_eq!(final_metrics.blocks_pending, 0);
    assert_eq!(final_metrics.steps_started, 15);
    assert_eq!(final_metrics.steps_completed, 15);
    assert_eq!(final_metrics.sessions_completed, 1);
    assert_eq!(executor.calls.len(), 15);
    assert_eq!(
        final_metrics.execution_duration_ns,
        final_metrics.prepare_duration_ns
            + final_metrics.denoise_duration_ns
            + final_metrics.validate_duration_ns
            + final_metrics.commit_duration_ns
    );
}

#[test]
fn synthetic_failure_isolated_and_reported_without_gpu() {
    struct FailOnFirst;
    impl DiffusionExecutor for FailOnFirst {
        fn execute_step(&mut self, _step: DiffusionStep) -> Result<(), String> {
            Err("synthetic CPU failure".to_owned())
        }
    }

    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let failed = orchestrator
        .submit(plan(2, 1), CancellationToken::new(), TaskPriority::High)
        .unwrap();
    let independent = orchestrator
        .submit(plan(1, 1), CancellationToken::new(), TaskPriority::Low)
        .unwrap();

    assert!(matches!(
        orchestrator.run_one(&mut FailOnFirst).unwrap(),
        DiffusionRunOutcome::StepFailed { .. }
    ));
    assert_eq!(orchestrator.session(failed).unwrap().state, DiffusionSessionState::Failed);
    assert_eq!(orchestrator.metrics().sessions_failed, 1);
    assert_eq!(orchestrator.metrics().steps_failed, 1);
    assert_eq!(orchestrator.metrics().blocks_pending, 1);

    let mut executor = RecordingExecutor::default();
    loop {
        match orchestrator.run_one(&mut executor).unwrap() {
            DiffusionRunOutcome::SessionCompleted { .. } => break,
            DiffusionRunOutcome::StepCompleted { .. } => {}
            other => panic!("unexpected independent-session outcome: {other:?}"),
        }
    }
    assert_eq!(orchestrator.session(independent).unwrap().state, DiffusionSessionState::Completed);
    assert_eq!(orchestrator.metrics().sessions_completed, 1);
}

#[test]
fn synthetic_cancellation_is_observed_at_a_safe_boundary() {
    let token = CancellationToken::new();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let id = orchestrator
        .submit(plan(2, 1), token.clone(), TaskPriority::Normal)
        .unwrap();

    struct CancelAfterOne(CancellationToken);
    impl DiffusionExecutor for CancelAfterOne {
        fn execute_step(&mut self, _step: DiffusionStep) -> Result<(), String> {
            self.0.cancel();
            Ok(())
        }
    }

    assert!(matches!(
        orchestrator.run_one(&mut CancelAfterOne(token)).unwrap(),
        DiffusionRunOutcome::SessionCancelled { .. }
    ));
    let metrics = orchestrator.metrics();
    assert_eq!(orchestrator.session(id).unwrap().state, DiffusionSessionState::Cancelled);
    assert_eq!(metrics.steps_completed, 1);
    assert_eq!(metrics.steps_cancelled, 0);
    assert_eq!(metrics.sessions_cancelled, 1);
    assert_eq!(metrics.blocks_pending, 0);
}

#[test]
fn pre_cancelled_plan_is_visible_but_never_scheduled() {
    let token = CancellationToken::new();
    token.cancel();
    let mut orchestrator = DiffusionOrchestrator::new(SchedulerConfig::default()).unwrap();
    let id = orchestrator.submit(plan(2, 1), token, TaskPriority::Normal).unwrap();

    assert_eq!(orchestrator.session(id).unwrap().state, DiffusionSessionState::Cancelled);
    assert_eq!(orchestrator.metrics().blocks_submitted, 2);
    assert_eq!(orchestrator.metrics().blocks_pending, 0);
    assert_eq!(orchestrator.metrics().steps_started, 0);
    assert!(matches!(
        orchestrator.run_one(&mut RecordingExecutor::default()).unwrap(),
        DiffusionRunOutcome::Idle
    ));
}
