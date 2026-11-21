//! Model-agnostic request lifecycle and execution orchestration for Claymore.
//!
//! M6 deliberately provides a synchronous runtime core. It owns request IDs,
//! per-request context, cancellation and lifecycle reporting, but it does not
//! create CUDA resources, schedule concurrent work, tokenize prompts, or
//! implement model-specific generation. Those responsibilities belong to
//! later phases and to model strategies behind the boundary below.
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use thiserror::Error;

/// Opaque, monotonically allocated identifier for one runtime request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(u64);

impl RequestId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Cooperative cancellation signal shared with the caller and strategy.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Strategies should check this between bounded units
    /// of work; cancellation cannot forcibly interrupt a blocking backend call.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Model-neutral input envelope. The runtime treats payload bytes as opaque.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RequestInput {
    pub payload: Vec<u8>,
    pub metadata: BTreeMap<String, String>,
}

impl RequestInput {
    pub fn new(payload: impl Into<Vec<u8>>) -> Self {
        Self {
            payload: payload.into(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Model-neutral output envelope. No tensor or tokenizer semantics are implied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestOutput {
    pub payload: Vec<u8>,
}

impl RequestOutput {
    pub fn new(payload: impl Into<Vec<u8>>) -> Self {
        Self {
            payload: payload.into(),
        }
    }
}

/// Explicit lifecycle states for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Created,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl RequestState {
    const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Error returned by runtime construction or request admission.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RuntimeError {
    #[error("max_input_bytes must be greater than zero")]
    InvalidConfig,
    #[error("request payload is {actual_bytes} bytes; configured maximum is {max_bytes}")]
    InputTooLarge {
        actual_bytes: usize,
        max_bytes: usize,
    },
    #[error("request was cancelled")]
    Cancelled,
    #[error("generation strategy failed: {0}")]
    StrategyFailed(String),
    #[error("request identifier space exhausted")]
    RequestIdExhausted,
}

/// Limits that can be enforced without knowing the model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub max_input_bytes: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_input_bytes: 1024 * 1024,
        }
    }
}

impl RuntimeConfig {
    pub fn validate(self) -> Result<Self, RuntimeError> {
        if self.max_input_bytes == 0 {
            return Err(RuntimeError::InvalidConfig);
        }
        Ok(self)
    }
}

/// Read-only request-scoped information exposed to a generation strategy.
pub struct ExecutionContext<'a> {
    request_id: RequestId,
    metadata: &'a BTreeMap<String, String>,
    cancellation: CancellationToken,
}

impl<'a> ExecutionContext<'a> {
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        self.metadata
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    pub fn check_cancelled(&self) -> Result<(), RuntimeError> {
        if self.cancellation.is_cancelled() {
            Err(RuntimeError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Strategy boundary: model architecture and generation mode stay outside the
/// runtime. Implementations may be thread-affine; no Send/Sync bound is imposed.
pub trait GenerationStrategy {
    fn execute(
        &mut self,
        context: &mut ExecutionContext<'_>,
        input: &RequestInput,
    ) -> Result<RequestOutput, String>;
}

/// Immutable result of one synchronous request execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReport {
    pub request_id: RequestId,
    pub state: RequestState,
    /// Includes the initial Created state and exactly one terminal state.
    pub transitions: Vec<RequestState>,
    pub output: Option<RequestOutput>,
    pub error: Option<RuntimeError>,
}

impl ExecutionReport {
    fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            state: RequestState::Created,
            transitions: vec![RequestState::Created],
            output: None,
            error: None,
        }
    }

    fn transition(&mut self, next: RequestState) {
        debug_assert!(!self.state.is_terminal(), "terminal request state cannot transition");
        debug_assert!(
            matches!(
                (self.state, next),
                (RequestState::Created, RequestState::Running)
                    | (RequestState::Created, RequestState::Cancelled)
                    | (RequestState::Running, RequestState::Completed)
                    | (RequestState::Running, RequestState::Failed)
                    | (RequestState::Running, RequestState::Cancelled)
            ),
            "invalid request lifecycle transition"
        );
        self.state = next;
        self.transitions.push(next);
    }
}

/// Synchronous, model-agnostic runtime.
///
/// The runtime owns request lifecycle and delegates actual work to one strategy.
/// It intentionally does not promise concurrent execution or CUDA thread
/// migration; the current CUDA backend remains confined to its owner thread.
pub struct Runtime<S> {
    config: RuntimeConfig,
    strategy: S,
    next_request_id: u64,
}

impl<S: GenerationStrategy> Runtime<S> {
    pub fn new(config: RuntimeConfig, strategy: S) -> Result<Self, RuntimeError> {
        Ok(Self {
            config: config.validate()?,
            strategy,
            next_request_id: 1,
        })
    }

    pub fn config(&self) -> RuntimeConfig {
        self.config
    }

    pub fn strategy(&self) -> &S {
        &self.strategy
    }

    pub fn strategy_mut(&mut self) -> &mut S {
        &mut self.strategy
    }

    pub fn execute(
        &mut self,
        input: RequestInput,
        cancellation: CancellationToken,
    ) -> Result<ExecutionReport, RuntimeError> {
        if input.payload.len() > self.config.max_input_bytes {
            return Err(RuntimeError::InputTooLarge {
                actual_bytes: input.payload.len(),
                max_bytes: self.config.max_input_bytes,
            });
        }

        let request_id = RequestId(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(RuntimeError::RequestIdExhausted)?;

        let mut report = ExecutionReport::new(request_id);
        if cancellation.is_cancelled() {
            report.transition(RequestState::Cancelled);
            report.error = Some(RuntimeError::Cancelled);
            return Ok(report);
        }

        report.transition(RequestState::Running);
        let mut context = ExecutionContext {
            request_id,
            metadata: &input.metadata,
            cancellation: cancellation.clone(),
        };

        let result = self.strategy.execute(&mut context, &input);
        if cancellation.is_cancelled() {
            report.transition(RequestState::Cancelled);
            report.error = Some(RuntimeError::Cancelled);
        } else {
            match result {
                Ok(output) => {
                    report.transition(RequestState::Completed);
                    report.output = Some(output);
                }
                Err(error) => {
                    report.transition(RequestState::Failed);
                    report.error = Some(RuntimeError::StrategyFailed(error));
                }
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct EchoStrategy;

    impl GenerationStrategy for EchoStrategy {
        fn execute(
            &mut self,
            context: &mut ExecutionContext<'_>,
            input: &RequestInput,
        ) -> Result<RequestOutput, String> {
            context.check_cancelled().map_err(|error| error.to_string())?;
            Ok(RequestOutput::new(input.payload.clone()))
        }
    }

    struct FailingStrategy;

    impl GenerationStrategy for FailingStrategy {
        fn execute(
            &mut self,
            _context: &mut ExecutionContext<'_>,
            _input: &RequestInput,
        ) -> Result<RequestOutput, String> {
            Err("synthetic strategy failure".to_owned())
        }
    }

    struct CancellingStrategy(CancellationToken);

    impl GenerationStrategy for CancellingStrategy {
        fn execute(
            &mut self,
            _context: &mut ExecutionContext<'_>,
            input: &RequestInput,
        ) -> Result<RequestOutput, String> {
            self.0.cancel();
            Ok(RequestOutput::new(input.payload.clone()))
        }
    }

    struct MustNotRunStrategy(Cell<bool>);

    impl GenerationStrategy for MustNotRunStrategy {
        fn execute(
            &mut self,
            _context: &mut ExecutionContext<'_>,
            _input: &RequestInput,
        ) -> Result<RequestOutput, String> {
            self.0.set(true);
            Ok(RequestOutput::new([]))
        }
    }

    #[test]
    fn runtime_executes_a_model_neutral_request_end_to_end() {
        let mut runtime = Runtime::new(RuntimeConfig::default(), EchoStrategy).unwrap();
        let input = RequestInput::new(b"hello".to_vec()).with_metadata("trace", "test-1");
        let report = runtime.execute(input, CancellationToken::new()).unwrap();

        assert_eq!(report.request_id.get(), 1);
        assert_eq!(report.state, RequestState::Completed);
        assert_eq!(
            report.transitions,
            [
                RequestState::Created,
                RequestState::Running,
                RequestState::Completed
            ]
        );
        assert_eq!(report.output, Some(RequestOutput::new(b"hello".to_vec())));
        assert_eq!(report.error, None);
    }

    #[test]
    fn request_ids_are_monotonic_and_runtime_state_is_reused() {
        let mut runtime = Runtime::new(RuntimeConfig::default(), EchoStrategy).unwrap();
        let first = runtime
            .execute(RequestInput::new([1]), CancellationToken::new())
            .unwrap();
        let second = runtime
            .execute(RequestInput::new([2]), CancellationToken::new())
            .unwrap();
        assert_eq!(first.request_id.get(), 1);
        assert_eq!(second.request_id.get(), 2);
    }

    #[test]
    fn pre_cancelled_request_never_enters_strategy() {
        let called = Cell::new(false);
        let mut runtime =
            Runtime::new(RuntimeConfig::default(), MustNotRunStrategy(called)).unwrap();
        let token = CancellationToken::new();
        token.cancel();
        let report = runtime
            .execute(RequestInput::new([1]), token)
            .unwrap();
        assert_eq!(report.state, RequestState::Cancelled);
        assert_eq!(report.transitions, [RequestState::Created, RequestState::Cancelled]);
        assert_eq!(report.error, Some(RuntimeError::Cancelled));
        assert!(!runtime.strategy().0.get());
    }

    #[test]
    fn cancellation_during_strategy_discards_output() {
        let token = CancellationToken::new();
        let mut runtime = Runtime::new(
            RuntimeConfig::default(),
            CancellingStrategy(token.clone()),
        )
        .unwrap();
        let report = runtime
            .execute(RequestInput::new([1, 2]), token)
            .unwrap();
        assert_eq!(report.state, RequestState::Cancelled);
        assert_eq!(report.output, None);
        assert_eq!(report.error, Some(RuntimeError::Cancelled));
    }

    #[test]
    fn strategy_failure_is_reported_as_failed_request() {
        let mut runtime = Runtime::new(RuntimeConfig::default(), FailingStrategy).unwrap();
        let report = runtime
            .execute(RequestInput::new([1]), CancellationToken::new())
            .unwrap();
        assert_eq!(report.state, RequestState::Failed);
        assert_eq!(
            report.error,
            Some(RuntimeError::StrategyFailed(
                "synthetic strategy failure".to_owned()
            ))
        );
    }

    #[test]
    fn oversized_input_is_rejected_before_request_id_is_consumed() {
        let config = RuntimeConfig { max_input_bytes: 2 };
        let mut runtime = Runtime::new(config, EchoStrategy).unwrap();
        assert_eq!(
            runtime.execute(RequestInput::new([1, 2, 3]), CancellationToken::new()),
            Err(RuntimeError::InputTooLarge {
                actual_bytes: 3,
                max_bytes: 2
            })
        );
        let report = runtime
            .execute(RequestInput::new([1, 2]), CancellationToken::new())
            .unwrap();
        assert_eq!(report.request_id.get(), 1);
    }

    #[test]
    fn invalid_runtime_configuration_is_rejected() {
        assert!(matches!(
            Runtime::new(RuntimeConfig { max_input_bytes: 0 }, EchoStrategy),
            Err(RuntimeError::InvalidConfig)
        ));
    }
}
