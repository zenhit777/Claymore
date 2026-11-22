//! Explicit operation contracts and deterministic backend dispatch.
//!
//! Every M8 operation is described before it is executed: dtype, layout,
//! shape rules, aliasing/mutability, and the backend capabilities required to
//! run it. Dispatch never casts, broadcasts, silently falls back to the CPU or
//! copies host memory to the device. A rejected operation is always explained
//! by a typed reason.

use std::fmt;

use claymore_core::MemoryTier;
use claymore_device::{CpuDevice, DType, Device, DeviceKind};

use crate::ops::{self, OpError};
use crate::{Bf16, Tensor};

/// The complete set of tensor operations covered by the M8 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Add,
    MatMul,
    SoftmaxLastDim,
    RmsNorm,
}

impl OpKind {
    pub const ALL: [OpKind; 4] = [
        OpKind::Add,
        OpKind::MatMul,
        OpKind::SoftmaxLastDim,
        OpKind::RmsNorm,
    ];

    pub fn name(self) -> &'static str {
        match self {
            OpKind::Add => "add",
            OpKind::MatMul => "matmul",
            OpKind::SoftmaxLastDim => "softmax_last_dim",
            OpKind::RmsNorm => "rms_norm",
        }
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Layout forms accepted by an operation. Strided inputs are never silently
/// materialized into a contiguous temporary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutRequirement {
    Contiguous,
}

impl fmt::Display for LayoutRequirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutRequirement::Contiguous => formatter.write_str("contiguous"),
        }
    }
}

/// M8 aliasing rule: inputs are read-only and the result is always a freshly
/// allocated tensor. There is no in-place or output-aliasing path yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasRule {
    ReadOnlyInputsFreshOutput,
}

/// A backend declares what it can actually execute. Capabilities are facts
/// supplied by the backend implementation, not assumptions made by dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSupport {
    pub kind: OpKind,
    pub dtype: DType,
    pub input_tier: MemoryTier,
    pub output_tier: MemoryTier,
    pub layout: LayoutRequirement,
    pub alias_rule: AliasRule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub name: &'static str,
    pub device_kind: DeviceKind,
    pub supported_ops: &'static [OpSupport],
}

impl BackendCapabilities {
    /// Returns the exact matching support entry, never a compatible-looking
    /// substitute such as BF16 for F32 or strided for contiguous.
    pub fn support_for(
        &self,
        kind: OpKind,
        dtype: DType,
        input_tier: MemoryTier,
        output_tier: MemoryTier,
        layout: LayoutRequirement,
        alias_rule: AliasRule,
    ) -> Option<&'static OpSupport> {
        self.supported_ops.iter().find(|support| {
            support.kind == kind
                && support.dtype == dtype
                && support.input_tier == input_tier
                && support.output_tier == output_tier
                && support.layout == layout
                && support.alias_rule == alias_rule
        })
    }
}

/// A fully described operation. Tensors are borrowed: execution never mutates
/// an input and never takes ownership away from the caller.
#[derive(Debug)]
pub enum TensorOp<'a> {
    Add {
        lhs: &'a Tensor,
        rhs: &'a Tensor,
    },
    MatMul {
        lhs: &'a Tensor,
        rhs: &'a Tensor,
    },
    SoftmaxLastDim {
        input: &'a Tensor,
    },
    RmsNorm {
        input: &'a Tensor,
        scale: &'a Tensor,
        epsilon: f32,
    },
}

impl<'a> TensorOp<'a> {
    pub fn kind(&self) -> OpKind {
        match self {
            TensorOp::Add { .. } => OpKind::Add,
            TensorOp::MatMul { .. } => OpKind::MatMul,
            TensorOp::SoftmaxLastDim { .. } => OpKind::SoftmaxLastDim,
            TensorOp::RmsNorm { .. } => OpKind::RmsNorm,
        }
    }

    pub fn inputs(&self) -> Vec<&'a Tensor> {
        match self {
            TensorOp::Add { lhs, rhs } | TensorOp::MatMul { lhs, rhs } => vec![lhs, rhs],
            TensorOp::SoftmaxLastDim { input } => vec![input],
            TensorOp::RmsNorm { input, scale, .. } => vec![input, scale],
        }
    }

    pub fn output_shape(&self) -> Vec<usize> {
        match self {
            TensorOp::Add { lhs, .. } => lhs.layout().shape().to_vec(),
            TensorOp::MatMul { lhs, rhs } => {
                vec![lhs.layout().shape()[0], rhs.layout().shape()[1]]
            }
            TensorOp::SoftmaxLastDim { input } => input.layout().shape().to_vec(),
            TensorOp::RmsNorm { input, .. } => input.layout().shape().to_vec(),
        }
    }

    pub fn alias_rule(&self) -> AliasRule {
        AliasRule::ReadOnlyInputsFreshOutput
    }

    /// Validate shape, dtype, layout, epsilon and device consistency before
    /// any backend is considered. This is the single source of truth for the
    /// CPU reference and CUDA execution paths.
    pub fn validate(&self) -> Result<ValidatedOp, OpError> {
        let inputs = self.inputs();
        let first = inputs.first().ok_or(OpError::InvalidRank)?;
        let dtype = first.layout().dtype();
        // FP16 (and every other metadata-only dtype) is explicitly refused:
        // no CPU reference, kernel or conversion path exists for it.
        if !matches!(dtype, DType::F32 | DType::BF16) {
            return Err(OpError::UnsupportedDType);
        }
        for input in &inputs {
            if input.layout().dtype() != dtype {
                return Err(OpError::MixedDType);
            }
            if !input.layout().is_contiguous() {
                return Err(OpError::NonContiguous);
            }
            if input.storage().device_instance_id() != first.storage().device_instance_id() {
                return Err(OpError::MixedDevices);
            }
        }
        match self {
            TensorOp::Add { lhs, rhs } => {
                if lhs.layout().shape() != rhs.layout().shape() {
                    return Err(OpError::ShapeMismatch);
                }
            }
            TensorOp::MatMul { lhs, rhs } => {
                let left = lhs.layout().shape();
                let right = rhs.layout().shape();
                if left.len() != 2 || right.len() != 2 || left[1] != right[0] {
                    return Err(OpError::ShapeMismatch);
                }
            }
            TensorOp::SoftmaxLastDim { input } => {
                if input.layout().shape().is_empty() {
                    return Err(OpError::InvalidRank);
                }
            }
            TensorOp::RmsNorm {
                input,
                scale,
                epsilon,
            } => {
                if input.layout().shape().is_empty() {
                    return Err(OpError::InvalidRank);
                }
                let width = input.layout().shape().last().copied().unwrap_or(0);
                if scale.layout().shape() != [width] {
                    return Err(OpError::ScaleMismatch);
                }
                if !epsilon.is_finite() || *epsilon < 0.0 {
                    return Err(OpError::InvalidEpsilon);
                }
            }
        }
        Ok(ValidatedOp {
            kind: self.kind(),
            dtype,
            shape: self.output_shape(),
            input_tier: first.storage().tier(),
            output_tier: first.storage().tier(),
            layout: LayoutRequirement::Contiguous,
            alias_rule: self.alias_rule(),
            device_instance_id: first.storage().device_instance_id(),
        })
    }
}

/// Backend-neutral launch dimensions derived from a validated operation. It
/// contains no CUDA types and can be checked without a GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeSignature {
    Elementwise {
        numel: usize,
    },
    MatMul {
        m: usize,
        k: usize,
        n: usize,
    },
    LastDim {
        rows: usize,
        width: usize,
        epsilon: Option<f32>,
    },
}

/// CUDA grid limits for the launch configurations used by the M8 kernels.
const CUDA_GRID_X_MAX: usize = 2_147_483_647;
const CUDA_GRID_Y_MAX: usize = 65_535;

fn cuda_grid_limit(signature: ShapeSignature) -> Option<UnsupportedReason> {
    let (grid_x, grid_y) = match signature {
        ShapeSignature::Elementwise { numel } => (numel.div_ceil(256), 1),
        ShapeSignature::MatMul { m, n, .. } => (n.div_ceil(16), m.div_ceil(16)),
        ShapeSignature::LastDim { rows, .. } => (rows, 1),
    };
    if grid_x > CUDA_GRID_X_MAX {
        return Some(UnsupportedReason::LaunchLimit {
            axis: "x",
            blocks: grid_x,
            maximum: CUDA_GRID_X_MAX,
        });
    }
    if grid_y > CUDA_GRID_Y_MAX {
        return Some(UnsupportedReason::LaunchLimit {
            axis: "y",
            blocks: grid_y,
            maximum: CUDA_GRID_Y_MAX,
        });
    }
    None
}

impl TensorOp<'_> {
    pub fn shape_signature(&self) -> ShapeSignature {
        match self {
            TensorOp::Add { lhs, .. } => ShapeSignature::Elementwise {
                numel: lhs.layout().numel(),
            },
            TensorOp::MatMul { lhs, rhs } => ShapeSignature::MatMul {
                m: lhs.layout().shape()[0],
                k: lhs.layout().shape()[1],
                n: rhs.layout().shape()[1],
            },
            TensorOp::SoftmaxLastDim { input } => {
                let width = input.layout().shape().last().copied().unwrap_or(0);
                ShapeSignature::LastDim {
                    rows: input.layout().numel() / width,
                    width,
                    epsilon: None,
                }
            }
            TensorOp::RmsNorm { input, epsilon, .. } => {
                let width = input.layout().shape().last().copied().unwrap_or(0);
                ShapeSignature::LastDim {
                    rows: input.layout().numel() / width,
                    width,
                    epsilon: Some(*epsilon),
                }
            }
        }
    }
}

/// Result of the contract checks; used by dispatch and by backend adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedOp {
    pub kind: OpKind,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub input_tier: MemoryTier,
    pub output_tier: MemoryTier,
    pub layout: LayoutRequirement,
    pub alias_rule: AliasRule,
    pub device_instance_id: claymore_device::DeviceInstanceId,
}

/// Reasons a backend may not execute an operation. Every refusal is typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsupportedReason {
    OperationNotImplemented,
    /// The operation's CUDA launch grid exceeds a hardware axis limit.
    LaunchLimit {
        axis: &'static str,
        blocks: usize,
        maximum: usize,
    },
    DType { dtype: DType },
    Layout { layout: LayoutRequirement },
    Tier { tier: MemoryTier },
    Aliasing { rule: AliasRule },
    DeviceKind { kind: DeviceKind },
    LowerPriority,
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnsupportedReason::OperationNotImplemented => {
                formatter.write_str("operation is not implemented by this backend")
            }
            UnsupportedReason::LaunchLimit { axis, blocks, maximum } => {
                write!(formatter, "CUDA grid axis {axis} requires {blocks} blocks; maximum is {maximum}")
            }
            UnsupportedReason::DType { dtype } => {
                write!(formatter, "dtype {dtype:?} is not implemented")
            }
            UnsupportedReason::Layout { layout } => write!(formatter, "requires {layout} layout"),
            UnsupportedReason::Tier { tier } => {
                write!(formatter, "tier {tier:?} is not implemented")
            }
            UnsupportedReason::Aliasing { rule } => {
                write!(formatter, "aliasing rule {rule:?} is not implemented")
            }
            UnsupportedReason::DeviceKind { kind } => {
                write!(formatter, "device kind {kind:?} is not implemented")
            }
            UnsupportedReason::LowerPriority => {
                formatter.write_str("capable, but a higher-priority backend was selected")
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DispatchError {
    UnknownBackend(String),
    NoCapableBackend {
        op: OpKind,
        refusals: Vec<(String, UnsupportedReason)>,
    },
    Contract(OpError),
}

impl From<OpError> for DispatchError {
    fn from(error: OpError) -> Self {
        DispatchError::Contract(error)
    }
}

impl fmt::Display for DispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchError::UnknownBackend(name) => write!(formatter, "unknown backend {name}"),
            DispatchError::NoCapableBackend { op, refusals } => {
                write!(formatter, "no backend can execute {op}:")?;
                for (backend, reason) in refusals {
                    write!(formatter, " {backend} refuses ({reason}),")?;
                }
                Ok(())
            }
            DispatchError::Contract(error) => write!(formatter, "invalid operation: {error}"),
        }
    }
}

impl std::error::Error for DispatchError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchPlan {
    pub backend: &'static str,
    pub op: OpKind,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendDecision {
    Selected(DispatchPlan),
    Refused {
        backend: &'static str,
        reason: UnsupportedReason,
    },
}

/// Complete, human-readable explanation of a dispatch decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchReport {
    pub op: OpKind,
    pub decisions: Vec<BackendDecision>,
}

impl fmt::Display for DispatchReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "dispatch for {}", self.op)?;
        for decision in &self.decisions {
            match decision {
                BackendDecision::Selected(plan) => {
                    writeln!(formatter, "  SELECT {} — {}", plan.backend, plan.reason)?;
                }
                BackendDecision::Refused { backend, reason } => {
                    writeln!(formatter, "  REFUSE {backend} — {reason}")?;
                }
            }
        }
        Ok(())
    }
}

/// A backend announces its real capabilities and performs execution. It must
/// not fall back to another backend internally.
pub trait TensorComputeBackend {
    fn capabilities(&self) -> BackendCapabilities;
    fn execute(&self, op: &TensorOp<'_>) -> Result<Tensor, OpError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendPreference {
    /// Deterministic registration-order selection among capable backends.
    Auto,
    /// Require a named backend. A refusal is an error, never a fallback.
    Explicit(String),
}

#[derive(Debug)]
pub struct ExecutionRequest<'a> {
    pub op: TensorOp<'a>,
    pub preference: BackendPreference,
}

impl<'a> ExecutionRequest<'a> {
    pub fn new(op: TensorOp<'a>) -> Self {
        Self {
            op,
            preference: BackendPreference::Auto,
        }
    }

    pub fn on(mut self, backend: impl Into<String>) -> Self {
        self.preference = BackendPreference::Explicit(backend.into());
        self
    }
}

#[derive(Debug)]
pub enum ExecutionError {
    Dispatch(DispatchError),
    Operation(OpError),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionError::Dispatch(error) => error.fmt(formatter),
            ExecutionError::Operation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<DispatchError> for ExecutionError {
    fn from(error: DispatchError) -> Self {
        ExecutionError::Dispatch(error)
    }
}

impl From<OpError> for ExecutionError {
    fn from(error: OpError) -> Self {
        ExecutionError::Operation(error)
    }
}

/// Deterministic dispatcher. Backends are evaluated in registration order;
/// the first exact capability match wins. No implicit conversions or fallbacks.
pub struct TensorDispatcher {
    backends: Vec<Box<dyn TensorComputeBackend>>,
}

impl TensorDispatcher {
    pub fn new(backends: Vec<Box<dyn TensorComputeBackend>>) -> Self {
        Self { backends }
    }

    pub fn backend_names(&self) -> Vec<&'static str> {
        self.backends
            .iter()
            .map(|backend| backend.capabilities().name)
            .collect()
    }

    fn refusal(
        capabilities: &BackendCapabilities,
        validated: &ValidatedOp,
        op: &TensorOp<'_>,
    ) -> UnsupportedReason {
        if capabilities.device_kind != DeviceKind::Cpu
            && capabilities.device_kind != DeviceKind::Cuda
        {
            return UnsupportedReason::DeviceKind {
                kind: capabilities.device_kind,
            };
        }
        if capabilities.device_kind == DeviceKind::Cuda {
            if let Some(reason) = cuda_grid_limit(op.shape_signature()) {
                return reason;
            }
        }
        let support = capabilities.support_for(
            validated.kind,
            validated.dtype,
            validated.input_tier,
            validated.output_tier,
            validated.layout,
            validated.alias_rule,
        );
        match support {
            Some(_) => UnsupportedReason::OperationNotImplemented,
            None if !capabilities
                .supported_ops
                .iter()
                .any(|entry| entry.kind == validated.kind) =>
            {
                UnsupportedReason::OperationNotImplemented
            }
            None if !capabilities
                .supported_ops
                .iter()
                .any(|entry| entry.dtype == validated.dtype) =>
            {
                UnsupportedReason::DType {
                    dtype: validated.dtype,
                }
            }
            None if !capabilities
                .supported_ops
                .iter()
                .any(|entry| entry.input_tier == validated.input_tier) =>
            {
                UnsupportedReason::Tier {
                    tier: validated.input_tier,
                }
            }
            None if !capabilities
                .supported_ops
                .iter()
                .any(|entry| entry.layout == validated.layout) =>
            {
                UnsupportedReason::Layout {
                    layout: validated.layout,
                }
            }
            None => UnsupportedReason::Aliasing {
                rule: validated.alias_rule,
            },
        }
    }

    fn is_supported(
        capabilities: &BackendCapabilities,
        validated: &ValidatedOp,
        op: &TensorOp<'_>,
    ) -> bool {
        (capabilities.device_kind != DeviceKind::Cuda
            || cuda_grid_limit(op.shape_signature()).is_none())
            && capabilities
                .support_for(
                    validated.kind,
                    validated.dtype,
                    validated.input_tier,
                    validated.output_tier,
                    validated.layout,
                    validated.alias_rule,
                )
                .is_some()
    }

    /// Explain every candidate decision without executing anything.
    pub fn explain(&self, request: &ExecutionRequest<'_>) -> Result<DispatchReport, DispatchError> {
        let validated = request.op.validate()?;
        let mut report = DispatchReport {
            op: validated.kind,
            decisions: Vec::new(),
        };
        match &request.preference {
            BackendPreference::Auto => {
                let mut selected = false;
                for backend in &self.backends {
                    let capabilities = backend.capabilities();
                    if Self::is_supported(&capabilities, &validated, &request.op) {
                        if selected {
                            report.decisions.push(BackendDecision::Refused {
                                backend: capabilities.name,
                                reason: UnsupportedReason::LowerPriority,
                            });
                        } else {
                            report.decisions.push(BackendDecision::Selected(DispatchPlan {
                                backend: capabilities.name,
                                op: validated.kind,
                                reason: format!(
                                    "exact capability match: {} / {:?} / {} / {:?} / {:?}; first capable backend in registration order",
                                    validated.kind,
                                    validated.dtype,
                                    validated.layout,
                                    validated.input_tier,
                                    validated.output_tier
                                ),
                            }));
                            selected = true;
                        }
                    } else {
                        report.decisions.push(BackendDecision::Refused {
                            backend: capabilities.name,
                            reason: Self::refusal(&capabilities, &validated, &request.op),
                        });
                    }
                }
                if !report
                    .decisions
                    .iter()
                    .any(|decision| matches!(decision, BackendDecision::Selected(_)))
                {
                    let refusals = report
                        .decisions
                        .iter()
                        .map(|decision| match decision {
                            BackendDecision::Refused { backend, reason } => {
                                ((*backend).to_owned(), reason.clone())
                            }
                            BackendDecision::Selected(_) => unreachable!(),
                        })
                        .collect();
                    return Err(DispatchError::NoCapableBackend {
                        op: validated.kind,
                        refusals,
                    });
                }
            }
            BackendPreference::Explicit(name) => {
                let Some(backend) = self
                    .backends
                    .iter()
                    .find(|backend| backend.capabilities().name == name)
                else {
                    return Err(DispatchError::UnknownBackend(name.clone()));
                };
                let capabilities = backend.capabilities();
                if Self::is_supported(&capabilities, &validated) {
                    report
                        .decisions
                        .push(BackendDecision::Selected(DispatchPlan {
                            backend: capabilities.name,
                            op: validated.kind,
                            reason: "explicitly requested backend has an exact capability match"
                                .into(),
                        }));
                } else {
                    let reason = Self::refusal(&capabilities, &validated);
                    report.decisions.push(BackendDecision::Refused {
                        backend: capabilities.name,
                        reason: reason.clone(),
                    });
                    return Err(DispatchError::NoCapableBackend {
                        op: validated.kind,
                        refusals: vec![(capabilities.name.to_owned(), reason)],
                    });
                }
            }
        }
        Ok(report)
    }

    /// Return the selected plan. Outputs can be inspected with `explain`.
    pub fn plan(&self, request: &ExecutionRequest<'_>) -> Result<DispatchPlan, DispatchError> {
        let report = self.explain(request)?;
        report
            .decisions
            .into_iter()
            .find_map(|decision| match decision {
                BackendDecision::Selected(plan) => Some(plan),
                BackendDecision::Refused { .. } => None,
            })
            .ok_or(DispatchError::NoCapableBackend {
                op: request.op.kind(),
                refusals: Vec::new(),
            })
    }

    pub fn execute(&self, request: &ExecutionRequest<'_>) -> Result<Tensor, ExecutionError> {
        let plan = self.plan(request)?;
        let backend = self
            .backends
            .iter()
            .find(|backend| backend.capabilities().name == plan.backend)
            .expect("a planned backend must be registered");
        Ok(backend.execute(&request.op)?)
    }
}

/// CPU F32 reference backend. It is intentionally restricted to Host buffers
/// on the CpuDevice: no VRAM allocation, no hidden host/device transfer.
pub struct CpuReferenceBackend {
    device: CpuDevice,
}

impl Default for CpuReferenceBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuReferenceBackend {
    pub fn new() -> Self {
        Self {
            device: CpuDevice::new(),
        }
    }

    /// Attach the reference backend to an existing CPU device instance so
    /// callers can create input tensors on the exact device used for reads.
    pub fn with_device(device: CpuDevice) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &CpuDevice {
        &self.device
    }

    /// Allocate a contiguous F32 Host tensor owned by this backend's device.
    pub fn tensor(&self, shape: Vec<usize>, values: &[f32]) -> Result<Tensor, OpError> {
        let layout = crate::TensorLayout::contiguous(shape, DType::F32)?;
        let expected = layout.numel();
        if values.len() != expected {
            return Err(OpError::ShapeMismatch);
        }
        let mut storage = self.device.allocate(
            MemoryTier::Host,
            u64::try_from(layout.span_bytes())
                .map_err(|_| OpError::Device(claymore_device::DeviceError::SizeOverflow))?,
        )?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        self.device.write(storage.as_mut(), 0, &bytes)?;
        Ok(Tensor::with_layout(storage, layout, 0)?)
    }

    /// Allocate a contiguous BF16 Host tensor with exact 16-bit storage.
    pub fn tensor_bf16(&self, shape: Vec<usize>, values: &[Bf16]) -> Result<Tensor, OpError> {
        let layout = crate::TensorLayout::contiguous(shape, DType::BF16)?;
        if values.len() != layout.numel() {
            return Err(OpError::ShapeMismatch);
        }
        let mut storage = self.device.allocate(
            MemoryTier::Host,
            u64::try_from(layout.span_bytes())
                .map_err(|_| OpError::Device(claymore_device::DeviceError::SizeOverflow))?,
        )?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        self.device.write(storage.as_mut(), 0, &bytes)?;
        Ok(Tensor::with_layout(storage, layout, 0)?)
    }
}

const CPU_OPS: [OpSupport; 8] = [
    OpSupport {
        kind: OpKind::Add,
        dtype: DType::F32,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::MatMul,
        dtype: DType::F32,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::SoftmaxLastDim,
        dtype: DType::F32,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::RmsNorm,
        dtype: DType::F32,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::Add,
        dtype: DType::BF16,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::MatMul,
        dtype: DType::BF16,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::SoftmaxLastDim,
        dtype: DType::BF16,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
    OpSupport {
        kind: OpKind::RmsNorm,
        dtype: DType::BF16,
        input_tier: MemoryTier::Host,
        output_tier: MemoryTier::Host,
        layout: LayoutRequirement::Contiguous,
        alias_rule: AliasRule::ReadOnlyInputsFreshOutput,
    },
];

impl TensorComputeBackend for CpuReferenceBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "cpu-reference",
            device_kind: DeviceKind::Cpu,
            supported_ops: &CPU_OPS,
        }
    }

    fn execute(&self, op: &TensorOp<'_>) -> Result<Tensor, OpError> {
        op.validate()?;
        match op {
            TensorOp::Add { lhs, rhs } => ops::add(&self.device, lhs, rhs),
            TensorOp::MatMul { lhs, rhs } => ops::matmul(&self.device, lhs, rhs),
            TensorOp::SoftmaxLastDim { input } => ops::softmax_last_dim(&self.device, input),
            TensorOp::RmsNorm {
                input,
                scale,
                epsilon,
            } => ops::rms_norm(&self.device, input, scale, *epsilon),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_core::MemoryTier;
    use claymore_device::{DType, Device};

    fn host_tensor(device: &CpuDevice, shape: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut storage = device
            .allocate(MemoryTier::Host, bytes.len() as u64)
            .unwrap();
        device.write(storage.as_mut(), 0, &bytes).unwrap();
        Tensor::new(storage, shape, DType::F32).unwrap()
    }

    fn dispatcher() -> (TensorDispatcher, CpuDevice) {
        let backend = CpuReferenceBackend::new();
        let device = backend.device().clone();
        (TensorDispatcher::new(vec![Box::new(backend)]), device)
    }

    fn f32_values(device: &CpuDevice, tensor: &Tensor) -> Vec<f32> {
        device
            .read(
                tensor.storage(),
                tensor.offset_bytes(),
                tensor.layout().span_bytes(),
            )
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn cuda_launch_grid_limits_are_axis_specific_and_typed() {
        assert_eq!(cuda_grid_limit(ShapeSignature::MatMul {
            m: 65_535 * 16,
            k: 1,
            n: 16,
        }), None);
        assert_eq!(
            cuda_grid_limit(ShapeSignature::MatMul {
                m: 65_536 * 16,
                k: 1,
                n: 16,
            }),
            Some(UnsupportedReason::LaunchLimit {
                axis: "y",
                blocks: 65_536,
                maximum: CUDA_GRID_Y_MAX,
            })
        );
        assert_eq!(
            cuda_grid_limit(ShapeSignature::LastDim {
                rows: CUDA_GRID_X_MAX + 1,
                width: 1,
                epsilon: None,
            }),
            Some(UnsupportedReason::LaunchLimit {
                axis: "x",
                blocks: CUDA_GRID_X_MAX + 1,
                maximum: CUDA_GRID_X_MAX,
            })
        );
    }

    #[test]
    fn dispatch_explains_an_exact_match() {
        let (dispatcher, device) = dispatcher();
        let lhs = host_tensor(&device, vec![2], &[1.0, 2.0]);
        let rhs = host_tensor(&device, vec![2], &[3.0, 4.0]);
        let request = ExecutionRequest::new(TensorOp::Add {
            lhs: &lhs,
            rhs: &rhs,
        });
        let report = dispatcher.explain(&request).unwrap();
        let text = report.to_string();
        assert!(text.contains("SELECT cpu-reference"), "{text}");
        assert!(text.contains("exact capability match"), "{text}");
        assert_eq!(dispatcher.plan(&request).unwrap().backend, "cpu-reference");
    }

    #[test]
    fn dispatch_executes_through_the_cpu_reference_backend() {
        let (dispatcher, device) = dispatcher();
        let lhs = host_tensor(&device, vec![2], &[1.0, 2.0]);
        let rhs = host_tensor(&device, vec![2], &[3.0, 4.0]);
        let result = dispatcher
            .execute(&ExecutionRequest::new(TensorOp::Add {
                lhs: &lhs,
                rhs: &rhs,
            }))
            .unwrap();
        assert_eq!(f32_values(&device, &result), [4.0, 6.0]);
    }

    #[test]
    fn dispatch_rejects_unknown_and_unsupported_backends_without_fallback() {
        let (dispatcher, device) = dispatcher();
        let lhs = host_tensor(&device, vec![2], &[1.0, 2.0]);
        let rhs = host_tensor(&device, vec![2], &[3.0, 4.0]);
        let request = ExecutionRequest::new(TensorOp::Add {
            lhs: &lhs,
            rhs: &rhs,
        })
        .on("nonexistent");
        assert_eq!(
            dispatcher.plan(&request).unwrap_err(),
            DispatchError::UnknownBackend("nonexistent".into())
        );
        assert_eq!(
            dispatcher.explain(&request).unwrap_err(),
            DispatchError::UnknownBackend("nonexistent".into())
        );
    }

    #[test]
    fn cpu_dispatch_refuses_vram_inputs_without_implicit_transfer() {
        let device = CpuDevice::new();
        let mut storage = device.allocate(MemoryTier::Vram, 8).unwrap();
        device.write(storage.as_mut(), 0, &[0; 8]).unwrap();
        let lhs = Tensor::new(storage, vec![2], DType::F32).unwrap();
        let rhs = host_tensor(&device, vec![2], &[3.0, 4.0]);
        let request = ExecutionRequest::new(TensorOp::Add {
            lhs: &lhs,
            rhs: &rhs,
        });
        assert!(matches!(
            dispatcher().0.plan(&request),
            Err(DispatchError::NoCapableBackend {
                refusals,
                ..
            }) if refusals.iter().any(|(_, reason)| matches!(reason, UnsupportedReason::Tier { .. }))
        ));
    }

    #[test]
    fn shape_signature_is_backend_neutral_and_checked_without_gpu() {
        let device = CpuDevice::new();
        let lhs = host_tensor(&device, vec![2, 3], &[1.0; 6]);
        let rhs = host_tensor(&device, vec![3, 4], &[2.0; 12]);
        assert_eq!(
            TensorOp::MatMul {
                lhs: &lhs,
                rhs: &rhs
            }
            .shape_signature(),
            ShapeSignature::MatMul { m: 2, k: 3, n: 4 }
        );
        let row = host_tensor(&device, vec![8], &[1.0; 8]);
        assert_eq!(
            TensorOp::SoftmaxLastDim { input: &row }.shape_signature(),
            ShapeSignature::LastDim {
                rows: 1,
                width: 8,
                epsilon: None
            }
        );
        let scale = host_tensor(&device, vec![8], &[1.0; 8]);
        assert_eq!(
            TensorOp::RmsNorm {
                input: &row,
                scale: &scale,
                epsilon: 1e-5
            }
            .shape_signature(),
            ShapeSignature::LastDim {
                rows: 1,
                width: 8,
                epsilon: Some(1e-5)
            }
        );
    }

    #[test]
    fn fp16_is_not_claimed_as_supported() {
        let device = CpuDevice::new();
        let mut storage = device.allocate(MemoryTier::Host, 4).unwrap();
        device.write(storage.as_mut(), 0, &[0; 4]).unwrap();
        let tensor = Tensor::new(storage, vec![2], DType::F16).unwrap();
        let request = ExecutionRequest::new(TensorOp::SoftmaxLastDim { input: &tensor });
        assert_eq!(
            dispatcher().0.plan(&request).unwrap_err(),
            DispatchError::Contract(OpError::UnsupportedDType)
        );
        for support in CPU_OPS {
            assert_ne!(support.dtype, DType::F16);
        }
    }

    #[test]
    fn bf16_is_dispatched_and_executed_by_the_cpu_reference() {
        let backend = CpuReferenceBackend::new();
        let device = backend.device().clone();
        let dispatcher = TensorDispatcher::new(vec![Box::new(backend)]);
        let mut lhs_storage = device.allocate(MemoryTier::Host, 4).unwrap();
        device.write(lhs_storage.as_mut(), 0, &[0; 4]).unwrap();
        let lhs = Tensor::new(lhs_storage, vec![2], DType::BF16).unwrap();
        let rhs = host_tensor(&device, vec![2], &[1.0, 2.0]);
        let mixed = ExecutionRequest::new(TensorOp::Add {
            lhs: &lhs,
            rhs: &rhs,
        });
        assert_eq!(
            dispatcher.plan(&mixed).unwrap_err(),
            DispatchError::Contract(OpError::MixedDType)
        );

        let rhs_bf16 = {
            let mut storage = device.allocate(MemoryTier::Host, 4).unwrap();
            device.write(storage.as_mut(), 0, &[0; 4]).unwrap();
            Tensor::new(storage, vec![2], DType::BF16).unwrap()
        };
        let request = ExecutionRequest::new(TensorOp::Add {
            lhs: &lhs,
            rhs: &rhs_bf16,
        });
        let plan = dispatcher.plan(&request).unwrap();
        assert_eq!(plan.backend, "cpu-reference");
        assert!(plan.reason.contains("BF16"), "{}", plan.reason);
        let output = dispatcher.execute(&request).unwrap();
        assert_eq!(output.layout().dtype(), DType::BF16);
        assert_eq!(output.layout().numel(), 2);
        assert_eq!(output.layout().span_bytes(), 4);
    }

    #[test]
    fn cpu_capabilities_cover_both_dtypes_for_all_four_operations() {
        let capabilities = CpuReferenceBackend::new().capabilities();
        assert_eq!(capabilities.supported_ops.len(), 8);
        for dtype in [DType::F32, DType::BF16] {
            for kind in OpKind::ALL {
                assert!(
                    capabilities
                        .support_for(
                            kind,
                            dtype,
                            MemoryTier::Host,
                            MemoryTier::Host,
                            LayoutRequirement::Contiguous,
                            AliasRule::ReadOnlyInputsFreshOutput,
                        )
                        .is_some(),
                    "missing support for {kind} / {dtype:?}"
                );
            }
        }
    }

    #[test]
    fn contract_rejects_non_contiguous_and_mixed_devices() {
        let first = CpuDevice::new();
        let second = CpuDevice::new();
        let storage = first.allocate(MemoryTier::Host, 40).unwrap();
        let strided = Tensor::with_layout(
            storage,
            crate::TensorLayout::strided(vec![2, 2], vec![3, 1], DType::F32).unwrap(),
            0,
        )
        .unwrap();
        let request = ExecutionRequest::new(TensorOp::SoftmaxLastDim { input: &strided });
        assert_eq!(
            dispatcher().0.plan(&request).unwrap_err(),
            DispatchError::Contract(OpError::NonContiguous)
        );

        let lhs = host_tensor(&first, vec![2], &[1.0, 2.0]);
        let rhs = host_tensor(&second, vec![2], &[3.0, 4.0]);
        let request = ExecutionRequest::new(TensorOp::Add {
            lhs: &lhs,
            rhs: &rhs,
        });
        assert_eq!(
            dispatcher().0.plan(&request).unwrap_err(),
            DispatchError::Contract(OpError::MixedDevices)
        );
    }
}
