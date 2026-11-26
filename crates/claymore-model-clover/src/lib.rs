//! Clover-35B-A3B-Preview model adapter boundary.
//!
//! Model-specific graph semantics live here. Tensor storage, device execution,
//! scheduling, and memory admission remain owned by the shared Claymore crates.

pub mod checkpoint;
pub mod config;
pub use checkpoint::{
    CloverCheckpointError, CloverCheckpointInventory, CloverInventoryIdentity, CloverShardIndex,
    ShardSummary, TensorSummary,
};
pub use config::{
    CloverConfig, CloverConfigError, LayerType, CONFIG_REVISION, CONFIG_SHA256, INDEX_SHA256,
    MODEL_ID,
};

/// Lifecycle marker retained until the model graph is wired to the shared
/// diffusion executor. This is not an inference engine state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloverEngineState {
    Uninitialized,
    Ready,
}
