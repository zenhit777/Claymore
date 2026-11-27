//! Clover-35B-A3B-Preview model adapter boundary.
//!
//! Model-specific graph semantics live here. Tensor storage, device execution,
//! scheduling, and memory admission remain owned by the shared Claymore crates.

pub mod checkpoint;
pub mod config;
pub mod executor;
pub mod generation;
pub mod graph;
pub mod moe;
pub mod residency;
pub mod weights;
pub use checkpoint::{
    CloverCheckpointError, CloverCheckpointInventory, CloverInventoryIdentity, CloverLoadPlan,
    CloverShardIndex, ShardLoadPlan, ShardSummary, TensorChunkPlan, TensorLoadPlan, TensorSummary,
};
pub use config::{
    CloverConfig, CloverConfigError, LayerType, CONFIG_REVISION, CONFIG_SHA256, INDEX_SHA256,
    MODEL_ID,
};
pub use executor::{CloverDiffusionExecutor, CloverExecutorError, CloverGenerationOutput};
pub use generation::{
    CloverGenerationError, CloverGenerationOptions, CloverGenerationPlan, CloverInvalidTokenPolicy,
    CloverPenaltyConfig,
};
pub use graph::{
    CloverActivation, CloverForwardMode, CloverGraph, CloverGraphBackend, CloverGraphError,
    CloverGraphOutput, CloverNormKind,
};
pub use moe::{
    execute_moe_cpu, project_router_logits_cpu, CloverExpertWeights, CloverMoeError,
    CloverMoeRouter, CloverRoutingPlan, CloverSharedExpertWeights, TokenRoute,
};
pub use weights::{CloverCpuWeightError, CloverCpuWeightStore};
pub use residency::{
    prefetch_requests_for_routing, CloverExpertCache, CloverExpertCacheError,
    CloverExpertCacheMetrics, CloverExpertGeometry, CloverExpertLoadRequest,
    CloverExpertLoadSchedule, CloverExpertLoadTask, CloverExpertPrefetchRequest,
    CloverExpertScheduleError, CloverSharedExpertGeometry, CloverSharedExpertLoadRequest,
};
