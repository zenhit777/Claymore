//! DiffusionGemma model adapter boundary.
//!
//! This crate owns model-specific configuration, checkpoint provenance,
//! reference sampling policy and generation state. Tensor storage, device
//! execution, scheduling and memory admission remain owned by the corresponding
//! Claymore crates.

pub mod canvas;
pub mod checkpoint;
pub mod config;
pub mod executor;
pub mod generation;
pub mod rng;
pub mod safetensors;
pub mod sampling;
pub mod sha256;
mod strict_json;

pub use canvas::{CanvasError, CanvasMetrics, CanvasPosition, CanvasTokenState, GemmaCanvas};
pub use checkpoint::{
    expected_tensors, Checkpoint, CheckpointError, CheckpointIdentity, ShardIndex, TensorLocation,
};
pub use config::{
    GemmaConfigError, GemmaModelConfig, LayerAttention, TextConfig, VisionConfig, MODEL_ID,
};
pub use executor::{
    BlockId, BlockState, CommittedBlock, DiffusionGemmaBackend, GemmaDiffusionExecutor,
    GemmaExecutorError, IterationMetrics,
};
pub use generation::{
    DiffusionGenerationConfig, GenerationConfigError, GenerationPolicy, SamplerConfig,
};
pub use rng::SplitMix64;
pub use safetensors::{
    SafetensorsError, SafetensorsFile, TensorEntry, WeightDType, MAX_HEADER_BYTES,
};
pub use sampling::{
    apply_temperature, argmax_rows, mean_entropy, sample_rows, token_entropies, Acceptance,
    EntropyBoundSampler, SamplingError, StableAndConfident,
};
pub use sha256::{hex as sha256_hex, sha256, Sha256Stream};
