//! DiffusionGemma model adapter boundary.
//!
//! This crate owns model-specific configuration and generation state. Tensor
//! storage, device execution, scheduling and memory admission remain owned by
//! the corresponding Claymore crates.
pub mod canvas;
pub mod config;

pub use canvas::{CanvasError, CanvasMetrics, CanvasPosition, CanvasTokenState, GemmaCanvas};
pub use config::{GemmaConfigError, GemmaModelConfig, MODEL_ID};
