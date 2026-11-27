//! Shared, model-agnostic checkpoint metadata and Safetensors primitives.
//!
//! Model adapters own their pinned tensor inventories and architecture-specific
//! validation. This crate owns only strict JSON parsing, Safetensors container
//! validation and streaming SHA-256 primitives.

pub mod safetensors;
pub mod sha256;
pub mod strict_json;
