//! Shared types and configuration primitives for Claymore.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    Bf16,
    Fp16,
    Fp32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTier {
    Vram,
    PinnedHost,
    Host,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub soft_vram_bytes: u64,
    pub hard_vram_bytes: u64,
}

impl MemoryBudget {
    pub fn validate(&self) -> bool {
        self.soft_vram_bytes <= self.hard_vram_bytes
    }
}
