//! Memory placement and expert-cache policy layer.
use claymore_core::MemoryTier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub tier: MemoryTier,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    Adaptive,
}

/// Conservative admission check. This is a policy primitive, not a CUDA allocator.
pub fn can_admit(used_bytes: u64, requested_bytes: u64, hard_limit_bytes: u64) -> bool {
    used_bytes.checked_add(requested_bytes).is_some_and(|n| n <= hard_limit_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_over_budget_allocation() {
        assert!(!can_admit(90, 20, 100));
        assert!(can_admit(80, 20, 100));
    }
}
