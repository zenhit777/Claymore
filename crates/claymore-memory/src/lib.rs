//! Deterministic memory orchestration policies for Claymore.
//!
//! This crate plans placements and safe evictions. It deliberately does not
//! allocate CUDA memory or initiate transfers; those operations belong to a
//! backend that must honor the plans and report completed CUDA events.

use std::collections::HashMap;

use claymore_core::MemoryTier;
use thiserror::Error;

mod lifecycle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudgets {
    pub vram_soft_bytes: u64,
    pub vram_hard_bytes: u64,
    pub host_bytes: u64,
    pub pinned_host_bytes: u64,
    pub expert_cache_target_bytes: u64,
    pub expert_cache_max_bytes: u64,
}

impl MemoryBudgets {
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.vram_soft_bytes > self.vram_hard_bytes {
            return Err(MemoryError::InvalidBudget(
                "soft VRAM budget exceeds hard limit",
            ));
        }
        if self.expert_cache_target_bytes > self.expert_cache_max_bytes {
            return Err(MemoryError::InvalidBudget(
                "expert cache target exceeds maximum",
            ));
        }
        if self.expert_cache_max_bytes > self.vram_hard_bytes {
            return Err(MemoryError::InvalidBudget(
                "expert cache maximum exceeds VRAM hard limit",
            ));
        }
        if self.pinned_host_bytes > self.host_bytes {
            return Err(MemoryError::InvalidBudget(
                "pinned host budget exceeds host budget",
            ));
        }
        Ok(())
    }

    fn limit_for(&self, tier: MemoryTier) -> u64 {
        match tier {
            MemoryTier::Vram => self.vram_hard_bytes,
            MemoryTier::PinnedHost => self.pinned_host_bytes,
            MemoryTier::Host => self.host_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub host_total_bytes: u64,
    pub host_available_bytes: u64,
}

impl MemorySnapshot {
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.vram_free_bytes > self.vram_total_bytes {
            return Err(MemoryError::InvalidSnapshot("free VRAM exceeds total VRAM"));
        }
        if self.host_available_bytes > self.host_total_bytes {
            return Err(MemoryError::InvalidSnapshot(
                "available host RAM exceeds total RAM",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    pub id: String,
    pub tier: MemoryTier,
    pub bytes: u64,
    pub last_used_tick: u64,
    pub hit_count: u64,
    /// Earliest logical tick at which the allocation is known to be idle.
    pub safe_after_tick: u64,
    pub expert: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionPlan {
    pub tier: MemoryTier,
    pub requested_bytes: u64,
    pub evict_ids: Vec<String>,
    pub projected_usage_bytes: u64,
    pub limit_bytes: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MemoryError {
    #[error("invalid memory budget: {0}")]
    InvalidBudget(&'static str),
    #[error("invalid memory snapshot: {0}")]
    InvalidSnapshot(&'static str),
    #[error("allocation already exists: {0}")]
    DuplicateAllocation(String),
    #[error("allocation not found: {0}")]
    AllocationNotFound(String),
    #[error("allocation is still in flight: {0}")]
    AllocationBusy(String),
    #[error("request of {requested} bytes exceeds {tier:?} budget ({limit} bytes)")]
    RequestTooLarge {
        tier: MemoryTier,
        requested: u64,
        limit: u64,
    },
    #[error("not enough reclaimable memory in {tier:?}: need {needed} bytes, can reclaim {reclaimable} bytes")]
    InsufficientReclaimable {
        tier: MemoryTier,
        needed: u64,
        reclaimable: u64,
    },
}

#[derive(Debug)]
pub struct MemoryOrchestrator {
    budgets: MemoryBudgets,
    policy: EvictionPolicy,
    allocations: HashMap<String, Allocation>,
    reload_costs_ns: HashMap<String, u64>,
}

impl MemoryOrchestrator {
    pub fn new(budgets: MemoryBudgets, policy: EvictionPolicy) -> Result<Self, MemoryError> {
        budgets.validate()?;
        Ok(Self {
            budgets,
            policy,
            allocations: HashMap::new(),
            reload_costs_ns: HashMap::new(),
        })
    }

    pub fn budgets(&self) -> MemoryBudgets {
        self.budgets
    }

    pub fn usage_bytes(&self, tier: MemoryTier) -> u64 {
        self.allocations
            .values()
            .filter(|a| a.tier == tier)
            .fold(0_u64, |sum, a| sum.saturating_add(a.bytes))
    }

    pub fn allocation(&self, id: &str) -> Option<&Allocation> {
        self.allocations.get(id)
    }

    /// Register an allocation after the backend has successfully allocated it.
    pub fn record_allocation(&mut self, allocation: Allocation) -> Result<(), MemoryError> {
        if self.allocations.contains_key(&allocation.id) {
            return Err(MemoryError::DuplicateAllocation(allocation.id));
        }
        let limit = self.budgets.limit_for(allocation.tier);
        let projected = self
            .usage_bytes(allocation.tier)
            .saturating_add(allocation.bytes);
        if allocation.expert && allocation.tier == MemoryTier::Vram {
            let expert_projected = self.expert_cache_usage_bytes().saturating_add(allocation.bytes);
            if expert_projected > self.budgets.expert_cache_max_bytes {
                return Err(MemoryError::InsufficientReclaimable {
                    tier: MemoryTier::Vram,
                    needed: expert_projected - self.budgets.expert_cache_max_bytes,
                    reclaimable: 0,
                });
            }
        }
        if projected > limit {
            return Err(MemoryError::InsufficientReclaimable {
                tier: allocation.tier,
                needed: projected - limit,
                reclaimable: 0,
            });
        }
        self.allocations.insert(allocation.id.clone(), allocation);
        Ok(())
    }

    /// Record the observed or estimated cost of reloading an allocation.
    /// Adaptive eviction retains allocations with higher reload costs.
    pub fn record_reload_cost(&mut self, id: &str, cost_ns: u64) -> Result<(), MemoryError> {
        if !self.allocations.contains_key(id) {
            return Err(MemoryError::AllocationNotFound(id.to_owned()));
        }
        self.reload_costs_ns.insert(id.to_owned(), cost_ns);
        Ok(())
    }

    pub fn reload_cost_ns(&self, id: &str) -> Option<u64> {
        self.reload_costs_ns.get(id).copied()
    }

    pub fn touch(&mut self, id: &str, tick: u64) -> Result<(), MemoryError> {
        let allocation = self
            .allocations
            .get_mut(id)
            .ok_or_else(|| MemoryError::AllocationNotFound(id.to_owned()))?;
        allocation.last_used_tick = tick;
        allocation.hit_count = allocation.hit_count.saturating_add(1);
        Ok(())
    }

    /// Update the fence/event completion horizon reported by the backend.
    pub fn mark_in_flight_until(
        &mut self,
        id: &str,
        safe_after_tick: u64,
    ) -> Result<(), MemoryError> {
        let allocation = self
            .allocations
            .get_mut(id)
            .ok_or_else(|| MemoryError::AllocationNotFound(id.to_owned()))?;
        allocation.safe_after_tick = safe_after_tick;
        Ok(())
    }

    /// Build a deterministic eviction plan without mutating tracked allocations.
    /// The backend must wait for relevant device events, free the selected
    /// allocations, then call confirm_eviction for each successfully freed id.
    pub fn plan_admission(
        &self,
        tier: MemoryTier,
        requested_bytes: u64,
        now_tick: u64,
    ) -> Result<AdmissionPlan, MemoryError> {
        let hard_limit = self.budgets.limit_for(tier);
        let soft_limit = if tier == MemoryTier::Vram {
            self.budgets.vram_soft_bytes
        } else {
            hard_limit
        };
        if requested_bytes > hard_limit {
            return Err(MemoryError::RequestTooLarge {
                tier,
                requested: requested_bytes,
                limit: hard_limit,
            });
        }
        let current = self.usage_bytes(tier);
        let projected = current.saturating_add(requested_bytes);
        if projected <= soft_limit {
            return Ok(AdmissionPlan {
                tier,
                requested_bytes,
                evict_ids: Vec::new(),
                projected_usage_bytes: projected,
                limit_bytes: hard_limit,
            });
        }

        // VRAM soft budget is an eviction target; the hard budget remains the
        // admission boundary. If the target cannot be reached, reclaim every
        // safe candidate and admit only if the hard ceiling is still respected.
        let needed_for_soft = projected - soft_limit;
        let mut candidates: Vec<&Allocation> = self.allocations.values()
            .filter(|a| a.tier == tier && a.safe_after_tick <= now_tick)
            .collect();
        candidates.sort_by(|a, b| {
            self.eviction_key(a, now_tick).cmp(&self.eviction_key(b, now_tick))
                // HashMap iteration order is intentionally unspecified. An ID
                // tie-break makes equal-priority eviction plans reproducible.
                .then_with(|| a.id.cmp(&b.id))
        });
        let mut selected = Vec::new();
        let mut reclaimed = 0_u64;
        for candidate in candidates {
            selected.push(candidate.id.clone());
            reclaimed = reclaimed.saturating_add(candidate.bytes);
            if reclaimed >= needed_for_soft {
                break;
            }
        }

        let projected_after_eviction = current.saturating_sub(reclaimed).saturating_add(requested_bytes);
        if projected_after_eviction > hard_limit {
            return Err(MemoryError::InsufficientReclaimable {
                tier,
                needed: projected_after_eviction - hard_limit,
                reclaimable: reclaimed,
            });
        }
        Ok(AdmissionPlan {
            tier,
            requested_bytes,
            evict_ids: selected,
            projected_usage_bytes: projected_after_eviction,
            limit_bytes: hard_limit,
        })
    }

    /// Plan expert-cache admission independently from the general VRAM budget.
    /// Only idle expert allocations are candidates; the plan never mutates state.
    pub fn plan_expert_admission(
        &self,
        requested_bytes: u64,
        now_tick: u64,
    ) -> Result<AdmissionPlan, MemoryError> {
        let limit = self.budgets.expert_cache_max_bytes;
        if requested_bytes > limit {
            return Err(MemoryError::RequestTooLarge {
                tier: MemoryTier::Vram,
                requested: requested_bytes,
                limit,
            });
        }
        let current = self.expert_cache_usage_bytes();
        let projected = current.saturating_add(requested_bytes);
        if projected <= limit {
            return Ok(AdmissionPlan {
                tier: MemoryTier::Vram,
                requested_bytes,
                evict_ids: Vec::new(),
                projected_usage_bytes: projected,
                limit_bytes: limit,
            });
        }
        let needed = projected - limit;
        let mut candidates: Vec<&Allocation> = self.allocations.values()
            .filter(|a| a.expert && a.tier == MemoryTier::Vram && a.safe_after_tick <= now_tick)
            .collect();
        candidates.sort_by(|a, b| {
            self.eviction_key(a, now_tick).cmp(&self.eviction_key(b, now_tick))
                .then_with(|| a.id.cmp(&b.id))
        });
        let mut selected = Vec::new();
        let mut reclaimed = 0_u64;
        for candidate in candidates {
            selected.push(candidate.id.clone());
            reclaimed = reclaimed.saturating_add(candidate.bytes);
            if reclaimed >= needed { break; }
        }
        if reclaimed < needed {
            return Err(MemoryError::InsufficientReclaimable {
                tier: MemoryTier::Vram,
                needed,
                reclaimable: reclaimed,
            });
        }
        Ok(AdmissionPlan {
            tier: MemoryTier::Vram,
            requested_bytes,
            evict_ids: selected,
            projected_usage_bytes: current.saturating_sub(reclaimed).saturating_add(requested_bytes),
            limit_bytes: limit,
        })
    }

    /// Confirm that the backend has actually freed an allocation.
    pub fn confirm_eviction(&mut self, id: &str, now_tick: u64) -> Result<Allocation, MemoryError> {
        let allocation = self
            .allocations
            .get(id)
            .ok_or_else(|| MemoryError::AllocationNotFound(id.to_owned()))?;
        if allocation.safe_after_tick > now_tick {
            return Err(MemoryError::AllocationBusy(id.to_owned()));
        }
        self.reload_costs_ns.remove(id);
        Ok(self
            .allocations
            .remove(id)
            .expect("allocation was checked above"))
    }

    fn eviction_key(&self, allocation: &Allocation, now_tick: u64) -> (u64, u64, u64) {
        let age = now_tick.saturating_sub(allocation.last_used_tick);
        match self.policy {
            EvictionPolicy::Lru => (0, u64::MAX - age, allocation.hit_count),
            EvictionPolicy::Lfu => (0, allocation.hit_count, u64::MAX - age),
            EvictionPolicy::Adaptive => {
                // Prefer cold, old, cheap-to-reload allocations.
                let recency_penalty = allocation.last_used_tick;
                let reload_cost = self.reload_costs_ns.get(&allocation.id).copied().unwrap_or(0);
                (
                    0,
                    allocation.hit_count
                        .saturating_mul(2)
                        .saturating_add(recency_penalty / 8)
                        .saturating_add(reload_cost),
                    allocation.bytes,
                )
            }
        }
    }

    pub fn expert_cache_usage_bytes(&self) -> u64 {
        self.allocations
            .values()
            .filter(|a| a.expert && a.tier == MemoryTier::Vram)
            .fold(0_u64, |sum, a| sum.saturating_add(a.bytes))
    }

    pub fn expert_cache_target_bytes(&self) -> u64 {
        self.budgets.expert_cache_target_bytes
    }

    pub fn expert_cache_max_bytes(&self) -> u64 {
        self.budgets.expert_cache_max_bytes
    }
}

mod device;
pub use device::{DeviceMemoryError, DeviceMemoryManager};

mod executor;
pub use executor::{MemoryExecutionMetrics, MemoryTask, MemoryTaskExecutor, MemoryTaskExecutorError};

#[cfg(feature = "cuda-driver")]
mod cuda;
#[cfg(feature = "cuda-driver")]
pub use cuda::{CudaMemoryError, CudaMemoryManager, TransferToken};

#[cfg(test)]
mod tests {
    use super::*;

    fn budgets() -> MemoryBudgets {
        MemoryBudgets {
            vram_soft_bytes: 80,
            vram_hard_bytes: 100,
            host_bytes: 1_000,
            pinned_host_bytes: 200,
            expert_cache_target_bytes: 60,
            expert_cache_max_bytes: 90,
        }
    }

    fn allocation(id: &str, bytes: u64, last_used_tick: u64, safe_after_tick: u64) -> Allocation {
        Allocation {
            id: id.into(),
            tier: MemoryTier::Vram,
            bytes,
            last_used_tick,
            hit_count: 1,
            safe_after_tick,
            expert: true,
        }
    }

    #[test]
    fn rejects_invalid_budget_relationships() {
        let mut invalid = budgets();
        invalid.vram_soft_bytes = 101;
        assert!(matches!(
            invalid.validate(),
            Err(MemoryError::InvalidBudget(_))
        ));
    }

    #[test]
    fn admission_succeeds_without_eviction_when_under_limit() {
        let manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        let plan = manager.plan_admission(MemoryTier::Vram, 40, 0).unwrap();
        assert!(plan.evict_ids.is_empty());
        assert_eq!(plan.projected_usage_bytes, 40);
    }

    #[test]
    fn vram_soft_budget_triggers_proactive_eviction_before_hard_ceiling() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        manager.record_allocation(allocation("old", 50, 1, 0)).unwrap();
        manager.record_allocation(allocation("new", 35, 9, 0)).unwrap();

        let plan = manager.plan_admission(MemoryTier::Vram, 10, 10).unwrap();
        assert_eq!(plan.evict_ids, vec!["old"]);
        assert_eq!(plan.projected_usage_bytes, 45);
        assert_eq!(plan.limit_bytes, 100);
    }

    #[test]
    fn vram_soft_target_is_best_effort_but_hard_ceiling_is_mandatory() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        manager.record_allocation(allocation("busy", 90, 1, 50)).unwrap();

        let plan = manager.plan_admission(MemoryTier::Vram, 5, 10).unwrap();
        assert!(plan.evict_ids.is_empty());
        assert_eq!(plan.projected_usage_bytes, 95);
        assert_eq!(plan.limit_bytes, 100);
        assert!(matches!(
            manager.plan_admission(MemoryTier::Vram, 15, 10),
            Err(MemoryError::InsufficientReclaimable { .. })
        ));
    }

    #[test]
    fn lru_plan_selects_oldest_idle_allocation() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .record_allocation(allocation("old", 60, 1, 0))
            .unwrap();
        manager
            .record_allocation(allocation("new", 30, 9, 0))
            .unwrap();
        let plan = manager.plan_admission(MemoryTier::Vram, 30, 10).unwrap();
        assert_eq!(plan.evict_ids, vec!["old"]);
        assert_eq!(
            manager.usage_bytes(MemoryTier::Vram),
            90,
            "planning must not mutate state"
        );
    }

    #[test]
    fn equal_priority_evictions_use_lexicographic_allocation_id() {
        for policy in [
            EvictionPolicy::Lru,
            EvictionPolicy::Lfu,
            EvictionPolicy::Adaptive,
        ] {
            let mut manager = MemoryOrchestrator::new(budgets(), policy).unwrap();
            for id in ["gamma", "alpha", "beta"] {
                manager.record_allocation(allocation(id, 30, 5, 0)).unwrap();
            }

            // A 50-byte admission needs two 30-byte victims. Equal policy
            // scores must therefore use the stable ID tie-break for both.
            for _ in 0..32 {
                let plan = manager.plan_admission(MemoryTier::Vram, 50, 5).unwrap();
                assert_eq!(plan.evict_ids, vec!["alpha", "beta"], "policy: {policy:?}");
            }
        }
    }

    #[test]
    fn adaptive_eviction_prefers_the_lower_reload_cost() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Adaptive).unwrap();
        manager.record_allocation(allocation("expensive", 40, 5, 0)).unwrap();
        manager.record_allocation(allocation("cheap", 40, 5, 0)).unwrap();
        manager.record_reload_cost("expensive", 10_000).unwrap();
        manager.record_reload_cost("cheap", 10).unwrap();

        let plan = manager.plan_admission(MemoryTier::Vram, 20, 5).unwrap();
        assert_eq!(plan.evict_ids, vec!["cheap"]);
        assert_eq!(manager.reload_cost_ns("expensive"), Some(10_000));
    }

    #[test]
    fn in_flight_allocations_are_never_selected() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .record_allocation(allocation("busy", 70, 1, 50))
            .unwrap();
        assert!(matches!(
            manager.plan_admission(MemoryTier::Vram, 40, 10),
            Err(MemoryError::InsufficientReclaimable { .. })
        ));
        assert!(matches!(
            manager.confirm_eviction("busy", 10),
            Err(MemoryError::AllocationBusy(_))
        ));
    }

    #[test]
    fn confirm_eviction_releases_accounted_capacity() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .record_allocation(allocation("expert-1", 70, 1, 0))
            .unwrap();
        let removed = manager.confirm_eviction("expert-1", 1).unwrap();
        assert_eq!(removed.bytes, 70);
        assert_eq!(manager.usage_bytes(MemoryTier::Vram), 0);
    }

    #[test]
    fn expert_cache_admission_enforces_max_and_selects_only_idle_experts() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        manager.record_allocation(allocation("expert-old", 50, 1, 0)).unwrap();
        manager.record_allocation(allocation("expert-busy", 30, 2, 50)).unwrap();
        let mut ordinary = allocation("ordinary", 10, 0, 0);
        ordinary.expert = false;
        manager.record_allocation(ordinary).unwrap();

        let plan = manager.plan_expert_admission(50, 10).unwrap();
        assert_eq!(plan.evict_ids, vec!["expert-old"]);
        assert_eq!(plan.limit_bytes, 90);
        assert_eq!(manager.expert_cache_usage_bytes(), 80);
        assert!(matches!(
            manager.plan_expert_admission(91, 10),
            Err(MemoryError::RequestTooLarge { limit: 90, .. })
        ));
        assert!(matches!(
            manager.plan_expert_admission(70, 10),
            Err(MemoryError::InsufficientReclaimable { .. })
        ));
    }

    #[test]
    fn detects_duplicate_allocations() {
        let mut manager = MemoryOrchestrator::new(budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .record_allocation(allocation("same", 10, 0, 0))
            .unwrap();
        assert!(matches!(
            manager.record_allocation(allocation("same", 10, 0, 0)),
            Err(MemoryError::DuplicateAllocation(_))
        ));
    }
}

#[cfg(feature = "cuda-driver")]
mod cuda_executor;
#[cfg(feature = "cuda-driver")]
pub use cuda_executor::{CudaMemoryTask, CudaTransferExecutor, CudaTransferExecutorError, CudaTransferMetrics};
