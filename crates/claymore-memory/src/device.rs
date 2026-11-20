//! Backend-neutral memory adapter for CPU contract tests and future devices.
use std::collections::HashMap;

use claymore_core::MemoryTier;
use claymore_device::{Device, DeviceBuffer, DeviceError};
use thiserror::Error;

use crate::{Allocation, EvictionPolicy, MemoryBudgets, MemoryError, MemoryOrchestrator};

/// Owns backend allocations while delegating admission and eviction policy to
/// the existing memory orchestrator.
pub struct DeviceMemoryManager<D: Device> {
    device: D,
    policy: MemoryOrchestrator,
    buffers: HashMap<String, Box<dyn DeviceBuffer>>,
    last_tick: Option<u64>,
}

#[derive(Debug, Error)]
pub enum DeviceMemoryError {
    #[error(transparent)]
    Policy(#[from] MemoryError),
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("allocation {0} does not exist")]
    UnknownAllocation(String),
    #[error("allocation {0} already exists")]
    DuplicateAllocation(String),
    #[error("logical tick moved backwards: previous {previous}, provided {provided}")]
    NonMonotonicTick { previous: u64, provided: u64 },
    #[error("expert allocations must reside in VRAM")]
    ExpertMustUseVram,
    #[error("source and destination allocations must be distinct")]
    AliasedCopy,
}

impl<D: Device> DeviceMemoryManager<D> {
    pub fn new(device: D, budgets: MemoryBudgets, eviction: EvictionPolicy) -> Result<Self, DeviceMemoryError> {
        Ok(Self {
            device,
            policy: MemoryOrchestrator::new(budgets, eviction)?,
            buffers: HashMap::new(),
            last_tick: None,
        })
    }

    pub fn device(&self) -> &D { &self.device }
    pub fn usage_bytes(&self, tier: MemoryTier) -> u64 { self.policy.usage_bytes(tier) }
    pub fn allocation(&self, id: &str) -> Option<&Allocation> { self.policy.allocation(id) }

    pub fn allocate(&mut self, id: impl Into<String>, tier: MemoryTier, bytes: u64, tick: u64, expert: bool) -> Result<(), DeviceMemoryError> {
        self.advance_tick(tick)?;
        let id = id.into();
        if self.policy.allocation(&id).is_some() || self.buffers.contains_key(&id) {
            return Err(DeviceMemoryError::DuplicateAllocation(id));
        }
        if expert && tier != MemoryTier::Vram { return Err(DeviceMemoryError::ExpertMustUseVram); }

        let plan = self.policy.plan_admission(tier, bytes, tick)?;
        for evict_id in plan.evict_ids { self.release_inner(&evict_id, tick)?; }

        let buffer = self.device.allocate(tier, bytes)?;
        let allocation = Allocation {
            id: id.clone(), tier, bytes, last_used_tick: tick, hit_count: 0,
            safe_after_tick: tick, expert,
        };
        if let Err(error) = self.policy.record_allocation(allocation) {
            drop(buffer);
            return Err(error.into());
        }
        self.buffers.insert(id, buffer);
        Ok(())
    }

    pub fn write(&mut self, id: &str, offset: usize, data: &[u8], tick: u64) -> Result<(), DeviceMemoryError> {
        self.advance_tick(tick)?;
        let buffer = self.buffers.get_mut(id).ok_or_else(|| DeviceMemoryError::UnknownAllocation(id.to_owned()))?;
        self.device.write(buffer.as_mut(), offset, data)?;
        self.policy.touch(id, tick)?;
        Ok(())
    }

    pub fn read(&mut self, id: &str, offset: usize, len: usize, tick: u64) -> Result<Vec<u8>, DeviceMemoryError> {
        self.advance_tick(tick)?;
        let buffer = self.buffers.get(id).ok_or_else(|| DeviceMemoryError::UnknownAllocation(id.to_owned()))?;
        let bytes = self.device.read(buffer.as_ref(), offset, len)?;
        self.policy.touch(id, tick)?;
        Ok(bytes)
    }

    /// Synchronous copy: the destination is ready before this method returns.
    pub fn copy(&mut self, source_id: &str, destination_id: &str, bytes: usize, tick: u64) -> Result<(), DeviceMemoryError> {
        self.advance_tick(tick)?;
        if source_id == destination_id { return Err(DeviceMemoryError::AliasedCopy); }
        let source = self.buffers.get(source_id).ok_or_else(|| DeviceMemoryError::UnknownAllocation(source_id.to_owned()))?;
        let payload = self.device.read(source.as_ref(), 0, bytes)?;
        let destination = self.buffers.get_mut(destination_id).ok_or_else(|| DeviceMemoryError::UnknownAllocation(destination_id.to_owned()))?;
        self.device.write(destination.as_mut(), 0, &payload)?;
        self.policy.touch(source_id, tick)?;
        self.policy.touch(destination_id, tick)?;
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), DeviceMemoryError> {
        self.device.synchronize()?;
        Ok(())
    }

    pub fn release(&mut self, id: &str, tick: u64) -> Result<Allocation, DeviceMemoryError> {
        self.advance_tick(tick)?;
        self.release_inner(id, tick)
    }

    fn release_inner(&mut self, id: &str, tick: u64) -> Result<Allocation, DeviceMemoryError> {
        if !self.buffers.contains_key(id) { return Err(DeviceMemoryError::UnknownAllocation(id.to_owned())); }
        let allocation = self.policy.allocation(id).ok_or_else(|| DeviceMemoryError::UnknownAllocation(id.to_owned()))?;
        if allocation.safe_after_tick > tick { return Err(MemoryError::AllocationBusy(id.to_owned()).into()); }
        let allocation = self.policy.confirm_eviction(id, tick)?;
        self.buffers.remove(id);
        Ok(allocation)
    }

    fn advance_tick(&mut self, tick: u64) -> Result<(), DeviceMemoryError> {
        if let Some(previous) = self.last_tick {
            if tick < previous { return Err(DeviceMemoryError::NonMonotonicTick { previous, provided: tick }); }
        }
        self.last_tick = Some(tick);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_device::CpuDevice;

    fn budgets() -> MemoryBudgets {
        MemoryBudgets { vram_soft_bytes: 64, vram_hard_bytes: 128, host_bytes: 256, pinned_host_bytes: 32, expert_cache_target_bytes: 32, expert_cache_max_bytes: 64 }
    }

    #[test]
    fn cpu_adapter_allocates_copies_and_releases_buffers() {
        let mut manager = DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager.allocate("source", MemoryTier::Host, 8, 1, false).unwrap();
        manager.allocate("destination", MemoryTier::Vram, 8, 2, false).unwrap();
        manager.write("source", 0, &[9, 8, 7, 6], 3).unwrap();
        manager.copy("source", "destination", 4, 4).unwrap();
        assert_eq!(manager.read("destination", 0, 4, 5).unwrap(), [9, 8, 7, 6]);
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 8);
        manager.release("source", 6).unwrap();
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 0);
        manager.synchronize().unwrap();
    }

    #[test]
    fn cpu_adapter_rejects_duplicate_ids_and_non_monotonic_ticks() {
        let mut manager = DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager.allocate("buffer", MemoryTier::Host, 8, 4, false).unwrap();
        assert!(matches!(manager.allocate("buffer", MemoryTier::Host, 8, 5, false), Err(DeviceMemoryError::DuplicateAllocation(_))));
        assert!(matches!(manager.read("buffer", 0, 1, 3), Err(DeviceMemoryError::NonMonotonicTick { .. })));
    }

    #[test]
    fn cpu_adapter_uses_memory_policy_for_deterministic_eviction() {
        let mut manager = DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager.allocate("old", MemoryTier::Host, 160, 1, false).unwrap();
        manager.allocate("new", MemoryTier::Host, 80, 2, false).unwrap();
        manager.allocate("replacement", MemoryTier::Host, 200, 3, false).unwrap();
        assert!(manager.allocation("old").is_none());
        assert!(manager.allocation("replacement").is_some());
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 200);
    }
}
