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
    #[error("completion horizon {safe_after_tick} precedes current tick {current_tick}")]
    InvalidCompletionHorizon { current_tick: u64, safe_after_tick: u64 },
    #[error("completion horizon regressed from {previous} to {provided}")]
    CompletionHorizonRegression { previous: u64, provided: u64 },
}

impl<D: Device> DeviceMemoryManager<D> {
    pub fn new(
        device: D,
        budgets: MemoryBudgets,
        eviction: EvictionPolicy,
    ) -> Result<Self, DeviceMemoryError> {
        Ok(Self {
            device,
            policy: MemoryOrchestrator::new(budgets, eviction)?,
            buffers: HashMap::new(),
            last_tick: None,
        })
    }

    pub fn device(&self) -> &D {
        &self.device
    }
    pub fn usage_bytes(&self, tier: MemoryTier) -> u64 {
        self.policy.usage_bytes(tier)
    }
    pub fn allocation(&self, id: &str) -> Option<&Allocation> {
        self.policy.allocation(id)
    }

    pub fn allocate(
        &mut self,
        id: impl Into<String>,
        tier: MemoryTier,
        bytes: u64,
        tick: u64,
        expert: bool,
    ) -> Result<(), DeviceMemoryError> {
        self.advance_tick(tick)?;
        let id = id.into();
        if self.policy.allocation(&id).is_some() || self.buffers.contains_key(&id) {
            return Err(DeviceMemoryError::DuplicateAllocation(id));
        }
        if expert && tier != MemoryTier::Vram {
            return Err(DeviceMemoryError::ExpertMustUseVram);
        }

        if expert {
            let expert_plan = self.policy.plan_expert_admission(bytes, tick)?;
            for evict_id in expert_plan.evict_ids {
                self.release_inner(&evict_id, tick)?;
            }
        }
        let plan = self.policy.plan_admission(tier, bytes, tick)?;
        for evict_id in plan.evict_ids {
            self.release_inner(&evict_id, tick)?;
        }

        let buffer = self.device.allocate(tier, bytes)?;
        let allocation = Allocation {
            id: id.clone(),
            tier,
            bytes,
            last_used_tick: tick,
            hit_count: 0,
            safe_after_tick: tick,
            expert,
        };
        if let Err(error) = self.policy.record_allocation(allocation) {
            drop(buffer);
            return Err(error.into());
        }
        self.buffers.insert(id, buffer);
        Ok(())
    }

    pub fn write(
        &mut self,
        id: &str,
        offset: usize,
        data: &[u8],
        tick: u64,
    ) -> Result<(), DeviceMemoryError> {
        self.advance_tick(tick)?;
        let buffer = self
            .buffers
            .get_mut(id)
            .ok_or_else(|| DeviceMemoryError::UnknownAllocation(id.to_owned()))?;
        self.device.write(buffer.as_mut(), offset, data)?;
        self.policy.touch(id, tick)?;
        Ok(())
    }

    pub fn read(
        &mut self,
        id: &str,
        offset: usize,
        len: usize,
        tick: u64,
    ) -> Result<Vec<u8>, DeviceMemoryError> {
        self.advance_tick(tick)?;
        let buffer = self
            .buffers
            .get(id)
            .ok_or_else(|| DeviceMemoryError::UnknownAllocation(id.to_owned()))?;
        let bytes = self.device.read(buffer.as_ref(), offset, len)?;
        self.policy.touch(id, tick)?;
        Ok(bytes)
    }

    /// Synchronous copy: the destination is ready before this method returns.
    pub fn copy(
        &mut self,
        source_id: &str,
        destination_id: &str,
        bytes: usize,
        tick: u64,
    ) -> Result<(), DeviceMemoryError> {
        self.advance_tick(tick)?;
        if source_id == destination_id {
            return Err(DeviceMemoryError::AliasedCopy);
        }
        let [source, destination] = self.buffers.get_disjoint_mut([source_id, destination_id]);
        let source =
            source.ok_or_else(|| DeviceMemoryError::UnknownAllocation(source_id.to_owned()))?;
        let destination = destination
            .ok_or_else(|| DeviceMemoryError::UnknownAllocation(destination_id.to_owned()))?;
        self.device
            .copy(source.as_ref(), destination.as_mut(), bytes)?;
        self.policy.touch(source_id, tick)?;
        self.policy.touch(destination_id, tick)?;
        Ok(())
    }

    /// Mark an allocation as protected until the backend-reported logical tick.
    /// This is the CPU/backend-neutral guard used by schedulers and executors;
    /// callers must not advance the horizon until their operation is complete.
    pub fn mark_in_flight_until(
        &mut self,
        id: &str,
        safe_after_tick: u64,
        tick: u64,
    ) -> Result<(), DeviceMemoryError> {
        self.advance_tick(tick)?;
        if safe_after_tick < tick {
            return Err(DeviceMemoryError::InvalidCompletionHorizon {
                current_tick: tick,
                safe_after_tick,
            });
        }
        let previous = self.policy.allocation(id)
            .ok_or_else(|| DeviceMemoryError::UnknownAllocation(id.to_owned()))?
            .safe_after_tick;
        if safe_after_tick < previous {
            return Err(DeviceMemoryError::CompletionHorizonRegression {
                previous,
                provided: safe_after_tick,
            });
        }
        self.policy.mark_in_flight_until(id, safe_after_tick)?;
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
        if !self.buffers.contains_key(id) {
            return Err(DeviceMemoryError::UnknownAllocation(id.to_owned()));
        }
        let allocation = self
            .policy
            .allocation(id)
            .ok_or_else(|| DeviceMemoryError::UnknownAllocation(id.to_owned()))?;
        if !crate::lifecycle::release_is_safe(allocation.safe_after_tick, tick) {
            return Err(MemoryError::AllocationBusy(id.to_owned()).into());
        }
        // The backend must confirm release before the allocation is removed
        // from ownership or logical accounting. On failure, both remain intact
        // so the caller can diagnose or retry without inventing free capacity.
        self.buffers
            .get_mut(id)
            .expect("buffer existence was checked above")
            .release()?;
        let buffer = self
            .buffers
            .remove(id)
            .expect("buffer existence was checked above");
        drop(buffer);
        let allocation = self.policy.confirm_eviction(id, tick)?;
        Ok(allocation)
    }

    fn advance_tick(&mut self, tick: u64) -> Result<(), DeviceMemoryError> {
        if let Some((previous, provided)) = crate::lifecycle::backwards_tick(self.last_tick, tick) {
            return Err(DeviceMemoryError::NonMonotonicTick { previous, provided });
        }
        self.last_tick = Some(tick);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claymore_device::{CpuDevice, DeviceCapabilities, DeviceInfo};

    struct FailingReleaseBuffer {
        inner: Box<dyn DeviceBuffer>,
        fail_next_release: bool,
    }

    impl DeviceBuffer for FailingReleaseBuffer {
        fn device_id(&self) -> &str {
            self.inner.device_id()
        }

        fn device_instance_id(&self) -> claymore_device::DeviceInstanceId {
            self.inner.device_instance_id()
        }

        fn tier(&self) -> MemoryTier {
            self.inner.tier()
        }

        fn len(&self) -> usize {
            self.inner.len()
        }

        fn release(&mut self) -> Result<(), DeviceError> {
            if self.fail_next_release {
                self.fail_next_release = false;
                return Err(DeviceError::ReleaseFailed(
                    "injected backend failure".to_owned(),
                ));
            }
            self.inner.release()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    struct FailingReleaseDevice {
        inner: CpuDevice,
    }

    impl Device for FailingReleaseDevice {
        fn info(&self) -> &DeviceInfo {
            self.inner.info()
        }

        fn capabilities(&self) -> DeviceCapabilities {
            self.inner.capabilities()
        }

        fn allocate(
            &self,
            tier: MemoryTier,
            bytes: u64,
        ) -> Result<Box<dyn DeviceBuffer>, DeviceError> {
            Ok(Box::new(FailingReleaseBuffer {
                inner: self.inner.allocate(tier, bytes)?,
                fail_next_release: true,
            }))
        }

        fn write(
            &self,
            buffer: &mut dyn DeviceBuffer,
            offset: usize,
            data: &[u8],
        ) -> Result<(), DeviceError> {
            let buffer = buffer
                .as_any_mut()
                .downcast_mut::<FailingReleaseBuffer>()
                .ok_or(DeviceError::BufferTypeMismatch)?;
            self.inner.write(buffer.inner.as_mut(), offset, data)
        }

        fn read(
            &self,
            buffer: &dyn DeviceBuffer,
            offset: usize,
            len: usize,
        ) -> Result<Vec<u8>, DeviceError> {
            let buffer = buffer
                .as_any()
                .downcast_ref::<FailingReleaseBuffer>()
                .ok_or(DeviceError::BufferTypeMismatch)?;
            self.inner.read(buffer.inner.as_ref(), offset, len)
        }

        fn copy(
            &self,
            source: &dyn DeviceBuffer,
            destination: &mut dyn DeviceBuffer,
            bytes: usize,
        ) -> Result<(), DeviceError> {
            let source = source
                .as_any()
                .downcast_ref::<FailingReleaseBuffer>()
                .ok_or(DeviceError::BufferTypeMismatch)?;
            let destination = destination
                .as_any_mut()
                .downcast_mut::<FailingReleaseBuffer>()
                .ok_or(DeviceError::BufferTypeMismatch)?;
            self.inner
                .copy(source.inner.as_ref(), destination.inner.as_mut(), bytes)
        }

        fn synchronize(&self) -> Result<(), DeviceError> {
            Ok(())
        }
    }

    fn budgets() -> MemoryBudgets {
        MemoryBudgets {
            vram_soft_bytes: 64,
            vram_hard_bytes: 128,
            host_bytes: 256,
            pinned_host_bytes: 32,
            expert_cache_target_bytes: 32,
            expert_cache_max_bytes: 64,
        }
    }

    #[test]
    fn cpu_adapter_allocates_copies_and_releases_buffers() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .allocate("source", MemoryTier::Host, 8, 1, false)
            .unwrap();
        manager
            .allocate("destination", MemoryTier::Vram, 8, 2, false)
            .unwrap();
        manager.write("source", 0, &[9, 8, 7, 6], 3).unwrap();
        manager.copy("source", "destination", 4, 4).unwrap();
        assert_eq!(manager.read("destination", 0, 4, 5).unwrap(), [9, 8, 7, 6]);
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 8);
        manager.release("source", 6).unwrap();
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 0);
        manager.synchronize().unwrap();
    }

    #[test]
    fn failed_backend_release_keeps_buffer_and_accounting_for_retry() {
        let device = FailingReleaseDevice {
            inner: CpuDevice::new(),
        };
        let mut manager = DeviceMemoryManager::new(device, budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .allocate("buffer", MemoryTier::Host, 16, 1, false)
            .unwrap();
        manager.write("buffer", 0, &[1, 2, 3], 2).unwrap();

        assert!(matches!(
            manager.release("buffer", 3),
            Err(DeviceMemoryError::Device(DeviceError::ReleaseFailed(_)))
        ));
        assert!(manager.allocation("buffer").is_some());
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 16);
        assert_eq!(manager.read("buffer", 0, 3, 4).unwrap(), [1, 2, 3]);

        manager.release("buffer", 4).unwrap();
        assert!(manager.allocation("buffer").is_none());
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 0);
    }

    #[test]
    fn release_confirms_backend_release_before_removing_accounting() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .allocate("buffer", MemoryTier::Host, 16, 1, false)
            .unwrap();
        let released = manager.release("buffer", 2).unwrap();
        assert_eq!(released.id, "buffer");
        assert!(manager.allocation("buffer").is_none());
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 0);
    }

    #[test]
    fn cpu_adapter_rejects_duplicate_ids_and_non_monotonic_ticks() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .allocate("buffer", MemoryTier::Host, 8, 4, false)
            .unwrap();
        assert!(matches!(
            manager.allocate("buffer", MemoryTier::Host, 8, 5, false),
            Err(DeviceMemoryError::DuplicateAllocation(_))
        ));
        assert!(matches!(
            manager.read("buffer", 0, 1, 3),
            Err(DeviceMemoryError::NonMonotonicTick { .. })
        ));
    }

    #[test]
    fn cpu_adapter_uses_memory_policy_for_deterministic_eviction() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .allocate("old", MemoryTier::Host, 160, 1, false)
            .unwrap();
        manager
            .allocate("new", MemoryTier::Host, 80, 2, false)
            .unwrap();
        manager
            .allocate("replacement", MemoryTier::Host, 200, 3, false)
            .unwrap();
        assert!(manager.allocation("old").is_none());
        assert!(manager.allocation("replacement").is_some());
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 200);
    }
    #[test]
    fn expert_allocations_must_use_vram() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        assert!(matches!(
            manager.allocate("expert", MemoryTier::Host, 8, 1, true),
            Err(DeviceMemoryError::ExpertMustUseVram)
        ));
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 0);
        assert!(manager.allocation("expert").is_none());
    }

    #[test]
    fn aliased_copy_is_rejected_without_mutating_the_buffer() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager
            .allocate("buffer", MemoryTier::Host, 4, 1, false)
            .unwrap();
        manager.write("buffer", 0, &[1, 2, 3, 4], 2).unwrap();

        assert!(matches!(
            manager.copy("buffer", "buffer", 4, 3),
            Err(DeviceMemoryError::AliasedCopy)
        ));
        assert_eq!(manager.read("buffer", 0, 4, 4).unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn in_flight_guard_prevents_release_and_eviction_until_horizon() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager.allocate("protected", MemoryTier::Vram, 80, 1, true).unwrap();
        manager.mark_in_flight_until("protected", 10, 2).unwrap();

        assert!(matches!(
            manager.release("protected", 9),
            Err(DeviceMemoryError::Policy(MemoryError::AllocationBusy(_)))
        ));
        assert!(matches!(
            manager.allocate("replacement", MemoryTier::Vram, 80, 9, false),
            Err(DeviceMemoryError::Policy(MemoryError::InsufficientReclaimable { .. }))
        ));

        manager.release("protected", 10).unwrap();
        manager.allocate("replacement", MemoryTier::Vram, 80, 11, false).unwrap();
    }

    #[test]
    fn expert_cache_max_is_enforced_by_cpu_device_manager() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager.allocate("expert-a", MemoryTier::Vram, 60, 1, true).unwrap();
        manager.allocate("expert-b", MemoryTier::Vram, 40, 2, true).unwrap();
        assert_eq!(manager.usage_bytes(MemoryTier::Vram), 40);
        assert!(manager.allocation("expert-a").is_none());
        assert!(manager.allocation("expert-b").is_some());
    }

    #[test]
    fn completion_horizon_cannot_be_moved_backwards() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        manager.allocate("guarded", MemoryTier::Host, 8, 1, false).unwrap();
        manager.mark_in_flight_until("guarded", 10, 2).unwrap();
        assert!(matches!(
            manager.mark_in_flight_until("guarded", 8, 3),
            Err(DeviceMemoryError::CompletionHorizonRegression {
                previous: 10,
                provided: 8
            })
        ));
        assert!(matches!(
            manager.mark_in_flight_until("guarded", 2, 3),
            Err(DeviceMemoryError::InvalidCompletionHorizon {
                current_tick: 3,
                safe_after_tick: 2
            })
        ));
    }

    #[test]
    fn zero_sized_allocation_is_rejected_without_accounting() {
        let mut manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        assert!(matches!(
            manager.allocate("empty", MemoryTier::Host, 0, 1, false),
            Err(DeviceMemoryError::Device(DeviceError::ZeroSizedAllocation))
        ));
        assert_eq!(manager.usage_bytes(MemoryTier::Host), 0);
        assert!(manager.allocation("empty").is_none());
    }

    #[test]
    fn device_capabilities_are_reported_consistently_by_manager() {
        let manager =
            DeviceMemoryManager::new(CpuDevice::new(), budgets(), EvictionPolicy::Lru).unwrap();
        let capabilities = manager.device().capabilities();
        assert!(capabilities.device_memory);
        assert!(capabilities.host_memory);
        assert!(!capabilities.pinned_host_memory);
        assert!(!capabilities.asynchronous_transfers);
        assert!(!capabilities.tensor_compute);
    }
}
