//! CUDA-backed allocation execution for the host-side memory policy.
//!
//! This adapter is opt-in. Policy tests remain hardware-independent. Every
//! asynchronous transfer is associated with a real CUDA event; allocations
//! touched by unfinished transfers remain non-evictable until synchronization.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::{
    Allocation, EvictionPolicy, MemoryBudgets, MemoryError, MemoryOrchestrator, MemorySnapshot,
};
use claymore_core::MemoryTier;
use claymore_cuda::{
    CudaDeviceBuffer, CudaDriver, CudaError, CudaEvent, CudaStream, PinnedHostBuffer,
};

const IN_FLIGHT: u64 = crate::lifecycle::IN_FLIGHT_TICK;
static NEXT_TRANSFER_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_transfer_id() -> Result<u64, CudaMemoryError> {
    let mut current = NEXT_TRANSFER_ID.load(Ordering::Relaxed);
    loop {
        let next = current
            .checked_add(1)
            .ok_or(CudaMemoryError::TransferIdExhausted)?;
        match NEXT_TRANSFER_ID.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(current),
            Err(observed) => current = observed,
        }
    }
}

fn validate_monotonic_tick(previous: Option<u64>, provided: u64) -> Result<(), CudaMemoryError> {
    if let Some((previous, provided)) = crate::lifecycle::backwards_tick(previous, provided) {
        return Err(CudaMemoryError::NonMonotonicTick { previous, provided });
    }
    Ok(())
}

fn ensure_release_safe(id: &str, safe_after_tick: u64, tick: u64) -> Result<(), CudaMemoryError> {
    if !crate::lifecycle::release_is_safe(safe_after_tick, tick) {
        return Err(MemoryError::AllocationBusy(id.to_owned()).into());
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum CudaMemoryError {
    #[error(transparent)]
    Policy(#[from] MemoryError),
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("allocation {0} does not exist")]
    UnknownAllocation(String),
    #[error("allocation {0} has the wrong memory tier")]
    WrongTier(String),
    #[error("allocation size must be greater than zero")]
    ZeroSizedAllocation,
    #[error("byte range {offset}..{end} exceeds pinned buffer capacity {capacity}")]
    BufferRangeOutOfBounds {
        offset: usize,
        end: usize,
        capacity: usize,
    },
    #[error("CUDA reports only {free} free bytes; {requested} bytes were requested")]
    InsufficientDeviceMemory { requested: u64, free: u64 },
    #[error("logical tick moved backwards: previous {previous}, provided {provided}")]
    NonMonotonicTick { previous: u64, provided: u64 },
    #[error("CUDA memory manager is poisoned after an uncertain transfer synchronization failure")]
    ManagerPoisoned,
    #[error("CUDA transfer token identifier space exhausted")]
    TransferIdExhausted,
    #[error("CUDA transfer token {0} is not owned by this manager")]
    UnknownTransferToken(u64),
}

#[derive(Debug, Clone)]
pub struct TransferToken {
    manager_id: u64,
    id: u64,
    allocation_ids: Vec<String>,
    completed: Arc<AtomicBool>,
}

impl TransferToken {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn allocation_ids(&self) -> &[String] {
        &self.allocation_ids
    }
}

pub struct CudaMemoryManager {
    manager_id: u64,
    driver: CudaDriver,
    transfer_stream: CudaStream,
    policy: MemoryOrchestrator,
    device: HashMap<String, CudaDeviceBuffer>,
    pinned: HashMap<String, PinnedHostBuffer>,
    pending: Vec<(u64, Vec<String>, CudaEvent, Arc<AtomicBool>)>,
    last_tick: Option<u64>,
    poisoned: bool,
}

impl CudaMemoryManager {
    pub fn new(
        ordinal: u32,
        budgets: MemoryBudgets,
        eviction: EvictionPolicy,
    ) -> Result<Self, CudaMemoryError> {
        let manager_id = allocate_transfer_id()?;
        let driver = CudaDriver::new(ordinal)?;
        let transfer_stream = driver.create_stream()?;
        let policy = MemoryOrchestrator::new(budgets, eviction)?;
        Ok(Self {
            manager_id,
            driver,
            transfer_stream,
            policy,
            device: HashMap::new(),
            pinned: HashMap::new(),
            pending: Vec::new(),
            last_tick: None,
            poisoned: false,
        })
    }

    /// Enforce one monotonic logical clock for every operation on this manager.
    fn advance_tick(&mut self, tick: u64) -> Result<(), CudaMemoryError> {
        if self.poisoned {
            return Err(CudaMemoryError::ManagerPoisoned);
        }
        validate_monotonic_tick(self.last_tick, tick)?;
        self.last_tick = Some(tick);
        Ok(())
    }

    pub fn device_info(&self) -> &claymore_cuda::CudaDeviceInfo {
        self.driver.device_info()
    }

    pub fn snapshot(
        &self,
        host_total_bytes: u64,
        host_available_bytes: u64,
    ) -> Result<MemorySnapshot, CudaMemoryError> {
        let (vram_free_bytes, vram_total_bytes) = self.driver.memory_info()?;
        let snapshot = MemorySnapshot {
            vram_total_bytes,
            vram_free_bytes,
            host_total_bytes,
            host_available_bytes,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn usage_bytes(&self, tier: MemoryTier) -> u64 {
        self.policy.usage_bytes(tier)
    }

    pub fn allocate_device(
        &mut self,
        id: impl Into<String>,
        bytes: u64,
        tick: u64,
        expert: bool,
    ) -> Result<(), CudaMemoryError> {
        self.advance_tick(tick)?;
        let id = id.into();
        if bytes == 0 {
            return Err(CudaMemoryError::ZeroSizedAllocation);
        }
        self.ensure_new_id(&id)?;
        self.make_room(MemoryTier::Vram, bytes, tick)?;
        let (free, _) = self.driver.memory_info()?;
        if bytes > free {
            return Err(CudaMemoryError::InsufficientDeviceMemory {
                requested: bytes,
                free,
            });
        }
        let buffer = self.driver.allocate(bytes)?;
        let allocation = Allocation {
            id: id.clone(),
            tier: MemoryTier::Vram,
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
        self.device.insert(id, buffer);
        Ok(())
    }

    pub fn allocate_pinned(
        &mut self,
        id: impl Into<String>,
        bytes: u64,
        tick: u64,
    ) -> Result<(), CudaMemoryError> {
        self.advance_tick(tick)?;
        let id = id.into();
        if bytes == 0 {
            return Err(CudaMemoryError::ZeroSizedAllocation);
        }
        self.ensure_new_id(&id)?;
        self.make_room(MemoryTier::PinnedHost, bytes, tick)?;
        let buffer = self.driver.allocate_pinned(bytes)?;
        let allocation = Allocation {
            id: id.clone(),
            tier: MemoryTier::PinnedHost,
            bytes,
            last_used_tick: tick,
            hit_count: 0,
            safe_after_tick: tick,
            expert: false,
        };
        if let Err(error) = self.policy.record_allocation(allocation) {
            drop(buffer);
            return Err(error.into());
        }
        self.pinned.insert(id, buffer);
        Ok(())
    }

    /// Writes bytes into a pinned buffer before an H2D transfer.
    pub fn write_pinned(
        &mut self,
        id: &str,
        offset: usize,
        data: &[u8],
        tick: u64,
    ) -> Result<(), CudaMemoryError> {
        self.advance_tick(tick)?;
        self.wait_for_allocation(id, tick)?;
        let buffer = self
            .pinned
            .get_mut(id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(id.to_owned()))?;
        let end =
            offset
                .checked_add(data.len())
                .ok_or(CudaMemoryError::BufferRangeOutOfBounds {
                    offset,
                    end: usize::MAX,
                    capacity: buffer.len(),
                })?;
        if end > buffer.len() {
            return Err(CudaMemoryError::BufferRangeOutOfBounds {
                offset,
                end,
                capacity: buffer.len(),
            });
        }
        // SAFETY: the range is checked and no transfer can be in flight while
        // the caller mutably borrows this manager to write the buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.as_mut_ptr().add(offset),
                data.len(),
            );
        }
        Ok(())
    }

    /// Waits for outstanding D2H work and returns a copy of the requested bytes.
    pub fn read_pinned(
        &mut self,
        id: &str,
        offset: usize,
        len: usize,
        tick: u64,
    ) -> Result<Vec<u8>, CudaMemoryError> {
        self.advance_tick(tick)?;
        self.wait_for_allocation(id, tick)?;
        let buffer = self
            .pinned
            .get_mut(id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(id.to_owned()))?;
        let end = offset
            .checked_add(len)
            .ok_or(CudaMemoryError::BufferRangeOutOfBounds {
                offset,
                end: usize::MAX,
                capacity: buffer.len(),
            })?;
        if end > buffer.len() {
            return Err(CudaMemoryError::BufferRangeOutOfBounds {
                offset,
                end,
                capacity: buffer.len(),
            });
        }
        // SAFETY: the range is checked and wait_for_allocation completed every
        // CUDA operation that could write this pinned buffer.
        let bytes = unsafe { std::slice::from_raw_parts(buffer.as_mut_ptr().add(offset), len) };
        Ok(bytes.to_vec())
    }

    pub fn copy_h2d(
        &mut self,
        pinned_id: &str,
        device_id: &str,
        bytes: usize,
        tick: u64,
    ) -> Result<TransferToken, CudaMemoryError> {
        self.advance_tick(tick)?;
        let transfer_id = allocate_transfer_id()?;
        let event = self.driver.create_event()?;
        let source = self
            .pinned
            .get(pinned_id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(pinned_id.to_owned()))?;
        let destination = self
            .device
            .get_mut(device_id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(device_id.to_owned()))?;
        // SAFETY: both allocations stay owned by this manager until the event completes.
        unsafe {
            destination.copy_from_pinned_async(source, bytes, &self.transfer_stream)?;
        }
        if let Err(error) = event.record(&self.transfer_stream) {
            if let Err(sync_error) = self.transfer_stream.synchronize() {
                // The copy may still be using either allocation. Fail closed:
                // prevent every subsequent operation and keep both allocations
                // non-evictable until manager destruction can synchronize again.
                self.poisoned = true;
                self.policy.mark_in_flight_until(pinned_id, IN_FLIGHT)?;
                self.policy.mark_in_flight_until(device_id, IN_FLIGHT)?;
                return Err(sync_error.into());
            }
            return Err(error.into());
        }
        self.policy.mark_in_flight_until(pinned_id, IN_FLIGHT)?;
        self.policy.mark_in_flight_until(device_id, IN_FLIGHT)?;
        let allocation_ids = vec![pinned_id.to_owned(), device_id.to_owned()];
        let completed = Arc::new(AtomicBool::new(false));
        self.pending.push((
            transfer_id,
            allocation_ids.clone(),
            event,
            Arc::clone(&completed),
        ));
        self.policy.touch(pinned_id, tick)?;
        self.policy.touch(device_id, tick)?;
        Ok(TransferToken {
            manager_id: self.manager_id,
            id: transfer_id,
            allocation_ids,
            completed,
        })
    }

    pub fn copy_d2h(
        &mut self,
        device_id: &str,
        pinned_id: &str,
        bytes: usize,
        tick: u64,
    ) -> Result<TransferToken, CudaMemoryError> {
        self.advance_tick(tick)?;
        let transfer_id = allocate_transfer_id()?;
        let event = self.driver.create_event()?;
        let source = self
            .device
            .get(device_id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(device_id.to_owned()))?;
        let destination = self
            .pinned
            .get_mut(pinned_id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(pinned_id.to_owned()))?;
        // SAFETY: both allocations stay owned by this manager until the event completes.
        unsafe {
            source.copy_to_pinned_async(destination, bytes, &self.transfer_stream)?;
        }
        if let Err(error) = event.record(&self.transfer_stream) {
            if let Err(sync_error) = self.transfer_stream.synchronize() {
                // The copy may still be using either allocation. Fail closed:
                // prevent every subsequent operation and keep both allocations
                // non-evictable until manager destruction can synchronize again.
                self.poisoned = true;
                self.policy.mark_in_flight_until(device_id, IN_FLIGHT)?;
                self.policy.mark_in_flight_until(pinned_id, IN_FLIGHT)?;
                return Err(sync_error.into());
            }
            return Err(error.into());
        }
        self.policy.mark_in_flight_until(device_id, IN_FLIGHT)?;
        self.policy.mark_in_flight_until(pinned_id, IN_FLIGHT)?;
        let allocation_ids = vec![device_id.to_owned(), pinned_id.to_owned()];
        let completed = Arc::new(AtomicBool::new(false));
        self.pending.push((
            transfer_id,
            allocation_ids.clone(),
            event,
            Arc::clone(&completed),
        ));
        self.policy.touch(device_id, tick)?;
        self.policy.touch(pinned_id, tick)?;
        Ok(TransferToken {
            manager_id: self.manager_id,
            id: transfer_id,
            allocation_ids,
            completed,
        })
    }

    /// Poll every pending transfer without blocking and retire only events
    /// that CUDA positively reports as complete.
    ///
    /// If any event query fails, the error is returned and that event remains
    /// tracked with its allocation guards intact. Earlier positively completed
    /// events may already have been retired; an error is never treated as success.
    pub fn poll_completed_transfers(&mut self, tick: u64) -> Result<usize, CudaMemoryError> {
        self.advance_tick(tick)?;
        let mut retired = 0;
        let mut index = 0;
        while index < self.pending.len() {
            // Query before mutating the tracker. Any driver error leaves this
            // event and every allocation it protects untouched.
            if !self.pending[index].2.is_complete()? {
                index += 1;
                continue;
            }

            let (_, ids, _event, completed) = self.pending.remove(index);
            completed.store(true, Ordering::Release);
            for completed_id in ids {
                let still_in_flight = self.pending.iter().any(|(_, pending_ids, _, _)| {
                    pending_ids.iter().any(|pending_id| pending_id == &completed_id)
                });
                if !still_in_flight {
                    self.policy.mark_in_flight_until(&completed_id, tick)?;
                }
            }
            retired += 1;
        }
        Ok(retired)
    }

    /// Query the associated CUDA event without blocking. A token is complete
    /// after its event query succeeds or an explicit wait synchronizes it.
    /// This query does not retire the pending event or release allocation guards;
    /// call wait_for_transfer (or wait_for_allocation) to retire the tracked work.
    pub fn is_transfer_complete(&self, token: &TransferToken) -> Result<bool, CudaMemoryError> {
        if token.manager_id != self.manager_id {
            return Err(CudaMemoryError::UnknownTransferToken(token.id));
        }
        if token.completed.load(Ordering::Acquire) {
            return Ok(true);
        }
        if let Some((_, _, event, completed)) = self.pending.iter().find(|(id, ids, _, state)| {
            *id == token.id && *ids == token.allocation_ids && Arc::ptr_eq(state, &token.completed)
        }) {
            let is_complete = event.is_complete()?;
            if is_complete {
                completed.store(true, Ordering::Release);
            }
            Ok(is_complete)
        } else {
            Err(CudaMemoryError::UnknownTransferToken(token.id))
        }
    }

    /// Wait for the exact transfer represented by the token. A synchronization
    /// error leaves the event and its allocation protections tracked.
    pub fn wait_for_transfer(
        &mut self,
        token: &TransferToken,
        tick: u64,
    ) -> Result<(), CudaMemoryError> {
        self.advance_tick(tick)?;
        if token.manager_id != self.manager_id {
            return Err(CudaMemoryError::UnknownTransferToken(token.id));
        }
        let allocation_ids = self
            .pending
            .iter()
            .find(|(id, ids, _, state)| {
                *id == token.id
                    && *ids == token.allocation_ids
                    && Arc::ptr_eq(state, &token.completed)
            })
            .map(|(_, ids, _, _)| ids.clone());
        if let Some(allocation_ids) = allocation_ids {
            if let Some(id) = allocation_ids.first() {
                self.wait_for_allocation(id, tick)?;
            }
            token.completed.store(true, Ordering::Release);
            return Ok(());
        }
        if token.completed.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(CudaMemoryError::UnknownTransferToken(token.id))
        }
    }

    pub fn wait_for_allocation(&mut self, id: &str, tick: u64) -> Result<(), CudaMemoryError> {
        self.advance_tick(tick)?;
        let mut index = 0;
        while index < self.pending.len() {
            let matches = self.pending[index]
                .1
                .iter()
                .any(|candidate| candidate == id);
            if matches {
                // Synchronize before removing the event. On failure the event and
                // every allocation it protects remain tracked and non-evictable.
                self.pending[index].2.synchronize()?;
                let (_, ids, _event, completed) = self.pending.remove(index);
                completed.store(true, Ordering::Release);
                for completed_id in ids {
                    let still_in_flight = self.pending.iter().any(|(_, pending_ids, _, _)| {
                        pending_ids
                            .iter()
                            .any(|pending_id| pending_id == &completed_id)
                    });
                    if !still_in_flight {
                        self.policy.mark_in_flight_until(&completed_id, tick)?;
                    }
                }
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub fn release(&mut self, id: &str, tick: u64) -> Result<Allocation, CudaMemoryError> {
        self.advance_tick(tick)?;
        self.wait_for_allocation(id, tick)?;
        let allocation = self
            .policy
            .allocation(id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(id.to_owned()))?;
        ensure_release_safe(id, allocation.safe_after_tick, tick)?;
        let tier = allocation.tier;
        match tier {
            MemoryTier::Vram => {
                // SAFETY: wait_for_allocation synchronized every event involving id.
                unsafe {
                    self.device
                        .get_mut(id)
                        .ok_or_else(|| CudaMemoryError::WrongTier(id.to_owned()))?
                        .free()?;
                }
                let _ = self.device.remove(id);
            }
            MemoryTier::PinnedHost => {
                // SAFETY: wait_for_allocation synchronized every event involving id.
                unsafe {
                    self.pinned
                        .get_mut(id)
                        .ok_or_else(|| CudaMemoryError::WrongTier(id.to_owned()))?
                        .free()?;
                }
                let _ = self.pinned.remove(id);
            }
            MemoryTier::Host => return Err(CudaMemoryError::WrongTier(id.to_owned())),
        }
        Ok(self.policy.confirm_eviction(id, tick)?)
    }

    fn ensure_new_id(&self, id: &str) -> Result<(), CudaMemoryError> {
        if self.policy.allocation(id).is_some()
            || self.device.contains_key(id)
            || self.pinned.contains_key(id)
        {
            return Err(MemoryError::DuplicateAllocation(id.to_owned()).into());
        }
        Ok(())
    }

    fn make_room(
        &mut self,
        tier: MemoryTier,
        bytes: u64,
        tick: u64,
    ) -> Result<(), CudaMemoryError> {
        // Reclaim transfers already completed by the device before planning
        // evictions. Polling is non-blocking; pending/error states stay guarded.
        self.poll_completed_transfers(tick)?;
        let plan = self.policy.plan_admission(tier, bytes, tick)?;
        for id in plan.evict_ids {
            self.release(&id, tick)?;
        }
        Ok(())
    }
}

impl Drop for CudaMemoryManager {
    fn drop(&mut self) {
        // Do not destroy buffers while queued DMA may still reference them.
        // Only a successful synchronization of the transfer stream proves that
        // every operation enqueued on it has completed. Older events in
        // `pending` cannot prove completion of a later copy whose event record
        // failed, so never use them as a fallback.
        let safe_to_drop = self.transfer_stream.synchronize().is_ok();
        if safe_to_drop {
            for (_, _, event, completed) in self.pending.drain(..) {
                completed.store(true, Ordering::Release);
                drop(event);
            }
        } else {
            // A failed driver synchronization leaves buffer lifetime uncertain.
            // Leak these CUDA owners rather than risk freeing memory still in use.
            for (_, buffer) in self.device.drain() {
                std::mem::forget(buffer);
            }
            for (_, buffer) in self.pinned.drain() {
                std::mem::forget(buffer);
            }
            for (_, _, event, _) in self.pending.drain(..) {
                std::mem::forget(event);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_token_clones_share_completion_state() {
        let completed = Arc::new(AtomicBool::new(false));
        let token = TransferToken {
            manager_id: 1,
            id: 2,
            allocation_ids: vec!["device".to_owned(), "pinned".to_owned()],
            completed: Arc::clone(&completed),
        };
        let clone = token.clone();

        clone.completed.store(true, Ordering::Release);
        assert!(token.completed.load(Ordering::Acquire));
    }

    #[test]
    fn logical_ticks_must_be_monotonic() {
        assert!(validate_monotonic_tick(None, 10).is_ok());
        assert!(validate_monotonic_tick(Some(10), 10).is_ok());
        assert!(validate_monotonic_tick(Some(10), 11).is_ok());
        assert!(matches!(
            validate_monotonic_tick(Some(10), 9),
            Err(CudaMemoryError::NonMonotonicTick {
                previous: 10,
                provided: 9
            })
        ));
    }

    #[test]
    fn release_guard_rejects_tick_before_safe_horizon() {
        assert!(matches!(
            ensure_release_safe("buffer", 10, 9),
            Err(CudaMemoryError::Policy(MemoryError::AllocationBusy(id))) if id == "buffer"
        ));
        assert!(ensure_release_safe("buffer", 10, 10).is_ok());
    }

    #[test]
    #[ignore = "requires a CUDA driver and a usable NVIDIA device"]
    fn cuda_manager_reports_device_memory() {
        let gib = 1024_u64.pow(3);
        let budgets = MemoryBudgets {
            vram_soft_bytes: 26 * gib,
            vram_hard_bytes: 28 * gib,
            host_bytes: 80 * gib,
            pinned_host_bytes: 8 * gib,
            expert_cache_target_bytes: 14 * gib,
            expert_cache_max_bytes: 16 * gib,
        };
        let mut manager = CudaMemoryManager::new(0, budgets, EvictionPolicy::Lru).unwrap();
        let (free, total) = manager.driver.memory_info().unwrap();
        assert!(free <= total);
        assert!(manager.device_info().compute_capability.is_some());
        let peer_driver = CudaDriver::new(0).unwrap();
        assert!(manager.driver.shares_context_with(&peer_driver));

        let expected = b"clay";
        manager
            .allocate_pinned("host", expected.len() as u64, 1)
            .unwrap();
        manager
            .allocate_device("device", expected.len() as u64, 1, false)
            .unwrap();
        manager.write_pinned("host", 0, expected, 2).unwrap();
        let h2d = manager
            .copy_h2d("host", "device", expected.len(), 3)
            .unwrap();
        // Polling is explicitly non-blocking; the transfer may or may not have
        // completed by this point. Either result is valid.
        let _ = manager.is_transfer_complete(&h2d).unwrap();
        manager.wait_for_transfer(&h2d, 4).unwrap();
        let d2h = manager
            .copy_d2h("device", "host", expected.len(), 5)
            .unwrap();
        manager.wait_for_transfer(&d2h, 6).unwrap();
        assert_eq!(
            manager.read_pinned("host", 0, expected.len(), 7).unwrap(),
            expected
        );
        manager.release("device", 8).unwrap();
        manager.release("host", 8).unwrap();
    }
}
