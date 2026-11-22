//! CUDA-backed allocation execution for the host-side memory policy.
//!
//! This adapter is opt-in. Policy tests remain hardware-independent. Every
//! asynchronous transfer is associated with a real CUDA event; allocations
//! touched by unfinished transfers remain non-evictable until synchronization.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::{
    AdmissionPlan, Allocation, EvictionPolicy, MemoryBudgets, MemoryError, MemoryOrchestrator,
    MemorySnapshot,
};
use claymore_core::MemoryTier;
use claymore_cuda::{
    CudaComputeEngine, CudaDeviceBuffer, CudaDriver, CudaError, CudaEvent, CudaStream,
    PinnedHostBuffer,
};
use claymore_device::DType;
use claymore_tensor::{OpError, TensorOp};

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
    #[error(transparent)]
    Tensor(#[from] claymore_tensor::OpError),
    #[error("tensor launch inputs do not match the operation descriptor")]
    TensorInputMismatch,
    #[error("compute output allocation is missing from the VRAM map")]
    ComputeOutputMissing,
}

/// Private operation boundary for event lifecycle. Production delegates to
/// CUDA Driver API handles; tests inject deterministic event outcomes without
/// constructing or submitting fabricated handles to a real driver.
trait CudaEventOps {
    fn is_complete(&self) -> Result<bool, CudaError>;
    fn synchronize(&self) -> Result<(), CudaError>;
}

impl CudaEventOps for CudaEvent {
    fn is_complete(&self) -> Result<bool, CudaError> {
        CudaEvent::is_complete(self)
    }
    fn synchronize(&self) -> Result<(), CudaError> {
        CudaEvent::synchronize(self)
    }
}

trait CudaStreamOps {
    fn synchronize(&self) -> Result<(), CudaError>;
}
impl CudaStreamOps for CudaStream {
    fn synchronize(&self) -> Result<(), CudaError> {
        CudaStream::synchronize(self)
    }
}

trait CudaBufferRelease {
    fn release(&mut self) -> Result<(), CudaError>;
}
impl CudaBufferRelease for CudaDeviceBuffer {
    fn release(&mut self) -> Result<(), CudaError> {
        // SAFETY: callers synchronize every event that may reference this buffer.
        unsafe { self.free() }
    }
}
impl CudaBufferRelease for PinnedHostBuffer {
    fn release(&mut self) -> Result<(), CudaError> {
        // SAFETY: callers synchronize every event that may reference this buffer.
        unsafe { self.free() }
    }
}

struct PendingTransfer<E> {
    id: u64,
    allocation_ids: Vec<String>,
    event: E,
    completed: Arc<AtomicBool>,
}

/// State shared by the production CUDA manager and deterministic manager-level
/// tests. Keeping policy and pending events together prevents tests from
/// validating helper functions while bypassing the manager's transfer state.
struct CudaTransferTracker<E> {
    policy: MemoryOrchestrator,
    pending: Vec<PendingTransfer<E>>,
}

impl<E: CudaEventOps> CudaTransferTracker<E> {
    fn poll(&mut self, tick: u64) -> Result<usize, CudaMemoryError> {
        poll_pending_transfers(&mut self.policy, &mut self.pending, tick)
    }

    fn wait_for_allocation(&mut self, id: &str, tick: u64) -> Result<(), CudaMemoryError> {
        wait_for_pending_allocation(&mut self.policy, &mut self.pending, id, tick)
    }

    fn plan_admission(
        &mut self,
        tier: MemoryTier,
        bytes: u64,
        tick: u64,
    ) -> Result<AdmissionPlan, CudaMemoryError> {
        plan_admission_after_poll(&mut self.policy, &mut self.pending, tier, bytes, tick)
    }
}

fn poll_pending_transfers<E: CudaEventOps>(
    policy: &mut MemoryOrchestrator,
    pending: &mut Vec<PendingTransfer<E>>,
    tick: u64,
) -> Result<usize, CudaMemoryError> {
    let mut retired = 0;
    let mut index = 0;
    while index < pending.len() {
        if !pending[index].event.is_complete()? {
            index += 1;
            continue;
        }
        let ids = pending[index].allocation_ids.clone();
        for completed_id in &ids {
            let still_in_flight = pending.iter().enumerate().any(|(pending_index, transfer)| {
                pending_index != index
                    && transfer
                        .allocation_ids
                        .iter()
                        .any(|pending_id| pending_id == completed_id)
            });
            if !still_in_flight {
                policy.mark_in_flight_until(completed_id, tick)?;
            }
        }
        let transfer = pending.remove(index);
        transfer.completed.store(true, Ordering::Release);
        retired += 1;
    }
    Ok(retired)
}

fn plan_admission_after_poll<E: CudaEventOps>(
    policy: &mut MemoryOrchestrator,
    pending: &mut Vec<PendingTransfer<E>>,
    tier: MemoryTier,
    bytes: u64,
    tick: u64,
) -> Result<AdmissionPlan, CudaMemoryError> {
    // This ordering is the fail-closed contract used by CudaMemoryManager::make_room.
    poll_pending_transfers(policy, pending, tick)?;
    Ok(policy.plan_admission(tier, bytes, tick)?)
}

fn wait_for_pending_allocation<E: CudaEventOps>(
    policy: &mut MemoryOrchestrator,
    pending: &mut Vec<PendingTransfer<E>>,
    id: &str,
    tick: u64,
) -> Result<(), CudaMemoryError> {
    let mut index = 0;
    while index < pending.len() {
        if pending[index]
            .allocation_ids
            .iter()
            .any(|candidate| candidate == id)
        {
            pending[index].event.synchronize()?;
            let transfer = pending.remove(index);
            transfer.completed.store(true, Ordering::Release);
            for completed_id in transfer.allocation_ids {
                let still_in_flight = pending.iter().any(|other| {
                    other
                        .allocation_ids
                        .iter()
                        .any(|pending_id| pending_id == &completed_id)
                });
                if !still_in_flight {
                    policy.mark_in_flight_until(&completed_id, tick)?;
                }
            }
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn synchronize_stream<S: CudaStreamOps>(stream: &S) -> Result<(), CudaError> {
    stream.synchronize()
}

fn release_and_confirm<B: CudaBufferRelease>(
    buffer: &mut B,
    policy: &mut MemoryOrchestrator,
    id: &str,
    tick: u64,
) -> Result<Allocation, CudaMemoryError> {
    buffer.release()?;
    Ok(policy.confirm_eviction(id, tick)?)
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
    /// The single compute stream reserved by the hardware profile. It is kept
    /// separate from the DMA stream so kernel work never uses the legacy
    /// default stream and never implicitly synchronizes with transfers.
    compute_stream: CudaStream,
    compute_engine: CudaComputeEngine,
    transfers: CudaTransferTracker<CudaEvent>,
    device: HashMap<String, CudaDeviceBuffer>,
    pinned: HashMap<String, PinnedHostBuffer>,
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
        let compute_stream = driver.create_stream()?;
        let compute_engine = driver.create_compute_engine()?;
        let policy = MemoryOrchestrator::new(budgets, eviction)?;
        Ok(Self {
            manager_id,
            driver,
            transfer_stream,
            compute_stream,
            compute_engine,
            transfers: CudaTransferTracker {
                policy,
                pending: Vec::new(),
            },
            device: HashMap::new(),
            pinned: HashMap::new(),
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
        self.transfers.policy.usage_bytes(tier)
    }

    /// Policy accounting entry for one managed allocation, if it is live.
    pub fn allocation(&self, id: &str) -> Option<&Allocation> {
        self.transfers.policy.allocation(id)
    }

    /// Store an observed/estimated reload cost for adaptive eviction.
    pub fn record_reload_cost(
        &mut self,
        id: &str,
        cost_ns: u64,
        tick: u64,
    ) -> Result<(), CudaMemoryError> {
        self.advance_tick(tick)?;
        self.transfers.policy.record_reload_cost(id, cost_ns)?;
        Ok(())
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
        if expert {
            self.make_room_for_expert(bytes, tick)?;
        } else {
            self.make_room(MemoryTier::Vram, bytes, tick)?;
        }
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
        if let Err(error) = self.transfers.policy.record_allocation(allocation) {
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
        if let Err(error) = self.transfers.policy.record_allocation(allocation) {
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
                self.transfers
                    .policy
                    .mark_in_flight_until(pinned_id, IN_FLIGHT)?;
                self.transfers
                    .policy
                    .mark_in_flight_until(device_id, IN_FLIGHT)?;
                return Err(sync_error.into());
            }
            return Err(error.into());
        }
        self.transfers
            .policy
            .mark_in_flight_until(pinned_id, IN_FLIGHT)?;
        self.transfers
            .policy
            .mark_in_flight_until(device_id, IN_FLIGHT)?;
        let allocation_ids = vec![pinned_id.to_owned(), device_id.to_owned()];
        let completed = Arc::new(AtomicBool::new(false));
        self.transfers.pending.push(PendingTransfer {
            id: transfer_id,
            allocation_ids: allocation_ids.clone(),
            event,
            completed: Arc::clone(&completed),
        });
        self.transfers.policy.touch(pinned_id, tick)?;
        self.transfers.policy.touch(device_id, tick)?;
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
                self.transfers
                    .policy
                    .mark_in_flight_until(device_id, IN_FLIGHT)?;
                self.transfers
                    .policy
                    .mark_in_flight_until(pinned_id, IN_FLIGHT)?;
                return Err(sync_error.into());
            }
            return Err(error.into());
        }
        self.transfers
            .policy
            .mark_in_flight_until(device_id, IN_FLIGHT)?;
        self.transfers
            .policy
            .mark_in_flight_until(pinned_id, IN_FLIGHT)?;
        let allocation_ids = vec![device_id.to_owned(), pinned_id.to_owned()];
        let completed = Arc::new(AtomicBool::new(false));
        self.transfers.pending.push(PendingTransfer {
            id: transfer_id,
            allocation_ids: allocation_ids.clone(),
            event,
            completed: Arc::clone(&completed),
        });
        self.transfers.policy.touch(device_id, tick)?;
        self.transfers.policy.touch(pinned_id, tick)?;
        Ok(TransferToken {
            manager_id: self.manager_id,
            id: transfer_id,
            allocation_ids,
            completed,
        })
    }

    /// Enqueue one native CUDA kernel for a validated M8 tensor operation.
    ///
    /// Output allocation has already gone through `allocate_device` (and thus
    /// through the memory policy). Every touched allocation is marked in flight
    /// before the launch and is protected by a real CUDA event recorded on the
    /// dedicated compute stream. The returned token must be waited on with
    /// `wait_for_transfer` before the caller may read, release or evict any of
    /// those allocations. A launch/event failure leaves the guards in place.
    pub fn launch_tensor_op(
        &mut self,
        op: &TensorOp<'_>,
        input_ids: &[&str],
        output_id: &str,
        tick: u64,
    ) -> Result<TransferToken, CudaMemoryError> {
        self.advance_tick(tick)?;
        let validated = op.validate()?;
        let inputs = op.inputs();
        if input_ids.len() != inputs.len() {
            return Err(CudaMemoryError::TensorInputMismatch);
        }
        if !self.device.contains_key(output_id) {
            return Err(CudaMemoryError::ComputeOutputMissing);
        }
        let mut allocation_ids: Vec<String> = input_ids.iter().map(|id| (*id).to_owned()).collect();
        allocation_ids.push(output_id.to_owned());
        let mut pointers = Vec::with_capacity(allocation_ids.len());
        for (index, id) in input_ids.iter().enumerate() {
            let buffer = self
                .device
                .get(*id)
                .ok_or_else(|| CudaMemoryError::UnknownAllocation((*id).to_owned()))?;
            let tensor = inputs[index];
            let offset = tensor.offset_bytes();
            let end = offset
                .checked_add(tensor.layout().span_bytes())
                .ok_or(CudaMemoryError::Tensor(OpError::Tensor(
                    claymore_tensor::TensorError::SizeOverflow,
                )))?;
            if end > buffer.len() {
                return Err(CudaMemoryError::Cuda(CudaError::BufferTooSmall {
                    requested: end,
                    capacity: buffer.len(),
                }));
            }
            let pointer = buffer
                .device_ptr()
                .checked_add(u64::try_from(offset).map_err(|_| CudaError::SizeOverflow)?)
                .ok_or(CudaError::SizeOverflow)?;
            pointers.push(pointer);
        }
        let output_buffer = self
            .device
            .get(output_id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(output_id.to_owned()))?;
        let output_elements = op.output_shape().into_iter().try_fold(1usize, |n, d| {
            n.checked_mul(d).ok_or(CudaError::SizeOverflow)
        })?;
        let output_bytes = output_elements
            .checked_mul(validated.dtype.element_size_bytes())
            .ok_or(CudaError::SizeOverflow)?;
        if output_bytes > output_buffer.len() {
            return Err(CudaMemoryError::Cuda(CudaError::BufferTooSmall {
                requested: output_bytes,
                capacity: output_buffer.len(),
            }));
        }
        pointers.push(output_buffer.device_ptr());

        let shapes = op.shape_signature();
        // Allocate fallible bookkeeping resources before submitting device work.
        let transfer_id = allocate_transfer_id()?;
        let event = self.driver.create_event()?;

        let engine = &self.compute_engine;
        let stream = &self.compute_stream;
        let output_ptr = pointers[pointers.len() - 1];
        // Dispatch is exact on both dtype and shape signature. There is no
        // cross-dtype launch path: BF16 uses its own kernels, F32 keeps its own.
        let launch = match (validated.dtype, shapes) {
            (DType::F32, claymore_tensor::ShapeSignature::Elementwise { numel }) => {
                engine.add(pointers[0], pointers[1], output_ptr, numel, stream)
            }
            (DType::BF16, claymore_tensor::ShapeSignature::Elementwise { numel }) => {
                engine.add_bf16(pointers[0], pointers[1], output_ptr, numel, stream)
            }
            (DType::F32, claymore_tensor::ShapeSignature::MatMul { m, k, n }) => {
                engine.matmul(pointers[0], pointers[1], output_ptr, m, k, n, stream)
            }
            (DType::BF16, claymore_tensor::ShapeSignature::MatMul { m, k, n }) => {
                engine.matmul_bf16(pointers[0], pointers[1], output_ptr, m, k, n, stream)
            }
            (
                DType::F32,
                claymore_tensor::ShapeSignature::LastDim {
                    rows,
                    width,
                    epsilon,
                },
            ) => match epsilon {
                Some(value) => engine.rms_norm(
                    pointers[0],
                    pointers[1],
                    output_ptr,
                    rows,
                    width,
                    value,
                    stream,
                ),
                None => engine.softmax_last_dim(pointers[0], output_ptr, rows, width, stream),
            },
            (
                DType::BF16,
                claymore_tensor::ShapeSignature::LastDim {
                    rows,
                    width,
                    epsilon,
                },
            ) => match epsilon {
                Some(value) => engine.rms_norm_bf16(
                    pointers[0],
                    pointers[1],
                    output_ptr,
                    rows,
                    width,
                    value,
                    stream,
                ),
                None => engine.softmax_last_dim_bf16(pointers[0], output_ptr, rows, width, stream),
            },
            (_dtype, _) => {
                return Err(CudaMemoryError::Tensor(OpError::UnsupportedDType));
            }
        };
        if let Err(error) = launch {
            // A driver error may be reported after work was partially submitted.
            // Prove stream quiescence before returning control to the caller.
            if let Err(sync_error) = synchronize_stream(&self.compute_stream) {
                self.poisoned = true;
                for id in &allocation_ids {
                    let _ = self.transfers.policy.mark_in_flight_until(id, IN_FLIGHT);
                }
                return Err(sync_error.into());
            }
            return Err(error.into());
        }

        // The manager is exclusively borrowed for this whole method, so no
        // release/eviction can race between enqueue and installing these guards.
        let mut guarded = Vec::with_capacity(allocation_ids.len());
        for id in &allocation_ids {
            if let Err(error) = self.transfers.policy.mark_in_flight_until(id, IN_FLIGHT) {
                match synchronize_stream(&self.compute_stream) {
                    Ok(()) => {
                        for guarded_id in &guarded {
                            self.transfers.policy.mark_in_flight_until(guarded_id, tick)?;
                        }
                        return Err(error.into());
                    }
                    Err(sync_error) => {
                        self.poisoned = true;
                        for guarded_id in &allocation_ids {
                            let _ = self.transfers.policy.mark_in_flight_until(guarded_id, IN_FLIGHT);
                        }
                        return Err(sync_error.into());
                    }
                }
            }
            guarded.push(id.clone());
        }

        if let Err(error) = event.record(&self.compute_stream) {
            match synchronize_stream(&self.compute_stream) {
                Ok(()) => {
                    for id in &allocation_ids {
                        self.transfers.policy.mark_in_flight_until(id, tick)?;
                    }
                    return Err(error.into());
                }
                Err(sync_error) => {
                    self.poisoned = true;
                    for id in &allocation_ids {
                        let _ = self.transfers.policy.mark_in_flight_until(id, IN_FLIGHT);
                    }
                    return Err(sync_error.into());
                }
            }
        }
        self.transfers.pending.push(PendingTransfer {
            id: transfer_id,
            allocation_ids: allocation_ids.clone(),
            event,
            completed: Arc::new(AtomicBool::new(false)),
        });
        let completed = Arc::clone(&self.transfers.pending.last().unwrap().completed);
        for id in &allocation_ids {
            self.transfers.policy.touch(id, tick)?;
        }
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
    /// tracked with its allocation guards intact. A completed event is also kept
    /// tracked until every guard no longer covered by another pending transfer
    /// has been lifted successfully; policy errors therefore remain retryable.
    /// Earlier completed events may already have been retired; an error is never
    /// treated as success.
    pub fn poll_completed_transfers(&mut self, tick: u64) -> Result<usize, CudaMemoryError> {
        self.advance_tick(tick)?;
        self.transfers.poll(tick)
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
        if let Some(transfer) = self.transfers.pending.iter().find(|transfer| {
            transfer.id == token.id
                && transfer.allocation_ids == token.allocation_ids
                && Arc::ptr_eq(&transfer.completed, &token.completed)
        }) {
            let is_complete = transfer.event.is_complete()?;
            if is_complete {
                transfer.completed.store(true, Ordering::Release);
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
            .transfers
            .pending
            .iter()
            .find(|transfer| {
                transfer.id == token.id
                    && transfer.allocation_ids == token.allocation_ids
                    && Arc::ptr_eq(&transfer.completed, &token.completed)
            })
            .map(|transfer| transfer.allocation_ids.clone());
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
        self.transfers.wait_for_allocation(id, tick)
    }

    pub fn release(&mut self, id: &str, tick: u64) -> Result<Allocation, CudaMemoryError> {
        self.advance_tick(tick)?;
        self.wait_for_allocation(id, tick)?;
        let allocation = self
            .transfers
            .policy
            .allocation(id)
            .ok_or_else(|| CudaMemoryError::UnknownAllocation(id.to_owned()))?;
        ensure_release_safe(id, allocation.safe_after_tick, tick)?;
        let tier = allocation.tier;
        match tier {
            MemoryTier::Vram => {
                let buffer = self
                    .device
                    .get_mut(id)
                    .ok_or_else(|| CudaMemoryError::WrongTier(id.to_owned()))?;
                let released = release_and_confirm(buffer, &mut self.transfers.policy, id, tick)?;
                self.device.remove(id);
                Ok(released)
            }
            MemoryTier::PinnedHost => {
                let buffer = self
                    .pinned
                    .get_mut(id)
                    .ok_or_else(|| CudaMemoryError::WrongTier(id.to_owned()))?;
                let released = release_and_confirm(buffer, &mut self.transfers.policy, id, tick)?;
                self.pinned.remove(id);
                Ok(released)
            }
            MemoryTier::Host => Err(CudaMemoryError::WrongTier(id.to_owned())),
        }
    }

    fn ensure_new_id(&self, id: &str) -> Result<(), CudaMemoryError> {
        if self.transfers.policy.allocation(id).is_some()
            || self.device.contains_key(id)
            || self.pinned.contains_key(id)
        {
            return Err(MemoryError::DuplicateAllocation(id.to_owned()).into());
        }
        Ok(())
    }

    fn make_room_for_expert(&mut self, bytes: u64, tick: u64) -> Result<(), CudaMemoryError> {
        self.poll_completed_transfers(tick)?;
        // Compute one victim set against both constraints. Independently
        // planning expert and general VRAM admission can over-evict when the
        // two plans choose different candidates.
        let plan = self
            .transfers
            .policy
            .plan_expert_vram_admission(bytes, tick)?;
        for id in plan.evict_ids {
            self.release(&id, tick)?;
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
        // evictions. Polling errors abort admission before any victim is released.
        let plan = self.transfers.plan_admission(tier, bytes, tick)?;
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
        let safe_to_drop = synchronize_stream(&self.transfer_stream).is_ok()
            && synchronize_stream(&self.compute_stream).is_ok();
        if safe_to_drop {
            for transfer in self.transfers.pending.drain(..) {
                transfer.completed.store(true, Ordering::Release);
                drop(transfer.event);
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
            for transfer in self.transfers.pending.drain(..) {
                std::mem::forget(transfer.event);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_policy() -> MemoryOrchestrator {
        MemoryOrchestrator::new(
            MemoryBudgets {
                vram_soft_bytes: 8,
                vram_hard_bytes: 16,
                host_bytes: 32,
                pinned_host_bytes: 8,
                expert_cache_target_bytes: 0,
                expert_cache_max_bytes: 0,
            },
            EvictionPolicy::Lru,
        )
        .unwrap()
    }

    fn guarded_allocation(policy: &mut MemoryOrchestrator, id: &str) {
        policy
            .record_allocation(Allocation {
                id: id.to_owned(),
                tier: MemoryTier::Vram,
                bytes: 4,
                last_used_tick: 0,
                hit_count: 0,
                safe_after_tick: IN_FLIGHT,
                expert: false,
            })
            .unwrap();
    }

    struct FakeEvent {
        complete: bool,
        query_error: Option<i32>,
        sync_error: Option<i32>,
    }

    impl CudaEventOps for FakeEvent {
        fn is_complete(&self) -> Result<bool, CudaError> {
            if let Some(code) = self.query_error {
                Err(CudaError::Driver {
                    operation: "cuEventQuery",
                    code,
                })
            } else {
                Ok(self.complete)
            }
        }

        fn synchronize(&self) -> Result<(), CudaError> {
            if let Some(code) = self.sync_error {
                Err(CudaError::Driver {
                    operation: "cuEventSynchronize",
                    code,
                })
            } else {
                Ok(())
            }
        }
    }

    struct FakeStream(Option<i32>);
    impl CudaStreamOps for FakeStream {
        fn synchronize(&self) -> Result<(), CudaError> {
            match self.0 {
                Some(code) => Err(CudaError::Driver {
                    operation: "cuStreamSynchronize",
                    code,
                }),
                None => Ok(()),
            }
        }
    }

    struct FakeBufferRelease(Option<i32>);
    impl CudaBufferRelease for FakeBufferRelease {
        fn release(&mut self) -> Result<(), CudaError> {
            match self.0 {
                Some(code) => Err(CudaError::Driver {
                    operation: "cuMemFree",
                    code,
                }),
                None => Ok(()),
            }
        }
    }

    #[test]
    fn injected_not_ready_event_keeps_event_and_allocation_guard() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "buffer");
        let completed = Arc::new(AtomicBool::new(false));
        let mut pending = vec![PendingTransfer {
            id: 1,
            allocation_ids: vec!["buffer".to_owned()],
            event: FakeEvent {
                complete: false,
                query_error: None,
                sync_error: None,
            },
            completed: Arc::clone(&completed),
        }];

        assert_eq!(
            poll_pending_transfers(&mut policy, &mut pending, 10).unwrap(),
            0
        );
        assert_eq!(pending.len(), 1);
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(
            policy.allocation("buffer").unwrap().safe_after_tick,
            IN_FLIGHT
        );
    }

    #[test]
    fn injected_query_failure_preserves_pending_event_and_every_guard() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "buffer");
        let completed = Arc::new(AtomicBool::new(false));
        let mut pending = vec![PendingTransfer {
            id: 2,
            allocation_ids: vec!["buffer".to_owned()],
            event: FakeEvent {
                complete: false,
                query_error: Some(719),
                sync_error: None,
            },
            completed: Arc::clone(&completed),
        }];

        assert!(matches!(
            poll_pending_transfers(&mut policy, &mut pending, 10),
            Err(CudaMemoryError::Cuda(CudaError::Driver {
                operation: "cuEventQuery",
                code: 719
            }))
        ));
        assert_eq!(pending.len(), 1);
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(
            policy.allocation("buffer").unwrap().safe_after_tick,
            IN_FLIGHT
        );
    }

    #[test]
    fn injected_event_synchronization_failure_preserves_pending_event_and_guards() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "buffer");
        let completed = Arc::new(AtomicBool::new(false));
        let mut pending = vec![PendingTransfer {
            id: 6,
            allocation_ids: vec!["buffer".to_owned()],
            event: FakeEvent {
                complete: false,
                query_error: None,
                sync_error: Some(722),
            },
            completed: Arc::clone(&completed),
        }];

        assert!(matches!(
            wait_for_pending_allocation(&mut policy, &mut pending, "buffer", 10),
            Err(CudaMemoryError::Cuda(CudaError::Driver {
                operation: "cuEventSynchronize",
                code: 722
            }))
        ));
        assert_eq!(pending.len(), 1);
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(
            policy.allocation("buffer").unwrap().safe_after_tick,
            IN_FLIGHT
        );
    }

    #[test]
    fn successful_event_retirement_lifts_guards_only_after_last_shared_event() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "buffer");
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let mut pending = vec![
            PendingTransfer {
                id: 3,
                allocation_ids: vec!["buffer".to_owned()],
                event: FakeEvent {
                    complete: true,
                    query_error: None,
                    sync_error: None,
                },
                completed: Arc::clone(&first),
            },
            PendingTransfer {
                id: 4,
                allocation_ids: vec!["buffer".to_owned()],
                event: FakeEvent {
                    complete: false,
                    query_error: None,
                    sync_error: None,
                },
                completed: Arc::clone(&second),
            },
        ];

        assert_eq!(
            poll_pending_transfers(&mut policy, &mut pending, 10).unwrap(),
            1
        );
        assert!(first.load(Ordering::Acquire));
        assert!(!second.load(Ordering::Acquire));
        assert_eq!(pending.len(), 1);
        assert_eq!(
            policy.allocation("buffer").unwrap().safe_after_tick,
            IN_FLIGHT
        );

        pending[0].event = FakeEvent {
            complete: true,
            query_error: None,
            sync_error: None,
        };
        assert_eq!(
            poll_pending_transfers(&mut policy, &mut pending, 11).unwrap(),
            1
        );
        assert!(second.load(Ordering::Acquire));
        assert!(pending.is_empty());
        assert_eq!(policy.allocation("buffer").unwrap().safe_after_tick, 11);
    }

    #[test]
    fn unrelated_event_query_error_fails_closed_before_admission_planning() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "unrelated");
        let completed = Arc::new(AtomicBool::new(false));
        let mut pending = vec![PendingTransfer {
            id: 5,
            allocation_ids: vec!["unrelated".to_owned()],
            event: FakeEvent {
                complete: false,
                query_error: Some(999),
                sync_error: None,
            },
            completed,
        }];

        let admission =
            plan_admission_after_poll(&mut policy, &mut pending, MemoryTier::Vram, 8, 10)
                .map(|plan| plan.evict_ids);
        assert!(matches!(admission, Err(CudaMemoryError::Cuda(_))));
        assert_eq!(pending.len(), 1);
        assert!(policy.allocation("unrelated").is_some());
        assert_eq!(
            policy.allocation("unrelated").unwrap().safe_after_tick,
            IN_FLIGHT
        );
    }

    #[test]
    fn failed_backend_release_retains_policy_ownership_until_retry_succeeds() {
        let mut policy = fake_policy();
        policy
            .record_allocation(Allocation {
                id: "buffer".to_owned(),
                tier: MemoryTier::Vram,
                bytes: 4,
                last_used_tick: 0,
                hit_count: 0,
                safe_after_tick: 0,
                expert: false,
            })
            .unwrap();
        let mut backend = FakeBufferRelease(Some(700));

        assert!(matches!(
            release_and_confirm(&mut backend, &mut policy, "buffer", 1),
            Err(CudaMemoryError::Cuda(CudaError::Driver {
                operation: "cuMemFree",
                code: 700
            }))
        ));
        assert!(policy.allocation("buffer").is_some());
        assert_eq!(policy.usage_bytes(MemoryTier::Vram), 4);

        backend.0 = None;
        release_and_confirm(&mut backend, &mut policy, "buffer", 1).unwrap();
        assert!(policy.allocation("buffer").is_none());
        assert_eq!(policy.usage_bytes(MemoryTier::Vram), 0);
    }

    #[test]
    fn injected_stream_synchronization_error_is_not_treated_as_success() {
        assert!(matches!(
            synchronize_stream(&FakeStream(Some(901))),
            Err(CudaError::Driver {
                operation: "cuStreamSynchronize",
                code: 901
            })
        ));
        assert!(synchronize_stream(&FakeStream(None)).is_ok());
    }

    // These tests exercise the exact transfer-state component owned by
    // CudaMemoryManager, rather than calling the free helper functions directly.
    #[test]
    fn manager_tracker_poll_and_admission_share_the_production_state() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "guarded");
        let completed = Arc::new(AtomicBool::new(false));
        let mut tracker = CudaTransferTracker {
            policy,
            pending: vec![PendingTransfer {
                id: 41,
                allocation_ids: vec!["guarded".to_owned()],
                event: FakeEvent {
                    complete: false,
                    query_error: None,
                    sync_error: None,
                },
                completed: Arc::clone(&completed),
            }],
        };

        // NOT_READY must preserve the manager-owned pending record and guard.
        assert_eq!(tracker.poll(10).unwrap(), 0);
        assert_eq!(tracker.pending.len(), 1);
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(
            tracker
                .policy
                .allocation("guarded")
                .unwrap()
                .safe_after_tick,
            IN_FLIGHT
        );

        // A query failure through the manager tracker must abort admission
        // before a plan can release any victim.
        tracker.pending[0].event.query_error = Some(719);
        assert!(matches!(
            tracker.plan_admission(MemoryTier::Vram, 8, 11),
            Err(CudaMemoryError::Cuda(CudaError::Driver {
                operation: "cuEventQuery",
                code: 719
            }))
        ));
        assert_eq!(tracker.pending.len(), 1);
        assert!(tracker.policy.allocation("guarded").is_some());
        assert_eq!(
            tracker
                .policy
                .allocation("guarded")
                .unwrap()
                .safe_after_tick,
            IN_FLIGHT
        );
        assert!(!completed.load(Ordering::Acquire));
    }

    #[test]
    fn manager_tracker_wait_and_shared_event_retirement_preserve_guards() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "shared");
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let mut tracker = CudaTransferTracker {
            policy,
            pending: vec![
                PendingTransfer {
                    id: 42,
                    allocation_ids: vec!["shared".to_owned()],
                    event: FakeEvent {
                        complete: true,
                        query_error: None,
                        sync_error: None,
                    },
                    completed: Arc::clone(&first),
                },
                PendingTransfer {
                    id: 43,
                    allocation_ids: vec!["shared".to_owned()],
                    event: FakeEvent {
                        complete: false,
                        query_error: None,
                        sync_error: None,
                    },
                    completed: Arc::clone(&second),
                },
            ],
        };

        assert_eq!(tracker.poll(10).unwrap(), 1);
        assert!(first.load(Ordering::Acquire));
        assert!(!second.load(Ordering::Acquire));
        assert_eq!(tracker.pending.len(), 1);
        assert_eq!(
            tracker.policy.allocation("shared").unwrap().safe_after_tick,
            IN_FLIGHT
        );

        tracker.pending[0].event.complete = true;
        assert_eq!(tracker.poll(11).unwrap(), 1);
        assert!(second.load(Ordering::Acquire));
        assert!(tracker.pending.is_empty());
        assert_eq!(
            tracker.policy.allocation("shared").unwrap().safe_after_tick,
            11
        );
    }

    #[test]
    fn manager_tracker_wait_failure_keeps_event_and_every_guard() {
        let mut policy = fake_policy();
        guarded_allocation(&mut policy, "buffer");
        let completed = Arc::new(AtomicBool::new(false));
        let mut tracker = CudaTransferTracker {
            policy,
            pending: vec![PendingTransfer {
                id: 44,
                allocation_ids: vec!["buffer".to_owned()],
                event: FakeEvent {
                    complete: false,
                    query_error: None,
                    sync_error: Some(722),
                },
                completed: Arc::clone(&completed),
            }],
        };

        assert!(matches!(
            tracker.wait_for_allocation("buffer", 10),
            Err(CudaMemoryError::Cuda(CudaError::Driver {
                operation: "cuEventSynchronize",
                code: 722
            }))
        ));
        assert_eq!(tracker.pending.len(), 1);
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(
            tracker.policy.allocation("buffer").unwrap().safe_after_tick,
            IN_FLIGHT
        );
    }

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
    fn hardware_poll_retires_completed_transfer_before_admission_without_wait() {
        use std::time::{Duration, Instant};

        let budgets = MemoryBudgets {
            vram_soft_bytes: 1024 * 1024,
            vram_hard_bytes: 1024 * 1024,
            host_bytes: 2 * 1024 * 1024,
            pinned_host_bytes: 64 * 1024,
            expert_cache_target_bytes: 0,
            expert_cache_max_bytes: 0,
        };
        let mut manager =
            CudaMemoryManager::new(0, budgets, EvictionPolicy::Lru).expect("CUDA memory manager");
        manager
            .allocate_pinned("host", 4096, 1)
            .expect("pinned allocation");
        manager
            .allocate_device("victim", 768 * 1024, 2, false)
            .expect("victim allocation");
        let expected = vec![0x5a_u8; 4096];
        manager
            .write_pinned("host", 0, &expected, 3)
            .expect("pinned write");
        manager
            .copy_h2d("host", "victim", expected.len(), 4)
            .expect("H2D enqueue");

        // Exercise the actual manager-level polling API until the event is
        // positively completed and retired. No token wait or allocation wait
        // is permitted in this scenario.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut tick = 5_u64;
        let retired = loop {
            let count = manager
                .poll_completed_transfers(tick)
                .expect("non-blocking transfer poll");
            tick += 1;
            if count == 1 {
                break count;
            }
            assert!(
                Instant::now() < deadline,
                "completed H2D event was not retired within the deadline"
            );
            std::thread::yield_now();
        };
        assert_eq!(retired, 1);
        assert_eq!(manager.usage_bytes(MemoryTier::Vram), 768 * 1024);

        // Admission pressure must now evict the completed victim without an
        // explicit wait_for_transfer or wait_for_allocation call.
        manager
            .allocate_device("replacement", 512 * 1024, tick, false)
            .expect("admission after poll-driven retirement");
        assert_eq!(manager.usage_bytes(MemoryTier::Vram), 512 * 1024);
        assert!(matches!(
            manager.release("victim", tick + 1),
            Err(CudaMemoryError::UnknownAllocation(id)) if id == "victim"
        ));
        manager
            .release("replacement", tick + 1)
            .expect("replacement release");
        manager.release("host", tick + 1).expect("pinned release");
        println!(
            "M5-HW-POLL-01 retired={retired}; admission evicted completed victim without explicit wait"
        );
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
