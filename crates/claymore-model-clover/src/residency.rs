use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use claymore_checkpoint::safetensors::{SafetensorsError, SafetensorsFile, WeightDType};
use claymore_core::MemoryTier;
use claymore_device::{CpuDevice, Device};
use claymore_memory::{DeviceMemoryError, DeviceMemoryManager, EvictionPolicy, MemoryBudgets};
use claymore_scheduler::{Scheduler, SchedulerError, TaskId, TaskKind, TaskPriority, TaskSpec};
use claymore_tensor::Bf16;
use thiserror::Error;

use crate::{
    CloverConfig, CloverExpertWeights, CloverLoadPlan, CloverRoutingPlan,
    CloverSharedExpertWeights, ShardLoadPlan, TensorLoadPlan,
};

const PREFIX_BYTES: u64 = 8;
const BF16_BYTES: usize = 2;
const MAX_EXPERT_CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CloverExpertCacheError {
    #[error("invalid expert dimensions: {0}")]
    InvalidDimensions(&'static str),
    #[error("missing expert tensor: {0}")]
    MissingTensor(String),
    #[error("invalid expert tensor metadata: {0}")]
    InvalidTensor(String),
    #[error("load-plan chunk exceeds maximum chunk size")]
    ChunkTooLarge,
    #[error("load-plan chunks do not cover the expert slice")]
    IncompleteChunkCoverage,
    #[error("expert streaming was cancelled")]
    Cancelled,
    #[error("expert allocation exists but its chunk stream is incomplete")]
    IncompleteLoad,
    #[error("expert byte offset or size overflow")]
    SizeOverflow,
    #[error("expert stream I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("expert Safetensors shard validation failed: {0}")]
    Safetensors(#[from] SafetensorsError),
    #[error("expert shard changed after checkpoint inspection: {0}")]
    StaleShard(String),
    #[error("shared memory manager failed: {0}")]
    Memory(#[from] DeviceMemoryError),
    #[error("expert load failed ({cause}) and allocation rollback also failed: {rollback}")]
    RollbackFailed {
        cause: String,
        rollback: DeviceMemoryError,
    },
}

/// Scheduler-safe task: identifiers and byte ranges only, never tensor buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloverExpertLoadTask {
    pub expert_id: String,
    pub shard_name: String,
    pub tensor_name: String,
    /// Offset within the named tensor; never trusted without revalidating its shard.
    pub tensor_offset: u64,
    /// Legacy absolute offset retained for diagnostics; transfer uses tensor_offset.
    pub file_offset: u64,
    pub target_offset: usize,
    pub length_bytes: usize,
}

#[derive(Debug, Error)]
pub enum CloverExpertScheduleError {
    #[error("cannot schedule an expert load without chunk tasks")]
    EmptyPlan,
    #[error("one scheduler prefetch task cannot contain chunks for multiple experts")]
    MixedExpertPlan,
    #[error("shared scheduler rejected the expert load graph: {0}")]
    Scheduler(#[from] SchedulerError),
}

/// Scheduler graph plus byte-range descriptors owned by the transfer executor.
/// The scheduler stores only the prefetch/compute task ids and their dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloverExpertLoadSchedule {
    load_task: TaskId,
    chunks: Vec<CloverExpertLoadTask>,
}

impl CloverExpertLoadSchedule {
    pub fn submit(
        scheduler: &mut Scheduler,
        chunks: Vec<CloverExpertLoadTask>,
        priority: TaskPriority,
    ) -> Result<Self, CloverExpertScheduleError> {
        Self::submit_block(scheduler, vec![chunks], priority)
    }

    /// Submit one block-level prefetch task. Its external chunk descriptors may
    /// cover several routed experts and the shared expert; the scheduler still
    /// stores only one task id, never tensor or buffer payloads.
    pub fn submit_block(
        scheduler: &mut Scheduler,
        chunk_groups: Vec<Vec<CloverExpertLoadTask>>,
        priority: TaskPriority,
    ) -> Result<Self, CloverExpertScheduleError> {
        if chunk_groups.is_empty() {
            return Err(CloverExpertScheduleError::EmptyPlan);
        }
        let mut chunks = Vec::new();
        for group in chunk_groups {
            let first = group.first().ok_or(CloverExpertScheduleError::EmptyPlan)?;
            if group.iter().any(|chunk| chunk.expert_id != first.expert_id) {
                return Err(CloverExpertScheduleError::MixedExpertPlan);
            }
            chunks.extend(group);
        }
        if chunks.is_empty() {
            return Err(CloverExpertScheduleError::EmptyPlan);
        }
        let load_task =
            scheduler.submit(TaskSpec::new(TaskKind::Prefetch).with_priority(priority))?;
        Ok(Self { load_task, chunks })
    }

    pub fn load_task(&self) -> TaskId {
        self.load_task
    }

    pub fn chunks(&self) -> &[CloverExpertLoadTask] {
        &self.chunks
    }

    pub fn submit_compute(
        &self,
        scheduler: &mut Scheduler,
        priority: TaskPriority,
    ) -> Result<TaskId, CloverExpertScheduleError> {
        let spec = TaskSpec::new(TaskKind::Compute)
            .with_priority(priority)
            .depends_on(self.load_task);
        Ok(scheduler.submit(spec)?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloverExpertGeometry {
    pub num_experts: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl CloverExpertGeometry {
    pub fn from_config(config: &CloverConfig) -> Self {
        Self {
            num_experts: config.num_experts,
            hidden_size: config.hidden_size,
            intermediate_size: config.moe_intermediate_size,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloverExpertPrefetchRequest {
    pub layer: usize,
    pub expert: usize,
    pub priority: i32,
}

pub struct CloverExpertLoadRequest<'a> {
    pub checkpoint_root: &'a Path,
    pub plan: &'a CloverLoadPlan,
    pub layer: usize,
    pub expert: usize,
    pub geometry: CloverExpertGeometry,
    pub tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloverSharedExpertGeometry {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl CloverSharedExpertGeometry {
    pub fn from_config(config: &CloverConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.shared_expert_intermediate_size,
        }
    }
}

pub struct CloverSharedExpertLoadRequest<'a> {
    pub checkpoint_root: &'a Path,
    pub plan: &'a CloverLoadPlan,
    pub layer: usize,
    pub geometry: CloverSharedExpertGeometry,
    pub tick: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CloverExpertCacheMetrics {
    pub expert_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub resident_experts: u64,
    pub evictions: u64,
    pub bytes_transferred: u64,
    pub failed_loads: u64,
    pub cancelled_loads: u64,
    pub admission_failures: u64,
    pub wait_count: u64,
    pub in_flight_marks: u64,
}

pub struct CloverExpertCache<D: Device> {
    manager: DeviceMemoryManager<D>,
    known_residents: BTreeSet<String>,
    ready_residents: BTreeSet<String>,
    metrics: CloverExpertCacheMetrics,
}

impl<D: Device> CloverExpertCache<D> {
    pub fn new(
        device: D,
        budgets: MemoryBudgets,
        eviction: EvictionPolicy,
    ) -> Result<Self, CloverExpertCacheError> {
        Ok(Self {
            manager: DeviceMemoryManager::new(device, budgets, eviction)?,
            known_residents: BTreeSet::new(),
            ready_residents: BTreeSet::new(),
            metrics: CloverExpertCacheMetrics::default(),
        })
    }

    pub fn metrics(&self) -> CloverExpertCacheMetrics {
        self.metrics
    }

    pub fn usage_bytes(&self) -> u64 {
        self.manager.usage_bytes(MemoryTier::Vram)
    }

    pub fn is_resident(&self, layer: usize, expert: usize) -> bool {
        let id = allocation_id(layer, expert);
        self.ready_residents.contains(&id) && self.manager.allocation(&id).is_some()
    }

    pub fn load_tasks(
        plan: &CloverLoadPlan,
        layer: usize,
        expert: usize,
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Vec<CloverExpertLoadTask>, CloverExpertCacheError> {
        build_load_tasks(
            plan,
            layer,
            expert,
            num_experts,
            hidden_size,
            intermediate_size,
        )
    }

    pub fn shared_load_tasks(
        plan: &CloverLoadPlan,
        layer: usize,
        geometry: CloverSharedExpertGeometry,
    ) -> Result<Vec<CloverExpertLoadTask>, CloverExpertCacheError> {
        build_shared_expert_load_tasks(
            plan,
            layer,
            geometry.hidden_size,
            geometry.intermediate_size,
        )
    }

    pub fn ensure_resident(
        &mut self,
        request: CloverExpertLoadRequest<'_>,
    ) -> Result<String, CloverExpertCacheError> {
        self.ensure_resident_with_cancel(request, || false)
    }

    /// Cancellation is checked between bounded chunk transfers. Because each
    /// Device::write is synchronous, a cancelled partial load can be safely
    /// released at the current logical tick.
    pub fn ensure_resident_with_cancel(
        &mut self,
        request: CloverExpertLoadRequest<'_>,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<String, CloverExpertCacheError> {
        let id = allocation_id(request.layer, request.expert);
        let tasks = build_load_tasks(
            request.plan,
            request.layer,
            request.expert,
            request.geometry.num_experts,
            request.geometry.hidden_size,
            request.geometry.intermediate_size,
        )?;
        let bytes = allocation_bytes(
            request.geometry.hidden_size,
            request.geometry.intermediate_size,
        )?;
        self.ensure_tasks_resident(
            id,
            request.checkpoint_root,
            request.plan,
            tasks,
            bytes,
            request.tick,
            is_cancelled,
        )
    }

    pub fn ensure_shared_expert_resident(
        &mut self,
        request: CloverSharedExpertLoadRequest<'_>,
    ) -> Result<String, CloverExpertCacheError> {
        self.ensure_shared_expert_resident_with_cancel(request, || false)
    }

    pub fn ensure_shared_expert_resident_with_cancel(
        &mut self,
        request: CloverSharedExpertLoadRequest<'_>,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<String, CloverExpertCacheError> {
        let id = shared_allocation_id(request.layer);
        let tasks = build_shared_expert_load_tasks(
            request.plan,
            request.layer,
            request.geometry.hidden_size,
            request.geometry.intermediate_size,
        )?;
        let bytes = shared_allocation_bytes(
            request.geometry.hidden_size,
            request.geometry.intermediate_size,
        )?;
        self.ensure_tasks_resident(
            id,
            request.checkpoint_root,
            request.plan,
            tasks,
            bytes,
            request.tick,
            is_cancelled,
        )
    }

    fn ensure_tasks_resident(
        &mut self,
        id: String,
        root: &Path,
        plan: &CloverLoadPlan,
        tasks: Vec<CloverExpertLoadTask>,
        bytes: u64,
        tick: u64,
        mut is_cancelled: impl FnMut() -> bool,
    ) -> Result<String, CloverExpertCacheError> {
        self.metrics.expert_requests = self.metrics.expert_requests.saturating_add(1);
        if is_cancelled() {
            self.metrics.cancelled_loads = self.metrics.cancelled_loads.saturating_add(1);
            return Err(CloverExpertCacheError::Cancelled);
        }

        if self.manager.allocation(&id).is_some() {
            if !self.ready_residents.contains(&id) {
                // A previous failed rollback may have retained its allocation
                // for safety. Retry the release before admitting a fresh load.
                self.manager.release(&id, tick).map_err(|rollback| {
                    CloverExpertCacheError::RollbackFailed {
                        cause: "previous expert load left an incomplete allocation".to_owned(),
                        rollback,
                    }
                })?;
                self.known_residents.remove(&id);
                self.ready_residents.remove(&id);
                self.metrics.resident_experts = self.ready_residents.len() as u64;
            } else {
                let _ = self.manager.read(&id, 0, 1, tick)?;
                self.metrics.cache_hits = self.metrics.cache_hits.saturating_add(1);
                return Ok(id);
            }
        }

        self.metrics.cache_misses = self.metrics.cache_misses.saturating_add(1);
        self.prune_evicted();
        if let Err(error) = self
            .manager
            .allocate(&id, MemoryTier::Vram, bytes, tick, true)
        {
            self.metrics.admission_failures = self.metrics.admission_failures.saturating_add(1);
            return Err(error.into());
        }
        self.known_residents.insert(id.clone());
        self.prune_evicted();

        let mut open_shards = BTreeMap::new();
        for task in &tasks {
            if is_cancelled() {
                self.metrics.cancelled_loads = self.metrics.cancelled_loads.saturating_add(1);
                if let Err(rollback) = self.rollback_allocation(&id, tick) {
                    return Err(CloverExpertCacheError::RollbackFailed {
                        cause: CloverExpertCacheError::Cancelled.to_string(),
                        rollback,
                    });
                }
                return Err(CloverExpertCacheError::Cancelled);
            }
            if !open_shards.contains_key(&task.shard_name) {
                let expected = match plan
                    .shards
                    .iter()
                    .find(|shard| shard.name == task.shard_name)
                {
                    Some(shard) => shard,
                    None => {
                        self.metrics.failed_loads = self.metrics.failed_loads.saturating_add(1);
                        if let Err(rollback) = self.rollback_allocation(&id, tick) {
                            return Err(CloverExpertCacheError::RollbackFailed {
                                cause: CloverExpertCacheError::StaleShard(
                                    task.shard_name.clone(),
                                )
                                .to_string(),
                                rollback,
                            });
                        }
                        return Err(CloverExpertCacheError::StaleShard(
                            task.shard_name.clone(),
                        ));
                    }
                };
                match open_planned_shard(root, expected) {
                    Ok(opened) => {
                        open_shards.insert(task.shard_name.clone(), opened);
                    }
                    Err(error) => {
                        self.metrics.failed_loads = self.metrics.failed_loads.saturating_add(1);
                        if let Err(rollback) = self.rollback_allocation(&id, tick) {
                            return Err(CloverExpertCacheError::RollbackFailed {
                                cause: error.to_string(),
                                rollback,
                            });
                        }
                        return Err(error);
                    }
                }
            }
            let shard = open_shards
                .get(&task.shard_name)
                .expect("validated shard was inserted");
            match stream_task(&mut self.manager, shard, &id, task, tick) {
                Ok(()) => {
                    self.metrics.bytes_transferred = self
                        .metrics
                        .bytes_transferred
                        .saturating_add(task.length_bytes as u64);
                }
                Err(error) => {
                    self.metrics.failed_loads = self.metrics.failed_loads.saturating_add(1);
                    if let Err(rollback) = self.rollback_allocation(&id, tick) {
                        return Err(CloverExpertCacheError::RollbackFailed {
                            cause: error.to_string(),
                            rollback,
                        });
                    }
                    return Err(error);
                }
            }
        }
        self.ready_residents.insert(id.clone());
        self.metrics.resident_experts = self.ready_residents.len() as u64;
        Ok(id)
    }

    pub fn is_shared_expert_resident(&self, layer: usize) -> bool {
        let id = shared_allocation_id(layer);
        self.ready_residents.contains(&id) && self.manager.allocation(&id).is_some()
    }

    pub fn mark_shared_expert_in_flight_until(
        &mut self,
        layer: usize,
        safe_after_tick: u64,
        current_tick: u64,
    ) -> Result<(), CloverExpertCacheError> {
        let id = shared_allocation_id(layer);
        if !self.ready_residents.contains(&id) {
            return Err(CloverExpertCacheError::IncompleteLoad);
        }
        self.manager
            .mark_in_flight_until(&id, safe_after_tick, current_tick)?;
        self.metrics.in_flight_marks = self.metrics.in_flight_marks.saturating_add(1);
        Ok(())
    }

    pub fn release_shared_expert(
        &mut self,
        layer: usize,
        tick: u64,
    ) -> Result<(), CloverExpertCacheError> {
        let id = shared_allocation_id(layer);
        self.manager.release(&id, tick)?;
        self.known_residents.remove(&id);
        self.ready_residents.remove(&id);
        self.metrics.resident_experts = self.ready_residents.len() as u64;
        Ok(())
    }

    /// Load experts by descending priority, then ascending layer/expert id.
    /// The stable order is independent of the caller's hash/map iteration order.
    pub fn prefetch_ordered(
        &mut self,
        checkpoint_root: &Path,
        plan: &CloverLoadPlan,
        geometry: CloverExpertGeometry,
        requests: &[CloverExpertPrefetchRequest],
        tick: u64,
    ) -> Result<Vec<String>, CloverExpertCacheError> {
        let ordered = ordered_prefetch_requests(requests);
        let mut loaded = Vec::with_capacity(ordered.len());
        for request in ordered {
            loaded.push(self.ensure_resident(CloverExpertLoadRequest {
                checkpoint_root,
                plan,
                layer: request.layer,
                expert: request.expert,
                geometry,
                tick,
            })?);
        }
        Ok(loaded)
    }

    fn rollback_allocation(&mut self, id: &str, tick: u64) -> Result<(), DeviceMemoryError> {
        // DeviceMemoryManager retains the allocation and its accounting when
        // release fails. Keep the resident bookkeeping intact as well; the
        // next request retries release instead of becoming permanently stuck.
        self.manager.release(id, tick)?;
        self.known_residents.remove(id);
        self.ready_residents.remove(id);
        self.metrics.resident_experts = self.ready_residents.len() as u64;
        Ok(())
    }

    pub fn mark_in_flight_until(
        &mut self,
        layer: usize,
        expert: usize,
        safe_after_tick: u64,
        current_tick: u64,
    ) -> Result<(), CloverExpertCacheError> {
        let id = allocation_id(layer, expert);
        if !self.ready_residents.contains(&id) {
            return Err(CloverExpertCacheError::IncompleteLoad);
        }
        self.manager
            .mark_in_flight_until(&id, safe_after_tick, current_tick)?;
        self.metrics.in_flight_marks = self.metrics.in_flight_marks.saturating_add(1);
        Ok(())
    }

    pub fn release(
        &mut self,
        layer: usize,
        expert: usize,
        tick: u64,
    ) -> Result<(), CloverExpertCacheError> {
        let id = allocation_id(layer, expert);
        self.manager.release(&id, tick)?;
        self.known_residents.remove(&id);
        self.ready_residents.remove(&id);
        self.metrics.resident_experts = self.ready_residents.len() as u64;
        Ok(())
    }

    pub fn synchronize(&mut self) -> Result<(), CloverExpertCacheError> {
        self.manager.synchronize()?;
        self.metrics.wait_count = self.metrics.wait_count.saturating_add(1);
        Ok(())
    }

    fn prune_evicted(&mut self) {
        let before = self.known_residents.len();
        self.known_residents
            .retain(|id| self.manager.allocation(id).is_some());
        self.ready_residents
            .retain(|id| self.manager.allocation(id).is_some());
        let removed = before.saturating_sub(self.known_residents.len()) as u64;
        self.metrics.evictions = self.metrics.evictions.saturating_add(removed);
        self.metrics.resident_experts = self.ready_residents.len() as u64;
    }
}

/// Convert the block coreset (or routed top-k when no coreset is active)
/// into a deduplicated expert prefetch set for one layer.
pub fn prefetch_requests_for_routing(
    routing: &CloverRoutingPlan,
    layer: usize,
    priority: i32,
) -> Vec<CloverExpertPrefetchRequest> {
    let mut experts = BTreeSet::new();
    if let Some(coresets) = &routing.coreset {
        for coreset in coresets {
            experts.extend(coreset.iter().copied());
        }
    }
    // Invalid tokens are intentionally not coreset-masked. Include their
    // actually selected experts as well; zero-weight masked experts stay out.
    for route in &routing.routes {
        for (&expert, &weight) in route.expert_ids.iter().zip(&route.weights) {
            if weight > 0.0 {
                experts.insert(expert);
            }
        }
    }
    experts
        .into_iter()
        .map(|expert| CloverExpertPrefetchRequest {
            layer,
            expert,
            priority,
        })
        .collect()
}

fn ordered_prefetch_requests(
    requests: &[CloverExpertPrefetchRequest],
) -> Vec<CloverExpertPrefetchRequest> {
    let mut ordered = requests.to_vec();
    ordered.sort_by(|left, right| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.layer.cmp(&right.layer))
            .then_with(|| left.expert.cmp(&right.expert))
    });
    ordered
}

fn shared_allocation_id(layer: usize) -> String {
    format!("clover.layer.{layer}.shared_expert")
}

fn shared_allocation_bytes(
    hidden: usize,
    intermediate: usize,
) -> Result<u64, CloverExpertCacheError> {
    let projections = intermediate
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(BF16_BYTES))
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let gate = hidden
        .checked_mul(BF16_BYTES)
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    u64::try_from(
        projections
            .checked_add(gate)
            .ok_or(CloverExpertCacheError::SizeOverflow)?,
    )
    .map_err(|_| CloverExpertCacheError::SizeOverflow)
}

fn build_shared_expert_load_tasks(
    plan: &CloverLoadPlan,
    layer: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<Vec<CloverExpertLoadTask>, CloverExpertCacheError> {
    if hidden == 0
        || intermediate == 0
        || plan.max_chunk_bytes == 0
        || plan.max_chunk_bytes > MAX_EXPERT_CHUNK_BYTES
    {
        return Err(CloverExpertCacheError::InvalidDimensions(
            "shared expert dimensions and maximum chunk size must be valid",
        ));
    }
    let mlp = format!("model.layers.{layer}.mlp");
    let gate_name = format!("{mlp}.shared_expert.gate_proj.weight");
    let up_name = format!("{mlp}.shared_expert.up_proj.weight");
    let down_name = format!("{mlp}.shared_expert.down_proj.weight");
    let shared_gate_name = format!("{mlp}.shared_expert_gate.weight");
    let projection_bytes = intermediate
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(BF16_BYTES))
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let down_bytes = hidden
        .checked_mul(intermediate)
        .and_then(|n| n.checked_mul(BF16_BYTES))
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let gate_bytes = hidden
        .checked_mul(BF16_BYTES)
        .ok_or(CloverExpertCacheError::SizeOverflow)?;

    let gate = find_tensor(plan, &gate_name)?;
    let up = find_tensor(plan, &up_name)?;
    let down = find_tensor(plan, &down_name)?;
    let shared_gate = find_tensor(plan, &shared_gate_name)?;
    validate_tensor(
        gate.1,
        &[intermediate, hidden],
        (hidden * BF16_BYTES) as u64,
        plan,
    )?;
    validate_tensor(
        up.1,
        &[intermediate, hidden],
        (hidden * BF16_BYTES) as u64,
        plan,
    )?;
    validate_tensor(
        down.1,
        &[hidden, intermediate],
        (intermediate * BF16_BYTES) as u64,
        plan,
    )?;
    validate_tensor(shared_gate.1, &[1, hidden], gate_bytes as u64, plan)?;

    let id = shared_allocation_id(layer);
    let mut tasks = Vec::new();
    let mut target = 0usize;
    for ((shard, tensor), name, bytes) in [
        (gate, gate_name.as_str(), projection_bytes),
        (up, up_name.as_str(), projection_bytes),
        (down, down_name.as_str(), down_bytes),
        (shared_gate, shared_gate_name.as_str(), gate_bytes),
    ] {
        append_tasks(&mut tasks, (shard, tensor), 0, bytes, target, name, &id)?;
        target = target
            .checked_add(bytes)
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
    }
    Ok(tasks)
}

fn allocation_id(layer: usize, expert: usize) -> String {
    format!("clover.layer.{layer}.expert.{expert}")
}

fn allocation_bytes(hidden: usize, intermediate: usize) -> Result<u64, CloverExpertCacheError> {
    let gate_up = intermediate
        .checked_mul(2)
        .and_then(|n| n.checked_mul(hidden))
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let down = hidden
        .checked_mul(intermediate)
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let total = gate_up
        .checked_add(down)
        .and_then(|n| n.checked_mul(BF16_BYTES))
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    u64::try_from(total).map_err(|_| CloverExpertCacheError::SizeOverflow)
}

fn build_load_tasks(
    plan: &CloverLoadPlan,
    layer: usize,
    expert: usize,
    experts: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<Vec<CloverExpertLoadTask>, CloverExpertCacheError> {
    if experts == 0
        || expert >= experts
        || hidden == 0
        || intermediate == 0
        || plan.max_chunk_bytes == 0
        || plan.max_chunk_bytes > MAX_EXPERT_CHUNK_BYTES
    {
        return Err(CloverExpertCacheError::InvalidDimensions(
            "invalid model or expert dimensions",
        ));
    }
    let gate_name = format!("model.layers.{layer}.mlp.experts.gate_up_proj");
    let down_name = format!("model.layers.{layer}.mlp.experts.down_proj");
    let gate_bytes = intermediate
        .checked_mul(2)
        .and_then(|n| n.checked_mul(hidden))
        .and_then(|n| n.checked_mul(BF16_BYTES))
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let down_bytes = hidden
        .checked_mul(intermediate)
        .and_then(|n| n.checked_mul(BF16_BYTES))
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let doubled_intermediate = intermediate
        .checked_mul(2)
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let gate_shape = vec![experts, doubled_intermediate, hidden];
    let down_shape = vec![experts, hidden, intermediate];
    let (gate_shard, gate) = find_tensor(plan, &gate_name)?;
    let (down_shard, down) = find_tensor(plan, &down_name)?;
    validate_tensor(gate, &gate_shape, gate_bytes as u64, plan)?;
    validate_tensor(down, &down_shape, down_bytes as u64, plan)?;

    let id = allocation_id(layer, expert);
    let mut tasks = Vec::new();
    append_tasks(
        &mut tasks,
        (gate_shard, gate),
        expert,
        gate_bytes,
        0,
        &gate_name,
        &id,
    )?;
    append_tasks(
        &mut tasks,
        (down_shard, down),
        expert,
        down_bytes,
        gate_bytes,
        &down_name,
        &id,
    )?;
    Ok(tasks)
}

fn find_tensor<'a>(
    plan: &'a CloverLoadPlan,
    name: &str,
) -> Result<(&'a ShardLoadPlan, &'a TensorLoadPlan), CloverExpertCacheError> {
    for shard in &plan.shards {
        if let Some(tensor) = shard.tensors.iter().find(|t| t.name == name) {
            return Ok((shard, tensor));
        }
    }
    Err(CloverExpertCacheError::MissingTensor(name.to_owned()))
}

fn validate_tensor(
    item: &TensorLoadPlan,
    shape: &[usize],
    expert_bytes: u64,
    plan: &CloverLoadPlan,
) -> Result<(), CloverExpertCacheError> {
    if item.dtype != WeightDType::BF16
        || item.shape != shape
        || item.byte_len
            != expert_bytes
                .checked_mul(shape[0] as u64)
                .ok_or(CloverExpertCacheError::SizeOverflow)?
    {
        return Err(CloverExpertCacheError::InvalidTensor(item.name.clone()));
    }
    if item
        .chunks
        .iter()
        .any(|c| c.length_bytes == 0 || c.length_bytes > plan.max_chunk_bytes)
    {
        return Err(CloverExpertCacheError::ChunkTooLarge);
    }
    Ok(())
}

fn append_tasks(
    tasks: &mut Vec<CloverExpertLoadTask>,
    (shard, tensor): (&ShardLoadPlan, &TensorLoadPlan),
    expert: usize,
    expert_bytes: usize,
    target_base: usize,
    tensor_name: &str,
    id: &str,
) -> Result<(), CloverExpertCacheError> {
    let start = (expert as u64)
        .checked_mul(expert_bytes as u64)
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let end = start
        .checked_add(expert_bytes as u64)
        .ok_or(CloverExpertCacheError::SizeOverflow)?;
    let mut cursor = start;

    for chunk in &tensor.chunks {
        let chunk_start = chunk.tensor_offset;
        let chunk_end = chunk_start
            .checked_add(chunk.length_bytes as u64)
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let first = start.max(chunk_start);
        let last = end.min(chunk_end);
        if first >= last {
            continue;
        }
        if first != cursor {
            return Err(CloverExpertCacheError::IncompleteChunkCoverage);
        }
        let delta = first - chunk_start;
        let shard_offset = chunk
            .shard_payload_offset
            .checked_add(delta)
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let file_offset = PREFIX_BYTES
            .checked_add(shard.header_bytes)
            .and_then(|v| v.checked_add(shard_offset))
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let target_offset = target_base
            .checked_add(
                usize::try_from(first - start).map_err(|_| CloverExpertCacheError::SizeOverflow)?,
            )
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let length_bytes =
            usize::try_from(last - first).map_err(|_| CloverExpertCacheError::SizeOverflow)?;
        let payload_end = shard_offset
            .checked_add(length_bytes as u64)
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let file_end = file_offset
            .checked_add(length_bytes as u64)
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        if payload_end > shard.payload_bytes || file_end > shard.file_bytes {
            return Err(CloverExpertCacheError::InvalidTensor(
                tensor_name.to_owned(),
            ));
        }
        tasks.push(CloverExpertLoadTask {
            expert_id: id.to_owned(),
            shard_name: shard.name.clone(),
            tensor_name: tensor_name.to_owned(),
            tensor_offset: first,
            file_offset,
            target_offset,
            length_bytes,
        });
        cursor = last;
    }
    if cursor != end {
        return Err(CloverExpertCacheError::IncompleteChunkCoverage);
    }
    Ok(())
}

impl CloverExpertCache<CpuDevice> {
    /// Materialize one already-resident routed expert for the CPU reference path.
    pub fn read_routed_weights(
        &mut self,
        layer: usize,
        expert: usize,
        geometry: CloverExpertGeometry,
        tick: u64,
    ) -> Result<CloverExpertWeights, CloverExpertCacheError> {
        let id = allocation_id(layer, expert);
        if !self.ready_residents.contains(&id) {
            return Err(CloverExpertCacheError::IncompleteLoad);
        }
        let byte_len = usize::try_from(allocation_bytes(
            geometry.hidden_size,
            geometry.intermediate_size,
        )?)
        .map_err(|_| CloverExpertCacheError::SizeOverflow)?;
        let bytes = self.manager.read(&id, 0, byte_len, tick)?;
        let gate_up_bytes = geometry
            .intermediate_size
            .checked_mul(2)
            .and_then(|n| n.checked_mul(geometry.hidden_size))
            .and_then(|n| n.checked_mul(BF16_BYTES))
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let (gate_up, down) = bytes.split_at(gate_up_bytes);
        Ok(CloverExpertWeights {
            hidden_size: geometry.hidden_size,
            intermediate_size: geometry.intermediate_size,
            gate_up_proj: decode_bf16(gate_up),
            down_proj: decode_bf16(down),
        })
    }

    /// Materialize the already-resident shared expert for the CPU reference path.
    pub fn read_shared_expert_weights(
        &mut self,
        layer: usize,
        geometry: CloverSharedExpertGeometry,
        tick: u64,
    ) -> Result<CloverSharedExpertWeights, CloverExpertCacheError> {
        let id = shared_allocation_id(layer);
        if !self.ready_residents.contains(&id) {
            return Err(CloverExpertCacheError::IncompleteLoad);
        }
        let byte_len = usize::try_from(shared_allocation_bytes(
            geometry.hidden_size,
            geometry.intermediate_size,
        )?)
        .map_err(|_| CloverExpertCacheError::SizeOverflow)?;
        let bytes = self.manager.read(&id, 0, byte_len, tick)?;
        let projection_bytes = geometry
            .intermediate_size
            .checked_mul(geometry.hidden_size)
            .and_then(|n| n.checked_mul(BF16_BYTES))
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let gate_bytes = geometry
            .hidden_size
            .checked_mul(BF16_BYTES)
            .ok_or(CloverExpertCacheError::SizeOverflow)?;
        let up_start = projection_bytes;
        let down_start = up_start + projection_bytes;
        let shared_gate_start = down_start + projection_bytes;
        Ok(CloverSharedExpertWeights {
            hidden_size: geometry.hidden_size,
            intermediate_size: geometry.intermediate_size,
            gate_proj: decode_bf16(&bytes[..projection_bytes]),
            up_proj: decode_bf16(&bytes[up_start..down_start]),
            down_proj: decode_bf16(&bytes[down_start..shared_gate_start]),
            shared_gate: decode_bf16(&bytes[shared_gate_start..shared_gate_start + gate_bytes]),
        })
    }
}

fn decode_bf16(bytes: &[u8]) -> Vec<Bf16> {
    bytes
        .chunks_exact(BF16_BYTES)
        .map(|pair| Bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

fn open_planned_shard(
    root: &Path,
    expected: &ShardLoadPlan,
) -> Result<SafetensorsFile, CloverExpertCacheError> {
    let file = SafetensorsFile::open(root.join(&expected.name))?;
    if file.file_len() != expected.file_bytes
        || file.header_len() != expected.header_bytes
        || file.data_len() != expected.payload_bytes
        || file.tensor_count() != expected.tensors.len()
    {
        return Err(CloverExpertCacheError::StaleShard(expected.name.clone()));
    }

    for tensor in &expected.tensors {
        let entry = file
            .tensor(&tensor.name)
            .ok_or_else(|| CloverExpertCacheError::StaleShard(expected.name.clone()))?;
        if entry.dtype() != tensor.dtype
            || entry.shape() != tensor.shape
            || entry.byte_len() != tensor.byte_len
        {
            return Err(CloverExpertCacheError::StaleShard(expected.name.clone()));
        }
        if let Some(first_chunk) = tensor.chunks.first() {
            let expected_start = first_chunk
                .shard_payload_offset
                .checked_sub(first_chunk.tensor_offset)
                .ok_or(CloverExpertCacheError::SizeOverflow)?;
            if entry.data_start() != expected_start {
                return Err(CloverExpertCacheError::StaleShard(expected.name.clone()));
            }
        }
    }
    Ok(file)
}

fn stream_task<D: Device>(
    manager: &mut DeviceMemoryManager<D>,
    shard: &SafetensorsFile,
    id: &str,
    task: &CloverExpertLoadTask,
    tick: u64,
) -> Result<(), CloverExpertCacheError> {
    let bytes =
        shard.read_tensor_range(&task.tensor_name, task.tensor_offset, task.length_bytes)?;
    manager.write(id, task.target_offset, &bytes, tick)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ShardLoadPlan, TensorChunkPlan};

    fn fixture_header(entries: &[(&str, &[usize], u64, u64)]) -> Vec<u8> {
        let tensors = entries
            .iter()
            .map(|(name, shape, start, end)| {
                let shape = shape
                    .iter()
                    .map(|dimension| dimension.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    r#""{name}":{{"dtype":"BF16","shape":[{shape}],"data_offsets":[{start},{end}]}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let mut header = format!("{{{tensors}}}").into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        header
    }

    fn write_safetensors_fixture(
        root: &Path,
        shard_name: &str,
        entries: &[(&str, &[usize], u64, u64)],
        payload_bytes: usize,
    ) -> (u64, u64) {
        use std::fs;
        fs::create_dir_all(root).unwrap();
        let header = fixture_header(entries);
        let header_bytes = header.len() as u64;
        let file_bytes = 8 + header_bytes + payload_bytes as u64;
        let mut bytes = Vec::with_capacity(file_bytes as usize);
        bytes.extend_from_slice(&header_bytes.to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend((0..payload_bytes).map(|value| value as u8));
        fs::write(root.join(shard_name), bytes).unwrap();
        (header_bytes, file_bytes)
    }

    fn fixture_plan(shard_name: &str) -> CloverLoadPlan {
        let entries = [
            (
                "model.layers.0.mlp.experts.gate_up_proj",
                &[2, 2, 2][..],
                0,
                16,
            ),
            (
                "model.layers.0.mlp.experts.down_proj",
                &[2, 2, 1][..],
                16,
                24,
            ),
        ];
        let header_bytes = fixture_header(&entries).len() as u64;
        CloverLoadPlan {
            max_chunk_bytes: 8,
            total_payload_bytes: 24,
            shards: vec![ShardLoadPlan {
                name: shard_name.to_owned(),
                file_bytes: 8 + header_bytes + 24,
                header_bytes,
                payload_bytes: 24,
                tensors: vec![
                    TensorLoadPlan {
                        name: entries[0].0.to_owned(),
                        shape: entries[0].1.to_vec(),
                        dtype: WeightDType::BF16,
                        byte_len: 16,
                        chunks: vec![
                            TensorChunkPlan {
                                tensor_offset: 0,
                                shard_payload_offset: 0,
                                length_bytes: 8,
                            },
                            TensorChunkPlan {
                                tensor_offset: 8,
                                shard_payload_offset: 8,
                                length_bytes: 8,
                            },
                        ],
                    },
                    TensorLoadPlan {
                        name: entries[1].0.to_owned(),
                        shape: entries[1].1.to_vec(),
                        dtype: WeightDType::BF16,
                        byte_len: 8,
                        chunks: vec![TensorChunkPlan {
                            tensor_offset: 0,
                            shard_payload_offset: 16,
                            length_bytes: 8,
                        }],
                    },
                ],
            }],
        }
    }

    fn write_fixture(root: &Path, shard_name: &str) {
        let entries = [
            ("model.layers.0.mlp.experts.gate_up_proj", &[2, 2, 2][..], 0, 16),
            ("model.layers.0.mlp.experts.down_proj", &[2, 2, 1][..], 16, 24),
        ];
        write_safetensors_fixture(root, shard_name, &entries, 24);
    }

    fn fixture_budgets(expert_cache_max_bytes: u64) -> MemoryBudgets {
        MemoryBudgets {
            vram_soft_bytes: 24,
            vram_hard_bytes: 24,
            host_bytes: 128,
            pinned_host_bytes: 0,
            expert_cache_target_bytes: expert_cache_max_bytes,
            expert_cache_max_bytes,
        }
    }

    fn fixture_geometry() -> CloverExpertGeometry {
        CloverExpertGeometry {
            num_experts: 2,
            hidden_size: 2,
            intermediate_size: 1,
        }
    }

    #[test]
    fn expert_load_tasks_intersect_tensor_chunks_without_exceeding_bound() {
        let plan = fixture_plan("model-00001-of-00001.safetensors");

        let tasks = build_load_tasks(&plan, 0, 1, 2, 2, 1).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].file_offset, 8 + plan.shards[0].header_bytes + 8);
        assert_eq!(tasks[0].target_offset, 0);
        assert_eq!(tasks[0].length_bytes, 8);
        assert_eq!(tasks[1].file_offset, 8 + plan.shards[0].header_bytes + 20);
        assert_eq!(tasks[1].target_offset, 8);
        assert_eq!(tasks[1].length_bytes, 4);
        assert!(tasks
            .iter()
            .all(|task| task.length_bytes <= plan.max_chunk_bytes));
        assert!(tasks
            .iter()
            .all(|task| task.expert_id == "clover.layer.0.expert.1"));
    }

    #[test]
    fn expert_load_plan_rejects_non_bf16_and_incomplete_chunk_coverage() {
        let mut plan = CloverLoadPlan {
            max_chunk_bytes: 8,
            total_payload_bytes: 24,
            shards: vec![ShardLoadPlan {
                name: "model.safetensors".to_owned(),
                file_bytes: 48,
                header_bytes: 16,
                payload_bytes: 24,
                tensors: vec![
                    TensorLoadPlan {
                        name: "model.layers.0.mlp.experts.gate_up_proj".to_owned(),
                        shape: vec![2, 2, 2],
                        dtype: WeightDType::F32,
                        byte_len: 16,
                        chunks: vec![TensorChunkPlan {
                            tensor_offset: 0,
                            shard_payload_offset: 0,
                            length_bytes: 16,
                        }],
                    },
                    TensorLoadPlan {
                        name: "model.layers.0.mlp.experts.down_proj".to_owned(),
                        shape: vec![2, 2, 1],
                        dtype: WeightDType::BF16,
                        byte_len: 8,
                        chunks: vec![TensorChunkPlan {
                            tensor_offset: 0,
                            shard_payload_offset: 16,
                            length_bytes: 8,
                        }],
                    },
                ],
            }],
        };
        assert!(matches!(
            build_load_tasks(&plan, 0, 0, 2, 2, 1),
            Err(CloverExpertCacheError::InvalidTensor(_))
        ));
        plan.shards[0].tensors[0].dtype = WeightDType::BF16;
        plan.shards[0].tensors[0].chunks[0].length_bytes = 8;
        assert!(matches!(
            build_load_tasks(&plan, 0, 1, 2, 2, 1),
            Err(CloverExpertCacheError::IncompleteChunkCoverage)
        ));
    }

    #[test]
    fn cpu_cache_streams_and_protects_expert_residency() {
        use claymore_device::CpuDevice;
        use std::fs;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(1);
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-expert-stream-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();

        let shard_name = "model-00001-of-00001.safetensors";
        write_fixture(&root, shard_name);
        let plan = fixture_plan(shard_name);

        let mut cache =
            CloverExpertCache::new(CpuDevice::new(), fixture_budgets(24), EvictionPolicy::Lru)
                .unwrap();
        let request = CloverExpertLoadRequest {
            checkpoint_root: &root,
            plan: &plan,
            layer: 0,
            expert: 1,
            geometry: fixture_geometry(),
            tick: 1,
        };
        let mut checks = 0;
        let error = cache.ensure_resident_with_cancel(request, || {
            checks += 1;
            checks == 3
        });
        assert!(matches!(error, Err(CloverExpertCacheError::Cancelled)));
        assert!(!cache.is_resident(0, 1));
        assert_eq!(cache.usage_bytes(), 0);
        assert_eq!(cache.metrics().bytes_transferred, 8);
        assert_eq!(cache.metrics().cancelled_loads, 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_chunk_read_rolls_back_the_admitted_expert() {
        use claymore_device::CpuDevice;

        let root = std::env::temp_dir().join("claymore-clover-missing-shard");
        let plan = fixture_plan("missing.safetensors");
        let mut cache =
            CloverExpertCache::new(CpuDevice::new(), fixture_budgets(24), EvictionPolicy::Lru)
                .unwrap();
        let error = cache.ensure_resident(CloverExpertLoadRequest {
            checkpoint_root: &root,
            plan: &plan,
            layer: 0,
            expert: 1,
            geometry: fixture_geometry(),
            tick: 1,
        });
        assert!(matches!(error, Err(CloverExpertCacheError::Safetensors(_))));
        assert!(!cache.is_resident(0, 1));
        assert_eq!(cache.usage_bytes(), 0);
        assert_eq!(cache.metrics().failed_loads, 1);
    }

    #[test]
    fn expert_stream_rejects_a_shard_replaced_with_different_tensor_offsets() {
        use claymore_device::CpuDevice;
        use std::fs;

        let root = std::env::temp_dir().join(format!(
            "claymore-clover-stale-shard-{}",
            std::process::id()
        ));
        let shard_name = "model.safetensors";
        write_fixture(&root, shard_name);
        let plan = fixture_plan(shard_name);

        let changed_entries = [
            (
                "model.layers.0.mlp.experts.gate_up_proj",
                &[2, 2, 2][..],
                8,
                24,
            ),
            ("model.layers.0.mlp.experts.down_proj", &[2, 2, 1][..], 0, 8),
        ];
        let (changed_header, changed_file) =
            write_safetensors_fixture(&root, shard_name, &changed_entries, 24);
        assert_eq!(changed_header, plan.shards[0].header_bytes);
        assert_eq!(changed_file, plan.shards[0].file_bytes);

        let mut cache =
            CloverExpertCache::new(CpuDevice::new(), fixture_budgets(24), EvictionPolicy::Lru)
                .unwrap();
        let error = cache.ensure_resident(CloverExpertLoadRequest {
            checkpoint_root: &root,
            plan: &plan,
            layer: 0,
            expert: 1,
            geometry: fixture_geometry(),
            tick: 1,
        });
        assert!(matches!(error, Err(CloverExpertCacheError::StaleShard(_))));
        assert!(!cache.is_resident(0, 1));
        assert_eq!(cache.usage_bytes(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn in_flight_expert_is_not_evicted_under_cache_pressure() {
        use claymore_device::CpuDevice;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(200);
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-pressure-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let shard_name = "model.safetensors";
        write_fixture(&root, shard_name);
        let plan = fixture_plan(shard_name);
        let mut cache =
            CloverExpertCache::new(CpuDevice::new(), fixture_budgets(12), EvictionPolicy::Lru)
                .unwrap();

        cache
            .ensure_resident(CloverExpertLoadRequest {
                checkpoint_root: &root,
                plan: &plan,
                layer: 0,
                expert: 0,
                geometry: fixture_geometry(),
                tick: 1,
            })
            .unwrap();
        cache.mark_in_flight_until(0, 0, 5, 2).unwrap();

        let blocked = cache.ensure_resident(CloverExpertLoadRequest {
            checkpoint_root: &root,
            plan: &plan,
            layer: 0,
            expert: 1,
            geometry: fixture_geometry(),
            tick: 3,
        });
        assert!(blocked.is_err());
        assert!(cache.is_resident(0, 0));
        assert!(!cache.is_resident(0, 1));
        assert_eq!(cache.metrics().admission_failures, 1);

        cache
            .ensure_resident(CloverExpertLoadRequest {
                checkpoint_root: &root,
                plan: &plan,
                layer: 0,
                expert: 1,
                geometry: fixture_geometry(),
                tick: 5,
            })
            .unwrap();
        assert!(!cache.is_resident(0, 0));
        assert!(cache.is_resident(0, 1));
        assert_eq!(cache.metrics().evictions, 1);
        assert_eq!(cache.usage_bytes(), 12);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prefetch_order_is_priority_then_layer_then_expert() {
        let requests = [
            CloverExpertPrefetchRequest {
                layer: 1,
                expert: 2,
                priority: 1,
            },
            CloverExpertPrefetchRequest {
                layer: 0,
                expert: 3,
                priority: 5,
            },
            CloverExpertPrefetchRequest {
                layer: 0,
                expert: 1,
                priority: 5,
            },
            CloverExpertPrefetchRequest {
                layer: 0,
                expert: 0,
                priority: 7,
            },
        ];
        let ordered = ordered_prefetch_requests(&requests);
        assert_eq!(
            ordered,
            vec![
                CloverExpertPrefetchRequest {
                    layer: 0,
                    expert: 0,
                    priority: 7,
                },
                CloverExpertPrefetchRequest {
                    layer: 0,
                    expert: 1,
                    priority: 5,
                },
                CloverExpertPrefetchRequest {
                    layer: 0,
                    expert: 3,
                    priority: 5,
                },
                CloverExpertPrefetchRequest {
                    layer: 1,
                    expert: 2,
                    priority: 1,
                },
            ]
        );
    }

    #[test]
    fn shared_expert_weights_use_the_same_bounded_cache_and_in_flight_guards() {
        use claymore_device::CpuDevice;
        use std::fs;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(300);
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-shared-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let shard_name = "shared.safetensors";
        let entries = [
            (
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
                &[1, 2][..],
                0,
                4,
            ),
            (
                "model.layers.0.mlp.shared_expert.up_proj.weight",
                &[1, 2][..],
                4,
                8,
            ),
            (
                "model.layers.0.mlp.shared_expert.down_proj.weight",
                &[2, 1][..],
                8,
                12,
            ),
            (
                "model.layers.0.mlp.shared_expert_gate.weight",
                &[1, 2][..],
                12,
                16,
            ),
        ];
        let (header_bytes, file_bytes) = write_safetensors_fixture(&root, shard_name, &entries, 16);

        let tensor = |name: &str, shape: Vec<usize>, offset: u64| TensorLoadPlan {
            name: name.to_owned(),
            shape,
            dtype: WeightDType::BF16,
            byte_len: 4,
            chunks: vec![TensorChunkPlan {
                tensor_offset: 0,
                shard_payload_offset: offset,
                length_bytes: 4,
            }],
        };
        let plan = CloverLoadPlan {
            max_chunk_bytes: 8,
            total_payload_bytes: 16,
            shards: vec![ShardLoadPlan {
                name: shard_name.to_owned(),
                file_bytes,
                header_bytes,
                payload_bytes: 16,
                tensors: vec![
                    tensor(
                        "model.layers.0.mlp.shared_expert.gate_proj.weight",
                        vec![1, 2],
                        0,
                    ),
                    tensor(
                        "model.layers.0.mlp.shared_expert.up_proj.weight",
                        vec![1, 2],
                        4,
                    ),
                    tensor(
                        "model.layers.0.mlp.shared_expert.down_proj.weight",
                        vec![2, 1],
                        8,
                    ),
                    tensor(
                        "model.layers.0.mlp.shared_expert_gate.weight",
                        vec![1, 2],
                        12,
                    ),
                ],
            }],
        };
        let budgets = MemoryBudgets {
            vram_soft_bytes: 16,
            vram_hard_bytes: 16,
            host_bytes: 128,
            pinned_host_bytes: 0,
            expert_cache_target_bytes: 16,
            expert_cache_max_bytes: 16,
        };
        let mut cache =
            CloverExpertCache::new(CpuDevice::new(), budgets, EvictionPolicy::Lru).unwrap();
        let geometry = CloverSharedExpertGeometry {
            hidden_size: 2,
            intermediate_size: 1,
        };
        let shared_tasks =
            CloverExpertCache::<CpuDevice>::shared_load_tasks(&plan, 0, geometry).unwrap();
        assert_eq!(
            shared_tasks
                .iter()
                .map(|task| task.target_offset)
                .collect::<Vec<_>>(),
            vec![0, 4, 8, 12]
        );
        cache
            .ensure_shared_expert_resident(CloverSharedExpertLoadRequest {
                checkpoint_root: &root,
                plan: &plan,
                layer: 0,
                geometry,
                tick: 1,
            })
            .unwrap();
        assert!(cache.is_shared_expert_resident(0));
        assert_eq!(cache.usage_bytes(), 16);

        let weights = cache.read_shared_expert_weights(0, geometry, 2).unwrap();
        assert_eq!(weights.gate_proj.len(), 2);
        assert_eq!(weights.up_proj.len(), 2);
        assert_eq!(weights.down_proj.len(), 2);
        assert_eq!(weights.shared_gate.len(), 2);
        assert!(weights
            .shared_gate
            .iter()
            .all(|value| value.to_f32().is_finite()));

        cache.mark_shared_expert_in_flight_until(0, 4, 3).unwrap();
        assert!(cache.release_shared_expert(0, 3).is_err());
        cache.release_shared_expert(0, 4).unwrap();
        assert!(!cache.is_shared_expert_resident(0));
        assert_eq!(cache.usage_bytes(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scheduler_graph_carries_only_ids_and_compute_dependency() {
        use claymore_scheduler::{Scheduler, SchedulerConfig, TaskKind, TaskPriority, TaskState};

        let plan = fixture_plan("model.safetensors");
        let chunks = build_load_tasks(&plan, 0, 1, 2, 2, 1).unwrap();
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let schedule =
            CloverExpertLoadSchedule::submit(&mut scheduler, chunks, TaskPriority::High).unwrap();

        let load = scheduler.task(schedule.load_task()).unwrap();
        assert_eq!(load.kind, TaskKind::Prefetch);
        assert_eq!(load.state, TaskState::Ready);
        assert_eq!(schedule.chunks().len(), 2);
        assert!(schedule
            .chunks()
            .iter()
            .all(|chunk| chunk.expert_id == "clover.layer.0.expert.1"));

        let compute = schedule
            .submit_compute(&mut scheduler, TaskPriority::Normal)
            .unwrap();
        let compute_snapshot = scheduler.task(compute).unwrap();
        assert_eq!(compute_snapshot.kind, TaskKind::Compute);
        assert_eq!(compute_snapshot.dependencies, vec![schedule.load_task()]);

        assert_eq!(
            scheduler.start_next(TaskKind::Prefetch),
            Some(schedule.load_task())
        );
        scheduler.complete(schedule.load_task()).unwrap();
        assert_eq!(scheduler.start_next(TaskKind::Compute), Some(compute));
    }

    #[test]
    fn prefetch_set_uses_coreset_not_masked_top_k_experts() {
        let routing = crate::CloverMoeRouter {
            num_experts: 4,
            experts_per_token: 2,
            coreset_enabled: true,
            coreset_top_p: 0.9,
        }
        .route_batched(
            &[
                8.0, 1.0, 0.0, -1.0, 8.0, 1.0, 0.0, -1.0, -1.0, 0.0, 8.0, 1.0, -1.0, 0.0, 8.0, 1.0,
            ],
            2,
            2,
            &[true, true, true, true],
            true,
        )
        .unwrap();
        let requests = prefetch_requests_for_routing(&routing, 7, 5);
        assert_eq!(
            requests,
            vec![
                CloverExpertPrefetchRequest {
                    layer: 7,
                    expert: 0,
                    priority: 5,
                },
                CloverExpertPrefetchRequest {
                    layer: 7,
                    expert: 2,
                    priority: 5,
                },
            ]
        );
    }

    #[test]
    fn prefetch_ordered_loads_requests_in_deterministic_priority_order() {
        use claymore_device::CpuDevice;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(400);
        let root = std::env::temp_dir().join(format!(
            "claymore-clover-prefetch-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let shard_name = "model.safetensors";
        write_fixture(&root, shard_name);
        let plan = fixture_plan(shard_name);
        let mut cache =
            CloverExpertCache::new(CpuDevice::new(), fixture_budgets(24), EvictionPolicy::Lru)
                .unwrap();
        let requests = [
            CloverExpertPrefetchRequest {
                layer: 0,
                expert: 1,
                priority: 1,
            },
            CloverExpertPrefetchRequest {
                layer: 0,
                expert: 0,
                priority: 2,
            },
        ];
        let loaded = cache
            .prefetch_ordered(&root, &plan, fixture_geometry(), &requests, 1)
            .unwrap();
        assert_eq!(
            loaded,
            vec!["clover.layer.0.expert.0", "clover.layer.0.expert.1"]
        );
        assert_eq!(cache.metrics().resident_experts, 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prefetch_set_keeps_unmasked_routes_for_invalid_tokens() {
        let routing = crate::CloverMoeRouter {
            num_experts: 4,
            experts_per_token: 2,
            coreset_enabled: true,
            coreset_top_p: 0.9,
        }
        .route(&[1.0, 2.0, 3.0, 4.0], 1, &[false], true)
        .unwrap();
        assert_eq!(routing.coreset, Some(vec![Vec::new()]));
        let requests = prefetch_requests_for_routing(&routing, 0, 1);
        assert_eq!(
            requests
                .iter()
                .map(|request| request.expert)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn block_compute_waits_for_all_routed_and_shared_expert_chunks() {
        use claymore_scheduler::{Scheduler, SchedulerConfig, TaskKind, TaskPriority};

        let plan = fixture_plan("model.safetensors");
        let expert_zero = build_load_tasks(&plan, 0, 0, 2, 2, 1).unwrap();
        let expert_one = build_load_tasks(&plan, 0, 1, 2, 2, 1).unwrap();
        let mut scheduler = Scheduler::new(SchedulerConfig::default()).unwrap();
        let schedule = CloverExpertLoadSchedule::submit_block(
            &mut scheduler,
            vec![expert_zero, expert_one],
            TaskPriority::High,
        )
        .unwrap();
        assert_eq!(schedule.chunks().len(), 4);
        assert!(schedule
            .chunks()
            .iter()
            .any(|chunk| chunk.expert_id == "clover.layer.0.expert.0"));
        assert!(schedule
            .chunks()
            .iter()
            .any(|chunk| chunk.expert_id == "clover.layer.0.expert.1"));

        let compute = schedule
            .submit_compute(&mut scheduler, TaskPriority::Normal)
            .unwrap();
        let compute_snapshot = scheduler.task(compute).unwrap();
        assert_eq!(compute_snapshot.dependencies, vec![schedule.load_task()]);

        assert_eq!(
            scheduler.start_next(TaskKind::Prefetch),
            Some(schedule.load_task())
        );
        scheduler.complete(schedule.load_task()).unwrap();
        assert_eq!(scheduler.start_next(TaskKind::Compute), Some(compute));
    }
}
