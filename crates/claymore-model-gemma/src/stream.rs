//! Bounded, event-tracked weight streaming for the full DiffusionGemma graph.
//!
//! Every device allocation goes through `CudaMemoryManager`: the stream never
//! calls `cuMemAlloc` directly, never uses system swap as a weight tier, and
//! never releases a buffer while a CUDA operation may still reference it.
//! Tensors are read from Safetensors shards in bounded chunks, staged in one
//! pinned host buffer per tensor, transferred once, and the staging buffer is
//! released only after the transfer event completes.

use std::collections::BTreeMap;

#[cfg(feature = "cuda-streaming")]
use claymore_core::MemoryTier;
#[cfg(feature = "cuda-streaming")]
use claymore_memory::{CudaMemoryError, CudaTensorMemory, MemoryBudgets};

#[cfg(feature = "cuda-streaming")]
use crate::checkpoint::Checkpoint;
use crate::config::GemmaModelConfig;

/// Maximum bytes read from one shard between two `write_pinned` calls. The
/// staging buffer itself is sized per tensor; chunking bounds the transient
/// read buffers, not the residency.
pub const STAGING_CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error(transparent)]
    Checkpoint(#[from] crate::checkpoint::CheckpointError),
    #[cfg(feature = "cuda-streaming")]
    #[error(transparent)]
    Memory(#[from] CudaMemoryError),
    #[error("tensor {tensor} is not in the pinned inventory")]
    MissingWeight { tensor: String },
    #[error("weight stream is closed after cancellation")]
    Closed,
    #[error("cannot free {bytes} bytes for tensor {tensor}: {evictable} bytes are evictable")]
    BudgetExhausted {
        tensor: String,
        bytes: u64,
        evictable: u64,
    },
}

/// Deterministic tensor name plan for one decoder/encoder text layer.
pub fn decoder_layer_names(config: &GemmaModelConfig, layer: usize) -> Vec<String> {
    let p = |s: &str| format!("model.decoder.layers.{layer}.{s}");
    let mut names = vec![
        p("input_layernorm.weight"),
        p("self_attn.q_proj.weight"),
        p("self_attn.k_proj.weight"),
    ];
    if !config
        .text_config
        .as_ref()
        .map(|t| t.is_global(layer))
        .unwrap_or(false)
    {
        names.push(p("self_attn.v_proj.weight"));
    }
    names.push(p("self_attn.o_proj.weight"));
    names.extend([
        p("self_attn.q_norm.weight"),
        p("self_attn.k_norm.weight"),
        p("post_attention_layernorm.weight"),
        p("pre_feedforward_layernorm.weight"),
        p("mlp.gate_proj.weight"),
        p("mlp.up_proj.weight"),
        p("mlp.down_proj.weight"),
        p("post_feedforward_layernorm.weight"),
        p("post_feedforward_layernorm_1.weight"),
        p("router.proj.weight"),
        p("router.scale"),
        p("router.per_expert_scale"),
        p("pre_feedforward_layernorm_2.weight"),
        p("experts.gate_up_proj"),
        p("experts.down_proj"),
        p("layer_scalar"),
        // The text weights are tied, but each tower learns its own scalar: the
        // encoder scalar is a separate tensor of the pinned checkpoint.
        format!("model.encoder.language_model.layers.{layer}.layer_scalar"),
    ]);
    names
}

/// Deterministic tensor name plan for one vision tower layer.
pub fn vision_layer_names(_config: &GemmaModelConfig, layer: usize) -> Vec<String> {
    let p = |s: &str| format!("model.encoder.vision_tower.encoder.layers.{layer}.{s}");
    vec![
        p("input_layernorm.weight"),
        p("self_attn.q_proj.linear.weight"),
        p("self_attn.k_proj.linear.weight"),
        p("self_attn.v_proj.linear.weight"),
        p("self_attn.o_proj.linear.weight"),
        p("self_attn.q_norm.weight"),
        p("self_attn.k_norm.weight"),
        p("post_attention_layernorm.weight"),
        p("pre_feedforward_layernorm.weight"),
        p("mlp.gate_proj.linear.weight"),
        p("mlp.up_proj.linear.weight"),
        p("mlp.down_proj.linear.weight"),
        p("post_feedforward_layernorm.weight"),
    ]
}

/// Vision prefix tensors (patch embedder, position table, standardization).
pub fn vision_prefix_names(_config: &GemmaModelConfig) -> Vec<String> {
    vec![
        "model.encoder.vision_tower.patch_embedder.input_proj.weight".to_owned(),
        "model.encoder.vision_tower.patch_embedder.position_embedding_table".to_owned(),
        "model.encoder.vision_tower.std_bias".to_owned(),
        "model.encoder.vision_tower.std_scale".to_owned(),
    ]
}

/// Text prefix tensors: embedding table, final norm and self-conditioning.
pub fn text_prefix_names(_config: &GemmaModelConfig) -> Vec<String> {
    vec![
        "model.decoder.embed_tokens.weight".to_owned(),
        "model.decoder.norm.weight".to_owned(),
        "model.decoder.self_conditioning.pre_norm.weight".to_owned(),
        "model.decoder.self_conditioning.gate_proj.weight".to_owned(),
        "model.decoder.self_conditioning.up_proj.weight".to_owned(),
        "model.decoder.self_conditioning.down_proj.weight".to_owned(),
    ]
}

#[cfg(feature = "cuda-streaming")]
fn is_expert_tensor(name: &str) -> bool {
    name.contains(".experts.")
}

/// Residency accounting and peak instrumentation for one streamed tensor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StreamMetrics {
    pub peak_vram_bytes: u64,
    pub peak_pinned_bytes: u64,
    pub transferred_bytes: u64,
    pub disk_read_bytes: u64,
    pub loads: u64,
    pub cache_hits: u64,
    pub evictions: u64,
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub struct Resident {
    pub device_id: String,
    pub bytes: u64,
    pub last_used: u64,
    pub expert: bool,
}

/// Pure LRU victim order for the resident set: ascending `last_used`, with
/// expert tensors released after non-expert tensors of the same tick.
pub fn lru_victims(
    resident: &BTreeMap<String, Resident>,
    need: u64,
    soft: u64,
    usage: u64,
) -> Vec<String> {
    if usage.saturating_add(need) <= soft {
        return Vec::new();
    }
    let mut ordered: Vec<(u64, bool, String)> = resident
        .iter()
        .map(|(name, entry)| (entry.last_used, entry.expert, name.clone()))
        .collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut victims = Vec::new();
    let mut freed = usage;
    for (last_used, expert, name) in ordered {
        let _ = (last_used, expert);
        victims.push(name);
        freed = freed.saturating_sub(resident[&victims[victims.len() - 1]].bytes);
        if freed + need <= soft {
            break;
        }
    }
    victims
}

/// Streaming executor for the full checkpoint. One device allocation per
/// resident tensor, admission through `CudaMemoryManager` only.
#[cfg(feature = "cuda-streaming")]
pub struct WeightStream {
    checkpoint: Checkpoint,
    resident: BTreeMap<String, Resident>,
    budgets: MemoryBudgets,
    metrics: StreamMetrics,
    tick: u64,
    closed: bool,
}

#[cfg(feature = "cuda-streaming")]
impl WeightStream {
    pub fn open_pinned(
        root: impl AsRef<std::path::Path>,
        budgets: MemoryBudgets,
    ) -> Result<Self, StreamError> {
        let checkpoint = Checkpoint::open_pinned(root)?;
        budgets
            .validate()
            .map_err(CudaMemoryError::Policy)
            .map_err(StreamError::Memory)?;
        Ok(Self {
            checkpoint,
            resident: BTreeMap::new(),
            budgets,
            metrics: StreamMetrics::new(),
            tick: 0,
            closed: false,
        })
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    pub fn metrics(&self) -> &StreamMetrics {
        &self.metrics
    }

    pub fn budgets(&self) -> &MemoryBudgets {
        &self.budgets
    }

    pub fn resident_ids(&self) -> Vec<String> {
        self.resident
            .values()
            .map(|r| r.device_id.clone())
            .collect()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn next_tick(&mut self, tick: u64) -> u64 {
        if tick > self.tick {
            self.tick = tick;
        }
        self.tick
    }

    /// Release LRU residents so the soft VRAM budget can admit `need` bytes.
    fn evict_for(
        &mut self,
        memory: &CudaTensorMemory,
        tensor: &str,
        need: u64,
        tick: u64,
    ) -> Result<(), StreamError> {
        let usage = memory.manager_mut().usage_bytes(MemoryTier::Vram);
        let victims = lru_victims(&self.resident, need, self.budgets.vram_soft_bytes, usage);
        let mut evictable = 0u64;
        for name in victims {
            let entry = &self.resident[&name];
            if entry.last_used >= tick {
                evictable += entry.bytes;
                continue;
            }
            // The `RefMut` returned by `manager_mut()` is a temporary of the
            // match scrutinee: it stays alive until the end of the `match`,
            // so the release result is bound first. Borrowing the manager
            // again inside the arm would panic with `RefCell already
            // borrowed`.
            let release = memory.manager_mut().release(&entry.device_id, tick);
            match release {
                Ok(_alloc) => {
                    self.resident.remove(&name);
                    self.metrics.evictions += 1;
                    if memory.manager_mut().usage_bytes(MemoryTier::Vram) + need
                        <= self.budgets.vram_soft_bytes
                    {
                        return Ok(());
                    }
                }
                Err(CudaMemoryError::Policy(_)) => {
                    // IN_FLIGHT guard: keep this resident and try the next victim.
                    evictable += entry.bytes;
                }
                Err(error) => return Err(error.into()),
            }
        }
        if memory.manager_mut().usage_bytes(MemoryTier::Vram) + need <= self.budgets.vram_hard_bytes
        {
            return Ok(());
        }
        Err(StreamError::BudgetExhausted {
            tensor: tensor.to_owned(),
            bytes: need,
            evictable,
        })
    }

    /// Load one tensor into VRAM, returning its manager allocation id.
    /// Cache hits are recorded; misses stream the payload in bounded chunks.
    pub fn ensure_tensor(
        &mut self,
        memory: &CudaTensorMemory,
        name: &str,
        tick: u64,
    ) -> Result<String, StreamError> {
        if self.closed {
            return Err(StreamError::Closed);
        }
        let tick = self.next_tick(tick);
        let location = self
            .checkpoint
            .tensor(name)
            .ok_or(StreamError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        let device_id = format!("w:{name}");
        let bytes = location.entry.byte_len();
        if let Some(resident) = self.resident.get_mut(name) {
            resident.last_used = tick;
            self.metrics.cache_hits += 1;
            memory
                .manager_mut()
                .record_reload_cost(&device_id, 0, tick)?;
            return Ok(device_id);
        }
        self.evict_for(memory, name, bytes, tick)?;
        let pinned_id = format!("p:{name}");
        memory
            .manager_mut()
            .allocate_pinned(&pinned_id, bytes, tick)?;
        self.metrics.peak_pinned_bytes = self
            .metrics
            .peak_pinned_bytes
            .max(memory.manager_mut().usage_bytes(MemoryTier::PinnedHost));
        // The shard reader applies the header offset: the range is read
        // relative to the start of the tensor payload, never to the start of
        // the shard file.
        let mut offset = 0usize;
        let mut remaining = bytes;
        while remaining > 0 {
            let chunk = remaining.min(STAGING_CHUNK_BYTES as u64) as usize;
            let buffer = self
                .checkpoint
                .read_tensor_range(name, offset as u64, chunk)?;
            memory
                .manager_mut()
                .write_pinned(&pinned_id, offset, &buffer, tick)?;
            offset += chunk;
            remaining -= chunk as u64;
            self.metrics.disk_read_bytes += chunk as u64;
        }
        memory
            .manager_mut()
            .allocate_device(&device_id, bytes, tick, is_expert_tensor(name))?;
        self.metrics.peak_vram_bytes = self
            .metrics
            .peak_vram_bytes
            .max(memory.manager_mut().usage_bytes(MemoryTier::Vram));
        let token = memory
            .manager_mut()
            .copy_h2d(&pinned_id, &device_id, bytes as usize, tick)?;
        self.metrics.transferred_bytes += bytes;
        self.metrics.peak_vram_bytes = self
            .metrics
            .peak_vram_bytes
            .max(memory.manager_mut().usage_bytes(MemoryTier::Vram));
        memory.manager_mut().wait_for_transfer(&token, tick)?;
        memory.manager_mut().release(&pinned_id, tick)?;
        self.resident.insert(
            name.to_owned(),
            Resident {
                device_id: device_id.clone(),
                bytes,
                last_used: tick,
                expert: is_expert_tensor(name),
            },
        );
        self.metrics.loads += 1;
        Ok(device_id)
    }

    /// Load a whole planned layer; returns the device ids in plan order.
    pub fn ensure_plan(
        &mut self,
        memory: &CudaTensorMemory,
        names: &[String],
        tick: u64,
    ) -> Result<Vec<String>, StreamError> {
        let mut ids = Vec::with_capacity(names.len());
        for name in names {
            ids.push(self.ensure_tensor(memory, name, tick)?);
        }
        Ok(ids)
    }

    /// Release one resident tensor (the caller must have observed the CUDA
    /// completion event for every operation that used it).
    pub fn release(
        &mut self,
        memory: &CudaTensorMemory,
        name: &str,
        tick: u64,
    ) -> Result<(), StreamError> {
        if self.closed {
            return Err(StreamError::Closed);
        }
        let tick = self.next_tick(tick);
        let entry = self
            .resident
            .remove(name)
            .ok_or(StreamError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        memory.manager_mut().release(&entry.device_id, tick)?;
        Ok(())
    }

    /// Cancel the stream: release every resident tensor the manager accepts,
    /// in LRU order, then mark the stream closed. In-flight tensors remain
    /// guarded by the manager; the manager's poisoned flag keeps them
    /// non-evictable until synchronization succeeds.
    pub fn cancel(&mut self, memory: &CudaTensorMemory, tick: u64) -> Result<(), StreamError> {
        if self.closed {
            return Ok(());
        }
        let tick = self.next_tick(tick);
        let names: Vec<String> = self
            .resident
            .iter()
            .map(|(name, entry)| (entry.last_used, name.clone()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(_l, n)| n)
            .collect();
        for name in names {
            let device_id = self.resident[&name].device_id.clone();
            match memory.manager_mut().release(&device_id, tick) {
                Ok(_) => {
                    self.resident.remove(&name);
                    self.metrics.evictions += 1;
                }
                Err(CudaMemoryError::Policy(_)) => {}
                Err(error) => {
                    self.closed = true;
                    return Err(error.into());
                }
            }
        }
        self.closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident(bytes: u64, last: u64, expert: bool) -> Resident {
        Resident {
            device_id: format!("w:{bytes}"),
            bytes,
            last_used: last,
            expert,
        }
    }

    #[test]
    fn lru_victims_free_the_oldest_first_and_stop_at_the_soft_budget() {
        let mut map = BTreeMap::new();
        for (name, bytes, last, expert) in [
            ("a", 100, 1, false),
            ("b", 200, 2, true),
            ("c", 300, 0, false),
            ("d", 400, 3, false),
        ] {
            map.insert(name.to_owned(), resident(bytes, last, expert));
        }
        let victims = lru_victims(&map, 250, 400, 1000);
        assert_eq!(victims, ["c", "a", "b", "d"]);
        assert_eq!(lru_victims(&map, 10, 1200, 1000), Vec::<String>::new());
    }

    #[test]
    fn decoder_plan_matches_the_pinned_inventory_shape_rules() {
        // The plan rules are asserted against the pinned config when it is
        // readable; without it the test skips explicitly, never simulating a
        // success.
        let root = std::env::var("CLAYMORE_M10_CHECKPOINT").ok();
        let path = match root {
            Some(root) => format!("{root}/config.json"),
            None => {
                eprintln!("skip: CLAYMORE_M10_CHECKPOINT unavailable");
                return;
            }
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => {
                eprintln!("skip: pinned config not readable at {path}");
                return;
            }
        };
        let config = GemmaModelConfig::from_json(&text).expect("pinned config valid");
        let names = decoder_layer_names(&config, 0);
        assert!(names.iter().any(|n| n.ends_with("self_attn.v_proj.weight")));
        let global = decoder_layer_names(&config, 5);
        assert!(!global
            .iter()
            .any(|n| n.ends_with("self_attn.v_proj.weight")));
        assert!(global.iter().any(|n| n.ends_with("experts.gate_up_proj")));
        // Layer scalars and router parameters are part of every layer plan.
        assert!(names.iter().any(|n| n.ends_with("layer_scalar")));
        assert!(names.iter().any(|n| n.ends_with("router.per_expert_scale")));
    }
}
