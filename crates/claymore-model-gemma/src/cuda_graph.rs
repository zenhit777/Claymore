//! CUDA execution graph for the full DiffusionGemma model (M10.4).
//!
//! Every primitive runs on the device through `TensorDispatcher`; every
//! weight tensor is admitted through `CudaTensorMemory` (the shared
//! `CudaMemoryManager` owner); every activation crosses the pinned staging
//! boundary explicitly. Per-expert matrices are read from the Safetensors
//! payload at the exact element offset and uploaded as their own resident
//! tensors. The composition mirrors the pinned upstream reference
//! (`graph.rs` / `fullgraph.rs`) op by op, so the CUDA campaign is a
//! differential run against the CPU reference on the same input bits.
//!
//! Activations are device tensors identified by arena indices. The host is
//! used only for pure composition logic that has no contract operation
//! (row concat of cache blocks, mask building, position-embedding addition,
//! spatial pooling, self-conditioning softmax): these steps read downloaded
//! activations and re-upload the composed result, always through pinned
//! staging.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

use claymore_core::MemoryTier;
use claymore_memory::{CudaTensorBackend, CudaTensorError, CudaTensorMemory};
use claymore_tensor::{Bf16, ExecutionRequest, Tensor, TensorDispatcher, TensorOp};

use crate::checkpoint::Checkpoint;
use crate::config::GemmaModelConfig;

/// Chunk bound for reading one payload slice from a shard.
pub const READ_CHUNK_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error(transparent)]
    Memory(#[from] CudaTensorError),
    #[error(transparent)]
    Execution(#[from] claymore_tensor::ExecutionError),
    #[error("tensor {tensor} is not in the pinned inventory")]
    MissingWeight { tensor: String },
    #[error("io error while reading shard {shard}: {source}")]
    Io {
        shard: String,
        source: std::io::Error,
    },
    #[error("slice {name} is out of the tensor payload range")]
    SliceOutOfBounds { name: String },
    #[error("arena index {index} is out of range")]
    BadIndex { index: usize },
}

/// CUDA-side graph owner: one memory owner, one dispatcher, one arena of
/// live tensors, one LRU weight residency map.
pub struct CudaGraph {
    memory: CudaTensorMemory,
    dispatcher: TensorDispatcher,
    checkpoint: Checkpoint,
    arena: Vec<Tensor>,
    resident: BTreeMap<String, usize>,
    tick: u64,
    peak_vram_bytes: u64,
    uploaded_bytes: u64,
    downloaded_bytes: u64,
    launches: u64,
}

impl CudaGraph {
    pub fn new(memory: CudaTensorMemory, checkpoint: Checkpoint) -> Self {
        Self {
            dispatcher: TensorDispatcher::new(vec![Box::new(CudaTensorBackend::new(&memory))]),
            memory,
            checkpoint,
            arena: Vec::new(),
            resident: BTreeMap::new(),
            tick: 0,
            peak_vram_bytes: 0,
            uploaded_bytes: 0,
            downloaded_bytes: 0,
            launches: 0,
        }
    }

    pub fn memory(&self) -> &CudaTensorMemory {
        &self.memory
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    /// `(peak VRAM bytes, uploaded bytes, downloaded bytes, launches)`.
    pub fn metrics(&self) -> (u64, u64, u64, u64) {
        (
            self.peak_vram_bytes,
            self.uploaded_bytes,
            self.downloaded_bytes,
            self.launches,
        )
    }

    fn record_peak(&mut self) {
        let usage = self.memory.usage_bytes(MemoryTier::Vram);
        if usage > self.peak_vram_bytes {
            self.peak_vram_bytes = usage;
        }
    }

    fn index(&self, index: usize) -> Result<&Tensor, GraphError> {
        self.arena.get(index).ok_or(GraphError::BadIndex { index })
    }

    fn push(&mut self, tensor: Tensor) -> usize {
        self.arena.push(tensor);
        self.uploaded_bytes = self.uploaded_bytes;
        self.arena.len() - 1
    }

    // ------------------------------------------------------------------
    // Uploads and downloads (pinned staging, manager admission).

    pub fn upload_bf16(&mut self, shape: Vec<usize>, bits: &[Bf16]) -> Result<usize, GraphError> {
        let tensor = self.memory.upload_bf16(shape, bits)?;
        self.uploaded_bytes += tensor.layout().span_bytes() as u64;
        self.record_peak();
        Ok(self.push(tensor))
    }

    pub fn upload_f32(&mut self, shape: Vec<usize>, values: &[f32]) -> Result<usize, GraphError> {
        let tensor = self.memory.upload_f32(shape, values)?;
        self.uploaded_bytes += tensor.layout().span_bytes() as u64;
        self.record_peak();
        Ok(self.push(tensor))
    }

    pub fn download_bf16(&mut self, index: usize) -> Result<Vec<Bf16>, GraphError> {
        let bits = self.memory.download_bf16(self.index(index)?)?;
        self.downloaded_bytes += self.arena[index].layout().span_bytes() as u64;
        Ok(bits)
    }

    pub fn download_f32(&mut self, index: usize) -> Result<Vec<f32>, GraphError> {
        let values = self.memory.download_f32(self.index(index)?)?;
        self.downloaded_bytes += self.arena[index].layout().span_bytes() as u64;
        Ok(values)
    }

    // ------------------------------------------------------------------
    // Weight residency.

    /// Read the full payload of one tensor from its shard, in bounded chunks.
    pub fn read_payload(&self, name: &str) -> Result<Vec<Bf16>, GraphError> {
        let location = self
            .checkpoint
            .tensor(name)
            .ok_or(GraphError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        self.read_payload_range(name, 0, location.entry.numel() as usize)
    }

    /// Read one contiguous sub-slice of a tensor payload (per-expert matrices).
    pub fn read_slice(
        &self,
        name: &str,
        start_element: usize,
        count_elements: usize,
    ) -> Result<Vec<Bf16>, GraphError> {
        let location = self
            .checkpoint
            .tensor(name)
            .ok_or(GraphError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        if start_element + count_elements > location.entry.numel() as usize {
            return Err(GraphError::SliceOutOfBounds {
                name: name.to_owned(),
            });
        }
        self.read_payload_range(name, start_element, count_elements)
    }

    fn read_payload_range(
        &self,
        name: &str,
        start_element: usize,
        count_elements: usize,
    ) -> Result<Vec<Bf16>, GraphError> {
        let location = self
            .checkpoint
            .tensor(name)
            .ok_or(GraphError::MissingWeight {
                tensor: name.to_owned(),
            })?;
        let shard_path = location.shard.path().display().to_string();
        let mut file =
            std::fs::File::open(location.shard.path()).map_err(|source| GraphError::Io {
                shard: shard_path.clone(),
                source,
            })?;
        let start = location.entry.data_start() + (start_element * 2) as u64;
        file.seek(SeekFrom::Start(start))
            .map_err(|source| GraphError::Io {
                shard: shard_path.clone(),
                source,
            })?;
        let bytes = count_elements * 2;
        let mut raw = Vec::with_capacity(bytes);
        let mut remaining = bytes;
        while remaining > 0 {
            let chunk = remaining.min(READ_CHUNK_BYTES);
            let mut buffer = vec![0u8; chunk];
            file.read_exact(&mut buffer)
                .map_err(|source| GraphError::Io {
                    shard: shard_path.clone(),
                    source,
                })?;
            raw.extend_from_slice(&buffer);
            remaining -= chunk;
        }
        Ok(raw
            .chunks_exact(2)
            .map(|b| Bf16::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// Residency-managed BF16 tensor for a full checkpoint tensor name.
    pub fn weight(&mut self, name: &str) -> Result<usize, GraphError> {
        if let Some(&index) = self.resident.get(name) {
            return Ok(index);
        }
        let shape = self
            .checkpoint
            .tensor(name)
            .ok_or(GraphError::MissingWeight {
                tensor: name.to_owned(),
            })?
            .entry
            .shape()
            .to_vec();
        let bits = self.read_payload(name)?;
        let index = self.upload_bf16(shape, &bits)?;
        self.resident.insert(name.to_owned(), index);
        Ok(index)
    }

    /// Residency-managed BF16 tensor for one sub-slice (per-expert matrix).
    pub fn weight_slice(
        &mut self,
        name: &str,
        start_element: usize,
        shape: Vec<usize>,
    ) -> Result<usize, GraphError> {
        let key = format!("{name}#{start_element}");
        if let Some(&index) = self.resident.get(&key) {
            return Ok(index);
        }
        let count = shape.iter().product::<usize>();
        let bits = self.read_slice(name, start_element, count)?;
        let index = self.upload_bf16(shape, &bits)?;
        self.resident.insert(key, index);
        Ok(index)
    }

    /// Release one resident weight after its last CUDA use has completed.
    pub fn release_resident(&mut self, key: &str) -> Result<(), GraphError> {
        if let Some(index) = self.resident.remove(key) {
            let id = self.memory.buffer_id(self.index(index)?)?;
            self.tick += 1;
            self.memory
                .manager_mut()
                .release(&id, self.tick)
                .map_err(CudaTensorError::from)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Launch helpers.

    /// Execute one op. The dispatcher borrows the graph immutably, so every
    /// tensor reference is taken from the arena in the same expression.
    fn exec_op(&self, op: TensorOp<'_>) -> Result<Tensor, GraphError> {
        Ok(self.dispatcher.execute(&ExecutionRequest::new(op))?)
    }

    fn finish(&mut self, output: Tensor) -> Result<usize, GraphError> {
        self.launches += 1;
        self.record_peak();
        Ok(self.push(output))
    }

    pub fn linear(&mut self, input: usize, weight: usize) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        let weight = self.index(weight)?;
        self.finish(self.exec_op(TensorOp::Linear { input, weight })?)
    }

    pub fn rms_norm(
        &mut self,
        input: usize,
        scale: usize,
        epsilon: f32,
    ) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        let scale = self.index(scale)?;
        self.finish(self.exec_op(TensorOp::RmsNorm {
            input,
            scale,
            epsilon,
        })?)
    }

    pub fn rope(&mut self, input: usize, cos: usize, sin: usize) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        let cos = self.index(cos)?;
        let sin = self.index(sin)?;
        self.finish(self.exec_op(TensorOp::RotaryEmbed { input, cos, sin })?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        query: usize,
        key: usize,
        value: usize,
        mask: Option<usize>,
        scale: f32,
    ) -> Result<usize, GraphError> {
        let query = self.index(query)?;
        let key = self.index(key)?;
        let value = self.index(value)?;
        let mask_tensor = match mask {
            Some(index) => Some(self.index(index)?),
            None => None,
        };
        self.finish(self.exec_op(TensorOp::Attention {
            query,
            key,
            value,
            mask: mask_tensor,
            scale,
        })?)
    }

    pub fn softmax(&mut self, input: usize) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        self.finish(self.exec_op(TensorOp::SoftmaxLastDim { input })?)
    }

    pub fn add(&mut self, lhs: usize, rhs: usize) -> Result<usize, GraphError> {
        let lhs = self.index(lhs)?;
        let rhs = self.index(rhs)?;
        self.finish(self.exec_op(TensorOp::Add { lhs, rhs })?)
    }

    pub fn scale(&mut self, input: usize, factor: usize) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        let factor = self.index(factor)?;
        self.finish(self.exec_op(TensorOp::Scale { input, factor })?)
    }

    pub fn scale_bias(
        &mut self,
        input: usize,
        scale: usize,
        bias: usize,
    ) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        let scale = self.index(scale)?;
        let bias = self.index(bias)?;
        self.finish(self.exec_op(TensorOp::ScaleBias { input, scale, bias })?)
    }

    pub fn softcap(&mut self, input: usize, cap: f32) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        self.finish(self.exec_op(TensorOp::SoftCap { input, cap })?)
    }

    pub fn widen(&mut self, input: usize) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        self.finish(self.exec_op(TensorOp::Widen { input })?)
    }

    pub fn router_top_k(
        &mut self,
        scores: usize,
        per_expert_scale: usize,
        top_k: usize,
    ) -> Result<usize, GraphError> {
        let scores = self.index(scores)?;
        let scale = self.index(per_expert_scale)?;
        self.finish(self.exec_op(TensorOp::RouterTopK {
            scores,
            per_expert_scale: scale,
            top_k,
        })?)
    }

    pub fn gelu_mul_up(&mut self, gate: usize, up: usize) -> Result<usize, GraphError> {
        let gate = self.index(gate)?;
        let up = self.index(up)?;
        self.finish(self.exec_op(TensorOp::GeluMulUp { gate, up })?)
    }

    pub fn gather_rows(&mut self, input: usize, indices: usize) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        let indices = self.index(indices)?;
        self.finish(self.exec_op(TensorOp::GatherRows { input, indices })?)
    }

    pub fn scatter_add(
        &mut self,
        source: usize,
        weights: usize,
        rows: usize,
        out_rows: usize,
    ) -> Result<usize, GraphError> {
        let source = self.index(source)?;
        let weights = self.index(weights)?;
        let rows = self.index(rows)?;
        self.finish(self.exec_op(TensorOp::ScatterAdd {
            source,
            weights,
            rows,
            out_rows,
        })?)
    }

    pub fn embed_gather(
        &mut self,
        ids: usize,
        table: usize,
        scale: f32,
    ) -> Result<usize, GraphError> {
        let ids = self.index(ids)?;
        let table = self.index(table)?;
        self.finish(self.exec_op(TensorOp::EmbedGather { ids, table, scale })?)
    }

    pub fn row_mask(&mut self, input: usize, flags: usize) -> Result<usize, GraphError> {
        let input = self.index(input)?;
        let flags = self.index(flags)?;
        self.finish(self.exec_op(TensorOp::RowMask { input, flags })?)
    }

    /// Drop one arena entry, releasing its allocation through the manager.
    pub fn release(&mut self, index: usize) -> Result<(), GraphError> {
        if index >= self.arena.len() {
            return Err(GraphError::BadIndex { index });
        }
        let id = self.memory.buffer_id(&self.arena[index])?;
        self.tick += 1;
        self.memory
            .manager_mut()
            .release(&id, self.tick)
            .map_err(CudaTensorError::from)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Host-side composition helpers (no contract op exists for row concat,
    // mask construction, position-embedding addition, spatial pooling and
    // the self-conditioning softmax; these run on downloaded activations and
    // re-upload through pinned staging).

    pub fn host_bits(&mut self, index: usize) -> Result<Vec<Bf16>, GraphError> {
        self.download_bf16(index)
    }

    pub fn upload_bits(&mut self, shape: Vec<usize>, bits: Vec<Bf16>) -> Result<usize, GraphError> {
        self.upload_bf16(shape, &bits)
    }

    /// Standard rope tables `[tokens, head_dim]` F32, with the optional
    /// `partial_rotary_factor` of the global layers.
    pub fn rope_tables(
        head_dim: usize,
        theta: f32,
        positions: &[usize],
        partial: Option<f32>,
    ) -> (Vec<f32>, Vec<f32>) {
        let rotated = match partial {
            Some(f) => 2 * (head_dim as f32 * f / 2.0) as usize,
            None => head_dim,
        };
        let inv: Vec<f32> = (0..rotated / 2)
            .map(|i| theta.powf((2 * i) as f32 / rotated as f32).recip())
            .collect();
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for p in positions {
            let mut row_c = vec![1.0f32; head_dim];
            let mut row_s = vec![0.0f32; head_dim];
            for i in 0..inv.len() {
                let a = *p as f32 * inv[i];
                row_c[i] = a.cos();
                row_c[i + rotated / 2] = a.cos();
                row_s[i] = a.sin();
                row_s[i + rotated / 2] = a.sin();
            }
            cos.extend(row_c);
            sin.extend(row_s);
        }
        (cos, sin)
    }

    /// Axial 2D vision rope tables `[patches, head_dim]` F32, recomposed as
    /// `[h, h, w, w]` per patch.
    pub fn vision_rope_tables(
        head_dim: usize,
        theta: f32,
        positions: &[(u32, u32)],
    ) -> (Vec<f32>, Vec<f32>) {
        let spatial = head_dim / 2;
        let inv: Vec<f32> = (0..spatial / 2)
            .map(|i| theta.powf((2 * i) as f32 / spatial as f32).recip())
            .collect();
        let mut cos_out = Vec::with_capacity(positions.len() * head_dim);
        let mut sin_out = Vec::with_capacity(positions.len() * head_dim);
        for (x, y) in positions {
            let ch: Vec<f32> = inv.iter().map(|f| (*x as f32 * f).cos()).collect();
            let sh: Vec<f32> = inv.iter().map(|f| (*x as f32 * f).sin()).collect();
            let cw: Vec<f32> = inv.iter().map(|f| (*y as f32 * f).cos()).collect();
            let sw: Vec<f32> = inv.iter().map(|f| (*y as f32 * f).sin()).collect();
            cos_out.extend(ch.iter().chain(ch.iter()).chain(cw.iter()).chain(cw.iter()));
            sin_out.extend(sh.iter().chain(sh.iter()).chain(sw.iter()).chain(sw.iter()));
        }
        (cos_out, sin_out)
    }

    /// Additive mask `[q_len, kv_len]` F32: causal/windowed visibility,
    /// blocked keys as `-inf`.
    pub fn build_mask(
        positions: &[usize],
        kv_len: usize,
        sliding: Option<usize>,
        is_causal: bool,
    ) -> Option<Vec<f32>> {
        let q_len = positions.len();
        let mut values = Vec::with_capacity(q_len * kv_len);
        let mut any_masked = false;
        for &q_abs in positions {
            for k_abs in 0..kv_len {
                let visible = if is_causal && k_abs > q_abs {
                    false
                } else if let Some(w) = sliding {
                    (q_abs as i64 - k_abs as i64).abs() < w as i64
                } else {
                    true
                };
                values.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
                any_masked |= !visible;
            }
        }
        any_masked.then(|| values)
    }

    pub fn bf16_of(values: &[f32]) -> Vec<Bf16> {
        values.iter().map(|v| Bf16::from_f32(*v)).collect()
    }

    pub fn f32_of(bits: &[Bf16]) -> Vec<f32> {
        bits.iter().map(|b| b.to_f32()).collect()
    }
}

// ------------------------------------------------------------------
// Graph composition (mirrors `fullgraph.rs` op by op).

impl CudaGraph {
    /// Text prefix: scaled embedding gather for `ids`.
    pub fn embed_text(&mut self, ids: &[u32]) -> Result<usize, GraphError> {
        let table = self.weight("model.decoder.embed_tokens.weight")?;
        let hidden = self
            .checkpoint
            .config()
            .text_config
            .as_ref()
            .unwrap()
            .hidden_size
            .unwrap();
        let scale = (hidden as f32).powf(-0.5);
        let ids_t = self.upload_f32(
            vec![ids.len()],
            &ids.iter().map(|i| *i as f32).collect::<Vec<f32>>(),
        )?;
        self.embed_gather(ids_t, table, scale)
    }

    /// Per-layer KV cache on host: `(kv_heads, head_dim, rows, keys_bits, values_bits)`.
    pub fn append_cache(
        cache: &mut Vec<(usize, usize, usize, Vec<Bf16>, Vec<Bf16>)>,
        layer: usize,
        heads: usize,
        dim: usize,
        keys_bits: Vec<Bf16>,
        values_bits: Vec<Bf16>,
    ) {
        if cache.len() <= layer {
            cache.push((
                heads,
                dim,
                keys_bits.len() / (heads * dim),
                keys_bits,
                values_bits,
            ));
            return;
        }
        let added = keys_bits.len() / (heads * dim);
        let entry = &mut cache[layer];
        entry.3.extend(keys_bits);
        entry.4.extend(values_bits);
        entry.2 += added;
    }

    /// One text attention layer. Returns the `[tokens, hidden]` output index
    /// (`o_proj` applied), appending the encoder cache when `append` is true.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_layer(
        &mut self,
        prefix: &str,
        layer: usize,
        x: usize,
        cache: &mut Vec<(usize, usize, usize, Vec<Bf16>, Vec<Bf16>)>,
        positions: &[usize],
        head_dim: usize,
        heads: usize,
        kv_heads: usize,
        rope_theta: f32,
        partial: Option<f32>,
        sliding: Option<usize>,
        is_causal: bool,
        eps: f32,
        append: bool,
    ) -> Result<usize, GraphError> {
        let p = |n: &str| format!("model.{prefix}.layers.{layer}.self_attn.{n}");
        let tokens = self.arena[x].layout().shape()[0];
        let (cos, sin) = Self::rope_tables(head_dim, rope_theta, positions, partial);
        let cos_idx = self.upload_f32(vec![positions.len(), head_dim], &cos)?;
        let sin_idx = self.upload_f32(vec![positions.len(), head_dim], &sin)?;

        let q_w = self.weight(&p("q_proj.weight"))?;
        let q = self.linear(x, q_w)?;
        let q_bits = self.host_bits(q)?;
        let q3 = self.upload_bits(vec![tokens, heads, head_dim], q_bits)?;
        let q_norm_w = self.weight(&p("q_norm.weight"))?;
        let qn = self.rms_norm(q3, q_norm_w, eps)?;
        let qr = self.rope(qn, cos_idx, sin_idx)?;

        let k_w = self.weight(&p("k_proj.weight"))?;
        let k = self.linear(x, k_w)?;
        let k_bits = self.host_bits(k)?;
        let k3 = self.upload_bits(vec![tokens, kv_heads, head_dim], k_bits)?;
        let k_norm_w = self.weight(&p("k_norm.weight"))?;
        let kn = self.rms_norm(k3, k_norm_w, eps)?;
        let kr = self.rope(kn, cos_idx, sin_idx)?;

        let has_v = self.checkpoint.tensor(&p("v_proj.weight")).is_some();
        let v3_idx = if has_v {
            let v_w = self.weight(&p("v_proj.weight"))?;
            let v = self.linear(x, v_w)?;
            let v_bits = self.host_bits(v)?;
            self.upload_bits(vec![tokens, kv_heads, head_dim], v_bits)?
        } else {
            k3
        };
        let ones = {
            let width = head_dim;
            let v = vec![1.0f32; width];
            self.upload_f32(vec![width], &v)?
        };
        // `v_norm` is an unscaled RMSNorm in the pinned code; the contract
        // applies the scale vector, so the ones vector is the scale.
        let vn = self.rms_norm(v3_idx, ones, eps)?;

        let kr_bits = self.host_bits(kr)?;
        let vn_bits = self.host_bits(vn)?;
        if append {
            Self::append_cache(cache, layer, kv_heads, head_dim, kr_bits, vn_bits);
            let entry = &cache[layer];
            let kv_len = entry.2;
            let keys_idx = self.upload_bits(vec![kv_len, kv_heads, head_dim], entry.3.clone())?;
            let values_idx = self.upload_bits(vec![kv_len, kv_heads, head_dim], entry.4.clone())?;
            let mask_values = Self::build_mask(positions, kv_len, sliding, is_causal);
            let mask_idx = match mask_values {
                Some(values) => Some(self.upload_f32(vec![positions.len(), kv_len], &values)?),
                None => None,
            };
            let attn = self.attention(qr, keys_idx, values_idx, mask_idx, 1.0)?;
            let attn_bits = self.host_bits(attn)?;
            let flat = self.upload_bits(vec![tokens, heads * head_dim], attn_bits)?;
            let o_w = self.weight(&p("o_proj.weight"))?;
            self.linear(flat, o_w)
        } else {
            let entry = &cache[layer];
            let kv_len = entry.2 + tokens;
            let mut keys = entry.3.clone();
            keys.extend(kr_bits);
            let mut values = entry.4.clone();
            values.extend(vn_bits);
            let keys_idx = self.upload_bits(vec![kv_len, kv_heads, head_dim], keys)?;
            let values_idx = self.upload_bits(vec![kv_len, kv_heads, head_dim], values)?;
            let mask_values = Self::build_mask(positions, kv_len, sliding, is_causal);
            let mask_idx = match mask_values {
                Some(values) => Some(self.upload_f32(vec![positions.len(), kv_len], &values)?),
                None => None,
            };
            let attn = self.attention(qr, keys_idx, values_idx, mask_idx, 1.0)?;
            let attn_bits = self.host_bits(attn)?;
            let flat = self.upload_bits(vec![tokens, heads * head_dim], attn_bits)?;
            let o_w = self.weight(&p("o_proj.weight"))?;
            self.linear(flat, o_w)
        }
    }

    /// Dense gated MLP on device tensors: `gate`, `up`, `down` weight names.
    pub fn dense_mlp(
        &mut self,
        x: usize,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
    ) -> Result<usize, GraphError> {
        let gate_w = self.weight(gate_name)?;
        let up_w = self.weight(up_name)?;
        let down_w = self.weight(down_name)?;
        let g = self.linear(x, gate_w)?;
        let u = self.linear(x, up_w)?;
        let a = self.gelu_mul_up(g, u)?;
        self.linear(a, down_w)
    }

    /// Router: unscaled RMSNorm, per-column `scale * root_scale`, projection,
    /// widening, top-k. Returns the `[tokens, 2*top_k]` device tensor.
    pub fn router_layer(
        &mut self,
        x: usize,
        layer_prefix: &str,
        layer: usize,
        eps: f32,
        root_scale: f32,
        top_k: usize,
    ) -> Result<usize, GraphError> {
        let width = self.arena[x].layout().shape()[1];
        let ones = self.upload_f32(vec![width], &vec![1.0f32; width])?;
        let normed = self.rms_norm(x, ones, eps)?;
        let scale_w = self.weight(&format!("model.{layer_prefix}.layers.{layer}.router.scale"))?;
        // scale_bias multiplies by `scale` and subtracts `bias`; the root
        // factor is folded into the scale vector on the host.
        let scale_bits = self.host_bits(scale_w)?;
        let scale_values: Vec<f32> = Self::f32_of(&scale_bits)
            .iter()
            .map(|v| v * root_scale)
            .collect();
        let scale_idx = self.upload_f32(vec![width], &scale_values)?;
        let zero_bias = self.upload_f32(vec![width], &vec![0.0f32; width])?;
        let scaled = self.scale_bias(normed, scale_idx, zero_bias)?;
        let proj_w = self.weight(&format!(
            "model.{layer_prefix}.layers.{layer}.router.proj.weight"
        ))?;
        let scores = self.linear(scaled, proj_w)?;
        let scores_f32 = self.widen(scores)?;
        let per_expert = self.weight(&format!(
            "model.{layer_prefix}.layers.{layer}.router.per_expert_scale"
        ))?;
        self.router_top_k(scores_f32, per_expert, top_k)
    }

    /// MoE experts: gather, per-expert gate/up/down launches, weighted
    /// scatter-add back into `[tokens, hidden]`.
    pub fn experts_layer(
        &mut self,
        x: usize,
        topk_idx: usize,
        gate_up_name: &str,
        down_name: &str,
        experts: usize,
        inter: usize,
        top_k: usize,
    ) -> Result<usize, GraphError> {
        let tokens = self.arena[x].layout().shape()[0];
        let hidden = self.arena[x].layout().shape()[1];
        let pairs = tokens * top_k;
        let topk = self.download_f32(topk_idx)?;
        let weights = &topk[..pairs];
        let indices = &topk[pairs..];
        let indices_idx = self.upload_f32(vec![pairs], indices)?;
        let weights_idx = self.upload_f32(vec![pairs], weights)?;
        let gathered = self.gather_rows(x, indices_idx)?;
        let mut out_rows = Vec::new();
        // Group pairs by expert to launch one linear chain per expert.
        let mut by_expert: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for p in 0..pairs {
            by_expert.entry(indices[p] as usize).or_default().push(p);
        }
        // Launch per expert on the gathered rows, then scatter the results.
        for (e, pair_list) in by_expert {
            let gate_start = e * 2 * inter * hidden;
            let gate_w = self.weight_slice(gate_up_name, gate_start, vec![inter, hidden])?;
            let up_w = self.weight_slice(
                gate_up_name,
                gate_start + inter * hidden,
                vec![inter, hidden],
            )?;
            let down_w = self.weight_slice(down_name, e * hidden * inter, vec![hidden, inter])?;
            let sub_idx = self.upload_f32(
                vec![pair_list.len()],
                &pair_list.iter().map(|p| *p as f32).collect::<Vec<f32>>(),
            )?;
            let sub = self.gather_rows(gathered, sub_idx)?;
            let g = self.linear(sub, gate_w)?;
            let u = self.linear(sub, up_w)?;
            let a = self.gelu_mul_up(g, u)?;
            let o = self.linear(a, down_w)?;
            // Scatter the expert outputs back to the token rows.
            let rows = self.upload_f32(
                vec![pair_list.len()],
                &pair_list
                    .iter()
                    .map(|p| p / top_k)
                    .map(|t| t as f32)
                    .collect::<Vec<f32>>(),
            )?;
            let sub_weights = self.upload_f32(
                vec![pair_list.len()],
                &pair_list.iter().map(|p| weights[*p]).collect::<Vec<f32>>(),
            )?;
            let scattered = self.scatter_add(o, sub_weights, rows, tokens)?;
            out_rows.push(scattered);
        }
        let _ = experts;
        // Sum the per-expert scatter results into one tensor.
        let mut acc = out_rows
            .first()
            .copied()
            .ok_or(GraphError::BadIndex { index: 0 })?;
        for next in out_rows.iter().skip(1) {
            acc = self.add(acc, *next)?;
        }
        Ok(acc)
    }

    /// One full text layer (encoder or decoder), mirroring the pinned
    /// composition: input norm, attention, post-attention norm, residual,
    /// dense MLP + MoE with router, post norms, residual, layer scalar.
    #[allow(clippy::too_many_arguments)]
    pub fn text_layer(
        &mut self,
        prefix: &str,
        layer: usize,
        x: usize,
        cache: &mut Vec<(usize, usize, usize, Vec<Bf16>, Vec<Bf16>)>,
        positions: &[usize],
        head_dim: usize,
        heads: usize,
        kv_heads: usize,
        rope_theta: f32,
        partial: Option<f32>,
        sliding: Option<usize>,
        is_causal: bool,
        eps: f32,
        append: bool,
        hidden: usize,
        inter: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<usize, GraphError> {
        let norm_prefix = format!("model.decoder.layers.{layer}");
        let input_norm = self.weight(&format!("{norm_prefix}.input_layernorm.weight"))?;
        let normed = self.rms_norm(x, input_norm, eps)?;
        let attn = self.attention_layer(
            prefix, layer, normed, cache, positions, head_dim, heads, kv_heads, rope_theta,
            partial, sliding, is_causal, eps, append,
        )?;
        let post_attn = self.weight(&format!("{norm_prefix}.post_attention_layernorm.weight"))?;
        let post = self.rms_norm(attn, post_attn, eps)?;
        let x = self.add(x, post)?;
        // Dense branch.
        let pre_ff1 = self.weight(&format!("{norm_prefix}.pre_feedforward_layernorm.weight"))?;
        let pre1 = self.rms_norm(x, pre_ff1, eps)?;
        let mlp = self.dense_mlp(
            pre1,
            &format!("{norm_prefix}.mlp.gate_proj.weight"),
            &format!("{norm_prefix}.mlp.up_proj.weight"),
            &format!("{norm_prefix}.mlp.down_proj.weight"),
        )?;
        let pff1_w = self.weight(&format!(
            "{norm_prefix}.post_feedforward_layernorm_1.weight"
        ))?;
        let pff1 = self.rms_norm(mlp, pff1_w, eps)?;
        // MoE branch.
        let pre2_w = self.weight(&format!("{norm_prefix}.pre_feedforward_layernorm_2.weight"))?;
        let pre2 = self.rms_norm(x, pre2_w, eps)?;
        let root_scale = (hidden as f32).powf(-0.5);
        let topk = self.router_layer(pre2, "decoder", layer, eps, root_scale, top_k)?;
        let exp = self.experts_layer(
            pre2,
            topk,
            &format!("{norm_prefix}.experts.gate_up_proj"),
            &format!("{norm_prefix}.experts.down_proj"),
            experts,
            inter,
            top_k,
        )?;
        let pff2_w = self.weight(&format!(
            "{norm_prefix}.post_feedforward_layernorm_2.weight"
        ))?;
        let pff2 = self.rms_norm(exp, pff2_w, eps)?;
        let comb = self.add(pff1, pff2)?;
        let pff_w = self.weight(&format!("{norm_prefix}.post_feedforward_layernorm.weight"))?;
        let pff = self.rms_norm(comb, pff_w, eps)?;
        let y = self.add(x, pff)?;
        // Learned layer scalar as a constant scale on the last dimension.
        let scalar_w = self.weight(&format!("{norm_prefix}.layer_scalar"))?;
        let scalar_bits = self.host_bits(scalar_w)?;
        let scalar = Self::f32_of(&scalar_bits)[0];
        let scale_vec = self.upload_f32(vec![hidden], &vec![scalar; hidden])?;
        let zero_vec = self.upload_f32(vec![hidden], &vec![0.0f32; hidden])?;
        self.scale_bias(y, scale_vec, zero_vec)
    }

    /// Encoder causal prefill: fills the KV cache layer by layer and returns
    /// the final normalized hidden `[tokens, hidden]`.
    pub fn encoder_forward(
        &mut self,
        ids: &[u32],
        cache: &mut Vec<(usize, usize, usize, Vec<Bf16>, Vec<Bf16>)>,
    ) -> Result<usize, GraphError> {
        let text = self.checkpoint.config().text_config.clone().unwrap();
        let hidden = text.hidden_size.unwrap();
        let inter = text.intermediate_size.unwrap_or(704);
        let experts = text.num_experts.unwrap_or(128);
        let top_k = text.top_k_experts.unwrap_or(8);
        let eps = text.rms_norm_eps.unwrap_or(1e-6) as f32;
        let heads = text.num_attention_heads.unwrap();
        let mut x = self.embed_text(ids)?;
        let positions: Vec<usize> = (0..ids.len()).collect();
        for layer in 0..text.num_hidden_layers.unwrap() {
            let global = text.layer_types.get(layer) == Some(&"full_attention".to_owned());
            let head_dim = if global {
                text.global_head_dim.unwrap()
            } else {
                text.head_dim.unwrap()
            };
            let kv_heads = if global {
                text.num_global_key_value_heads.unwrap()
            } else {
                text.num_key_value_heads.unwrap()
            };
            let rope = if global { 1_000_000.0f32 } else { 10_000.0f32 };
            let partial = if global { Some(0.25f32) } else { None };
            let sliding = if global {
                None
            } else {
                Some(text.sliding_window.unwrap_or(1024))
            };
            x = self.text_layer(
                "decoder", layer, x, cache, &positions, head_dim, heads, kv_heads, rope, partial,
                sliding, true, eps, true, hidden, inter, experts, top_k,
            )?;
        }
        let final_norm = self.weight("model.decoder.norm.weight")?;
        self.rms_norm(x, final_norm, eps)
    }

    /// Decoder canvas pass with self-conditioning; `self_cond` is F32 logits
    /// `[canvas, vocab]` from the previous denoising step.
    pub fn decoder_forward(
        &mut self,
        canvas: &[u32],
        cache: &mut Vec<(usize, usize, usize, Vec<Bf16>, Vec<Bf16>)>,
        self_cond: Option<&Vec<f32>>,
    ) -> Result<usize, GraphError> {
        let text = self.checkpoint.config().text_config.clone().unwrap();
        let hidden = text.hidden_size.unwrap();
        let vocab = text.vocab_size.unwrap();
        let inter = text.intermediate_size.unwrap_or(704);
        let experts = text.num_experts.unwrap_or(128);
        let top_k = text.top_k_experts.unwrap_or(8);
        let eps = text.rms_norm_eps.unwrap_or(1e-6) as f32;
        let heads = text.num_attention_heads.unwrap();
        let scale = (hidden as f32).powf(-0.5);
        let embed = self.embed_text(canvas)?;
        let mut inputs = embed;
        if let Some(logits) = self_cond {
            // Softmax in F32 on host, cast to BF16, matmul with the embedding
            // table, embedding scale, self-conditioning MLP.
            let mut probs = vec![0.0f32; logits.len()];
            for r in 0..canvas.len() {
                let row = &logits[r * vocab..(r + 1) * vocab];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for i in 0..vocab {
                    let p = (row[i] - max).exp();
                    probs[r * vocab + i] = p;
                    sum += p;
                }
                for v in probs[r * vocab..(r + 1) * vocab].iter_mut() {
                    *v /= sum;
                }
            }
            let probs_idx = self.upload_bits(vec![canvas.len(), vocab], Self::bf16_of(&probs))?;
            let table = self.weight("model.decoder.embed_tokens.weight")?;
            let transposed = {
                let bits = self.read_payload("model.decoder.embed_tokens.weight")?;
                let mut out = Vec::with_capacity(bits.len());
                for col in 0..hidden {
                    for row in 0..vocab {
                        out.push(bits[row * hidden + col]);
                    }
                }
                out
            };
            let table_t = self.upload_bits(vec![vocab, hidden], transposed)?;
            let _ = table;
            let soft = self.matmul(probs_idx, table_t)?;
            let scale_vec = self.upload_f32(vec![1], &vec![scale])?;
            let soft_scaled = self.scale(soft, scale_vec)?;
            let pre_norm = self.weight("model.decoder.self_conditioning.pre_norm.weight")?;
            let normed = self.rms_norm(soft_scaled, pre_norm, eps)?;
            let sc = self.dense_mlp(
                normed,
                "model.decoder.self_conditioning.gate_proj.weight",
                "model.decoder.self_conditioning.up_proj.weight",
                "model.decoder.self_conditioning.down_proj.weight",
            )?;
            inputs = self.add(inputs, sc)?;
            let ones = self.upload_f32(vec![hidden], &vec![1.0f32; hidden])?;
            inputs = self.rms_norm(inputs, ones, eps)?;
        }
        let prefix_len = cache.first().map(|entry| entry.2).unwrap_or(0);
        let positions: Vec<usize> = (prefix_len..prefix_len + canvas.len()).collect();
        for layer in 0..text.num_hidden_layers.unwrap() {
            let global = text.layer_types.get(layer) == Some(&"full_attention".to_owned());
            let head_dim = if global {
                text.global_head_dim.unwrap()
            } else {
                text.head_dim.unwrap()
            };
            let kv_heads = if global {
                text.num_global_key_value_heads.unwrap()
            } else {
                text.num_key_value_heads.unwrap()
            };
            let rope = if global { 1_000_000.0f32 } else { 10_000.0f32 };
            let partial = if global { Some(0.25f32) } else { None };
            let sliding = if global {
                None
            } else {
                Some(text.sliding_window.unwrap_or(1024))
            };
            inputs = self.text_layer(
                "decoder", layer, inputs, cache, &positions, head_dim, heads, kv_heads, rope,
                partial, sliding, false, eps, false, hidden, inter, experts, top_k,
            )?;
        }
        let final_norm = self.weight("model.decoder.norm.weight")?;
        self.rms_norm(inputs, final_norm, eps)
    }

    /// Tied lm head with final softcapping in F32.
    pub fn lm_logits(&mut self, hidden_idx: usize) -> Result<usize, GraphError> {
        let text = self.checkpoint.config().text_config.clone().unwrap();
        let cap = text.final_logit_softcapping.unwrap() as f32;
        let hidden = text.hidden_size.unwrap();
        let vocab = text.vocab_size.unwrap();
        let transposed = {
            let bits = self.read_payload("model.decoder.embed_tokens.weight")?;
            let mut out = Vec::with_capacity(bits.len());
            for col in 0..hidden {
                for row in 0..vocab {
                    out.push(bits[row * hidden + col]);
                }
            }
            out
        };
        let table_t = self.upload_bits(vec![vocab, hidden], transposed)?;
        let logits = self.matmul(hidden_idx, table_t)?;
        let logits_f32 = self.widen(logits)?;
        let cap_vec = self.upload_f32(vec![1], &vec![cap])?;
        let divided = self.scale(logits_f32, cap_vec)?;
        self.softcap(divided, cap)
    }

    pub fn matmul(&mut self, lhs: usize, rhs: usize) -> Result<usize, GraphError> {
        let lhs = self.index(lhs)?;
        let rhs = self.index(rhs)?;
        self.finish(self.exec_op(TensorOp::MatMul { lhs, rhs })?)
    }

    /// Vision tower: patch embedder, 27 layers with axial 2D rope, pooling
    /// and standardization. Returns `(post_encoder [patches, hidden],
    /// pooled_standardized [out_len, hidden])` as arena indices.
    pub fn vision_forward(
        &mut self,
        pixels: &[f32],
        positions: &[(u32, u32)],
        pooling_kernel: usize,
    ) -> Result<(usize, usize), GraphError> {
        let vision = self.checkpoint.config().vision_config.clone().unwrap();
        let hidden = vision.hidden_size.unwrap();
        let heads = vision.num_attention_heads.unwrap();
        let kv_heads = vision.num_key_value_heads.unwrap();
        let head_dim = vision.head_dim.unwrap();
        let eps = vision.rms_norm_eps.unwrap_or(1e-6) as f32;
        let patches = positions.len();
        let per_patch = pixels.len() / patches;
        let scaled: Vec<f32> = pixels.iter().map(|v| 2.0 * (v - 0.5)).collect();
        let pixels_idx = self.upload_bits(vec![patches, per_patch], Self::bf16_of(&scaled))?;
        let input_proj =
            self.weight("model.encoder.vision_tower.patch_embedder.input_proj.weight")?;
        let mut x = self.linear(pixels_idx, input_proj)?;
        // Position embeddings: table [2, 10240, hidden], added on host.
        let table_bits = self
            .read_payload("model.encoder.vision_tower.patch_embedder.position_embedding_table")?;
        let psize = vision.position_embedding_size.unwrap();
        let tv = Self::f32_of(&table_bits);
        let mut x_bits = self.host_bits(x)?;
        for (i, (xpos, ypos)) in positions.iter().enumerate() {
            let xbase = (*xpos as usize) * hidden;
            let ybase = psize * hidden + (*ypos as usize) * hidden;
            for h in 0..hidden {
                let summed = x_bits[i * hidden + h].to_f32() + tv[xbase + h] + tv[ybase + h];
                x_bits[i * hidden + h] = Bf16::from_f32(summed);
            }
        }
        x = self.upload_bits(vec![patches, hidden], x_bits)?;
        let (cos, sin) = Self::vision_rope_tables(head_dim, 100.0f32, positions);
        let cos_idx = self.upload_f32(vec![patches, head_dim], &cos)?;
        let sin_idx = self.upload_f32(vec![patches, head_dim], &sin)?;
        for layer in 0..vision.num_hidden_layers.unwrap() {
            let p = |n: &str| format!("model.encoder.vision_tower.encoder.layers.{layer}.{n}");
            let input_norm = self.weight(&p("input_layernorm.weight"))?;
            let normed = self.rms_norm(x, input_norm, eps)?;
            let q_w = self.weight(&p("self_attn.q_proj.linear.weight"))?;
            let q = self.linear(normed, q_w)?;
            let q_bits = self.host_bits(q)?;
            let q3 = self.upload_bits(vec![patches, heads, head_dim], q_bits)?;
            let q_norm_w = self.weight(&p("self_attn.q_norm.weight"))?;
            let qn = self.rms_norm(q3, q_norm_w, eps)?;
            let qr = self.rope(qn, cos_idx, sin_idx)?;
            let k_w = self.weight(&p("self_attn.k_proj.linear.weight"))?;
            let k = self.linear(normed, k_w)?;
            let k_bits = self.host_bits(k)?;
            let k3 = self.upload_bits(vec![patches, kv_heads, head_dim], k_bits)?;
            let k_norm_w = self.weight(&p("self_attn.k_norm.weight"))?;
            let kn = self.rms_norm(k3, k_norm_w, eps)?;
            let kr = self.rope(kn, cos_idx, sin_idx)?;
            let v_w = self.weight(&p("self_attn.v_proj.linear.weight"))?;
            let v = self.linear(normed, v_w)?;
            let v_bits = self.host_bits(v)?;
            let v3 = self.upload_bits(vec![patches, kv_heads, head_dim], v_bits)?;
            let ones = self.upload_f32(vec![head_dim], &vec![1.0f32; head_dim])?;
            let vn = self.rms_norm(v3, ones, eps)?;
            let attn = self.attention(qr, kr, vn, None, 1.0)?;
            let attn_bits = self.host_bits(attn)?;
            let flat = self.upload_bits(vec![patches, heads * head_dim], attn_bits)?;
            let o_w = self.weight(&p("self_attn.o_proj.linear.weight"))?;
            let o = self.linear(flat, o_w)?;
            let post_attn = self.weight(&p("post_attention_layernorm.weight"))?;
            let post = self.rms_norm(o, post_attn, eps)?;
            x = self.add(x, post)?;
            let pre_ff = self.weight(&p("pre_feedforward_layernorm.weight"))?;
            let pre = self.rms_norm(x, pre_ff, eps)?;
            let mlp = self.dense_mlp(
                pre,
                &p("mlp.gate_proj.linear.weight"),
                &p("mlp.up_proj.linear.weight"),
                &p("mlp.down_proj.linear.weight"),
            )?;
            let pff_w = self.weight(&p("post_feedforward_layernorm.weight"))?;
            let pff = self.rms_norm(mlp, pff_w, eps)?;
            x = self.add(x, pff)?;
        }
        // Pooling and standardization run on host activations, then the
        // pooled result is re-uploaded through pinned staging.
        let out_len = patches / (pooling_kernel * pooling_kernel);
        let max_x = positions.iter().map(|p| p.0).max().unwrap() + 1;
        let xv = Self::f32_of(&self.host_bits(x)?);
        let inv_k2 = 1.0f32 / (pooling_kernel * pooling_kernel) as f32;
        let mut pooled = vec![0.0f32; out_len * hidden];
        for (i, (xpos, ypos)) in positions.iter().enumerate() {
            let gx = (*xpos as usize) / pooling_kernel;
            let gy = (*ypos as usize) / pooling_kernel;
            let group = gx + (max_x as usize / pooling_kernel) * gy;
            for h in 0..hidden {
                pooled[group * hidden + h] += xv[i * hidden + h] * inv_k2;
            }
        }
        let root = (hidden as f32).sqrt();
        let pooled_scaled: Vec<f32> = pooled.iter().map(|v| v * root).collect();
        let bias_bits = self.read_payload("model.encoder.vision_tower.std_bias")?;
        let scale_bits = self.read_payload("model.encoder.vision_tower.std_scale")?;
        let bv = Self::f32_of(&bias_bits);
        let sv = Self::f32_of(&scale_bits);
        let mut out = vec![0.0f32; out_len * hidden];
        for g in 0..out_len {
            for h in 0..hidden {
                out[g * hidden + h] = (pooled_scaled[g * hidden + h] - bv[h]) * sv[h];
            }
        }
        let pooled_idx = self.upload_bits(vec![out_len, hidden], Self::bf16_of(&out))?;
        Ok((x, pooled_idx))
    }

    /// `DiffusionGemmaMultimodalEmbedder`: unscaled RMSNorm + projection.
    pub fn embed_vision_forward(&mut self, pooled: usize) -> Result<usize, GraphError> {
        let vision = self.checkpoint.config().vision_config.clone().unwrap();
        let eps = vision.rms_norm_eps.unwrap_or(1e-6) as f32;
        let width = self.arena[pooled].layout().shape()[1];
        let ones = self.upload_f32(vec![width], &vec![1.0f32; width])?;
        let normed = self.rms_norm(pooled, ones, eps)?;
        let proj = self.weight("model.encoder.embed_vision.embedding_projection.weight")?;
        self.linear(normed, proj)
    }

    /// One full denoising block on the card: canvas -> decoder -> logits ->
    /// EntropyBound acceptance, repeating until the stopping criteria fire.
    /// `rng` draws uniform vocabulary indices. Returns the accepted canvas.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_block(
        &mut self,
        canvas: Vec<u32>,
        cache: &mut Vec<(usize, usize, usize, Vec<Bf16>, Vec<Bf16>)>,
        policy: &crate::generation_loop::SamplerPolicy,
        rng: &mut dyn FnMut(usize) -> usize,
    ) -> Result<Vec<u32>, GraphError> {
        use crate::generation_loop::{
            accept_canvas, mean_entropy, position_entropy, should_stop, CanvasState,
        };
        let vocab = policy.vocab_size;
        let temperature = 0.4f32;
        let mut state = CanvasState {
            canvas: canvas.clone(),
            accepted: vec![false; canvas.len()],
            logits: Vec::new(),
            step: 0,
            stable_count: 0,
            last_accepted: canvas.clone(),
        };
        let mut self_cond: Option<Vec<f32>> = None;
        for step in 0..policy.max_denoising_steps {
            // Linear temperature schedule t_min 0.4 -> t_max 0.8.
            let temperature =
                0.4 + (0.8 - 0.4) * (step as f32) / (policy.max_denoising_steps - 1) as f32;
            let hidden = self.decoder_forward(&state.canvas, cache, self_cond.as_ref())?;
            let logits_idx = self.lm_logits(hidden)?;
            let logits = self.download_f32(logits_idx)?;
            let entropies = position_entropy(&logits, canvas.len(), vocab, temperature);
            let argmax: Vec<u32> = (0..canvas.len())
                .map(|c| {
                    let row = &logits[c * vocab..(c + 1) * vocab];
                    let mut best = 0usize;
                    let mut best_value = f32::NEG_INFINITY;
                    for (i, v) in row.iter().enumerate() {
                        if *v > best_value {
                            best_value = *v;
                            best = i;
                        }
                    }
                    best as u32
                })
                .collect();
            let (canvas_next, accepted) = accept_canvas(&state, &argmax, &entropies, policy, rng);
            let mean = mean_entropy(&entropies);
            let stop = should_stop(&state, &accepted, &canvas_next, mean, policy);
            state.canvas = canvas_next;
            state.accepted = accepted;
            state.last_accepted = state.canvas.clone();
            state.step = step + 1;
            state.logits = logits;
            self_cond = Some(state.logits.clone());
            if stop {
                return Ok(state.canvas);
            }
        }
        Ok(state.canvas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_chunk_bound_is_positive() {
        assert!(READ_CHUNK_BYTES >= 1024 * 1024);
    }

    #[test]
    fn graph_errors_are_typed() {
        let error = GraphError::MissingWeight {
            tensor: "model.decoder.norm.weight".to_owned(),
        };
        assert!(format!("{error}").contains("not in the pinned inventory"));
        let error = GraphError::SliceOutOfBounds {
            name: "experts.gate_up_proj".to_owned(),
        };
        assert!(format!("{error}").contains("out of the tensor payload range"));
    }
}
