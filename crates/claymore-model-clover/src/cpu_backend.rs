//! Concrete CPU execution backend for the Clover graph.
//!
//! The backend explicitly owns every runtime resource of one generation
//! session: the validated checkpoint inventory and its bounded tile reader,
//! the FP32 GDN convolution/recurrent states, the FP32 KV caches, and the
//! M11.3 expert residency cache built on the shared memory manager. It
//! implements [`CloverGraphBackend`] by composing the pinned CPU primitives
//! from `claymore-tensor`; no arithmetic is re-derived here.
//!
//! State discipline:
//! - preview is strictly read-only with respect to clean GDN and KV state;
//! - causal prefill/replay advances each layer state exactly once, after all
//!   fallible reads for that layer have succeeded;
//! - a failing read leaves the caller's states untouched for that operation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use claymore_device::{CpuDevice, Device};
use claymore_memory::{EvictionPolicy, MemoryBudgets};
use claymore_runtime::CancellationToken;
use claymore_tensor::{
    attention_fp32_scores, attention_output_gate, block_conv_silu, block_preview, causal_conv_silu,
    full_attention_mask, gdn_gates, recurrent_commit, rms_norm_silu_gate, rms_norm_zero_centered,
    rope_tables_partial, rotary_embed_partial, Bf16, GdnDimensions, GdnOutputDType, Tensor,
};

use crate::residency::{
    prefetch_requests_for_routing, CloverExpertCache, CloverExpertCacheMetrics,
    CloverExpertGeometry, CloverExpertLoadRequest, CloverSharedExpertGeometry,
    CloverSharedExpertLoadRequest,
};
use crate::weights::{
    CloverCpuWeightStore, CloverFullAttentionGeometry, CloverFullAttentionWeights,
    CloverGdnGeometry, CloverGdnWeights,
};
use crate::{
    CloverActivation, CloverCheckpointInventory, CloverConfig, CloverGraphBackend, CloverLoadPlan,
    CloverNormKind, LayerType,
};

pub const EMBED_TENSOR: &str = "model.embed_tokens.weight";
pub const FINAL_NORM_TENSOR: &str = "model.norm.weight";
pub const LM_HEAD_TENSOR: &str = "lm_head.weight";

fn layer_norm_tensor(layer: usize, kind: CloverNormKind) -> String {
    let leaf = match kind {
        CloverNormKind::Input => "input_layernorm.weight",
        CloverNormKind::PostAttention => "post_attention_layernorm.weight",
    };
    format!("model.layers.{layer}.{leaf}")
}

fn router_tensor(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gate.weight")
}

#[derive(Debug, Clone)]
pub struct CloverCpuBackendOptions {
    pub expert_budgets: MemoryBudgets,
    pub eviction: EvictionPolicy,
    pub cancellation: Option<CancellationToken>,
}

struct GdnLayerState {
    conv_history: Vec<f32>,
    ssm_state: Vec<f32>,
    committed_tokens: usize,
}

struct KvLayerState {
    keys: Vec<f32>,
    values: Vec<f32>,
    tokens: usize,
}

enum LayerWeights {
    Gdn(CloverGdnWeights),
    Attention(CloverFullAttentionWeights),
}

/// Concrete fallible CPU backend used by generation and integration paths.
pub struct CloverCpuBackend<D: Device = CpuDevice> {
    config: CloverConfig,
    inventory: CloverCheckpointInventory,
    plan: CloverLoadPlan,
    root: PathBuf,
    device: CpuDevice,
    expert_cache: CloverExpertCache<D>,
    expert_geometry: CloverExpertGeometry,
    shared_geometry: CloverSharedExpertGeometry,
    weights: Vec<Option<LayerWeights>>,
    gdn: Vec<Option<GdnLayerState>>,
    kv: Vec<Option<KvLayerState>>,
    gdn_dims: GdnDimensions,
    gdn_channels: usize,
    gdn_key_width: usize,
    gdn_value_width: usize,
    kernel_width: usize,
    rope_theta: f64,
    partial_factor: f64,
    eps: f32,
    tick: u64,
    cancellation: Option<CancellationToken>,
    resident: std::collections::BTreeSet<(usize, usize)>,
    resident_shared: std::collections::BTreeSet<usize>,
    expert_hard_bytes: u64,
}

impl<D: Device> CloverCpuBackend<D> {
    /// Open a backend over an inspected checkpoint using the caller-provided
    /// expert device. Tests inject a device whose release fails to prove that
    /// cleanup errors are propagated and residency stays retryable (P4).
    pub fn new_with_expert_device(
        config: &CloverConfig,
        inventory: CloverCheckpointInventory,
        options: CloverCpuBackendOptions,
        expert_device: D,
    ) -> Result<Self, String> {
        let gdn_dims = GdnDimensions {
            key_heads: config.linear_num_key_heads,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
        };
        let gdn_key_width = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or("GDN key geometry overflow")?;
        let gdn_value_width = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or("GDN value geometry overflow")?;
        let gdn_channels = gdn_key_width
            .checked_mul(2)
            .and_then(|n| n.checked_add(gdn_value_width))
            .ok_or("GDN channel geometry overflow")?;
        let kernel_width = config.linear_conv_kernel_dim;
        if kernel_width < 2 {
            return Err("GDN convolution kernel width must be at least two".to_owned());
        }
        let store = CloverCpuWeightStore::new(&inventory);
        let mut weights = Vec::with_capacity(config.layer_types.len());
        let mut gdn = Vec::with_capacity(config.layer_types.len());
        let mut kv = Vec::with_capacity(config.layer_types.len());
        for (layer, layer_type) in config.layer_types.iter().copied().enumerate() {
            match layer_type {
                LayerType::LinearAttention => {
                    let resolved = store
                        .gdn_weights(
                            layer,
                            CloverGdnGeometry {
                                hidden_size: config.hidden_size,
                                key_heads: gdn_dims.key_heads,
                                key_dim: gdn_dims.key_dim,
                                value_heads: gdn_dims.value_heads,
                                value_dim: gdn_dims.value_dim,
                                kernel_width,
                            },
                        )
                        .map_err(|error| error.to_string())?;
                    weights.push(Some(LayerWeights::Gdn(resolved)));
                    gdn.push(Some(GdnLayerState {
                        conv_history: vec![0.0; gdn_channels * (kernel_width - 1)],
                        ssm_state: vec![
                            0.0;
                            gdn_dims.value_heads * gdn_dims.value_dim * gdn_dims.key_dim
                        ],
                        committed_tokens: 0,
                    }));
                    kv.push(None);
                }
                LayerType::FullAttention => {
                    let resolved = store
                        .full_attention_weights(
                            layer,
                            CloverFullAttentionGeometry {
                                hidden_size: config.hidden_size,
                                query_heads: config.num_attention_heads,
                                kv_heads: config.num_key_value_heads,
                                head_dim: config.head_dim,
                            },
                        )
                        .map_err(|error| error.to_string())?;
                    weights.push(Some(LayerWeights::Attention(resolved)));
                    gdn.push(None);
                    kv.push(Some(KvLayerState {
                        keys: Vec::new(),
                        values: Vec::new(),
                        tokens: 0,
                    }));
                }
            }
        }
        let expert_cache =
            CloverExpertCache::new(expert_device, options.expert_budgets, options.eviction)
                .map_err(|error| error.to_string())?;
        let root = inventory.root().to_path_buf();
        let plan = inventory.load_plan().map_err(|error| error.to_string())?;
        Ok(Self {
            config: config.clone(),
            root,
            inventory,
            plan,
            device: CpuDevice::new(),
            expert_cache,
            expert_geometry: CloverExpertGeometry::from_config(config),
            shared_geometry: CloverSharedExpertGeometry::from_config(config),
            weights,
            gdn,
            kv,
            gdn_dims,
            gdn_channels,
            gdn_key_width,
            gdn_value_width,
            kernel_width,
            rope_theta: config.rope_parameters.rope_theta,
            partial_factor: config.partial_rotary_factor,
            eps: config.rms_norm_eps as f32,
            tick: 0,
            cancellation: options.cancellation,
            resident: std::collections::BTreeSet::new(),
            resident_shared: std::collections::BTreeSet::new(),
            expert_hard_bytes: options.expert_budgets.vram_hard_bytes,
        })
    }

    pub fn gdn_geometry(&self) -> CloverGdnGeometry {
        CloverGdnGeometry {
            hidden_size: self.config.hidden_size,
            key_heads: self.gdn_dims.key_heads,
            key_dim: self.gdn_dims.key_dim,
            value_heads: self.gdn_dims.value_heads,
            value_dim: self.gdn_dims.value_dim,
            kernel_width: self.kernel_width,
        }
    }

    pub fn expert_hard_bytes(&self) -> u64 {
        self.expert_hard_bytes
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    fn store(&self) -> CloverCpuWeightStore<'_> {
        CloverCpuWeightStore::new(&self.inventory)
    }

    fn gdn_weights(&self, layer: usize) -> Result<&CloverGdnWeights, String> {
        match self.weights[layer].as_ref() {
            Some(LayerWeights::Gdn(weights)) => Ok(weights),
            _ => Err(format!("layer {layer} is not a GDN layer")),
        }
    }

    fn attention_weights(&self, layer: usize) -> Result<&CloverFullAttentionWeights, String> {
        match self.weights[layer].as_ref() {
            Some(LayerWeights::Attention(weights)) => Ok(weights),
            _ => Err(format!("layer {layer} is not a full-attention layer")),
        }
    }

    fn activation(&self, tokens: usize, values: Vec<Bf16>) -> Result<CloverActivation, String> {
        CloverActivation::new(tokens, self.config.hidden_size, values)
            .map_err(|error| error.to_string())
    }

    pub fn cache_metrics(&self) -> CloverExpertCacheMetrics {
        self.expert_cache.metrics()
    }

    pub fn expert_usage_bytes(&self) -> u64 {
        self.expert_cache.usage_bytes()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(claymore_runtime::CancellationToken::is_cancelled)
    }

    /// Test and diagnostics accessors: the exact clean state layout.
    pub fn gdn_conv_history(&self, layer: usize) -> &[f32] {
        self.gdn[layer]
            .as_ref()
            .map(|state| state.conv_history.as_slice())
            .unwrap_or(&[])
    }
    pub fn gdn_ssm_state(&self, layer: usize) -> &[f32] {
        self.gdn[layer]
            .as_ref()
            .map(|state| state.ssm_state.as_slice())
            .unwrap_or(&[])
    }
    pub fn kv_cache(&self, layer: usize) -> (&[f32], &[f32]) {
        match self.kv[layer].as_ref() {
            Some(state) => (state.keys.as_slice(), state.values.as_slice()),
            None => (&[], &[]),
        }
    }
    pub fn cached_kv_tokens(&self, layer: usize) -> usize {
        self.kv[layer].as_ref().map_or(0, |state| state.tokens)
    }

    /// Clean GDN state snapshot `(conv history, recurrent state)` for one
    /// layer, used by non-mutation and session-lifecycle tests.
    pub fn gdn_state(&self, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        self.gdn[layer]
            .as_ref()
            .map(|state| (state.conv_history.clone(), state.ssm_state.clone()))
    }

    /// Number of tokens causally committed into the GDN state of one layer.
    pub fn gdn_committed_tokens(&self, layer: usize) -> usize {
        self.gdn[layer]
            .as_ref()
            .map_or(0, |state| state.committed_tokens)
    }
}

impl CloverCpuBackend<CpuDevice> {
    /// Open a backend over an inspected checkpoint with the default host
    /// `CpuDevice` expert cache. Every layer's compact weights and projection
    /// shapes are resolved eagerly so a schema mismatch fails before any
    /// forward runs.
    pub fn new(
        config: &CloverConfig,
        inventory: CloverCheckpointInventory,
        options: CloverCpuBackendOptions,
    ) -> Result<Self, String> {
        Self::new_with_expert_device(config, inventory, options, CpuDevice::new())
    }
}

impl<D: Device> CloverGraphBackend for CloverCpuBackend<D> {
    fn embed_tokens(&mut self, token_ids: &[u32]) -> Result<CloverActivation, String> {
        let values = self
            .store()
            .embed_tokens(EMBED_TENSOR, token_ids)
            .map_err(|error| error.to_string())?;
        CloverActivation::new(token_ids.len(), self.config.hidden_size, values)
            .map_err(|error| error.to_string())
    }

    fn rms_norm(
        &mut self,
        layer: usize,
        kind: CloverNormKind,
        input: &CloverActivation,
    ) -> Result<CloverActivation, String> {
        let values = self
            .store()
            .rms_norm_zero_centered(
                &layer_norm_tensor(layer, kind),
                input.values(),
                input.tokens(),
                self.config.hidden_size,
                self.eps,
            )
            .map_err(|error| error.to_string())?;
        self.activation(input.tokens(), values)
    }

    fn forward_gdn(
        &mut self,
        layer: usize,
        input: &CloverActivation,
        valid: &[bool],
        preview: bool,
    ) -> Result<CloverActivation, String> {
        let tokens = input.tokens();
        if valid.len() != tokens {
            return Err(format!(
                "GDN validity mask has {} entries, expected {tokens}",
                valid.len()
            ));
        }
        let weights = self.gdn_weights(layer)?.clone();
        let hidden = self.config.hidden_size;
        // Pinned upstream zeroes invalid token rows once, before every GDN
        // projection (modeling_clover_moe.py CloverGatedDeltaNet.forward).
        let mut masked_x: Vec<Bf16> = input.values().to_vec();
        for (token, &is_valid) in valid.iter().enumerate() {
            if !is_valid {
                for value in &mut masked_x[token * hidden..(token + 1) * hidden] {
                    *value = Bf16::from_f32(0.0);
                }
            }
        }
        let raw = self
            .store()
            .linear(&weights.in_proj_qkv, &masked_x, tokens)
            .map_err(|error| error.to_string())?;
        let raw: Vec<f32> = raw.iter().map(|value| value.to_f32()).collect();
        let a = self
            .store()
            .linear(&weights.in_proj_a, &masked_x, tokens)
            .map_err(|error| error.to_string())?;
        let b = self
            .store()
            .linear(&weights.in_proj_b, &masked_x, tokens)
            .map_err(|error| error.to_string())?;
        let z = self
            .store()
            .linear(&weights.in_proj_z, &masked_x, tokens)
            .map_err(|error| error.to_string())?;
        let a: Vec<f32> = a.iter().map(|value| value.to_f32()).collect();
        let b: Vec<f32> = b.iter().map(|value| value.to_f32()).collect();

        let state = self.gdn[layer]
            .as_ref()
            .ok_or_else(|| format!("layer {layer} is not a GDN layer"))?;
        let (mixed, next_history, mut next_ssm) = if preview {
            let mixed = gdn_preview_conv_stage(
                &raw,
                tokens,
                self.gdn_channels,
                &state.conv_history,
                &weights.conv1d,
                &weights.conv_right,
                self.kernel_width,
                valid,
            )?;
            (mixed, None, None)
        } else {
            let (mixed, next_history) = gdn_causal_conv_stage(
                &raw,
                tokens,
                self.gdn_channels,
                &state.conv_history,
                &weights.conv1d,
                self.kernel_width,
                valid,
            );
            (mixed, Some(next_history), None)
        };
        let (q, k, v) = split_qkv(
            &mixed,
            tokens,
            self.gdn_channels,
            self.gdn_key_width,
            self.gdn_value_width,
        );
        let (decay, beta) = gdn_gates_stage(
            &a,
            &b,
            &weights.a_log,
            &weights.dt_bias,
            tokens,
            self.gdn_dims.value_heads,
        );
        let core = if preview {
            block_preview(
                self.gdn_dims,
                &q,
                &k,
                &v,
                &decay,
                valid,
                &state.ssm_state,
                (self.gdn_dims.key_dim as f32).sqrt().recip(),
                GdnOutputDType::BF16,
                &weights
                    .softmax_logit_scale
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                &weights
                    .softmax_history_bias
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                &weights
                    .softmax_history_c
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| error.to_string())?
        } else {
            let mut working = state.ssm_state.clone();
            let output = recurrent_commit(
                self.gdn_dims,
                &q,
                &k,
                &v,
                &decay,
                &beta,
                valid,
                GdnOutputDType::BF16,
                &mut working,
            )
            .map_err(|error| error.to_string())?;
            next_ssm.replace(working);
            output
        };
        let core_bf16: Vec<Bf16> = core.iter().map(|value| Bf16::from_f32(*value)).collect();
        let normed = gdn_norm_gate_stage(&self.device, &core_bf16, &weights.norm, &z, self.eps)?;
        // out_proj is the last fallible read of this layer: clean state is
        // advanced only after it succeeded.
        let output = self
            .store()
            .linear(&weights.out_proj, &normed, tokens)
            .map_err(|error| error.to_string())?;
        if let Some(history) = next_history {
            let state = self.gdn[layer]
                .as_mut()
                .ok_or_else(|| format!("layer {layer} is not a GDN layer"))?;
            state.conv_history = history;
            state.ssm_state = next_ssm.expect("causal path builds recurrent state");
            state.committed_tokens = state
                .committed_tokens
                .checked_add(tokens)
                .ok_or("GDN committed token count overflow")?;
        }
        self.activation(tokens, output)
    }

    fn forward_full_attention(
        &mut self,
        layer: usize,
        input: &CloverActivation,
        valid: &[bool],
        prefix_valid: &[bool],
        positions: &[usize],
        preview: bool,
    ) -> Result<CloverActivation, String> {
        let tokens = input.tokens();
        if valid.len() != tokens || positions.len() != tokens {
            return Err("attention validity or position mask mismatch".to_owned());
        }
        let weights = self.attention_weights(layer)?.clone();
        let x = input.values();
        let store = self.store();
        let q_proj_out = store
            .linear(&weights.q_proj, x, tokens)
            .map_err(|error| error.to_string())?;
        let k_proj_out = store
            .linear(&weights.k_proj, x, tokens)
            .map_err(|error| error.to_string())?;
        let v_proj_out = store
            .linear(&weights.v_proj, x, tokens)
            .map_err(|error| error.to_string())?;

        let state = self.kv[layer]
            .as_ref()
            .ok_or_else(|| format!("layer {layer} is not a full-attention layer"))?;
        if state.tokens != prefix_valid.len() {
            return Err(format!(
                "attention cache holds {} tokens but the prefix validity mask has {}",
                state.tokens,
                prefix_valid.len()
            ));
        }
        let stage = attention_stage(
            &self.device,
            AttentionDType::F32,
            &q_proj_out,
            &k_proj_out,
            &v_proj_out,
            &state.keys,
            &state.values,
            prefix_valid,
            valid,
            positions,
            preview,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            &weights.q_norm,
            &weights.k_norm,
            self.eps,
            self.rope_theta,
            self.partial_factor,
        )?;

        // o_proj is the last fallible read: the cache is appended only after
        // the whole causal operation succeeded.
        let output = store
            .linear(&weights.o_proj, &stage.gated, tokens)
            .map_err(|error| error.to_string())?;
        if !preview {
            let state = self.kv[layer]
                .as_mut()
                .ok_or_else(|| format!("layer {layer} is not a full-attention layer"))?;
            state.keys.extend_from_slice(&stage.block_keys);
            state.values.extend_from_slice(&stage.block_values);
            state.tokens = state
                .tokens
                .checked_add(tokens)
                .ok_or("attention cache token count overflow")?;
        }
        self.activation(tokens, output)
    }

    fn router_logits(
        &mut self,
        layer: usize,
        input: &CloverActivation,
    ) -> Result<Vec<f32>, String> {
        self.store()
            .router_logits(&router_tensor(layer), input.values(), input.tokens())
            .map_err(|error| error.to_string())
    }

    fn forward_moe(
        &mut self,
        layer: usize,
        input: &CloverActivation,
        routing: &crate::CloverRoutingPlan,
        _valid: &[bool],
        _preview: bool,
    ) -> Result<CloverActivation, String> {
        let tokens = input.tokens();
        // Deterministic prefetch order (priority descending, then layer and
        // expert ascending), identical to CloverExpertCache::prefetch_ordered.
        let mut requests = prefetch_requests_for_routing(routing, layer, 0);
        requests.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.layer.cmp(&right.layer))
                .then_with(|| left.expert.cmp(&right.expert))
        });
        let cancellation = self.cancellation.clone();
        let cancelled = move || {
            cancellation
                .as_ref()
                .is_some_and(claymore_runtime::CancellationToken::is_cancelled)
        };
        for request in requests {
            let tick = self.next_tick();
            self.expert_cache
                .ensure_resident_with_cancel(
                    CloverExpertLoadRequest {
                        checkpoint_root: &self.root,
                        plan: &self.plan,
                        layer,
                        expert: request.expert,
                        geometry: self.expert_geometry,
                        tick,
                    },
                    &cancelled,
                )
                .map_err(|error| format!("expert {request:?} residency failed: {error}"))?;
            self.resident.insert((layer, request.expert));
        }
        // Read each routed expert's weights immediately after its residency
        // check: under a budget smaller than the routed set, the LRU cache may
        // evict an expert between a separate ensure pass and its use.
        let mut routed = BTreeSet::new();
        for route in &routing.routes {
            for (&expert, &weight) in route.expert_ids.iter().zip(&route.weights) {
                if weight > 0.0 {
                    routed.insert(expert);
                }
            }
        }
        let mut experts = BTreeMap::new();
        for expert in routed {
            let tick = self.next_tick();
            self.expert_cache
                .ensure_resident_with_cancel(
                    CloverExpertLoadRequest {
                        checkpoint_root: &self.root,
                        plan: &self.plan,
                        layer,
                        expert,
                        geometry: self.expert_geometry,
                        tick,
                    },
                    &cancelled,
                )
                .map_err(|error| format!("expert {expert} residency failed: {error}"))?;
            self.resident.insert((layer, expert));
            let tick = self.next_tick();
            experts.insert(
                expert,
                self.expert_cache
                    .read_routed_weights(layer, expert, self.expert_geometry, tick)
                    .map_err(|error| format!("expert {expert} read failed: {error}"))?,
            );
        }
        let shared = if self.config.num_shared_experts > 0 {
            let tick = self.next_tick();
            self.expert_cache
                .ensure_shared_expert_resident_with_cancel(
                    CloverSharedExpertLoadRequest {
                        checkpoint_root: &self.root,
                        plan: &self.plan,
                        layer,
                        geometry: self.shared_geometry,
                        tick,
                    },
                    &cancelled,
                )
                .map_err(|error| format!("shared expert residency failed: {error}"))?;
            self.resident_shared.insert(layer);
            let tick = self.next_tick();
            Some(
                self.expert_cache
                    .read_shared_expert_weights(layer, self.shared_geometry, tick)
                    .map_err(|error| format!("shared expert read failed: {error}"))?,
            )
        } else {
            None
        };
        let values = crate::moe::execute_moe_cpu(
            input.values(),
            tokens,
            self.config.hidden_size,
            routing,
            &experts,
            shared.as_ref(),
        )
        .map_err(|error| error.to_string())?;
        self.activation(tokens, values)
    }

    fn final_norm(&mut self, input: &CloverActivation) -> Result<CloverActivation, String> {
        let values = self
            .store()
            .rms_norm_zero_centered(
                FINAL_NORM_TENSOR,
                input.values(),
                input.tokens(),
                self.config.hidden_size,
                self.eps,
            )
            .map_err(|error| error.to_string())?;
        self.activation(input.tokens(), values)
    }

    fn lm_head(&mut self, input: &CloverActivation) -> Result<Vec<f32>, String> {
        let logits = self
            .store()
            .linear(LM_HEAD_TENSOR, input.values(), input.tokens())
            .map_err(|error| error.to_string())?;
        Ok(logits.iter().map(|value| value.to_f32()).collect())
    }

    fn reset_session(&mut self) -> Result<(), String> {
        for state in self.gdn.iter_mut().flatten() {
            state.conv_history.fill(0.0);
            state.ssm_state.fill(0.0);
            state.committed_tokens = 0;
        }
        for state in self.kv.iter_mut().flatten() {
            state.keys.clear();
            state.values.clear();
            state.tokens = 0;
        }
        // Release every expert this request made resident. A failed release
        // must keep the residency identifier so the next attempt can retry it
        // (M11.4 review P4): dropping the id would strand the allocation with
        // no way to free it. All failures are collected and reported.
        let mut errors: Vec<String> = Vec::new();
        let resident = std::mem::take(&mut self.resident);
        for (layer, expert) in resident {
            let tick = self.next_tick();
            if let Err(error) = self.expert_cache.release(layer, expert, tick) {
                self.resident.insert((layer, expert));
                errors.push(format!(
                    "expert release layer={layer} expert={expert}: {error}"
                ));
            }
        }
        let shared = std::mem::take(&mut self.resident_shared);
        for layer in shared {
            let tick = self.next_tick();
            if let Err(error) = self.expert_cache.release_shared_expert(layer, tick) {
                self.resident_shared.insert(layer);
                errors.push(format!("shared expert release layer={layer}: {error}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

// ---------------------------------------------------------------------------
// Composition stages over the pinned claymore-tensor primitives.
//
// These wrappers own no semantics of their own: they map checkpoint tensor
// layouts onto the pinned reference signatures. M11.4 tests drive each stage
// directly against the upstream CPU fixtures and the full forward against an
// independent composition of the same primitives.
// ---------------------------------------------------------------------------

/// One host BF16 tensor with contiguous layout.
fn bf16_tensor(device: &CpuDevice, values: &[Bf16], shape: Vec<usize>) -> Result<Tensor, String> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    host_tensor(device, &bytes, shape, claymore_device::DType::BF16)
}

fn f32_tensor(device: &CpuDevice, values: &[f32], shape: Vec<usize>) -> Result<Tensor, String> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    host_tensor(device, &bytes, shape, claymore_device::DType::F32)
}

fn host_tensor(
    device: &CpuDevice,
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: claymore_device::DType,
) -> Result<Tensor, String> {
    let mut storage = device
        .allocate(claymore_core::MemoryTier::Host, bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    device
        .write(storage.as_mut(), 0, bytes)
        .map_err(|error| error.to_string())?;
    Tensor::new(storage, shape, dtype).map_err(|error| error.to_string())
}

fn tensor_to_bf16(device: &CpuDevice, tensor: &Tensor) -> Result<Vec<Bf16>, String> {
    let bytes = read_host(device, tensor)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| Bf16::from_ne_bytes([pair[0], pair[1]]))
        .collect())
}

fn tensor_to_f32(device: &CpuDevice, tensor: &Tensor) -> Result<Vec<f32>, String> {
    let bytes = read_host(device, tensor)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four byte chunk")))
        .collect())
}

fn read_host(device: &CpuDevice, tensor: &Tensor) -> Result<Vec<u8>, String> {
    device
        .read(
            tensor.storage(),
            tensor.offset_bytes(),
            tensor.layout().span_bytes(),
        )
        .map_err(|error| error.to_string())
}

/// Pinned GDN gate derivation: decay stays FP32, beta is rounded to the BF16
/// activation dtype exactly as the pinned module executes it.
pub(crate) fn gdn_gates_stage(
    a: &[f32],
    b: &[f32],
    a_log: &[Bf16],
    dt_bias: &[Bf16],
    tokens: usize,
    value_heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let a_log: Vec<f32> = a_log.iter().map(|value| value.to_f32()).collect();
    let dt_bias: Vec<f32> = dt_bias.iter().map(|value| value.to_f32()).collect();
    gdn_gates(
        a,
        b,
        &a_log,
        &dt_bias,
        tokens,
        value_heads,
        GdnOutputDType::BF16,
    )
    .expect("GDN gate shapes are validated at backend construction")
}

/// Causal GDN convolution over the flattened `[tokens, channels]` projection.
/// Weights use the checkpoint `[channels, 1, width]` layout.
pub(crate) fn gdn_causal_conv_stage(
    raw: &[f32],
    tokens: usize,
    channels: usize,
    history: &[f32],
    weights: &[Bf16],
    kernel_width: usize,
    valid: &[bool],
) -> (Vec<f32>, Vec<f32>) {
    let weights: Vec<f32> = weights.iter().map(|value| value.to_f32()).collect();
    causal_conv_silu(
        raw,
        tokens,
        channels,
        history,
        &weights,
        kernel_width,
        valid,
        GdnOutputDType::BF16,
    )
    .expect("GDN convolution shapes are validated at backend construction")
}

/// Bidirectional block-preview convolution reading clean history only, with
/// independent right-context weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_preview_conv_stage(
    raw: &[f32],
    tokens: usize,
    channels: usize,
    clean_history: &[f32],
    weights: &[Bf16],
    right_weights: &[Bf16],
    kernel_width: usize,
    valid: &[bool],
) -> Result<Vec<f32>, String> {
    let weights: Vec<f32> = weights.iter().map(|value| value.to_f32()).collect();
    let right_weights: Vec<f32> = right_weights.iter().map(|value| value.to_f32()).collect();
    block_conv_silu(
        raw,
        tokens,
        channels,
        clean_history,
        &weights,
        &right_weights,
        kernel_width,
        valid,
        GdnOutputDType::BF16,
    )
    .map_err(|error| error.to_string())
}

/// Split the post-convolution `[tokens, channels]` mixed projection into the
/// pinned `(q, k, v)` head layouts.
pub(crate) fn split_qkv(
    mixed: &[f32],
    tokens: usize,
    channels: usize,
    key_width: usize,
    value_width: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; tokens * key_width];
    let mut k = vec![0.0f32; tokens * key_width];
    let mut v = vec![0.0f32; tokens * value_width];
    for token in 0..tokens {
        let row = &mixed[token * channels..(token + 1) * channels];
        q[token * key_width..(token + 1) * key_width].copy_from_slice(&row[..key_width]);
        k[token * key_width..(token + 1) * key_width]
            .copy_from_slice(&row[key_width..2 * key_width]);
        v[token * value_width..(token + 1) * value_width]
            .copy_from_slice(&row[2 * key_width..2 * key_width + value_width]);
    }
    (q, k, v)
}

/// Pinned GDN output normalization: non-zero-centered RMSNorm over the value
/// head width, gated by `silu(z)`, rounded once to BF16.
pub(crate) fn gdn_norm_gate_stage(
    device: &CpuDevice,
    core: &[Bf16],
    norm_weight: &[Bf16],
    gate: &[Bf16],
    epsilon: f32,
) -> Result<Vec<Bf16>, String> {
    let width = norm_weight.len();
    if width == 0 || core.len() % width != 0 || gate.len() != core.len() {
        return Err("GDN norm gate layout mismatch".to_owned());
    }
    let rows = core.len() / width;
    let input = bf16_tensor(device, core, vec![rows, width])?;
    let scale = bf16_tensor(device, norm_weight, vec![width])?;
    let gate = bf16_tensor(device, gate, vec![rows, width])?;
    let output = rms_norm_silu_gate(device, &input, &scale, &gate, epsilon)
        .map_err(|error| error.to_string())?;
    tensor_to_bf16(device, &output)
}

/// Pinned zero-centered RMSNorm over fixed-width rows.
pub(crate) fn zero_centered_norm_stage(
    device: &CpuDevice,
    rows: &[Bf16],
    row_count: usize,
    width: usize,
    weight: &[Bf16],
    epsilon: f32,
) -> Result<Vec<Bf16>, String> {
    if width == 0 || weight.len() != width || rows.len() != row_count * width {
        return Err("zero-centered norm layout mismatch".to_owned());
    }
    let input = bf16_tensor(device, rows, vec![row_count, width])?;
    let scale = bf16_tensor(device, weight, vec![width])?;
    let output = rms_norm_zero_centered(device, &input, &scale, epsilon)
        .map_err(|error| error.to_string())?;
    tensor_to_bf16(device, &output)
}

/// Pinned partial RoPE over `[tokens, heads, head_dim]` BF16 rows. Cosine and
/// sine tables are rounded to the BF16 activation dtype before the rotation,
/// exactly as `CloverAttention.rotary` stores them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_stage(
    device: &CpuDevice,
    data: &[Bf16],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    positions: &[usize],
    theta: f64,
    partial_factor: f64,
) -> Result<Vec<Bf16>, String> {
    if positions.len() != tokens || data.len() != tokens * heads * head_dim {
        return Err("RoPE layout mismatch".to_owned());
    }
    let positions: Vec<u32> = positions
        .iter()
        .map(|position| {
            u32::try_from(*position).map_err(|_| format!("position {position} exceeds u32 range"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (cos, sin) = rope_tables_partial(&positions, head_dim, theta, partial_factor)
        .map_err(|error| error.to_string())?;
    let rotary_dim = cos.len() / tokens;
    let cos_bf16: Vec<Bf16> = cos.iter().map(|value| Bf16::from_f32(*value)).collect();
    let sin_bf16: Vec<Bf16> = sin.iter().map(|value| Bf16::from_f32(*value)).collect();
    let input = bf16_tensor(device, data, vec![tokens, heads, head_dim])?;
    let cos = bf16_tensor(device, &cos_bf16, vec![tokens, rotary_dim])?;
    let sin = bf16_tensor(device, &sin_bf16, vec![tokens, rotary_dim])?;
    let output =
        rotary_embed_partial(device, &input, &cos, &sin).map_err(|error| error.to_string())?;
    tensor_to_bf16(device, &output)
}

/// Activation dtype driving the attention score and value chain. FP32 selects
/// the runtime contract with FP32 KV state; BF16 reproduces the pinned M11.2
/// fixtures where the cached values are BF16.
#[derive(Clone, Copy)]
pub(crate) enum AttentionDType {
    /// Reproduces the pinned M11.2 attention fixtures where the KV cache is
    /// BF16; the runtime backend itself uses the FP32 cache. Constructed only
    /// by the fixture-parity test.
    #[allow(dead_code)]
    Bf16,
    F32,
}

pub(crate) struct AttentionStageOutput {
    /// Gated attention activations `[tokens, query_heads * head_dim]` in the
    /// BF16 model dtype, ready for `o_proj`.
    pub gated: Vec<Bf16>,
    /// Normalized, rotated new keys widened to the FP32 state layout.
    pub block_keys: Vec<f32>,
    /// New values widened to the FP32 state layout.
    pub block_values: Vec<f32>,
}

/// One complete full-attention operation over the pinned primitives: Q/gate
/// de-interleaving, zero-centered Q/K norms, partial RoPE, prefix-plus-block
/// masking, grouped-query attention with FP32 scores, and the output gate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_stage(
    device: &CpuDevice,
    dtype: AttentionDType,
    q_proj_out: &[Bf16],
    k_proj_out: &[Bf16],
    v_proj_out: &[Bf16],
    cached_keys: &[f32],
    cached_values: &[f32],
    prefix_valid: &[bool],
    block_valid: &[bool],
    positions: &[usize],
    preview: bool,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_norm_weight: &[Bf16],
    k_norm_weight: &[Bf16],
    epsilon: f32,
    rope_theta: f64,
    partial_factor: f64,
) -> Result<AttentionStageOutput, String> {
    let tokens = block_valid.len();
    let query_width = query_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    if tokens == 0
        || q_proj_out.len() != tokens * 2 * query_width
        || k_proj_out.len() != tokens * kv_width
        || v_proj_out.len() != tokens * kv_width
    {
        return Err("attention projection layout mismatch".to_owned());
    }
    // Pinned split: `q_proj` output views each head as [head | gate]
    // (modeling_clover_moe.py `view(b, t, heads, 2 * head_dim).chunk(2, -1)`).
    let mut q = Vec::with_capacity(tokens * query_width);
    let mut gate = Vec::with_capacity(tokens * query_width);
    for row in q_proj_out.chunks_exact(2 * query_width) {
        for head in 0..query_heads {
            let base = head * 2 * head_dim;
            q.extend_from_slice(&row[base..base + head_dim]);
            gate.extend_from_slice(&row[base + head_dim..base + 2 * head_dim]);
        }
    }
    let q_normed = zero_centered_norm_stage(
        device,
        &q,
        tokens * query_heads,
        head_dim,
        q_norm_weight,
        epsilon,
    )?;
    let k_normed = zero_centered_norm_stage(
        device,
        k_proj_out,
        tokens * kv_heads,
        head_dim,
        k_norm_weight,
        epsilon,
    )?;
    let q_rotated = rope_stage(
        device,
        &q_normed,
        tokens,
        query_heads,
        head_dim,
        positions,
        rope_theta,
        partial_factor,
    )?;
    let k_rotated = rope_stage(
        device,
        &k_normed,
        tokens,
        kv_heads,
        head_dim,
        positions,
        rope_theta,
        partial_factor,
    )?;

    let cached_tokens = cached_keys.len() / kv_width;
    if cached_tokens != prefix_valid.len() || cached_values.len() != cached_keys.len() {
        return Err("KV cache length does not match the prefix validity mask".to_owned());
    }
    let mask = full_attention_mask(prefix_valid, block_valid, preview)
        .map_err(|error| error.to_string())?;
    let scale = (head_dim as f32).sqrt().recip();

    let gated = match dtype {
        AttentionDType::Bf16 => {
            let mut keys: Vec<Bf16> = cached_keys
                .iter()
                .map(|value| Bf16::from_f32(*value))
                .collect();
            keys.extend_from_slice(&k_rotated);
            let mut values: Vec<Bf16> = cached_values
                .iter()
                .map(|value| Bf16::from_f32(*value))
                .collect();
            values.extend_from_slice(v_proj_out);
            let q_tensor = bf16_tensor(device, &q_rotated, vec![tokens, query_heads, head_dim])?;
            let k_tensor = bf16_tensor(
                device,
                &keys,
                vec![keys.len() / kv_width, kv_heads, head_dim],
            )?;
            let v_tensor = bf16_tensor(
                device,
                &values,
                vec![values.len() / kv_width, kv_heads, head_dim],
            )?;
            let mask_tensor = f32_tensor(device, &mask, vec![tokens, prefix_valid.len() + tokens])?;
            let attended = attention_fp32_scores(
                device,
                &q_tensor,
                &k_tensor,
                &v_tensor,
                Some(&mask_tensor),
                scale,
            )
            .map_err(|error| error.to_string())?;
            let attended = tensor_to_bf16(device, &attended)?;
            let attended: Vec<Bf16> = {
                // [tokens, query_heads, head_dim] -> [tokens, query_width]
                attended
            };
            let attended_flat: Vec<Bf16> = attended;
            let attended = bf16_tensor(device, &attended_flat, vec![tokens, query_width])?;
            let gate_tensor = bf16_tensor(device, &gate, vec![tokens, query_width])?;
            let gated = attention_output_gate(device, &attended, &gate_tensor)
                .map_err(|error| error.to_string())?;
            tensor_to_bf16(device, &gated)?
        }
        AttentionDType::F32 => {
            let mut keys: Vec<f32> = cached_keys.to_vec();
            keys.extend(k_rotated.iter().map(|value| value.to_f32()));
            let mut values: Vec<f32> = cached_values.to_vec();
            values.extend(v_proj_out.iter().map(|value| value.to_f32()));
            let q_f32: Vec<f32> = q_rotated.iter().map(|value| value.to_f32()).collect();
            let q_tensor = f32_tensor(device, &q_f32, vec![tokens, query_heads, head_dim])?;
            let k_tensor = f32_tensor(
                device,
                &keys,
                vec![keys.len() / kv_width, kv_heads, head_dim],
            )?;
            let v_tensor = f32_tensor(
                device,
                &values,
                vec![values.len() / kv_width, kv_heads, head_dim],
            )?;
            let mask_tensor = f32_tensor(device, &mask, vec![tokens, prefix_valid.len() + tokens])?;
            let attended = attention_fp32_scores(
                device,
                &q_tensor,
                &k_tensor,
                &v_tensor,
                Some(&mask_tensor),
                scale,
            )
            .map_err(|error| error.to_string())?;
            let attended = tensor_to_f32(device, &attended)?;
            let gate_f32: Vec<f32> = gate.iter().map(|value| value.to_f32()).collect();
            let attended = f32_tensor(device, &attended, vec![tokens, query_width])?;
            let gate_tensor = f32_tensor(device, &gate_f32, vec![tokens, query_width])?;
            let gated = attention_output_gate(device, &attended, &gate_tensor)
                .map_err(|error| error.to_string())?;
            let gated = tensor_to_f32(device, &gated)?;
            // Activations stay BF16 for the downstream dense projections.
            gated.iter().map(|value| Bf16::from_f32(*value)).collect()
        }
    };

    let block_keys: Vec<f32> = k_rotated.iter().map(|value| value.to_f32()).collect();
    let block_values: Vec<f32> = v_proj_out.iter().map(|value| value.to_f32()).collect();
    Ok(AttentionStageOutput {
        gated,
        block_keys,
        block_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_checkpoint::{tiny_clover_config, SynthCloverCheckpoint, SYNTH_MODEL_SHARD};
    use crate::{CloverGraph, CloverMoeRouter, CloverRoutingPlan};
    use claymore_tensor::{clover_gdn_upstream_fixtures as fixtures, GdnOutputDType};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "claymore-clover-backend-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn budgets(experts: usize, layer_expert_bytes: u64, shared_bytes: u64) -> MemoryBudgets {
        let hard = layer_expert_bytes * experts as u64 + shared_bytes;
        MemoryBudgets {
            vram_soft_bytes: hard / 2,
            vram_hard_bytes: hard,
            host_bytes: hard,
            pinned_host_bytes: hard,
            expert_cache_target_bytes: hard,
            expert_cache_max_bytes: hard,
        }
    }

    fn open_backend(
        config: &CloverConfig,
        root: &std::path::Path,
        budget_experts: usize,
    ) -> CloverCpuBackend {
        let inventory = CloverCheckpointInventory::inspect(root).unwrap();
        let expert_bytes = (config.moe_intermediate_size * 2 * config.hidden_size
            + config.hidden_size * config.moe_intermediate_size)
            * 2;
        let shared_bytes = (3 * config.shared_expert_intermediate_size * config.hidden_size
            + config.hidden_size)
            * 2;
        let cancellation = CancellationToken::new();
        CloverCpuBackend::new(
            config,
            inventory,
            CloverCpuBackendOptions {
                expert_budgets: budgets(budget_experts, expert_bytes as u64, shared_bytes as u64),
                eviction: EvictionPolicy::Lru,
                cancellation: Some(cancellation),
            },
        )
        .unwrap()
    }

    fn fixture_backend(label: &str, budget_experts: usize) -> (PathBuf, CloverCpuBackend) {
        let config = tiny_clover_config();
        let root = temp_dir(label);
        SynthCloverCheckpoint::new(&config).write(&root);
        let backend = open_backend(&config, &root, budget_experts);
        (root, backend)
    }

    fn bf16(values: &[f32]) -> Vec<Bf16> {
        values.iter().copied().map(Bf16::from_f32).collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], bound: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
            let tolerance = bound * right.abs().max(1.0);
            assert!(
                (left - right).abs() <= tolerance,
                "{label}[{index}]: {left} vs {right}"
            );
        }
    }

    fn route(
        config: &CloverConfig,
        logits: &[f32],
        tokens: usize,
        valid: &[bool],
        preview: bool,
    ) -> CloverRoutingPlan {
        CloverMoeRouter::from_config(config)
            .route(logits, tokens, valid, preview)
            .unwrap()
    }

    // -- GDN stage parity against pinned upstream fixtures ------------------

    #[test]
    fn gdn_gates_stage_matches_pinned_gate_fixture() {
        // Fixture inputs from tools/m11/gen_cpu_reference.py GATE block.
        let a = [0.2f32, -0.4, 0.7, 0.1, -0.3, 0.6];
        let b = [0.3f32, -0.8, 0.5, 0.2, -0.1, 0.9];
        let a_log = bf16(&[0.1, -0.2]);
        let dt_bias = bf16(&[1.0, 0.8]);
        let (decay, beta) = gdn_gates_stage(&a, &b, &a_log, &dt_bias, 3, 2);
        assert_close(&decay, fixtures::GATE_DECAY, 2f32.powi(-6), "gate decay");
        assert_close(&beta, fixtures::GATE_BETA, 2f32.powi(-7), "gate beta");
    }

    #[test]
    fn gdn_causal_conv_stage_matches_pinned_conv_fixture() {
        let raw = [0.2f32, -0.4, 0.0, 0.0, 0.7, 0.1, -0.3, 0.6];
        let history = vec![0.1f32, -0.2, 0.3, -0.4, 0.2, 0.05];
        let weights = bf16(&[0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3]);
        let valid = [true, false, true, true];
        let (output, next_history) =
            gdn_causal_conv_stage(&raw, 4, 2, &history, &weights, 4, &valid);
        assert_close(&output, fixtures::CAUSAL_OUTPUT, 2f32.powi(-7), "conv out");
        assert_close(
            &next_history,
            fixtures::CAUSAL_STATE_OUT,
            2f32.powi(-7),
            "conv history",
        );
    }

    #[test]
    fn gdn_preview_conv_stage_matches_pinned_block_fixture() {
        let raw = [0.2f32, -0.4, 0.7, 0.1, -0.3, 0.6];
        let history = vec![0.1f32, -0.2, 0.3, -0.4, 0.2, 0.05];
        let weights = bf16(&[0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3]);
        let right = bf16(&[0.15, -0.25, 0.35, -0.1, 0.2, 0.3]);
        let valid = [true, true, false];
        let output =
            gdn_preview_conv_stage(&raw, 3, 2, &history, &weights, &right, 4, &valid).unwrap();
        assert_close(&output, fixtures::BLOCK_OUTPUT, 2f32.powi(-7), "block conv");
    }

    #[test]
    fn gdn_norm_gate_stage_matches_pinned_rmsnorm_fixture() {
        let input = bf16(&[0.2, -0.7, 1.1, 0.3, -0.4, 0.6, 0.1, -0.9]);
        let weight = bf16(&[0.5, 1.0, 1.5, 0.25]);
        let gate = bf16(&[-0.2, 0.4, 0.8, -1.0, 0.3, -0.5, 0.1, 0.7]);
        let device = CpuDevice::new();
        let output = gdn_norm_gate_stage(&device, &input, &weight, &gate, 1e-6).unwrap();
        let actual: Vec<f32> = output.iter().map(|value| value.to_f32()).collect();
        assert_close(
            &actual,
            fixtures::NORM_GATE_OUT,
            2f32.powi(-7),
            "rmsnorm silu gate",
        );
    }

    #[test]
    fn zero_centered_norm_stage_matches_pinned_norm_fixture() {
        let input = bf16(&[0.2, -0.7, 1.1, 0.3, -0.4, 0.6, 0.1, -0.9]);
        let weight = bf16(&[-0.25, 0.5, 0.0, 0.125]);
        let device = CpuDevice::new();
        let output = zero_centered_norm_stage(&device, &input, 2, 4, &weight, 1e-6).unwrap();
        let actual: Vec<f32> = output.iter().map(|value| value.to_f32()).collect();
        assert_close(
            &actual,
            fixtures::NORM_ZERO_OUT,
            2f32.powi(-7),
            "zero centered norm",
        );
    }

    #[test]
    fn rope_stage_matches_pinned_partial_rope_fixtures() {
        let device = CpuDevice::new();
        let input = bf16(&[
            0.2, -0.7, 1.1, 0.3, -0.4, 0.6, 0.1, -0.9, 0.5, 0.2, -0.3, 0.8, 0.1, -0.6, 0.7, -0.2,
            -0.1, 0.4, 0.6, -0.8, 0.3, 0.2, -0.5, 0.9, 0.7, -0.2, 0.1, 0.5, -0.8, 0.4, 0.3, -0.6,
        ]);
        let output = rope_stage(&device, &input, 2, 2, 8, &[3, 7], 10_000.0, 0.5).unwrap();
        let actual: Vec<f32> = output.iter().map(|value| value.to_f32()).collect();
        assert_close(&actual, fixtures::ROPE_OUTPUT, 2f32.powi(-7), "rope 50%");

        // Production configuration: head_dim 256, partial 0.25, theta 1e7.
        let big: Vec<Bf16> = (0..3 * 256)
            .map(|index| Bf16::from_f32((((index % 23) as f32) - 11.0) / 16.0))
            .collect();
        let output = rope_stage(&device, &big, 3, 1, 256, &[0, 1, 2], 10_000_000.0, 0.25).unwrap();
        let actual: Vec<f32> = output.iter().map(|value| value.to_f32()).collect();
        assert_close(
            &actual,
            fixtures::ROPE_BIG_OUTPUT,
            2f32.powi(-7),
            "rope production",
        );
    }

    // -- GDN state discipline -------------------------------------------------

    #[test]
    fn preview_forward_leaves_gdn_states_bitwise_identical() {
        let (_root, mut backend) = fixture_backend("gdn-preview-readonly", 16);
        let conv_before = backend.gdn_conv_history(0).to_vec();
        let ssm_before = backend.gdn_ssm_state(0).to_vec();
        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let valid = [true, true, true];
        let _ = backend.forward_gdn(0, &input, &valid, true).unwrap();
        assert_eq!(backend.gdn_conv_history(0), conv_before.as_slice());
        assert_eq!(backend.gdn_ssm_state(0), ssm_before.as_slice());
    }

    #[test]
    fn causal_forward_advances_gdn_states_with_exact_chunking() {
        let config = tiny_clover_config();
        let root_single = temp_dir("gdn-chunk-single");
        SynthCloverCheckpoint::new(&config).write(&root_single);
        let mut single = open_backend(&config, &root_single, 16);
        let root_chunked = temp_dir("gdn-chunk-split");
        SynthCloverCheckpoint::new(&config).write(&root_chunked);
        let mut chunked = open_backend(&config, &root_chunked, 16);

        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let valid = [true, true, true];
        single.forward_gdn(0, &input, &valid, false).unwrap();

        let first = CloverActivation::new(1, 4, vec![Bf16::from_f32(0.25); 4]).unwrap();
        let rest = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        chunked.forward_gdn(0, &first, &[true], false).unwrap();
        chunked.forward_gdn(0, &rest, &[true, true], false).unwrap();

        assert_eq!(single.gdn_conv_history(0), chunked.gdn_conv_history(0));
        assert_eq!(single.gdn_ssm_state(0), chunked.gdn_ssm_state(0));
        assert!(single.gdn_ssm_state(0).iter().any(|value| *value != 0.0));
    }

    #[test]
    fn causal_error_after_recurrent_stage_leaves_no_state_mutation() {
        let config = tiny_clover_config();
        let root = temp_dir("gdn-mid-failure");
        let mut checkpoint = SynthCloverCheckpoint::new(&config);
        // Force out_proj (read after the recurrent stage) to sit at the shard
        // tail so truncation fails only there.
        checkpoint.move_last("model.layers.0.linear_attn.out_proj.weight");
        checkpoint.write(&root);
        let mut backend = open_backend(&config, &root, 16);

        let conv_before = backend.gdn_conv_history(0).to_vec();
        let ssm_before = backend.gdn_ssm_state(0).to_vec();
        let shard = root.join(SYNTH_MODEL_SHARD);
        let bytes = fs::read(&shard).unwrap();
        // Keep header plus a prefix of the payload: the tail tensor read fails.
        fs::write(&shard, &bytes[..bytes.len() - 16]).unwrap();

        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let valid = [true, true, true];
        assert!(backend.forward_gdn(0, &input, &valid, false).is_err());
        assert_eq!(backend.gdn_conv_history(0), conv_before.as_slice());
        assert_eq!(backend.gdn_ssm_state(0), ssm_before.as_slice());
    }

    #[test]
    fn two_backends_isolate_gdn_state() {
        let (_root_a, mut backend_a) = fixture_backend("gdn-isolation-a", 16);
        let (_root_b, backend_b) = fixture_backend("gdn-isolation-b", 16);
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.5); 8]).unwrap();
        backend_a
            .forward_gdn(0, &input, &[true, true], false)
            .unwrap();
        assert!(backend_a.gdn_ssm_state(0).iter().any(|value| *value != 0.0));
        assert!(backend_b.gdn_ssm_state(0).iter().all(|value| *value == 0.0));
    }

    #[test]
    fn invalid_tokens_do_not_advance_gdn_state() {
        let config = tiny_clover_config();
        let root_all = temp_dir("gdn-invalid-all");
        SynthCloverCheckpoint::new(&config).write(&root_all);
        let mut all = open_backend(&config, &root_all, 16);
        let root_invalid = temp_dir("gdn-invalid-mixed");
        SynthCloverCheckpoint::new(&config).write(&root_invalid);
        let mut mixed = open_backend(&config, &root_invalid, 16);

        // Rows are identical constant vectors, so the mixed forward's valid
        // rows carry the same values as the all-valid two-token forward.
        let input_two = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        let all_out = all
            .forward_gdn(0, &input_two, &[true, true], false)
            .unwrap();
        assert_eq!(all_out.tokens(), 2);
        let input_three = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let mixed_out = mixed
            .forward_gdn(0, &input_three, &[true, false, true], false)
            .unwrap();
        assert_eq!(all.gdn_conv_history(0), mixed.gdn_conv_history(0));
        assert_eq!(all.gdn_ssm_state(0), mixed.gdn_ssm_state(0));
        // The invalid middle row must be zero after out_proj (all-zero gate).
        let hidden = 4usize;
        let invalid_row = &mixed_out.values()[hidden..2 * hidden];
        assert!(invalid_row
            .iter()
            .all(|value| value.to_f32() == 0.0 || value.to_f32() == -0.0));
    }

    #[test]
    fn gdn_forward_matches_independent_primitive_chain() {
        let (_root, mut backend) = fixture_backend("gdn-chain-parity", 16);
        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let valid = [true, true, true];

        // Independent chain over the same checkpoint tensors, run while the
        // backend state is still pristine.
        let store = CloverCpuWeightStore::new(&backend.inventory);
        let gdn_weights = store.gdn_weights(0, backend.gdn_geometry()).unwrap();
        let x = input.values().to_vec();
        let raw = store.linear(&gdn_weights.in_proj_qkv, &x, 3).unwrap();
        let raw_f32: Vec<f32> = raw.iter().map(|value| value.to_f32()).collect();
        let (mixed, _) = gdn_causal_conv_stage(
            &raw_f32,
            3,
            backend.gdn_channels,
            backend.gdn_conv_history(0),
            &gdn_weights.conv1d,
            backend.kernel_width,
            &valid,
        );
        let (q, k, v) = split_qkv(
            &mixed,
            3,
            backend.gdn_channels,
            backend.gdn_key_width,
            backend.gdn_value_width,
        );
        let a = store
            .linear(&gdn_weights.in_proj_a, &x, 3)
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let b = store
            .linear(&gdn_weights.in_proj_b, &x, 3)
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let (decay, beta) = gdn_gates_stage(
            &a,
            &b,
            &gdn_weights.a_log,
            &gdn_weights.dt_bias,
            3,
            backend.gdn_dims.value_heads,
        );
        let mut ssm = vec![0.0f32; backend.gdn_ssm_state(0).len()];
        let core = recurrent_commit(
            backend.gdn_dims,
            &q,
            &k,
            &v,
            &decay,
            &beta,
            &valid,
            GdnOutputDType::BF16,
            &mut ssm,
        )
        .unwrap();
        let core_bf16: Vec<Bf16> = core.iter().map(|value| Bf16::from_f32(*value)).collect();
        let z = store.linear(&gdn_weights.in_proj_z, &x, 3).unwrap();
        let normed = gdn_norm_gate_stage(
            &backend.device,
            &core_bf16,
            &gdn_weights.norm,
            &z,
            backend.eps,
        )
        .unwrap();
        let expected = store.linear(&gdn_weights.out_proj, &normed, 3).unwrap();

        let output = backend.forward_gdn(0, &input, &valid, false).unwrap();
        assert_eq!(output.values(), expected.as_slice());
    }

    // -- Full-attention -------------------------------------------------------

    #[test]
    fn attention_stage_matches_pinned_upstream_attention_fixture() {
        // Regenerate the M11.2 attention fixture inputs from the pinned
        // generator formulas and drive the backend attention stage end to end.
        let q_proj_out: Vec<Bf16> = {
            let q =
                |t: usize, h: usize, d: usize| ((((t * 64 + h * 4 + d) % 17) as f32) - 8.0) / 20.0;
            let gate =
                |t: usize, h: usize, d: usize| ((((t * 64 + h * 4 + d) % 11) as f32) - 5.0) / 8.0;
            let mut values = Vec::with_capacity(2 * 16 * 8);
            for t in 0..2 {
                for h in 0..16 {
                    for d in 0..4 {
                        values.push(Bf16::from_f32(q(t, h, d)));
                    }
                    for d in 0..4 {
                        values.push(Bf16::from_f32(gate(t, h, d)));
                    }
                }
            }
            values
        };
        let k_proj_out: Vec<Bf16> = (0..3 * 2 * 4)
            .map(|index| Bf16::from_f32((((index % 13) as f32) - 6.0) / 15.0))
            .collect();
        let v_proj_out: Vec<Bf16> = (0..3 * 2 * 4)
            .map(|index| Bf16::from_f32((((index % 19) as f32) - 9.0) / 20.0))
            .collect();
        let q_norm = bf16(&[0.0, 0.1, -0.1, 0.2]);
        let k_norm = bf16(&[0.1, -0.1, 0.2, 0.0]);
        let device = CpuDevice::new();

        // The fixture caches key/value row 0: normalized at zero-centered
        // norm and rotated at position 0 through the same pinned stages.
        let cached_k =
            zero_centered_norm_stage(&device, &k_proj_out[..2 * 4], 2, 4, &k_norm, 1e-6).unwrap();
        let cached_k = rope_stage(&device, &cached_k, 1, 2, 4, &[0], 10_000.0, 0.5).unwrap();
        let cached_k: Vec<f32> = cached_k.iter().map(|value| value.to_f32()).collect();
        let cached_v: Vec<f32> = v_proj_out[..2 * 4]
            .iter()
            .map(|value| value.to_f32())
            .collect();

        let gated = attention_stage(
            &device,
            AttentionDType::Bf16,
            &q_proj_out,
            &k_proj_out[2 * 4..],
            &v_proj_out[2 * 4..],
            &cached_k,
            &cached_v,
            &[true],
            &[true, false],
            &[1, 2],
            true,
            16,
            2,
            4,
            &q_norm,
            &k_norm,
            1e-6,
            10_000.0,
            0.5,
        )
        .unwrap()
        .gated;
        assert_close(
            &gated.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
            fixtures::ATTN_GATED_OUTPUT,
            2f32.powi(-7),
            "attention gated output",
        );
    }

    #[test]
    fn attention_respects_prefix_holes_and_invalid_queries() {
        // MASK_COMMIT_ALLOW fixture wiring: prefix [t,f,t], block [t,t,f,t],
        // commit mode. An invalid query row is all zero and a hole in the
        // prefix or an invalid block key never contributes: perturbing the
        // hidden value at blocked keys must not move valid outputs.
        let build = |k0: f32| -> Vec<Bf16> {
            let device = CpuDevice::new();
            let q_proj_out: Vec<Bf16> = (0..4 * 16)
                .map(|index| Bf16::from_f32((((index % 7) as f32) - 3.0) / 8.0))
                .collect();
            let k_new: Vec<Bf16> = (0..4 * 8)
                .map(|index| Bf16::from_f32((((index % 5) as f32) - 2.0) / 8.0))
                .collect();
            let v_new: Vec<Bf16> = (0..4 * 8)
                .map(|index| Bf16::from_f32((((index % 3) as f32) - 1.0) / 8.0))
                .collect();
            let cached_k: Vec<f32> = (0..3 * 8)
                .map(|index| {
                    if index == 0 {
                        k0
                    } else {
                        (((index % 11) as f32) - 5.0) / 8.0
                    }
                })
                .collect();
            let cached_v: Vec<f32> = (0..3 * 8)
                .map(|index| (((index % 9) as f32) - 4.0) / 8.0)
                .collect();
            attention_stage(
                &device,
                AttentionDType::F32,
                &q_proj_out,
                &k_new,
                &v_new,
                &cached_k,
                &cached_v,
                &[true, false, true],
                &[true, true, false, true],
                &[3, 4, 5, 6],
                false,
                1,
                1,
                8,
                &bf16(&[0.0; 8]),
                &bf16(&[0.0; 8]),
                1e-6,
                10_000.0,
                0.5,
            )
            .unwrap()
            .gated
        };
        let baseline = build(-5.0);
        let perturbed = build(7.0);
        // The two runs differ only at cached key row 0 (a visible prefix key),
        // so valid query rows must move while the invalid row stays zero.
        assert_eq!(baseline.len(), 4 * 8);
        let invalid_row = &baseline[2 * 8..3 * 8];
        assert!(invalid_row
            .iter()
            .all(|value| value.to_f32() == 0.0 || value.to_f32() == -0.0));
        // Perturbing key row 0 (a visible prefix key) must change valid rows.
        assert_ne!(
            baseline[0..8]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            perturbed[0..8]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn preview_forward_does_not_mutate_kv_cache() {
        let (_root, mut backend) = fixture_backend("attn-preview-readonly", 16);
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        let valid = [true, true];
        backend
            .forward_full_attention(1, &input, &valid, &[], &[0, 1], true)
            .unwrap();
        assert_eq!(backend.cached_kv_tokens(1), 0);
        assert!(backend.kv_cache(1).0.is_empty());
        assert!(backend.kv_cache(1).1.is_empty());
    }

    #[test]
    fn causal_forward_appends_exact_keys_and_values() {
        let (_root, mut backend) = fixture_backend("attn-commit-append", 16);
        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let valid = [true, true, true];
        backend
            .forward_full_attention(1, &input, &valid, &[], &[0, 1, 2], false)
            .unwrap();
        let appended = backend.cached_kv_tokens(1);
        assert_eq!(appended, 3);

        // A second causal step sees exactly the committed keys in its cache.
        let (keys_before, values_before) = backend.kv_cache(1);
        let keys_before = keys_before.to_vec();
        let values_before = values_before.to_vec();
        let input2 = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.125); 8]).unwrap();
        backend
            .forward_full_attention(
                1,
                &input2,
                &[true, true],
                &[true, true, true],
                &[3, 4],
                false,
            )
            .unwrap();
        let (keys, values) = backend.kv_cache(1);
        assert_eq!(&keys[..keys_before.len()], keys_before.as_slice());
        assert_eq!(&values[..values_before.len()], values_before.as_slice());
        assert_eq!(backend.cached_kv_tokens(1), 5);
    }

    #[test]
    fn padding_positions_keep_cache_alignment() {
        let (_root, mut backend) = fixture_backend("attn-padding", 16);
        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        backend
            .forward_full_attention(1, &input, &[true, false, true], &[], &[0, 0, 1], false)
            .unwrap();
        assert_eq!(backend.cached_kv_tokens(1), 3);
    }

    #[test]
    fn attention_rollback_on_io_failure_keeps_cache() {
        let config = tiny_clover_config();
        let root = temp_dir("attn-io-failure");
        let mut checkpoint = SynthCloverCheckpoint::new(&config);
        checkpoint.move_last("model.layers.1.self_attn.o_proj.weight");
        checkpoint.write(&root);
        let mut backend = open_backend(&config, &root, 16);
        let shard = root.join(SYNTH_MODEL_SHARD);
        let bytes = fs::read(&shard).unwrap();
        fs::write(&shard, &bytes[..bytes.len() - 16]).unwrap();
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        assert!(backend
            .forward_full_attention(1, &input, &[true, true], &[], &[0, 1], false)
            .is_err());
        assert_eq!(backend.cached_kv_tokens(1), 0);
    }

    #[test]
    fn two_backends_isolate_kv_state() {
        let (_root_a, mut backend_a) = fixture_backend("attn-isolation-a", 16);
        let (_root_b, backend_b) = fixture_backend("attn-isolation-b", 16);
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        backend_a
            .forward_full_attention(1, &input, &[true, true], &[], &[0, 1], false)
            .unwrap();
        assert_eq!(backend_a.cached_kv_tokens(1), 2);
        assert_eq!(backend_b.cached_kv_tokens(1), 0);
    }

    // -- MoE within the graph --------------------------------------------------

    #[test]
    fn router_logits_match_reference_projection() {
        use crate::moe::project_router_logits_cpu;
        let config = tiny_clover_config();
        let (_root, mut backend) = fixture_backend("moe-router", 16);
        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let logits = backend.router_logits(0, &input).unwrap();
        let store = CloverCpuWeightStore::new(&backend.inventory);
        let weight = store
            .read_bf16(
                &router_tensor(0),
                0,
                config.num_experts * config.hidden_size,
            )
            .unwrap();
        let expected = project_router_logits_cpu(
            input.values(),
            3,
            config.hidden_size,
            config.num_experts,
            &weight,
        )
        .unwrap();
        assert_eq!(logits, expected);
    }

    #[test]
    fn zero_weight_routes_skip_expert_loading() {
        let mut config = tiny_clover_config();
        config.moe_block_coreset_top_p = 0.02;
        let root = temp_dir("moe-coreset");
        SynthCloverCheckpoint::new(&config).write(&root);
        let mut backend = open_backend(&config, &root, 1);
        let input = CloverActivation::new(4, 4, vec![Bf16::from_f32(0.25); 16]).unwrap();
        let valid = [true, true, true, true];
        let logits = backend.router_logits(0, &input).unwrap();
        let routing = route(&config, &logits, 4, &valid, true);
        let selected: std::collections::BTreeSet<usize> = routing
            .routes
            .iter()
            .flat_map(|route| {
                route
                    .expert_ids
                    .iter()
                    .zip(&route.weights)
                    .filter(|(_, weight)| **weight > 0.0)
                    .map(|(expert, _)| *expert)
            })
            .collect();
        let masked: Vec<usize> = routing
            .routes
            .iter()
            .flat_map(|route| {
                route
                    .expert_ids
                    .iter()
                    .zip(&route.weights)
                    .filter(|(_, weight)| **weight == 0.0)
                    .map(|(expert, _)| *expert)
            })
            .collect();
        assert!(!masked.is_empty(), "coreset must mask some experts");

        let output = backend
            .forward_moe(0, &input, &routing, &valid, true)
            .unwrap();
        for expert in &masked {
            if !selected.contains(expert) {
                assert!(
                    !backend.expert_cache.is_resident(0, *expert),
                    "zero-weight expert {expert} must not load"
                );
            }
        }
        assert_eq!(output.tokens(), 4);
    }

    #[test]
    fn moe_matches_execute_moe_cpu_reference() {
        use crate::moe::execute_moe_cpu;
        let config = tiny_clover_config();
        let (_root, mut backend) = fixture_backend("moe-parity", 16);
        let input = CloverActivation::new(3, 4, vec![Bf16::from_f32(0.25); 12]).unwrap();
        let valid = [true, true, true];
        let logits = backend.router_logits(0, &input).unwrap();
        let routing = route(&config, &logits, 3, &valid, false);
        let output = backend
            .forward_moe(0, &input, &routing, &valid, false)
            .unwrap();

        let store = CloverCpuWeightStore::new(&backend.inventory);
        let mut experts = BTreeMap::new();
        for expert in routing.unique_experts() {
            if routing.routes.iter().any(|route| {
                route
                    .expert_ids
                    .iter()
                    .zip(&route.weights)
                    .any(|(id, weight)| *id == expert && *weight > 0.0)
            }) {
                experts.insert(
                    expert,
                    store
                        .routed_expert_weights(
                            0,
                            expert,
                            config.num_experts,
                            config.hidden_size,
                            config.moe_intermediate_size,
                        )
                        .unwrap(),
                );
            }
        }
        let shared = store
            .shared_expert_weights(
                0,
                config.hidden_size,
                config.shared_expert_intermediate_size,
            )
            .unwrap();
        let expected = execute_moe_cpu(
            input.values(),
            3,
            config.hidden_size,
            &routing,
            &experts,
            Some(&shared),
        )
        .unwrap();
        assert_eq!(output.values(), expected.as_slice());
    }

    #[test]
    fn missing_expert_weights_surface_as_error() {
        let config = tiny_clover_config();
        let root = temp_dir("moe-missing");
        SynthCloverCheckpoint::new(&config).write(&root);
        let mut backend = open_backend(&config, &root, 16);
        fs::remove_file(root.join(SYNTH_MODEL_SHARD)).unwrap();
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        let valid = [true, true];
        let logits = backend.router_logits(0, &input).unwrap();
        let routing = route(&config, &logits, 2, &valid, false);
        let error = backend
            .forward_moe(0, &input, &routing, &valid, false)
            .unwrap_err();
        assert!(
            error.contains("expert") || error.contains("stream") || error.contains("I/O"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn forward_moe_honours_cancellation_before_transfer() {
        let config = tiny_clover_config();
        let root = temp_dir("moe-cancel");
        SynthCloverCheckpoint::new(&config).write(&root);
        let inventory = CloverCheckpointInventory::inspect(&root).unwrap();
        let token = CancellationToken::new();
        let expert_bytes = ((config.moe_intermediate_size * 2 * config.hidden_size
            + config.hidden_size * config.moe_intermediate_size)
            * 2) as u64;
        let shared_bytes = ((3 * config.shared_expert_intermediate_size * config.hidden_size
            + config.hidden_size)
            * 2) as u64;
        let mut backend = CloverCpuBackend::new(
            &config,
            inventory,
            CloverCpuBackendOptions {
                expert_budgets: budgets(8, expert_bytes, shared_bytes),
                eviction: EvictionPolicy::Lru,
                cancellation: Some(token.clone()),
            },
        )
        .unwrap();
        token.cancel();
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        let valid = [true, true];
        let logits = backend.router_logits(0, &input).unwrap();
        let routing = route(&config, &logits, 2, &valid, false);
        assert!(backend
            .forward_moe(0, &input, &routing, &valid, false)
            .is_err());
        assert!(backend.cache_metrics().cancelled_loads >= 1);
    }

    #[test]
    fn expert_cache_respects_memory_pressure() {
        let config = tiny_clover_config();
        let root = temp_dir("moe-pressure");
        SynthCloverCheckpoint::new(&config).write(&root);
        // Budget: one routed expert plus the shared expert.
        let mut backend = open_backend(&config, &root, 1);
        let input = CloverActivation::new(4, 4, vec![Bf16::from_f32(0.25); 16]).unwrap();
        let valid = [true, true, true, true];
        let logits = backend.router_logits(0, &input).unwrap();
        let routing = route(&config, &logits, 4, &valid, false);
        let result = backend.forward_moe(0, &input, &routing, &valid, false);
        // Either experts evict within budget or admission fails; both are
        // legal, but usage must never exceed the hard budget by more than one
        // in-budget allocation and state must stay consistent.
        assert!(backend.expert_usage_bytes() <= backend.expert_hard_bytes());
        if result.is_err() {
            assert!(backend.cache_metrics().admission_failures >= 1);
        }
        let _ = result;
    }

    // -- Session lifecycle -----------------------------------------------------

    #[test]
    fn reset_session_clears_all_clean_state() {
        let (_root, mut backend) = fixture_backend("reset-session", 16);
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        backend
            .forward_gdn(0, &input, &[true, true], false)
            .unwrap();
        backend
            .forward_full_attention(1, &input, &[true, true], &[], &[0, 1], false)
            .unwrap();
        assert!(backend.cached_kv_tokens(1) > 0);
        assert!(backend.gdn_ssm_state(0).iter().any(|value| *value != 0.0));
        backend.reset_session().unwrap();
        assert_eq!(backend.cached_kv_tokens(1), 0);
        assert!(backend.gdn_ssm_state(0).iter().all(|value| *value == 0.0));
        assert!(backend
            .gdn_conv_history(0)
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(backend.cache_metrics().resident_experts, 0);
    }

    /// P4: when an expert release fails, `reset_session` must report it and
    /// keep the residency identifiers so the host can retry the release. The
    /// expert weights live in host memory; only the release path can fail.
    fn fault_release_backend(
        label: &str,
    ) -> (
        PathBuf,
        CloverCpuBackend<crate::residency::release_fault::FailOnceReleaseDevice>,
    ) {
        use crate::residency::release_fault::FailOnceReleaseDevice;
        let config = tiny_clover_config();
        let root = temp_dir(label);
        SynthCloverCheckpoint::new(&config).write(&root);
        let inventory = CloverCheckpointInventory::inspect(&root).unwrap();
        let expert_bytes = ((config.moe_intermediate_size * 2 * config.hidden_size
            + config.hidden_size * config.moe_intermediate_size)
            * 2) as u64;
        let shared_bytes = ((3 * config.shared_expert_intermediate_size * config.hidden_size
            + config.hidden_size)
            * 2) as u64;
        let backend = CloverCpuBackend::new_with_expert_device(
            &config,
            inventory,
            CloverCpuBackendOptions {
                expert_budgets: budgets(16, expert_bytes, shared_bytes),
                eviction: EvictionPolicy::Lru,
                cancellation: None,
            },
            FailOnceReleaseDevice::new(),
        )
        .unwrap();
        (root, backend)
    }

    fn route_layer0(
        config: &CloverConfig,
        backend: &mut CloverCpuBackend<crate::residency::release_fault::FailOnceReleaseDevice>,
    ) -> crate::CloverRoutingPlan {
        let input = CloverActivation::new(2, 4, vec![Bf16::from_f32(0.25); 8]).unwrap();
        let valid = [true, true];
        let logits = backend.router_logits(0, &input).unwrap();
        let routing = route(config, &logits, 2, &valid, false);
        backend
            .forward_moe(0, &input, &routing, &valid, false)
            .unwrap();
        routing
    }

    #[test]
    fn reset_session_reports_failed_releases_and_keeps_residency_retryable() {
        let config = tiny_clover_config();
        let (_root, mut backend) = fault_release_backend("reset-release-fault");
        route_layer0(&config, &mut backend);
        assert!(
            !backend.resident.is_empty(),
            "routed experts must be resident"
        );
        assert!(!backend.resident_shared.is_empty());
        assert!(backend.expert_usage_bytes() > 0);

        let error = backend
            .reset_session()
            .expect_err("a failing release must be reported, not swallowed");
        assert!(
            error.contains("expert release layer=0 expert="),
            "routed release failure missing: {error}"
        );
        assert!(
            error.contains("shared expert release layer=0:"),
            "shared release failure missing: {error}"
        );
        // The failed releases keep their residency identifiers: dropping them
        // would strand the allocation with no way to free it.
        assert!(
            !backend.resident.is_empty(),
            "failed routed releases must stay retryable"
        );
        assert!(
            !backend.resident_shared.is_empty(),
            "failed shared release must stay retryable"
        );
        assert!(backend.expert_usage_bytes() > 0);
    }

    #[test]
    fn retrying_reset_session_releases_every_resident_expert() {
        let config = tiny_clover_config();
        let (_root, mut backend) = fault_release_backend("reset-release-retry");
        route_layer0(&config, &mut backend);
        assert!(backend.reset_session().is_err(), "first release fails");

        backend
            .reset_session()
            .expect("the retry must release everything");
        assert!(backend.resident.is_empty());
        assert!(backend.resident_shared.is_empty());
        assert_eq!(backend.expert_usage_bytes(), 0);
        assert_eq!(backend.cache_metrics().resident_experts, 0);
    }

    #[test]
    fn embed_rejects_out_of_vocabulary_tokens() {
        let (_root, mut backend) = fixture_backend("embed-range", 16);
        assert!(backend.embed_tokens(&[9]).is_err());
        let activation = backend.embed_tokens(&[0, 7]).unwrap();
        assert_eq!(activation.tokens(), 2);
    }

    #[test]
    fn graph_forward_with_backend_produces_finite_logits() {
        let config = tiny_clover_config();
        let root = temp_dir("graph-forward");
        SynthCloverCheckpoint::new(&config).write(&root);
        let mut backend = open_backend(&config, &root, 16);
        let graph = CloverGraph::new(config.clone()).unwrap();
        let output = graph
            .forward_preview(&mut backend, &[3, 4], &[true, true], &[], &[0, 1])
            .unwrap();
        let logits = output.logits.expect("preview logits");
        assert_eq!(logits.len(), 2 * config.vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}
