//! CPU reference routing, block coreset, and MoE dispatch for the pinned Clover release.
//!
//! This module owns no device buffers and no scheduler tasks. It produces
//! immutable routing metadata; weights remain owned by the shared memory layer.

use std::collections::BTreeMap;
use std::ops::Range;

use thiserror::Error;

use crate::CloverConfig;
use claymore_tensor::Bf16;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CloverMoeError {
    #[error("router dimensions are invalid: {0}")]
    InvalidDimensions(&'static str),
    #[error("router logits contain a non-finite value at token {token}, expert {expert}")]
    NonFiniteLogit { token: usize, expert: usize },
    #[error("top-p must be finite and in (0, 1], got {0}")]
    InvalidTopP(f32),
    #[error("token {token} has no finite route after coreset filtering")]
    EmptyTokenRoute { token: usize },
    #[error("missing routed expert weights for expert {0}")]
    MissingExpert(usize),
    #[error("expert {expert} weight shape is invalid: {reason}")]
    InvalidExpertWeights { expert: usize, reason: &'static str },
    #[error("shared expert weight shape is invalid: {0}")]
    InvalidSharedWeights(&'static str),
    #[error("routing plan and input dimensions do not match")]
    RoutingShapeMismatch,
    #[error("numeric shape arithmetic overflow")]
    SizeOverflow,
}

/// Apply the pinned bias-free router projection to BF16 hidden states.
///
/// The linear result is rounded to BF16 exactly once, as in the upstream
/// module, then widened to FP32 for the router softmax and top-k selection.
pub fn project_router_logits_cpu(
    input: &[Bf16],
    token_count: usize,
    hidden_size: usize,
    num_experts: usize,
    router_weight: &[Bf16],
) -> Result<Vec<f32>, CloverMoeError> {
    let input_len = token_count
        .checked_mul(hidden_size)
        .ok_or(CloverMoeError::SizeOverflow)?;
    let weight_len = num_experts
        .checked_mul(hidden_size)
        .ok_or(CloverMoeError::SizeOverflow)?;
    let output_len = token_count
        .checked_mul(num_experts)
        .ok_or(CloverMoeError::SizeOverflow)?;
    if token_count == 0
        || hidden_size == 0
        || num_experts == 0
        || input.len() != input_len
        || router_weight.len() != weight_len
    {
        return Err(CloverMoeError::InvalidDimensions(
            "router projection expects input [tokens, hidden] and weight [experts, hidden]",
        ));
    }
    let mut logits = Vec::with_capacity(output_len);
    for token in 0..token_count {
        let row = &input[token * hidden_size..(token + 1) * hidden_size];
        for expert in 0..num_experts {
            let weights = &router_weight[expert * hidden_size..(expert + 1) * hidden_size];
            logits.push(dot_bf16(row, weights).to_f32());
        }
    }
    Ok(logits)
}

/// Immutable Clover MoE routing configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloverMoeRouter {
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub coreset_enabled: bool,
    pub coreset_top_p: f32,
}

impl CloverMoeRouter {
    pub fn from_config(config: &CloverConfig) -> Self {
        Self {
            num_experts: config.num_experts,
            experts_per_token: config.num_experts_per_tok,
            coreset_enabled: config.moe_block_coreset_enabled,
            coreset_top_p: config.moe_block_coreset_top_p as f32,
        }
    }

    pub fn validate(&self) -> Result<(), CloverMoeError> {
        if self.num_experts == 0
            || self.experts_per_token == 0
            || self.experts_per_token > self.num_experts
        {
            return Err(CloverMoeError::InvalidDimensions(
                "expert count and top-k must be non-zero, with top-k <= experts",
            ));
        }
        if !self.coreset_top_p.is_finite() || self.coreset_top_p <= 0.0 || self.coreset_top_p > 1.0
        {
            return Err(CloverMoeError::InvalidTopP(self.coreset_top_p));
        }
        Ok(())
    }

    /// Route FP32 logits shaped [tokens, experts]. Preview coreset is applied
    /// only when both preview and the pinned config enable it.
    ///
    /// Ties use stable descending order (lower expert index first), matching
    /// the pinned torch.argsort(descending=True, stable=True) implementation.
    /// Softmax, coreset accumulation and route-weight normalization are FP32.
    pub fn route(
        &self,
        logits: &[f32],
        token_count: usize,
        valid: &[bool],
        preview: bool,
    ) -> Result<CloverRoutingPlan, CloverMoeError> {
        self.route_batched(logits, 1, token_count, valid, preview)
    }

    /// Batched variant preserving Clover's per-sequence coreset boundary.
    /// Input tensors are flattened batch-major as [batch * tokens, experts].
    pub fn route_batched(
        &self,
        logits: &[f32],
        batch_size: usize,
        tokens_per_batch: usize,
        valid: &[bool],
        preview: bool,
    ) -> Result<CloverRoutingPlan, CloverMoeError> {
        self.validate()?;
        let token_count = batch_size
            .checked_mul(tokens_per_batch)
            .ok_or(CloverMoeError::SizeOverflow)?;
        if batch_size == 0
            || tokens_per_batch == 0
            || valid.len() != token_count
            || logits.len()
                != token_count
                    .checked_mul(self.num_experts)
                    .ok_or(CloverMoeError::SizeOverflow)?
        {
            return Err(CloverMoeError::InvalidDimensions(
                "logits must be [batch * tokens, num_experts] and valid must match",
            ));
        }
        for token in 0..token_count {
            for expert in 0..self.num_experts {
                if !logits[token * self.num_experts + expert].is_finite() {
                    return Err(CloverMoeError::NonFiniteLogit { token, expert });
                }
            }
        }

        let probabilities = (0..token_count)
            .map(|token| softmax(&logits[token * self.num_experts..(token + 1) * self.num_experts]))
            .collect::<Vec<_>>();

        let coreset = if preview && self.coreset_enabled {
            let mut batch_coresets = Vec::with_capacity(batch_size);
            for batch in 0..batch_size {
                batch_coresets.push(select_block_coreset(
                    &probabilities,
                    logits,
                    valid,
                    batch * tokens_per_batch..(batch + 1) * tokens_per_batch,
                    self.num_experts,
                    self.experts_per_token,
                    self.coreset_top_p,
                )?);
            }
            Some(batch_coresets)
        } else {
            None
        };

        let allowed = coreset.as_ref().map(|sets| {
            sets.iter()
                .map(|ids| {
                    let mut mask = vec![false; self.num_experts];
                    for &id in ids {
                        mask[id] = true;
                    }
                    mask
                })
                .collect::<Vec<_>>()
        });

        let mut routes = Vec::with_capacity(token_count);
        let mut dispatch: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for token in 0..token_count {
            let row = &logits[token * self.num_experts..(token + 1) * self.num_experts];
            let batch_mask = allowed
                .as_ref()
                .map(|masks| &masks[token / tokens_per_batch]);
            // Keep exactly top-k entries even when the coreset has fewer than
            // k experts: masked experts remain -inf with zero route weight,
            // matching torch.topk over the masked full expert dimension.
            let mut candidates: Vec<usize> = (0..self.num_experts).collect();
            candidates.sort_by(|&left, &right| {
                let left_score = if valid[token] && batch_mask.is_some_and(|mask| !mask[left]) {
                    f32::NEG_INFINITY
                } else {
                    row[left]
                };
                let right_score = if valid[token] && batch_mask.is_some_and(|mask| !mask[right]) {
                    f32::NEG_INFINITY
                } else {
                    row[right]
                };
                right_score
                    .total_cmp(&left_score)
                    .then_with(|| left.cmp(&right))
            });
            candidates.truncate(self.experts_per_token);

            let maximum = candidates
                .iter()
                .map(|&expert| {
                    if valid[token] && batch_mask.is_some_and(|mask| !mask[expert]) {
                        f32::NEG_INFINITY
                    } else {
                        row[expert]
                    }
                })
                .fold(f32::NEG_INFINITY, f32::max);
            if !maximum.is_finite() {
                return Err(CloverMoeError::EmptyTokenRoute { token });
            }
            let mut selected = Vec::with_capacity(candidates.len());
            let mut denominator = 0.0f32;
            for &expert in &candidates {
                let score = if valid[token] && batch_mask.is_some_and(|mask| !mask[expert]) {
                    0.0
                } else {
                    (row[expert] - maximum).exp()
                };
                selected.push(score);
                denominator += score;
            }
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(CloverMoeError::EmptyTokenRoute { token });
            }
            let weights: Vec<f32> = selected
                .into_iter()
                .map(|value| value / denominator)
                .collect();
            for (slot, &expert) in candidates.iter().enumerate() {
                dispatch.entry(expert).or_default().push((token, slot));
            }
            routes.push(TokenRoute {
                expert_ids: candidates,
                weights,
            });
        }

        Ok(CloverRoutingPlan {
            routes,
            coreset,
            dispatch,
        })
    }
}

/// Per-token top-k expert ids and normalized FP32 weights.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenRoute {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Dispatch is grouped by ascending expert id; token and slot order within
/// each expert follows the original flattened token order.
#[derive(Debug, Clone, PartialEq)]
pub struct CloverRoutingPlan {
    pub routes: Vec<TokenRoute>,
    pub coreset: Option<Vec<Vec<usize>>>,
    pub dispatch: BTreeMap<usize, Vec<(usize, usize)>>,
}

impl CloverRoutingPlan {
    pub fn unique_experts(&self) -> Vec<usize> {
        self.dispatch.keys().copied().collect()
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect();
    let sum: f32 = values.iter().sum();
    for value in &mut values {
        *value /= sum;
    }
    values
}

fn select_block_coreset(
    probabilities: &[Vec<f32>],
    logits: &[f32],
    valid: &[bool],
    token_range: Range<usize>,
    expert_count: usize,
    top_k: usize,
    top_p: f32,
) -> Result<Vec<usize>, CloverMoeError> {
    let mut totals = vec![0.0f32; expert_count];
    for token in token_range {
        if !valid[token] {
            continue;
        }
        let row = &logits[token * expert_count..(token + 1) * expert_count];
        let mut ids: Vec<usize> = (0..expert_count).collect();
        ids.sort_by(|&left, &right| {
            row[right]
                .total_cmp(&row[left])
                .then_with(|| left.cmp(&right))
        });
        ids.truncate(top_k);
        for expert in ids {
            totals[expert] += probabilities[token][expert];
        }
    }

    let total_mass: f32 = totals.iter().sum();
    if !total_mass.is_finite() {
        return Err(CloverMoeError::InvalidDimensions(
            "coreset mass is not finite",
        ));
    }
    // With no valid tokens the source's clamped zero-mass coreset is
    // immaterial: invalid tokens are deliberately not masked.
    if total_mass <= 0.0 {
        return Ok(Vec::new());
    }
    for value in &mut totals {
        *value /= total_mass;
    }

    let mut order: Vec<usize> = (0..expert_count).collect();
    order.sort_by(|&left, &right| {
        totals[right]
            .total_cmp(&totals[left])
            .then_with(|| left.cmp(&right))
    });
    let mut selected = Vec::new();
    let mut cumulative = 0.0f32;
    for expert in order {
        if cumulative < top_p || selected.is_empty() {
            selected.push(expert);
            cumulative += totals[expert];
        } else {
            break;
        }
    }
    Ok(selected)
}

/// BF16 weights for one routed expert. Matrices use row-major
/// [out_features, in_features] checkpoint layout.
#[derive(Debug, Clone)]
pub struct CloverExpertWeights {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_up_proj: Vec<Bf16>,
    pub down_proj: Vec<Bf16>,
}

impl CloverExpertWeights {
    pub fn validate(&self, expert: usize) -> Result<(), CloverMoeError> {
        let hidden = self.hidden_size;
        let intermediate = self.intermediate_size;
        let gate_up = intermediate
            .checked_mul(2)
            .and_then(|n| n.checked_mul(hidden))
            .ok_or(CloverMoeError::SizeOverflow)?;
        let down = hidden
            .checked_mul(intermediate)
            .ok_or(CloverMoeError::SizeOverflow)?;
        if hidden == 0
            || intermediate == 0
            || self.gate_up_proj.len() != gate_up
            || self.down_proj.len() != down
        {
            return Err(CloverMoeError::InvalidExpertWeights {
                expert,
                reason:
                    "expected gate_up [2 * intermediate, hidden] and down [hidden, intermediate]",
            });
        }
        Ok(())
    }
}

/// BF16 parameters of the single shared SwiGLU expert and its scalar gate.
#[derive(Debug, Clone)]
pub struct CloverSharedExpertWeights {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_proj: Vec<Bf16>,
    pub up_proj: Vec<Bf16>,
    pub down_proj: Vec<Bf16>,
    pub shared_gate: Vec<Bf16>,
}

impl CloverSharedExpertWeights {
    pub fn validate(&self) -> Result<(), CloverMoeError> {
        let hidden = self.hidden_size;
        let intermediate = self.intermediate_size;
        let projection_len = intermediate
            .checked_mul(hidden)
            .ok_or(CloverMoeError::SizeOverflow)?;
        let down_len = hidden
            .checked_mul(intermediate)
            .ok_or(CloverMoeError::SizeOverflow)?;
        if hidden == 0
            || intermediate == 0
            || self.gate_proj.len() != projection_len
            || self.up_proj.len() != projection_len
            || self.down_proj.len() != down_len
            || self.shared_gate.len() != hidden
        {
            return Err(CloverMoeError::InvalidSharedWeights(
                "gate/up [intermediate, hidden], down [hidden, intermediate], gate [hidden]",
            ));
        }
        Ok(())
    }
}

/// CPU BF16 reference for Clover's MoE accumulation contract.
///
/// Routed contributions are processed in ascending unique expert order. Each
/// down projection is accumulated in FP32, route-weighted and rounded once to
/// BF16, then added to the FP32 output accumulator. The accumulator is rounded
/// to BF16 before the shared expert is added.
pub fn execute_moe_cpu(
    input: &[Bf16],
    token_count: usize,
    hidden_size: usize,
    routing: &CloverRoutingPlan,
    experts: &BTreeMap<usize, CloverExpertWeights>,
    shared: Option<&CloverSharedExpertWeights>,
) -> Result<Vec<Bf16>, CloverMoeError> {
    if token_count == 0
        || hidden_size == 0
        || input.len()
            != token_count
                .checked_mul(hidden_size)
                .ok_or(CloverMoeError::SizeOverflow)?
        || routing.routes.len() != token_count
    {
        return Err(CloverMoeError::RoutingShapeMismatch);
    }
    let mut output = vec![0.0f32; input.len()];

    for (&expert_id, assignments) in &routing.dispatch {
        let mut active_assignments = Vec::new();
        for &(token, slot) in assignments {
            let route = routing
                .routes
                .get(token)
                .ok_or(CloverMoeError::RoutingShapeMismatch)?;
            let route_weight = *route
                .weights
                .get(slot)
                .ok_or(CloverMoeError::RoutingShapeMismatch)?;
            if route.expert_ids.get(slot) != Some(&expert_id)
                || !route_weight.is_finite()
                || route_weight < 0.0
            {
                return Err(CloverMoeError::RoutingShapeMismatch);
            }
            if route_weight > 0.0 {
                active_assignments.push((token, route_weight));
            }
        }
        // Coreset-masked experts remain in top-k metadata with exactly zero
        // weight. They must not be loaded or executed by the expert cache.
        if active_assignments.is_empty() {
            continue;
        }
        let weights = experts
            .get(&expert_id)
            .ok_or(CloverMoeError::MissingExpert(expert_id))?;
        weights.validate(expert_id)?;
        if weights.hidden_size != hidden_size {
            return Err(CloverMoeError::InvalidExpertWeights {
                expert: expert_id,
                reason: "hidden size does not match input",
            });
        }
        for (token, route_weight) in active_assignments {
            let x = &input[token * hidden_size..(token + 1) * hidden_size];
            let intermediate = weights.intermediate_size;
            let mut hidden = vec![Bf16::from_bits(0); intermediate];
            for (row, hidden_value) in hidden.iter_mut().enumerate() {
                let gate_start = row * hidden_size;
                let up_start = (intermediate + row) * hidden_size;
                let gate = dot_bf16(
                    x,
                    &weights.gate_up_proj[gate_start..gate_start + hidden_size],
                );
                let up = dot_bf16(x, &weights.gate_up_proj[up_start..up_start + hidden_size]);
                *hidden_value = Bf16::from_f32(silu(gate.to_f32()) * up.to_f32());
            }
            for output_row in 0..hidden_size {
                let start = output_row * intermediate;
                let down = dot_bf16_f32(&hidden, &weights.down_proj[start..start + intermediate]);
                let routed = Bf16::from_f32(down * route_weight);
                output[token * hidden_size + output_row] += routed.to_f32();
            }
        }
    }

    let mut result: Vec<Bf16> = output.into_iter().map(Bf16::from_f32).collect();
    if let Some(shared) = shared {
        shared.validate()?;
        if shared.hidden_size != hidden_size {
            return Err(CloverMoeError::InvalidSharedWeights(
                "shared hidden size does not match input",
            ));
        }
        for token in 0..token_count {
            let x = &input[token * hidden_size..(token + 1) * hidden_size];
            let mut hidden = vec![Bf16::from_bits(0); shared.intermediate_size];
            for (row, hidden_value) in hidden.iter_mut().enumerate() {
                let start = row * hidden_size;
                let gate = dot_bf16(x, &shared.gate_proj[start..start + hidden_size]);
                let up = dot_bf16(x, &shared.up_proj[start..start + hidden_size]);
                *hidden_value = Bf16::from_f32(silu(gate.to_f32()) * up.to_f32());
            }
            let mut shared_output = vec![Bf16::from_bits(0); hidden_size];
            for (row, output_value) in shared_output.iter_mut().enumerate() {
                let start = row * shared.intermediate_size;
                *output_value = dot_bf16(
                    &hidden,
                    &shared.down_proj[start..start + shared.intermediate_size],
                );
            }
            let gate = Bf16::from_f32(sigmoid(dot_bf16(x, &shared.shared_gate).to_f32()));
            for column in 0..hidden_size {
                let gated = Bf16::from_f32(shared_output[column].to_f32() * gate.to_f32());
                result[token * hidden_size + column] =
                    Bf16::from_f32(result[token * hidden_size + column].to_f32() + gated.to_f32());
            }
        }
    }
    Ok(result)
}

fn dot_bf16(input: &[Bf16], weights: &[Bf16]) -> Bf16 {
    Bf16::from_f32(dot_bf16_f32(input, weights))
}

fn dot_bf16_f32(input: &[Bf16], weights: &[Bf16]) -> f32 {
    input
        .iter()
        .zip(weights)
        .fold(0.0f32, |sum, (left, right)| {
            sum + left.to_f32() * right.to_f32()
        })
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(coreset: bool) -> CloverMoeRouter {
        CloverMoeRouter {
            num_experts: 4,
            experts_per_token: 2,
            coreset_enabled: coreset,
            coreset_top_p: 0.9,
        }
    }

    #[test]
    fn stable_top_k_breaks_equal_logits_by_expert_index() {
        let plan = router(false)
            .route(&[1.0, 4.0, 4.0, 0.0], 1, &[true], false)
            .unwrap();
        assert_eq!(plan.routes[0].expert_ids, vec![1, 2]);
        assert!((plan.routes[0].weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn coreset_aggregates_top_k_mass_only_over_valid_tokens() {
        let plan = router(true)
            .route(
                &[
                    8.0, 1.0, 0.0, -1.0, 8.0, 1.0, 0.0, -1.0, -1.0, -2.0, 9.0, 0.0,
                ],
                3,
                &[true, true, false],
                true,
            )
            .unwrap();
        assert!(plan.coreset.as_ref().unwrap()[0].contains(&0));
        assert!(!plan.coreset.as_ref().unwrap()[0].contains(&2));
        assert_eq!(plan.routes[2].expert_ids, vec![2, 3]);
    }

    #[test]
    fn block_coreset_is_computed_independently_per_batch_item() {
        let plan = router(true)
            .route_batched(
                &[
                    8.0, 1.0, 0.0, -1.0, 8.0, 1.0, 0.0, -1.0, -1.0, 0.0, 8.0, 1.0, -1.0, 0.0, 8.0,
                    1.0,
                ],
                2,
                2,
                &[true, true, true, true],
                true,
            )
            .unwrap();
        assert_eq!(plan.coreset, Some(vec![vec![0], vec![2]]));
        assert_eq!(plan.routes[0].expert_ids, vec![0, 1]);
        assert_eq!(plan.routes[2].expert_ids, vec![2, 0]);
    }

    #[test]
    fn coreset_is_not_applied_outside_preview() {
        let plan = router(true)
            .route(&[8.0, 1.0, 0.0, -1.0], 1, &[true], false)
            .unwrap();
        assert_eq!(plan.coreset, None);
        assert_eq!(plan.routes[0].expert_ids, vec![0, 1]);
    }

    #[test]
    fn non_finite_logits_fail_closed() {
        assert!(matches!(
            router(false).route(&[f32::NAN, 1.0, 0.0, -1.0], 1, &[true], false),
            Err(CloverMoeError::NonFiniteLogit {
                token: 0,
                expert: 0
            })
        ));
    }

    #[test]
    fn all_invalid_block_has_deterministic_noop_coreset() {
        let plan = router(true)
            .route(&[1.0, 2.0, 3.0, 4.0], 1, &[false], true)
            .unwrap();
        assert_eq!(plan.coreset, Some(vec![Vec::new()]));
        assert_eq!(plan.routes[0].expert_ids, vec![3, 2]);
    }

    #[test]
    fn routed_and_shared_expert_outputs_are_combined_in_bf16() {
        let one = Bf16::from_f32(1.0);
        let expert = CloverExpertWeights {
            hidden_size: 2,
            intermediate_size: 1,
            gate_up_proj: vec![one, one, one, one],
            down_proj: vec![one, one],
        };
        let experts = BTreeMap::from([(0, expert)]);
        let shared = CloverSharedExpertWeights {
            hidden_size: 2,
            intermediate_size: 1,
            gate_proj: vec![one, one],
            up_proj: vec![one, one],
            down_proj: vec![one, one],
            shared_gate: vec![one, one],
        };
        let routing = CloverRoutingPlan {
            routes: vec![TokenRoute {
                expert_ids: vec![0],
                weights: vec![1.0],
            }],
            coreset: None,
            dispatch: BTreeMap::from([(0, vec![(0, 0)])]),
        };
        let output = execute_moe_cpu(&[one, one], 1, 2, &routing, &experts, Some(&shared)).unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.to_f32().is_finite()));
        assert!(output.iter().all(|value| value.to_f32() > 0.0));
    }

    #[test]
    fn dispatch_groups_experts_in_deterministic_order() {
        let plan = router(false)
            .route(
                &[4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
                2,
                &[true, true],
                false,
            )
            .unwrap();
        assert_eq!(plan.unique_experts(), vec![0, 1, 2, 3]);
        assert_eq!(plan.dispatch[&0], vec![(0, 0)]);
    }

    #[test]
    fn router_projection_uses_bf16_linear_output() {
        let one = Bf16::from_f32(1.0);
        let logits = project_router_logits_cpu(&[one, one], 1, 2, 1, &[one, one]).unwrap();
        assert_eq!(logits, vec![2.0]);
        assert!(matches!(
            project_router_logits_cpu(&[one], 1, 2, 1, &[one, one]),
            Err(CloverMoeError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn zero_weight_masked_experts_are_not_required_by_cpu_execution() {
        let one = Bf16::from_f32(1.0);
        let expert = CloverExpertWeights {
            hidden_size: 2,
            intermediate_size: 1,
            gate_up_proj: vec![one, one, one, one],
            down_proj: vec![one, one],
        };
        let experts = BTreeMap::from([(0, expert)]);
        let routing = router(true)
            .route(&[8.0, 1.0, 0.0, -1.0], 1, &[true], true)
            .unwrap();
        assert_eq!(routing.routes[0].expert_ids, vec![0, 1]);
        assert_eq!(routing.routes[0].weights[1], 0.0);
        let output = execute_moe_cpu(&[one, one], 1, 2, &routing, &experts, None).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn router_preserves_top_eight_and_normalizes_selected_weights() {
        let router = CloverMoeRouter {
            num_experts: 10,
            experts_per_token: 8,
            coreset_enabled: false,
            coreset_top_p: 0.9,
        };
        let logits: Vec<f32> = (0..10).map(|expert| expert as f32).collect();
        let plan = router.route(&logits, 1, &[true], false).unwrap();
        assert_eq!(plan.routes[0].expert_ids, vec![9, 8, 7, 6, 5, 4, 3, 2]);
        assert_eq!(plan.routes[0].weights.len(), 8);
        assert!((plan.routes[0].weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

}
