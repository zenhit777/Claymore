//! Clover's model-specific forward graph over Claymore's shared tensor/device
//! and diffusion-runtime boundaries.
//!
//! The graph owns layer order and routing semantics only. A backend owns weight
//! materialization, tensor operations, clean GDN/attention states and expert
//! residency. Preview calls must be read-only with respect to clean states;
//! causal calls are the only calls allowed to advance them.

use claymore_tensor::Bf16;
use thiserror::Error;

use crate::{
    CloverConfig, CloverConfigError, CloverMoeError, CloverMoeRouter, CloverRoutingPlan, LayerType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloverNormKind {
    Input,
    PostAttention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CloverActivation {
    tokens: usize,
    hidden_size: usize,
    values: Vec<Bf16>,
}

impl CloverActivation {
    pub fn new(
        tokens: usize,
        hidden_size: usize,
        values: Vec<Bf16>,
    ) -> Result<Self, CloverGraphError> {
        let expected = tokens
            .checked_mul(hidden_size)
            .ok_or(CloverGraphError::SizeOverflow)?;
        if tokens == 0 || hidden_size == 0 || values.len() != expected {
            return Err(CloverGraphError::ActivationShape {
                tokens,
                hidden_size,
                values: values.len(),
            });
        }
        Ok(Self {
            tokens,
            hidden_size,
            values,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn values(&self) -> &[Bf16] {
        &self.values
    }
    pub fn into_values(self) -> Vec<Bf16> {
        self.values
    }

    fn validate_like(
        &self,
        expected_tokens: usize,
        expected_hidden: usize,
        layer: Option<usize>,
        operation: &'static str,
    ) -> Result<(), CloverGraphError> {
        if self.tokens != expected_tokens || self.hidden_size != expected_hidden {
            return Err(CloverGraphError::BackendShape {
                layer,
                operation,
                expected_tokens,
                expected_hidden,
                actual_tokens: self.tokens,
                actual_hidden: self.hidden_size,
            });
        }
        Ok(())
    }
}

/// One operation boundary in the complete Clover decoder layer.
///
/// Implementations must not mutate clean caches for GDN/full-attention preview
/// calls. Causal calls may advance clean recurrent and KV state.
pub trait CloverGraphBackend {
    fn embed_tokens(&mut self, token_ids: &[u32]) -> Result<CloverActivation, String>;

    fn rms_norm(
        &mut self,
        layer: usize,
        kind: CloverNormKind,
        input: &CloverActivation,
    ) -> Result<CloverActivation, String>;

    fn forward_gdn(
        &mut self,
        layer: usize,
        input: &CloverActivation,
        valid: &[bool],
        preview: bool,
    ) -> Result<CloverActivation, String>;

    fn forward_full_attention(
        &mut self,
        layer: usize,
        input: &CloverActivation,
        valid: &[bool],
        prefix_valid: &[bool],
        positions: &[usize],
        preview: bool,
    ) -> Result<CloverActivation, String>;

    /// Return the bias-free router projection rounded to BF16 then widened to
    /// FP32, shaped [tokens, num_experts].
    fn router_logits(&mut self, layer: usize, input: &CloverActivation)
        -> Result<Vec<f32>, String>;

    /// Execute routed top-k experts and the learned shared expert. The backend
    /// must honor zero-weight routes and retain all weight/cache ownership.
    fn forward_moe(
        &mut self,
        layer: usize,
        input: &CloverActivation,
        routing: &CloverRoutingPlan,
        valid: &[bool],
        preview: bool,
    ) -> Result<CloverActivation, String>;

    fn final_norm(&mut self, input: &CloverActivation) -> Result<CloverActivation, String>;

    /// Project final BF16 hidden states to FP32 vocabulary logits. Called only
    /// for block previews; causal prefill/replay avoids LM-head work.
    fn lm_head(&mut self, input: &CloverActivation) -> Result<Vec<f32>, String>;

    /// Release request-local clean caches after a terminal session.
    fn reset_session(&mut self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloverForwardMode {
    Preview,
    Causal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CloverGraphOutput {
    pub hidden: CloverActivation,
    /// Present only for Preview mode, shaped [tokens, vocab_size].
    pub logits: Option<Vec<f32>>,
    /// One immutable top-k/coreset plan for each decoder layer.
    pub routing: Vec<CloverRoutingPlan>,
}

#[derive(Debug, Error)]
pub enum CloverGraphError {
    #[error(transparent)]
    Config(#[from] CloverConfigError),
    #[error(transparent)]
    Moe(#[from] CloverMoeError),
    #[error("activation shape is invalid: tokens={tokens}, hidden={hidden_size}, values={values}")]
    ActivationShape {
        tokens: usize,
        hidden_size: usize,
        values: usize,
    },
    #[error("backend {operation} returned {actual_tokens}x{actual_hidden}, expected {expected_tokens}x{expected_hidden} at layer {layer:?}")]
    BackendShape {
        layer: Option<usize>,
        operation: &'static str,
        expected_tokens: usize,
        expected_hidden: usize,
        actual_tokens: usize,
        actual_hidden: usize,
    },
    #[error("backend operation {operation} failed at layer {layer:?}: {message}")]
    Backend {
        layer: Option<usize>,
        operation: &'static str,
        message: String,
    },
    #[error("router logits for layer {layer} have {actual} values, expected {expected}")]
    RouterLogitShape {
        layer: usize,
        actual: usize,
        expected: usize,
    },
    #[error("LM head returned {actual} logits, expected {expected}")]
    LogitShape { actual: usize, expected: usize },
    #[error("LM head returned a non-finite logit")]
    NonFiniteLogit,
    #[error("validity and position arrays must match the current token count")]
    InputShape,
    #[error("preview token count {requested} must be positive, no larger than configured block size {configured}, and divide it exactly")]
    InvalidPreviewBlockSize { requested: usize, configured: usize },
    #[error("prefix plus current tokens exceeds Clover's maximum context")]
    ContextExceeded,
    #[error("a valid token position exceeds Clover's maximum context")]
    PositionOutOfRange,
    #[error("activation size arithmetic overflow")]
    SizeOverflow,
}

/// Immutable composition of the Clover layer pattern and MoE router policy.
#[derive(Debug, Clone)]
pub struct CloverGraph {
    config: CloverConfig,
    layers: Vec<LayerType>,
    router: CloverMoeRouter,
}

impl CloverGraph {
    /// Build a graph for a validated Clover-compatible config. Reduced configs
    /// are accepted for CPU fixtures; production should use new_pinned.
    pub fn new(config: CloverConfig) -> Result<Self, CloverGraphError> {
        config.validate()?;
        let layers = config.layer_types.clone();
        let router = CloverMoeRouter::from_config(&config);
        router.validate()?;
        Ok(Self {
            config,
            layers,
            router,
        })
    }

    /// Require the exact audited Clover-35B-A3B-Preview architecture.
    pub fn new_pinned(config: CloverConfig) -> Result<Self, CloverGraphError> {
        config.validate_pinned()?;
        Self::new(config)
    }

    pub fn config(&self) -> &CloverConfig {
        &self.config
    }
    pub fn layer_types(&self) -> &[LayerType] {
        &self.layers
    }

    pub fn forward_preview<B: CloverGraphBackend>(
        &self,
        backend: &mut B,
        token_ids: &[u32],
        valid: &[bool],
        prefix_valid: &[bool],
        positions: &[usize],
    ) -> Result<CloverGraphOutput, CloverGraphError> {
        self.forward(
            backend,
            token_ids,
            valid,
            prefix_valid,
            positions,
            CloverForwardMode::Preview,
        )
    }

    /// Causal prefill/replay advances clean states and skips vocabulary projection.
    pub fn forward_causal<B: CloverGraphBackend>(
        &self,
        backend: &mut B,
        token_ids: &[u32],
        valid: &[bool],
        prefix_valid: &[bool],
        positions: &[usize],
    ) -> Result<CloverGraphOutput, CloverGraphError> {
        self.forward(
            backend,
            token_ids,
            valid,
            prefix_valid,
            positions,
            CloverForwardMode::Causal,
        )
    }

    fn forward<B: CloverGraphBackend>(
        &self,
        backend: &mut B,
        token_ids: &[u32],
        valid: &[bool],
        prefix_valid: &[bool],
        positions: &[usize],
        mode: CloverForwardMode,
    ) -> Result<CloverGraphOutput, CloverGraphError> {
        let tokens = token_ids.len();
        if tokens == 0 || valid.len() != tokens || positions.len() != tokens {
            return Err(CloverGraphError::InputShape);
        }
        if prefix_valid
            .len()
            .checked_add(tokens)
            .ok_or(CloverGraphError::SizeOverflow)?
            > self.config.max_position_embeddings
        {
            return Err(CloverGraphError::ContextExceeded);
        }
        if positions.iter().zip(valid).any(|(&position, &is_valid)| {
            is_valid && position >= self.config.max_position_embeddings
        }) {
            return Err(CloverGraphError::PositionOutOfRange);
        }
        let preview = mode == CloverForwardMode::Preview;
        if preview && (tokens > self.config.block_size || self.config.block_size % tokens != 0) {
            return Err(CloverGraphError::InvalidPreviewBlockSize {
                requested: tokens,
                configured: self.config.block_size,
            });
        }
        let mut hidden =
            backend
                .embed_tokens(token_ids)
                .map_err(|message| CloverGraphError::Backend {
                    layer: None,
                    operation: "embed_tokens",
                    message,
                })?;
        hidden.validate_like(tokens, self.config.hidden_size, None, "embed_tokens")?;

        let mut routing_plans = Vec::with_capacity(self.layers.len());
        for (layer, layer_type) in self.layers.iter().copied().enumerate() {
            let normalized = backend
                .rms_norm(layer, CloverNormKind::Input, &hidden)
                .map_err(|message| CloverGraphError::Backend {
                    layer: Some(layer),
                    operation: "input_layer_norm",
                    message,
                })?;
            normalized.validate_like(
                tokens,
                self.config.hidden_size,
                Some(layer),
                "input_layer_norm",
            )?;

            let attention = match layer_type {
                LayerType::LinearAttention => backend
                    .forward_gdn(layer, &normalized, valid, preview)
                    .map_err(|message| CloverGraphError::Backend {
                        layer: Some(layer),
                        operation: "gdn",
                        message,
                    })?,
                LayerType::FullAttention => backend
                    .forward_full_attention(
                        layer,
                        &normalized,
                        valid,
                        prefix_valid,
                        positions,
                        preview,
                    )
                    .map_err(|message| CloverGraphError::Backend {
                        layer: Some(layer),
                        operation: "full_attention",
                        message,
                    })?,
            };
            attention.validate_like(tokens, self.config.hidden_size, Some(layer), "attention")?;
            hidden = residual_add(&hidden, &attention)?;

            let moe_input = backend
                .rms_norm(layer, CloverNormKind::PostAttention, &hidden)
                .map_err(|message| CloverGraphError::Backend {
                    layer: Some(layer),
                    operation: "post_attention_layer_norm",
                    message,
                })?;
            moe_input.validate_like(
                tokens,
                self.config.hidden_size,
                Some(layer),
                "post_attention_layer_norm",
            )?;

            let router_logits = backend
                .router_logits(layer, &moe_input)
                .map_err(|message| CloverGraphError::Backend {
                    layer: Some(layer),
                    operation: "router_logits",
                    message,
                })?;
            let expected = tokens
                .checked_mul(self.config.num_experts)
                .ok_or(CloverGraphError::SizeOverflow)?;
            if router_logits.len() != expected {
                return Err(CloverGraphError::RouterLogitShape {
                    layer,
                    actual: router_logits.len(),
                    expected,
                });
            }
            let routing = self.router.route(&router_logits, tokens, valid, preview)?;
            let moe = backend
                .forward_moe(layer, &moe_input, &routing, valid, preview)
                .map_err(|message| CloverGraphError::Backend {
                    layer: Some(layer),
                    operation: "moe",
                    message,
                })?;
            moe.validate_like(tokens, self.config.hidden_size, Some(layer), "moe")?;
            hidden = residual_add(&hidden, &moe)?;
            routing_plans.push(routing);
        }

        let hidden = backend
            .final_norm(&hidden)
            .map_err(|message| CloverGraphError::Backend {
                layer: None,
                operation: "final_norm",
                message,
            })?;
        hidden.validate_like(tokens, self.config.hidden_size, None, "final_norm")?;

        let logits = if preview {
            let values = backend
                .lm_head(&hidden)
                .map_err(|message| CloverGraphError::Backend {
                    layer: None,
                    operation: "lm_head",
                    message,
                })?;
            let expected = tokens
                .checked_mul(self.config.vocab_size)
                .ok_or(CloverGraphError::SizeOverflow)?;
            if values.len() != expected {
                return Err(CloverGraphError::LogitShape {
                    actual: values.len(),
                    expected,
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(CloverGraphError::NonFiniteLogit);
            }
            Some(values)
        } else {
            None
        };

        Ok(CloverGraphOutput {
            hidden,
            logits,
            routing: routing_plans,
        })
    }
}

fn residual_add(
    left: &CloverActivation,
    right: &CloverActivation,
) -> Result<CloverActivation, CloverGraphError> {
    if left.tokens != right.tokens || left.hidden_size != right.hidden_size {
        return Err(CloverGraphError::BackendShape {
            layer: None,
            operation: "residual_add",
            expected_tokens: left.tokens,
            expected_hidden: left.hidden_size,
            actual_tokens: right.tokens,
            actual_hidden: right.hidden_size,
        });
    }
    let values = left
        .values
        .iter()
        .zip(&right.values)
        .map(|(a, b)| Bf16::from_f32(a.to_f32() + b.to_f32()))
        .collect();
    CloverActivation::new(left.tokens, left.hidden_size, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CloverConfig;

    const CONFIG: &str = include_str!("../../../docs/audits/fixtures/clover-config.json");

    #[derive(Default)]
    struct RecordingBackend {
        events: Vec<String>,
        hidden_size: usize,
        vocab_size: usize,
        num_experts: usize,
    }

    impl CloverGraphBackend for RecordingBackend {
        fn embed_tokens(&mut self, ids: &[u32]) -> Result<CloverActivation, String> {
            self.events.push("embed".to_owned());
            CloverActivation::new(
                ids.len(),
                self.hidden_size,
                vec![Bf16::from_f32(1.0); ids.len() * self.hidden_size],
            )
            .map_err(|error| error.to_string())
        }
        fn rms_norm(
            &mut self,
            layer: usize,
            kind: CloverNormKind,
            input: &CloverActivation,
        ) -> Result<CloverActivation, String> {
            self.events.push(format!("norm:{layer}:{kind:?}"));
            Ok(input.clone())
        }
        fn forward_gdn(
            &mut self,
            layer: usize,
            input: &CloverActivation,
            _valid: &[bool],
            preview: bool,
        ) -> Result<CloverActivation, String> {
            self.events.push(format!("gdn:{layer}:{preview}"));
            CloverActivation::new(
                input.tokens,
                input.hidden_size,
                vec![Bf16::from_f32(0.0); input.values.len()],
            )
            .map_err(|error| error.to_string())
        }
        fn forward_full_attention(
            &mut self,
            layer: usize,
            input: &CloverActivation,
            _valid: &[bool],
            prefix: &[bool],
            positions: &[usize],
            preview: bool,
        ) -> Result<CloverActivation, String> {
            self.events.push(format!(
                "attention:{layer}:{preview}:{}:{}",
                prefix.len(),
                positions[0]
            ));
            CloverActivation::new(
                input.tokens,
                input.hidden_size,
                vec![Bf16::from_f32(0.0); input.values.len()],
            )
            .map_err(|error| error.to_string())
        }
        fn router_logits(
            &mut self,
            layer: usize,
            input: &CloverActivation,
        ) -> Result<Vec<f32>, String> {
            self.events.push(format!("router:{layer}"));
            let experts = if self.num_experts == 0 {
                4
            } else {
                self.num_experts
            };
            Ok((0..input.tokens)
                .flat_map(|_| {
                    (0..experts).map(|expert| Bf16::from_f32(expert as f32 / 4.0).to_f32())
                })
                .collect())
        }
        fn forward_moe(
            &mut self,
            layer: usize,
            input: &CloverActivation,
            routing: &CloverRoutingPlan,
            _valid: &[bool],
            preview: bool,
        ) -> Result<CloverActivation, String> {
            self.events.push(format!(
                "moe:{layer}:{preview}:{}",
                routing.unique_experts().len()
            ));
            CloverActivation::new(
                input.tokens,
                input.hidden_size,
                vec![Bf16::from_f32(0.0); input.values.len()],
            )
            .map_err(|error| error.to_string())
        }
        fn final_norm(&mut self, input: &CloverActivation) -> Result<CloverActivation, String> {
            self.events.push("final_norm".to_owned());
            Ok(input.clone())
        }
        fn lm_head(&mut self, input: &CloverActivation) -> Result<Vec<f32>, String> {
            self.events.push("lm_head".to_owned());
            let mut logits = vec![0.0; input.tokens * self.vocab_size];
            for (token, row) in logits.chunks_exact_mut(self.vocab_size).enumerate() {
                row[0] = input.values[token * input.hidden_size].to_f32();
            }
            Ok(logits)
        }
    }

    fn tiny_config() -> CloverConfig {
        let mut c = CloverConfig::from_json(CONFIG).unwrap();
        c.hidden_size = 4;
        c.num_hidden_layers = 3;
        c.layer_types = vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ];
        c.num_experts = 4;
        c.num_experts_per_tok = 2;
        c.vocab_size = 8;
        c.mask_token_id = 7;
        c.block_size = 4;
        c.max_position_embeddings = 32;
        c.num_attention_heads = 1;
        c.num_key_value_heads = 1;
        c.linear_num_key_heads = 1;
        c.linear_num_value_heads = 1;
        c.moe_intermediate_size = 2;
        c.shared_expert_intermediate_size = 2;
        c.validate().unwrap();
        c
    }

    #[test]
    fn traverses_gdn_full_attention_and_moe_in_layer_order() {
        let graph = CloverGraph::new(tiny_config()).unwrap();
        let mut backend = RecordingBackend {
            hidden_size: 4,
            vocab_size: 8,
            ..Default::default()
        };
        let output = graph
            .forward_preview(&mut backend, &[1, 2], &[true, true], &[true; 3], &[3, 4])
            .unwrap();
        assert_eq!(output.routing.len(), 3);
        assert!(output.logits.is_some());
        assert_eq!(
            backend.events,
            vec![
                "embed",
                "norm:0:Input",
                "gdn:0:true",
                "norm:0:PostAttention",
                "router:0",
                "moe:0:true:2",
                "norm:1:Input",
                "attention:1:true:3:3",
                "norm:1:PostAttention",
                "router:1",
                "moe:1:true:2",
                "norm:2:Input",
                "gdn:2:true",
                "norm:2:PostAttention",
                "router:2",
                "moe:2:true:2",
                "final_norm",
                "lm_head",
            ]
        );
    }

    #[test]
    fn causal_graph_skips_lm_head_and_coreset() {
        let graph = CloverGraph::new(tiny_config()).unwrap();
        let mut backend = RecordingBackend {
            hidden_size: 4,
            vocab_size: 8,
            ..Default::default()
        };
        let output = graph
            .forward_causal(&mut backend, &[1], &[true], &[], &[0])
            .unwrap();
        assert!(output.logits.is_none());
        assert!(!backend.events.iter().any(|event| event == "lm_head"));
        assert!(output.routing.iter().all(|route| route.coreset.is_none()));
    }

    #[test]
    fn rejects_context_overflow_before_backend_execution() {
        let graph = CloverGraph::new(tiny_config()).unwrap();
        let mut backend = RecordingBackend {
            hidden_size: 4,
            vocab_size: 8,
            ..Default::default()
        };
        assert!(matches!(
            graph.forward_causal(&mut backend, &[1, 2], &[true; 2], &[true; 31], &[31, 32]),
            Err(CloverGraphError::ContextExceeded)
        ));
        assert!(backend.events.is_empty());
    }

    struct LayerReferenceBackend {
        hidden_size: usize,
        vocab_size: usize,
    }

    impl CloverGraphBackend for LayerReferenceBackend {
        fn embed_tokens(&mut self, ids: &[u32]) -> Result<CloverActivation, String> {
            CloverActivation::new(
                ids.len(),
                self.hidden_size,
                vec![Bf16::from_f32(1.0); ids.len() * self.hidden_size],
            )
            .map_err(|error| error.to_string())
        }

        fn rms_norm(
            &mut self,
            _layer: usize,
            _kind: CloverNormKind,
            input: &CloverActivation,
        ) -> Result<CloverActivation, String> {
            Ok(input.clone())
        }

        fn forward_gdn(
            &mut self,
            _layer: usize,
            input: &CloverActivation,
            _valid: &[bool],
            _preview: bool,
        ) -> Result<CloverActivation, String> {
            CloverActivation::new(
                input.tokens,
                input.hidden_size,
                vec![Bf16::from_f32(0.5); input.values.len()],
            )
            .map_err(|error| error.to_string())
        }

        fn forward_full_attention(
            &mut self,
            _layer: usize,
            input: &CloverActivation,
            _valid: &[bool],
            _prefix: &[bool],
            _positions: &[usize],
            _preview: bool,
        ) -> Result<CloverActivation, String> {
            CloverActivation::new(
                input.tokens,
                input.hidden_size,
                vec![Bf16::from_f32(1.0); input.values.len()],
            )
            .map_err(|error| error.to_string())
        }

        fn router_logits(
            &mut self,
            _layer: usize,
            input: &CloverActivation,
        ) -> Result<Vec<f32>, String> {
            Ok((0..input.tokens)
                .flat_map(|_| [0.0, 1.0, 2.0, 3.0])
                .collect())
        }

        fn forward_moe(
            &mut self,
            _layer: usize,
            input: &CloverActivation,
            _routing: &CloverRoutingPlan,
            _valid: &[bool],
            _preview: bool,
        ) -> Result<CloverActivation, String> {
            CloverActivation::new(
                input.tokens,
                input.hidden_size,
                vec![Bf16::from_f32(0.25); input.values.len()],
            )
            .map_err(|error| error.to_string())
        }

        fn final_norm(&mut self, input: &CloverActivation) -> Result<CloverActivation, String> {
            Ok(input.clone())
        }

        fn lm_head(&mut self, input: &CloverActivation) -> Result<Vec<f32>, String> {
            let mut logits = vec![0.0; input.tokens * self.vocab_size];
            for (token, row) in logits.chunks_exact_mut(self.vocab_size).enumerate() {
                row[0] = input.values[token * self.hidden_size].to_f32();
                row[1] = token as f32;
            }
            Ok(logits)
        }
    }

    #[test]
    fn injected_layer_references_match_residual_path_and_same_position_logits() {
        let graph = CloverGraph::new(tiny_config()).unwrap();
        let mut backend = LayerReferenceBackend {
            hidden_size: 4,
            vocab_size: 8,
        };
        let output = graph
            .forward_preview(&mut backend, &[1, 2], &[true, true], &[true, true], &[2, 3])
            .unwrap();

        // 1.0 + GDN 0.5 + MoE 0.25 + full-attention 1.0
        // + MoE 0.25 + GDN 0.5 + MoE 0.25 = 3.75.
        assert!(output
            .hidden
            .values()
            .iter()
            .all(|value| (value.to_f32() - 3.75).abs() < 0.01));
        let logits = output.logits.unwrap();
        assert_eq!(logits[0], 3.75);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[8], 3.75);
        assert_eq!(logits[9], 1.0);
        for routing in output.routing {
            assert_eq!(routing.routes[0].expert_ids, vec![3, 2]);
            assert!((routing.routes[0].weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn each_pinned_layer_uses_configured_top_eight_and_bf16_router_scores() {
        let mut config = tiny_config();
        config.num_experts = 256;
        config.num_experts_per_tok = 8;
        config.moe_block_coreset_enabled = false;
        let graph = CloverGraph::new(config).unwrap();
        let mut backend = RecordingBackend {
            hidden_size: 4,
            vocab_size: 8,
            num_experts: 256,
            ..RecordingBackend::default()
        };
        let output = graph
            .forward_preview(&mut backend, &[1], &[true], &[true], &[1])
            .unwrap();
        assert_eq!(output.routing.len(), 3);
        for routing in output.routing {
            let mut selected = routing.routes[0].expert_ids.clone();
            selected.sort_unstable();
            assert_eq!(selected, (248..256).collect::<Vec<_>>());
            assert_eq!(routing.routes[0].weights.len(), 8);
            assert!((routing.routes[0].weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        assert_eq!(output.logits.unwrap().len(), 8);
    }

    #[test]
    fn production_graph_requires_and_preserves_the_pinned_forty_layer_pattern() {
        let config = CloverConfig::from_json(CONFIG).unwrap();
        let graph = CloverGraph::new_pinned(config).unwrap();
        assert_eq!(graph.layer_types().len(), 40);
        assert_eq!(
            graph
                .layer_types()
                .iter()
                .filter(|kind| **kind == LayerType::FullAttention)
                .count(),
            10
        );
        assert_eq!(
            graph
                .layer_types()
                .iter()
                .filter(|kind| **kind == LayerType::LinearAttention)
                .count(),
            30
        );
    }

    #[test]
    fn preview_block_size_must_divide_the_configured_block() {
        let graph = CloverGraph::new(tiny_config()).unwrap();
        let mut backend = RecordingBackend {
            hidden_size: 4,
            vocab_size: 8,
            ..RecordingBackend::default()
        };
        assert!(matches!(
            graph.forward_preview(&mut backend, &[1, 2, 3], &[true; 3], &[], &[0, 1, 2],),
            Err(CloverGraphError::InvalidPreviewBlockSize {
                requested: 3,
                configured: 4,
            })
        ));
        assert!(backend.events.is_empty());
    }
}
