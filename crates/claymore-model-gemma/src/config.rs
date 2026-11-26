//! Pinned-model configuration parsing and validation for DiffusionGemma.
//!
//! The M10 adapter targets exactly one public checkpoint revision. Structural
//! validation (dtype, dimensions, layer layout) is separated from pin
//! validation so that synthetic tests can exercise the parser while real
//! execution must pass both.

use std::collections::BTreeMap;

use serde::Deserialize;
use thiserror::Error;

/// Exact supported Hugging Face checkpoint identifier.
pub const MODEL_ID: &str = "google/diffusiongemma-26B-A4B-it";

/// Immutable Hugging Face revision audited for M10.
pub const CHECKPOINT_REVISION: &str = "cd40195554d3cfcb56f34b9adfb66efc790ed5b6";

/// Number of Safetensors shards in the pinned revision.
pub const PINNED_SHARD_COUNT: usize = 11;

/// Number of tensors declared by the pinned shard index.
pub const PINNED_TENSOR_COUNT: usize = 1047;

/// Total tensor payload in bytes declared by the pinned shard index.
pub const PINNED_TOTAL_BYTES: u64 = 51_647_562_456;

/// Parameter count declared by the pinned shard index (all tensors are BF16).
pub const PINNED_TOTAL_PARAMETERS: u64 = 25_823_778_864;

/// SHA-256 of the pinned `config.json`.
pub const PINNED_CONFIG_SHA256: &str =
    "13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506";

/// SHA-256 of the pinned `model.safetensors.index.json`.
pub const PINNED_INDEX_SHA256: &str =
    "6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a";

/// SHA-256 of the pinned `generation_config.json`.
pub const PINNED_GENERATION_CONFIG_SHA256: &str =
    "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200";

/// Audited initial canvas length (also present in the pinned config).
pub const PINNED_CANVAS_LENGTH: usize = 256;

/// Audited text hidden size.
pub const PINNED_HIDDEN_SIZE: usize = 2816;

/// Audited decoder/encoder text layer count.
pub const PINNED_TEXT_LAYERS: usize = 30;

/// Audited vision tower layer count.
pub const PINNED_VISION_LAYERS: usize = 27;

/// Audited vocabulary size.
pub const PINNED_VOCAB_SIZE: usize = 262_144;

/// One or many token identifiers, as accepted by the pinned configs.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TokenIdList {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIdList {
    pub fn ids(&self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.clone(),
        }
    }
}

/// RoPE parameters for one attention layer type.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

/// Text-tower configuration subset required by the pinned graph.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TextConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: BTreeMap<String, RopeParameters>,
    #[serde(default)]
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdList>,
}

/// Audited attention geometry for one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerAttention {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub global: bool,
}

impl TextConfig {
    pub fn layer_type(&self, layer: usize) -> Option<&str> {
        self.layer_types.get(layer).map(String::as_str)
    }

    /// Attention geometry for a decoder/encoder text layer. Global layers use
    /// their own head dimension and KV-head count in the pinned graph.
    pub fn layer_attention(&self, layer: usize) -> Option<LayerAttention> {
        let global = self.layer_type(layer)? == "full_attention";
        let heads = self.num_attention_heads?;
        Some(LayerAttention {
            heads,
            kv_heads: if global {
                self.num_global_key_value_heads?
            } else {
                self.num_key_value_heads?
            },
            head_dim: if global {
                self.global_head_dim?
            } else {
                self.head_dim?
            },
            global,
        })
    }

    pub fn is_global(&self, layer: usize) -> bool {
        self.layer_type(layer) == Some("full_attention")
    }
}

/// Vision tower subset pinned for provenance/weight validation.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VisionConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub patch_size: Option<usize>,
    #[serde(default)]
    pub position_embedding_size: Option<usize>,
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
}

/// Parsed subset of the upstream configuration. Unknown fields are preserved
/// by serde's default forward-compatible behavior; model execution must still
/// validate the full pinned revision before loading any weights.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GemmaModelConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub canvas_length: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdList>,
    #[serde(default)]
    pub boi_token_id: Option<u32>,
    #[serde(default)]
    pub eoi_token_id: Option<u32>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub text_config: Option<TextConfig>,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
}

impl GemmaModelConfig {
    pub fn from_json(json: &str) -> Result<Self, GemmaConfigError> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate only invariants established by the model audit. Missing
    /// optional architecture fields are not guessed or silently defaulted.
    pub fn validate(&self) -> Result<(), GemmaConfigError> {
        for dtype in [self.dtype.as_deref(), self.torch_dtype.as_deref()]
            .into_iter()
            .flatten()
        {
            if !is_bf16(dtype) {
                return Err(GemmaConfigError::UnsupportedWeightDType(dtype.to_owned()));
            }
        }

        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("vocab_size", self.vocab_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("canvas_length", self.canvas_length),
        ] {
            if value == Some(0) {
                return Err(GemmaConfigError::ZeroDimension(name));
            }
        }

        if let Some(text) = &self.text_config {
            text.validate()?;
        }
        if let Some(vision) = &self.vision_config {
            vision.validate()?;
        }
        Ok(())
    }

    /// Require this configuration to be exactly the audited checkpoint.
    /// Structural validation must already have passed.
    pub fn validate_pinned(&self) -> Result<(), GemmaConfigError> {
        let mut check = PinCheck::default();
        check.equal(
            "model_type",
            self.model_type.as_deref(),
            Some("diffusion_gemma"),
        );
        check.contains(
            "architectures",
            &self.architectures,
            "DiffusionGemmaForBlockDiffusion",
        );
        check.equal("dtype", self.dtype.as_deref(), Some("bfloat16"));
        check.equal(
            "canvas_length",
            self.canvas_length,
            Some(PINNED_CANVAS_LENGTH),
        );
        check.equal("tie_word_embeddings", self.tie_word_embeddings, Some(true));
        check.equal(
            "eos_token_id",
            self.eos_token_id.as_ref().map(TokenIdList::ids),
            Some(vec![1, 106]),
        );
        check.equal("boi_token_id", self.boi_token_id, Some(255_999));
        check.equal("eoi_token_id", self.eoi_token_id, Some(258_882));
        check.equal("image_token_id", self.image_token_id, Some(258_880));

        let text = self
            .text_config
            .as_ref()
            .ok_or_else(|| GemmaConfigError::PinnedMismatch {
                field: "text_config".to_owned(),
                expected: "present".to_owned(),
                actual: "missing".to_owned(),
            })?;
        check.equal(
            "text_config.model_type",
            text.model_type.as_deref(),
            Some("diffusion_gemma_text"),
        );
        check.equal("text_config.dtype", text.dtype.as_deref(), Some("bfloat16"));
        check.equal(
            "text_config.hidden_size",
            text.hidden_size,
            Some(PINNED_HIDDEN_SIZE),
        );
        check.equal(
            "text_config.num_hidden_layers",
            text.num_hidden_layers,
            Some(PINNED_TEXT_LAYERS),
        );
        check.equal(
            "text_config.num_attention_heads",
            text.num_attention_heads,
            Some(16),
        );
        check.equal(
            "text_config.num_key_value_heads",
            text.num_key_value_heads,
            Some(8),
        );
        check.equal(
            "text_config.num_global_key_value_heads",
            text.num_global_key_value_heads,
            Some(2),
        );
        check.equal("text_config.head_dim", text.head_dim, Some(256));
        check.equal(
            "text_config.global_head_dim",
            text.global_head_dim,
            Some(512),
        );
        check.equal(
            "text_config.intermediate_size",
            text.intermediate_size,
            Some(2112),
        );
        check.equal(
            "text_config.moe_intermediate_size",
            text.moe_intermediate_size,
            Some(704),
        );
        check.equal("text_config.num_experts", text.num_experts, Some(128));
        check.equal("text_config.top_k_experts", text.top_k_experts, Some(8));
        check.equal(
            "text_config.vocab_size",
            text.vocab_size,
            Some(PINNED_VOCAB_SIZE),
        );
        check.equal(
            "text_config.sliding_window",
            text.sliding_window,
            Some(1024),
        );
        check.equal(
            "text_config.max_position_embeddings",
            text.max_position_embeddings,
            Some(262_144),
        );
        check.float_equal("text_config.rms_norm_eps", text.rms_norm_eps, Some(1e-6));
        check.float_equal(
            "text_config.final_logit_softcapping",
            text.final_logit_softcapping,
            Some(30.0),
        );
        check.equal(
            "text_config.hidden_activation",
            text.hidden_activation.as_deref(),
            Some("gelu_pytorch_tanh"),
        );
        check.equal(
            "text_config.use_bidirectional_attention",
            text.use_bidirectional_attention.as_deref(),
            Some("vision"),
        );
        check.equal(
            "text_config.attention_bias",
            text.attention_bias,
            Some(false),
        );
        check.equal(
            "text_config.tie_word_embeddings",
            text.tie_word_embeddings,
            Some(true),
        );
        let expected_layer_types: Vec<String> = (0..PINNED_TEXT_LAYERS)
            .map(|layer| {
                if layer % 6 == 5 {
                    "full_attention".to_owned()
                } else {
                    "sliding_attention".to_owned()
                }
            })
            .collect();
        check.equal(
            "text_config.layer_types",
            if text.layer_types.is_empty() {
                None
            } else {
                Some(text.layer_types.clone())
            },
            Some(expected_layer_types),
        );
        check_rope(
            &mut check,
            "text_config.rope_parameters.sliding_attention",
            text.rope_parameters.get("sliding_attention"),
            10_000.0,
            "default",
            None,
        );
        check_rope(
            &mut check,
            "text_config.rope_parameters.full_attention",
            text.rope_parameters.get("full_attention"),
            1_000_000.0,
            "proportional",
            Some(0.25),
        );

        let vision =
            self.vision_config
                .as_ref()
                .ok_or_else(|| GemmaConfigError::PinnedMismatch {
                    field: "vision_config".to_owned(),
                    expected: "present".to_owned(),
                    actual: "missing".to_owned(),
                })?;
        check.equal(
            "vision_config.model_type",
            vision.model_type.as_deref(),
            Some("gemma4_vision"),
        );
        check.equal(
            "vision_config.dtype",
            vision.dtype.as_deref(),
            Some("bfloat16"),
        );
        check.equal("vision_config.hidden_size", vision.hidden_size, Some(1152));
        check.equal(
            "vision_config.num_hidden_layers",
            vision.num_hidden_layers,
            Some(PINNED_VISION_LAYERS),
        );
        check.equal(
            "vision_config.num_attention_heads",
            vision.num_attention_heads,
            Some(16),
        );
        check.equal(
            "vision_config.num_key_value_heads",
            vision.num_key_value_heads,
            Some(16),
        );
        check.equal("vision_config.head_dim", vision.head_dim, Some(72));
        check.equal(
            "vision_config.global_head_dim",
            vision.global_head_dim,
            Some(72),
        );
        check.equal(
            "vision_config.intermediate_size",
            vision.intermediate_size,
            Some(4304),
        );
        check.equal("vision_config.patch_size", vision.patch_size, Some(16));
        check.equal(
            "vision_config.position_embedding_size",
            vision.position_embedding_size,
            Some(10_240),
        );
        check.float_equal(
            "vision_config.rms_norm_eps",
            vision.rms_norm_eps,
            Some(1e-6),
        );

        check.finish()
    }

    pub fn requires_bf16(&self) -> bool {
        [self.dtype.as_deref(), self.torch_dtype.as_deref()]
            .into_iter()
            .flatten()
            .any(is_bf16)
    }
}

impl TextConfig {
    pub fn validate(&self) -> Result<(), GemmaConfigError> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            (
                "num_global_key_value_heads",
                self.num_global_key_value_heads,
            ),
            ("head_dim", self.head_dim),
            ("global_head_dim", self.global_head_dim),
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_experts", self.num_experts),
            ("top_k_experts", self.top_k_experts),
            ("vocab_size", self.vocab_size),
            ("sliding_window", self.sliding_window),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == Some(0) {
                return Err(GemmaConfigError::ZeroDimension(name));
            }
        }
        if let Some(dtype) = self.dtype.as_deref() {
            if !is_bf16(dtype) {
                return Err(GemmaConfigError::UnsupportedWeightDType(dtype.to_owned()));
            }
        }
        if let (Some(layers), false) = (self.num_hidden_layers, self.layer_types.is_empty()) {
            if layers != self.layer_types.len() {
                return Err(GemmaConfigError::LayerCountMismatch {
                    layers,
                    types: self.layer_types.len(),
                });
            }
        }
        for layer_type in &self.layer_types {
            if !matches!(layer_type.as_str(), "sliding_attention" | "full_attention") {
                return Err(GemmaConfigError::UnsupportedLayerType(layer_type.clone()));
            }
        }
        if let (Some(heads), Some(kv_heads)) = (self.num_attention_heads, self.num_key_value_heads)
        {
            if heads % kv_heads != 0 {
                return Err(GemmaConfigError::HeadCountMismatch { heads, kv_heads });
            }
        }
        if let (Some(experts), Some(top_k)) = (self.num_experts, self.top_k_experts) {
            if top_k > experts {
                return Err(GemmaConfigError::TopKExceedsExperts { experts, top_k });
            }
        }
        for (layer_type, params) in &self.rope_parameters {
            if let Some(theta) = params.rope_theta {
                if !theta.is_finite() || theta <= 0.0 {
                    return Err(GemmaConfigError::InvalidRopeTheta {
                        layer_type: layer_type.clone(),
                        theta,
                    });
                }
            }
            if let Some(factor) = params.partial_rotary_factor {
                if !factor.is_finite() || factor <= 0.0 || factor > 1.0 {
                    return Err(GemmaConfigError::InvalidPartialRotaryFactor {
                        layer_type: layer_type.clone(),
                        factor,
                    });
                }
            }
            match params.rope_type.as_deref() {
                None | Some("default") | Some("proportional") => {}
                Some(other) => return Err(GemmaConfigError::UnsupportedRopeType(other.to_owned())),
            }
        }
        Ok(())
    }
}

impl VisionConfig {
    pub fn validate(&self) -> Result<(), GemmaConfigError> {
        for (name, value) in [
            ("vision.hidden_size", self.hidden_size),
            ("vision.num_hidden_layers", self.num_hidden_layers),
            ("vision.num_attention_heads", self.num_attention_heads),
            ("vision.num_key_value_heads", self.num_key_value_heads),
            ("vision.head_dim", self.head_dim),
            ("vision.global_head_dim", self.global_head_dim),
            ("vision.intermediate_size", self.intermediate_size),
            ("vision.patch_size", self.patch_size),
            (
                "vision.position_embedding_size",
                self.position_embedding_size,
            ),
        ] {
            if value == Some(0) {
                return Err(GemmaConfigError::ZeroDimension(name));
            }
        }
        if let Some(dtype) = self.dtype.as_deref() {
            if !is_bf16(dtype) {
                return Err(GemmaConfigError::UnsupportedWeightDType(dtype.to_owned()));
            }
        }
        if let (Some(heads), Some(kv_heads)) = (self.num_attention_heads, self.num_key_value_heads)
        {
            if heads % kv_heads != 0 {
                return Err(GemmaConfigError::HeadCountMismatch { heads, kv_heads });
            }
        }
        Ok(())
    }
}

fn is_bf16(dtype: &str) -> bool {
    dtype.eq_ignore_ascii_case("bfloat16") || dtype.eq_ignore_ascii_case("bf16")
}

fn check_rope(
    check: &mut PinCheck,
    field: &str,
    params: Option<&RopeParameters>,
    theta: f64,
    rope_type: &str,
    partial: Option<f64>,
) {
    match params {
        None => check.equal(
            field,
            None::<&RopeParameters>,
            Some(&RopeParameters {
                rope_theta: Some(theta),
                rope_type: Some(rope_type.to_owned()),
                partial_rotary_factor: partial,
            }),
        ),
        Some(params) => {
            check.float_equal(
                &format!("{field}.rope_theta"),
                params.rope_theta,
                Some(theta),
            );
            check.equal(
                &format!("{field}.rope_type"),
                params.rope_type.as_deref(),
                Some(rope_type),
            );
            check.float_equal(
                &format!("{field}.partial_rotary_factor"),
                params.partial_rotary_factor,
                partial,
            );
        }
    }
}

#[derive(Default)]
struct PinCheck {
    mismatches: Vec<(String, String, String)>,
}

impl PinCheck {
    fn equal<T: PartialEq + std::fmt::Debug>(
        &mut self,
        field: &str,
        actual: Option<T>,
        expected: Option<T>,
    ) {
        if actual != expected {
            self.mismatches.push((
                field.to_owned(),
                format!("{expected:?}"),
                format!("{actual:?}"),
            ));
        }
    }

    fn contains(&mut self, field: &str, values: &[String], expected: &str) {
        if !values.iter().any(|value| value == expected) {
            self.mismatches.push((
                field.to_owned(),
                format!("contains {expected:?}"),
                format!("{values:?}"),
            ));
        }
    }

    fn float_equal(&mut self, field: &str, actual: Option<f64>, expected: Option<f64>) {
        let equal = match (actual, expected) {
            (None, None) => true,
            (Some(a), Some(e)) => a == e,
            _ => false,
        };
        if !equal {
            self.mismatches.push((
                field.to_owned(),
                format!("{expected:?}"),
                format!("{actual:?}"),
            ));
        }
    }

    fn finish(self) -> Result<(), GemmaConfigError> {
        match self.mismatches.into_iter().next() {
            None => Ok(()),
            Some((field, expected, actual)) => Err(GemmaConfigError::PinnedMismatch {
                field,
                expected,
                actual,
            }),
        }
    }
}

#[derive(Debug, Error)]
pub enum GemmaConfigError {
    #[error("invalid DiffusionGemma config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported DiffusionGemma weight dtype: {0}; only explicit BF16 is accepted")]
    UnsupportedWeightDType(String),
    #[error("configuration field {0} must be non-zero")]
    ZeroDimension(&'static str),
    #[error("config declares {layers} layers but {types} layer types")]
    LayerCountMismatch { layers: usize, types: usize },
    #[error("unsupported attention layer type: {0}")]
    UnsupportedLayerType(String),
    #[error("num_attention_heads {heads} is not divisible by num_key_value_heads {kv_heads}")]
    HeadCountMismatch { heads: usize, kv_heads: usize },
    #[error("top_k_experts {top_k} exceeds num_experts {experts}")]
    TopKExceedsExperts { experts: usize, top_k: usize },
    #[error("invalid rope_theta {theta} for layer type {layer_type}")]
    InvalidRopeTheta { layer_type: String, theta: f64 },
    #[error("invalid partial_rotary_factor {factor} for layer type {layer_type}")]
    InvalidPartialRotaryFactor { layer_type: String, factor: f64 },
    #[error("unsupported rope_type {0}")]
    UnsupportedRopeType(String),
    #[error("pinned configuration mismatch at {field}: expected {expected}, got {actual}")]
    PinnedMismatch {
        field: String,
        expected: String,
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_CONFIG: &str =
        include_str!("../../../docs/audits/fixtures/diffusiongemma-config.json");

    #[test]
    fn parses_bf16_config_without_assuming_optional_fields() {
        let config =
            GemmaModelConfig::from_json(r#"{"torch_dtype":"bfloat16","hidden_size":2048}"#)
                .unwrap();
        assert!(config.requires_bf16());
        assert_eq!(config.hidden_size, Some(2048));
        assert_eq!(config.num_hidden_layers, None);
    }

    #[test]
    fn rejects_implicit_or_unsupported_precision() {
        let error = GemmaModelConfig::from_json(r#"{"torch_dtype":"float16"}"#).unwrap_err();
        assert!(matches!(error, GemmaConfigError::UnsupportedWeightDType(_)));
        let error = GemmaModelConfig::from_json(r#"{"dtype":"float16"}"#).unwrap_err();
        assert!(matches!(error, GemmaConfigError::UnsupportedWeightDType(_)));
    }

    #[test]
    fn rejects_zero_dimensions() {
        let config = GemmaModelConfig {
            hidden_size: Some(0),
            ..GemmaModelConfig::from_json("{}").unwrap()
        };
        assert_eq!(
            config.validate().unwrap_err().to_string(),
            "configuration field hidden_size must be non-zero"
        );
    }

    #[test]
    fn accepts_exactly_the_pinned_configuration() {
        let config = GemmaModelConfig::from_json(PINNED_CONFIG).unwrap();
        config.validate_pinned().unwrap();
        let text = config.text_config.as_ref().unwrap();
        assert_eq!(text.layer_attention(0).unwrap().head_dim, 256);
        assert_eq!(text.layer_attention(5).unwrap().head_dim, 512);
        assert_eq!(text.layer_attention(5).unwrap().kv_heads, 2);
        assert!(text.is_global(29));
        assert!(!text.is_global(28));
        assert_eq!(config.canvas_length, Some(256));
    }

    #[test]
    fn pinned_validation_refuses_any_material_change() {
        let mut config = GemmaModelConfig::from_json(PINNED_CONFIG).unwrap();
        config.canvas_length = Some(128);
        let error = config.validate_pinned().unwrap_err();
        assert!(error.to_string().contains("canvas_length"), "{error}");

        let mut config = GemmaModelConfig::from_json(PINNED_CONFIG).unwrap();
        config.text_config.as_mut().unwrap().head_dim = Some(128);
        let error = config.validate_pinned().unwrap_err();
        assert!(error.to_string().contains("head_dim"), "{error}");

        let mut config = GemmaModelConfig::from_json(PINNED_CONFIG).unwrap();
        config.text_config.as_mut().unwrap().rope_parameters.clear();
        assert!(config.validate_pinned().is_err());
    }

    #[test]
    fn layer_geometry_matches_the_weight_shapes() {
        let config = GemmaModelConfig::from_json(PINNED_CONFIG).unwrap();
        let text = config.text_config.as_ref().unwrap();
        let sliding = text.layer_attention(0).unwrap();
        assert_eq!(
            (
                sliding.heads * sliding.head_dim,
                sliding.kv_heads * sliding.head_dim
            ),
            (4096, 2048)
        );
        let global = text.layer_attention(5).unwrap();
        assert_eq!(
            (
                global.heads * global.head_dim,
                global.kv_heads * global.head_dim
            ),
            (8192, 1024)
        );
    }

    #[test]
    fn rejects_inconsistent_layer_geometry() {
        let base = r#"{
            "text_config": {
                "num_hidden_layers": 2,
                "layer_types": ["sliding_attention"],
                "num_attention_heads": 3,
                "num_key_value_heads": 2,
                "num_experts": 4,
                "top_k_experts": 5
            }
        }"#;
        let error = GemmaModelConfig::from_json(base).unwrap_err();
        assert!(matches!(error, GemmaConfigError::LayerCountMismatch { .. }));

        let error = GemmaModelConfig::from_json(
            r#"{"text_config":{"num_attention_heads":3,"num_key_value_heads":2}}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GemmaConfigError::HeadCountMismatch { .. }));

        let error = GemmaModelConfig::from_json(
            r#"{"text_config":{"num_attention_heads":4,"num_key_value_heads":2,"num_experts":4,"top_k_experts":5}}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GemmaConfigError::TopKExceedsExperts { .. }));
    }

    #[test]
    fn rejects_unknown_layer_types_and_bad_rope_parameters() {
        let error =
            GemmaModelConfig::from_json(r#"{"text_config":{"layer_types":["cross_attention"]}}"#)
                .unwrap_err();
        assert!(matches!(error, GemmaConfigError::UnsupportedLayerType(_)));

        let error = GemmaModelConfig::from_json(
            r#"{"text_config":{"rope_parameters":{"sliding_attention":{"rope_theta":-1.0}}}}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GemmaConfigError::InvalidRopeTheta { .. }));

        let error = GemmaModelConfig::from_json(
            r#"{"text_config":{"rope_parameters":{"full_attention":{"partial_rotary_factor":1.5}}}}"#,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GemmaConfigError::InvalidPartialRotaryFactor { .. }
        ));
    }
}
