//! Typed, fail-closed configuration contract for the pinned Clover preview.
//!
//! This module validates architecture metadata only. It does not load weights,
//! allocate device memory, or claim that a checkpoint is executable.

use serde::Deserialize;
use thiserror::Error;

/// Hugging Face repository identifier.
pub const MODEL_ID: &str = "Diffuspace/Clover-35B-A3B-Preview";

/// Immutable revision that introduced the audited public config.json.
pub const CONFIG_REVISION: &str = "4b0402b65e523d0183a14c431955438714cb16ff";

pub const HIDDEN_SIZE: usize = 2048;
pub const LAYER_COUNT: usize = 40;
pub const EXPERT_COUNT: usize = 256;
pub const EXPERTS_PER_TOKEN: usize = 8;
pub const SHARED_EXPERTS: usize = 1;
pub const VOCAB_SIZE: usize = 248_320;
pub const MAX_CONTEXT: usize = 131_072;
pub const DEFAULT_BLOCK_SIZE: usize = 32;

/// A layer's execution family in the published checkpoint.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RopeParameters {
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DenoisingConfig {
    pub algorithm: String,
    pub threshold: f64,
}

/// The subset of Clover's upstream config that affects graph construction,
/// weight interpretation, memory planning, or generation semantics.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct CloverConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attn_output_gate: bool,
    pub block_size: usize,
    pub bos_token_id: u32,
    pub dtype: String,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub mamba_ssm_dtype: String,
    pub mask_token_id: u32,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub moe_block_coreset_enabled: bool,
    pub moe_block_coreset_top_p: f64,
    pub moe_intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_router_logits: bool,
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub shared_expert_intermediate_size: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
    pub num_shared_experts: usize,
    pub attention_backend: String,
    pub denoising_config: DenoisingConfig,
}

impl CloverConfig {
    pub fn from_json(json: &str) -> Result<Self, CloverConfigError> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate internal consistency without assuming that any arbitrary
    /// Clover-like configuration is the supported pinned checkpoint.
    pub fn validate(&self) -> Result<(), CloverConfigError> {
        for (name, value) in [
            ("block_size", self.block_size),
            ("full_attention_interval", self.full_attention_interval),
            ("head_dim", self.head_dim),
            ("hidden_size", self.hidden_size),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_key_value_heads", self.num_key_value_heads),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            ),
            ("vocab_size", self.vocab_size),
            ("num_shared_experts", self.num_shared_experts),
        ] {
            if value == 0 {
                return Err(CloverConfigError::ZeroDimension(name));
            }
        }

        if !is_bf16(&self.dtype) {
            return Err(CloverConfigError::UnsupportedDType(self.dtype.clone()));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(CloverConfigError::InvalidFloat("rms_norm_eps"));
        }
        if !self.partial_rotary_factor.is_finite()
            || self.partial_rotary_factor <= 0.0
            || self.partial_rotary_factor > 1.0
            || self.rope_parameters.partial_rotary_factor != self.partial_rotary_factor
        {
            return Err(CloverConfigError::InvalidFloat("partial_rotary_factor"));
        }
        if !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
            || self.rope_parameters.rope_type != "default"
        {
            return Err(CloverConfigError::InvalidRope);
        }
        if !self.attention_dropout.is_finite() || !(0.0..=1.0).contains(&self.attention_dropout) {
            return Err(CloverConfigError::InvalidFloat("attention_dropout"));
        }
        if !self.moe_block_coreset_top_p.is_finite()
            || self.moe_block_coreset_top_p <= 0.0
            || self.moe_block_coreset_top_p > 1.0
        {
            return Err(CloverConfigError::InvalidFloat("moe_block_coreset_top_p"));
        }
        if !self.denoising_config.threshold.is_finite()
            || self.denoising_config.threshold <= 0.0
            || self.denoising_config.threshold > 1.0
        {
            return Err(CloverConfigError::InvalidFloat(
                "denoising_config.threshold",
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(CloverConfigError::LayerCountMismatch {
                declared: self.num_hidden_layers,
                actual: self.layer_types.len(),
            });
        }
        if self.num_experts_per_tok > self.num_experts {
            return Err(CloverConfigError::TopKExceedsExperts {
                top_k: self.num_experts_per_tok,
                experts: self.num_experts,
            });
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(CloverConfigError::InvalidGroupedQueryGeometry);
        }
        if self.architectures.len() != 1 || self.architectures[0] != "CloverMoeModelLM" {
            return Err(CloverConfigError::UnsupportedArchitecture);
        }
        if self.model_type != "clover_moe" {
            return Err(CloverConfigError::UnsupportedModelType);
        }
        if self.hidden_act != "silu" || self.denoising_config.algorithm != "low_confidence" {
            return Err(CloverConfigError::UnsupportedExecutionSemantics);
        }
        Ok(())
    }

    /// Require the exact architecture geometry audited for the initial
    /// Clover-35B-A3B-Preview integration.
    pub fn validate_pinned(&self) -> Result<(), CloverConfigError> {
        self.validate()?;
        let expected_layers: Vec<_> = (0..LAYER_COUNT)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();

        let matches = self.hidden_size == HIDDEN_SIZE
            && self.num_hidden_layers == LAYER_COUNT
            && self.layer_types == expected_layers
            && self.num_experts == EXPERT_COUNT
            && self.num_experts_per_tok == EXPERTS_PER_TOKEN
            && self.num_shared_experts == SHARED_EXPERTS
            && self.vocab_size == VOCAB_SIZE
            && self.max_position_embeddings == MAX_CONTEXT
            && self.block_size == DEFAULT_BLOCK_SIZE
            && self.full_attention_interval == 4
            && self.num_attention_heads == 16
            && self.num_key_value_heads == 2
            && self.head_dim == 256
            && self.linear_num_key_heads == 16
            && self.linear_num_value_heads == 32
            && self.linear_key_head_dim == 128
            && self.linear_value_head_dim == 128
            && self.linear_conv_kernel_dim == 4
            && self.dtype == "bfloat16"
            && self.mamba_ssm_dtype == "float32"
            && self.hidden_act == "silu"
            && self.attn_output_gate
            && self.moe_block_coreset_enabled
            && (self.moe_block_coreset_top_p - 0.9).abs() < f64::EPSILON
            && (self.partial_rotary_factor - 0.25).abs() < f64::EPSILON
            && (self.rope_parameters.rope_theta - 10_000_000.0).abs() < f64::EPSILON
            && self.denoising_config.algorithm == "low_confidence"
            && (self.denoising_config.threshold - 0.95).abs() < f64::EPSILON
            && !self.tie_word_embeddings
            && self.use_cache;

        if matches {
            Ok(())
        } else {
            Err(CloverConfigError::PinnedArchitectureMismatch)
        }
    }

    pub fn layer_type(&self, index: usize) -> Option<LayerType> {
        self.layer_types.get(index).copied()
    }

    pub fn is_full_attention_layer(&self, index: usize) -> bool {
        self.layer_type(index) == Some(LayerType::FullAttention)
    }

    pub fn gdn_layer_count(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerType::LinearAttention)
            .count()
    }

    pub fn full_attention_layer_count(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerType::FullAttention)
            .count()
    }
}

fn is_bf16(dtype: &str) -> bool {
    dtype.eq_ignore_ascii_case("bfloat16") || dtype.eq_ignore_ascii_case("bf16")
}

#[derive(Debug, Error)]
pub enum CloverConfigError {
    #[error("invalid Clover config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("configuration field {0} must be non-zero")]
    ZeroDimension(&'static str),
    #[error("unsupported Clover weight dtype {0}; explicit BF16 is required")]
    UnsupportedDType(String),
    #[error("configuration float field {0} is invalid")]
    InvalidFloat(&'static str),
    #[error("unsupported Clover RoPE configuration")]
    InvalidRope,
    #[error("configuration declares {declared} layers but contains {actual} layer types")]
    LayerCountMismatch { declared: usize, actual: usize },
    #[error("MoE top-k {top_k} exceeds expert count {experts}")]
    TopKExceedsExperts { top_k: usize, experts: usize },
    #[error("full-attention query heads are not divisible by KV heads")]
    InvalidGroupedQueryGeometry,
    #[error("unsupported model architecture")]
    UnsupportedArchitecture,
    #[error("unsupported model_type")]
    UnsupportedModelType,
    #[error("unsupported activation or denoising algorithm")]
    UnsupportedExecutionSemantics,
    #[error("configuration does not match the pinned Clover architecture")]
    PinnedArchitectureMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_CONFIG: &str = include_str!("../../../docs/audits/fixtures/clover-config.json");

    #[test]
    fn parses_and_validates_the_pinned_architecture() {
        let config = CloverConfig::from_json(PINNED_CONFIG).unwrap();
        config.validate_pinned().unwrap();
        assert_eq!(config.gdn_layer_count(), 30);
        assert_eq!(config.full_attention_layer_count(), 10);
        assert!(config.is_full_attention_layer(3));
        assert!(!config.is_full_attention_layer(2));
    }

    #[test]
    fn rejects_non_bf16_weights_and_invalid_moe_top_k() {
        let config = CloverConfig::from_json(PINNED_CONFIG).unwrap();
        let mut wrong_dtype = config.clone();
        wrong_dtype.dtype = "float16".to_owned();
        assert!(matches!(
            wrong_dtype.validate(),
            Err(CloverConfigError::UnsupportedDType(_))
        ));

        let mut wrong_top_k = config;
        wrong_top_k.num_experts_per_tok = 257;
        assert!(matches!(
            wrong_top_k.validate(),
            Err(CloverConfigError::TopKExceedsExperts { .. })
        ));
    }

    #[test]
    fn rejects_layer_pattern_and_geometry_drift() {
        let mut config = CloverConfig::from_json(PINNED_CONFIG).unwrap();
        config.layer_types.swap(0, 3);
        assert!(matches!(
            config.validate_pinned(),
            Err(CloverConfigError::PinnedArchitectureMismatch)
        ));

        let mut config = CloverConfig::from_json(PINNED_CONFIG).unwrap();
        config.num_attention_heads = 15;
        assert!(matches!(
            config.validate(),
            Err(CloverConfigError::InvalidGroupedQueryGeometry)
        ));
    }

    #[test]
    fn rejects_invalid_probability_and_rope_values() {
        let mut config = CloverConfig::from_json(PINNED_CONFIG).unwrap();
        config.moe_block_coreset_top_p = f64::NAN;
        assert!(matches!(
            config.validate(),
            Err(CloverConfigError::InvalidFloat("moe_block_coreset_top_p"))
        ));

        let mut config = CloverConfig::from_json(PINNED_CONFIG).unwrap();
        config.rope_parameters.rope_theta = 0.0;
        assert!(matches!(
            config.validate(),
            Err(CloverConfigError::InvalidRope)
        ));
    }
}
