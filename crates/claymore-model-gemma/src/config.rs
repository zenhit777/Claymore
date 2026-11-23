//! Pinned-model configuration parsing and validation for DiffusionGemma.

use serde::Deserialize;
use thiserror::Error;

/// Exact supported Hugging Face checkpoint identifier.
pub const MODEL_ID: &str = "google/diffusiongemma-26B-A4B-it";

/// Parsed subset of the upstream configuration. Unknown fields are preserved
/// by serde's default forward-compatible behavior; model execution must still
/// validate the full pinned revision before loading any weights.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GemmaModelConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
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
        if let Some(dtype) = self.torch_dtype.as_deref() {
            if !dtype.eq_ignore_ascii_case("bfloat16") && !dtype.eq_ignore_ascii_case("bf16") {
                return Err(GemmaConfigError::UnsupportedWeightDType(dtype.to_owned()));
            }
        }

        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("vocab_size", self.vocab_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
        ] {
            if value == Some(0) {
                return Err(GemmaConfigError::ZeroDimension(name));
            }
        }
        Ok(())
    }

    pub fn requires_bf16(&self) -> bool {
        self.torch_dtype.as_deref().is_some_and(|dtype| {
            dtype.eq_ignore_ascii_case("bfloat16") || dtype.eq_ignore_ascii_case("bf16")
        })
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
