//! Pinned generation policy configuration for DiffusionGemma.
//!
//! Two upstream descriptions exist in the checkpoint:
//! - `generation_config.json` (+ the Transformers generation loop) drives the
//!   text generation path audited for M10;
//! - `scheduler_config.json` describes a separate diffusers pipeline
//!   abstraction (`BlockRefinementScheduler`, 32 tokens/32 steps).
//!
//! M10 follows the Transformers `EntropyBoundSampler` path because it is the
//! policy attached to `DiffusionGemmaForBlockDiffusion.generate` and to the
//! pinned checkpoint's `generation_config.json`. The diffusers scheduler is
//! recorded for provenance and is deliberately not merged into this policy.

use serde::Deserialize;
use thiserror::Error;

use crate::config::TokenIdList;

/// Audited denoising step budget.
pub const PINNED_MAX_DENOISING_STEPS: u32 = 48;

/// Audited entropy bound (nats) for canvas acceptance.
pub const PINNED_ENTROPY_BOUND: f32 = 0.1;

/// Audited temperature at the first denoising step.
pub const PINNED_T_MAX: f32 = 0.8;

/// Audited temperature at the last denoising step.
pub const PINNED_T_MIN: f32 = 0.4;

/// Audited mean-entropy threshold for adaptive stopping.
pub const PINNED_CONFIDENCE_THRESHOLD: f32 = 0.005;

/// Audited number of identical argmax canvases required for stability.
pub const PINNED_STABILITY_THRESHOLD: u32 = 1;

/// Audited EOS ids used by the generation loop.
pub const PINNED_EOS_TOKEN_IDS: [u32; 3] = [1, 106, 50];

/// Audited maximum number of generated tokens.
pub const PINNED_MAX_NEW_TOKENS: usize = 256;

/// Audited padding id.
pub const PINNED_PAD_TOKEN_ID: u32 = 0;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SamplerConfig {
    #[serde(default, rename = "_cls_name")]
    pub class_name: Option<String>,
    #[serde(default)]
    pub entropy_bound: Option<f32>,
}

/// Parsed generation configuration. Validation is separate so synthetic tests
/// can use reduced schedules while real execution must pass the pin checks.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DiffusionGenerationConfig {
    #[serde(default)]
    pub max_denoising_steps: Option<u32>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdList>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub sampler_config: Option<SamplerConfig>,
    #[serde(default)]
    pub confidence_threshold: Option<f32>,
    #[serde(default)]
    pub stability_threshold: Option<u32>,
    #[serde(default)]
    pub t_max: Option<f32>,
    #[serde(default)]
    pub t_min: Option<f32>,
}

impl DiffusionGenerationConfig {
    pub fn from_json(json: &str) -> Result<Self, GenerationConfigError> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), GenerationConfigError> {
        if self.max_denoising_steps == Some(0) {
            return Err(GenerationConfigError::Zero("max_denoising_steps"));
        }
        if self.max_new_tokens == Some(0) {
            return Err(GenerationConfigError::Zero("max_new_tokens"));
        }
        if let Some(sampler) = &self.sampler_config {
            if let Some(bound) = sampler.entropy_bound {
                if !bound.is_finite() || bound <= 0.0 {
                    return Err(GenerationConfigError::InvalidEntropyBound(bound));
                }
            }
        }
        if let Some(threshold) = self.confidence_threshold {
            if !threshold.is_finite() || threshold <= 0.0 {
                return Err(GenerationConfigError::InvalidConfidenceThreshold(threshold));
            }
        }
        if let (Some(max), Some(min)) = (self.t_max, self.t_min) {
            if !max.is_finite() || !min.is_finite() || min <= 0.0 || max < min {
                return Err(GenerationConfigError::InvalidTemperatureSchedule { min, max });
            }
        }
        if let Some(ids) = &self.eos_token_id {
            if ids.ids().is_empty() {
                return Err(GenerationConfigError::EmptyEos);
            }
        }
        Ok(())
    }

    pub fn validate_pinned(&self) -> Result<(), GenerationConfigError> {
        let mut mismatch = None;
        let mut expect = |field: &'static str, actual: String, expected: String| {
            if actual != expected && mismatch.is_none() {
                mismatch = Some(GenerationConfigError::PinnedMismatch {
                    field,
                    expected,
                    actual,
                });
            }
        };
        expect(
            "max_denoising_steps",
            format!("{:?}", self.max_denoising_steps),
            format!("Some({PINNED_MAX_DENOISING_STEPS})"),
        );
        expect(
            "max_new_tokens",
            format!("{:?}", self.max_new_tokens),
            format!("Some({PINNED_MAX_NEW_TOKENS})"),
        );
        expect(
            "eos_token_id",
            format!("{:?}", self.eos_token_id.as_ref().map(TokenIdList::ids)),
            format!("Some({:?})", PINNED_EOS_TOKEN_IDS.to_vec()),
        );
        expect(
            "pad_token_id",
            format!("{:?}", self.pad_token_id),
            format!("Some({PINNED_PAD_TOKEN_ID})"),
        );
        expect(
            "confidence_threshold",
            format!("{:?}", self.confidence_threshold),
            format!("Some({PINNED_CONFIDENCE_THRESHOLD})"),
        );
        expect(
            "stability_threshold",
            format!("{:?}", self.stability_threshold),
            format!("Some({PINNED_STABILITY_THRESHOLD})"),
        );
        expect(
            "t_max",
            format!("{:?}", self.t_max),
            format!("Some({PINNED_T_MAX})"),
        );
        expect(
            "t_min",
            format!("{:?}", self.t_min),
            format!("Some({PINNED_T_MIN})"),
        );
        let sampler = self.sampler_config.as_ref();
        expect(
            "sampler_config._cls_name",
            format!(
                "{:?}",
                sampler.and_then(|config| config.class_name.as_deref())
            ),
            "Some(\"EntropyBoundSamplerConfig\")".to_owned(),
        );
        expect(
            "sampler_config.entropy_bound",
            format!("{:?}", sampler.and_then(|config| config.entropy_bound)),
            format!("Some({PINNED_ENTROPY_BOUND})"),
        );
        match mismatch {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[derive(Debug, Error)]
pub enum GenerationConfigError {
    #[error("invalid DiffusionGemma generation config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("generation configuration field {0} must be non-zero")]
    Zero(&'static str),
    #[error("entropy_bound {0} must be finite and positive")]
    InvalidEntropyBound(f32),
    #[error("confidence_threshold {0} must be finite and positive")]
    InvalidConfidenceThreshold(f32),
    #[error("invalid temperature schedule: t_min {min}, t_max {max}")]
    InvalidTemperatureSchedule { min: f32, max: f32 },
    #[error("eos_token_id must not be empty")]
    EmptyEos,
    #[error("generation configuration field {0} is required")]
    MissingField(&'static str),
    #[error("denoising step {step} is outside 1..={maximum}")]
    StepOutOfRange { step: u32, maximum: u32 },
    #[error(
        "pinned generation configuration mismatch at {field}: expected {expected}, got {actual}"
    )]
    PinnedMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
}

/// Runtime copy of the validated generation policy.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationPolicy {
    pub max_denoising_steps: u32,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: u32,
    pub entropy_bound: f32,
    pub confidence_threshold: f32,
    pub stability_threshold: u32,
    pub t_max: f32,
    pub t_min: f32,
}

impl GenerationPolicy {
    /// Build a policy with explicit values. Used by synthetic tests; the
    /// pinned checkpoint path flows through `from_config`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_denoising_steps: u32,
        max_new_tokens: usize,
        eos_token_ids: Vec<u32>,
        pad_token_id: u32,
        entropy_bound: f32,
        confidence_threshold: f32,
        stability_threshold: u32,
        t_max: f32,
        t_min: f32,
    ) -> Result<Self, GenerationConfigError> {
        let policy = Self {
            max_denoising_steps,
            max_new_tokens,
            eos_token_ids,
            pad_token_id,
            entropy_bound,
            confidence_threshold,
            stability_threshold,
            t_max,
            t_min,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn from_config(config: &DiffusionGenerationConfig) -> Result<Self, GenerationConfigError> {
        config.validate()?;
        let sampler = config
            .sampler_config
            .as_ref()
            .ok_or(GenerationConfigError::MissingField("sampler_config"))?;
        let policy = Self {
            max_denoising_steps: config
                .max_denoising_steps
                .ok_or(GenerationConfigError::MissingField("max_denoising_steps"))?,
            max_new_tokens: config
                .max_new_tokens
                .ok_or(GenerationConfigError::MissingField("max_new_tokens"))?,
            eos_token_ids: config
                .eos_token_id
                .as_ref()
                .ok_or(GenerationConfigError::MissingField("eos_token_id"))?
                .ids(),
            pad_token_id: config
                .pad_token_id
                .ok_or(GenerationConfigError::MissingField("pad_token_id"))?,
            entropy_bound: sampler
                .entropy_bound
                .ok_or(GenerationConfigError::MissingField(
                    "sampler_config.entropy_bound",
                ))?,
            confidence_threshold: config
                .confidence_threshold
                .ok_or(GenerationConfigError::MissingField("confidence_threshold"))?,
            stability_threshold: config
                .stability_threshold
                .ok_or(GenerationConfigError::MissingField("stability_threshold"))?,
            t_max: config
                .t_max
                .ok_or(GenerationConfigError::MissingField("t_max"))?,
            t_min: config
                .t_min
                .ok_or(GenerationConfigError::MissingField("t_min"))?,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn pinned() -> Self {
        Self {
            max_denoising_steps: PINNED_MAX_DENOISING_STEPS,
            max_new_tokens: PINNED_MAX_NEW_TOKENS,
            eos_token_ids: PINNED_EOS_TOKEN_IDS.to_vec(),
            pad_token_id: PINNED_PAD_TOKEN_ID,
            entropy_bound: PINNED_ENTROPY_BOUND,
            confidence_threshold: PINNED_CONFIDENCE_THRESHOLD,
            stability_threshold: PINNED_STABILITY_THRESHOLD,
            t_max: PINNED_T_MAX,
            t_min: PINNED_T_MIN,
        }
    }

    pub fn validate(&self) -> Result<(), GenerationConfigError> {
        if self.max_denoising_steps == 0 {
            return Err(GenerationConfigError::Zero("max_denoising_steps"));
        }
        if self.max_new_tokens == 0 {
            return Err(GenerationConfigError::Zero("max_new_tokens"));
        }
        if !self.entropy_bound.is_finite() || self.entropy_bound <= 0.0 {
            return Err(GenerationConfigError::InvalidEntropyBound(
                self.entropy_bound,
            ));
        }
        if !self.confidence_threshold.is_finite() || self.confidence_threshold <= 0.0 {
            return Err(GenerationConfigError::InvalidConfidenceThreshold(
                self.confidence_threshold,
            ));
        }
        if !self.t_max.is_finite()
            || !self.t_min.is_finite()
            || self.t_min <= 0.0
            || self.t_max < self.t_min
        {
            return Err(GenerationConfigError::InvalidTemperatureSchedule {
                min: self.t_min,
                max: self.t_max,
            });
        }
        if self.eos_token_ids.is_empty() {
            return Err(GenerationConfigError::EmptyEos);
        }
        Ok(())
    }

    /// Linear temperature schedule used by
    /// `LinearTemperatureScheduleLogitsProcessor`. `remaining_step` counts
    /// down from `max_denoising_steps` to 1, exactly as the reference loop.
    pub fn temperature(&self, remaining_step: u32) -> Result<f32, GenerationConfigError> {
        if remaining_step == 0 || remaining_step > self.max_denoising_steps {
            return Err(GenerationConfigError::StepOutOfRange {
                step: remaining_step,
                maximum: self.max_denoising_steps,
            });
        }
        let ratio = remaining_step as f32 / self.max_denoising_steps as f32;
        Ok(self.t_min + (self.t_max - self.t_min) * ratio)
    }

    pub fn canvas_count(&self) -> usize {
        self.max_new_tokens
            .div_ceil(crate::config::PINNED_CANVAS_LENGTH)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_GENERATION: &str =
        include_str!("../../../docs/audits/fixtures/diffusiongemma-generation-config.json");

    #[test]
    fn accepts_and_pins_the_audited_generation_config() {
        let config = DiffusionGenerationConfig::from_json(PINNED_GENERATION).unwrap();
        config.validate_pinned().unwrap();
        let policy = GenerationPolicy::from_config(&config).unwrap();
        assert_eq!(policy, GenerationPolicy::pinned());
        assert_eq!(policy.temperature(48).unwrap(), PINNED_T_MAX);
        assert_eq!(
            policy.temperature(1).unwrap(),
            PINNED_T_MIN + (PINNED_T_MAX - PINNED_T_MIN) / 48.0
        );
        assert!(policy.temperature(0).is_err());
        assert!(policy.temperature(49).is_err());
        assert_eq!(policy.canvas_count(), 1);
    }

    #[test]
    fn rejects_invalid_policies_and_pin_drift() {
        let mut config = DiffusionGenerationConfig::from_json(PINNED_GENERATION).unwrap();
        config.sampler_config.as_mut().unwrap().entropy_bound = Some(0.0);
        assert!(config.validate().is_err());

        let mut config = DiffusionGenerationConfig::from_json(PINNED_GENERATION).unwrap();
        config.t_max = Some(0.2);
        assert!(config.validate().is_err());

        let mut config = DiffusionGenerationConfig::from_json(PINNED_GENERATION).unwrap();
        config.stability_threshold = Some(2);
        assert!(matches!(
            config.validate_pinned(),
            Err(GenerationConfigError::PinnedMismatch { .. })
        ));

        assert!(GenerationPolicy::new(0, 256, vec![1], 0, 0.1, 0.005, 1, 0.8, 0.4,).is_err());
        assert!(GenerationPolicy::new(48, 256, vec![], 0, 0.1, 0.005, 1, 0.8, 0.4,).is_err());
    }

    #[test]
    fn missing_fields_are_not_silently_defaulted() {
        assert!(matches!(
            DiffusionGenerationConfig::from_json("{}")
                .and_then(|config| GenerationPolicy::from_config(&config)),
            Err(GenerationConfigError::MissingField(_))
        ));
    }
}
