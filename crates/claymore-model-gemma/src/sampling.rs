//! Faithful implementation of the pinned sampling policy:
//!
//! - linear temperature schedule (`LinearTemperatureScheduleLogitsProcessor`);
//! - entropy-bound canvas acceptance (`EntropyBoundSampler`);
//! - uniform re-noising of every non-accepted position;
//! - stable-and-confident adaptive stopping (`StableAndConfidentStoppingCriteria`).
//!
//! The policy is separated from model execution: it consumes processed logits
//! and returns token IDs/scalars only. It never touches tensors or devices.

use std::collections::VecDeque;

use thiserror::Error;

use crate::generation::GenerationPolicy;
use crate::rng::SplitMix64;

/// One step's acceptance decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Acceptance {
    pub mask: Vec<bool>,
    pub accepted_count: usize,
    pub renoised_count: usize,
}

/// Entropy-bound canvas sampler with the same acceptance rule as the pinned
/// `EntropyBoundSampler`.
#[derive(Debug, Clone)]
pub struct EntropyBoundSampler {
    entropy_bound: f32,
    canvas_length: usize,
    vocabulary: usize,
    accepted: Vec<bool>,
}

impl EntropyBoundSampler {
    pub fn new(
        entropy_bound: f32,
        canvas_length: usize,
        vocabulary: usize,
    ) -> Result<Self, SamplingError> {
        if canvas_length == 0 {
            return Err(SamplingError::EmptyCanvas);
        }
        if vocabulary == 0 {
            return Err(SamplingError::EmptyVocabulary);
        }
        if !entropy_bound.is_finite() || entropy_bound <= 0.0 {
            return Err(SamplingError::InvalidEntropyBound(entropy_bound));
        }
        Ok(Self {
            entropy_bound,
            canvas_length,
            vocabulary,
            accepted: vec![false; canvas_length],
        })
    }

    pub fn from_policy(policy: &GenerationPolicy, canvas_length: usize, vocabulary: usize) -> Self {
        Self::new(policy.entropy_bound, canvas_length, vocabulary)
            .expect("validated generation policy")
    }

    pub const fn entropy_bound(&self) -> f32 {
        self.entropy_bound
    }

    pub const fn canvas_length(&self) -> usize {
        self.canvas_length
    }

    pub const fn vocabulary(&self) -> usize {
        self.vocabulary
    }

    /// Uniform canvas initialization, equivalent to
    /// `torch.randint(0, vocab_size, (1, canvas_length))`.
    pub fn initialize_canvas(&self, rng: &mut SplitMix64) -> Vec<u32> {
        (0..self.canvas_length)
            .map(|_| rng.below(self.vocabulary as u32))
            .collect()
    }

    /// Reference acceptance rule: sort token entropies ascending and accept a
    /// prefix whose sum *excluding the newly accepted token* is bounded:
    ///
    /// `sum_{i<k} entropy_i <= entropy_bound`.
    ///
    /// The first (lowest-entropy) token is therefore always accepted.
    pub fn accept(&mut self, entropies: &[f32]) -> Result<Acceptance, SamplingError> {
        if entropies.len() != self.canvas_length {
            return Err(SamplingError::ShapeMismatch {
                expected: self.canvas_length,
                actual: entropies.len(),
            });
        }
        for (position, entropy) in entropies.iter().enumerate() {
            if !entropy.is_finite() || *entropy < 0.0 {
                return Err(SamplingError::InvalidEntropy {
                    position,
                    value: *entropy,
                });
            }
        }

        let mut order: Vec<usize> = (0..self.canvas_length).collect();
        // Stable, index-ordered tie-breaking keeps the policy deterministic.
        order.sort_by(|&left, &right| {
            entropies[left]
                .total_cmp(&entropies[right])
                .then_with(|| left.cmp(&right))
        });
        let mut cumulative = 0.0f32;
        let mut mask = vec![false; self.canvas_length];
        for &position in &order {
            if cumulative > self.entropy_bound {
                break;
            }
            mask[position] = true;
            cumulative += entropies[position];
        }
        let accepted_count = mask.iter().filter(|accepted| **accepted).count();
        self.accepted = mask.clone();
        Ok(Acceptance {
            mask,
            accepted_count,
            renoised_count: self.canvas_length - accepted_count,
        })
    }

    /// Apply the acceptance mask to the denoiser predictions.
    pub fn accept_canvas(
        &self,
        current: &[u32],
        denoiser: &[u32],
    ) -> Result<Vec<u32>, SamplingError> {
        if current.len() != self.canvas_length || denoiser.len() != self.canvas_length {
            return Err(SamplingError::ShapeMismatch {
                expected: self.canvas_length,
                actual: current.len().min(denoiser.len()),
            });
        }
        Ok(current
            .iter()
            .zip(denoiser)
            .zip(&self.accepted)
            .map(|((current, denoiser), accepted)| if *accepted { *denoiser } else { *current })
            .collect())
    }

    /// Replace every non-accepted position with a fresh uniform token. Accepted
    /// positions are preserved exactly.
    pub fn renoise(
        &self,
        accepted_canvas: &[u32],
        rng: &mut SplitMix64,
    ) -> Result<Vec<u32>, SamplingError> {
        if accepted_canvas.len() != self.canvas_length {
            return Err(SamplingError::ShapeMismatch {
                expected: self.canvas_length,
                actual: accepted_canvas.len(),
            });
        }
        if accepted_canvas
            .iter()
            .any(|token| *token as usize >= self.vocabulary)
        {
            return Err(SamplingError::TokenOutOfVocabulary);
        }
        Ok(accepted_canvas
            .iter()
            .zip(&self.accepted)
            .map(|(token, accepted)| {
                if *accepted {
                    *token
                } else {
                    rng.below(self.vocabulary as u32)
                }
            })
            .collect())
    }

    pub fn accepted_mask(&self) -> &[bool] {
        &self.accepted
    }
}

/// Stable-and-confident adaptive stopping, with the reference's off-by-one
/// semantics: the first call can never be stable because the history starts
/// unset, and `stability_threshold = 1` requires two identical argmax canvases.
#[derive(Debug, Clone)]
pub struct StableAndConfident {
    stability_threshold: u32,
    confidence_threshold: f32,
    history: VecDeque<Vec<u32>>,
}

impl StableAndConfident {
    pub fn new(stability_threshold: u32, confidence_threshold: f32) -> Result<Self, SamplingError> {
        if !confidence_threshold.is_finite() || confidence_threshold <= 0.0 {
            return Err(SamplingError::InvalidConfidenceThreshold(
                confidence_threshold,
            ));
        }
        Ok(Self {
            stability_threshold,
            confidence_threshold,
            history: VecDeque::new(),
        })
    }

    pub fn from_policy(policy: &GenerationPolicy) -> Self {
        Self::new(policy.stability_threshold, policy.confidence_threshold)
            .expect("validated generation policy")
    }

    pub fn evaluate(
        &mut self,
        argmax_canvas: &[u32],
        mean_entropy: f32,
    ) -> Result<bool, SamplingError> {
        if !mean_entropy.is_finite() || mean_entropy < 0.0 {
            return Err(SamplingError::InvalidEntropy {
                position: 0,
                value: mean_entropy,
            });
        }
        let stable = if self.stability_threshold == 0 {
            true
        } else {
            self.history.len() == self.stability_threshold as usize
                && self
                    .history
                    .iter()
                    .all(|canvas| canvas.as_slice() == argmax_canvas)
        };
        if self.stability_threshold > 0 {
            if self.history.len() == self.stability_threshold as usize {
                self.history.pop_front();
            }
            self.history.push_back(argmax_canvas.to_vec());
        }
        let confident = mean_entropy < self.confidence_threshold;
        Ok(stable && confident)
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    pub fn stability_threshold(&self) -> u32 {
        self.stability_threshold
    }

    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}

/// Apply the pinned linear temperature schedule. `remaining_step` counts down
/// from `max_denoising_steps` (first step) to 1 (last step).
pub fn apply_temperature(
    logits: &[f32],
    vocabulary: usize,
    temperature: f32,
) -> Result<Vec<f32>, SamplingError> {
    if vocabulary == 0 || logits.is_empty() || logits.len() % vocabulary != 0 {
        return Err(SamplingError::InvalidLogitShape {
            length: logits.len(),
            vocabulary,
        });
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature(temperature));
    }
    let mut scaled = Vec::with_capacity(logits.len());
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(SamplingError::NonFiniteLogit { index, value });
        }
        scaled.push(value / temperature);
    }
    Ok(scaled)
}

/// Token entropies in nats for each canvas row of `[canvas, vocabulary]` logits.
pub fn token_entropies(
    logits: &[f32],
    vocabulary: usize,
    canvas_length: usize,
) -> Result<Vec<f32>, SamplingError> {
    if vocabulary == 0 || logits.len() != vocabulary * canvas_length {
        return Err(SamplingError::InvalidLogitShape {
            length: logits.len(),
            vocabulary,
        });
    }
    let mut entropies = Vec::with_capacity(canvas_length);
    for (row, values) in logits.chunks_exact(vocabulary).enumerate() {
        let mut max = f32::NEG_INFINITY;
        for (column, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SamplingError::NonFiniteLogit {
                    index: row * vocabulary + column,
                    value,
                });
            }
            max = max.max(value);
        }
        if max == f32::NEG_INFINITY {
            return Err(SamplingError::NonFiniteLogit {
                index: row * vocabulary,
                value: max,
            });
        }
        let mut sum = 0.0f32;
        for value in values {
            sum += (*value - max).exp();
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SamplingError::NonFiniteLogit {
                index: row * vocabulary,
                value: sum,
            });
        }
        let log_sum = sum.ln();
        let mut entropy = 0.0f32;
        for value in values {
            let probability = (*value - max).exp() / sum;
            if probability > 0.0 {
                entropy -= probability * (*value - max - log_sum);
            }
        }
        if !entropy.is_finite() || entropy < 0.0 {
            return Err(SamplingError::InvalidEntropy {
                position: row,
                value: entropy,
            });
        }
        entropies.push(entropy);
    }
    Ok(entropies)
}

/// First-index argmax for every row (PyTorch `argmax` tie behaviour).
pub fn argmax_rows(
    logits: &[f32],
    vocabulary: usize,
    canvas_length: usize,
) -> Result<Vec<u32>, SamplingError> {
    if vocabulary == 0 || logits.len() != vocabulary * canvas_length {
        return Err(SamplingError::InvalidLogitShape {
            length: logits.len(),
            vocabulary,
        });
    }
    if vocabulary > u32::MAX as usize {
        return Err(SamplingError::VocabularyTooLarge(vocabulary));
    }
    let mut output = Vec::with_capacity(canvas_length);
    for values in logits.chunks_exact(vocabulary) {
        let mut best = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        for (index, value) in values.iter().copied().enumerate() {
            if value.is_nan() {
                return Err(SamplingError::NonFiniteLogit { index, value });
            }
            if value > best {
                best = value;
                best_index = index as u32;
            }
        }
        output.push(best_index);
    }
    Ok(output)
}

/// Sample one token per row from `softmax(logits)` without replacement count 1.
/// The distribution is equivalent to the reference; the underlying RNG stream
/// is Claymore's deterministic SplitMix64, not Torch's global generator.
pub fn sample_rows(
    logits: &[f32],
    vocabulary: usize,
    canvas_length: usize,
    rng: &mut SplitMix64,
) -> Result<Vec<u32>, SamplingError> {
    let entropies = token_entropies(logits, vocabulary, canvas_length)?;
    let _ = entropies;
    if vocabulary > u32::MAX as usize {
        return Err(SamplingError::VocabularyTooLarge(vocabulary));
    }
    let mut output = Vec::with_capacity(canvas_length);
    for values in logits.chunks_exact(vocabulary) {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probabilities = Vec::with_capacity(vocabulary);
        for value in values {
            let probability = (*value - max).exp();
            sum += probability;
            probabilities.push(probability);
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SamplingError::NonFiniteLogit {
                index: 0,
                value: sum,
            });
        }
        let draw = rng.next_f64();
        let mut cumulative = 0.0f64;
        let mut chosen = None;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += f64::from(*probability / sum);
            if draw < cumulative {
                chosen = Some(index as u32);
                break;
            }
        }
        output.push(chosen.unwrap_or_else(|| {
            probabilities
                .iter()
                .rposition(|probability| *probability > 0.0)
                .unwrap_or(0) as u32
        }));
    }
    Ok(output)
}

/// Mean entropy over canvas positions, used by the stopping criterion.
pub fn mean_entropy(entropies: &[f32]) -> Result<f32, SamplingError> {
    if entropies.is_empty() {
        return Err(SamplingError::EmptyCanvas);
    }
    let mut sum = 0.0f32;
    for (position, entropy) in entropies.iter().copied().enumerate() {
        if !entropy.is_finite() || entropy < 0.0 {
            return Err(SamplingError::InvalidEntropy {
                position,
                value: entropy,
            });
        }
        sum += entropy;
    }
    Ok(sum / entropies.len() as f32)
}

#[derive(Debug, Error, PartialEq)]
pub enum SamplingError {
    #[error("canvas must contain at least one position")]
    EmptyCanvas,
    #[error("vocabulary must be non-zero")]
    EmptyVocabulary,
    #[error("vocabulary {0} does not fit in a token id")]
    VocabularyTooLarge(usize),
    #[error("entropy bound {0} must be finite and positive")]
    InvalidEntropyBound(f32),
    #[error("confidence threshold {0} must be finite and positive")]
    InvalidConfidenceThreshold(f32),
    #[error("invalid temperature {0}")]
    InvalidTemperature(f32),
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },
    #[error("logit buffer of {length} elements is not a [canvas, {vocabulary}] matrix")]
    InvalidLogitShape { length: usize, vocabulary: usize },
    #[error("non-finite logit at index {index}: {value}")]
    NonFiniteLogit { index: usize, value: f32 },
    #[error("invalid entropy at position {position}: {value}")]
    InvalidEntropy { position: usize, value: f32 },
    #[error("canvas token is outside the vocabulary")]
    TokenOutOfVocabulary,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::GenerationPolicy;

    fn policy() -> GenerationPolicy {
        GenerationPolicy::new(8, 256, vec![1, 106, 50], 0, 0.1, 0.005, 1, 0.8, 0.4).unwrap()
    }

    fn peaked_logits(vocabulary: usize, length: usize, peak: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocabulary * length];
        for row in 0..length {
            logits[row * vocabulary + (peak + row) % vocabulary] = 20.0;
        }
        logits
    }

    #[test]
    fn entropy_of_uniform_and_peaked_rows_matches_definition() {
        let uniform = vec![0.0f32; 4];
        let entropies = token_entropies(&uniform, 4, 1).unwrap();
        assert!((entropies[0] - 4.0f32.ln()).abs() < 1e-6);

        let one_hot = [1000.0, -1000.0, -1000.0, -1000.0];
        let entropies = token_entropies(&one_hot, 4, 1).unwrap();
        assert!(entropies[0] < 1e-6);
        assert_eq!(mean_entropy(&entropies).unwrap(), entropies[0]);
    }

    #[test]
    fn entropy_rejects_nan_infinite_and_empty_inputs() {
        assert!(matches!(
            token_entropies(&[f32::NAN], 1, 1),
            Err(SamplingError::NonFiniteLogit { .. })
        ));
        assert!(matches!(
            token_entropies(&[f32::INFINITY], 1, 1),
            Err(SamplingError::NonFiniteLogit { .. })
        ));
        assert!(matches!(
            token_entropies(&[], 1, 1),
            Err(SamplingError::InvalidLogitShape { .. })
        ));
        assert!(matches!(mean_entropy(&[]), Err(SamplingError::EmptyCanvas)));
    }

    #[test]
    fn acceptance_is_the_reference_entropy_bound_prefix() {
        let mut sampler = EntropyBoundSampler::new(0.05, 5, 100).unwrap();
        let entropies = [0.01f32, 0.2, 0.02, 0.03, 0.5];
        let acceptance = sampler.accept(&entropies).unwrap();
        // Sorted: 0.01, 0.02, 0.03, 0.2, 0.5.
        // previous sums: 0, 0.01, 0.03, 0.06, 0.26.
        assert_eq!(acceptance.mask, [true, false, true, true, false]);
        assert_eq!(acceptance.accepted_count, 3);
        assert_eq!(acceptance.renoised_count, 2);

        let current = [1, 2, 3, 4, 5];
        let denoiser = [11, 12, 13, 14, 15];
        let accepted = sampler.accept_canvas(&current, &denoiser).unwrap();
        assert_eq!(accepted, [11, 2, 13, 14, 5]);
        let mut rng = SplitMix64::new(9);
        let renoised = sampler.renoise(&accepted, &mut rng).unwrap();
        assert_eq!(renoised[0], 11);
        assert_eq!(renoised[2], 13);
        assert_eq!(renoised[3], 14);
        assert!(renoised[1] < 100);
        assert!(renoised[4] < 100);
    }

    #[test]
    fn acceptance_always_accepts_the_lowest_entropy_token() {
        let mut sampler = EntropyBoundSampler::new(1e-9, 3, 10).unwrap();
        let acceptance = sampler.accept(&[5.0, 0.001, 4.0]).unwrap();
        assert!(acceptance.mask[1]);
        assert_eq!(acceptance.accepted_count, 1);
    }

    #[test]
    fn acceptance_rejects_invalid_entropy_and_shapes() {
        let mut sampler = EntropyBoundSampler::new(0.1, 2, 10).unwrap();
        assert!(matches!(
            sampler.accept(&[0.0]),
            Err(SamplingError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            sampler.accept(&[0.0, f32::NAN]),
            Err(SamplingError::InvalidEntropy { .. })
        ));
        assert!(matches!(
            sampler.accept(&[-0.1, 0.0]),
            Err(SamplingError::InvalidEntropy { .. })
        ));
        assert!(matches!(
            sampler.renoise(&[20], &mut SplitMix64::new(1)),
            Err(SamplingError::ShapeMismatch { .. }) | Err(SamplingError::TokenOutOfVocabulary)
        ));
        let mut single = EntropyBoundSampler::new(0.1, 1, 10).unwrap();
        single.accept(&[0.0]).unwrap();
        assert!(matches!(
            single.renoise(&[20], &mut SplitMix64::new(1)),
            Err(SamplingError::TokenOutOfVocabulary)
        ));
    }

    #[test]
    fn renoise_only_replaces_non_accepted_positions() {
        let mut sampler = EntropyBoundSampler::new(10.0, 3, 50).unwrap();
        sampler.accept(&[0.0, 0.0, 0.0]).unwrap();
        let original = [7, 8, 9];
        let mut rng = SplitMix64::new(3);
        let renoised = sampler.renoise(&original, &mut rng).unwrap();
        assert_eq!(renoised, original);

        let mut sampler = EntropyBoundSampler::new(1e-9, 3, 50).unwrap();
        sampler.accept(&[1.0, 0.0, 2.0]).unwrap();
        let mut rng = SplitMix64::new(3);
        let renoised = sampler.renoise(&original, &mut rng).unwrap();
        assert_eq!(renoised[1], original[1]);
        assert!(renoised[0] < 50);
        assert!(renoised[2] < 50);

        // The same seed replays the same re-noising stream.
        let mut replay = SplitMix64::new(3);
        assert_eq!(renoised, sampler.renoise(&original, &mut replay).unwrap());
    }

    #[test]
    fn temperature_scaling_is_applied_before_sampling() {
        let scaled = apply_temperature(&[0.8, 1.6], 2, 0.8).unwrap();
        assert_eq!(scaled, [1.0, 2.0]);
        assert!(matches!(
            apply_temperature(&[0.0], 1, 0.0),
            Err(SamplingError::InvalidTemperature(_))
        ));
        assert!(matches!(
            apply_temperature(&[f32::NAN], 1, 0.8),
            Err(SamplingError::NonFiniteLogit { .. })
        ));
        assert!(matches!(
            apply_temperature(&[0.0], 2, 0.8),
            Err(SamplingError::InvalidLogitShape { .. })
        ));
    }

    #[test]
    fn argmax_uses_first_index_on_ties_and_rejects_nan() {
        assert_eq!(argmax_rows(&[1.0, 3.0, 3.0], 3, 1).unwrap(), [1]);
        assert!(matches!(
            argmax_rows(&[1.0, f32::NAN], 2, 1),
            Err(SamplingError::NonFiniteLogit { .. })
        ));
    }

    #[test]
    fn sampling_is_deterministic_for_a_fixed_seed() {
        let logits = peaked_logits(16, 4, 3);
        let mut first = SplitMix64::new(0xDEAD_BEEF);
        let mut second = SplitMix64::new(0xDEAD_BEEF);
        let a = sample_rows(&logits, 16, 4, &mut first).unwrap();
        let b = sample_rows(&logits, 16, 4, &mut second).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, [3, 4, 5, 6]);
    }

    #[test]
    fn stability_requires_the_reference_history_window() {
        let mut stop = StableAndConfident::new(1, 0.005).unwrap();
        assert!(!stop.evaluate(&[1, 2], 0.0).unwrap());
        assert!(stop.evaluate(&[1, 2], 0.0).unwrap());
        assert!(!stop.evaluate(&[1, 3], 0.0).unwrap());
        assert!(stop.evaluate(&[1, 3], 0.0).unwrap());
        assert!(!stop.evaluate(&[1, 3], 0.005).unwrap());
        stop.reset();
        assert!(!stop.evaluate(&[1, 3], 0.0).unwrap());

        let mut stop = StableAndConfident::new(2, 0.005).unwrap();
        assert!(!stop.evaluate(&[4], 0.0).unwrap());
        assert!(!stop.evaluate(&[4], 0.0).unwrap());
        assert!(stop.evaluate(&[4], 0.0).unwrap());
        assert!(!stop.evaluate(&[5], 0.0).unwrap());

        let mut always = StableAndConfident::new(0, 0.005).unwrap();
        assert!(always.evaluate(&[1], 0.0).unwrap());
        assert!(always.evaluate(&[2], 0.004).unwrap());
        assert!(!always.evaluate(&[2], 0.005).unwrap());
    }

    #[test]
    fn policy_builders_produce_the_pinned_sampler_values() {
        let policy = policy();
        let sampler = EntropyBoundSampler::from_policy(&policy, 4, 100);
        assert_eq!(sampler.entropy_bound(), 0.1);
        let stop = StableAndConfident::from_policy(&policy);
        assert_eq!(stop.stability_threshold(), 1);
        assert_eq!(stop.confidence_threshold(), 0.005);
    }
}
