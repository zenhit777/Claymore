//! Deterministic SplitMix64 randomness for canvas initialization and
//! re-noising. The reference uses `torch.randint`/`torch.multinomial`; exact
//! stream identity is impossible across RNGs, so Claymore records its seed and
//! exposes deterministic, reproducible draws for tests and audits.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound` with rejection sampling (no modulo bias).
    pub fn below(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "bound must be positive");
        let bound64 = u64::from(bound);
        let zone = u64::MAX - (u64::MAX % bound64);
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % bound64) as u32;
            }
        }
    }

    /// Uniform f64 in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_and_bounded() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            let value = a.below(262_144);
            assert_eq!(value, b.below(262_144));
            assert!(value < 262_144);
        }
        assert_eq!(a.next_f64(), b.next_f64());
        let value = a.next_f64();
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn known_sequence_is_pinned() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn single_value_bound_is_zero() {
        let mut rng = SplitMix64::new(7);
        assert_eq!(rng.below(1), 0);
    }
}
