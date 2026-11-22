//! Explicit IEEE-754 bfloat16 (BF16) scalar representation.
//!
//! # Storage format
//!
//! BF16 is a *binary16 storage / binary32 exponent* format: 1 sign bit,
//! 8 exponent bits and 7 explicit significand bits (8 bits of precision with
//! the implicit leading one). It is stored as the exact 16-bit pattern in a
//! `u16`. It is not the same format as FP16 (IEEE-754 binary16, 5 exponent
//! bits), and this crate never treats the two as interchangeable.
//!
//! # Equality, ordering and hashing
//!
//! `Bf16` implements `PartialEq`/`Eq` by comparing the **raw 16-bit pattern**,
//! *not* the numeric value. As a consequence `+0` (`0x0000`) and `-0` (`0x8000`)
//! are **not** equal, and two distinct NaN payloads are **not** equal even
//! though both widen to a NaN. This is deliberate: a `Bf16` is a bit container
//! whose representation must survive CPU/CUDA byte transfers unchanged, and
//! bitwise equality is the only notion consistent with that contract. There
//! is intentionally **no** `PartialOrd`, `Ord` or `Hash` implementation:
//! bitwise ordering would place every negative pattern above every positive
//! one, which would be misleading. Code that needs numeric ordering must map
//! the sign-magnitude pattern to a monotonic integer key (flip the sign bit,
//! or complement the negative half) or widen to FP32.
//!
//! # Conversion rules (documented and tested)
//!
//! * `from_bits` / `to_bits` are exact bitwise views. Because BF16 uses the
//!   same exponent width as FP32, shifting the 16-bit pattern into the high
//!   half of an FP32 word reproduces normal, subnormal, zero and infinite
//!   values exactly.
//! * `from_f32` rounds to nearest, ties to even (`RNE`). The low 16 FP32 bits
//!   are discarded: bit 15 is the rounding bit and bits 14..0 form the sticky
//!   bit. Carrying out of the significand increments the exponent, so overflow
//!   to infinity and subnormal-to-normal transitions follow IEEE-754 rules.
//! * All NaNs are canonicalized to the quiet BF16 pattern `0x7FC0` on
//!   `from_f32`; `to_f32` quiets any signaling BF16 NaN (`0x00400000` in the
//!   FP32 significand) so producing a value never raises a spurious FP
//!   exception in the reference implementation.
//! * Signed zero is preserved in both directions.
//!
//! Arithmetic is deliberately *not* implemented on `Bf16` itself. Operations
//! convert to FP32, accumulate/evaluate in FP32 and round the final result
//! back to BF16 with `from_f32`. This keeps the numerical policy explicit and
//! identical between the CPU reference and the CUDA kernels.

use claymore_device::DType;

/// A single BF16 value stored as its exact 16-bit pattern.
///
/// Equality (`PartialEq`/`Eq`) compares that raw pattern bitwise, not the
/// numeric value; see the module documentation for the full contract.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Bf16(u16);

impl Bf16 {
    /// Canonical quiet NaN (exponent all ones, quiet bit set).
    pub const QUIET_NAN_BITS: u16 = 0x7FC0;
    /// Positive infinity.
    pub const INFINITY_BITS: u16 = 0x7F80;
    /// Negative infinity.
    pub const NEG_INFINITY_BITS: u16 = 0xFF80;
    /// Positive zero.
    pub const ZERO_BITS: u16 = 0x0000;
    /// Negative zero.
    pub const NEG_ZERO_BITS: u16 = 0x8000;
    /// Smallest positive subnormal (2^-133).
    pub const MIN_SUBNORMAL_BITS: u16 = 0x0001;
    /// Smallest positive normal (2^-126).
    pub const MIN_NORMAL_BITS: u16 = 0x0080;
    /// Largest finite value.
    pub const MAX_FINITE_BITS: u16 = 0x7F7F;

    /// Exact bitwise construction. No validation or conversion is performed.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Exact bitwise access.
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// The dtype metadata matching this scalar representation.
    pub const fn dtype() -> DType {
        DType::BF16
    }

    /// Native-endian byte representation (two bytes, matching
    /// `DType::BF16::element_size_bytes()`).
    pub const fn to_ne_bytes(self) -> [u8; 2] {
        self.0.to_ne_bytes()
    }

    /// Construct from two native-endian bytes.
    pub const fn from_ne_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_ne_bytes(bytes))
    }

    /// Convert from FP32 using round-to-nearest-even.
    pub fn from_f32(value: f32) -> Self {
        if value.is_nan() {
            return Self(Self::QUIET_NAN_BITS);
        }
        let bits = value.to_bits();
        let truncated = (bits >> 16) as u16;
        let round_bit = (bits >> 15) & 1 == 1;
        let sticky = bits & 0x7FFF != 0;
        let least_significant = truncated & 1 == 1;
        if round_bit && (sticky || least_significant) {
            Self(truncated.wrapping_add(1))
        } else {
            Self(truncated)
        }
    }

    /// Convert to FP32 exactly. BF16 NaN payloads are quieted.
    pub fn to_f32(self) -> f32 {
        let bits = (self.0 as u32) << 16;
        let value = f32::from_bits(bits);
        if value.is_nan() {
            f32::from_bits(bits | 0x0040_0000)
        } else {
            value
        }
    }

    pub fn is_nan(self) -> bool {
        self.to_f32().is_nan()
    }

    pub fn is_infinite(self) -> bool {
        self.0 & 0x7F80 == 0x7F80 && self.0 & 0x007F == 0
    }

    pub fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    pub fn is_zero(self) -> bool {
        self.0 & 0x7FFF == 0
    }

    pub fn is_sign_negative(self) -> bool {
        self.0 & 0x8000 != 0
    }
}

impl std::fmt::Debug for Bf16 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("Bf16")
            .field(&format_args!("{:#06x}", self.0))
            .field(&self.to_f32())
            .finish()
    }
}

impl std::fmt::Display for Bf16 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_f32().fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact power of two constructed from the IEEE-754 bit pattern.
    /// `f32::powi` underflows for exponents below -126, so it cannot be used
    /// to build the FP32 subnormals needed to test BF16 subnormal rounding.
    fn pow2(exponent: i32) -> f32 {
        assert!((-149..=127).contains(&exponent));
        if exponent >= -126 {
            f32::from_bits(((exponent + 127) as u32) << 23)
        } else {
            f32::from_bits(1u32 << (exponent + 149))
        }
    }

    #[test]
    fn bitwise_roundtrip_is_exact_for_every_16_bit_pattern() {
        for bits in 0..=u16::MAX {
            let value = Bf16::from_bits(bits);
            assert_eq!(value.to_bits(), bits);
            assert_eq!(Bf16::from_ne_bytes(value.to_ne_bytes()), value);
        }
    }

    #[test]
    fn exact_values_convert_in_both_directions() {
        for (value, bits) in [
            (0.0f32, 0x0000u16),
            (-0.0f32, 0x8000),
            (1.0, 0x3F80),
            (-1.0, 0xBF80),
            (2.0, 0x4000),
            (0.5, 0x3F00),
            (256.0, 0x4380),
            (f32::INFINITY, 0x7F80),
            (f32::NEG_INFINITY, 0xFF80),
        ] {
            assert_eq!(Bf16::from_f32(value).to_bits(), bits, "from {value}");
        }
        assert_eq!(Bf16::from_bits(0x3F80).to_f32(), 1.0);
        assert_eq!(Bf16::from_bits(0xBF80).to_f32(), -1.0);
        assert_eq!(Bf16::from_bits(0x0001).to_f32(), pow2(-133));
        assert_eq!(Bf16::from_bits(0x0080).to_f32(), pow2(-126));
        assert_eq!(Bf16::from_bits(0x007F).to_f32(), 127.0 * pow2(-133));
    }

    #[test]
    fn round_to_nearest_even_ties_and_neighbours() {
        // BF16 ULP at 1.0 is 2^-7; half an ULP is 2^-8.
        let half_ulp = pow2(-8);
        let tie = 1.0 + half_ulp;
        let below_tie = f32::from_bits(tie.to_bits() - 1);
        let above_tie = f32::from_bits(tie.to_bits() + 1);
        let one = Bf16::from_bits(0x3F80);
        assert_eq!(Bf16::from_f32(tie), one, "tie rounds to even");
        assert_eq!(Bf16::from_f32(below_tie), one, "below the tie rounds down");
        assert_eq!(
            Bf16::from_f32(above_tie).to_bits(),
            0x3F81,
            "above the tie rounds up"
        );
        // 1.0 + 3*half_ulp is the tie between mantissa 1 (odd) and 2 (even).
        assert_eq!(
            Bf16::from_f32(1.0 + 3.0 * half_ulp).to_bits(),
            0x3F82,
            "tie rounds to the even significand"
        );
    }

    #[test]
    fn special_values_and_limits_are_defined() {
        assert_eq!(Bf16::from_f32(f32::NAN).to_bits(), Bf16::QUIET_NAN_BITS);
        assert_eq!(Bf16::from_f32(-f32::NAN).to_bits(), Bf16::QUIET_NAN_BITS);
        assert!(Bf16::from_bits(Bf16::QUIET_NAN_BITS).is_nan());
        // A signaling BF16 NaN is quieted on the way to FP32.
        assert!(Bf16::from_bits(0x7F81).to_f32().is_nan());
        assert_eq!(Bf16::from_f32(f32::INFINITY).to_bits(), Bf16::INFINITY_BITS);
        assert_eq!(
            Bf16::from_f32(f32::NEG_INFINITY).to_bits(),
            Bf16::NEG_INFINITY_BITS
        );
        assert_eq!(Bf16::from_bits(Bf16::INFINITY_BITS).to_f32(), f32::INFINITY);
        assert_eq!(
            Bf16::from_bits(Bf16::NEG_INFINITY_BITS).to_f32(),
            f32::NEG_INFINITY
        );
        assert!(Bf16::from_f32(0.0).is_zero());
        assert!(!Bf16::from_f32(0.0).is_sign_negative());
        assert!(Bf16::from_f32(-0.0).is_sign_negative());
        assert_eq!(Bf16::from_f32(-0.0).to_bits(), Bf16::NEG_ZERO_BITS);
        assert_eq!(Bf16::from_f32(f32::MAX).to_bits(), Bf16::INFINITY_BITS);
        assert_eq!(Bf16::from_f32(-f32::MAX).to_bits(), Bf16::NEG_INFINITY_BITS);
    }

    #[test]
    fn subnormal_rounding_and_halfway_to_zero() {
        assert_eq!(
            Bf16::from_f32(pow2(-133)).to_bits(),
            Bf16::MIN_SUBNORMAL_BITS
        );
        // Half of the smallest BF16 subnormal rounds to even (zero).
        assert_eq!(Bf16::from_f32(pow2(-134)).to_bits(), Bf16::ZERO_BITS);
        // 1.5 ULP above zero is a tie between subnormal 1 (odd) and 2 (even).
        assert_eq!(Bf16::from_f32(3.0 * pow2(-134)).to_bits(), 0x0002);
        // Just below that tie rounds down to subnormal 1.
        assert_eq!(
            Bf16::from_f32(f32::from_bits(0x0001_7FFF)).to_bits(),
            Bf16::MIN_SUBNORMAL_BITS
        );
        // Largest FP32 subnormal rounds up to the smallest normal BF16.
        assert_eq!(
            Bf16::from_f32(f32::from_bits(0x007F_FFFF)).to_bits(),
            Bf16::MIN_NORMAL_BITS
        );
        // Smallest normal FP32 is exactly the smallest normal BF16.
        assert_eq!(
            Bf16::from_f32(f32::MIN_POSITIVE).to_bits(),
            Bf16::MIN_NORMAL_BITS
        );
    }

    #[test]
    fn every_bf16_value_roundtrips_through_fp32() {
        for bits in 0..=u16::MAX {
            let value = Bf16::from_bits(bits);
            let roundtrip = Bf16::from_f32(value.to_f32());
            if value.is_nan() {
                assert!(roundtrip.is_nan());
            } else {
                assert_eq!(roundtrip, value, "bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn fp32_grid_rounds_monotonically() {
        // A monotonic input sequence must produce a non-decreasing BF16
        // sequence (ordering covers NaNs only by their bit pattern here).
        let mut previous = 0u16;
        for step in 0..20_000u32 {
            let value = f32::from_bits(step * 20_000);
            let bits = Bf16::from_f32(value).to_bits();
            assert!(
                bits >= previous || bits & 0x7FFF == 0x7F80,
                "{bits:#x} < {previous:#x}"
            );
            previous = bits;
        }
    }

    #[test]
    fn widening_is_exactly_sign_symmetric_for_every_pattern() {
        // Widening is a pure bit operation (shift + quiet OR). The quiet bit
        // does not overlap the sign bit, so flipping the BF16 sign bit flips
        // exactly the FP32 sign bit for *all* 65,536 patterns, including NaNs
        // (their payloads are preserved by the shift).
        for bits in 0..=u16::MAX {
            let positive = Bf16::from_bits(bits).to_f32().to_bits();
            let negative = Bf16::from_bits(bits ^ 0x8000).to_f32().to_bits();
            assert_eq!(negative, positive ^ 0x8000_0000, "bits {bits:#06x}");
        }
    }

    #[test]
    fn nan_widening_quiets_and_from_f32_canonicalizes_for_every_pattern() {
        for bits in 0..=u16::MAX {
            let value = Bf16::from_bits(bits);
            if value.is_nan() {
                // The quiet bit is set; sign and NaN payload are otherwise
                // preserved verbatim by the widening shift.
                assert_eq!(
                    value.to_f32().to_bits(),
                    ((bits as u32) << 16) | 0x0040_0000,
                    "bits {bits:#06x}"
                );
                // Every NaN pattern (any sign/payload) collapses to +qNaN.
                assert_eq!(
                    Bf16::from_f32(value.to_f32()).to_bits(),
                    Bf16::QUIET_NAN_BITS,
                    "bits {bits:#06x}"
                );
            } else {
                assert!(!value.to_f32().is_nan(), "bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn widening_is_monotonic_in_magnitude_for_all_non_nan_patterns() {
        // Widening of a positive pattern is exactly (bits << 16), which is
        // monotonic in the FP32 bit pattern. The negative half mirrors it.
        // NaNs are excluded because their payloads are not ordered.
        let mut previous = 0u32;
        for magnitude in 0..=0x7F7Fu16 {
            let widened = Bf16::from_bits(magnitude).to_f32().to_bits();
            assert!(widened >= previous, "non-monotonic at {magnitude:#06x}");
            previous = widened;

            let negative = Bf16::from_bits(magnitude | 0x8000).to_f32().to_bits();
            assert_eq!(
                negative & 0x7FFF_FFFF,
                widened,
                "magnitude mismatch at {magnitude:#06x}"
            );
        }
        // Infinities bound the finite range from both sides.
        assert_eq!(
            Bf16::from_bits(Bf16::INFINITY_BITS).to_f32().to_bits(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            Bf16::from_bits(Bf16::NEG_INFINITY_BITS).to_f32().to_bits(),
            f32::NEG_INFINITY.to_bits()
        );
    }

    #[test]
    fn equality_compares_raw_bits_not_numeric_value() {
        let pos_zero = Bf16::from_bits(Bf16::ZERO_BITS);
        let neg_zero = Bf16::from_bits(Bf16::NEG_ZERO_BITS);
        assert_eq!(pos_zero, Bf16::from_bits(0x0000));
        assert_ne!(pos_zero, neg_zero, "+0 and -0 have different patterns");
        // ...yet both widen to FP32 zero (sign masked out here).
        assert_eq!(
            pos_zero.to_f32().to_bits() & 0x7FFF_FFFF,
            neg_zero.to_f32().to_bits() & 0x7FFF_FFFF
        );

        let nan_a = Bf16::from_bits(0x7FC0);
        let nan_b = Bf16::from_bits(0x7FC1);
        assert!(nan_a.is_nan() && nan_b.is_nan());
        assert_ne!(nan_a, nan_b, "distinct NaN payloads are distinct values");
        assert_eq!(nan_a, nan_a, "bitwise equality is still reflexive");
        // from_f32 canonicalizes every NaN to the single +qNaN pattern.
        assert_eq!(Bf16::from_f32(f32::NAN), Bf16::from_f32(-f32::NAN));
        assert_eq!(Bf16::from_f32(f32::NAN), nan_a);
    }
}
