//! Shared lifecycle invariants used by backend-specific memory managers.
//!
//! These helpers centralize logical-clock and release-horizon semantics without
//! forcing CPU and CUDA resource ownership into one backend abstraction.

pub(crate) const IN_FLIGHT_TICK: u64 = u64::MAX;

/// Returns the backwards transition, if any. Equal ticks are intentionally valid.
pub(crate) fn backwards_tick(previous: Option<u64>, provided: u64) -> Option<(u64, u64)> {
    previous
        .filter(|previous| provided < *previous)
        .map(|previous| (previous, provided))
}

/// A resource is releasable only when its backend completion horizon has passed.
pub(crate) const fn release_is_safe(safe_after_tick: u64, tick: u64) -> bool {
    safe_after_tick <= tick
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_tick_accepts_equal_and_forward_transitions() {
        assert_eq!(backwards_tick(None, 0), None);
        assert_eq!(backwards_tick(Some(4), 4), None);
        assert_eq!(backwards_tick(Some(4), 5), None);
    }

    #[test]
    fn monotonic_tick_rejects_backward_transition() {
        assert_eq!(backwards_tick(Some(4), 3), Some((4, 3)));
    }

    #[test]
    fn release_horizon_is_inclusive_and_never_early() {
        assert!(!release_is_safe(5, 4));
        assert!(release_is_safe(5, 5));
        assert!(release_is_safe(5, 6));
    }
}
