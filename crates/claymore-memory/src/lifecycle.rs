//! Shared lifecycle invariants used by backend-specific memory managers.
///
/// These helpers centralize logical-clock and release-horizon semantics without
/// forcing CPU and CUDA resource ownership into one backend abstraction.

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
