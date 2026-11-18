//! Clover-35B-A3B-preview integration boundary.
///
/// TODO: implement Candle model loading, GDN/block-diffusion execution,
/// MoE routing, expert residency/cache management, and attention kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloverEngineState {
    Uninitialized,
    Ready,
}
