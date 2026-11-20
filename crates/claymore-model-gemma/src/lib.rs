//! DiffusionGemma integration boundary.
///
/// TODO: implement Claymore-native model loading, bidirectional attention, diffusion
/// denoising/canvas state, and validated KV/state management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaEngineState {
    Uninitialized,
    Ready,
}
