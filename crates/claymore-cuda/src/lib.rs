//! CUDA backend boundary.
//!
//! The default build is hardware-independent and provides device inventory
//! types plus explicit capability reporting. Enable `cuda-driver` to compile
//! the CUDA Driver API backend. That feature requires the NVIDIA driver library
//! (`libcuda.so.1`) to be available at link and runtime.
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub ordinal: u32,
    pub name: String,
    pub total_memory_bytes: u64,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA driver backend was not compiled; rebuild with --features cuda-driver")]
    FeatureDisabled,
    #[error("CUDA driver error {code} during {operation}")]
    Driver { operation: &'static str, code: i32 },
    #[error("requested {requested} bytes, buffer capacity is {capacity}")]
    BufferTooSmall { requested: usize, capacity: usize },
    #[error("size conversion overflow")]
    SizeOverflow,
    #[error("CUDA returned an invalid device name")]
    InvalidDeviceName,
    #[error("CUDA device ordinal {0} is invalid or unavailable")]
    InvalidDeviceOrdinal(u32),
    #[error("zero-sized CUDA allocations are not supported")]
    ZeroSizedAllocation,
    #[error("CUDA resources belong to different contexts")]
    ContextMismatch,
    #[error("CUDA kernel module could not be loaded from embedded PTX")]
    KernelModuleLoad { code: i32 },
    #[error("CUDA kernel symbol {0} is missing from the embedded module")]
    MissingKernel(&'static str),
    #[error("invalid kernel launch configuration: {0}")]
    InvalidLaunchConfiguration(String),
}

pub trait DeviceInventory {
    fn devices(&self) -> Result<Vec<CudaDeviceInfo>, CudaError>;
}

#[cfg(not(feature = "cuda-driver"))]
pub struct CudaRuntime;

#[cfg(not(feature = "cuda-driver"))]
impl CudaRuntime {
    pub fn new() -> Result<Self, CudaError> {
        Err(CudaError::FeatureDisabled)
    }
}

#[cfg(feature = "cuda-driver")]
mod driver;

#[cfg(feature = "cuda-driver")]
pub use compute::CudaComputeEngine;
#[cfg(feature = "cuda-driver")]
pub use driver::CudaRuntime;
#[cfg(feature = "cuda-driver")]
pub use driver::{CudaDeviceBuffer, CudaDriver, CudaEvent, CudaStream, PinnedHostBuffer};
#[cfg(feature = "cuda-driver")]
mod compute;

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "cuda-driver"))]
    use super::*;
    #[test]
    fn disabled_backend_reports_actionable_error() {
        #[cfg(not(feature = "cuda-driver"))]
        assert!(matches!(
            CudaRuntime::new(),
            Err(CudaError::FeatureDisabled)
        ));
    }
}
