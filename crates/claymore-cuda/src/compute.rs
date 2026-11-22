//! Native CUDA kernel execution for the M8 F32 operations.
//!
//! The PTX is generated from `kernels/m8_f32.cu` for `sm_120` and checked in,
//! so the default build (and CI) never needs `nvcc` or a GPU. A compute engine
//! loads the module eagerly through the Driver API and launches kernels only
//! on a stream belonging to the same retained primary context. Every launch
//! is explicit: there is no implicit host copy, no CPU fallback and no global
//! `cuCtxSynchronize` inside this layer.

use std::ffi::{c_uint, c_void, CStr, CString};
use std::ptr;
use std::rc::Rc;

use crate::driver::{
    check, ensure_same_context, ContextInner, CuFunctionRaw, CuModuleRaw, CudaStream, CUDA_SUCCESS,
};
use crate::CudaError;

const EMBEDDED_PTX: &str = include_str!("kernels/m8_f32.ptx");
const ADD_KERNEL: &[u8] = b"claymore_add_f32\0";
const MATMUL_KERNEL: &[u8] = b"claymore_matmul_f32\0";
const SOFTMAX_KERNEL: &[u8] = b"claymore_softmax_last_dim_f32\0";
const RMS_NORM_KERNEL: &[u8] = b"claymore_rms_norm_f32\0";
const ELEMENTWISE_BLOCK: u32 = 256;
const MATMUL_TILE: u32 = 16;

struct ModuleInner {
    raw: CuModuleRaw,
    context: Rc<ContextInner>,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if self.context.activate().is_ok() {
            // SAFETY: this owner balances exactly one successful module load.
            unsafe {
                let _ = crate::driver::cuModuleUnload(self.raw);
            }
        }
    }
}

struct Kernels {
    add: CuFunctionRaw,
    matmul: CuFunctionRaw,
    softmax: CuFunctionRaw,
    rms_norm: CuFunctionRaw,
}

/// Owns the embedded PTX module. Cheap to clone; all clones share one module
/// and one primary context, matching Claymore's single-compute-stream rule.
#[derive(Clone)]
pub struct CudaComputeEngine {
    context: Rc<ContextInner>,
    module: Rc<ModuleInner>,
    kernels: Rc<Kernels>,
}

impl CudaComputeEngine {
    pub(crate) fn new(context: Rc<ContextInner>) -> Result<Self, CudaError> {
        context.activate()?;
        let image = CString::new(EMBEDDED_PTX)
            .map_err(|_| CudaError::InvalidLaunchConfiguration("PTX contains a NUL byte".into()))?;
        let mut raw = ptr::null_mut();
        // SAFETY: image is a NUL-terminated PTX string and raw is valid output storage.
        let code = unsafe { crate::driver::cuModuleLoadData(&mut raw, image.as_ptr().cast()) };
        if code != CUDA_SUCCESS {
            return Err(CudaError::KernelModuleLoad { code });
        }
        let module = Rc::new(ModuleInner {
            raw,
            context: Rc::clone(&context),
        });
        let kernels = Rc::new(Kernels {
            add: load_function(&module, ADD_KERNEL, "claymore_add_f32")?,
            matmul: load_function(&module, MATMUL_KERNEL, "claymore_matmul_f32")?,
            softmax: load_function(&module, SOFTMAX_KERNEL, "claymore_softmax_last_dim_f32")?,
            rms_norm: load_function(&module, RMS_NORM_KERNEL, "claymore_rms_norm_f32")?,
        });
        Ok(Self {
            context,
            module,
            kernels,
        })
    }

    fn prepare(&self, stream: &CudaStream) -> Result<(), CudaError> {
        ensure_same_context(self.context.raw, stream.context.raw)?;
        ensure_same_context(self.context.raw, self.module.context.raw)?;
        self.context.activate()
    }

    /// Elementwise F32 addition. All three device pointers must reference live
    /// allocations of at least `numel` F32 elements in this context.
    pub fn add(
        &self,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if numel == 0 {
            return Err(invalid("add requires at least one element"));
        }
        let count = u64::try_from(numel).map_err(|_| invalid("element count overflow"))?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK)?;
        let params = [p(&lhs), p(&rhs), p(&output), p(&count)];
        // SAFETY: the module and stream belong to the current context; the
        // caller guarantees that every device pointer stays live until the
        // stream's completion event has been observed.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.add,
                    grid,
                    1,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(add)",
            )
        }
    }

    /// Rank-2 row-major F32 matrix multiplication.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        lhs: u64,
        rhs: u64,
        output: u64,
        rows: usize,
        inner: usize,
        cols: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if rows == 0 || inner == 0 || cols == 0 {
            return Err(invalid("matmul dimensions must be non-zero"));
        }
        let rows64 = u64::try_from(rows).map_err(|_| invalid("row count overflow"))?;
        let inner64 = u64::try_from(inner).map_err(|_| invalid("inner dimension overflow"))?;
        let cols64 = u64::try_from(cols).map_err(|_| invalid("column count overflow"))?;
        let grid_x = grid_for(cols64, MATMUL_TILE)?;
        let grid_y = grid_for(rows64, MATMUL_TILE)?;
        let params = [
            p(&lhs),
            p(&rhs),
            p(&output),
            p(&rows64),
            p(&inner64),
            p(&cols64),
        ];
        // SAFETY: see `add`; the caller owns every referenced allocation.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.matmul,
                    grid_x,
                    grid_y,
                    1,
                    MATMUL_TILE,
                    MATMUL_TILE,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(matmul)",
            )
        }
    }

    /// Stable softmax over the last dimension, one CUDA block per row.
    pub fn softmax_last_dim(
        &self,
        input: u64,
        output: u64,
        rows: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if rows == 0 || width == 0 {
            return Err(invalid("softmax dimensions must be non-zero"));
        }
        let rows64 = u64::try_from(rows).map_err(|_| invalid("row count overflow"))?;
        let width64 = u64::try_from(width).map_err(|_| invalid("width overflow"))?;
        let grid = grid_for(rows64, 1)?;
        let params = [p(&input), p(&output), p(&rows64), p(&width64)];
        // SAFETY: see `add`; dynamic shared memory holds one float per thread.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.softmax,
                    grid,
                    1,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    ELEMENTWISE_BLOCK * 4,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(softmax_last_dim)",
            )
        }
    }

    /// RMS normalization over the last dimension with an explicit F32 scale.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        input: u64,
        scale: u64,
        output: u64,
        rows: usize,
        width: usize,
        epsilon: f32,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if rows == 0 || width == 0 {
            return Err(invalid("rms_norm dimensions must be non-zero"));
        }
        let rows64 = u64::try_from(rows).map_err(|_| invalid("row count overflow"))?;
        let width64 = u64::try_from(width).map_err(|_| invalid("width overflow"))?;
        let grid = grid_for(rows64, 1)?;
        let params = [
            p(&input),
            p(&scale),
            p(&output),
            p(&rows64),
            p(&width64),
            &epsilon as *const f32 as *const c_void,
        ];
        // SAFETY: see `add`; dynamic shared memory holds one float per thread.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.rms_norm,
                    grid,
                    1,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    ELEMENTWISE_BLOCK * 4,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(rms_norm)",
            )
        }
    }
}

fn invalid(message: &str) -> CudaError {
    CudaError::InvalidLaunchConfiguration(message.to_owned())
}

fn p(value: &u64) -> *const c_void {
    value as *const u64 as *const c_void
}

fn grid_for(total: u64, block: u32) -> Result<c_uint, CudaError> {
    let blocks = total.div_ceil(u64::from(block));
    c_uint::try_from(blocks).map_err(|_| invalid("grid dimension exceeds CUDA limit"))
}

fn load_function(
    module: &ModuleInner,
    name: &[u8],
    label: &'static str,
) -> Result<CuFunctionRaw, CudaError> {
    let symbol = CStr::from_bytes_with_nul(name).map_err(|_| {
        CudaError::InvalidLaunchConfiguration("kernel name is not NUL-terminated".into())
    })?;
    let mut function = ptr::null_mut();
    let code =
        unsafe { crate::driver::cuModuleGetFunction(&mut function, module.raw, symbol.as_ptr()) };
    if code != CUDA_SUCCESS {
        return Err(CudaError::MissingKernel(label));
    }
    Ok(function)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_ptx_targets_blackwell_sm_120() {
        assert!(
            EMBEDDED_PTX.contains(".target sm_120"),
            "embedded PTX must target the RTX 5090 compute capability 12.0"
        );
        for symbol in [ADD_KERNEL, MATMUL_KERNEL, SOFTMAX_KERNEL, RMS_NORM_KERNEL] {
            assert!(
                EMBEDDED_PTX.contains(std::str::from_utf8(symbol).unwrap().trim_end_matches('\0'))
            );
        }
    }

    #[test]
    fn grid_rounding_and_limits_are_explicit() {
        assert_eq!(grid_for(1, ELEMENTWISE_BLOCK).unwrap(), 1);
        assert_eq!(grid_for(257, ELEMENTWISE_BLOCK).unwrap(), 2);
        assert_eq!(grid_for(4096, ELEMENTWISE_BLOCK).unwrap(), 16);
        assert!(matches!(
            grid_for(u64::MAX, 1),
            Err(CudaError::InvalidLaunchConfiguration(_))
        ));
    }
}
