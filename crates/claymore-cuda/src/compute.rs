//! Native CUDA kernel execution for the M8 F32 and BF16 operations.
//!
//! Both PTX images are generated from the checked-in CUDA sources under
//! `kernels/` for `sm_120`, so the default build (and CI) never needs `nvcc`
//! or a GPU. A compute engine loads both modules eagerly through the Driver
//! API and launches kernels only on a stream belonging to the same retained
//! primary context. Every launch is explicit: no implicit host copy, no CPU
//! fallback and no global `cuCtxSynchronize` inside this layer.
//!
//! BF16 kernels widen to FP32, accumulate/evaluate in FP32 and narrow with
//! round-to-nearest-even at the output, exactly like the CPU reference.

use std::ffi::{c_uint, c_void, CStr, CString};
use std::ptr;
use std::rc::Rc;

use crate::driver::{
    check, ensure_same_context, ContextInner, CuFunctionRaw, CuModuleRaw, CudaStream, CUDA_SUCCESS,
};
use crate::CudaError;

const EMBEDDED_PTX: &str = include_str!("kernels/m8_f32.ptx");
const EMBEDDED_PTX_BF16: &str = include_str!("kernels/m8_bf16.ptx");

const ADD_KERNEL: &[u8] = b"claymore_add_f32\0";
const MATMUL_KERNEL: &[u8] = b"claymore_matmul_f32\0";
const SOFTMAX_KERNEL: &[u8] = b"claymore_softmax_last_dim_f32\0";
const RMS_NORM_KERNEL: &[u8] = b"claymore_rms_norm_f32\0";

const ADD_KERNEL_BF16: &[u8] = b"claymore_add_bf16\0";
const MATMUL_KERNEL_BF16: &[u8] = b"claymore_matmul_bf16\0";
const SOFTMAX_KERNEL_BF16: &[u8] = b"claymore_softmax_last_dim_bf16\0";
const RMS_NORM_KERNEL_BF16: &[u8] = b"claymore_rms_norm_bf16\0";

const ELEMENTWISE_BLOCK: u32 = 256;
const MATMUL_TILE: u32 = 16;
const CUDA_GRID_X_MAX: u64 = 2_147_483_647;
const CUDA_GRID_Y_MAX: u64 = 65_535;

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
    add_bf16: CuFunctionRaw,
    matmul_bf16: CuFunctionRaw,
    softmax_bf16: CuFunctionRaw,
    rms_norm_bf16: CuFunctionRaw,
}

/// Owns the embedded PTX modules. Cheap to clone; all clones share one module
/// per dtype and one primary context, matching Claymore's single-compute-stream
/// rule.
#[derive(Clone)]
pub struct CudaComputeEngine {
    context: Rc<ContextInner>,
    module: Rc<ModuleInner>,
    module_bf16: Rc<ModuleInner>,
    kernels: Rc<Kernels>,
}

impl CudaComputeEngine {
    pub(crate) fn new(context: Rc<ContextInner>) -> Result<Self, CudaError> {
        context.activate()?;
        let module = Rc::new(load_module(Rc::clone(&context), EMBEDDED_PTX)?);
        let module_bf16 = Rc::new(load_module(Rc::clone(&context), EMBEDDED_PTX_BF16)?);
        let kernels = Rc::new(Kernels {
            add: load_function(&module, ADD_KERNEL, "claymore_add_f32")?,
            matmul: load_function(&module, MATMUL_KERNEL, "claymore_matmul_f32")?,
            softmax: load_function(&module, SOFTMAX_KERNEL, "claymore_softmax_last_dim_f32")?,
            rms_norm: load_function(&module, RMS_NORM_KERNEL, "claymore_rms_norm_f32")?,
            add_bf16: load_function(&module_bf16, ADD_KERNEL_BF16, "claymore_add_bf16")?,
            matmul_bf16: load_function(&module_bf16, MATMUL_KERNEL_BF16, "claymore_matmul_bf16")?,
            softmax_bf16: load_function(
                &module_bf16,
                SOFTMAX_KERNEL_BF16,
                "claymore_softmax_last_dim_bf16",
            )?,
            rms_norm_bf16: load_function(
                &module_bf16,
                RMS_NORM_KERNEL_BF16,
                "claymore_rms_norm_bf16",
            )?,
        });
        Ok(Self {
            context,
            module,
            module_bf16,
            kernels,
        })
    }

    fn prepare(&self, stream: &CudaStream) -> Result<(), CudaError> {
        ensure_same_context(self.context.raw, stream.context.raw)?;
        ensure_same_context(self.context.raw, self.module.context.raw)?;
        ensure_same_context(self.context.raw, self.module_bf16.context.raw)?;
        self.context.activate()
    }

    /// Elementwise F32 addition. All three device pointers must reference live
    /// allocations of at least `numel` F32 elements in this context.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn add(
        &self,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let count = checked_count(numel, "add requires at least one element")?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX)?;
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
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn matmul(
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
        let (rows64, inner64, cols64) = checked_matmul(rows, inner, cols)?;
        let grid_x = grid_for(cols64, MATMUL_TILE, CUDA_GRID_X_MAX)?;
        let grid_y = grid_for(rows64, MATMUL_TILE, CUDA_GRID_Y_MAX)?;
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
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn softmax_last_dim(
        &self,
        input: u64,
        output: u64,
        rows: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (rows64, width64) = checked_last_dim(rows, width)?;
        let grid = grid_for(rows64, 1, CUDA_GRID_X_MAX)?;
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
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn rms_norm(
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
        let (rows64, width64) = checked_last_dim(rows, width)?;
        let grid = grid_for(rows64, 1, CUDA_GRID_X_MAX)?;
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

    /// Elementwise BF16 addition. Pointers address 16-bit elements; the kernel
    /// widens to FP32 and rounds the sum to nearest-even BF16.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn add_bf16(
        &self,
        lhs: u64,
        rhs: u64,
        output: u64,
        numel: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let count = checked_count(numel, "add_bf16 requires at least one element")?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX)?;
        let params = [p(&lhs), p(&rhs), p(&output), p(&count)];
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.add_bf16,
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
                "cuLaunchKernel(add_bf16)",
            )
        }
    }

    /// Rank-2 row-major BF16 matrix multiplication with FP32 accumulation.
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn matmul_bf16(
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
        let (rows64, inner64, cols64) = checked_matmul(rows, inner, cols)?;
        let grid_x = grid_for(cols64, MATMUL_TILE, CUDA_GRID_X_MAX)?;
        let grid_y = grid_for(rows64, MATMUL_TILE, CUDA_GRID_Y_MAX)?;
        let params = [
            p(&lhs),
            p(&rhs),
            p(&output),
            p(&rows64),
            p(&inner64),
            p(&cols64),
        ];
        // SAFETY: identical lifetime contract to `matmul`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.matmul_bf16,
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
                "cuLaunchKernel(matmul_bf16)",
            )
        }
    }

    /// Stable BF16 softmax over the last dimension, one CUDA block per row.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn softmax_last_dim_bf16(
        &self,
        input: u64,
        output: u64,
        rows: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (rows64, width64) = checked_last_dim(rows, width)?;
        let grid = grid_for(rows64, 1, CUDA_GRID_X_MAX)?;
        let params = [p(&input), p(&output), p(&rows64), p(&width64)];
        // SAFETY: identical lifetime contract to `softmax_last_dim`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.softmax_bf16,
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
                "cuLaunchKernel(softmax_last_dim_bf16)",
            )
        }
    }

    /// RMS normalization over the last dimension with an explicit BF16 scale.
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. The allocations must remain live and
    /// inaccessible to conflicting operations until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn rms_norm_bf16(
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
        let (rows64, width64) = checked_last_dim(rows, width)?;
        let grid = grid_for(rows64, 1, CUDA_GRID_X_MAX)?;
        let params = [
            p(&input),
            p(&scale),
            p(&output),
            p(&rows64),
            p(&width64),
            &epsilon as *const f32 as *const c_void,
        ];
        // SAFETY: identical lifetime contract to `rms_norm`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.rms_norm_bf16,
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
                "cuLaunchKernel(rms_norm_bf16)",
            )
        }
    }
}

fn load_module(context: Rc<ContextInner>, image: &str) -> Result<ModuleInner, CudaError> {
    let image = CString::new(image)
        .map_err(|_| CudaError::InvalidLaunchConfiguration("PTX contains a NUL byte".into()))?;
    let mut raw = ptr::null_mut();
    // SAFETY: image is a NUL-terminated PTX string and raw is valid output storage.
    let code = unsafe { crate::driver::cuModuleLoadData(&mut raw, image.as_ptr().cast()) };
    if code != CUDA_SUCCESS {
        return Err(CudaError::KernelModuleLoad { code });
    }
    Ok(ModuleInner { raw, context })
}

fn checked_count(numel: usize, message: &str) -> Result<u64, CudaError> {
    if numel == 0 {
        return Err(invalid(message));
    }
    u64::try_from(numel).map_err(|_| invalid("element count overflow"))
}

fn checked_matmul(rows: usize, inner: usize, cols: usize) -> Result<(u64, u64, u64), CudaError> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Err(invalid("matmul dimensions must be non-zero"));
    }
    Ok((
        u64::try_from(rows).map_err(|_| invalid("row count overflow"))?,
        u64::try_from(inner).map_err(|_| invalid("inner dimension overflow"))?,
        u64::try_from(cols).map_err(|_| invalid("column count overflow"))?,
    ))
}

fn checked_last_dim(rows: usize, width: usize) -> Result<(u64, u64), CudaError> {
    if rows == 0 || width == 0 {
        return Err(invalid("softmax/rms_norm dimensions must be non-zero"));
    }
    Ok((
        u64::try_from(rows).map_err(|_| invalid("row count overflow"))?,
        u64::try_from(width).map_err(|_| invalid("width overflow"))?,
    ))
}

fn invalid(message: &str) -> CudaError {
    CudaError::InvalidLaunchConfiguration(message.to_owned())
}

fn p(value: &u64) -> *const c_void {
    value as *const u64 as *const c_void
}

fn grid_for(total: u64, block: u32, maximum: u64) -> Result<c_uint, CudaError> {
    let blocks = total.div_ceil(u64::from(block));
    if blocks > maximum {
        return Err(CudaError::InvalidLaunchConfiguration(format!(
            "grid dimension requires {blocks} blocks; CUDA axis maximum is {maximum}"
        )));
    }
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
        for ptx in [EMBEDDED_PTX, EMBEDDED_PTX_BF16] {
            assert!(
                ptx.contains(".target sm_120"),
                "embedded PTX must target the RTX 5090 compute capability 12.0"
            );
        }
        for symbol in [ADD_KERNEL, MATMUL_KERNEL, SOFTMAX_KERNEL, RMS_NORM_KERNEL] {
            assert!(
                EMBEDDED_PTX.contains(std::str::from_utf8(symbol).unwrap().trim_end_matches('\0'))
            );
        }
        for symbol in [
            ADD_KERNEL_BF16,
            MATMUL_KERNEL_BF16,
            SOFTMAX_KERNEL_BF16,
            RMS_NORM_KERNEL_BF16,
        ] {
            assert!(EMBEDDED_PTX_BF16
                .contains(std::str::from_utf8(symbol).unwrap().trim_end_matches('\0')));
        }
    }

    #[test]
    fn bf16_ptx_declares_round_to_nearest_even_and_fp32_accumulation() {
        // Guards against accidentally regenerating with a different rounding
        // mode or with native BF16 accumulation.
        assert!(EMBEDDED_PTX_BF16.contains("cvt.rn.bf16.f32"));
        assert!(EMBEDDED_PTX_BF16.contains("fma.rn.f32"));
        assert!(!EMBEDDED_PTX_BF16.contains(".target sm_90"));
    }

    #[test]
    fn grid_rounding_and_axis_limits_are_explicit() {
        assert_eq!(grid_for(1, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX).unwrap(), 1);
        assert_eq!(grid_for(257, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX).unwrap(), 2);
        assert_eq!(grid_for(4096, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX).unwrap(), 16);
        assert_eq!(
            grid_for(65_535 * 16, MATMUL_TILE, CUDA_GRID_Y_MAX).unwrap(),
            65_535
        );
        assert!(matches!(
            grid_for(65_536 * 16, MATMUL_TILE, CUDA_GRID_Y_MAX),
            Err(CudaError::InvalidLaunchConfiguration(_))
        ));
        assert!(matches!(
            grid_for(CUDA_GRID_X_MAX + 1, 1, CUDA_GRID_X_MAX),
            Err(CudaError::InvalidLaunchConfiguration(_))
        ));
        assert!(matches!(
            grid_for(u64::MAX, 1, CUDA_GRID_X_MAX),
            Err(CudaError::InvalidLaunchConfiguration(_))
        ));
    }
}
