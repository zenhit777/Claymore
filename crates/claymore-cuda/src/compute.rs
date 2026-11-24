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
const EMBEDDED_PTX_M10: &str = include_str!("kernels/m10.ptx");

const ADD_KERNEL: &[u8] = b"claymore_add_f32\0";
const MATMUL_KERNEL: &[u8] = b"claymore_matmul_f32\0";
const SOFTMAX_KERNEL: &[u8] = b"claymore_softmax_last_dim_f32\0";
const RMS_NORM_KERNEL: &[u8] = b"claymore_rms_norm_f32\0";

const ADD_KERNEL_BF16: &[u8] = b"claymore_add_bf16\0";
const MATMUL_KERNEL_BF16: &[u8] = b"claymore_matmul_bf16\0";
const SOFTMAX_KERNEL_BF16: &[u8] = b"claymore_softmax_last_dim_bf16\0";
const RMS_NORM_KERNEL_BF16: &[u8] = b"claymore_rms_norm_bf16\0";

// M10 diffusion primitives. Each symbol is F32/BF16 specific: dispatch never
// shares a kernel across dtypes.
const LINEAR_F32: &[u8] = b"claymore_linear_f32\0";
const ROTARY_F32: &[u8] = b"claymore_rotary_embed_f32\0";
const ATTENTION_F32: &[u8] = b"claymore_attention_f32\0";
const SOFT_CAP_F32: &[u8] = b"claymore_soft_cap_f32\0";
const SCALE_F32: &[u8] = b"claymore_scale_f32\0";
const LINEAR_BF16: &[u8] = b"claymore_linear_bf16\0";
const ROTARY_BF16: &[u8] = b"claymore_rotary_embed_bf16\0";
const ATTENTION_BF16: &[u8] = b"claymore_attention_bf16\0";
const SOFT_CAP_BF16: &[u8] = b"claymore_soft_cap_bf16\0";
const SCALE_BF16: &[u8] = b"claymore_scale_bf16\0";

const WIDEN_BF16_F32: &[u8] = b"claymore_widen_bf16_f32\0";
const ROUTER_TOP_K_F32: &[u8] = b"claymore_router_top_k_f32\0";
const GELU_MUL_UP_F32: &[u8] = b"claymore_gelu_mul_up_f32\0";
const GELU_MUL_UP_BF16: &[u8] = b"claymore_gelu_mul_up_bf16\0";
const GATHER_ROWS_F32: &[u8] = b"claymore_gather_rows_f32\0";
const GATHER_ROWS_BF16: &[u8] = b"claymore_gather_rows_bf16\0";
const SCATTER_ADD_F32: &[u8] = b"claymore_scatter_add_f32\0";
const SCATTER_ADD_BF16: &[u8] = b"claymore_scatter_add_bf16\0";
const EMBED_GATHER_F32: &[u8] = b"claymore_embed_gather_f32\0";
const EMBED_GATHER_BF16: &[u8] = b"claymore_embed_gather_bf16\0";
const ROW_MASK_F32: &[u8] = b"claymore_row_mask_f32\0";
const ROW_MASK_BF16: &[u8] = b"claymore_row_mask_bf16\0";
const SCALE_BIAS_F32: &[u8] = b"claymore_scale_bias_f32\0";
const SCALE_BIAS_BF16: &[u8] = b"claymore_scale_bias_bf16\0";

const ELEMENTWISE_BLOCK: u32 = 256;
const MATMUL_TILE: u32 = 16;
const LINEAR_TILE: u32 = 64;
const ROTARY_BLOCK: u32 = 256;
const ATTENTION_THREADS: u32 = 64;
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
    linear_f32: CuFunctionRaw,
    rotary_f32: CuFunctionRaw,
    attention_f32: CuFunctionRaw,
    soft_cap_f32: CuFunctionRaw,
    scale_f32: CuFunctionRaw,
    linear_bf16: CuFunctionRaw,
    rotary_bf16: CuFunctionRaw,
    attention_bf16: CuFunctionRaw,
    soft_cap_bf16: CuFunctionRaw,
    scale_bf16: CuFunctionRaw,
    widen_bf16_f32: CuFunctionRaw,
    router_top_k_f32: CuFunctionRaw,
    gelu_mul_up_f32: CuFunctionRaw,
    gelu_mul_up_bf16: CuFunctionRaw,
    gather_rows_f32: CuFunctionRaw,
    gather_rows_bf16: CuFunctionRaw,
    scatter_add_f32: CuFunctionRaw,
    scatter_add_bf16: CuFunctionRaw,
    embed_gather_f32: CuFunctionRaw,
    embed_gather_bf16: CuFunctionRaw,
    row_mask_f32: CuFunctionRaw,
    row_mask_bf16: CuFunctionRaw,
    scale_bias_f32: CuFunctionRaw,
    scale_bias_bf16: CuFunctionRaw,
}

/// Owns the embedded PTX modules. Cheap to clone; all clones share one module
/// per dtype and one primary context, matching Claymore's single-compute-stream
/// rule.
#[derive(Clone)]
pub struct CudaComputeEngine {
    context: Rc<ContextInner>,
    module: Rc<ModuleInner>,
    module_bf16: Rc<ModuleInner>,
    module_m10: Rc<ModuleInner>,
    kernels: Rc<Kernels>,
}

impl CudaComputeEngine {
    pub(crate) fn new(context: Rc<ContextInner>) -> Result<Self, CudaError> {
        context.activate()?;
        let module = Rc::new(load_module(Rc::clone(&context), EMBEDDED_PTX)?);
        let module_bf16 = Rc::new(load_module(Rc::clone(&context), EMBEDDED_PTX_BF16)?);
        let module_m10 = Rc::new(load_module(Rc::clone(&context), EMBEDDED_PTX_M10)?);
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
            linear_f32: load_function(&module_m10, LINEAR_F32, "claymore_linear_f32")?,
            rotary_f32: load_function(&module_m10, ROTARY_F32, "claymore_rotary_embed_f32")?,
            attention_f32: load_function(&module_m10, ATTENTION_F32, "claymore_attention_f32")?,
            soft_cap_f32: load_function(&module_m10, SOFT_CAP_F32, "claymore_soft_cap_f32")?,
            scale_f32: load_function(&module_m10, SCALE_F32, "claymore_scale_f32")?,
            linear_bf16: load_function(&module_m10, LINEAR_BF16, "claymore_linear_bf16")?,
            rotary_bf16: load_function(&module_m10, ROTARY_BF16, "claymore_rotary_embed_bf16")?,
            attention_bf16: load_function(&module_m10, ATTENTION_BF16, "claymore_attention_bf16")?,
            soft_cap_bf16: load_function(&module_m10, SOFT_CAP_BF16, "claymore_soft_cap_bf16")?,
            scale_bf16: load_function(&module_m10, SCALE_BF16, "claymore_scale_bf16")?,
            widen_bf16_f32: load_function(&module_m10, WIDEN_BF16_F32, "claymore_widen_bf16_f32")?,
            router_top_k_f32: load_function(
                &module_m10,
                ROUTER_TOP_K_F32,
                "claymore_router_top_k_f32",
            )?,
            gelu_mul_up_f32: load_function(
                &module_m10,
                GELU_MUL_UP_F32,
                "claymore_gelu_mul_up_f32",
            )?,
            gelu_mul_up_bf16: load_function(
                &module_m10,
                GELU_MUL_UP_BF16,
                "claymore_gelu_mul_up_bf16",
            )?,
            gather_rows_f32: load_function(
                &module_m10,
                GATHER_ROWS_F32,
                "claymore_gather_rows_f32",
            )?,
            gather_rows_bf16: load_function(
                &module_m10,
                GATHER_ROWS_BF16,
                "claymore_gather_rows_bf16",
            )?,
            scatter_add_f32: load_function(
                &module_m10,
                SCATTER_ADD_F32,
                "claymore_scatter_add_f32",
            )?,
            scatter_add_bf16: load_function(
                &module_m10,
                SCATTER_ADD_BF16,
                "claymore_scatter_add_bf16",
            )?,
            embed_gather_f32: load_function(
                &module_m10,
                EMBED_GATHER_F32,
                "claymore_embed_gather_f32",
            )?,
            embed_gather_bf16: load_function(
                &module_m10,
                EMBED_GATHER_BF16,
                "claymore_embed_gather_bf16",
            )?,
            row_mask_f32: load_function(&module_m10, ROW_MASK_F32, "claymore_row_mask_f32")?,
            row_mask_bf16: load_function(&module_m10, ROW_MASK_BF16, "claymore_row_mask_bf16")?,
            scale_bias_f32: load_function(&module_m10, SCALE_BIAS_F32, "claymore_scale_bias_f32")?,
            scale_bias_bf16: load_function(
                &module_m10,
                SCALE_BIAS_BF16,
                "claymore_scale_bias_bf16",
            )?,
        });
        Ok(Self {
            context,
            module,
            module_bf16,
            module_m10,
            kernels,
        })
    }

    fn prepare(&self, stream: &CudaStream) -> Result<(), CudaError> {
        ensure_same_context(self.context.raw, stream.context.raw)?;
        ensure_same_context(self.context.raw, self.module.context.raw)?;
        ensure_same_context(self.context.raw, self.module_bf16.context.raw)?;
        ensure_same_context(self.context.raw, self.module_m10.context.raw)?;
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

    /// Dense linear projection: `[tokens, in_features] x [out_features, in_features]`
    /// -> `[tokens, out_features]`. Every product is accumulated in FP32.
    ///
    /// Pointers are device addresses (the driver's `CUdeviceptr`).
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context and stay live until the stream completion
    /// event has been observed by the caller.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn linear(
        &self,
        input: u64,
        weight: u64,
        output: u64,
        tokens: usize,
        in_features: usize,
        out_features: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.linear_dispatch(
            input,
            weight,
            output,
            tokens,
            in_features,
            out_features,
            false,
            stream,
        )
    }

    /// BF16 variant of `linear`.
    ///
    /// Pointers are device addresses (the driver's `CUdeviceptr`).
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context and stay live until the stream completion
    /// event has been observed by the caller.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn linear_bf16(
        &self,
        input: u64,
        weight: u64,
        output: u64,
        tokens: usize,
        in_features: usize,
        out_features: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.linear_dispatch(
            input,
            weight,
            output,
            tokens,
            in_features,
            out_features,
            true,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn linear_dispatch(
        &self,
        input: u64,
        weight: u64,
        output: u64,
        tokens: usize,
        in_features: usize,
        out_features: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (tokens64, inner64, cols64) = checked_matmul(tokens, in_features, out_features)?;
        let grid_x = grid_for(cols64, LINEAR_TILE, CUDA_GRID_X_MAX)?;
        let grid_y = grid_for(tokens64, 1, CUDA_GRID_Y_MAX)?;
        let params = [
            p(&input),
            p(&weight),
            p(&output),
            p(&tokens64),
            p(&inner64),
            p(&cols64),
        ];
        let function = if bf16 {
            self.kernels.linear_bf16
        } else {
            self.kernels.linear_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
                    grid_x,
                    grid_y,
                    1,
                    LINEAR_TILE,
                    1,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(linear)",
            )
        }
    }

    /// Apply rotary embeddings to `[tokens, heads, head_dim]` activations.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rotary_embed(
        &self,
        input: u64,
        cos: u64,
        sin: u64,
        output: u64,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.rotary_dispatch(
            input, cos, sin, output, tokens, heads, head_dim, false, stream,
        )
    }

    /// BF16 variant of `rotary_embed`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context and stay live until the stream completion
    /// event has been observed by the caller.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rotary_embed_bf16(
        &self,
        input: u64,
        cos: u64,
        sin: u64,
        output: u64,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.rotary_dispatch(
            input, cos, sin, output, tokens, heads, head_dim, true, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn rotary_dispatch(
        &self,
        input: u64,
        cos: u64,
        sin: u64,
        output: u64,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if tokens == 0 || heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
            return Err(invalid(
                "rotary dimensions must be non-zero with an even head_dim",
            ));
        }
        let (tokens64, heads64, dim64) = (tokens as u64, heads as u64, head_dim as u64);
        let grid_x = grid_for(heads64, 1, CUDA_GRID_X_MAX)?;
        let grid_y = grid_for(tokens64, 1, CUDA_GRID_Y_MAX)?;
        let params = [
            p(&input),
            p(&cos),
            p(&sin),
            p(&output),
            p(&tokens64),
            p(&heads64),
            p(&dim64),
        ];
        let function = if bf16 {
            self.kernels.rotary_bf16
        } else {
            self.kernels.rotary_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
                    grid_x,
                    grid_y,
                    1,
                    ROTARY_BLOCK,
                    1,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(rotary_embed)",
            )
        }
    }

    /// Grouped-query multi-head attention with an optional additive mask.
    ///
    /// `mask` is 0 when no mask tensor is supplied.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention(
        &self,
        query: u64,
        key: u64,
        value: u64,
        mask: u64,
        output: u64,
        scale: f32,
        query_tokens: usize,
        key_tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        masked: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.attention_dispatch(
            query,
            key,
            value,
            mask,
            output,
            scale,
            query_tokens,
            key_tokens,
            heads,
            kv_heads,
            head_dim,
            masked,
            false,
            stream,
        )
    }

    /// BF16 variant of `attention`.
    ///
    /// `mask` is 0 when no mask tensor is supplied.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context and stay live until the stream completion
    /// event has been observed by the caller.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_bf16(
        &self,
        query: u64,
        key: u64,
        value: u64,
        mask: u64,
        output: u64,
        scale: f32,
        query_tokens: usize,
        key_tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        masked: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.attention_dispatch(
            query,
            key,
            value,
            mask,
            output,
            scale,
            query_tokens,
            key_tokens,
            heads,
            kv_heads,
            head_dim,
            masked,
            true,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn attention_dispatch(
        &self,
        query: u64,
        key: u64,
        value: u64,
        mask: u64,
        output: u64,
        scale: f32,
        query_tokens: usize,
        key_tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        masked: bool,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if query_tokens == 0 || key_tokens == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(invalid("attention dimensions must be non-zero"));
        }
        if heads % kv_heads != 0 {
            return Err(invalid("attention heads must be a multiple of kv_heads"));
        }
        if masked && mask == 0 {
            return Err(invalid("masked attention requires a mask pointer"));
        }
        // The kernel keeps the score and probability vectors in dynamic shared
        // memory: two FP32 vectors of `key_tokens` entries.
        let shared = (key_tokens as u32)
            .checked_mul(2)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("attention shared memory overflow"))?;
        if shared > 48 * 1024 {
            return Err(invalid(
                "attention key length exceeds the 48 KiB static shared memory budget",
            ));
        }
        let query64 = query_tokens as u64;
        let key64 = key_tokens as u64;
        let heads64 = heads as u64;
        let kv64 = kv_heads as u64;
        let dim64 = head_dim as u64;
        let masked_flag: u64 = if masked { 1 } else { 0 };
        let params = [
            p(&query),
            p(&key),
            p(&value),
            p(&mask),
            p(&output),
            &scale as *const f32 as *const c_void,
            p(&query64),
            p(&key64),
            p(&heads64),
            p(&kv64),
            p(&dim64),
            p(&masked_flag),
        ];
        let function = if bf16 {
            self.kernels.attention_bf16
        } else {
            self.kernels.attention_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
                    heads64 as c_uint,
                    query64 as c_uint,
                    1,
                    ATTENTION_THREADS,
                    1,
                    1,
                    shared,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(attention)",
            )
        }
    }

    /// Logit soft-capping: `cap * tanh(x / cap)`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    pub unsafe fn soft_cap(
        &self,
        input: u64,
        output: u64,
        numel: usize,
        cap: f32,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.soft_cap_dispatch(input, output, numel, cap, false, stream)
    }

    /// BF16 variant of `soft_cap`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context and stay live until the stream completion
    /// event has been observed by the caller.
    pub unsafe fn soft_cap_bf16(
        &self,
        input: u64,
        output: u64,
        numel: usize,
        cap: f32,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.soft_cap_dispatch(input, output, numel, cap, true, stream)
    }

    unsafe fn soft_cap_dispatch(
        &self,
        input: u64,
        output: u64,
        numel: usize,
        cap: f32,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let count = checked_count(numel, "soft_cap requires at least one element")?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX)?;
        let params = [
            p(&input),
            p(&output),
            p(&count),
            &cap as *const f32 as *const c_void,
        ];
        let function = if bf16 {
            self.kernels.soft_cap_bf16
        } else {
            self.kernels.soft_cap_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
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
                "cuLaunchKernel(soft_cap)",
            )
        }
    }

    /// Multiply by an explicit factor tensor. `factor_width` is 1 for a
    /// broadcast scalar factor, or `width` for a per-column factor.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context and stay live until the stream completion
    /// event has been observed by the caller.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scale(
        &self,
        input: u64,
        factor: u64,
        output: u64,
        numel: usize,
        width: usize,
        factor_width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.scale_dispatch(
            input,
            factor,
            output,
            numel,
            width,
            factor_width,
            false,
            stream,
        )
    }

    /// BF16 variant of `scale`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context and stay live until the stream completion
    /// event has been observed by the caller.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scale_bf16(
        &self,
        input: u64,
        factor: u64,
        output: u64,
        numel: usize,
        width: usize,
        factor_width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.scale_dispatch(
            input,
            factor,
            output,
            numel,
            width,
            factor_width,
            true,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn scale_dispatch(
        &self,
        input: u64,
        factor: u64,
        output: u64,
        numel: usize,
        width: usize,
        factor_width: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let count = checked_count(numel, "scale requires at least one element")?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX)?;
        // A scalar factor needs no column rule, so `width` stays non-zero for
        // the per-column case only; the kernel ignores it when factor_width is 1.
        let columns = u64::try_from(width).map_err(|_| invalid("width overflow"))?;
        let factors = u64::try_from(factor_width).map_err(|_| invalid("factor width overflow"))?;
        if factor_width != 1 && factor_width != width {
            return Err(invalid(
                "scale factor must have one element or the last dimension",
            ));
        }
        if factor_width == 1 && width == 0 {
            return Err(invalid("scale requires a non-empty tensor"));
        }
        let params = [
            p(&input),
            p(&factor),
            p(&output),
            p(&count),
            p(&columns),
            p(&factors),
        ];
        let function = if bf16 {
            self.kernels.scale_bf16
        } else {
            self.kernels.scale_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
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
                "cuLaunchKernel(scale)",
            )
        }
    }

    /// Zero a live device allocation. Used to prepare scatter-add outputs,
    /// the only accumulation kernel in the contract; atomics require a
    /// pre-zeroed destination.
    /// # Safety
    /// `output` must reference a live allocation of at least `bytes` bytes in
    /// this engine's CUDA context.
    pub unsafe fn zero(
        &self,
        output: u64,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if bytes == 0 {
            return Err(invalid("zero-sized memset"));
        }
        // SAFETY: the memset targets a live allocation on this stream.
        unsafe {
            check(
                crate::driver::cuMemsetD8Async(output, 0, bytes, stream.raw),
                "cuMemsetD8Async",
            )
        }
    }

    /// Bit-exact BF16 widening to F32. The BF16 input uses `numel` elements;
    /// the F32 output must reserve `numel * 4` bytes.
    /// # Safety
    /// Both pointers must reference live, sufficiently sized allocations in
    /// this engine's CUDA context.
    pub unsafe fn widen_bf16(
        &self,
        input: u64,
        output: u64,
        numel: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let count = checked_count(numel, "widen requires at least one element")?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX)?;
        let params = [p(&input), p(&output), p(&count)];
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.widen_bf16_f32,
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
                "cuLaunchKernel(widen)",
            )
        }
    }

    /// Router selection for `[tokens, experts]` FP32 scores. One block per
    /// token; the block covers every expert, so `experts` must not exceed the
    /// launch block width.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn router_top_k(
        &self,
        scores: u64,
        per_expert_scale: u64,
        output: u64,
        tokens: usize,
        experts: usize,
        top_k: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        if tokens == 0 || experts == 0 || top_k == 0 || top_k > experts {
            return Err(invalid("router dimensions are invalid"));
        }
        if experts as u32 > ELEMENTWISE_BLOCK {
            return Err(invalid(
                "router expert count exceeds the launch block width",
            ));
        }
        // probs + the 32-lane block reduction buffer, in dynamic shared memory.
        let shared = (experts as u32)
            .checked_mul(4)
            .and_then(|n| n.checked_add(32 * 4))
            .ok_or_else(|| invalid("router shared memory overflow"))?;
        if shared > 48 * 1024 {
            return Err(invalid("router shared memory exceeds 48 KiB"));
        }
        let tokens64 = u64::try_from(tokens).map_err(|_| invalid("token count overflow"))?;
        let experts64 = u64::try_from(experts).map_err(|_| invalid("expert count overflow"))?;
        let top_k64 = u64::try_from(top_k).map_err(|_| invalid("top-k overflow"))?;
        let params = [
            p(&scores),
            p(&per_expert_scale),
            p(&output),
            p(&tokens64),
            p(&experts64),
            p(&top_k64),
        ];
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    self.kernels.router_top_k_f32,
                    tokens64 as c_uint,
                    1,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    shared,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(router_top_k)",
            )
        }
    }

    /// `gelu_pytorch_tanh(gate) * up` over `[rows, cols]`, F32.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    pub unsafe fn gelu_mul_up(
        &self,
        gate: u64,
        up: u64,
        output: u64,
        numel: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.gelu_mul_up_dispatch(gate, up, output, numel, false, stream)
    }

    /// BF16 variant of `gelu_mul_up`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    pub unsafe fn gelu_mul_up_bf16(
        &self,
        gate: u64,
        up: u64,
        output: u64,
        numel: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.gelu_mul_up_dispatch(gate, up, output, numel, true, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn gelu_mul_up_dispatch(
        &self,
        gate: u64,
        up: u64,
        output: u64,
        numel: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let count = checked_count(numel, "gelu_mul_up requires at least one element")?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX)?;
        let params = [p(&gate), p(&up), p(&output), p(&count)];
        let function = if bf16 {
            self.kernels.gelu_mul_up_bf16
        } else {
            self.kernels.gelu_mul_up_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
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
                "cuLaunchKernel(gelu_mul_up)",
            )
        }
    }

    /// Row gather with F32-encoded indices: `input [rows, width]`,
    /// `indices [out_rows]` -> `output [out_rows, width]`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    pub unsafe fn gather_rows(
        &self,
        input: u64,
        indices: u64,
        output: u64,
        out_rows: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.gather_rows_dispatch(input, indices, output, out_rows, width, false, stream)
    }

    /// BF16 variant of `gather_rows`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    pub unsafe fn gather_rows_bf16(
        &self,
        input: u64,
        indices: u64,
        output: u64,
        out_rows: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.gather_rows_dispatch(input, indices, output, out_rows, width, true, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn gather_rows_dispatch(
        &self,
        input: u64,
        indices: u64,
        output: u64,
        out_rows: usize,
        width: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (rows64, width64) = checked_last_dim(out_rows, width)?;
        let grid_y = grid_for(width64, ELEMENTWISE_BLOCK, CUDA_GRID_Y_MAX)?;
        let params = [p(&input), p(&indices), p(&output), p(&rows64), p(&width64)];
        let function = if bf16 {
            self.kernels.gather_rows_bf16
        } else {
            self.kernels.gather_rows_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
                    rows64 as c_uint,
                    grid_y,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(gather_rows)",
            )
        }
    }

    /// Weighted scatter-add with FP32 atomics. The destination `output` must
    /// be zeroed by `zero` on the same stream before this launch; the
    /// accumulation order is not deterministic, so the differential budget is
    /// a model-derived bound rather than a ULP budget.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. `output` must be pre-zeroed.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scatter_add(
        &self,
        source: u64,
        weights: u64,
        rows: u64,
        output: u64,
        pairs: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.scatter_add_dispatch(source, weights, rows, output, pairs, width, false, stream)
    }

    /// BF16 variant of `scatter_add` (bf16 atomic accumulation, matching the
    /// pinned upstream bf16 index_add behaviour).
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context. `output` must be pre-zeroed.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scatter_add_bf16(
        &self,
        source: u64,
        weights: u64,
        rows: u64,
        output: u64,
        pairs: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.scatter_add_dispatch(source, weights, rows, output, pairs, width, true, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn scatter_add_dispatch(
        &self,
        source: u64,
        weights: u64,
        rows: u64,
        output: u64,
        pairs: usize,
        width: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (pairs64, width64) = checked_last_dim(pairs, width)?;
        let grid_y = grid_for(width64, ELEMENTWISE_BLOCK, CUDA_GRID_Y_MAX)?;
        let params = [
            p(&source),
            p(&weights),
            p(&rows),
            p(&output),
            p(&pairs64),
            p(&width64),
        ];
        let function = if bf16 {
            self.kernels.scatter_add_bf16
        } else {
            self.kernels.scatter_add_f32
        };
        // SAFETY: identical lifetime contract to `add`; `output` is pre-zeroed.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
                    pairs64 as c_uint,
                    grid_y,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(scatter_add)",
            )
        }
    }

    /// Embedding gather with F32-encoded ids and an explicit scale:
    /// `ids [tokens]`, `table [vocab, width]` -> `output [tokens, width]`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn embed_gather(
        &self,
        ids: u64,
        table: u64,
        output: u64,
        tokens: usize,
        width: usize,
        scale: f32,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.embed_gather_dispatch(ids, table, output, tokens, width, scale, false, stream)
    }

    /// BF16 variant of `embed_gather`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn embed_gather_bf16(
        &self,
        ids: u64,
        table: u64,
        output: u64,
        tokens: usize,
        width: usize,
        scale: f32,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.embed_gather_dispatch(ids, table, output, tokens, width, scale, true, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn embed_gather_dispatch(
        &self,
        ids: u64,
        table: u64,
        output: u64,
        tokens: usize,
        width: usize,
        scale: f32,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (tokens64, width64) = checked_last_dim(tokens, width)?;
        let grid_y = grid_for(width64, ELEMENTWISE_BLOCK, CUDA_GRID_Y_MAX)?;
        let params = [
            p(&ids),
            p(&table),
            p(&output),
            p(&tokens64),
            p(&width64),
            &scale as *const f32 as *const c_void,
        ];
        let function = if bf16 {
            self.kernels.embed_gather_bf16
        } else {
            self.kernels.embed_gather_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
                    tokens64 as c_uint,
                    grid_y,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(embed_gather)",
            )
        }
    }

    /// Per-row zeroing by F32 flags: `flags [rows]`, `flags[row] == 1.0`
    /// yields zeros for that row.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    pub unsafe fn row_mask(
        &self,
        input: u64,
        flags: u64,
        output: u64,
        rows: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.row_mask_dispatch(input, flags, output, rows, width, false, stream)
    }

    /// BF16 variant of `row_mask`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    pub unsafe fn row_mask_bf16(
        &self,
        input: u64,
        flags: u64,
        output: u64,
        rows: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.row_mask_dispatch(input, flags, output, rows, width, true, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn row_mask_dispatch(
        &self,
        input: u64,
        flags: u64,
        output: u64,
        rows: usize,
        width: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (rows64, width64) = checked_last_dim(rows, width)?;
        let grid_y = grid_for(width64, ELEMENTWISE_BLOCK, CUDA_GRID_Y_MAX)?;
        let params = [p(&input), p(&flags), p(&output), p(&rows64), p(&width64)];
        let function = if bf16 {
            self.kernels.row_mask_bf16
        } else {
            self.kernels.row_mask_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
                    rows64 as c_uint,
                    grid_y,
                    1,
                    ELEMENTWISE_BLOCK,
                    1,
                    1,
                    0,
                    stream.raw,
                    params.as_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(row_mask)",
            )
        }
    }

    /// `(x * scale) - bias` over the last dimension, F32.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scale_bias(
        &self,
        input: u64,
        scale: u64,
        bias: u64,
        output: u64,
        numel: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.scale_bias_dispatch(input, scale, bias, output, numel, width, false, stream)
    }

    /// BF16 variant of `scale_bias`.
    /// # Safety
    /// Every pointer must reference a live, sufficiently sized allocation in
    /// this engine's CUDA context.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn scale_bias_bf16(
        &self,
        input: u64,
        scale: u64,
        bias: u64,
        output: u64,
        numel: usize,
        width: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.scale_bias_dispatch(input, scale, bias, output, numel, width, true, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn scale_bias_dispatch(
        &self,
        input: u64,
        scale: u64,
        bias: u64,
        output: u64,
        numel: usize,
        width: usize,
        bf16: bool,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        self.prepare(stream)?;
        let (count, width64) = checked_last_dim(numel, width)?;
        let grid = grid_for(count, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX)?;
        let params = [
            p(&input),
            p(&scale),
            p(&bias),
            p(&output),
            p(&count),
            p(&width64),
        ];
        let function = if bf16 {
            self.kernels.scale_bias_bf16
        } else {
            self.kernels.scale_bias_f32
        };
        // SAFETY: identical lifetime contract to `add`.
        unsafe {
            check(
                crate::driver::cuLaunchKernel(
                    function,
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
                "cuLaunchKernel(scale_bias)",
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
        for ptx in [EMBEDDED_PTX, EMBEDDED_PTX_BF16, EMBEDDED_PTX_M10] {
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
        for symbol in [
            LINEAR_F32,
            ROTARY_F32,
            ATTENTION_F32,
            SOFT_CAP_F32,
            LINEAR_BF16,
            ROTARY_BF16,
            ATTENTION_BF16,
            SOFT_CAP_BF16,
        ] {
            assert!(EMBEDDED_PTX_M10
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
        assert_eq!(
            grid_for(257, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX).unwrap(),
            2
        );
        assert_eq!(
            grid_for(4096, ELEMENTWISE_BLOCK, CUDA_GRID_X_MAX).unwrap(),
            16
        );
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
