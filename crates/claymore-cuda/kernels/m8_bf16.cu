/*
 * Claymore M8 BF16 reference kernels.
 *
 * This file is the source of truth for crates/claymore-cuda/src/kernels/m8_bf16.ptx.
 * Regenerate the embedded PTX with (CUDA 13.0 or newer, Blackwell sm_120):
 *
 *   nvcc -arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false \
 *     -o crates/claymore-cuda/src/kernels/m8_bf16.ptx \
 *     crates/claymore-cuda/kernels/m8_bf16.cu
 *
 * Numerical policy (identical to the CPU reference in claymore-tensor):
 *   - BF16 values are widened bit-exactly to FP32 with __bfloat162float;
 *   - all intermediates, including matmul accumulation and every reduction,
 *     are evaluated in FP32;
 *   - only the final result is narrowed with __float2bfloat16_rn, i.e.
 *     round-to-nearest-even;
 *   - no FP16 path, no implicit cast, no host fallback.
 *
 * The build never invokes nvcc: the generated PTX is checked in and loaded
 * through the CUDA Driver API at runtime. PTX target sm_120 is native for the
 * RTX 5090 (compute capability 12.0).
 */

#include "claymore_math.cuh"

extern "C" __global__ void claymore_add_bf16(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    uint64_t numel) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    out[index] = to_bf16_rn(to_f32(a[index]) + to_f32(b[index]));
  }
}

extern "C" __global__ void claymore_matmul_bf16(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    uint64_t rows,
    uint64_t inner,
    uint64_t cols) {
  uint64_t row = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }
  float accumulator = 0.0f;
  const __nv_bfloat16 *a_row = a + row * inner;
  for (uint64_t k = 0; k < inner; ++k) {
    accumulator = fmaf(to_f32(a_row[k]), to_f32(b[k * cols + col]), accumulator);
  }
  out[row * cols + col] = to_bf16_rn(accumulator);
}

extern "C" __global__ void claymore_softmax_last_dim_bf16(
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ out,
    uint64_t rows,
    uint64_t width) {
  extern __shared__ float shared[];
  uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  // Same reduction order as the FP32 kernel and the CPU reference: max and sum
  // over the widened values in index order, then a division by the sum, with
  // the single narrowing to BF16 at output.
  //
  // `claymore_softmax_row_bf16` is a single-thread contract: it writes each
  // probability from the shared scratch state. Only one thread of the block
  // may run it.
  if (threadIdx.x == 0) {
    claymore_softmax_row_bf16(input + row * width, out + row * width, width, shared);
  }
}

extern "C" __global__ void claymore_rms_norm_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ scale,
    __nv_bfloat16 *__restrict__ out,
    uint64_t rows,
    uint64_t width,
    float epsilon) {
  extern __shared__ float shared[];
  uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __nv_bfloat16 *x = input + row * width;
  __nv_bfloat16 *y = out + row * width;
  unsigned int tid = threadIdx.x;

  // Sequential row reduction on one thread, bit-exact against the CPU
  // reference's `sum` over the widened values.
  if (tid == 0) {
    float local_sum = 0.0f;
    for (uint64_t index = 0; index < width; ++index) {
      float value = to_f32(x[index]);
      local_sum += value * value;
    }
    shared[0] = local_sum;
  }
  __syncthreads();
  float inv_rms = 1.0f / sqrtf(shared[0] / (float)width + epsilon);
  for (uint64_t index = tid; index < width; index += blockDim.x) {
    float value = to_f32(x[index]);
    y[index] = to_bf16_rn(value * inv_rms * to_f32(scale[index]));
  }
}
