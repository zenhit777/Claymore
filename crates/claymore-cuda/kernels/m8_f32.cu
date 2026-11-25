/*
 * Claymore M8 F32 reference kernels.
 *
 * This file is the source of truth for crates/claymore-cuda/src/kernels/m8_f32.ptx.
 * Regenerate the embedded PTX with (CUDA 13.0 or newer, Blackwell sm_120):
 *
 *   nvcc -arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false \
 *     -o crates/claymore-cuda/src/kernels/m8_f32.ptx \
 *     crates/claymore-cuda/kernels/m8_f32.cu
 *
 * The build itself never invokes nvcc: the generated PTX is checked in and
 * loaded through the CUDA Driver API at runtime. The PTX target sm_120 is
 * native for the RTX 5090 (compute capability 12.0).
 */

#include "claymore_math.cuh"

// The FP32 helpers and the portable `expf`/`expm1f`/`tanhf` live in
// `claymore_math.cuh`, shared with `m10.cu` and `m8_bf16.cu` so the CUDA chain
// has one definition of each.
extern "C" __global__ void claymore_add_f32(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ out,
    uint64_t numel) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    out[index] = a[index] + b[index];
  }
}

extern "C" __global__ void claymore_matmul_f32(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ out,
    uint64_t rows,
    uint64_t inner,
    uint64_t cols) {
  uint64_t row = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }
  float accumulator = 0.0f;
  const float *a_row = a + row * inner;
  for (uint64_t k = 0; k < inner; ++k) {
    accumulator = fmaf(a_row[k], b[k * cols + col], accumulator);
  }
  out[row * cols + col] = accumulator;
}

extern "C" __global__ void claymore_softmax_last_dim_f32(
    const float *__restrict__ input,
    float *__restrict__ out,
    uint64_t rows,
    uint64_t width) {
  extern __shared__ float shared[];
  uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  // The row is reduced on one thread in index order, exactly as the CPU
  // reference reduces a row: max, sum, then a division by that sum. A
  // block-wide tree reduction and a reciprocal multiply each drift one ULP per
  // row, and the BF16 chain amplifies it. `shared` keeps the launch contract.
  claymore_softmax_row(input + row * width, out + row * width, width);
}

extern "C" __global__ void claymore_rms_norm_f32(
    const float *__restrict__ input,
    const float *__restrict__ scale,
    float *__restrict__ out,
    uint64_t rows,
    uint64_t width,
    float epsilon) {
  extern __shared__ float shared[];
  uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float *x = input + row * width;
  float *y = out + row * width;
  unsigned int tid = threadIdx.x;

  // The sum of squares accumulates on one thread in index order: the CPU
  // reference sums a row sequentially, and a tree reduction differs from that
  // order by one ULP in `inv_rms`, which the 2816-term projections amplify.
  if (tid == 0) {
    float local_sum = 0.0f;
    for (uint64_t index = 0; index < width; ++index) {
      local_sum += x[index] * x[index];
    }
    shared[0] = local_sum;
  }
  __syncthreads();
  float inv_rms = 1.0f / sqrtf(shared[0] / (float)width + epsilon);
  for (uint64_t index = tid; index < width; index += blockDim.x) {
    y[index] = x[index] * inv_rms * scale[index];
  }
}
