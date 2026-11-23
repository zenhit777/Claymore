/*
 * Claymore M8 F32 reference kernels.
 *
 * This file is the source of truth for crates/claymore-cuda/src/kernels/m8_f32.ptx.
 * Regenerate the embedded PTX with (CUDA 13.0 or newer, Blackwell sm_120):
 *
 *   nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
 *     -o crates/claymore-cuda/src/kernels/m8_f32.ptx \
 *     crates/claymore-cuda/kernels/m8_f32.cu
 *
 * The build itself never invokes nvcc: the generated PTX is checked in and
 * loaded through the CUDA Driver API at runtime. The PTX target sm_120 is
 * native for the RTX 5090 (compute capability 12.0).
 */

#include <math.h>
#include <stdint.h>

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
  const float *x = input + row * width;
  float *y = out + row * width;
  unsigned int tid = threadIdx.x;

  float local_max = -INFINITY;
  for (uint64_t index = tid; index < width; index += blockDim.x) {
    local_max = fmaxf(local_max, x[index]);
  }
  shared[tid] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  float max_value = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (uint64_t index = tid; index < width; index += blockDim.x) {
    local_sum += expf(x[index] - max_value);
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float inv_sum = 1.0f / shared[0];
  for (uint64_t index = tid; index < width; index += blockDim.x) {
    y[index] = expf(x[index] - max_value) * inv_sum;
  }
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

  float local_sum = 0.0f;
  for (uint64_t index = tid; index < width; index += blockDim.x) {
    local_sum = fmaf(x[index], x[index], local_sum);
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float inv_rms = 1.0f / sqrtf(shared[0] / (float)width + epsilon);
  for (uint64_t index = tid; index < width; index += blockDim.x) {
    y[index] = x[index] * inv_rms * scale[index];
  }
}
