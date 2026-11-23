/*
 * Claymore M10 diffusion primitive kernels (F32 and BF16).
 *
 * This file is the source of truth for
 * crates/claymore-cuda/src/kernels/m10.ptx. Regenerate with (CUDA 13.0 or
 * newer, Blackwell sm_120):
 *
 *   nvcc -arch=compute_120 -code=compute_120 -O3 -ptx \
 *     -o crates/claymore-cuda/src/kernels/m10.ptx \
 *     crates/claymore-cuda/kernels/m10.cu
 *
 * The build never invokes nvcc: the generated PTX is checked in and loaded
 * through the CUDA Driver API at runtime. sm_120 is native for the RTX 5090.
 *
 * Numerical policy (identical to the CPU reference in
 * claymore-tensor/src/diffusion.rs):
 *   - BF16 values are widened bit-exactly to FP32;
 *   - all products, reductions and the softmax are evaluated in FP32;
 *   - only the final result is narrowed with round-to-nearest-even;
 *   - no FP16 path, no implicit cast, no host fallback;
 *   - NaN and Inf scores propagate: no sanitisation is performed.
 *
 * The M10 kernels are intentionally small and shape-explicit. The CPU
 * reference remains the deterministic ground truth; these kernels are the
 * GPU execution path that must be differentially validated against it.
 */

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shared device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float to_f32(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 to_bf16_rn(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ int32_t bfloat_bits(float value) {
  // BF16 round-to-nearest-even, reinterpreted through the documented CUDA
  // helpers: no access to the opaque storage of `__nv_bfloat16`.
  uint32_t bits = __float_as_uint(value);
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t bias = 0x7fffu + lsb;
  uint32_t rounded = (bits + bias) >> 16;
  return (int32_t)(uint16_t)rounded;
}

__device__ __forceinline__ float bits_to_f32(int32_t bits) {
  // Exact BF16 -> FP32 widening (zero low 16 bits).
  return __uint_as_float(((uint32_t)(uint16_t)bits) << 16);
}

// Stable softmax over one row. `scores` is the FP32 score vector of length
// `width` (per row), `out` receives the FP32 probabilities.
//
// A fully masked row (max == -inf) yields zeros instead of NaN, matching the
// CPU reference. NaN or +Inf scores propagate as NaN.
__device__ void softmax_row(const float *scores, float *out, uint64_t width) {
  float max_value = -INFINITY;
  bool has_nan = false;
  for (uint64_t i = 0; i < width; ++i) {
    if (isnan(scores[i])) {
      has_nan = true;
      break;
    }
    max_value = fmaxf(max_value, scores[i]);
  }
  if (has_nan || isinf(max_value)) {
    for (uint64_t i = 0; i < width; ++i) {
      out[i] = NAN;
    }
    return;
  }
  if (max_value == -INFINITY) {
    for (uint64_t i = 0; i < width; ++i) {
      out[i] = 0.0f;
    }
    return;
  }
  float sum = 0.0f;
  for (uint64_t i = 0; i < width; ++i) {
    out[i] = expf(scores[i] - max_value);
    sum += out[i];
  }
  float inv_sum = (sum > 0.0f && isfinite(sum)) ? 1.0f / sum : NAN;
  for (uint64_t i = 0; i < width; ++i) {
    out[i] *= inv_sum;
  }
}

// ---------------------------------------------------------------------------
// F32 kernels
// ---------------------------------------------------------------------------

extern "C" __global__ void claymore_linear_f32(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float *__restrict__ out,
    uint64_t tokens,
    uint64_t in_features,
    uint64_t out_features) {
  uint64_t token = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t out_col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || out_col >= out_features) {
    return;
  }
  float accumulator = 0.0f;
  const float *row = input + token * in_features;
  const float *weight_row = weight + out_col * in_features;
  for (uint64_t k = 0; k < in_features; ++k) {
    accumulator = fmaf(row[k], weight_row[k], accumulator);
  }
  out[token * out_features + out_col] = accumulator;
}

extern "C" __global__ void claymore_rotary_embed_f32(
    const float *__restrict__ input,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    float *__restrict__ out,
    uint64_t tokens,
    uint64_t heads,
    uint64_t head_dim) {
  uint64_t token = blockIdx.y;
  uint64_t head = blockIdx.x;
  if (token >= tokens || head >= heads) {
    return;
  }
  uint64_t half = head_dim / 2;
  uint64_t tid = threadIdx.x;
  const float *in = input + (token * heads + head) * head_dim;
  float *dst = out + (token * heads + head) * head_dim;
  for (uint64_t pair = tid; pair < half; pair += blockDim.x) {
    float first = in[pair];
    float second = in[pair + half];
    float c = cos[token * head_dim + pair];
    float s = sin[token * head_dim + pair];
    dst[pair] = first * c - second * s;
    dst[pair + half] = second * c + first * s;
  }
}

extern "C" __global__ void claymore_attention_f32(
    const float *__restrict__ query,
    const float *__restrict__ key,
    const float *__restrict__ value,
    const float *__restrict__ mask,
    float *__restrict__ out,
    float scale,
    uint64_t query_tokens,
    uint64_t key_tokens,
    uint64_t heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    int32_t masked) {
  uint64_t token = blockIdx.y;
  uint64_t head = blockIdx.x;
  if (token >= query_tokens || head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  float *scores = shared;                        // key_tokens
  float *probabilities = shared + key_tokens;    // key_tokens
  const uint64_t groups = heads / kv_heads;
  const uint64_t kv_head = head / groups;
  const float *q = query + (token * heads + head) * head_dim;
  uint64_t tid = threadIdx.x;

  for (uint64_t index = tid; index < key_tokens; index += blockDim.x) {
    const float *k = key + (index * kv_heads + kv_head) * head_dim;
    float accumulator = 0.0f;
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      accumulator = fmaf(q[dim], k[dim], accumulator);
    }
    float score = accumulator * scale;
    if (masked && mask != nullptr) {
      score += mask[token * key_tokens + index];
    }
    scores[index] = score;
  }
  __syncthreads();
  if (tid == 0) {
    softmax_row(scores, probabilities, key_tokens);
  }
  __syncthreads();
  for (uint64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (uint64_t index = 0; index < key_tokens; ++index) {
      const float *v = value + (index * kv_heads + kv_head) * head_dim;
      accumulator = fmaf(probabilities[index], v[dim], accumulator);
    }
    out[(token * heads + head) * head_dim + dim] = accumulator;
  }
}

extern "C" __global__ void claymore_soft_cap_f32(
    const float *__restrict__ input,
    float *__restrict__ out,
    uint64_t numel,
    float cap) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    out[index] = cap * tanhf(input[index] / cap);
  }
}

extern "C" __global__ void claymore_scale_f32(
    const float *__restrict__ input,
    const float *__restrict__ factor,
    float *__restrict__ out,
    uint64_t numel,
    uint64_t width,
    uint64_t factor_width) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    // A factor of one element broadcasts over the whole tensor; a factor of
    // `width` elements broadcasts over the last dimension.
    uint64_t factor_index = factor_width == 1 ? 0 : (index % width);
    out[index] = input[index] * factor[factor_index];
  }
}

// ---------------------------------------------------------------------------
// BF16 kernels
// ---------------------------------------------------------------------------

extern "C" __global__ void claymore_linear_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    uint64_t tokens,
    uint64_t in_features,
    uint64_t out_features) {
  uint64_t token = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t out_col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || out_col >= out_features) {
    return;
  }
  float accumulator = 0.0f;
  const __nv_bfloat16 *row = input + token * in_features;
  const __nv_bfloat16 *weight_row = weight + out_col * in_features;
  for (uint64_t k = 0; k < in_features; ++k) {
    accumulator = fmaf(to_f32(row[k]), to_f32(weight_row[k]), accumulator);
  }
  out[token * out_features + out_col] = to_bf16_rn(accumulator);
}

extern "C" __global__ void claymore_rotary_embed_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ cos,
    const __nv_bfloat16 *__restrict__ sin,
    __nv_bfloat16 *__restrict__ out,
    uint64_t tokens,
    uint64_t heads,
    uint64_t head_dim) {
  uint64_t token = blockIdx.y;
  uint64_t head = blockIdx.x;
  if (token >= tokens || head >= heads) {
    return;
  }
  uint64_t half = head_dim / 2;
  uint64_t tid = threadIdx.x;
  const __nv_bfloat16 *in = input + (token * heads + head) * head_dim;
  __nv_bfloat16 *dst = out + (token * heads + head) * head_dim;
  for (uint64_t pair = tid; pair < half; pair += blockDim.x) {
    float first = to_f32(in[pair]);
    float second = to_f32(in[pair + half]);
    float c = to_f32(cos[token * head_dim + pair]);
    float s = to_f32(sin[token * head_dim + pair]);
    dst[pair] = to_bf16_rn(first * c - second * s);
    dst[pair + half] = to_bf16_rn(second * c + first * s);
  }
}

extern "C" __global__ void claymore_attention_bf16(
    const __nv_bfloat16 *__restrict__ query,
    const __nv_bfloat16 *__restrict__ key,
    const __nv_bfloat16 *__restrict__ value,
    const float *__restrict__ mask,
    __nv_bfloat16 *__restrict__ out,
    float scale,
    uint64_t query_tokens,
    uint64_t key_tokens,
    uint64_t heads,
    uint64_t kv_heads,
    uint64_t head_dim,
    int32_t masked) {
  uint64_t token = blockIdx.y;
  uint64_t head = blockIdx.x;
  if (token >= query_tokens || head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  float *scores = shared;
  float *probabilities = shared + key_tokens;
  const uint64_t groups = heads / kv_heads;
  const uint64_t kv_head = head / groups;
  const __nv_bfloat16 *q = query + (token * heads + head) * head_dim;
  uint64_t tid = threadIdx.x;

  for (uint64_t index = tid; index < key_tokens; index += blockDim.x) {
    const __nv_bfloat16 *k = key + (index * kv_heads + kv_head) * head_dim;
    float accumulator = 0.0f;
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      accumulator = fmaf(to_f32(q[dim]), to_f32(k[dim]), accumulator);
    }
    float score = accumulator * scale;
    if (masked && mask != nullptr) {
      score += mask[token * key_tokens + index];
    }
    scores[index] = score;
  }
  __syncthreads();
  if (tid == 0) {
    softmax_row(scores, probabilities, key_tokens);
  }
  __syncthreads();
  for (uint64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (uint64_t index = 0; index < key_tokens; ++index) {
      const __nv_bfloat16 *v = value + (index * kv_heads + kv_head) * head_dim;
      // The reference rounds the softmax output to the input dtype before the
      // value reduction: reproduce that explicit narrowing here.
      float probability =
          bits_to_f32(bfloat_bits(probabilities[index]));
      accumulator = fmaf(probability, to_f32(v[dim]), accumulator);
    }
    out[(token * heads + head) * head_dim + dim] = to_bf16_rn(accumulator);
  }
}

extern "C" __global__ void claymore_soft_cap_bf16(
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ out,
    uint64_t numel,
    float cap) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    float value = to_f32(input[index]);
    out[index] = to_bf16_rn(cap * tanhf(value / cap));
  }
}

extern "C" __global__ void claymore_scale_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ factor,
    __nv_bfloat16 *__restrict__ out,
    uint64_t numel,
    uint64_t width,
    uint64_t factor_width) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    uint64_t factor_index = factor_width == 1 ? 0 : (index % width);
    float product = to_f32(input[index]) * to_f32(factor[factor_index]);
    out[index] = to_bf16_rn(product);
  }
}
// ---------------------------------------------------------------------------
// M10.2 kernels: MoE router, gated activation, gather/scatter, embedding,
// row masking and explicit scale-bias. The same numerical policy applies:
// widen BF16 to FP32, accumulate in FP32, round once with RN-even.
// ---------------------------------------------------------------------------

extern "C" __global__ void claymore_widen_bf16_f32(
    const __nv_bfloat16 *__restrict__ input,
    float *__restrict__ out,
    uint64_t numel) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    out[index] = to_f32(input[index]);
  }
}

extern "C" __global__ void claymore_router_top_k_f32(
    const float *__restrict__ scores,
    const float *__restrict__ per_expert_scale,
    float *__restrict__ out,
    uint64_t tokens,
    uint64_t experts,
    uint64_t top_k) {
  uint64_t token = blockIdx.x;
  if (token >= tokens) {
    return;
  }
  extern __shared__ float shared[];
  float *probs = shared;
  const float *row = scores + token * experts;
  uint64_t tid = threadIdx.x;
  float max_value = -INFINITY;
  for (uint64_t index = tid; index < experts; index += blockDim.x) {
    max_value = fmaxf(max_value, row[index]);
  }
  __syncthreads();
  // One thread per expert is guaranteed (blockDim >= experts).
  if (tid < experts) {
    probs[tid] = expf(row[tid] - max_value);
  }
  __syncthreads();
  float sum = 0.0f;
  for (uint64_t index = tid; index < experts; index += blockDim.x) {
    sum += probs[index];
  }
  __shared__ float block_sum[32];
  // Reduce the sum across the block.
  if (tid < 32) {
    float partial = 0.0f;
    for (uint64_t index = tid; index < experts; index += 32) {
      partial += probs[index];
    }
    block_sum[tid] = partial;
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      total += block_sum[lane];
    }
    for (uint64_t index = 0; index < experts; ++index) {
      probs[index] /= total;
    }
    // Deterministic top-k: descending probability, ascending index on ties.
    float weight_sum = 0.0f;
    for (uint64_t slot = 0; slot < top_k; ++slot) {
      uint64_t best = 0;
      float best_value = probs[0];
      for (uint64_t index = 1; index < experts; ++index) {
        if (probs[index] > best_value) {
          best_value = probs[index];
          best = index;
        }
      }
      weight_sum += best_value;
      probs[best] = -INFINITY;
      out[token * 2 * top_k + slot] = best_value;
      out[token * 2 * top_k + top_k + slot] = (float)best;
    }
    for (uint64_t slot = 0; slot < top_k; ++slot) {
      float weight = out[token * 2 * top_k + slot] / weight_sum;
      uint64_t expert = (uint64_t)out[token * 2 * top_k + top_k + slot];
      out[token * 2 * top_k + slot] = weight * per_expert_scale[expert];
    }
  }
}

extern "C" __global__ void claymore_gelu_mul_up_f32(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    float *__restrict__ out,
    uint64_t numel) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  const float k = sqrtf(2.0f / 3.14159265358979323846f);
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    float g = gate[index];
    float inner = k * (g + 0.044715f * g * g * g);
    out[index] = 0.5f * g * (1.0f + tanhf(inner)) * up[index];
  }
}

extern "C" __global__ void claymore_gelu_mul_up_bf16(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    uint64_t numel) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  const float k = sqrtf(2.0f / 3.14159265358979323846f);
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    float g = to_f32(gate[index]);
    float inner = k * (g + 0.044715f * g * g * g);
    out[index] = to_bf16_rn(0.5f * g * (1.0f + tanhf(inner)) * to_f32(up[index]));
  }
}

extern "C" __global__ void claymore_gather_rows_f32(
    const float *__restrict__ input,
    const float *__restrict__ indices,
    float *__restrict__ out,
    uint64_t out_rows,
    uint64_t width) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= out_rows || column >= width) {
    return;
  }
  uint64_t source = (uint64_t)indices[row];
  out[row * width + column] = input[source * width + column];
}

extern "C" __global__ void claymore_gather_rows_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const float *__restrict__ indices,
    __nv_bfloat16 *__restrict__ out,
    uint64_t out_rows,
    uint64_t width) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= out_rows || column >= width) {
    return;
  }
  uint64_t source = (uint64_t)indices[row];
  out[row * width + column] = input[source * width + column];
}

extern "C" __global__ void claymore_scatter_add_f32(
    const float *__restrict__ source,
    const float *__restrict__ weights,
    const float *__restrict__ rows,
    float *__restrict__ out,
    uint64_t pairs,
    uint64_t width) {
  uint64_t pair = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (pair >= pairs || column >= width) {
    return;
  }
  uint64_t row = (uint64_t)rows[pair];
  atomicAdd(&out[row * width + column], weights[pair] * source[pair * width + column]);
}

extern "C" __global__ void claymore_scatter_add_bf16(
    const __nv_bfloat16 *__restrict__ source,
    const float *__restrict__ weights,
    const float *__restrict__ rows,
    __nv_bfloat16 *__restrict__ out,
    uint64_t pairs,
    uint64_t width) {
  uint64_t pair = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (pair >= pairs || column >= width) {
    return;
  }
  uint64_t row = (uint64_t)rows[pair];
  // BF16 atomic add is rounded at every accumulation step, matching the
  // pinned upstream bf16 index_add_ CUDA behaviour. Differential tolerance is
  // a model-derived bound, not a ULP budget, because the accumulation order is
  // not deterministic.
  __nv_bfloat16 value = to_bf16_rn(weights[pair] * to_f32(source[pair * width + column]));
  atomicAdd(&out[row * width + column], value);
}

extern "C" __global__ void claymore_embed_gather_f32(
    const float *__restrict__ ids,
    const float *__restrict__ table,
    float *__restrict__ out,
    uint64_t tokens,
    uint64_t width,
    float scale) {
  uint64_t token = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= tokens || column >= width) {
    return;
  }
  uint64_t row = (uint64_t)ids[token];
  out[token * width + column] = table[row * width + column] * scale;
}

extern "C" __global__ void claymore_embed_gather_bf16(
    const float *__restrict__ ids,
    const __nv_bfloat16 *__restrict__ table,
    __nv_bfloat16 *__restrict__ out,
    uint64_t tokens,
    uint64_t width,
    float scale) {
  uint64_t token = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= tokens || column >= width) {
    return;
  }
  uint64_t row = (uint64_t)ids[token];
  out[token * width + column] = to_bf16_rn(to_f32(table[row * width + column]) * scale);
}

extern "C" __global__ void claymore_row_mask_f32(
    const float *__restrict__ input,
    const float *__restrict__ flags,
    float *__restrict__ out,
    uint64_t rows,
    uint64_t width) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || column >= width) {
    return;
  }
  out[row * width + column] = flags[row] == 1.0f ? 0.0f : input[row * width + column];
}

extern "C" __global__ void claymore_row_mask_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const float *__restrict__ flags,
    __nv_bfloat16 *__restrict__ out,
    uint64_t rows,
    uint64_t width) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || column >= width) {
    return;
  }
  out[row * width + column] =
      flags[row] == 1.0f ? to_bf16_rn(0.0f) : input[row * width + column];
}

extern "C" __global__ void claymore_scale_bias_f32(
    const float *__restrict__ input,
    const float *__restrict__ scale,
    const float *__restrict__ bias,
    float *__restrict__ out,
    uint64_t numel,
    uint64_t width) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    uint64_t column = index % width;
    out[index] = fmaf(input[index], scale[column], -bias[column]);
  }
}

extern "C" __global__ void claymore_scale_bias_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ scale,
    const __nv_bfloat16 *__restrict__ bias,
    __nv_bfloat16 *__restrict__ out,
    uint64_t numel,
    uint64_t width) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    uint64_t column = index % width;
    out[index] = to_bf16_rn(
        fmaf(to_f32(input[index]), to_f32(scale[column]), -to_f32(bias[column])));
  }
}
