/*
 * Claymore M10 diffusion primitive kernels (F32 and BF16).
 *
 * This file is the source of truth for
 * crates/claymore-cuda/src/kernels/m10.ptx. Regenerate with (CUDA 13.0 or
 * newer, Blackwell sm_120):
 *
 *   nvcc -arch=compute_120 -code=compute_120 -O3 -ptx -fmad=false \
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

#include "claymore_math.cuh"

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
    // The upstream `rotate_half` convention reads the table entry of each
    // half independently (the generated tables repeat the frequencies).
    float c2 = cos[token * head_dim + pair + half];
    float s2 = sin[token * head_dim + pair + half];
    dst[pair] = first * c - second * s;
    dst[pair + half] = second * c2 + first * s2;
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
    claymore_softmax_row(scores, probabilities, key_tokens);
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
    out[index] = cap * claymore_tanhf(input[index] / cap);
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

extern "C" __global__ void claymore_scale_scalar_f32(
    const float *__restrict__ input,
    float *__restrict__ out,
    uint64_t numel,
    float factor) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    out[index] = input[index] * factor;
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
    // The upstream `rotate_half` convention reads the table entry of each
    // half independently (the generated tables repeat the frequencies).
    float c2 = to_f32(cos[token * head_dim + pair + half]);
    float s2 = to_f32(sin[token * head_dim + pair + half]);
    dst[pair] = to_bf16_rn(to_bf16_rn(first * c) - to_bf16_rn(second * s));
    dst[pair + half] = to_bf16_rn(to_bf16_rn(second * c2) + to_bf16_rn(first * s2));
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
    float score = to_bf16_rn(accumulator * scale);
    if (masked && mask != nullptr) {
      score = to_bf16_rn(score + mask[token * key_tokens + index]);
    }
    scores[index] = score;
  }
  __syncthreads();
  if (tid == 0) {
    claymore_softmax_row(scores, probabilities, key_tokens);
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
    out[index] = to_bf16_rn(cap * claymore_tanhf(value / cap));
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

// The factor is an FP32 scalar, exactly like the upstream's
// `bf16_tensor * python_float`: the product is evaluated in FP32 and rounded
// once to BF16. The scalar itself is never rounded to BF16.
extern "C" __global__ void claymore_scale_scalar_bf16(
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ out,
    uint64_t numel,
    float factor) {
  uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < numel; index += stride) {
    out[index] = to_bf16_rn(to_f32(input[index]) * factor);
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

// ---------------------------------------------------------------------------
// ATen-compatible top-k selection
// ---------------------------------------------------------------------------
//
// The pinned upstream selects its experts with `torch.topk(largest=True,
// sorted=True)` on the FP32 router probabilities. PyTorch's CPU selection is
// unstable inside a group of experts with identical probabilities, and the
// pinned checkpoint contains such tie groups at the top-k boundary, where the
// choice changes the model's weighted output. The helpers below reproduce
// ATen's algorithm exactly (`std::nth_element` / introselect followed by
// `std::sort` / introsort, with ATen's value-only comparator), one step at a
// time, so the device and the CPU reference select the same experts in the
// same order. The Rust mirror lives in `claymore-tensor/src/aten_topk.rs`.

__device__ __forceinline__ bool aten_less(float left, float right) {
  return (isnan(left) && !isnan(right)) || (left > right);
}

__device__ __forceinline__ void aten_push_heap(float *values, uint32_t *indices,
                                               int hole, int top, float value,
                                               uint32_t index) {
  int parent = (hole - 1) / 2;
  while (hole > top && aten_less(values[parent], value)) {
    values[hole] = values[parent];
    indices[hole] = indices[parent];
    hole = parent;
    parent = (hole - 1) / 2;
  }
  values[hole] = value;
  indices[hole] = index;
}

__device__ __forceinline__ void aten_adjust_heap(float *values, uint32_t *indices,
                                                  int hole, int len, float value,
                                                  uint32_t index) {
  int top = hole;
  int second_child = hole;
  while (second_child < (len - 1) / 2) {
    second_child = 2 * (second_child + 1);
    if (aten_less(values[second_child], values[second_child - 1])) {
      second_child -= 1;
    }
    values[hole] = values[second_child];
    indices[hole] = indices[second_child];
    hole = second_child;
  }
  if ((len & 1) == 0 && second_child == (len - 2) / 2) {
    second_child = 2 * (second_child + 1);
    values[hole] = values[second_child - 1];
    indices[hole] = indices[second_child - 1];
    hole = second_child - 1;
  }
  aten_push_heap(values, indices, hole, top, value, index);
}

__device__ __forceinline__ void aten_make_heap(float *values, uint32_t *indices,
                                                int len) {
  if (len < 2) {
    return;
  }
  int parent = (len - 2) / 2;
  while (true) {
    float value = values[parent];
    uint32_t index = indices[parent];
    aten_adjust_heap(values, indices, parent, len, value, index);
    if (parent == 0) {
      return;
    }
    parent -= 1;
  }
}

__device__ __forceinline__ void aten_heap_select(float *values, uint32_t *indices,
                                                  int first, int middle, int last) {
  int len = middle - first;
  // The heap lives in the `[first, middle)` window of the shared arrays.
  float *heap_values = values + first;
  uint32_t *heap_indices = indices + first;
  aten_make_heap(heap_values, heap_indices, len);
  for (int index = middle; index < last; ++index) {
    if (aten_less(values[index], heap_values[0])) {
      float value = values[index];
      uint32_t expert = indices[index];
      values[index] = heap_values[0];
      indices[index] = heap_indices[0];
      aten_adjust_heap(heap_values, heap_indices, 0, len, value, expert);
    }
  }
}

__device__ __forceinline__ void aten_partial_sort(float *values, uint32_t *indices,
                                                   int first, int middle, int last) {
  int len = middle - first;
  aten_heap_select(values, indices, first, middle, last);
  float *heap_values = values + first;
  uint32_t *heap_indices = indices + first;
  int length = len;
  while (length > 1) {
    length -= 1;
    float value = heap_values[length];
    uint32_t expert = heap_indices[length];
    heap_values[length] = heap_values[0];
    heap_indices[length] = heap_indices[0];
    aten_adjust_heap(heap_values, heap_indices, 0, length, value, expert);
  }
}

__device__ __forceinline__ void aten_move_median_to_first(float *values,
                                                           uint32_t *indices, int result,
                                                           int a, int b, int c) {
  if (aten_less(values[a], values[b])) {
    if (aten_less(values[b], values[c])) {
      float value = values[result];
      values[result] = values[b];
      values[b] = value;
      uint32_t expert = indices[result];
      indices[result] = indices[b];
      indices[b] = expert;
    } else if (aten_less(values[a], values[c])) {
      float value = values[result];
      values[result] = values[c];
      values[c] = value;
      uint32_t expert = indices[result];
      indices[result] = indices[c];
      indices[c] = expert;
    } else {
      float value = values[result];
      values[result] = values[a];
      values[a] = value;
      uint32_t expert = indices[result];
      indices[result] = indices[a];
      indices[a] = expert;
    }
  } else if (aten_less(values[a], values[c])) {
    float value = values[result];
    values[result] = values[a];
    values[a] = value;
    uint32_t expert = indices[result];
    indices[result] = indices[a];
    indices[a] = expert;
  } else if (aten_less(values[b], values[c])) {
    float value = values[result];
    values[result] = values[c];
    values[c] = value;
    uint32_t expert = indices[result];
    indices[result] = indices[c];
    indices[c] = expert;
  } else {
    float value = values[result];
    values[result] = values[b];
    values[b] = value;
    uint32_t expert = indices[result];
    indices[result] = indices[b];
    indices[b] = expert;
  }
}

__device__ __forceinline__ int aten_unguarded_partition(float *values,
                                                         uint32_t *indices, int first,
                                                         int last, int pivot_index) {
  float pivot = values[pivot_index];
  while (true) {
    while (aten_less(values[first], pivot)) {
      first += 1;
    }
    last -= 1;
    while (aten_less(pivot, values[last])) {
      last -= 1;
    }
    if (!(first < last)) {
      return first;
    }
    float value = values[first];
    values[first] = values[last];
    values[last] = value;
    uint32_t expert = indices[first];
    indices[first] = indices[last];
    indices[last] = expert;
    first += 1;
  }
}

__device__ __forceinline__ int aten_unguarded_partition_pivot(float *values,
                                                               uint32_t *indices, int first,
                                                               int last) {
  int mid = first + (last - first) / 2;
  aten_move_median_to_first(values, indices, first, first + 1, mid, last - 1);
  return aten_unguarded_partition(values, indices, first + 1, last, first);
}

__device__ __forceinline__ void aten_unguarded_linear_insert(float *values,
                                                              uint32_t *indices, int last) {
  float value = values[last];
  uint32_t expert = indices[last];
  int previous = last - 1;
  while (aten_less(value, values[previous])) {
    values[last] = values[previous];
    indices[last] = indices[previous];
    last = previous;
    previous -= 1;
  }
  values[last] = value;
  indices[last] = expert;
}

__device__ __forceinline__ void aten_insertion_sort(float *values, uint32_t *indices,
                                                     int first, int last) {
  if (first == last) {
    return;
  }
  for (int index = first + 1; index < last; ++index) {
    if (aten_less(values[index], values[first])) {
      float value = values[index];
      uint32_t expert = indices[index];
      for (int shift = index; shift > first; --shift) {
        values[shift] = values[shift - 1];
        indices[shift] = indices[shift - 1];
      }
      values[first] = value;
      indices[first] = expert;
    } else {
      aten_unguarded_linear_insert(values, indices, index);
    }
  }
}

__device__ __forceinline__ void aten_unguarded_insertion_sort(float *values,
                                                               uint32_t *indices, int first,
                                                               int last) {
  for (int index = first; index < last; ++index) {
    if (aten_less(values[index], values[first])) {
      float value = values[index];
      uint32_t expert = indices[index];
      for (int shift = index; shift > first; --shift) {
        values[shift] = values[shift - 1];
        indices[shift] = indices[shift - 1];
      }
      values[first] = value;
      indices[first] = expert;
    } else {
      aten_unguarded_linear_insert(values, indices, index);
    }
  }
}

__device__ __forceinline__ void aten_introsort_loop(float *values, uint32_t *indices,
                                                     int first, int last, int depth) {
  while (last - first > 16) {
    if (depth == 0) {
      aten_partial_sort(values, indices, first, last, last);
      return;
    }
    depth -= 1;
    int cut = aten_unguarded_partition_pivot(values, indices, first, last);
    aten_introsort_loop(values, indices, cut, last, depth);
    last = cut;
  }
}

__device__ __forceinline__ void aten_final_insertion_sort(float *values,
                                                           uint32_t *indices, int first,
                                                           int last) {
  if (last - first > 16) {
    aten_insertion_sort(values, indices, first, first + 16);
    aten_unguarded_insertion_sort(values, indices, first + 16, last);
  } else {
    aten_insertion_sort(values, indices, first, last);
  }
}

__device__ __forceinline__ void aten_std_sort(float *values, uint32_t *indices,
                                               int first, int last) {
  if (first != last) {
    int depth = 2 * (31 - __clz(last - first));
    aten_introsort_loop(values, indices, first, last, depth);
    aten_final_insertion_sort(values, indices, first, last);
  }
}

__device__ __forceinline__ void aten_introselect(float *values, uint32_t *indices,
                                                  int first, int nth, int last, int depth) {
  while (last - first > 3) {
    if (depth == 0) {
      // libstdc++ calls `__heap_select(first, nth + 1, last)` and then swaps
      // `first` and `nth` to place the nth largest element in its final
      // position.
      aten_heap_select(values, indices, first, nth + 1, last);
      float value = values[first];
      uint32_t expert = indices[first];
      values[first] = values[nth];
      indices[first] = indices[nth];
      values[nth] = value;
      indices[nth] = expert;
      return;
    }
    depth -= 1;
    int cut = aten_unguarded_partition_pivot(values, indices, first, last);
    if (cut <= nth) {
      first = cut;
    } else {
      last = cut;
    }
  }
  aten_insertion_sort(values, indices, first, last);
}

__device__ __forceinline__ void aten_nth_element(float *values, uint32_t *indices,
                                                  int first, int nth, int last) {
  if (first == last || nth == last) {
    return;
  }
  int depth = 2 * (31 - __clz(last - first));
  aten_introselect(values, indices, first, nth, last, depth);
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
  extern __shared__ char shared_raw[];
  float *probs = reinterpret_cast<float *>(shared_raw);
  uint32_t *indices = reinterpret_cast<uint32_t *>(probs + experts);

  const float *row = scores + token * experts;
  uint64_t tid = threadIdx.x;

  // Block-wide maximum. Every thread used to normalize the row with its own
  // partial maximum, which made the probabilities incomparable and turned the
  // selection into a pure tie-break.
  __shared__ float block_max[256];
  float local_max = -INFINITY;
  for (uint64_t index = tid; index < experts; index += blockDim.x) {
    local_max = fmaxf(local_max, row[index]);
  }
  block_max[tid] = local_max;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      block_max[tid] = fmaxf(block_max[tid], block_max[tid + stride]);
    }
    __syncthreads();
  }
  float max_value = block_max[0];

  if (tid < experts) {
    probs[tid] = claymore_expf(row[tid] - max_value);
    indices[tid] = (uint32_t)tid;
  }
  __syncthreads();

  if (tid == 0) {
    int count = (int)experts;
    int k = (int)top_k;
    // The CPU reference sums the exponentials sequentially in index order; the
    // same order is used here so both sides see identical probabilities.
    float sum = 0.0f;
    for (int index = 0; index < count; ++index) {
      sum += probs[index];
    }
    for (int index = 0; index < count; ++index) {
      probs[index] /= sum;
    }
    // ATen-compatible selection: `std::nth_element` then `std::sort`.
    aten_nth_element(probs, indices, 0, k - 1, count);
    aten_std_sort(probs, indices, 0, k - 1);
    float weight_sum = 0.0f;
    for (int slot = 0; slot < k; ++slot) {
      weight_sum += probs[slot];
    }
    for (int slot = 0; slot < k; ++slot) {
      uint32_t expert = indices[slot];
      float weight = probs[slot] / weight_sum * per_expert_scale[expert];
      out[token * 2 * top_k + slot] = weight;
      out[token * 2 * top_k + top_k + slot] = (float)expert;
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
    out[index] = 0.5f * g * (1.0f + claymore_tanhf(inner)) * up[index];
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
    // The pinned reference materializes gelu(gate) in the activation dtype
    // before multiplying by `up`, then narrows the product once more.
    float activation = to_bf16_rn(0.5f * g * (1.0f + claymore_tanhf(inner)));
    out[index] = to_bf16_rn(activation * to_f32(up[index]));
  }
}

extern "C" __global__ void claymore_gather_rows_f32(
    const float *__restrict__ input,
    const float *__restrict__ indices,
    float *__restrict__ out,
    uint64_t input_rows,
    uint64_t out_rows,
    uint64_t width) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= out_rows || column >= width) {
    return;
  }
  uint64_t source = (uint64_t)indices[row];
  // Refuse out-of-range indices: never read outside the input allocation.
  if (source >= input_rows) {
    return;
  }
  out[row * width + column] = input[source * width + column];
}

extern "C" __global__ void claymore_gather_rows_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const float *__restrict__ indices,
    __nv_bfloat16 *__restrict__ out,
    uint64_t input_rows,
    uint64_t out_rows,
    uint64_t width) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= out_rows || column >= width) {
    return;
  }
  uint64_t source = (uint64_t)indices[row];
  // Refuse out-of-range indices: never read outside the input allocation.
  if (source >= input_rows) {
    return;
  }
  out[row * width + column] = input[source * width + column];
}

// Scatter-add in the CPU reference's order: each destination row is
// accumulated by one thread over the pairs in increasing pair index, in FP32,
// and narrowed once at output. `atomicAdd` arrival order is not deterministic,
// and the BF16 atomic variant rounds at every step, so neither reproduces the
// reference bit for bit. The destination is written in full, so no pre-zero is
// needed.
extern "C" __global__ void claymore_scatter_add_f32(
    const float *__restrict__ source,
    const float *__restrict__ weights,
    const float *__restrict__ rows,
    float *__restrict__ out,
    uint64_t pairs,
    uint64_t width,
    uint64_t out_rows) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= out_rows || column >= width) {
    return;
  }
  float accumulator = 0.0f;
  for (uint64_t pair = 0; pair < pairs; ++pair) {
    if ((uint64_t)rows[pair] == row) {
      accumulator += weights[pair] * source[pair * width + column];
    }
  }
  out[row * width + column] = accumulator;
}

extern "C" __global__ void claymore_scatter_add_bf16(
    const __nv_bfloat16 *__restrict__ source,
    const float *__restrict__ weights,
    const float *__restrict__ rows,
    __nv_bfloat16 *__restrict__ out,
    uint64_t pairs,
    uint64_t width,
    uint64_t out_rows) {
  uint64_t row = blockIdx.x;
  uint64_t column = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= out_rows || column >= width) {
    return;
  }
  float accumulator = 0.0f;
  for (uint64_t pair = 0; pair < pairs; ++pair) {
    if ((uint64_t)rows[pair] == row) {
      accumulator += weights[pair] * to_f32(source[pair * width + column]);
    }
  }
  out[row * width + column] = to_bf16_rn(accumulator);
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
