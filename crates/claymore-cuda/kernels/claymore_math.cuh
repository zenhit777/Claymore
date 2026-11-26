/*
 * Claymore portable device math.
 *
 * Single source of truth for the FP32 helpers and transcendental functions used
 * by every CUDA kernel file (`m10.cu`, `m8_f32.cu`, `m8_bf16.cu`). The device
 * `expf`, `expm1f` and `tanhf` compile to `ex2.approx.ftz.f32` and
 * `rcp.approx.ftz.f32`, which are not correctly rounded, so the CUDA chain
 * cannot reproduce the CPU reference (glibc `expf`/`expm1f`/`tanhf`, reached
 * through Rust `f32::exp` and `f32::tanh`) bit for bit. Soft capping, GELU and
 * the softmax sit in the same amplification chain, so the kernels must produce
 * the CPU value bit for bit.
 *
 * These ports transcribe the glibc 2.39 implementations installed on the
 * target:
 *   - `sysdeps/ieee754/flt-32/e_expf.c` (EXP2F_TABLE_BITS = 5,
 *     EXP2F_POLY_ORDER = 3),
 *   - `sysdeps/ieee754/flt-32/s_expm1f.c`,
 *   - `sysdeps/ieee754/flt-32/s_tanhf.c`.
 * Constants are written as integer bit patterns and reinterpreted, so each one
 * is exactly the value read out of the installed libm.
 *
 * The ports rely on separate multiplies and adds: build every PTX with
 * `-fmad=false` (see the regeneration command in each kernel file header). The
 * Rust mirror in `crates/claymore-model-gemma/tests/kernel_order_parity.rs`
 * sweeps these ports against glibc on the CPU.
 */

#ifndef CLAYMORE_MATH_CUH
#define CLAYMORE_MATH_CUH

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

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

// glibc's EXP2F table (5 bits), as double bit patterns.
__constant__ unsigned long long claymore_exp2f_tab[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
    0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
    0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
    0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
    0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
    0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL};

__device__ __forceinline__ double claymore_bits_to_double(unsigned long long bits) {
  return __longlong_as_double(bits);
}

#define CLAYMORE_F32(bits) __int_as_float(bits)

__device__ __forceinline__ float claymore_expf(float x) {
  // glibc filters on `top12(x) & 0x7ff`, the top 12 bits of the FP32 word:
  // 0x42b is the slot of 88.0f (|x| >= 88), 0x7f8 the slot of infinity. The
  // overflow threshold is `x > 0x1.62e42ep6f` (log 2^128, 0x42b17217), the
  // underflow threshold `x < -0x1.9fe368p6f` (log 2^-150, 0xc2cff1b4).
  const unsigned int bits = __float_as_uint(x);
  const unsigned int abstop = (bits >> 20) & 0x7ffu;
  if (abstop >= 0x42bu) {
    if (bits == 0xff800000u) return 0.0f;
    if (abstop >= 0x7f8u) return x + x;
    if (x > __uint_as_float(0x42b17217u)) return __uint_as_float(0x7f800000u);
    if (x < __uint_as_float(0xc2cff1b4u)) return 0.0f;
  }
  const double invln = claymore_bits_to_double(0x40471547652b82feULL);
  const double shift = claymore_bits_to_double(0x4338000000000000ULL);
  const double xd = (double)x;
  const double z = fma(invln, xd, shift);
  const unsigned long long ki = (unsigned long long)__double_as_longlong(z);
  const double kd = z - shift;
  const double r = fma(invln, xd, -kd);
  const double s =
      claymore_bits_to_double(claymore_exp2f_tab[ki % 32] + (ki << 47));
  const double a = fma(r, claymore_bits_to_double(0x3ebc6af84b912394ULL),
                       claymore_bits_to_double(0x3f2ebfce50fac4f3ULL));
  const double r2 = r * r;
  const double y = fma(r, claymore_bits_to_double(0x3f962e42ff0c52d6ULL), 1.0);
  return (float)(fma(a, r2, y) * s);
}

__device__ __forceinline__ float claymore_expm1f(float x) {
  const int32_t hx0 = __float_as_int(x);
  const uint32_t xsb = (uint32_t)hx0 & 0x80000000u;
  const float huge = CLAYMORE_F32(0x7149f2ca); /* 1.0e+30 */
  float y = (xsb == 0) ? x : -x;
  uint32_t hx = (uint32_t)hx0 & 0x7fffffffu;
  float hi, lo, c, t, e, hxs, hfx, r1;
  int32_t k;

  if (hx >= 0x4195b844u) { /* |x| >= 27*ln2 */
    if (hx >= 0x42b17218u) { /* |x| >= 88.721... */
      if (hx > 0x7f800000u) return x + x; /* NaN */
      if (hx == 0x7f800000u) return (xsb == 0) ? x : -1.0f;
      if (x > CLAYMORE_F32(0x42b17180)) return huge * huge; /* overflow */
    }
    if (xsb != 0) {
      return CLAYMORE_F32(0x0da24260) - 1.0f; /* x < -27*ln2: -1 */
    }
  }

  if (hx > 0x3eb17218u) { /* |x| > 0.5*ln2 */
    if (hx < 0x3f851592u) { /* |x| < 1.5*ln2 */
      if (xsb == 0) {
        hi = x - CLAYMORE_F32(0x3f317180);
        lo = CLAYMORE_F32(0x3717f7d1);
        k = 1;
      } else {
        hi = x + CLAYMORE_F32(0x3f317180);
        lo = -CLAYMORE_F32(0x3717f7d1);
        k = -1;
      }
    } else {
      k = (int32_t)(CLAYMORE_F32(0x3fb8aa3b) * x + ((xsb == 0) ? 0.5f : -0.5f));
      t = (float)k;
      hi = x - t * CLAYMORE_F32(0x3f317180);
      lo = t * CLAYMORE_F32(0x3717f7d1);
    }
    x = hi - lo;
    c = (hi - x) - lo;
  } else if (hx < 0x33000000u) { /* |x| < 2^-25: return x */
    t = huge + x;
    return x - (t - (huge + x));
  } else {
    k = 0;
  }

  hfx = 0.5f * x;
  hxs = x * hfx;
  r1 = 1.0f + hxs * (CLAYMORE_F32(0xbd088889) + hxs * (CLAYMORE_F32(0x3ad00d01)
       + hxs * (CLAYMORE_F32(0xb8a670cd) + hxs * (CLAYMORE_F32(0x36867e54)
       + hxs * CLAYMORE_F32(0xb457edbb)))));
  t = 3.0f - r1 * hfx;
  e = hxs * ((r1 - t) / (6.0f - x * t));
  if (k == 0) {
    return x - (x * e - hxs);
  }
  e = (x * (e - c) - c);
  e -= hxs;
  if (k == -1) {
    return 0.5f * (x - e) - 0.5f;
  }
  if (k == 1) {
    if (x < -0.25f) {
      return -2.0f * (e - (x + 0.5f));
    }
    return 1.0f + 2.0f * (x - e);
  }
  if (k <= -2 || k > 56) {
    y = 1.0f - (e - x);
    y = __int_as_float(__float_as_int(y) + (k << 23));
    return y - 1.0f;
  }
  if (k < 23) {
    t = __int_as_float(0x3f800000 - (0x1000000 >> k)); /* t = 1 - 2^-k */
    y = t - (e - x);
    y = __int_as_float(__float_as_int(y) + (k << 23));
  } else {
    t = __int_as_float((0x7f - k) << 23); /* t = 2^-k */
    y = x - (e + t);
    y += 1.0f;
    y = __int_as_float(__float_as_int(y) + (k << 23));
  }
  return y;
}

__device__ __forceinline__ float claymore_tanhf(float x) {
  const int32_t jx = __float_as_int(x);
  const uint32_t ix = (uint32_t)jx & 0x7fffffffu;
  float z;
  if (ix >= 0x7f800000u) { /* inf or NaN */
    if (jx >= 0) return 1.0f / x + 1.0f;
    return 1.0f / x - 1.0f;
  }
  if (ix < 0x41b00000u) { /* |x| < 22 */
    if (ix == 0) return x;
    if (ix < 0x24000000u) return x * (1.0f + x); /* |x| < 2^-55 */
    if (ix >= 0x3f800000u) { /* |x| >= 1 */
      const float t = claymore_expm1f(2.0f * fabsf(x));
      z = 1.0f - 2.0f / (t + 2.0f);
    } else {
      const float t = claymore_expm1f(-2.0f * fabsf(x));
      z = -t / (t + 2.0f);
    }
  } else {
    z = 1.0f - CLAYMORE_F32(0x0da24260); /* 1 - 1e-30 */
  }
  return (jx >= 0) ? z : -z;
}

// Stable softmax over one row, in the CPU reference's order: the maximum and the
// sum are reduced in index order on a single thread, and each probability is a
// division by that sum. A block-wide tree reduction and a reciprocal multiply
// differ from the reference by one ULP per row, which the chain amplifies.
__device__ __forceinline__ void claymore_softmax_row(const float *scores,
                                                     float *out,
                                                     uint64_t width) {
  float max_value = -INFINITY;
  bool has_nan = false;
  for (uint64_t i = 0; i < width; ++i) {
    if (isnan(scores[i])) {
      has_nan = true;
      break;
    }
    max_value = fmaxf(max_value, scores[i]);
  }
  if (has_nan || max_value == INFINITY) {
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
    out[i] = claymore_expf(scores[i] - max_value);
    sum += out[i];
  }
  if (!(sum > 0.0f && isfinite(sum))) {
    for (uint64_t i = 0; i < width; ++i) {
      out[i] = NAN;
    }
    return;
  }
  for (uint64_t i = 0; i < width; ++i) {
    out[i] /= sum;
  }
}

// BF16 row softmax with the same reduction order as the CPU reference: the
// maximum and the sum run in index order on one thread over the widened FP32
// values, and each probability is the division of its exponential by that sum,
// narrowed to BF16 once.
//
// `state` is a three-slot scratch buffer: [max, sum, flag] with flag 0 =
// normal, 1 = all zeros (fully masked row), 2 = all NaN.
__device__ __forceinline__ void claymore_softmax_row_bf16(const __nv_bfloat16 *scores,
                                                           __nv_bfloat16 *out,
                                                           uint64_t width,
                                                           float *state) {
  float max_value = -INFINITY;
  bool has_nan = false;
  for (uint64_t i = 0; i < width; ++i) {
    const float value = to_f32(scores[i]);
    if (isnan(value)) {
      has_nan = true;
      break;
    }
    max_value = fmaxf(max_value, value);
  }
  float sum = 0.0f;
  int flag = 0;
  if (has_nan || max_value == INFINITY) {
    flag = 2;
  } else if (max_value == -INFINITY) {
    flag = 1;
  } else {
    for (uint64_t i = 0; i < width; ++i) {
      sum += claymore_expf(to_f32(scores[i]) - max_value);
    }
    if (!(sum > 0.0f && isfinite(sum))) {
      flag = 2;
    }
  }
  state[0] = max_value;
  state[1] = sum;
  state[2] = (float)flag;
  const __nv_bfloat16 fill = (flag == 1) ? to_bf16_rn(0.0f) : to_bf16_rn(NAN);
  for (uint64_t i = 0; i < width; ++i) {
    out[i] = (flag != 0)
                 ? fill
                 : to_bf16_rn(claymore_expf(to_f32(scores[i]) - max_value) / sum);
  }
}

#endif  // CLAYMORE_MATH_CUH
