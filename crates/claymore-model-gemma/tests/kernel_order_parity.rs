//! Kernel-order parity: the CUDA kernels must produce the CPU reference value
//! **bit for bit**, not just within a differential budget.
//!
//! This test holds a Rust transcription of each CUDA kernel's arithmetic (the
//! same expression tree, the same reduction order, the same narrowing points)
//! and asserts bit-exact equality against the CPU reference ops. It runs
//! entirely on the CPU simulator, so the parity contract is enforced in CI
//! without a GPU. If a kernel's order changes, the transcription here must
//! change with it and the equality must still hold; that is the point of the
//! test.
//!
//! The transcendental sections sweep the portable `expf`/`expm1f`/`tanhf` ports.
//! The ports are the **glibc 2.39** algorithms, and glibc 2.39 is the oracle:
//! that is the release the CPU reference of the RTX 5090 campaign ran on
//! (`ldd (Ubuntu GLIBC 2.39-0ubuntu8.12) 2.39`), so it is the release the
//! CUDA-to-CPU bit-parity was measured against. The installed libm of whatever
//! machine runs this test is *not* the oracle: newer glibc releases differ by one
//! ULP at a handful of inputs. Every swept input is therefore asserted against
//! the installed libm, except the inputs enumerated in `PINNED_GLIBC_2_39`, which
//! are asserted against the pinned glibc 2.39 result -- so the sweep stays
//! bit-exact everywhere and a port that drifts fails, on any host.
//! `report_divergence_against_installed_libm` (ignored) prints the full
//! divergence set against the installed libm when a runner image moves.
//!
//! The sweep covers every exponent slot, every branch-boundary neighbourhood,
//! the `k` boundaries of the argument reduction, subnormals and specials.

use claymore_model_gemma::graph::Host;
use claymore_tensor::{diffusion, ops};

fn bits_of(value: f32) -> u32 {
    value.to_bits()
}

fn f32_of_bits(bits: u32) -> f32 {
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Transcription of `claymore_math.cuh`
// ---------------------------------------------------------------------------

const EXP2F_TAB: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];

fn f64_of_bits(bits: u64) -> f64 {
    f64::from_bits(bits)
}

fn cuda_expf(x: f32) -> f32 {
    let bits = x.to_bits();
    let abstop = (bits >> 20) & 0x7ff;
    if abstop >= 0x42b {
        if bits == 0xff800000 {
            return 0.0;
        }
        if abstop >= 0x7f8 {
            return x + x;
        }
        if x > f32_of_bits(0x42b17217) {
            return f32_of_bits(0x7f800000);
        }
        if x < f32_of_bits(0xc2cff1b4) {
            return 0.0;
        }
    }
    let invln = f64_of_bits(0x40471547652b82fe);
    let shift = f64_of_bits(0x4338000000000000);
    let xd = x as f64;
    // The kernel computes these with `fma()`, so the mirror uses the fused
    // form: `mul_add` is Rust's fused multiply-add, not a separate multiply
    // and add.
    let z = invln.mul_add(xd, shift);
    let ki = (z.to_bits() as i64) as u64;
    let kd = z - shift;
    let r = invln.mul_add(xd, -kd);
    // Unsigned wrap-around, as in C: the table entry and the exponent shift
    // are added modulo 2^64.
    let s = f64_of_bits(EXP2F_TAB[(ki % 32) as usize].wrapping_add(ki << 47));
    let a = r.mul_add(
        f64_of_bits(0x3ebc6af84b912394),
        f64_of_bits(0x3f2ebfce50fac4f3),
    );
    let r2 = r * r;
    let y = r.mul_add(f64_of_bits(0x3f962e42ff0c52d6), 1.0);
    (a.mul_add(r2, y) * s) as f32
}

fn cuda_expm1f(mut x: f32) -> f32 {
    let hx0 = x.to_bits() as i32;
    let xsb = (hx0 as u32) & 0x8000_0000;
    let huge = f32_of_bits(0x7149_f2ca);
    let hx = (hx0 as u32) & 0x7fff_ffff;

    if hx >= 0x4195_b844 {
        if hx >= 0x42b1_7218 {
            if hx > 0x7f80_0000 {
                return x + x;
            }
            if hx == 0x7f80_0000 {
                return if xsb == 0 { x } else { -1.0 };
            }
            if x > f32_of_bits(0x42b1_7180) {
                return huge * huge;
            }
        }
        if xsb != 0 {
            return f32_of_bits(0x0da2_4260) - 1.0;
        }
    }

    // glibc leaves `c` undefined when the argument reduction is skipped; the
    // `k == 0` path returns before reading it. Rust needs an explicit value.
    let mut c = 0.0f32;
    let k: i32;
    if hx > 0x3eb1_7218 {
        let (hi, lo, kk) = if hx < 0x3f85_1592 {
            if xsb == 0 {
                (x - f32_of_bits(0x3f31_7180), f32_of_bits(0x3717_f7d1), 1)
            } else {
                (x + f32_of_bits(0x3f31_7180), -f32_of_bits(0x3717_f7d1), -1)
            }
        } else {
            let kk = (f32_of_bits(0x3fb8_aa3b) * x + (if xsb == 0 { 0.5 } else { -0.5 })) as i32;
            let tk = kk as f32;
            (
                x - tk * f32_of_bits(0x3f31_7180),
                tk * f32_of_bits(0x3717_f7d1),
                kk,
            )
        };
        x = hi - lo;
        c = (hi - x) - lo;
        k = kk;
    } else if hx < 0x3300_0000 {
        let t = huge + x;
        return x - (t - (huge + x));
    } else {
        k = 0;
    }

    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0
        + hxs
            * (f32_of_bits(0xbd08_8889)
                + hxs
                    * (f32_of_bits(0x3ad0_0d01)
                        + hxs
                            * (f32_of_bits(0xb8a6_70cd)
                                + hxs
                                    * (f32_of_bits(0x3686_7e54)
                                        + hxs * f32_of_bits(0xb457_edbb)))));
    let mut t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs);
    }
    e = x * (e - c) - c;
    e -= hxs;
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return 1.0 + 2.0 * (x - e);
    }
    let mut y;
    if k <= -2 || k > 56 {
        y = 1.0 - (e - x);
        y = f32_of_bits((y.to_bits() as i32).wrapping_add(k << 23) as u32);
        return y - 1.0;
    }
    if k < 23 {
        t = f32_of_bits(0x3f80_0000u32 - (0x0100_0000_u32 >> k));
        y = t - (e - x);
        y = f32_of_bits((y.to_bits() as i32).wrapping_add(k << 23) as u32);
    } else {
        t = f32_of_bits(((0x7f - k) << 23) as u32);
        y = x - (e + t);
        y += 1.0;
        y = f32_of_bits((y.to_bits() as i32).wrapping_add(k << 23) as u32);
    }
    y
}

fn cuda_tanhf(x: f32) -> f32 {
    let jx = x.to_bits() as i32;
    let ix = (jx as u32) & 0x7fff_ffff;
    let z;
    if ix >= 0x7f80_0000 {
        if jx >= 0 {
            return 1.0 / x + 1.0;
        }
        return 1.0 / x - 1.0;
    }
    if ix < 0x41b0_0000 {
        if ix == 0 {
            return x;
        }
        if ix < 0x2400_0000 {
            return x * (1.0 + x);
        }
        if ix >= 0x3f80_0000 {
            let t = cuda_expm1f(2.0 * x.abs());
            z = 1.0 - 2.0 / (t + 2.0);
        } else {
            let t = cuda_expm1f(-2.0 * x.abs());
            z = -t / (t + 2.0);
        }
    } else {
        z = 1.0 - f32_of_bits(0x0da2_4260);
    }
    if jx >= 0 {
        z
    } else {
        -z
    }
}

/// `claymore_softmax_row`: max and sum reduced in index order on one thread,
/// then a division by that sum.
fn cuda_softmax_row(scores: &[f32], out: &mut [f32]) {
    let width = scores.len();
    let mut max_value = f32::NEG_INFINITY;
    let mut has_nan = false;
    for score in scores {
        if score.is_nan() {
            has_nan = true;
            break;
        }
        max_value = max_value.max(*score);
    }
    if has_nan || max_value == f32::INFINITY {
        out.fill(f32::NAN);
        return;
    }
    if max_value == f32::NEG_INFINITY {
        out.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for i in 0..width {
        let value = cuda_expf(scores[i] - max_value);
        out[i] = value;
        sum += value;
    }
    if !(sum > 0.0 && sum.is_finite()) {
        out.fill(f32::NAN);
        return;
    }
    for value in out.iter_mut() {
        *value /= sum;
    }
}

/// `claymore_rms_norm_f32`: sequential sum of squares on one thread.
fn cuda_rms_norm_f32(input: &[f32], scale: &[f32], width: usize, epsilon: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for row in 0..input.len() / width {
        let x = &input[row * width..(row + 1) * width];
        let mut local_sum = 0.0f32;
        for value in x {
            local_sum += value * value;
        }
        let inv_rms = 1.0 / (local_sum / width as f32 + epsilon).sqrt();
        for (i, value) in x.iter().enumerate() {
            out[row * width + i] = value * inv_rms * scale[i];
        }
    }
    out
}

/// `claymore_scatter_add_f32`: each destination row accumulated in pair order,
/// FP32, narrowed once.
fn cuda_scatter_add_f32(
    source: &[f32],
    weights: &[f32],
    rows: &[f32],
    out_rows: usize,
    width: usize,
) -> Vec<f32> {
    let pairs = weights.len();
    let mut out = vec![0.0f32; out_rows * width];
    for row in 0..out_rows {
        for column in 0..width {
            let mut accumulator = 0.0f32;
            for pair in 0..pairs {
                if rows[pair] as usize == row {
                    accumulator += weights[pair] * source[pair * width + column];
                }
            }
            out[row * width + column] = accumulator;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tensor helpers on the CPU simulator
// ---------------------------------------------------------------------------

fn tensor_of(host: &Host, shape: Vec<usize>, values: &[f32]) -> claymore_tensor::Tensor {
    host.f32(shape, values)
}

fn values_of(host: &Host, tensor: &claymore_tensor::Tensor) -> Vec<f32> {
    host.to_f32(tensor)
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*state >> 32) as u32;
    let mantissa = bits & 0x0007_ffff;
    let exponent = 120 + (bits >> 20) % 20;
    f32::from_bits((exponent << 23) | (mantissa << 1)) - 4.0
}

// ---------------------------------------------------------------------------
// Transcendental parity against the pinned glibc 2.39 oracle
// ---------------------------------------------------------------------------

/// Inputs where the **installed** libm is measured to differ from glibc 2.39,
/// whose algorithms these ports implement. Columns: function, input bits,
/// glibc 2.39 result bits -- i.e. what the port must keep producing, which is
/// asserted here rather than compared to the host.
///
/// This is a measured set, not a tolerance. On the GitHub runner image
/// `ubuntu-26.04` (run `45533731837`, 2026-11-25T02:32:54Z, jobs
/// `138935118950` stable and `138935119267` MSRV), these two inputs are the
/// **only** divergences of the whole swept set: every other swept input
/// satisfied port == installed libm == glibc 2.39 there. Any new divergence,
/// anywhere else in the sweep, still fails the test on any host.
const PINNED_GLIBC_2_39: [(&str, u32, u32); 2] = [
    // expm1f(2^-23): port returns the argument exactly; glibc >= 2.4x returns
    // one ULP higher.
    ("expm1f", 0x3400_0000, 0x3400_0000),
    // tanhf(2^-7): glibc >= 2.4x returns one ULP higher.
    ("tanhf", 0x3c00_0000, 0x3bff_feaa),
];

fn pinned_glibc_2_39(name: &str, bits: u32) -> Option<u32> {
    PINNED_GLIBC_2_39
        .iter()
        .find(|(entry, input, _)| *entry == name && *input == bits)
        .map(|(_, _, expected)| *expected)
}

/// Swept inputs for `expf`: every exponent slot, the overflow/underflow window
/// around log 2^128 and log 2^-150, and the branch-boundary neighbourhoods.
fn expf_samples() -> Vec<u32> {
    let mut samples: Vec<u32> = Vec::new();
    for e in 0u32..=127 {
        for m in 0..64u32 {
            let bits = (e << 23) | ((m * 0x40000) & 0x800000);
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    for boundary in [
        0x42b1_7217u32,
        0x42b1_7218,
        0x42b1_7180,
        0x42b1_7181,
        0x7f80_0000,
        0x7f80_0001,
        0xff80_0000,
        0x0000_0000,
        0x8000_0000,
        0x42ce_8ecf,
        0x42ce_8ed0,
    ] {
        for d in -2i64..=2 {
            let bits = (boundary as i64 + d) as u32;
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    // The overflow/underflow window: the log 2^128 and log 2^-150 thresholds sit
    // inside exponent slots 133-134, which the 64-sample exponent grid can step
    // over. Sweep that window at a 1/512 relative pitch.
    for e in 132u32..=135 {
        for m in (0..0x800000u32).step_by(0x1000) {
            let bits = (e << 23) | m;
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    samples
}

#[test]
fn portable_expf_matches_glibc() {
    for bits in expf_samples() {
        let x = f32::from_bits(bits);
        let port = cuda_expf(x);
        let glibc = x.exp();
        assert_eq!(
            bits_of(port),
            bits_of(glibc),
            "expf({x:?}) port={port:?} glibc={glibc:?}"
        );
    }
}

/// Swept inputs for `expm1f`: exponent slots, reduction `k` boundaries, a dense
/// pitch across the reduction range, and the underflow boundaries.
fn expm1f_samples() -> Vec<u32> {
    let mut samples: Vec<u32> = Vec::new();
    for e in 0u32..=127 {
        for m in 0..64u32 {
            let bits = (e << 23) | ((m * 0x40000) & 0x800000);
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    for boundary in [
        0x4195_b844u32,
        0x42b1_7218,
        0x42b1_7180,
        0x42b1_7181,
        0x3eb1_7218,
        0x3eb1_7219,
        0x3f85_1592,
        0x3f85_1591,
        0x3300_0000,
        0x32ff_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x0000_0000,
        0x8000_0000,
    ] {
        for d in -2i64..=4 {
            let bits = (boundary as i64 + d) as u32;
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    // `k` boundaries of the argument reduction.
    for k in [-60i32, -3, -2, -1, 0, 1, 2, 22, 23, 56, 57, 88] {
        let x = (k as f32) / f32_of_bits(0x3fb8_aa3b);
        for d in 0..6u32 {
            samples.push(x.to_bits() + d);
            samples.push((x.to_bits() + d) | 0x8000_0000);
        }
    }
    // Dense pitch over the argument-reduction range: the `k` boundaries of the
    // reduction and the 27*ln2 / 88.72 filters sit at exponent slots 4-133,
    // which the 64-sample grid can step over.
    for e in 4u32..=133 {
        for m in (0..0x800000u32).step_by(0x1000) {
            let bits = (e << 23) | m;
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    samples
}

#[test]
fn portable_expm1f_matches_glibc() {
    let mut pinned = 0usize;
    for bits in expm1f_samples() {
        let x = f32::from_bits(bits);
        let port = cuda_expm1f(x);
        if let Some(expected) = pinned_glibc_2_39("expm1f", bits) {
            pinned += 1;
            assert_eq!(
                bits_of(port),
                expected,
                "expm1f({x:?}) moved off the pinned glibc 2.39 result"
            );
            continue;
        }
        let glibc = x.exp_m1();
        assert_eq!(
            bits_of(port),
            bits_of(glibc),
            "expm1f({x:?}) port={port:?} glibc={glibc:?}"
        );
    }
    assert!(
        pinned >= 1,
        "the pinned glibc 2.39 divergence point for expm1f is no longer swept"
    );
}

/// Swept inputs for `tanhf`: exponent slots down to 2^-60, the saturation and
/// branch boundaries, the specials, and a dense pitch below the 22 saturation.
fn tanhf_samples() -> Vec<u32> {
    let mut samples: Vec<u32> = Vec::new();
    for e in -60i32..=27 {
        for m in 0..64u32 {
            let bits = (((e + 127) as u32) << 23) | ((m * 0x40000) & 0x800000);
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    for boundary in [
        0x41b0_0000u32,
        0x41b0_0001,
        0x3f80_0000,
        0x3f7f_ffff,
        0x2400_0000,
        0x23ff_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x0000_0000,
        0x8000_0000,
    ] {
        for d in -2i64..=4 {
            let bits = (boundary as i64 + d) as u32;
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    for x in [
        100.0f32,
        1e30,
        1e38,
        f32::MAX,
        f32::MIN,
        22.0,
        -22.0,
        21.999999,
        1.0,
        -1.0,
    ] {
        samples.push(x.to_bits());
        samples.push(x.to_bits() | 0x8000_0000);
    }
    // Dense pitch below the 22 saturation: `tanh` feeds `±2|x|` to `expm1f`, so
    // every pitch step here exercises a distinct reduction path.
    for e in 0u32..=27 {
        for m in (0..0x800000u32).step_by(0x1000) {
            let bits = (e << 23) | m;
            samples.push(bits);
            samples.push(bits | 0x8000_0000);
        }
    }
    samples
}

#[test]
fn portable_tanhf_matches_glibc() {
    let mut pinned = 0usize;
    for bits in tanhf_samples() {
        let x = f32::from_bits(bits);
        let port = cuda_tanhf(x);
        if let Some(expected) = pinned_glibc_2_39("tanhf", bits) {
            pinned += 1;
            assert_eq!(
                bits_of(port),
                expected,
                "tanhf({x:?}) moved off the pinned glibc 2.39 result"
            );
            continue;
        }
        let glibc = x.tanh();
        assert_eq!(
            bits_of(port),
            bits_of(glibc),
            "tanhf({x:?}) port={port:?} glibc={glibc:?}"
        );
    }
    assert!(
        pinned >= 1,
        "the pinned glibc 2.39 divergence point for tanhf is no longer swept"
    );
}

/// Diagnostic, not a contract: prints every swept input where the port and the
/// **installed** libm disagree, with counts per function. Run it when the CI
/// image or the host libc moves, publish the output, and only then extend
/// `PINNED_GLIBC_2_39` from what was measured. It prints; it asserts nothing.
#[test]
#[ignore]
fn report_divergence_against_installed_libm() {
    fn sweep(
        name: &str,
        samples: &[u32],
        installed: fn(f32) -> f32,
        port: fn(f32) -> f32,
    ) -> (usize, usize) {
        let mut total = 0usize;
        let mut diverged = 0usize;
        for &bits in samples {
            let x = f32::from_bits(bits);
            let p = port(x);
            let g = installed(x);
            total += 1;
            if bits_of(p) != bits_of(g) {
                diverged += 1;
                eprintln!(
                    "{name}({x:?}) input=0x{bits:08x} port=0x{:08x} installed=0x{:08x}",
                    bits_of(p),
                    bits_of(g)
                );
            }
        }
        (total, diverged)
    }

    let expf = expf_samples();
    let expm1f = expm1f_samples();
    let tanhf = tanhf_samples();
    let mut totals = (0usize, 0usize);
    for (name, samples, installed, port) in [
        (
            "expf",
            &expf,
            f32::exp as fn(f32) -> f32,
            cuda_expf as fn(f32) -> f32,
        ),
        (
            "expm1f",
            &expm1f,
            f32::exp_m1 as fn(f32) -> f32,
            cuda_expm1f as fn(f32) -> f32,
        ),
        (
            "tanhf",
            &tanhf,
            f32::tanh as fn(f32) -> f32,
            cuda_tanhf as fn(f32) -> f32,
        ),
    ] {
        let (total, diverged) = sweep(name, samples, installed, port);
        eprintln!("{name}: {diverged} divergence(s) out of {total} swept inputs");
        totals.0 += total;
        totals.1 += diverged;
    }
    eprintln!(
        "total: {} divergence(s) out of {} swept inputs",
        totals.1, totals.0
    );
}

// ---------------------------------------------------------------------------
// Reduction-order parity against the CPU reference ops
// ---------------------------------------------------------------------------

#[test]
fn rms_norm_kernel_order_matches_the_reference_bit_for_bit() {
    let host = Host::new();
    for width in [2816usize, 1024, 256, 7] {
        let rows = 3;
        let mut state = 0x2345678u64 + width as u64;
        let values: Vec<f32> = (0..rows * width).map(|_| lcg(&mut state)).collect();
        let scale: Vec<f32> = (0..width).map(|_| lcg(&mut state)).collect();
        let input = tensor_of(&host, vec![rows, width], &values);
        let scale_tensor = tensor_of(&host, vec![width], &scale);
        let reference =
            ops::rms_norm(host.device(), &input, &scale_tensor, 1e-6).expect("rms_norm");
        let kernel = cuda_rms_norm_f32(&values, &scale, width, 1e-6);
        let reference_values = values_of(&host, &reference);
        for (i, (a, b)) in reference_values.iter().zip(kernel.iter()).enumerate() {
            assert_eq!(
                bits_of(*a),
                bits_of(*b),
                "rms_norm width={width} row={}/{} reference={a:?} kernel={b:?}",
                i / width,
                i % width
            );
        }
    }
}

#[test]
fn softmax_kernel_order_matches_the_reference_bit_for_bit() {
    let host = Host::new();
    for width in [2816usize, 1024, 256, 16, 7] {
        let rows = 3;
        let mut state = 0x9e3779u64 + width as u64;
        let values: Vec<f32> = (0..rows * width).map(|_| lcg(&mut state) * 40.0).collect();
        let input = tensor_of(&host, vec![rows, width], &values);
        let reference = ops::softmax_last_dim(host.device(), &input).expect("softmax");
        let mut kernel = vec![0.0f32; rows * width];
        for row in 0..rows {
            cuda_softmax_row(
                &values[row * width..(row + 1) * width],
                &mut kernel[row * width..(row + 1) * width],
            );
        }
        let reference_values = values_of(&host, &reference);
        for (i, (a, b)) in reference_values.iter().zip(kernel.iter()).enumerate() {
            assert_eq!(
                bits_of(*a),
                bits_of(*b),
                "softmax width={width} row={}/{} reference={a:?} kernel={b:?}",
                i / width,
                i % width
            );
        }
    }
}

#[test]
fn scatter_add_kernel_order_matches_the_reference_bit_for_bit() {
    let host = Host::new();
    // MoE shapes: pairs = tokens * top_k, destination rows = tokens.
    for (tokens, top_k, width) in [(16usize, 2, 2816), (7, 8, 1024), (3, 1, 256)] {
        let pairs = tokens * top_k;
        let mut state = 0x12345u64 + width as u64;
        let source: Vec<f32> = (0..pairs * width).map(|_| lcg(&mut state)).collect();
        let weights: Vec<f32> = (0..pairs).map(|_| lcg(&mut state)).collect();
        let rows: Vec<f32> = (0..pairs).map(|p| (p / top_k) as f32).collect();
        let source_tensor = tensor_of(&host, vec![pairs, width], &source);
        let weights_tensor = tensor_of(&host, vec![pairs], &weights);
        let rows_tensor = tensor_of(&host, vec![pairs], &rows);
        let reference = diffusion::scatter_add(
            host.device(),
            &source_tensor,
            &weights_tensor,
            &rows_tensor,
            tokens,
        )
        .expect("scatter_add");
        let kernel = cuda_scatter_add_f32(&source, &weights, &rows, tokens, width);
        let reference_values = values_of(&host, &reference);
        for (i, (a, b)) in reference_values.iter().zip(kernel.iter()).enumerate() {
            assert_eq!(
                bits_of(*a),
                bits_of(*b),
                "scatter_add tokens={tokens} top_k={top_k} width={width} cell {i} \
                 reference={a:?} kernel={b:?}"
            );
        }
    }
}

#[test]
fn masked_rows_keep_the_reference_conventions() {
    // Fully masked row: zeros, not NaN. NaN and -inf inputs: the reference
    // conventions must be reproduced by the kernel transcription.
    let scores = [f32::NEG_INFINITY; 8];
    let mut out = vec![1.0f32; 8];
    cuda_softmax_row(&scores, &mut out);
    assert!(
        out.iter().all(|v| *v == 0.0),
        "fully masked row must be zeros"
    );

    let mut scores = [0.5f32; 8];
    scores[3] = f32::NAN;
    let mut out = vec![0.0f32; 8];
    cuda_softmax_row(&scores, &mut out);
    assert!(out.iter().all(|v| v.is_nan()), "NaN score must propagate");

    let scores = [f32::NEG_INFINITY, 1.0, 2.0, 3.0];
    let mut out = vec![0.0f32; 4];
    cuda_softmax_row(&scores, &mut out);
    // A finite maximum with a -inf entry: the -inf term contributes 0.
    assert_eq!(bits_of(out[0]), bits_of(0.0f32));
    assert!(out[1..].iter().all(|v| v.is_finite()));
}
