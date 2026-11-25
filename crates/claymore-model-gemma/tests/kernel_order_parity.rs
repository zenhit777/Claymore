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
//! machine runs this test is *not* the oracle. Measured on the runner image
//! `ubuntu-26.04` (runs `45533731837` and `45538600058`, 2026-11-25), that libc
//! differs from glibc 2.39 across the sweep -- `expm1f(1.0)`, `expm1f(2^-23)`,
//! `tanhf(±2^-7)` and more -- so asserting against the host there is not a
//! parity contract, it is a moving target.
//!
//! The contract is therefore pinned to the oracle itself. Tier one, on every
//! host: the port must keep producing the glibc 2.39 results committed in
//! `tests/data/glibc_2_39_*.bin` (`regenerate_glibc_2_39_reference_tables`,
//! ignored, writes them and refuses any host whose libm disagrees with the
//! `SENTINELS_GLIBC_2_39` probe). Tier two, only when the installed libm passes
//! that probe -- i.e. only on a glibc 2.39 host: the port must also equal the
//! installed libm on **every swept input**, pinned or not, which is what keeps
//! the dense pitch under contract where the tables do not reach. On a host that
//! is not the oracle the tier-two comparison is skipped with a line on stdout,
//! and `report_divergence_against_installed_libm` (ignored) prints its full
//! divergence set for publication.
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

/// Sentinel inputs that identify the oracle: these are swept points where a
/// newer glibc (measured: the runner image `ubuntu-26.04`) differs from glibc
/// 2.39, whose algorithms these ports implement. Columns: function, input bits,
/// glibc 2.39 result bits.
///
/// They are a **host probe**, never a tolerance. If the installed libm returns
/// anything else at these points, that libm is not the oracle and the tests
/// stop comparing against it (the pinned reference table is still asserted in
/// full, on every host) and say so on stdout.
const SENTINELS_GLIBC_2_39: [(&str, u32, u32); 2] = [
    // expm1f(2^-23): the port and glibc 2.39 return the argument exactly;
    // glibc >= 2.4x returns one ULP higher.
    ("expm1f", 0x3400_0000, 0x3400_0000),
    // tanhf(2^-7): glibc >= 2.4x returns one ULP higher.
    ("tanhf", 0x3c00_0000, 0x3bff_feaa),
];

/// True when the installed libm agrees with glibc 2.39 on every sentinel.
fn host_is_pinned_glibc_2_39() -> bool {
    let probes = [
        ("expm1f", f32::exp_m1 as fn(f32) -> f32),
        ("tanhf", f32::tanh as fn(f32) -> f32),
    ];
    SENTINELS_GLIBC_2_39.iter().all(|(name, input, expected)| {
        let probe = probes
            .iter()
            .find(|(probe, _)| probe == name)
            .expect("a sentinel without an installed-libm probe")
            .1;
        bits_of(probe(f32_of_bits(*input))) == *expected
    })
}

/// Reference results of glibc 2.39, one file per function: magic `CMP1`, the
/// entry count as a little-endian `u32`, then that many little-endian
/// `(input bits, result bits)` pairs sorted by input bits. Written on the
/// glibc 2.39 reference host (the one that produced the golden inputs and the
/// GPU campaign) by `regenerate_glibc_2_39_reference_tables`.
const REFERENCE_GLIBC_2_39_EXPF: &[u8] = include_bytes!("data/glibc_2_39_expf.bin");
const REFERENCE_GLIBC_2_39_EXPM1F: &[u8] = include_bytes!("data/glibc_2_39_expm1f.bin");
const REFERENCE_GLIBC_2_39_TANHF: &[u8] = include_bytes!("data/glibc_2_39_tanhf.bin");

/// Decimation of the dense pitch held in the reference tables: the sweeps step
/// the mantissa by `0x1000`, the tables keep every 16th line (`0x10000`), which
/// still samples the reduction range more finely than the structural grids.
/// Structural inputs are always held: the grids build their mantissas as
/// multiples of `0x40000`, which is a multiple of `0x10000`.
const PINNED_PITCH: u32 = 0x1_0000;

/// Swept inputs, split into everything swept and what the reference table must
/// cover.
#[derive(Default)]
struct Sweep {
    all: Vec<u32>,
    pinned: Vec<u32>,
}

impl Sweep {
    /// A structural input, or a dense line whose mantissa is held: it joins the
    /// reference table.
    fn push(&mut self, bits: u32) {
        self.push_at(bits, true);
    }

    /// A dense-pitch line: swept always, held every `PINNED_PITCH`.
    fn push_dense(&mut self, bits: u32, mantissa: u32) {
        self.push_at(bits, mantissa % PINNED_PITCH == 0);
    }

    fn push_at(&mut self, bits: u32, held: bool) {
        self.all.push(bits);
        if held {
            self.pinned.push(bits);
        }
    }

    fn push_dense_pair(&mut self, bits: u32, mantissa: u32) {
        self.push_dense(bits, mantissa);
        self.push_dense(bits | 0x8000_0000, mantissa);
    }

    /// Held inputs, deduplicated and sorted by input bits: what a reference
    /// table must contain.
    fn pinned_inputs(&self) -> Vec<u32> {
        let mut inputs = self.pinned.clone();
        inputs.sort_unstable();
        inputs.dedup();
        inputs
    }
}

/// Parses a reference table, refusing anything that is not a sorted
/// duplicate-free sequence of pairs.
fn reference_table(name: &str, blob: &[u8]) -> Vec<(u32, u32)> {
    if blob.len() < 8 || &blob[..4] != b"CMP1" {
        panic!(
            "{name}: the committed reference table is not a `CMP1` table \
             ({} bytes). Regenerate it on the glibc 2.39 reference host with \
             cargo test -p claymore-model-gemma --test kernel_order_parity -- \
             --ignored regenerate_glibc_2_39_reference_tables",
            blob.len()
        );
    }
    let count = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
    if blob.len() != 8 + 8 * count {
        panic!(
            "{name}: the committed reference table announces {count} entries but \
             holds {} -- regenerate it on the glibc 2.39 reference host",
            (blob.len() - 8) / 8
        );
    }
    let mut table = Vec::with_capacity(count);
    for chunk in blob[8..].chunks_exact(8) {
        table.push((
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
        ));
    }
    assert!(
        table.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "{name}: the committed reference table is not sorted by input bits"
    );
    table
}

/// The contract, in two tiers.
///
/// Tier one, on **every** host: the port must keep producing the glibc 2.39
/// result recorded in the reference table. This is the part of the parity
/// contract that does not depend on the machine running the test.
///
/// Tier two, only when the installed libm *is* glibc 2.39 (sentinel probe):
/// the port must equal the installed libm on **every swept input**, pinned or
/// not. Comparing the port to a different libc is not a parity contract -- that
/// comparison is what turned the CI red on inputs nobody had measured (see the
/// module documentation), so on such a host it is replaced by the diagnostic
/// `report_divergence_against_installed_libm` and the skip is announced.
fn assert_transcendental_parity(
    name: &str,
    sweep: &Sweep,
    blob: &[u8],
    port: fn(f32) -> f32,
    installed: fn(f32) -> f32,
) {
    let table = reference_table(name, blob);
    assert_eq!(
        table.iter().map(|(input, _)| *input).collect::<Vec<u32>>(),
        sweep.pinned_inputs(),
        "{name}: the inputs held by the reference table and the inputs the sweep \
         pins no longer agree -- regenerate the table on the glibc 2.39 reference \
         host"
    );
    for (input, bits) in &table {
        let x = f32_of_bits(*input);
        let got = port(x);
        assert_eq!(
            bits_of(got),
            *bits,
            "{name}({x:?}) moved off the pinned glibc 2.39 result: port={got:?} \
             glibc 2.39={:?}",
            f32_of_bits(*bits)
        );
    }
    if host_is_pinned_glibc_2_39() {
        for input in &sweep.all {
            let x = f32_of_bits(*input);
            let got = port(x);
            let glibc = installed(x);
            assert_eq!(
                bits_of(got),
                bits_of(glibc),
                "{name}({x:?}) port={got:?} glibc={glibc:?}"
            );
        }
    } else {
        eprintln!(
            "{name}: installed libm is not glibc 2.39 (a sentinel input differs), \
             skipping the {}-input comparison against the host; the pinned table \
             of {} glibc 2.39 results was asserted in full. Publish this host's \
             divergence set with \
             cargo test -p claymore-model-gemma --test kernel_order_parity -- \
             --ignored report_divergence_against_installed_libm",
            sweep.all.len(),
            table.len()
        );
    }
}

/// Swept inputs for `expf`: every exponent slot, the overflow/underflow window
/// around log 2^128 and log 2^-150, and the branch-boundary neighbourhoods. The
/// exponent grids step the mantissa by `0x40000`, which wraps into 32 distinct
/// slots per exponent after masking; the dense pitch carries the finer coverage.
fn expf_samples() -> Sweep {
    let mut samples = Sweep::default();
    for e in 0u32..=127 {
        for m in 0..64u32 {
            let bits = (e << 23) | ((m * 0x40000) & 0x007f_ffff);
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
            samples.push_dense_pair(bits, m);
        }
    }
    samples
}

#[test]
fn portable_expf_matches_glibc() {
    let sweep = expf_samples();
    assert_transcendental_parity(
        "expf",
        &sweep,
        REFERENCE_GLIBC_2_39_EXPF,
        cuda_expf,
        f32::exp,
    );
}

/// Swept inputs for `expm1f`: exponent slots, reduction `k` boundaries, a dense
/// pitch across the reduction range, and the underflow boundaries.
fn expm1f_samples() -> Sweep {
    let mut samples = Sweep::default();
    for e in 0u32..=127 {
        for m in 0..64u32 {
            let bits = (e << 23) | ((m * 0x40000) & 0x007f_ffff);
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
            samples.push_dense_pair(bits, m);
        }
    }
    samples
}

#[test]
fn portable_expm1f_matches_glibc() {
    let sweep = expm1f_samples();
    assert_transcendental_parity(
        "expm1f",
        &sweep,
        REFERENCE_GLIBC_2_39_EXPM1F,
        cuda_expm1f,
        f32::exp_m1,
    );
}

/// Swept inputs for `tanhf`: exponent slots down to 2^-60, the saturation and
/// branch boundaries, the specials, and a dense pitch below the 22 saturation.
fn tanhf_samples() -> Sweep {
    let mut samples = Sweep::default();
    for e in -60i32..=27 {
        for m in 0..64u32 {
            let bits = (((e + 127) as u32) << 23) | ((m * 0x40000) & 0x007f_ffff);
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
            samples.push_dense_pair(bits, m);
        }
    }
    samples
}

#[test]
fn portable_tanhf_matches_glibc() {
    let sweep = tanhf_samples();
    assert_transcendental_parity(
        "tanhf",
        &sweep,
        REFERENCE_GLIBC_2_39_TANHF,
        cuda_tanhf,
        f32::tanh,
    );
}

/// Diagnostic, not a contract: prints every swept input where the port and the
/// **installed** libm disagree, with counts per function, plus how many of the
/// pinned reference entries that host's libm disagrees with. Run it when the CI
/// image or the host libc moves and publish the output: it is the measurement
/// behind a table regeneration, never a substitute for one. It prints; it
/// asserts nothing.
#[test]
#[ignore]
fn report_divergence_against_installed_libm() {
    fn sweep_all(
        name: &str,
        samples: &[u32],
        installed: fn(f32) -> f32,
        port: fn(f32) -> f32,
    ) -> (usize, usize, usize) {
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
        let mut distinct = samples.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        (total, diverged, distinct.len())
    }

    let expf = expf_samples();
    let expm1f = expm1f_samples();
    let tanhf = tanhf_samples();
    let mut totals = (0usize, 0usize, 0usize);
    for (name, samples, installed, port) in [
        (
            "expf",
            expf.all.as_slice(),
            f32::exp as fn(f32) -> f32,
            cuda_expf as fn(f32) -> f32,
        ),
        (
            "expm1f",
            expm1f.all.as_slice(),
            f32::exp_m1 as fn(f32) -> f32,
            cuda_expm1f as fn(f32) -> f32,
        ),
        (
            "tanhf",
            tanhf.all.as_slice(),
            f32::tanh as fn(f32) -> f32,
            cuda_tanhf as fn(f32) -> f32,
        ),
    ] {
        let (total, diverged, distinct) = sweep_all(name, samples, installed, port);
        eprintln!(
            "{name}: {diverged} divergence(s) out of {total} swept inputs ({distinct} distinct)"
        );
        totals.0 += total;
        totals.1 += diverged;
        totals.2 += distinct;
    }
    eprintln!(
        "total: {} divergence(s) out of {} swept inputs ({} distinct)",
        totals.1, totals.0, totals.2
    );

    for (name, blob, installed) in [
        (
            "expf",
            REFERENCE_GLIBC_2_39_EXPF,
            f32::exp as fn(f32) -> f32,
        ),
        (
            "expm1f",
            REFERENCE_GLIBC_2_39_EXPM1F,
            f32::exp_m1 as fn(f32) -> f32,
        ),
        (
            "tanhf",
            REFERENCE_GLIBC_2_39_TANHF,
            f32::tanh as fn(f32) -> f32,
        ),
    ] {
        let table = reference_table(name, blob);
        let stale = table
            .iter()
            .filter(|(input, expected)| bits_of(installed(f32_of_bits(*input))) != *expected)
            .count();
        eprintln!(
            "{name}: installed libm disagrees with the pinned glibc 2.39 table on \
             {stale} of {} entries",
            table.len()
        );
    }
}

/// Writes the glibc 2.39 reference tables consumed by the parity tests, from
/// the installed libm of the machine running it. That is only sound on the
/// reference host, so the sentinel probe gates the write.
///
/// Reference host as of this work: Ubuntu 24.04.5, `ldd (Ubuntu GLIBC
/// 2.39-0ubuntu8.12) 2.39`, the same host that produced the golden inputs and
/// ran the RTX 5090 campaign. Run
///
/// ```text
/// cargo test -p claymore-model-gemma --test kernel_order_parity -- \
///     --ignored regenerate_glibc_2_39_reference_tables
/// ```
///
/// then commit the three files with the measurement that justified it.
#[test]
#[ignore = "rewrites the committed reference tables; only sound on the glibc 2.39 reference host"]
fn regenerate_glibc_2_39_reference_tables() {
    assert!(
        host_is_pinned_glibc_2_39(),
        "the installed libm disagrees with the glibc 2.39 sentinels: this host is \
         not the oracle, so its results must not become the reference tables"
    );
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
    for (name, sweep, blob, installed) in [
        (
            "expf",
            expf_samples(),
            REFERENCE_GLIBC_2_39_EXPF,
            f32::exp as fn(f32) -> f32,
        ),
        (
            "expm1f",
            expm1f_samples(),
            REFERENCE_GLIBC_2_39_EXPM1F,
            f32::exp_m1 as fn(f32) -> f32,
        ),
        (
            "tanhf",
            tanhf_samples(),
            REFERENCE_GLIBC_2_39_TANHF,
            f32::tanh as fn(f32) -> f32,
        ),
    ] {
        let inputs = sweep.pinned_inputs();
        let mut bytes: Vec<u8> = Vec::with_capacity(8 + 8 * inputs.len());
        bytes.extend_from_slice(b"CMP1");
        bytes.extend_from_slice(&(inputs.len() as u32).to_le_bytes());
        for input in &inputs {
            bytes.extend_from_slice(&input.to_le_bytes());
            bytes.extend_from_slice(&bits_of(installed(f32_of_bits(*input))).to_le_bytes());
        }
        let path = format!("{dir}/glibc_2_39_{name}.bin");
        if blob == bytes.as_slice() {
            eprintln!("{path}: unchanged ({} entries)", inputs.len());
            continue;
        }
        std::fs::write(&path, &bytes).expect("write the reference table");
        eprintln!("{path}: rewritten ({} entries)", inputs.len());
    }
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
