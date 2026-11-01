//! Bit-exact ports of the NumPy float64 primitives the reference stack uses.
//!
//! The reference segmentation pipeline (NumPy 2.5.4 / SciPy 1.18.1) builds its
//! Gaussian kernels with `numpy.exp` and normalises them with `numpy.sum`.  Both
//! differ from the obvious Rust spelling in the last bit for some inputs, and
//! the pipeline's final step is a truncating `uint16` cast followed by `>=`, so
//! a one-ULP weight difference flips whole voxels between foreground and
//! background.  This module reproduces the upstream arithmetic order exactly:
//!
//! * [`pairwise_sum`] is NumPy's `TYPE_pairwise_sum` from
//!   `numpy/_core/src/umath/loops_utils.h.src` (8 accumulators, block size 128,
//!   recursive halving), which is what `ndarray.sum` uses for contiguous
//!   float64 data.
//! * [`exp`] is the AVX512 *normal* path of the bundled SVML `__svml_exp8_ha`
//!   (`numpy/_core/src/umath/svml`, SVML revision
//!   `bd9d46e0c627ab16f3d2641f7cdb1bc15d303ba4`), which is what `numpy.exp`
//!   dispatches to for float64 arrays on AVX512 hosts.
//!
//! See `THIRD_PARTY_NOTICES.md` for the upstream licences.

/// NumPy's pairwise summation order for contiguous float64 values.
///
/// This mirrors `DOUBLE_pairwise_sum` from `loops_utils.h.src`, including the
/// `-0.0` seed (so summing only `-0.0` keeps the sign) and the 8-accumulator
/// block for up to `PW_BLOCKSIZE` values.
pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    /// Number of values summed with 8 accumulators before recursing.
    const PW_BLOCKSIZE: usize = 128;

    if values.len() < 8 {
        let mut res = -0.0;
        for &v in values {
            res += v;
        }
        return res;
    }
    if values.len() <= PW_BLOCKSIZE {
        let n = values.len();
        let mut r = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += values[i + k];
            }
            i += 8;
        }
        // Combining the accumulators here avoids stack spills for the peel loop.
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        return res;
    }
    // Halve, keeping the recursion a multiple of the unroll factor.
    let mut n2 = values.len() / 2;
    n2 -= n2 % 8;
    pairwise_sum(&values[..n2]) + pairwise_sum(&values[n2..])
}

/// `numpy.exp` for float64 (SVML `__svml_exp8_ha` normal path).
///
/// The AVX512 sequence is transcribed from
/// `linux/avx512/svml_z0_exp_d_ha.s` (SVML revision
/// `bd9d46e0c627ab16f3d2641f7cdb1bc15d303ba4`) with every fused step kept as an
/// [`f64::mul_add`].  NumPy's scalar/vector tail goes through the same routine,
/// so this reproduces its results for every argument, not just the vector
/// lanes.
///
/// Only the normal path is ported: the assembly's argument-range compare is a
/// callout that raises flags/errno for extreme inputs (overflow, underflow,
/// NaN, huge magnitudes), none of which the Gaussian exponents reach.  Anything
/// outside the range where the table/scaling sequence is valid falls back to the
/// platform `exp`.
pub(crate) fn exp(x: f64) -> f64 {
    /// Range where the SVML normal path is valid; covers every Gaussian
    /// exponent (`-0.5 * x^2 / sigma^2` with scipy's 4-sigma truncation).
    const NORMAL_LIMIT: f64 = 700.0;

    if !(x > -NORMAL_LIMIT && x < NORMAL_LIMIT) {
        return x.exp();
    }

    // SVML forms `2^(52-4) * 1.5 + x * log2(e)` with round-to-zero, then
    // subtracts `2^(52-4) * 1.5`.  The magic constant is a multiple of 1/16, so
    // the subtraction is exact and the result is exactly floor(16 * log2e * x) /
    // 16: a range reduction to N = M + j/16 with the 16-entry table indexed by j.
    // Emulate it without touching the FPU rounding mode.
    let scaled = LOG2E * 16.0;
    let product = scaled * x;
    let residual = scaled.mul_add(x, -product);
    let mut k = product.floor();
    if residual < k - product {
        k -= 1.0;
    } else if residual >= (k + 1.0) - product {
        k += 1.0;
    }
    let m = k as i64;
    let j = m.rem_euclid(16) as usize;
    let shift = (m - j as i64) / 16;
    let n = m as f64 / 16.0;

    // R = x - N*ln(2)_high - N*ln(2)_low, then masked to |R| < 2 (a no-op for
    // well-reduced arguments, kept so the arithmetic matches the assembly).
    let r = (-n).mul_add(LN2_HI, x);
    let r = f64::from_bits((-n).mul_add(LN2_LO, r).to_bits() & R_MASK);

    // exp(r) - 1 = r * ((c4 + c5 r) r^2 + (c2 + c3 r)) r^2 + c0 + c1 r
    let r2 = r * r;
    let p = C5.mul_add(r, C4).mul_add(r2, C3.mul_add(r, C2));
    let p = p.mul_add(r2, C1.mul_add(r, C0));

    // exp(x) = 2^M * Th[j] * (1 + Tlr[j] + R * p)
    let scale = r.mul_add(p, TLR[j]);
    let res = scale.mul_add(TH[j], TH[j]);
    // vscalefpd by the integer part of N; |shift| <= 1010 for the range above.
    res * f64::from_bits(((shift + 1023) as u64) << 52)
}

/// `log2(e)` as used by the SVML sequence.
const LOG2E: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
/// `ln(2)` high part of the two-step range reduction.
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEFA_39EF);
/// `ln(2)` low part.
const LN2_LO: f64 = f64::from_bits(0x3C7A_BC9E_3B39_803F);
/// Polynomial coefficients c0..c5 of the `exp(r) - 1` expansion.
const C0: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FF70);
const C1: f64 = f64::from_bits(0x3FE0_0000_0000_D008);
const C2: f64 = f64::from_bits(0x3FC5_5555_5393_9732);
const C3: f64 = f64::from_bits(0x3FA5_5557_242D_68FE);
const C4: f64 = f64::from_bits(0x3F81_101C_BBC2_65C0);
const C5: f64 = f64::from_bits(0x3F57_4118_3694_0C04);
/// Clears the value's `2^0` exponent bit, i.e. enforces `|R| < 2`.
const R_MASK: u64 = 0xBFFF_FFFF_FFFF_FFFF;

/// `2^(j/16)` high parts; [`TLR`] carries the matching rounding-residual parts.
const TH: [f64; 16] = [
    f64::from_bits(0x3FF0_0000_0000_0000),
    f64::from_bits(0x3FF0_B558_6CF9_890F),
    f64::from_bits(0x3FF1_72B8_3C7D_517B),
    f64::from_bits(0x3FF2_387A_6E75_6238),
    f64::from_bits(0x3FF3_06FE_0A31_B715),
    f64::from_bits(0x3FF3_DEA6_4C12_3422),
    f64::from_bits(0x3FF4_BFDA_D536_2A27),
    f64::from_bits(0x3FF5_AB07_DD48_5429),
    f64::from_bits(0x3FF6_A09E_667F_3BCD),
    f64::from_bits(0x3FF7_A114_73EB_0187),
    f64::from_bits(0x3FF8_ACE5_422A_A0DB),
    f64::from_bits(0x3FF9_C491_82A3_F090),
    f64::from_bits(0x3FFA_E89F_995A_D3AD),
    f64::from_bits(0x3FFC_199B_DD85_529C),
    f64::from_bits(0x3FFD_5818_DCFB_A487),
    f64::from_bits(0x3FFE_A4AF_A2A4_90DA),
];
const TLR: [f64; 16] = [
    f64::from_bits(0x0000_0000_0000_0000),
    f64::from_bits(0x3C97_9AA6_5D83_7B6D),
    f64::from_bits(0xBC80_1B15_EAA5_9348),
    f64::from_bits(0x3C96_8EFD_E3A8_A894),
    f64::from_bits(0x3C83_4D75_4DB0_ABB6),
    f64::from_bits(0x3C85_9F48_A72A_4C6D),
    f64::from_bits(0x3C76_90CE_BB7A_AFB0),
    f64::from_bits(0x3C90_63E1_E21C_5409),
    f64::from_bits(0xBC93_B3EF_BF5E_2228),
    f64::from_bits(0xBC7B_32DC_B94D_A51D),
    f64::from_bits(0x3C8D_B72F_C1F0_EAB4),
    f64::from_bits(0x3C71_AFFC_2B91_CE27),
    f64::from_bits(0x3C8C_1A77_92CB_3387),
    f64::from_bits(0x3C73_6EAE_30AF_0CB3),
    f64::from_bits(0x3C74_A385_A63D_07A7),
    f64::from_bits(0xBC8F_F712_8FD3_91F0),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn read_fixture(name: &str) -> Vec<f64> {
        let path: std::path::PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
            .iter()
            .collect();
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(
            &bytes[0..4],
            b"TS3D",
            "bad fixture magic in {}",
            path.display()
        );
        assert_eq!(bytes[4], 1, "unsupported fixture version");
        assert_eq!(bytes[5], b'f', "expected an f64 fixture");
        let ndim = bytes[6] as usize;
        let mut count = 1usize;
        let mut off = 7;
        for _ in 0..ndim {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&bytes[off..off + 8]);
            count *= u64::from_le_bytes(raw) as usize;
            off += 8;
        }
        let payload = &bytes[off..];
        assert_eq!(payload.len(), count * 8);
        payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    }

    #[test]
    fn exp_matches_recorded_numpy_bits() {
        // Recorded from numpy.exp in the pinned reference environment for the
        // arguments scipy's _gaussian_kernel1d feeds to it.
        let args = read_fixture("numpy_exp_args.f64");
        let expected = read_fixture("numpy_exp_expected.f64");
        assert_eq!(args.len(), expected.len());
        for (i, (&x, &want)) in args.iter().zip(&expected).enumerate() {
            let got = exp(x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "exp({x:e}) [{i}] = {got:e}, numpy gives {want:e}"
            );
        }
    }

    #[test]
    fn pairwise_sum_matches_recorded_numpy_bits() {
        // Vectors where numpy's pairwise summation order is observable.
        let expected = read_fixture("numpy_pairwise_expected.f64");
        let block = read_fixture("numpy_pairwise_block.f64");
        let long = read_fixture("numpy_pairwise_long.f64");
        for (i, values) in [block, long].iter().enumerate() {
            let got = pairwise_sum(values);
            assert_eq!(
                got.to_bits(),
                expected[i].to_bits(),
                "pairwise sum of {} values = {got:?}, numpy gives {:?}",
                values.len(),
                expected[i]
            );
        }
    }
}
