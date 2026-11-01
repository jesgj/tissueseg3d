"""Generate the bit-exact NumPy/SciPy Gaussian parity fixtures.

The Rust port has to reproduce *the exact float64 values* that the pinned
reference stack produces, because the very last step of ``segment_nuclei`` is a
truncating ``uint16`` cast followed by ``>=``: a one-ULP difference in a kernel
weight flips whole voxels between foreground and background.

Two upstream implementation details decide those last bits:

  1. ``scipy.ndimage._filters._gaussian_kernel1d`` normalises the kernel with
     ``numpy.sum``, which uses *pairwise* summation (``loops_utils.h.src``:
     8 accumulators, block size 128, recursive halving), not a sequential sum.
  2. ``numpy.exp`` on ``float64`` is dispatched to the bundled SVML
     ``__svml_exp8_ha`` (``numpy/_core/src/umath/svml`` at NumPy tag
     ``v2.5.4``, SVML revision bd9d46e0c627ab16f3d2641f7cdb1bc15d303ba4), whose
     AVX512 normal path differs from ``libm``'s ``exp`` in the last ULP for
     some arguments.

This generator records, straight from the pinned environment, (a) the exact
normalised kernel weights, (b) ``numpy.exp`` on the raw kernel arguments, and
(c) the truncated ``uint16`` Gaussian output for the 264-case constant-volume
probe (``[3, 5, 7]`` volume, all-ones cell labels, ``median_size = 1``,
``z_ratio = 1``, ``sigma`` 0.00..3.00 step 0.05 plus 3.2/4/5.12/6/6.92, and
input values 1/1000/12345/65535).  No Rust code is used or trusted here.

Run with the reference environment:
    /tmp/tissueseg3d-reference-env/bin/python tools/gen_gaussian_parity_fixtures.py
"""
from __future__ import annotations

import pathlib
import sys

import math

import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d
from skimage.filters import gaussian

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reference_fixtures import write_fixture  # noqa: E402

TRUNCATE = 4.0

# Sigmas exercised by the segment_nuclei constant-volume parity probe.
PROBE_SIGMAS = [i / 20.0 for i in range(0, 61)] + [3.2, 4.0, 5.12, 6.0, 6.92]

# Extra sigmas that only need weight coverage: the small ones have a single
# sample kernel, the large ones push the normalising sum past numpy's 128
# element pairwise-summation block.
WEIGHT_SIGMAS = [s for s in PROBE_SIGMAS if s > 0.0] + [
    0.01, 0.02, 0.37, 0.5 / 6.92, 1.5 / 6.92, 12.0, 20.0, 50.0]

# segment_nuclei blurs the z axis with sigma / z_ratio.
Z_RATIOS = (1.0, 2.0, 6.92)

PROBE_SHAPE = (3, 5, 7)
PROBE_VALUES = (1, 1000, 12345, 65535)


def kernel_radius(sigma):
    return int(TRUNCATE * sigma + 0.5)


def kernel_arguments(sigma):
    """The raw ``exp`` arguments of ``_gaussian_kernel1d(sigma, 0, radius)``."""
    sigma2 = sigma * sigma
    x = np.arange(-kernel_radius(sigma), kernel_radius(sigma) + 1, dtype=np.int64)
    return (-0.5 / sigma2) * x ** 2


def write_exp_fixtures(out):
    args = []
    for sigma in WEIGHT_SIGMAS:
        if sigma == 0.0:
            continue
        for z in Z_RATIOS:
            args.append(kernel_arguments(sigma / z))
    flat = np.concatenate(args).astype(np.float64)
    # Deduplicate by bit pattern so the fixture stays small but keeps the
    # arguments scipy actually computes.
    bits = flat.view(np.int64)
    _, index = np.unique(bits, return_index=True)
    args = flat[np.sort(index)]
    write_fixture(out / 'numpy_exp_args.f64', args, 'f64')
    write_fixture(out / 'numpy_exp_expected.f64', np.exp(args), 'f64')


def write_kernel_weight_fixtures(out):
    sigmas = np.array(WEIGHT_SIGMAS, dtype=np.float64)
    blocks = [_gaussian_kernel1d(float(s), 0, kernel_radius(float(s))) for s in sigmas]
    weights = np.concatenate(blocks)
    write_fixture(out / 'gauss_kernel_sigmas.f64', sigmas, 'f64')
    write_fixture(out / 'gauss_kernel_weights.f64', weights, 'f64')

    # The normalised weights must sum to something a naive sequential
    # ``exp``/``sum`` port cannot reproduce, otherwise the fixture proves nothing.
    naive = 0
    for sigma, block in zip(sigmas, blocks):
        r = kernel_radius(float(sigma))
        # model of the pre-port Rust kernel: 1/(sigma*sigma) constant, libm
        # exp, sequential normalisation.
        sigma2 = 1.0 / (sigma * sigma)
        phi = [math.exp(-0.5 * sigma2 * float(x) * float(x))
               for x in range(-r, r + 1)]
        total = 0.0
        for v in phi:
            total += v
        old = np.array([v / total for v in phi])
        if old.tobytes() != block.tobytes():
            naive += 1
    assert naive > 0, 'fixture does not discriminate the numpy arithmetic'
    print(f'kernel weights: {naive} of {len(sigmas)} sigmas discriminate the old port')


def write_constant_probe_fixtures(out):
    """Truncated ``uint16`` Gaussian output of the constant-volume probe."""
    shape = PROBE_SHAPE
    cases = []
    for sigma in PROBE_SIGMAS:
        for value in PROBE_VALUES:
            img = np.full(shape, float(value))
            blurred = gaussian(img, sigma=(sigma, sigma, sigma), preserve_range=True)
            quantised = blurred.astype(np.uint16)
            cases.append((sigma, value, quantised))
    stacked = np.stack([q for _, _, q in cases]).reshape(len(cases), -1)
    write_fixture(out / 'gauss_const_blurred.u16', stacked, 'u16')
    # Self-describing case list, so the Rust test does not duplicate the ordering.
    write_fixture(out / 'gauss_const_sigmas.f64',
                  np.array([c[0] for c in cases], dtype=np.float64), 'f64')
    write_fixture(out / 'gauss_const_values.u16',
                  np.array([c[1] for c in cases], dtype=np.uint16), 'u16')

    # The probe is only meaningful where the truncation actually depends on
    # the last ULP: check that a sequential-sum/libm-exp port disagrees.
    disagreements = 0
    discriminating = set()
    for sigma, value, quantised in cases:
        if sigma == 0.0 or sigma in discriminating:
            continue
        r = kernel_radius(float(sigma))
        sigma2 = 1.0 / (sigma * sigma)
        phi = [math.exp(-0.5 * sigma2 * float(x) * float(x))
               for x in range(-r, r + 1)]
        total = 0.0
        for v in phi:
            total += v
        w = np.array([v / total for v in phi])
        vol = np.full(shape, float(value))
        for axis in range(3):
            acc = np.zeros(shape, dtype=np.float64)
            n = shape[axis]
            for z in range(shape[0]):
                for y in range(shape[1]):
                    for xx in range(shape[2]):
                        idx = (z, y, xx)
                        total = w[r] * vol[idx]
                        for k in range(r):
                            lo = max(idx[axis] - r + k, 0)
                            hi = min(idx[axis] + r - k, n - 1)
                            a = list(idx)
                            b = list(idx)
                            a[axis] = lo
                            b[axis] = hi
                            total += (vol[tuple(a)] + vol[tuple(b)]) * w[k]
                        acc[idx] = total
            vol = acc
        if not np.array_equal(vol.astype(np.uint16), quantised):
            disagreements += 1
            discriminating.add(sigma)
    assert disagreements > 0, 'fixture does not discriminate the numpy arithmetic'
    print(f'constant probe: {disagreements} sigmas discriminate the old port')


def write_pairwise_sum_fixtures(out):
    """Vectors where numpy's pairwise summation order is observable."""
    rng = np.random.default_rng(1)
    block = rng.uniform(-1.0, 2.0, 24)          # 8 accumulators + peel loop
    long = rng.uniform(-1.0, 2.0, 200)          # > block size: recursion
    expected = np.array([block.sum(), long.sum()], dtype=np.float64)
    for vec, want in ((block, expected[0]), (long, expected[1])):
        total = 0.0
        for v in vec:
            total += float(v)
        assert np.float64(total).tobytes() != want.tobytes(), \
            'fixture does not discriminate the pairwise order'
    write_fixture(out / 'numpy_pairwise_block.f64', block, 'f64')
    write_fixture(out / 'numpy_pairwise_long.f64', long, 'f64')
    write_fixture(out / 'numpy_pairwise_expected.f64', expected, 'f64')


def main():
    out = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
    out.mkdir(parents=True, exist_ok=True)
    write_exp_fixtures(out)
    write_pairwise_sum_fixtures(out)
    write_kernel_weight_fixtures(out)
    write_constant_probe_fixtures(out)
    print('wrote gaussian parity fixtures to', out)


if __name__ == '__main__':
    main()
