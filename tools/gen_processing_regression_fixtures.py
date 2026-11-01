"""Generate fixtures for the processing regressions fixed in the Rust port.

Three SciPy/scikit-image compatibility details are pinned here:

  1. ``median_filter`` with an *even* size uses the window offsets
     ``-(size // 2) .. size - size // 2`` (upper median), not ``-r..=r``.
  2. ``blob_dog`` (via ``peak_local_max`` on the DoG cube) compares a scale
     layer only against ``scale - 1 .. scale + 1``, not against every layer.
  3. ``scipy.ndimage.gaussian_filter1d`` accumulates the symmetric kernel as
     ``centre * w[centre]`` followed by the sample pairs from the outside in,
     which decides how a near-integer result truncates to ``uint16``.

Run with the reference environment:
    /tmp/tissueseg3d-reference-env/bin/python tools/gen_processing_regression_fixtures.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import median_filter
from skimage.feature import peak_local_max
from skimage.filters import gaussian

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reference_fixtures import write_fixture  # noqa: E402


def write_median_fixtures(out):
    """Even median sizes: scipy keeps the *upper* median of the exact window."""
    shape = (3, 9, 11)
    rng = np.random.default_rng(2024)
    m = rng.integers(0, 5, size=shape, dtype=np.uint32)  # duplicates on purpose
    write_fixture(out / 'median_even_input.u32', m, 'u32')
    for name, size in (
        ('median_size2_expected.u32', (1, 2, 2)),
        ('median_size4_expected.u32', (1, 4, 4)),
        ('median_size_2x4x2_expected.u32', (2, 4, 2)),
    ):
        write_fixture(out / name, median_filter(m, size=size), 'u32')


def _ascending_order_gaussian(value, sigma):
    """Model the *old* ascending-accumulation order for a constant volume."""
    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (1.0 / (sigma * sigma)) * x**2)
    w = w / w.sum()
    for _ in range(3):  # one pass per axis; a constant stays constant
        acc = 0.0
        for k in range(2 * radius + 1):
            acc += w[k] * value
        value = acc
    return value


def write_gaussian_quantization_fixtures(out):
    """Near-integer Gaussian results whose uint16 truncation depends on sfum order.

    ``gaussian(...).astype('uint16')`` truncates, so a value that is a single ULP
    below an integer truncates one lower.  These constants were picked because
    the SciPy accumulation order and the ascending order disagree on them.
    """
    shape = (2, 3, 4)
    cases = (
        ('gauss_const_sigma_0p5_1000', 0.5, 1000),
        ('gauss_const_sigma_0p75_max', 0.75, 65535),
        ('gauss_const_sigma_0p26_one', 0.26, 1),
    )
    for name, sigma, value in cases:
        img = np.full(shape, float(value))
        expected = gaussian(img, sigma=(sigma, sigma, sigma),
                            preserve_range=True).astype(np.uint16)
        assert _ascending_order_gaussian(float(value), sigma) != float(expected.flat[0]), (
            f'{name} does not discriminate the summation order')
        write_fixture(out / f'{name}_expected.u16', expected, 'u16')


def _dog_cube(img, sigmas, sigma_ratio):
    cube = np.empty(img.shape + (len(sigmas) - 1,), dtype=np.float64)
    previous = gaussian(img, sigma=sigmas[0], mode='reflect', preserve_range=True)
    for i, s in enumerate(sigmas[1:]):
        current = gaussian(img, sigma=s, mode='reflect', preserve_range=True)
        cube[..., i] = previous - current
        previous = current
    cube *= 1 / (sigma_ratio - 1)
    return cube


def write_dog_peak_fixtures(out):
    """DoG scale-space peaks for a deterministic multi-scale blob field.

    The scale axis is a real ``peak_local_max`` footprint dimension, so the
    expected peaks include a blob that only survives when a layer is compared
    with its immediate scale neighbours.
    """
    shape = (18, 20, 22)
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')
    rng = np.random.default_rng(4)
    img = np.zeros(shape, dtype=np.float64)
    for _ in range(14):
        cz, cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1]), rng.uniform(0, shape[2])
        s = rng.uniform(1.0, 6.0)
        a = rng.uniform(500, 60000)
        img += a * np.exp(-((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    img = (img * 65535 / img.max()).astype(np.float64)

    min_sigma, max_sigma, ratio, threshold = 1.0, 8.0, 1.6, 0.001
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(ratio) + 1))
    sigmas = np.array([min_sigma * ratio**i for i in range(k + 1)])
    cube = _dog_cube(img, sigmas, ratio)

    peaks = peak_local_max(cube, threshold_abs=threshold, threshold_rel=None,
                           exclude_border=(0, 0, 0, 0),
                           footprint=np.ones((3,) * (img.ndim + 1)))

    # The regression this fixture guards: comparing against every layer drops
    # peaks, so make sure the sample actually distinguishes the two rules.
    spatial = ndi.maximum_filter(cube, footprint=np.ones((3, 3, 3, 1)), mode='nearest')
    all_layers = (cube == spatial.max(axis=3, keepdims=True)) & (cube > threshold)
    assert all_layers.sum() != len(peaks), 'fixture does not discriminate the scale window'

    write_fixture(out / 'dog_peaks_input.f64', img, 'f64')
    write_fixture(out / 'dog_peaks_expected.u32', peaks.astype(np.uint32), 'u32')


def main():
    out = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
    out.mkdir(parents=True, exist_ok=True)
    write_median_fixtures(out)
    write_gaussian_quantization_fixtures(out)
    write_dog_peak_fixtures(out)
    print('wrote processing regression fixtures to', out)


if __name__ == '__main__':
    main()
