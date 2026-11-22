"""Tests of the Richardson-Lucy deconvolution (:mod:`pybones.imaging.deconvolution`)."""

import numpy as np
import pytest
from scipy import ndimage

from pybones.imaging.deconvolution import (
    DIVISION_FLOOR,
    GaussianPsf,
    gaussian_blur,
    iter_deconvolved_above,
    richardson_lucy,
)

PSF = GaussianPsf((1.2, 1.5, 1.5))
ITERATIONS = 3
SHAPE = (14, 40, 37)
LEVEL_PERCENTILE = 90


def noisy_volume(seed: int) -> np.ndarray:
    """Draw a blurred, noisy positive volume with a few bright tubes.

    Args:
        seed: Random seed.

    Returns:
        ``uint16`` volume of :data:`SHAPE`.
    """
    rng = np.random.default_rng(seed)
    volume = np.full(SHAPE, 3000.0)
    volume[:, 10, 5:30] = 30000.0
    volume[4:12, 20:23, 25] = 25000.0
    volume = ndimage.gaussian_filter(volume, (1.2, 1.5, 1.5)) + rng.normal(0, 300, SHAPE)
    return np.clip(volume, 0, 65535).astype(np.uint16)


def reference_blur(volume: np.ndarray, psf: GaussianPsf) -> np.ndarray:
    """Blur with SciPy's separable correlation, mirrored faces, the same sampled kernels.

    Args:
        volume: Volume to blur.
        psf: The Gaussian.

    Returns:
        The blurred volume (float64).
    """
    out = volume.astype(np.float64)
    for axis in range(3):
        out = ndimage.correlate1d(
            out, psf.kernel(axis).ravel().astype(np.float64), axis, mode="reflect"
        )
    return out


def test_the_blur_matches_an_independent_separable_correlation() -> None:
    """OpenCV in-plane plus the z sum equal SciPy's correlation, faces included."""
    volume = noisy_volume(0).astype(np.float32)
    np.testing.assert_allclose(gaussian_blur(volume, PSF), reference_blur(volume, PSF), rtol=1e-5)


def test_richardson_lucy_matches_its_definition() -> None:
    """Each iteration multiplies the estimate by the blurred ratio of image to blurred estimate."""
    image = noisy_volume(1).astype(np.float64)
    estimate = image.copy()
    for _ in range(ITERATIONS):
        ratio = image / np.maximum(reference_blur(estimate, PSF), DIVISION_FLOOR)
        estimate *= reference_blur(ratio, PSF)
    result = richardson_lucy(image.astype(np.float32), PSF, ITERATIONS)
    np.testing.assert_allclose(result, estimate, rtol=1e-3)


def test_deconvolution_sharpens_a_blurred_line() -> None:
    """A blurred bright line gets brighter and narrower again."""
    image = noisy_volume(2).astype(np.float32)
    sharp = richardson_lucy(image, PSF, ITERATIONS)
    assert sharp[7, 10, 15] > image[7, 10, 15]
    assert sharp[7, 13, 15] < image[7, 13, 15]  # beside the line, the blur's glow goes down


@pytest.mark.parametrize(("block", "workers"), [((4, 9), 1), ((5, 16), 3), ((100, 1000), 2)])
def test_blocks_give_the_whole_volume_result(block: tuple[int, int], workers: int) -> None:
    """Small blocks with their margins (many of them, threads or not) give the same mask."""
    volume = noisy_volume(3)
    whole = richardson_lucy(volume.astype(np.float32), PSF, ITERATIONS)
    level = float(np.percentile(whole, LEVEL_PERCENTILE))
    expected = whole >= level
    got = np.stack(list(iter_deconvolved_above(volume, PSF, ITERATIONS, level, block, workers)))
    # Float sums in another order can flip a voxel sitting exactly on the level.
    near_level = np.abs(whole - level) <= level * 1e-5
    assert np.array_equal(got[~near_level], expected[~near_level])


def test_an_axis_without_blur_is_left_alone() -> None:
    """A zero sigma gives a one-tap kernel: only the other axes blur."""
    volume = noisy_volume(4).astype(np.float32)
    flat = GaussianPsf((0.0, 1.5, 1.5))
    assert flat.radius[0] == 0
    np.testing.assert_allclose(gaussian_blur(volume, flat), reference_blur(volume, flat), rtol=1e-5)
