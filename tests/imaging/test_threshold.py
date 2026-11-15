"""Tests of the histogram and multi-Otsu thresholds (:mod:`pybones.imaging.threshold`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.imaging.threshold import (
    accumulate_histogram,
    class_means,
    two_level_otsu,
)
from pybones.utils.constants import UINT16_LEVELS

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_BINS = 1024


def brute_force_otsu(hist: "NDArray[np.integer]") -> tuple[int, int]:
    """Find the best threshold pair by looping over every pair (reference implementation).

    Args:
        hist: Per-level histogram.

    Returns:
        The pair maximising the explicit between-class variance.
    """
    levels = np.arange(hist.size, dtype=np.float64)
    p = hist / hist.sum()
    mu = (p * levels).sum()
    best, best_t = -1.0, (0, 0)
    for t1 in range(1, hist.size - 1):
        for t2 in range(t1 + 1, hist.size):
            var = 0.0
            for lo, hi in ((0, t1), (t1, t2), (t2, hist.size)):
                w = p[lo:hi].sum()
                if w > 0:
                    var += w * ((p[lo:hi] * levels[lo:hi]).sum() / w - mu) ** 2
            if var > best:
                best, best_t = var, (t1, t2)
    return best_t


def test_matches_brute_force_on_small_histogram() -> None:
    """The vectorised search finds the brute-force optimum."""
    rng = np.random.default_rng(1)
    hist = rng.integers(0, 50, size=40)
    hist[[5, 18, 33]] += 400
    assert two_level_otsu(hist, N_BINS) == brute_force_otsu(hist)


def test_separates_trimodal_data_mid_gap() -> None:
    """Three well-separated modes are split in the gaps between them."""
    rng = np.random.default_rng(2)
    data = (
        np.concatenate([rng.normal(m, 400, 50_000) for m in (3_000, 14_000, 30_000)])
        .clip(0, UINT16_LEVELS - 1)
        .astype(np.uint16)
    )
    t_low, t_high = two_level_otsu(accumulate_histogram([data.reshape(300, 500)]), N_BINS)
    assert 5_000 < t_low < 12_000
    assert 16_000 < t_high < 28_000


def test_accumulate_histogram_counts_every_pixel() -> None:
    """Every pixel of every image is counted once; non-uint16 images are rejected."""
    img = np.arange(1000, dtype=np.uint16).reshape(10, 100)
    hist = accumulate_histogram([img, img])
    assert hist.sum() == 2000
    assert hist[999] == 2
    with pytest.raises(TypeError):
        _ = accumulate_histogram([img.astype(np.float32)])  # pyright: ignore[reportArgumentType]


def test_rejects_degenerate_input() -> None:
    """Empty histograms and fewer than three levels have no 3-class split."""
    with pytest.raises(ValueError, match="Empty"):
        _ = two_level_otsu(np.zeros(10, dtype=np.int64), N_BINS)
    hist = np.zeros(10, dtype=np.int64)
    hist[[2, 3]] = 5
    with pytest.raises(ValueError, match="at least"):
        _ = two_level_otsu(hist, N_BINS)


def test_class_means() -> None:
    """Class means are weighted by the histogram; an empty class gives ``nan``."""
    hist = np.zeros(100, dtype=np.int64)
    hist[[10, 20]] = 1  # class 0 mean 15
    hist[50] = 3  # class 1 mean 50
    means = class_means(hist, (40, 80))
    assert means[:2] == [15.0, 50.0]
    assert np.isnan(means[2])
