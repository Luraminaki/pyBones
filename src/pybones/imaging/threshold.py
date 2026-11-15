"""Histogram accumulation and multi-level Otsu thresholding."""

import itertools
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.utils.constants import UINT16_LEVELS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

#: Number of intensity levels of a 16-bit image.
#: Classes produced by :func:`two_level_otsu`.
N_CLASSES = 3
#: Relative tolerance under which two Otsu scores are considered tied.
_TIE_TOLERANCE = 1e-12


def accumulate_histogram(images: "Iterable[NDArray[np.uint16]]") -> "NDArray[np.int64]":
    """Full-resolution intensity histogram of a set of ``uint16`` images.

    Uses ``cv2.calcHist`` (no temporary copy of the image), accumulated in
    ``int64`` so arbitrarily large stacks cannot overflow.

    Args:
        images: Iterable of 2D ``uint16`` images.

    Returns:
        Histogram with one bin per intensity level.

    Raises:
        TypeError: If an image is not ``uint16``.
    """
    hist = np.zeros(UINT16_LEVELS, dtype=np.int64)
    for img in images:
        if img.dtype != np.uint16:
            raise TypeError(f"Expected uint16 images, got {img.dtype}")
        h = cv2.calcHist([img], [0], None, [UINT16_LEVELS], [0, UINT16_LEVELS])
        hist += h.ravel().astype(np.int64)
    return hist


def _rebin(hist: "NDArray[np.integer]", n_bins: int) -> "tuple[NDArray[np.float64], int, int]":
    """Rebin a per-level histogram onto ``n_bins`` bins spanning its non-empty range.

    Args:
        hist: Per-intensity-level histogram.
        n_bins: Requested number of bins (fewer if the range has fewer levels).

    Returns:
        ``(binned, lo, width)``: bin counts, first non-empty level and number of levels
        covered.

    Raises:
        ValueError: If the histogram is empty.
    """
    nonzero = np.flatnonzero(hist)
    if nonzero.size == 0:
        raise ValueError("Empty histogram")
    lo, hi = int(nonzero[0]), int(nonzero[-1])
    width = hi - lo + 1
    n_bins = min(n_bins, width)
    idx = (np.arange(width, dtype=np.int64) * n_bins) // width
    binned = np.bincount(idx, weights=hist[lo : hi + 1], minlength=n_bins)
    return binned, lo, width


def class_means(hist: "NDArray[np.integer]", thresholds: "tuple[int, ...]") -> list[float]:
    """Mean intensity of each class delimited by ``thresholds`` (vectorized).

    Args:
        hist: Per-intensity-level histogram.
        thresholds: Increasing class boundaries ``t``; class ``k`` is
            ``t[k-1] <= v < t[k]``.

    Returns:
        One mean per class (``nan`` for an empty class).
    """
    levels = np.arange(hist.size, dtype=np.float64)
    edges = [0, *thresholds, hist.size]
    return [
        float(hist[lo:hi] @ levels[lo:hi] / weight)
        if (weight := hist[lo:hi].sum())
        else float("nan")
        for lo, hi in itertools.pairwise(edges)
    ]


def two_level_otsu(hist: "NDArray[np.integer]", n_bins: int) -> tuple[int, int]:
    """Two thresholds splitting a histogram into 3 classes (multi-Otsu).

    Exhaustively maximizes the between-class variance over every threshold
    pair, fully vectorized on an ``n_bins x n_bins`` grid. Equivalent to
    Matlab's ``multithresh(img, 2)`` but at a configurable resolution. When
    the optimum is a plateau (empty histogram gap), its centre is returned.

    Args:
        hist: Per-intensity-level histogram (e.g. from :func:`accumulate_histogram`).
        n_bins: Number of bins used for the search (precision vs. ``O(n_bins²)`` cost).

    Returns:
        ``(t_low, t_high)`` intensity levels such that the classes are
        ``v < t_low``, ``t_low <= v < t_high`` and ``v >= t_high``.

    Raises:
        ValueError: If the histogram is empty or has fewer than 3 distinct levels.
    """
    binned, lo, width = _rebin(hist, n_bins)
    n = binned.size
    if n < N_CLASSES:
        raise ValueError(f"Need at least {N_CLASSES} distinct intensity levels")
    p = binned / binned.sum()
    centers = np.arange(n, dtype=np.float64) + 0.5  # bin units; affine maps keep the argmax
    weight = np.cumsum(p)  # class-0 weight for a split after bin i
    moment = np.cumsum(p * centers)
    total = moment[-1]

    w0, m0 = weight[:, None], moment[:, None]  # split after bin i (rows)
    w01, m01 = weight[None, :], moment[None, :]  # split after bin j (cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Between-class variance up to a constant: sum_k M_k² / W_k.
        score: NDArray[np.float64] = (
            m0**2 / w0 + (m01 - m0) ** 2 / (w01 - w0) + (total - m01) ** 2 / (1.0 - w01)
        )
    i_idx, j_idx = np.triu_indices(n, k=1)
    valid: NDArray[np.float64] = score[i_idx, j_idx]
    valid[~np.isfinite(valid)] = -np.inf
    best = valid.max()
    tied = valid >= best - _TIE_TOLERANCE * abs(best)
    i, j = round(float(i_idx[tied].mean())), round(float(j_idx[tied].mean()))

    def first_level_of_bin(b: int) -> int:
        """Smallest level ``v`` with ``(v - lo) * n // width >= b`` (integer ceil division).

        Args:
            b: Bin index.

        Returns:
            The intensity level where bin ``b`` starts.
        """
        return lo + -(-(b * width) // n)

    return first_level_of_bin(i + 1), first_level_of_bin(j + 1)
