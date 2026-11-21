"""Tests of the z majority vote (:mod:`pybones.segmentation.zsmoothing`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.segmentation.labels import Label
from pybones.segmentation.zsmoothing import majority_vote, z_majority
from tests.helpers import truncated_window

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reference_vote(window: "list[NDArray[np.uint8]]", center: int) -> "NDArray[np.uint8]":
    """Vote the per-pixel mode naively (reference implementation).

    Ties keep the center label, else go to the smallest label.

    Args:
        window: Consecutive label slices.
        center: Index of the corrected slice in ``window``.

    Returns:
        The voted slice.
    """
    stack = np.stack(window)
    out = stack[center].copy()
    for idx in np.ndindex(out.shape):
        values = stack[(slice(None), *idx)].tolist()
        counts = {v: values.count(v) for v in values}
        best = max(counts.values())
        if counts[out[idx]] < best:
            out[idx] = min(v for v, c in counts.items() if c == best)
    return out


@pytest.mark.parametrize("size", [3, 5])
def test_majority_vote_matches_reference(size: int) -> None:
    """The vectorised vote (and its 3-slice fast path) matches the naive mode."""
    rng = np.random.default_rng(size)
    window = [rng.integers(0, len(Label), (12, 12), dtype=np.uint8) for _ in range(size)]
    for center in range(size):
        np.testing.assert_array_equal(majority_vote(window, center), reference_vote(window, center))


@pytest.mark.parametrize("window", [1, 3, 5])
def test_z_majority_streaming_matches_offline(window: int) -> None:
    """Streaming gives the offline vote, with windows truncated at the stack ends."""
    rng = np.random.default_rng(0)
    slices = [rng.integers(0, len(Label), (8, 8), dtype=np.uint8) for _ in range(9)]
    out = list(z_majority(iter(slices), window))
    assert len(out) == len(slices)
    for z, got in enumerate(out):
        lo, hi = truncated_window(z, window, len(slices))
        np.testing.assert_array_equal(got, reference_vote(slices[lo:hi], z - lo))
