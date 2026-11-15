"""Streaming majority-vote smoothing of label slices along z."""

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from pybones.segmentation.labels import Label

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from numpy.typing import NDArray

_FAST_PATH_WINDOW = 3


def majority_vote(window: "Sequence[NDArray[np.uint8]]", center: int) -> "NDArray[np.uint8]":
    """Vote the per-pixel most frequent label across ``window``.

    Ties keep the center label when it is among the most frequent; otherwise
    they go to the smallest label value.

    Args:
        window: Consecutive label slices (at most 255).
        center: Index in ``window`` of the slice being corrected.

    Returns:
        The voted label slice.
    """
    ref = window[center]
    best = ref.copy()
    if len(window) == _FAST_PATH_WINDOW and center == 1:
        # Majority of 3: the center only changes where both neighbours agree against it.
        prev, nxt = window[0], window[2]
        np.copyto(best, prev, where=(prev == nxt) & (prev != ref))
        return best
    best_count = np.zeros(ref.shape, dtype=np.uint8)
    for w in window:
        best_count += w == ref
    count = np.empty_like(best_count)
    for lab in Label:
        _ = count.fill(0)
        for w in window:
            count += w == lab
        better = count > best_count
        best[better] = lab
        _ = np.maximum(best_count, count, out=best_count)
    return best


def z_majority(slices: "Iterable[NDArray[np.uint8]]", window: int) -> "Iterator[NDArray[np.uint8]]":
    """Smooth a stream of label slices along z with a sliding majority vote.

    Only ``window`` slices are buffered. At the stack ends the window is
    truncated to the available neighbours.

    Args:
        slices: Label slices in z order.
        window: Odd window size (1 passes slices through unchanged).

    Yields:
        Smoothed label slices, in the same order.
    """
    if window == 1:
        yield from slices
        return
    half = window // 2
    buf: deque[NDArray[np.uint8]] = deque()
    start = 0  # z index of buf[0]
    next_out = 0

    def emit() -> "NDArray[np.uint8]":
        """Vote slice ``next_out`` and drop the slices no later window needs.

        Returns:
            The smoothed slice.
        """
        nonlocal start, next_out
        voted = majority_vote(list(buf), next_out - start)
        next_out += 1
        while start < next_out - half:
            _ = buf.popleft()
            start += 1
        return voted

    for z, s in enumerate(slices):
        buf.append(s)
        if next_out <= z - half:
            yield emit()
    while next_out < start + len(buf):
        yield emit()
