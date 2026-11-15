"""Tests of the sliding z windows (:mod:`pybones.utils.windows`)."""

from typing import TYPE_CHECKING

import pytest

from pybones.utils.windows import mirror_index, sliding_windows

if TYPE_CHECKING:
    from collections.abc import Iterator

DEPTH = 5


@pytest.mark.parametrize(("k", "expected"), [(-2, 2), (-1, 1), (0, 0), (4, 4), (5, 3), (6, 2)])
def test_mirror_index(k: int, expected: int) -> None:
    """Indices outside the stack are reflected, edges excluded."""
    assert mirror_index(k, DEPTH) == expected


def test_windows_are_centred_and_mirrored() -> None:
    """Each window is centred on its item and mirrored at the ends."""
    windows = list(sliding_windows(iter(range(DEPTH)), DEPTH, 2))
    assert windows == [
        [2, 1, 0, 1, 2],
        [1, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 3],
        [2, 3, 4, 3, 2],
    ]


def test_items_are_pulled_lazily() -> None:
    """Only the items the current window needs are pulled from the source."""
    pulled: list[int] = []

    def source() -> "Iterator[int]":
        """Record the items pulled.

        Yields:
            ``0 .. 9``.
        """
        for i in range(10):
            pulled.append(i)
            yield i

    windows = sliding_windows(source(), 10, 1)
    _ = next(windows)
    assert pulled == [0, 1]


def test_single_item_stack() -> None:
    """A one-item stack mirrors onto itself."""
    assert list(sliding_windows(iter(["a"]), 1, 3)) == [["a"] * 7]
