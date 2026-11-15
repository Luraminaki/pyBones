"""Tests of the ordered parallel map (:mod:`pybones.utils.parallel`)."""

import threading
import time
from typing import TYPE_CHECKING

import pytest

from pybones.utils.parallel import ordered_map

if TYPE_CHECKING:
    from collections.abc import Iterator

N_ITEMS = 10
DELAY_S = 0.001


def slow_for_small(x: int) -> int:
    """Square ``x``, taking longer for small values (so early items finish last).

    Args:
        x: Item, ``0 <= x < N_ITEMS``.

    Returns:
        ``x * x``.
    """
    time.sleep(DELAY_S * (N_ITEMS - x))
    return x * x


def identity(x: int) -> int:
    """Return the item unchanged.

    Args:
        x: Item.

    Returns:
        ``x``.
    """
    return x


def thread_id(_: int) -> int:
    """Identify the thread running the call.

    Args:
        _: Item (unused).

    Returns:
        The current thread identifier.
    """
    return threading.get_ident()


@pytest.mark.parametrize("workers", [1, 4])
def test_results_keep_input_order(workers: int) -> None:
    """Results come in input order even when later items finish first."""
    results = list(ordered_map(slow_for_small, range(N_ITEMS), workers, 2))
    assert results == [x * x for x in range(N_ITEMS)]


def test_in_flight_work_is_bounded() -> None:
    """Never more than ``workers + extra_in_flight`` items are pulled ahead."""
    consumed = 0
    lock = threading.Lock()

    def items() -> "Iterator[int]":
        """Count the items pulled from the source.

        Yields:
            ``0 .. 99``.
        """
        nonlocal consumed
        for i in range(100):
            with lock:
                consumed += 1
            yield i

    workers, extra = 2, 1
    results = ordered_map(identity, items(), workers=workers, extra_in_flight=extra)
    _ = next(results)
    assert consumed <= workers + extra


def test_single_worker_runs_in_calling_thread() -> None:
    """With one worker, every call runs in the calling thread."""
    caller = threading.get_ident()
    assert set(ordered_map(thread_id, range(5), 1, 0)) == {caller}
