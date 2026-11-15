"""Order-preserving, memory-bounded parallel map over a thread pool."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from concurrent.futures import Future

#: Name prefix of the worker threads (``pybones-worker_0``...), as shown in the logs.
WORKER_THREAD_PREFIX = "pybones-worker"


def ordered_map[T, R](
    fn: "Callable[[T], R]", items: "Iterable[T]", workers: int, extra_in_flight: int
) -> "Iterator[R]":
    """Apply ``fn`` to ``items`` in a thread pool, yielding results in input order.

    ``items`` is consumed lazily and at most ``workers + extra_in_flight`` calls
    are pending at any time, so memory stays bounded for long streams. Threads
    pay off for OpenCV / numpy / zlib work, which releases the GIL.

    Args:
        fn: Function applied to each item.
        items: Input stream.
        workers: Thread count; ``1`` runs sequentially in the calling thread.
        extra_in_flight: Pending calls allowed beyond ``workers``.

    Yields:
        ``fn(item)`` for each item, in order.
    """
    if workers == 1:
        yield from map(fn, items)
        return
    max_in_flight = workers + extra_in_flight
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=WORKER_THREAD_PREFIX) as pool:
        pending: deque[Future[R]] = deque()
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= max_in_flight:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
