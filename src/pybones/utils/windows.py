"""Sliding z windows over a stream of slices."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def mirror_index(k: int, depth: int) -> int:
    """Reflect an out-of-range slice index into ``[0, depth)`` (edge-excluded reflection).

    ``-1 -> 1``, ``depth -> depth - 2``; a 1-slice stack always maps to 0.

    Args:
        k: Slice index, possibly outside the stack.
        depth: Number of slices.

    Returns:
        The reflected index.
    """
    if k < 0:
        k = -k
    if k >= depth:
        k = 2 * (depth - 1) - k
    return min(max(k, 0), depth - 1)


def sliding_windows[T](items: "Iterable[T]", depth: int, half: int) -> "Iterator[list[T]]":
    """Stream, for each item, the ``2 * half + 1`` items centred on it (mirrored at the ends).

    Only the items of the current window are kept. A yielded window holds
    references, so items still used by pending computations stay alive until
    those finish, while the stream itself never buffers more than the window.

    Args:
        items: Items in z order (e.g. slices).
        depth: Number of items in the stream.
        half: Items needed on each side.

    Yields:
        One window per item, in order; the centre item is ``window[half]``.
    """
    buffer: dict[int, T] = {}
    source = iter(items)
    loaded = 0
    for z in range(depth):
        while loaded < min(z + half + 1, depth):
            buffer[loaded] = next(source)
            loaded += 1
        yield [buffer[mirror_index(k, depth)] for k in range(z - half, z + half + 1)]
        # Later windows start at z + 1 - half; mirrored indices never fall below it.
        _ = buffer.pop(z - half, None)
