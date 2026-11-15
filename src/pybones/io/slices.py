"""Lazy access to stacks of 2D slices stored as image files (directory or zip archive)."""

import re
import zipfile
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Self, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType

    from numpy.typing import NDArray

    from pybones.config.schema import IOConfig

_DIGITS = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[str | int, ...]:
    """Build a sort key ordering embedded integers numerically (``s2`` < ``s10``).

    Args:
        name: File name or path.

    Returns:
        A tuple alternating text chunks and integers, usable as a sort key.
    """
    return tuple(int(tok) if tok.isdigit() else tok.lower() for tok in _DIGITS.split(name))


class SliceStack:
    """Ordered, lazily decoded stack of grayscale slices.

    Only one slice is decoded per :meth:`read` call, so memory use stays at a
    single slice regardless of the stack size. Slices are read from a directory
    or directly from a zip archive (no extraction needed).

    Example:
        >>> with SliceStack(Path("data_set.zip"), config.io) as stack:
        ...     for img in stack:
        ...         ...

    Attributes:
        source: Path of the directory or archive.
        names: Slice file names, naturally sorted.
    """

    def __init__(self, source: "Path", io_config: "IOConfig") -> None:
        """Index the slices available in ``source``.

        Args:
            source: Directory containing slice images, or a zip archive of them.
            io_config: Slice discovery rules (extensions, ignored names).

        Raises:
            FileNotFoundError: If ``source`` does not exist.
            ValueError: If ``source`` is not a directory or zip, or holds no slice.
        """
        self.source: Path = source
        self._config: IOConfig = io_config
        self._zip: zipfile.ZipFile | None = None
        if source.is_dir():
            names = [p.name for p in source.iterdir() if p.is_file() and self._is_slice(p.name)]
        elif zipfile.is_zipfile(source):
            self._zip = zipfile.ZipFile(source)
            names = [n for n in self._zip.namelist() if self._is_slice(n)]
        elif not source.exists():
            raise FileNotFoundError(source)
        else:
            raise ValueError(f"{source} is neither a directory nor a zip archive")
        if not names:
            raise ValueError(f"No slice image {io_config.slice_extensions} in {source}")
        self.names: list[str] = sorted(names, key=natural_key)
        self._first: NDArray | None = None

    def _is_slice(self, name: str) -> bool:
        """Tell whether an archive or directory entry is a slice image.

        Args:
            name: Entry name (POSIX path inside a zip archive).

        Returns:
            ``True`` for an image with a slice extension, outside ignored names and
            directories.
        """
        path = PurePosixPath(name)
        return (
            path.suffix.lower() in self._config.slice_extensions
            and not path.name.startswith(self._config.ignored_name_prefixes)
            and not set(path.parts) & set(self._config.ignored_directories)
        )

    def __len__(self) -> int:
        """Count the slices.

        Returns:
            Number of slices.
        """
        return len(self.names)

    def __iter__(self) -> "Iterator[NDArray]":
        """Iterate over the slices in order, decoding one at a time.

        Yields:
            Each slice (see :meth:`read`).
        """
        for i in range(len(self)):
            yield self.read(i)

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            This stack.
        """
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        """Close the underlying archive, if any.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc: That exception, if any.
            tb: Its traceback, if any.
        """
        self.close()

    def close(self) -> None:
        """Release the underlying archive handle, if any."""
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def read(self, index: int) -> "NDArray":
        """Decode one slice as a single-channel array, keeping its bit depth.

        Thread-safe: concurrent reads from the same archive are supported.

        Args:
            index: Slice index (negative indices allowed).

        Returns:
            The 2D slice (typically ``uint16`` for µCT data).

        Raises:
            ValueError: If the file cannot be decoded as an image.
        """
        name = self.names[index]
        if self._zip is not None:
            buffer = np.frombuffer(self._zip.read(name), dtype=np.uint8)
        else:
            buffer = np.fromfile(self.source / name, dtype=np.uint8)
        # IMREAD_ANYDEPTH alone = grayscale, original bit depth preserved.
        img = cast("NDArray | None", cv2.imdecode(buffer, cv2.IMREAD_ANYDEPTH))
        if img is None:
            raise ValueError(f"Cannot decode slice {name!r}")
        return img

    @property
    def slice_shape(self) -> tuple[int, int]:
        """``(height, width)`` of a slice, read from the first one."""
        height, width = self._first_slice.shape
        return height, width

    @property
    def dtype(self) -> np.dtype:
        """Pixel dtype of the slices, read from the first one."""
        return self._first_slice.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(depth, height, width)`` of the stack."""
        return (len(self), *self.slice_shape)

    @property
    def _first_slice(self) -> "NDArray":
        """First slice, decoded once and cached (it gives the shape and dtype)."""
        if self._first is None:
            self._first = self.read(0)
        return self._first
