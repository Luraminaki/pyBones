"""On-disk ``uint8`` label volumes (``.npy``), written slice by slice and read memory-mapped."""

from typing import TYPE_CHECKING, Self

import cv2
import numpy as np

from pybones.utils.constants import UINT8_LEVELS

if TYPE_CHECKING:
    import io
    from pathlib import Path
    from types import TracebackType

    from numpy.typing import NDArray

_NPY_UINT8 = "|u1"


class LabelVolumeWriter:
    """Stream ``(Z, Y, X)`` ``uint8`` slices into a ``.npy`` file.

    Plain sequential writes are used rather than a writable memmap so that no
    dirty mapped pages accumulate in the process RSS. A per-value histogram of
    the written voxels is maintained on the fly.

    Example:
        >>> with LabelVolumeWriter(path, (z, y, x)) as writer:
        ...     for lab in slices:
        ...         writer.write(lab)
    """

    def __init__(self, path: "Path", shape: tuple[int, int, int]) -> None:
        """Prepare a writer (the file is created on ``__enter__``).

        Args:
            path: Destination ``.npy`` file (overwritten).
            shape: ``(depth, height, width)`` of the volume.
        """
        self.path: Path = path
        self.shape: tuple[int, int, int] = shape
        self.written: int = 0
        self.counts: NDArray[np.int64] = np.zeros(UINT8_LEVELS, dtype=np.int64)
        self._fh: io.BufferedWriter | None = None

    def __enter__(self) -> Self:
        """Create the file and write the ``.npy`` header.

        Returns:
            This writer.
        """
        self._fh = self.path.open("wb")
        header = {"descr": _NPY_UINT8, "fortran_order": False, "shape": self.shape}
        np.lib.format.write_array_header_1_0(self._fh, header)
        return self

    def write(self, labels: "NDArray[np.uint8]") -> None:
        """Append the next slice.

        Args:
            labels: ``(height, width)`` uint8 slice.

        Raises:
            ValueError: If the slice has the wrong shape or dtype, or the volume is full.
            RuntimeError: If the writer is used outside its ``with`` block.
        """
        if labels.shape != self.shape[1:] or labels.dtype != np.uint8:
            raise ValueError(f"Expected uint8 {self.shape[1:]}, got {labels.dtype} {labels.shape}")
        if self.written >= self.shape[0]:
            raise ValueError(f"Volume already holds {self.shape[0]} slices")
        if self._fh is None:
            raise RuntimeError("LabelVolumeWriter.write() called outside its `with` block")
        _ = self._fh.write(np.ascontiguousarray(labels).data)
        self.counts += (
            cv2.calcHist([labels], [0], None, [UINT8_LEVELS], [0, UINT8_LEVELS])
            .ravel()
            .astype(np.int64)
        )
        self.written += 1

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        """Close the file; check completeness when no error occurred.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc: That exception, if any.
            tb: Its traceback, if any.

        Raises:
            RuntimeError: If fewer slices than announced were written.
        """
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        if exc_type is None and self.written != self.shape[0]:
            raise RuntimeError(f"Wrote {self.written} slices, expected {self.shape[0]}")


def open_volume(path: "Path") -> "NDArray[np.uint8]":
    """Memory-map a label volume read-only (no RAM cost until slices are touched).

    Args:
        path: ``.npy`` label volume.

    Returns:
        The read-only ``(depth, height, width)`` volume.
    """
    return np.load(path, mmap_mode="r")
