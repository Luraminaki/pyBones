"""Richardson-Lucy deconvolution of a Gaussian blur, whole or streamed block by block.

The scanner blurs the image with a (measured) Gaussian point-spread function. Richardson-Lucy
undoes part of it by repeated corrections: blur the estimate, compare with the measured image,
multiply the estimate by the blurred ratio. A Gaussian is separable, so each blur is three 1D
passes (OpenCV in-plane, a weighted sum of slices along z).

Streaming: a voxel's value after ``n`` iterations depends only on the image within
``2 n r`` voxels (``r``, the kernel radius: two blurs per iteration). Blocks carrying that margin
on every side therefore give exactly the whole-volume result in their core, the stack's own
faces being mirrored in both cases. Blocks are processed by threads (OpenCV and numpy release
the GIL) and only their core is kept, so memory stays bounded whatever the stack size.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.utils.parallel import ordered_map

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

#: Kernel half-width in sigmas (a Gaussian keeps 99.7 % of its weight within 3 sigmas).
TRUNCATE_SIGMAS = 3.0
#: Smallest blurred estimate a ratio divides by (grey levels are >= 0).
DIVISION_FLOOR = 1e-3
#: Blocks queued beyond the busy workers.
_EXTRA_BLOCKS_IN_FLIGHT = 2
#: One-tap kernel: no filtering along that direction (read-only: shared).
_NO_KERNEL = np.ones((1, 1), np.float32)
_NO_KERNEL.flags.writeable = False


@dataclass(frozen=True, slots=True)
class GaussianPsf:
    """A separable Gaussian point-spread function, in voxels.

    Attributes:
        sigma: Standard deviation along ``(z, y, x)``, in voxels (0 = no blur on that axis).
    """

    sigma: tuple[float, float, float]

    @property
    def radius(self) -> tuple[int, int, int]:
        """Half-width of the sampled kernel along each axis, in voxels."""
        rz, ry, rx = (math.ceil(TRUNCATE_SIGMAS * s) for s in self.sigma)
        return rz, ry, rx

    def kernel(self, axis: int) -> "NDArray[np.float32]":
        """Sample the normalised 1D kernel of one axis.

        Args:
            axis: 0 (z), 1 (y) or 2 (x).

        Returns:
            ``(2 r + 1, 1)`` weights summing to 1 (a single 1 for an axis without blur).
        """
        r = self.radius[axis]
        if r == 0:
            return _NO_KERNEL
        return np.asarray(
            cv2.getGaussianKernel(2 * r + 1, self.sigma[axis], cv2.CV_32F), np.float32
        )

    def margin(self, iterations: int) -> tuple[int, int, int]:
        """Margin a block needs for an exact core after ``iterations`` (two blurs each).

        Args:
            iterations: Richardson-Lucy iterations.

        Returns:
            Margin along ``(z, y, x)``, in voxels.
        """
        mz, my, mx = (2 * iterations * r for r in self.radius)
        return mz, my, mx


def gaussian_blur(volume: "NDArray[np.float32]", psf: GaussianPsf) -> "NDArray[np.float32]":
    """Blur a volume, its faces mirrored (``abc|cba``).

    Args:
        volume: ``(Z, Y, X)`` float32 volume.
        psf: The Gaussian.

    Returns:
        The blurred volume (a new array).
    """
    rz, ry, rx = psf.radius
    nz, ny, nx = volume.shape
    padded = np.pad(volume, ((rz, rz), (ry, ry), (rx, rx)), mode="symmetric")
    kz, ky, kx = (psf.kernel(axis) for axis in range(3))
    planes = np.empty((padded.shape[0], ny, nx), np.float32)
    for z, plane in enumerate(padded):
        blurred = cv2.sepFilter2D(plane, cv2.CV_32F, kx, ky, borderType=cv2.BORDER_CONSTANT)
        planes[z] = blurred[ry : ry + ny, rx : rx + nx]
    # Along z: one OpenCV pass over the planes seen as a 2D image, one row per slice.
    rows = planes.reshape(planes.shape[0], ny * nx)
    along_z = cv2.sepFilter2D(rows, cv2.CV_32F, _NO_KERNEL, kz, borderType=cv2.BORDER_CONSTANT)
    return np.asarray(along_z, np.float32)[rz : rz + nz].reshape(nz, ny, nx)  # rows: a view


def richardson_lucy(
    image: "NDArray[np.float32]", psf: GaussianPsf, iterations: int
) -> "NDArray[np.float32]":
    """Deconvolve a volume (the Gaussian is symmetric: it is its own mirror).

    Args:
        image: ``(Z, Y, X)`` measured volume, grey levels >= 0.
        psf: The blur to undo.
        iterations: Corrections (0 returns a copy of the image).

    Returns:
        The estimate of the unblurred volume.
    """
    estimate = image.astype(np.float32, copy=True)
    for _ in range(iterations):
        blurred = np.maximum(gaussian_blur(estimate, psf), DIVISION_FLOOR)
        ratio = (image / blurred).astype(np.float32, copy=False)
        estimate *= gaussian_blur(ratio, psf)
    return estimate


@dataclass(frozen=True, slots=True)
class _Block:
    """One block of a slab: its core and the region read with its margin.

    Attributes:
        core: Core slices ``(z, y, x)`` in volume coordinates.
        outer: Core plus margin (clipped to the volume), in volume coordinates.
    """

    core: tuple[slice, slice, slice]
    outer: tuple[slice, slice, slice]

    @property
    def inner(self) -> tuple[slice, slice, slice]:
        """The core's slices relative to :attr:`outer`."""
        sz, sy, sx = (
            slice(c.start - o.start, c.stop - o.start)
            for c, o in zip(self.core, self.outer, strict=True)
        )
        return sz, sy, sx


def _spans(length: int, step: int, margin: int) -> "Iterator[tuple[slice, slice]]":
    """Cut an axis into cores of ``step`` voxels, each with its margin (clipped).

    Args:
        length: Axis length.
        step: Core length.
        margin: Margin on each side.

    Yields:
        ``(core, outer)`` slices along the axis.
    """
    for start in range(0, length, step):
        stop = min(length, start + step)
        yield slice(start, stop), slice(max(0, start - margin), min(length, stop + margin))


def iter_deconvolved_above(
    volume: "NDArray[np.uint16]",
    psf: GaussianPsf,
    iterations: int,
    level: float,
    block: tuple[int, int],
    workers: int,
) -> "Iterator[NDArray[np.bool_]]":
    """Deconvolve a volume block by block and yield, slice by slice, where it reaches a level.

    Args:
        volume: ``(Z, Y, X)`` grey levels (typically memory-mapped: only blocks are read).
        psf: The blur to undo.
        iterations: Richardson-Lucy iterations.
        level: Grey level the deconvolved value must reach.
        block: Core size of a block, ``(slices, pixels)``: slabs of that many slices, cut in
            square tiles of that many pixels.
        workers: Threads deconvolving blocks.

    Yields:
        For each slice, in order, the pixels whose deconvolved value is >= ``level``; equal to
        thresholding :func:`richardson_lucy` of the whole volume.
    """
    depth, height, width = volume.shape
    slab_slices, tile_px = block
    mz, my, mx = psf.margin(iterations)

    def above(region: _Block) -> "NDArray[np.bool_]":
        """Deconvolve one block and keep its core's mask.

        Args:
            region: The block.

        Returns:
            The core's mask.
        """
        image = np.asarray(volume[region.outer], dtype=np.float32)
        return richardson_lucy(image, psf, iterations)[region.inner] >= level

    for z_core, z_outer in _spans(depth, slab_slices, mz):
        blocks = [
            _Block((z_core, y_core, x_core), (z_outer, y_outer, x_outer))
            for y_core, y_outer in _spans(height, tile_px, my)
            for x_core, x_outer in _spans(width, tile_px, mx)
        ]
        slab = np.zeros((z_core.stop - z_core.start, height, width), bool)
        for region, mask in zip(
            blocks, ordered_map(above, blocks, workers, _EXTRA_BLOCKS_IN_FLIGHT), strict=True
        ):
            slab[(slice(None), *region.core[1:])] = mask
        yield from slab
