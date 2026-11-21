"""Distance to the background at chosen voxels (the :class:`~.protocols.DistanceProbe` step).

The squared 3D Euclidean distance separates by axis: at voxel ``(z, y, x)`` it is the
minimum over slices ``z'`` of the squared in-plane distance in slice ``z'`` plus
``((z - z') * dz)^2``. :class:`SliceDistanceProbe` uses that identity directly; it only
needs the slices within the largest radius of each point. :class:`ScipyDistanceProbe`
runs SciPy's exact 3D transform on tiles of the volume instead: two independent ways to
the same numbers, which the tests compare.
"""

import math
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _window_slices(radius_um: float, slice_um: float) -> int:
    """Count the slices on each side of a point that can hold its nearest background voxel.

    Args:
        radius_um: Largest distance measured.
        slice_um: Slice spacing.

    Returns:
        ``ceil(radius / spacing)``.
    """
    return math.ceil(radius_um / slice_um)


class SliceDistanceProbe:
    """Per-slice OpenCV distance maps (exact 2D), combined along z only at the points.

    Pros: exact; fast (~0.4 s on the sample: distances are combined only at the points);
    memory is one distance map and the points.
    Cons: square pixels only (OpenCV's transform has a single spacing; use the SciPy backend
    otherwise); a slice's whole map is computed even when few points need it.

    Attributes:
        max_um: Largest distance measured (beyond it, distances are capped).
    """

    def __init__(self, max_um: float) -> None:
        """Set the largest distance measured.

        Args:
            max_um: Cap of the distances, in µm.
        """
        self.max_um: float = max_um

    def distances(
        self,
        mask: "NDArray[np.bool_]",
        points: "NDArray[np.integer]",
        voxel_um: tuple[float, float, float],
    ) -> "NDArray[np.float32]":
        """Measure the distance from each point to the nearest background voxel.

        Args:
            mask: ``(Z, Y, X)`` mask.
            points: ``(N, 3)`` voxel indices ``(z, y, x)``.
            voxel_um: Voxel size ``(z, y, x)`` in µm.

        Returns:
            ``(N,)`` distances in µm, capped at :attr:`max_um`.

        Raises:
            ValueError: If the pixels are not square.
        """
        dz, dy, dx = voxel_um
        if not math.isclose(dy, dx):
            raise ValueError(f"square pixels needed (y {dy} µm, x {dx} µm): use the scipy backend")
        best = np.full(len(points), self.max_um**2, dtype=np.float64)
        if len(points) == 0:
            return best.astype(np.float32)
        window = _window_slices(self.max_um, dz)
        # Points sorted by z once: each slice finds the ones within reach by bisection.
        order = np.argsort(points[:, 0], kind="stable")
        z_sorted = points[order, 0]
        first = max(0, int(z_sorted[0]) - window)
        last = min(mask.shape[0], int(z_sorted[-1]) + window + 1)
        for z in range(first, last):
            lo, hi = np.searchsorted(z_sorted, [z - window, z + window + 1])
            if lo == hi:
                continue
            plane = np.ascontiguousarray(mask[z], dtype=np.uint8)
            if plane.all():  # no background in this slice: nothing to measure to
                continue
            near = order[lo:hi]
            inside = cast(
                "NDArray[np.float32]",
                cv2.distanceTransform(plane, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
            )
            in_plane = inside[points[near, 1], points[near, 2]].astype(np.float64) * dy
            across = ((points[near, 0] - z) * dz).astype(np.float64)
            best[near] = np.minimum(best[near], in_plane**2 + across**2)
        return np.sqrt(best).astype(np.float32)


class ScipyDistanceProbe:
    """SciPy's exact 3D Euclidean distance transform, on tiles of the volume.

    Each tile (``tile_slices`` x ``tile_pixels`` x ``tile_pixels`` voxels) is extended
    by the cap on every side, so the nearest background voxel of any point within the cap
    lies in it: the result is exact up to :attr:`max_um`, and each transform stays small
    (a whole-volume transform would take several GB).

    Pros: exact for any voxel shape; an independent implementation (the tests compare it).
    Cons: ~100x slower than :class:`SliceDistanceProbe` (~39 s on the sample): the margins
    are recomputed for every tile (about three times the volume's work at the defaults),
    and SciPy works in float64.

    Attributes:
        max_um: Largest distance measured (beyond it, distances are capped).
        tile_slices: Depth of a tile, in slices.
        tile_pixels: Side of a tile in-plane, in pixels.
    """

    def __init__(self, max_um: float, tile_slices: int, tile_pixels: int) -> None:
        """Set the cap and the tile size.

        Args:
            max_um: Cap of the distances, in µm.
            tile_slices: Depth of a tile, in slices.
            tile_pixels: Side of a tile in-plane, in pixels.
        """
        self.max_um: float = max_um
        self.tile_slices: int = tile_slices
        self.tile_pixels: int = tile_pixels

    def distances(
        self,
        mask: "NDArray[np.bool_]",
        points: "NDArray[np.integer]",
        voxel_um: tuple[float, float, float],
    ) -> "NDArray[np.float32]":
        """Measure the distance from each point to the nearest background voxel.

        Args:
            mask: ``(Z, Y, X)`` mask.
            points: ``(N, 3)`` voxel indices ``(z, y, x)``.
            voxel_um: Voxel size ``(z, y, x)`` in µm.

        Returns:
            ``(N,)`` distances in µm, capped at :attr:`max_um`.
        """
        out = np.full(len(points), self.max_um, dtype=np.float32)
        if len(points) == 0:
            return out
        sizes = (self.tile_slices, self.tile_pixels, self.tile_pixels)
        margins = [_window_slices(self.max_um, spacing) for spacing in voxel_um]
        # Points grouped by tile once (sorted by tile, then split where the tile changes).
        keys, which = np.unique(points // np.asarray(sizes), axis=0, return_inverse=True)
        order = np.argsort(which.ravel(), kind="stable")
        bounds = np.searchsorted(which.ravel()[order], np.arange(len(keys) + 1))
        for k, key in enumerate(keys):
            inside = order[bounds[k] : bounds[k + 1]]
            lo = [max(0, int(t) * s - m) for t, s, m in zip(key, sizes, margins, strict=True)]
            hi = [
                min(n, (int(t) + 1) * s + m)
                for t, s, m, n in zip(key, sizes, margins, mask.shape, strict=True)
            ]
            crop = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
            if crop.all():  # no background within reach: the cap stands
                continue
            field = ndimage.distance_transform_edt(crop, sampling=voxel_um)
            local = points[inside] - np.asarray(lo)
            out[inside] = np.minimum(field[local[:, 0], local[:, 1], local[:, 2]], self.max_um)
        return out
