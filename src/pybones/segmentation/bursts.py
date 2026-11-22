"""Burst cores from the bright-pixel density in a 3D box, streamed along z.

A burst (contrast agent leaked out of a ruptured vessel) is a 3D blob. Measuring
its bright-pixel density per slice makes it flicker along z: near the top and
bottom of a cluster, a slice only shows a sparse cross-section. Averaging the
per-slice (coarse-grid) densities over the burst window's slices gives the
density in a 3D box, which is consistent from one slice to the next.

The burst core of a slice is the union of

* its own (2D) dense areas: a cube-shaped 3D box erodes the ends of a cluster
  along z by up to half its depth, the 2D part never loses a cluster;
* the 3D-box dense areas that contain contrast agent in this slice: they fill the
  slices where a cluster's cross-section is too sparse in 2D (its tips, or
  flickering slices), and the guard stops a dense cluster's box from painting
  slices past its end.

The burst boundary grows the core over a finer density (a Gaussian in-plane),
averaged over the same slices: a 3D density. Measured per slice, a vessel lying in
the slice is a dense strip, and the boundary would follow it along its length.

The running sum costs one addition and one subtraction per slice whatever the
window size. Buffered: the coarse densities of the window, and the masks of the
slices waiting for their upper neighbours, bit-packed (8x smaller).
"""

import itertools
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pybones.imaging import morphology as morph
from pybones.segmentation.classifier import SliceMasks, upsample_grid

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

    from pybones.segmentation.classifier import SliceFeatures
    from pybones.segmentation.scales import PixelParams

#: Planes of the stacked coarse maps summed over the window: core density, boundary density.
_CORE, _BOUNDARY = 0, 1


@dataclass(frozen=True, slots=True)
class SliceBursts:
    """What the final labelling of a slice needs from the 3D burst detection.

    Attributes:
        masks: Masks of the slice.
        core: Pixel mask of the dense burst cores.
        boundary: 3D boundary density on the coarse grid (mean over the burst window).
    """

    masks: SliceMasks
    core: "NDArray[np.bool_]"
    boundary: "NDArray[np.float32]"


@dataclass(frozen=True, slots=True)
class _PackedSlice:
    """Masks of a slice waiting for its upper neighbours, bit-packed.

    Attributes:
        shape: Pixel shape of the slice.
        foreground: Packed foreground mask.
        bright: Packed bright-core mask.
        vessel: Packed vessel mask.
        core_2d: The slice's own (2D) dense cells, on the coarse grid (already small).
        tubularity: Quantised tubularity map, if computed (carried as is).
    """

    shape: tuple[int, int]
    foreground: "NDArray[np.uint8]"
    bright: "NDArray[np.uint8]"
    vessel: "NDArray[np.uint8]"
    core_2d: "NDArray[np.bool_]"
    tubularity: "NDArray[np.uint8] | None"

    @classmethod
    def pack(cls, feat: "SliceFeatures", params: "PixelParams") -> "_PackedSlice":
        """Pack the masks of a slice and threshold its 2D density.

        Args:
            feat: Features of the slice.
            params: Segmentation parameters in pixel units (burst density).

        Returns:
            The packed slice.
        """
        masks = feat.masks
        return cls(
            shape=(masks.foreground.shape[0], masks.foreground.shape[1]),
            foreground=np.packbits(masks.foreground),
            bright=np.packbits(masks.bright),
            vessel=np.packbits(masks.vessel),
            core_2d=feat.burst_density >= params.burst_density,
            tubularity=masks.tubularity,
        )

    def _unpack(self, bits: "NDArray[np.uint8]") -> "NDArray[np.bool_]":
        """Unpack one mask.

        Args:
            bits: Mask packed by :func:`numpy.packbits`.

        Returns:
            The boolean mask of shape :attr:`shape`.
        """
        size = self.shape[0] * self.shape[1]
        return np.unpackbits(bits, count=size).reshape(self.shape).view(np.bool_)

    def unpack(self) -> SliceMasks:
        """Restore the masks of the slice.

        Returns:
            The unpacked masks.
        """
        return SliceMasks(
            foreground=self._unpack(self.foreground),
            bright=self._unpack(self.bright),
            vessel=self._unpack(self.vessel),
            tubularity=self.tubularity,
        )


def iter_burst_cores(
    features: "Iterable[SliceFeatures]", params: "PixelParams"
) -> "Iterator[SliceBursts]":
    """Give each slice its burst core (2D core united with the guarded 3D one) and boundary.

    At the stack ends the box is truncated to the available slices (the densities
    are means over existing slices).

    Args:
        features: Slice features in z order.
        params: Segmentation parameters in pixel units (burst grid, window, density).

    Yields:
        The burst inputs of each slice, in z order.
    """
    stream = iter(features)
    first = next(stream, None)
    if first is None:
        return
    half = params.burst_window_z // 2
    waiting: deque[_PackedSlice] = deque()  # slices next_out.. not yet emitted
    maps: deque[NDArray[np.float32]] = deque()  # stacked coarse maps of slices start..last
    total = np.zeros((2, *first.burst_density.shape), dtype=np.float64)  # sum of `maps`
    start = 0  # z index of maps[0]
    next_out = 0

    def emit() -> SliceBursts:
        """Finish slice ``next_out``, whose upper neighbours are all in the window.

        Returns:
            The burst inputs of the slice.
        """
        nonlocal start, next_out
        core_3d = total[_CORE] >= params.burst_density * len(maps)
        boundary = (total[_BOUNDARY] / len(maps)).astype(np.float32)
        next_out += 1
        while start < next_out - half:
            total[...] -= maps.popleft()
            start += 1
        packed = waiting.popleft()
        masks = packed.unpack()
        core_2d = upsample_grid(packed.core_2d, params, packed.shape)
        guarded_3d = morph.reconstruct(
            masks.bright, upsample_grid(core_3d, params, packed.shape), params.connectivity
        )
        return SliceBursts(masks, core_2d | guarded_3d, boundary)

    for z, feat in enumerate(itertools.chain((first,), stream)):
        waiting.append(_PackedSlice.pack(feat, params))
        stacked = np.stack([feat.burst_density, feat.boundary_density])
        maps.append(stacked)
        total[...] += stacked
        if next_out <= z - half:
            yield emit()
    while waiting:
        yield emit()
