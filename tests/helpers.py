"""Synthetic data and reference implementations shared by the tests."""

from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.config.loader import load_config
from pybones.segmentation.scales import resolve_pixel_params
from pybones.utils.constants import UINT16_MAX

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from pybones.config.schema import PyBonesConfig
    from pybones.segmentation.scales import PixelParams

#: Intensities of the synthetic slices (16-bit grey levels).
BACKGROUND, BONE, VESSEL = 3_000, 14_000, 32_000
SLICE_SHAPE = (256, 256)
#: Slices written by the ``slice_dir`` fixture (see ``conftest.py``).
N_SYNTHETIC_SLICES = 7
_NOISE_STD = 300
_FILLED = -1

#: Overrides adapting the default (full-size µCT) parameters to 256 px synthetic slices, at the
#: default 3 µm voxel: 21 px burst window, 400 px² burst core, 5 px² bone blobs, 50 voxels.
SMALL_IMAGE_OVERRIDES = (
    "segmentation.burst.window_xy_um=63",
    "segmentation.burst.min_area_um2=3600",
    "segmentation.bone.min_area_um2=45",
    "thresholds.histogram_stride=2",
    "pipeline.workers=2",
    "refinement.min_bone_um3=1350",
)

#: Neighbour offsets ``(dz, dy, dx)`` of 26- and 6-connectivity.
NEIGHBOURS_26 = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
)
NEIGHBOURS_6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def small_config(*extra: str) -> "PyBonesConfig":
    """Build the default configuration adapted to synthetic slices.

    Args:
        *extra: Further ``section.key=value`` overrides.

    Returns:
        The validated configuration.
    """
    return load_config(overrides=(*SMALL_IMAGE_OVERRIDES, *extra))


def small_params(*extra: str) -> "PixelParams":
    """Resolve the pixel-unit parameters of :func:`small_config`.

    Args:
        *extra: Further ``section.key=value`` overrides.

    Returns:
        The pixel-unit parameters.
    """
    return resolve_pixel_params(small_config(*extra))


def cli_overrides() -> list[str]:
    """Express :data:`SMALL_IMAGE_OVERRIDES` as ``pybones`` command-line options.

    Returns:
        ``["--set", override, ...]``.
    """
    return [item for override in SMALL_IMAGE_OVERRIDES for item in ("--set", override)]


def synthetic_slice(seed: int) -> "NDArray[np.uint16]":
    """Draw a slice with a bone band, a vessel, a small dim blob and a dense burst cluster.

    Args:
        seed: Seed of the noise and of the burst speckle positions.

    Returns:
        A ``SLICE_SHAPE`` uint16 slice.
    """
    rng = np.random.default_rng(seed)
    img = np.full(SLICE_SHAPE, BACKGROUND, dtype=np.float32)
    img[20:60, :] = BONE  # cortical band
    _ = cv2.circle(img, (60, 150), 6, BONE + 2_000, _FILLED)  # dim rim of a vessel...
    _ = cv2.circle(img, (60, 150), 4, VESSEL, _FILLED)  # ...around its bright core
    _ = cv2.circle(img, (120, 120), 3, BONE, _FILLED)  # small dim blob: ambiguous in 2D
    for y, x in rng.integers(170, 230, size=(140, 2)):  # burst: dense bright speckles
        _ = cv2.circle(img, (int(x), int(y)), 3, VESSEL, _FILLED)
    img += rng.normal(0, _NOISE_STD, size=SLICE_SHAPE).astype(np.float32)
    return np.clip(img, 0, UINT16_MAX).astype(np.uint16)


def flood_fill_components(
    volume: "NDArray[np.bool_]", offsets: "Sequence[tuple[int, int, int]]"
) -> int:
    """Count 3D connected components by breadth-first search (reference implementation).

    Args:
        volume: Binary volume.
        offsets: Neighbour offsets defining the connectivity.

    Returns:
        Number of components.
    """
    seen = np.zeros(volume.shape, dtype=bool)
    count = 0
    for start in zip(*np.nonzero(volume), strict=True):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        queue = deque([start])
        while queue:
            z, y, x = queue.popleft()
            for dz, dy, dx in offsets:
                n = (z + dz, y + dy, x + dx)
                inside = all(0 <= c < s for c, s in zip(n, volume.shape, strict=True))
                if inside and volume[n] and not seen[n]:
                    seen[n] = True
                    queue.append(n)
    return count
