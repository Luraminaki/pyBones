"""Tests of the streamed 3D tubularity (:mod:`pybones.segmentation.tubularity`)."""

from typing import TYPE_CHECKING

import numpy as np

from pybones.segmentation import tubularity as tubularity_module
from pybones.segmentation.geometry import VoxelGeometry
from pybones.segmentation.tubularity import (
    CHUNK_VOXELS,
    TubularityParams,
    quantize_tubularity,
    scale_kernels,
    slice_tubularity,
)
from pybones.utils.windows import mirror_index, sliding_windows

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from numpy.typing import NDArray

GEO = VoxelGeometry(z=1.0, y=1.0, x=1.0)
SHAPE = (21, 40, 40)
CENTRE = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)
BG, FG = 100.0, 1100.0
LEVEL = BG + 0.3 * (FG - BG)
TUBE_RADIUS2 = 4
PLATE_HALF_THICKNESS = 1

#: Indicator of a shape, from centred coordinate grids ``(z, y, x)``.
type ShapeFunction = Callable[[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]], NDArray]


def params(sigmas: tuple[float, ...] = (1.5,)) -> TubularityParams:
    """Build tubularity settings on unit voxels.

    Args:
        sigmas: Scales in µm (= voxels).

    Returns:
        The settings.
    """
    return TubularityParams(scales=tuple(scale_kernels(s, GEO) for s in sigmas))


def z_tube(_z: "NDArray[np.int64]", y: "NDArray[np.int64]", x: "NDArray[np.int64]") -> "NDArray":
    """Indicate a tube along z through the centre.

    Args:
        _z: Centred z grid (unused: the tube spans every slice).
        y: Centred y grid.
        x: Centred x grid.

    Returns:
        Boolean indicator, of shape ``(1, Y, X)`` (``ogrid`` grids broadcast).
    """
    return y**2 + x**2 <= TUBE_RADIUS2


def yz_plate(_z: "NDArray[np.int64]", _y: "NDArray[np.int64]", x: "NDArray[np.int64]") -> "NDArray":
    """Indicate a plate orthogonal to x through the centre.

    Args:
        _z: Centred z grid (unused).
        _y: Centred y grid (unused).
        x: Centred x grid.

    Returns:
        Boolean indicator, of shape ``(1, 1, X)`` (``ogrid`` grids broadcast).
    """
    return np.abs(x) <= PLATE_HALF_THICKNESS


def volume_with(shape_fn: ShapeFunction) -> "NDArray[np.uint16]":
    """Draw a bright shape on a dark background.

    Args:
        shape_fn: Indicator of the shape.

    Returns:
        ``SHAPE`` uint16 volume.
    """
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    inside = np.broadcast_to(shape_fn(zz - CENTRE[0], yy - CENTRE[1], xx - CENTRE[2]), SHAPE)
    return np.where(inside, FG, BG).astype(np.uint16)


def stream_tubularity(
    vol: "NDArray[np.uint16]", p: TubularityParams
) -> "list[NDArray[np.float32]]":
    """Compute the tubularity of every slice the way the pipeline streams it.

    Args:
        vol: Volume to analyse.
        p: Tubularity settings.

    Returns:
        One map per slice; voxels below :data:`LEVEL` are not analysed.
    """
    floats = (s.astype(np.float32) for s in vol)
    return [
        slice_tubularity(window, window[p.half_z] >= LEVEL, p)
        for window in sliding_windows(floats, len(vol), p.half_z)
    ]


def centre_tubularity(vol: "NDArray[np.uint16]") -> float:
    """Measure the streamed tubularity at the centre voxel.

    Args:
        vol: Volume to analyse.

    Returns:
        Tube-vs-plate ratio at the centre.
    """
    maps = stream_tubularity(vol, params())
    return float(maps[CENTRE[0]][CENTRE[1], CENTRE[2]])


def test_tube_vs_plate() -> None:
    """A tube scores near 1, a plate near 0."""
    assert centre_tubularity(volume_with(z_tube)) > 0.9
    assert centre_tubularity(volume_with(yz_plate)) < 0.1


def test_streaming_matches_an_explicit_window() -> None:
    """Streaming equals the per-slice computation on mirrored windows, ends included."""
    rng = np.random.default_rng(3)
    vol = (BG + rng.random(SHAPE) * (FG - BG)).astype(np.uint16)
    p = params((1.0, 2.0))
    streamed = stream_tubularity(vol, p)
    floats = vol.astype(np.float32)
    for z in (0, 3, 10, 20):
        idx = [mirror_index(k, len(vol)) for k in range(z - p.half_z, z + p.half_z + 1)]
        expected = slice_tubularity([floats[i] for i in idx], vol[z] >= LEVEL, p)
        np.testing.assert_array_equal(streamed[z], expected)


def test_chunking_does_not_change_the_result(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Processing the voxels in small chunks gives the same map."""
    vol = volume_with(z_tube)
    window = [s.astype(np.float32) for s in vol[CENTRE[0] - 5 : CENTRE[0] + 6]]
    mask = vol[CENTRE[0]] >= LEVEL
    whole = slice_tubularity(window, mask, params())
    small_chunk = 3
    assert small_chunk < CHUNK_VOXELS
    monkeypatch.setattr(tubularity_module, "CHUNK_VOXELS", small_chunk)
    np.testing.assert_array_equal(slice_tubularity(window, mask, params()), whole)


def test_only_masked_voxels_are_analysed() -> None:
    """Voxels outside the mask stay 0."""
    vol = volume_with(z_tube)
    window = [s.astype(np.float32) for s in vol[CENTRE[0] - 5 : CENTRE[0] + 6]]
    mask = np.zeros(SHAPE[1:], bool)
    mask[CENTRE[1:]] = True
    out = slice_tubularity(window, mask, params())
    assert out[CENTRE[1:]] > 0.9
    assert np.count_nonzero(out) == 1


def test_quantisation_clips_and_rounds() -> None:
    """Quantisation to uint8 clips to ``[0, 1]`` and rounds."""
    q = quantize_tubularity(np.array([0.0, 0.5, 1.0, 1.5], np.float32))
    assert q.tolist() == [0, 128, 255, 255]
