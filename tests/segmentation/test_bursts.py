"""Tests of the streamed 3D burst cores (:mod:`pybones.segmentation.bursts`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.segmentation.bursts import iter_burst_cores
from pybones.segmentation.classifier import (
    SliceFeatures,
    SliceMasks,
    coarse_density,
    finalize_slice,
    upsample_grid,
)
from pybones.segmentation.labels import Label
from tests.helpers import small_params

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.segmentation.scales import PixelParams

SHAPE = (80, 80)
#: Slice thickness of the default configuration, in µm.
SLICE_UM = 3


def masks(bright: "NDArray[np.bool_]") -> SliceMasks:
    """Build slice masks whose only content is ``bright`` (no foreground, no vessel).

    Args:
        bright: Bright-core mask.

    Returns:
        The masks.
    """
    off = np.zeros(SHAPE, bool)
    return SliceMasks(foreground=off, bright=bright, vessel=off, tubularity=None)


def features_from_densities(
    densities: "NDArray[np.floating]", bright: "NDArray[np.bool_] | None" = None
) -> list[SliceFeatures]:
    """Build slice features with the given coarse densities.

    Args:
        densities: ``(Z, gy, gx)`` coarse densities, one grid per slice.
        bright: Bright mask of every slice (default: bright everywhere).

    Returns:
        One feature set per slice.
    """
    on = np.ones(SHAPE, bool) if bright is None else bright
    return [SliceFeatures(masks(on), d.astype(np.float32)) for d in densities]


def grid_shape(params: "PixelParams") -> tuple[int, int]:
    """Compute the coarse density grid shape of a ``SHAPE`` slice.

    Args:
        params: Pixel-unit parameters (grid block size).

    Returns:
        ``(rows, columns)`` of the grid.
    """
    gy, gx = params.burst_grid
    return -(-SHAPE[0] // gy), -(-SHAPE[1] // gx)


@pytest.mark.parametrize("window_z", [1, 3, 5])
def test_3d_core_is_the_mean_density_over_the_window(window_z: int) -> None:
    """The core is the 2D core united with the box mean over the (truncated) z window."""
    params = small_params(f"segmentation.burst.window_z_um={SLICE_UM * window_z}")
    assert params.burst_window_z == window_z
    rng = np.random.default_rng(window_z)
    dens = rng.random((9, *grid_shape(params)))
    out = list(iter_burst_cores(features_from_densities(dens), params))
    assert len(out) == len(dens)
    half = window_z // 2
    for z, (_, core) in enumerate(out):
        lo, hi = max(0, z - half), min(len(dens), z + half + 1)
        core_3d = dens[lo:hi].mean(axis=0) >= params.burst_density
        core_2d = dens[z] >= params.burst_density
        np.testing.assert_array_equal(core, upsample_grid(core_3d | core_2d, params, SHAPE))


def test_sparse_end_of_cluster_is_caught_in_3d_only() -> None:
    """A sparse slice at the tip of a dense cluster is a burst thanks to the 3D box."""
    params = small_params(f"segmentation.burst.window_z_um={3 * SLICE_UM}")
    dens = np.zeros((3, *grid_shape(params)))
    dens[0:2, 2:6, 2:6] = 0.9
    dens[2, 2:6, 2:6] = 0.2  # the cluster's tip: sparse cross-section
    cores = [core for _, core in iter_burst_cores(features_from_densities(dens), params)]
    gy, gx = params.burst_grid
    pixel = (3 * gy, 3 * gx)
    assert cores[2][pixel]  # (0.9 + 0.2) / 2 >= density
    assert dens[2, 3, 3] < params.burst_density  # 2D would miss it


def test_3d_core_needs_contrast_agent_in_the_slice() -> None:
    """The 3D core does not paint a slice that holds no contrast agent (the guard)."""
    params = small_params(f"segmentation.burst.window_z_um={3 * SLICE_UM}")
    dens = np.zeros((3, *grid_shape(params)))
    dens[0:2, 2:6, 2:6] = 0.9  # dense below; slice 2 has no bright pixel at all
    feats = features_from_densities(dens)
    feats[2] = SliceFeatures(masks(np.zeros(SHAPE, bool)), dens[2].astype(np.float32))
    cores = [core for _, core in iter_burst_cores(feats, params)]
    assert cores[1].any()
    assert not cores[2].any()


def test_empty_stream_yields_nothing() -> None:
    """No slice in, no core out."""
    assert list(iter_burst_cores([], small_params())) == []


def test_coarse_density_is_a_box_mean_of_bright_pixels() -> None:
    """No bright pixel gives zero density; all bright gives one."""
    params = small_params()
    bright = np.zeros(SHAPE, bool)
    assert not coarse_density(bright, params).any()
    np.testing.assert_allclose(coarse_density(~bright, params)[5, 5], 1.0)


def test_finalize_turns_a_core_into_a_burst() -> None:
    """A dense core over bright pixels becomes a burst region."""
    params = small_params()
    cluster = np.s_[5:30, 5:30]
    bright = np.zeros(SHAPE, bool)
    bright[cluster] = True
    slice_masks = SliceMasks(foreground=bright, bright=bright, vessel=bright, tubularity=None)
    core = np.zeros(SHAPE, bool)
    core[cluster] = True
    assert (finalize_slice(slice_masks, core, params)[cluster] == Label.BURST).all()
