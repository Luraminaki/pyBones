"""Tests of the streamed 3D burst cores (:mod:`pybones.segmentation.bursts`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.config.loader import load_config
from pybones.segmentation.bursts import iter_burst_cores
from pybones.segmentation.classifier import (
    SliceFeatures,
    SliceMasks,
    boundary_density,
    coarse_density,
    finalize_slice,
    grid_blocks,
    upsample_grid,
)
from pybones.segmentation.labels import Label
from pybones.segmentation.scales import resolve_pixel_params
from tests.helpers import small_params, truncated_window

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.segmentation.scales import PixelParams

SHAPE = (80, 80)
#: Slice thickness of the default configuration, in µm.
SLICE_UM = 3
#: A slice wide enough for a burst cluster and a long vessel leaving it.
WIDE = (120, 200)
CLUSTER = np.s_[30:90, 10:70]
#: A vessel lying in the slice (30 µm across), leaving the cluster to the right.
VESSEL = np.s_[55:65, 70:190]
#: The part of that vessel more than three boundary sigmas (36 µm) away from the cluster.
VESSEL_FAR = np.s_[55:65, 120:185]


def masks(bright: "NDArray[np.bool_]") -> SliceMasks:
    """Build slice masks whose only content is ``bright`` (no foreground, no vessel).

    Args:
        bright: Bright-core mask.

    Returns:
        The masks.
    """
    off = np.zeros(bright.shape, bool)
    return SliceMasks(foreground=off, bright=bright, vessel=off, tubularity=None)


def features_from_densities(
    densities: "NDArray[np.floating]",
    bright: "NDArray[np.bool_] | None" = None,
    boundaries: "NDArray[np.floating] | None" = None,
) -> list[SliceFeatures]:
    """Build slice features with the given coarse densities.

    Args:
        densities: ``(Z, gy, gx)`` coarse densities, one grid per slice.
        bright: Bright mask of every slice (default: bright everywhere).
        boundaries: ``(Z, gy, gx)`` boundary densities (default: zero).

    Returns:
        One feature set per slice.
    """
    on = np.ones(SHAPE, bool) if bright is None else bright
    fine = np.zeros_like(densities) if boundaries is None else boundaries
    return [
        SliceFeatures(masks(on), d.astype(np.float32), b.astype(np.float32))
        for d, b in zip(densities, fine, strict=True)
    ]


def features_of(bright: "NDArray[np.bool_]", params: "PixelParams") -> SliceFeatures:
    """Measure the burst densities of a slice holding only ``bright``.

    Args:
        bright: Bright-core mask.
        params: Pixel-unit parameters.

    Returns:
        The slice's features.
    """
    blocks = grid_blocks(bright, params)
    return SliceFeatures(
        masks(bright), coarse_density(blocks, params), boundary_density(blocks, params)
    )


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
    for z, item in enumerate(out):
        lo, hi = truncated_window(z, window_z, len(dens))
        core_3d = dens[lo:hi].mean(axis=0) >= params.burst_density
        core_2d = dens[z] >= params.burst_density
        np.testing.assert_array_equal(item.core, upsample_grid(core_3d | core_2d, params, SHAPE))


@pytest.mark.parametrize("window_z", [1, 3, 5])
def test_3d_boundary_is_the_mean_boundary_density_over_the_window(window_z: int) -> None:
    """The boundary density of a slice is the mean over the same (truncated) z window."""
    params = small_params(f"segmentation.burst.window_z_um={SLICE_UM * window_z}")
    rng = np.random.default_rng(window_z)
    dens = rng.random((9, *grid_shape(params)))
    fine = rng.random(dens.shape)
    out = list(iter_burst_cores(features_from_densities(dens, boundaries=fine), params))
    for z, item in enumerate(out):
        lo, hi = truncated_window(z, window_z, len(dens))
        np.testing.assert_allclose(item.boundary, fine[lo:hi].mean(axis=0), rtol=1e-6)


@pytest.mark.parametrize(("slices", "taken"), [(1, 1.0), (9, 0.0)])
def test_a_vessel_lying_in_one_slice_does_not_carry_the_burst(slices: int, taken: float) -> None:
    """A thick vessel in the slice plane is dense in 2D (the burst follows it), not in 3D."""
    params = resolve_pixel_params(
        load_config(overrides=[f"segmentation.burst.window_z_um={SLICE_UM * slices}"])
    )
    assert params.burst_window_z == slices
    cluster = np.zeros(WIDE, bool)
    cluster[CLUSTER] = True
    with_vessel = cluster.copy()
    with_vessel[VESSEL] = True
    stack = [cluster] * slices
    stack[slices // 2] = with_vessel
    middle = list(iter_burst_cores([features_of(b, params) for b in stack], params))[slices // 2]
    labels = finalize_slice(middle.masks, middle.core, middle.boundary, params)
    assert (labels[CLUSTER] == Label.BURST).all()
    assert (labels[VESSEL_FAR] == Label.BURST).mean() == taken


def dense_cluster_below_slice_2() -> "tuple[PixelParams, NDArray[np.float64]]":
    """A dense burst cluster on slices 0 and 1 of a three-slice stack, 3D window of 3 slices.

    Returns:
        The parameters, and the burst density per grid cell (slice 2 left empty).
    """
    params = small_params(f"segmentation.burst.window_z_um={3 * SLICE_UM}")
    dens = np.zeros((3, *grid_shape(params)))
    dens[0:2, 2:6, 2:6] = 0.9
    return params, dens


def test_sparse_end_of_cluster_is_caught_in_3d_only() -> None:
    """A sparse slice at the tip of a dense cluster is a burst thanks to the 3D box."""
    params, dens = dense_cluster_below_slice_2()
    dens[2, 2:6, 2:6] = 0.2  # the cluster's tip: sparse cross-section
    cores = [item.core for item in iter_burst_cores(features_from_densities(dens), params)]
    gy, gx = params.burst_grid
    pixel = (3 * gy, 3 * gx)
    assert cores[2][pixel]  # (0.9 + 0.2) / 2 >= density
    assert dens[2, 3, 3] < params.burst_density  # 2D would miss it


def test_3d_core_needs_contrast_agent_in_the_slice() -> None:
    """The 3D core does not paint a slice that holds no contrast agent (the guard)."""
    params, dens = dense_cluster_below_slice_2()  # slice 2 has no bright pixel at all
    feats = features_from_densities(dens)
    empty = np.zeros(SHAPE, bool)
    feats[2] = SliceFeatures(masks(empty), dens[2].astype(np.float32), feats[2].boundary_density)
    cores = [item.core for item in iter_burst_cores(feats, params)]
    assert cores[1].any()
    assert not cores[2].any()


def test_empty_stream_yields_nothing() -> None:
    """No slice in, no core out."""
    assert list(iter_burst_cores([], small_params())) == []


def test_coarse_densities_are_means_of_bright_pixels() -> None:
    """No bright pixel gives zero density; all bright gives one, in the box and the Gaussian."""
    params = small_params()
    bright = np.zeros(SHAPE, bool)
    for density in (coarse_density, boundary_density):
        assert not density(grid_blocks(bright, params), params).any()
        # The Gaussian's tails reach the (zero) border a little: 3 sigmas away.
        full = density(grid_blocks(~bright, params), params)[10, 10]
        np.testing.assert_allclose(full, 1.0, rtol=0.01)


def test_finalize_turns_a_core_into_a_burst() -> None:
    """A dense core over bright pixels becomes a burst region."""
    params = small_params()
    cluster = np.s_[5:30, 5:30]
    bright = np.zeros(SHAPE, bool)
    bright[cluster] = True
    slice_masks = SliceMasks(foreground=bright, bright=bright, vessel=bright, tubularity=None)
    core = np.zeros(SHAPE, bool)
    core[cluster] = True
    boundary = boundary_density(grid_blocks(bright, params), params)
    assert (finalize_slice(slice_masks, core, boundary, params)[cluster] == Label.BURST).all()
