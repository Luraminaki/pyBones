"""Tests of the µm -> pixel conversions (:mod:`pybones.segmentation.scales`)."""

from pybones.config.loader import load_config
from pybones.segmentation.scales import resolve_pixel_params


def test_default_physical_values_give_the_reference_pixel_values() -> None:
    """At 3 µm the µm configuration reproduces the original pixel parameters exactly."""
    p = resolve_pixel_params(load_config())
    assert p.smoothing_sigma == (1.0, 1.0)
    assert dict(p.vessel_core_ball)[0] == (3, 3)  # 9 µm rim in the slice itself
    assert p.edge_half_z == 4  # surround ball: 12 µm = 4 slices
    assert (p.vessel_min_area, p.bone_min_area, p.small_blob_max_area) == (6, 30, 700)
    assert (p.burst_grid, p.burst_window_blocks, p.burst_window_z) == ((4, 4), (10, 10), 41)
    assert p.burst_min_area == 2000
    assert p.burst_boundary_sigma == (12.0, 12.0)
    assert p.z_window == 1
    assert (p.min_isolated_vessel_voxels, p.min_vessel_voxels, p.min_bone_voxels) == (30, 30, 1000)


def test_sizes_follow_the_voxel_size() -> None:
    """Every size follows an anisotropic voxel (ellipses in-plane, slices along z)."""
    p = resolve_pixel_params(load_config(overrides=["acquisition.voxel_size_um=[6, 1.5, 3]"]))
    assert dict(p.vessel_core_ball)[0] == (6, 3)  # 9 µm on 1.5 µm rows, 3 µm columns
    assert [dz for dz, _ in p.vessel_core_ball] == [-1, 0, 1]  # 6 µm slices
    assert p.smoothing_sigma == (2.0, 1.0)
    assert p.burst_grid == (8, 4)  # 12 µm blocks on 1.5 / 3 µm pixels
    assert p.burst_window_blocks == (10, 10)  # 123 µm / 12 µm in both directions
    assert p.burst_window_z == 21  # 123 µm / 6 µm slices
    assert p.bone_min_area == 60  # 270 µm² / 4.5 µm²
    assert p.min_bone_voxels == 1000  # 27000 µm³ / 27 µm³


def test_windows_are_odd_and_at_least_one() -> None:
    """Windows along z always have a centre slice."""
    p = resolve_pixel_params(load_config(overrides=["segmentation.burst.window_z_um=6"]))
    assert p.burst_window_z == 3  # 2 slices rounded up to the next odd count
