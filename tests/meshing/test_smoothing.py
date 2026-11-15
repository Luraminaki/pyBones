"""Tests of the Taubin smoothing (:mod:`pybones.meshing.smoothing`)."""

import numpy as np
import pytest
from skimage.measure import marching_cubes, mesh_surface_area

from pybones.config.loader import load_config
from pybones.meshing.mesh import Mesh
from pybones.meshing.smoothing import taubin_smooth

RADIUS = 20
GRID = 56
CHUNK = 1 << 16
ISO_LEVEL = 0.5


@pytest.fixture(scope="module")
def voxel_sphere() -> Mesh:
    """Mesh a voxelised sphere without smoothing (staircase surface).

    Returns:
        The raw marching-cubes mesh.
    """
    c = GRID // 2
    zz, yy, xx = np.ogrid[:GRID, :GRID, :GRID]
    ball = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= RADIUS**2).astype(np.float32)
    verts, faces, _, _ = marching_cubes(np.pad(ball, 1), ISO_LEVEL)
    return Mesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int32))


def area_error(mesh: Mesh) -> float:
    """Measure the relative surface-area error against the ideal sphere.

    Args:
        mesh: Mesh of the sphere.

    Returns:
        ``area / (4 pi r²) - 1``.
    """
    return float(mesh_surface_area(mesh.vertices, mesh.faces)) / (4 * np.pi * RADIUS**2) - 1


def test_taubin_removes_staircase_and_keeps_volume(voxel_sphere: Mesh) -> None:
    """Smoothing removes the staircase area excess without shrinking the volume."""
    smoothed = taubin_smooth(voxel_sphere, load_config().mesh.taubin, CHUNK)
    assert area_error(voxel_sphere) > 0.05  # staircase inflates the area
    assert abs(area_error(smoothed)) < 0.015
    assert smoothed.volume(CHUNK) == pytest.approx(voxel_sphere.volume(CHUNK), rel=0.01)
    assert smoothed.faces is voxel_sphere.faces


def test_zero_iterations_is_identity(voxel_sphere: Mesh) -> None:
    """No iteration returns the mesh itself."""
    taubin = load_config(overrides=["mesh.taubin.iterations=0"]).mesh.taubin
    assert taubin_smooth(voxel_sphere, taubin, CHUNK) is voxel_sphere


def test_chunking_does_not_change_the_result(voxel_sphere: Mesh) -> None:
    """Processing faces in chunks gives the same vertices."""
    taubin = load_config().mesh.taubin
    whole = taubin_smooth(voxel_sphere, taubin, voxel_sphere.n_faces)
    chunked = taubin_smooth(voxel_sphere, taubin, 1000)
    np.testing.assert_allclose(chunked.vertices, whole.vertices, atol=1e-4)
