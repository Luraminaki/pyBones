"""Tests of surface extraction and STL files (:mod:`pybones.meshing`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.config.loader import load_config
from pybones.meshing.marching import extract_surface, occupancy
from pybones.meshing.stl import (
    STL_COUNT_DTYPE,
    STL_HEADER_BYTES,
    STL_RECORD,
    read_stl,
    write_stl,
)
from pybones.segmentation.labels import FOREGROUND_LABELS, Label

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.meshing.mesh import Mesh

RADIUS = 14
BALL_SHAPE = (40, 48, 56)
BALL_CENTRE = (20, 24, 28)
CHUNK = 1024
UNIT_VOXEL = (1.0, 1.0, 1.0)


def surface(
    volume: "NDArray[np.uint8]", label: Label, factor: int, voxel: tuple[float, float, float]
) -> "Mesh":
    """Mesh one label with the default settings and a given downsampling factor.

    Args:
        volume: Label volume.
        label: Label to mesh.
        factor: Downsampling factor of every label.
        voxel: Voxel size ``(z, y, x)``.

    Returns:
        The smoothed mesh.
    """
    downsample = {lab.key: factor for lab in FOREGROUND_LABELS}
    mesh_cfg = load_config(overrides=[f"mesh.downsample={downsample}".replace("'", '"')]).mesh
    return extract_surface(volume, label, mesh_cfg, voxel)


@pytest.fixture(scope="module")
def ball() -> "NDArray[np.uint8]":
    """Draw a vessel-labelled ball.

    Returns:
        The label volume.
    """
    zz, yy, xx = np.ogrid[: BALL_SHAPE[0], : BALL_SHAPE[1], : BALL_SHAPE[2]]
    cz, cy, cx = BALL_CENTRE
    vol = np.zeros(BALL_SHAPE, np.uint8)
    vol[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= RADIUS**2] = Label.VESSEL
    return vol


def test_occupancy_is_exact_block_mean() -> None:
    """Each block holds the exact fraction of its voxels with the label, inside a padding."""
    vol = np.zeros((4, 4, 4), np.uint8)
    vol[0, 0, 0] = vol[1, 1, 1] = Label.BURST
    occ = occupancy(vol, Label.BURST, factor=2)
    assert occ.shape == (4, 4, 4)  # 2x2x2 + one-voxel padding
    assert occ[1, 1, 1] == pytest.approx(2 / 8)
    assert occ.sum() == pytest.approx(2 / 8)


@pytest.mark.parametrize("factor", [1, 2])
def test_ball_mesh_is_closed_outward_and_has_right_volume(
    ball: "NDArray[np.uint8]", factor: int
) -> None:
    """The mesh is a closed manifold, faces outward, in µm and (x, y, z) order."""
    voxel = (2.0, 1.0, 1.0)  # z voxels are twice as long
    mesh = surface(ball, Label.VESSEL, factor, voxel)
    expected = 4 / 3 * np.pi * RADIUS**3 * voxel[0]
    assert mesh.volume(CHUNK) == pytest.approx(expected, rel=0.06)
    # Closed manifold: every undirected edge is shared by exactly two triangles.
    edges = np.sort(mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    cz, cy, cx = BALL_CENTRE
    np.testing.assert_allclose(
        mesh.vertices.mean(axis=0), [cx, cy, cz * voxel[0]], atol=0.6 * factor
    )


def test_stl_roundtrip_in_chunks(ball: "NDArray[np.uint8]", tmp_path: "Path") -> None:
    """A mesh written in several chunks reads back identically."""
    mesh = surface(ball, Label.VESSEL, 2, UNIT_VOXEL)
    assert mesh.n_faces > CHUNK  # exercises several chunks
    path = write_stl(mesh, tmp_path / "ball.stl", "test", CHUNK)
    expected_size = STL_HEADER_BYTES + STL_COUNT_DTYPE.itemsize + STL_RECORD.itemsize * mesh.n_faces
    assert path.stat().st_size == expected_size
    np.testing.assert_array_equal(read_stl(path), mesh.triangles(0, mesh.n_faces))


def test_absent_label_raises(ball: "NDArray[np.uint8]") -> None:
    """Meshing a label missing from the volume is an error."""
    with pytest.raises(ValueError, match="absent"):
        _ = surface(ball, Label.BONE, 2, UNIT_VOXEL)
