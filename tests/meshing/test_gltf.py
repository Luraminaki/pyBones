"""Tests of the binary glTF writer and reader (:mod:`pybones.meshing.gltf`) and mesh formats."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.config.loader import load_config
from pybones.meshing.formats import MeshFormat
from pybones.meshing.gltf import (
    ALIGNMENT,
    CHUNK_BIN,
    CHUNK_JSON,
    FLOAT_POSITION_BITS,
    GLB_MAGIC,
    GLB_VERSION,
    QUANTIZATION_EXTENSION,
    QUANTIZED_POSITION_BITS,
    read_glb,
    vertex_normals,
    write_glb,
)
from pybones.meshing.marching import extract_surface
from pybones.segmentation.labels import Label
from pybones.utils.constants import UINT16_MAX

if TYPE_CHECKING:
    from typing import Any

    from pybones.meshing.mesh import Mesh

#: A ball of vessel voxels, meshed at full resolution with anisotropic voxels.
RADIUS = 10
SHAPE = (28, 30, 32)
VOXEL_UM = (3.0, 1.5, 1.5)
CHUNK = 1024
NAME = "vessel"
#: 8-bit normals: one step of a component is 1/127.
NORMAL_TOLERANCE = 1.5 / 127


@pytest.fixture(scope="module")
def ball_mesh() -> "Mesh":
    """Mesh a ball of vessel voxels.

    Returns:
        The smoothed mesh, in µm.
    """
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    centre = [s // 2 for s in SHAPE]
    volume = np.zeros(SHAPE, np.uint8)
    inside = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= RADIUS**2
    volume[inside] = Label.VESSEL
    config = load_config(overrides=['mesh.downsample={"bone": 1, "vessel": 1, "burst": 1}'])
    return extract_surface(volume, Label.VESSEL, config.mesh, VOXEL_UM)


def glb_parts(path: Path) -> "tuple[bytes, dict[str, Any], bytes]":
    """Split a ``.glb`` file into its header, JSON document and binary chunk.

    Args:
        path: ``.glb`` file.

    Returns:
        The 12-byte header, the parsed JSON and the binary chunk.
    """
    data = path.read_bytes()
    json_length = int.from_bytes(data[12:16], "little")
    assert data[16:20] == CHUNK_JSON
    document = json.loads(data[20 : 20 + json_length])
    start = 20 + json_length
    bin_length = int.from_bytes(data[start : start + 4], "little")
    assert data[start + 4 : start + 8] == CHUNK_BIN
    assert start + 8 + bin_length == len(data)
    return data[:12], document, data[start + 8 :]


@pytest.mark.parametrize("bits", [QUANTIZED_POSITION_BITS, FLOAT_POSITION_BITS])
def test_glb_layout_follows_the_specification(ball_mesh: "Mesh", tmp_path: Path, bits: int) -> None:
    """Header, chunk types and lengths, 4-byte alignment and the quantization extension."""
    path = write_glb(ball_mesh, tmp_path / "ball.glb", NAME, bits, CHUNK)
    header, document, binary = glb_parts(path)
    assert header[:4] == GLB_MAGIC
    assert int.from_bytes(header[4:8], "little") == GLB_VERSION
    assert int.from_bytes(header[8:12], "little") == path.stat().st_size
    assert path.stat().st_size % ALIGNMENT == 0
    assert document["buffers"][0]["byteLength"] == len(binary)
    for view in document["bufferViews"]:
        assert view["byteOffset"] % ALIGNMENT == 0
        assert view.get("byteStride", ALIGNMENT) % ALIGNMENT == 0
    position = document["accessors"][0]
    assert {"min", "max"} <= set(position)  # required for POSITION
    quantized = QUANTIZATION_EXTENSION in document.get("extensionsRequired", [])
    assert quantized is (bits == QUANTIZED_POSITION_BITS)
    assert document["meshes"][0]["name"] == NAME


def test_float_positions_round_trip_exactly(ball_mesh: "Mesh", tmp_path: Path) -> None:
    """32-bit positions, normals and indices read back unchanged."""
    path = write_glb(ball_mesh, tmp_path / "ball.glb", NAME, FLOAT_POSITION_BITS, CHUNK)
    back = read_glb(path)
    assert back.position_bits == FLOAT_POSITION_BITS
    np.testing.assert_array_equal(back.vertices, ball_mesh.vertices)
    np.testing.assert_array_equal(back.faces, ball_mesh.faces)
    np.testing.assert_allclose(back.normals, vertex_normals(ball_mesh, CHUNK))


def test_quantized_positions_err_by_at_most_half_a_step(ball_mesh: "Mesh", tmp_path: Path) -> None:
    """16-bit positions are off by at most half of (bounding box size / 65535) per axis."""
    path = write_glb(ball_mesh, tmp_path / "ball.glb", NAME, QUANTIZED_POSITION_BITS, CHUNK)
    back = read_glb(path)
    vertices = ball_mesh.vertices.astype(np.float64)
    half_step = (vertices.max(axis=0) - vertices.min(axis=0)) / UINT16_MAX / 2
    error = np.abs(back.vertices - vertices).max(axis=0)
    assert (error <= half_step * (1 + 1e-6)).all()
    np.testing.assert_array_equal(back.faces, ball_mesh.faces)
    np.testing.assert_allclose(
        back.normals, vertex_normals(ball_mesh, CHUNK), atol=NORMAL_TOLERANCE
    )
    floats = write_glb(ball_mesh, tmp_path / "f.glb", NAME, FLOAT_POSITION_BITS, CHUNK)
    assert path.stat().st_size < floats.stat().st_size


def test_vertex_normals_point_outwards_and_ignore_chunking(ball_mesh: "Mesh") -> None:
    """On a ball, every vertex normal is a unit vector pointing away from the centre."""
    normals = vertex_normals(ball_mesh, CHUNK)
    np.testing.assert_allclose(vertex_normals(ball_mesh, ball_mesh.n_faces), normals, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    outward = ball_mesh.vertices - ball_mesh.vertices.mean(axis=0)
    assert (np.einsum("ij,ij->i", normals, outward) > 0).all()


def test_position_bits_are_16_or_32(ball_mesh: "Mesh", tmp_path: Path) -> None:
    """Any other precision is refused."""
    with pytest.raises(ValueError, match="16 or 32"):
        _ = write_glb(ball_mesh, tmp_path / "ball.glb", NAME, 8, CHUNK)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.stl", MeshFormat.STL), ("a.GLB", MeshFormat.GLB), ("dir.v2/a.glb", MeshFormat.GLB)],
)
def test_mesh_format_follows_the_extension(name: str, expected: MeshFormat) -> None:
    """The format is the file extension, whatever its case."""
    assert MeshFormat.from_path(Path(name)) is expected
    assert expected.suffix == f".{expected.value}"


@pytest.mark.parametrize("name", ["a.obj", "a", "a.stl.partial"])
def test_unknown_extensions_are_refused(name: str) -> None:
    """Only the supported extensions name a format."""
    with pytest.raises(ValueError, match="unsupported mesh format"):
        _ = MeshFormat.from_path(Path(name))
