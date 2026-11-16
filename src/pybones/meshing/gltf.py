"""Binary glTF 2.0 (``.glb``) writing and reading of a single indexed mesh.

Layout of a written file: a 12-byte header, a JSON chunk (the scene: one node, one
mesh, its accessors) and a binary chunk holding, in order, the positions, the vertex
normals and the triangle indices. Coordinates are in micrometres (``asset.extras``).

With 16-bit positions, the vertices are stored as unsigned integers over the mesh's
bounding box (``KHR_mesh_quantization``) and the node's ``scale`` / ``translation``
turn them back into micrometres: the rounding error is at most half a step, where a
step is the box size divided by 65,535 on each axis. Normals are then signed 8-bit.
With 32-bit positions, positions and normals are floats.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pybones import __version__
from pybones.utils.constants import UINT16_MAX

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.meshing.mesh import Mesh

#: Container layout (glTF 2.0 specification, "Binary glTF Layout").
GLB_MAGIC = b"glTF"
GLB_VERSION = 2
CHUNK_JSON = b"JSON"
CHUNK_BIN = b"BIN\0"
_U32 = np.dtype("<u4")
_HEADER_WORDS = 3
_CHUNK_HEADER_WORDS = 2
#: Chunks and vertex elements are aligned to 4 bytes; padding of each chunk type.
ALIGNMENT = 4
_JSON_PAD = b" "
_BIN_PAD = b"\0"
#: Accessor component types and primitive mode (glTF enumerations).
BYTE, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT = 5120, 5123, 5125, 5126
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
TRIANGLES = 4
QUANTIZATION_EXTENSION = "KHR_mesh_quantization"
#: Position precisions: quantized (unsigned 16-bit) or float.
QUANTIZED_POSITION_BITS = 16
FLOAT_POSITION_BITS = 32
_POSITION_LEVELS = UINT16_MAX
_NORMAL_LEVELS = np.iinfo(np.int8).max
_XYZ = 3
#: Bytes per vertex element: quantized positions are padded from 6 to 8 bytes and
#: normals from 3 to 4 (each element must start on a 4-byte boundary).
_QUANTIZED_POSITION_STRIDE = 8
_QUANTIZED_NORMAL_STRIDE = 4
_JSON_ENCODING = "utf-8"
UNIT = "micrometre"


@dataclass(frozen=True, slots=True)
class GlbMesh:
    """A mesh read back from a ``.glb`` file, in micrometres.

    Attributes:
        vertices: ``(V, 3)`` positions (de-quantized if stored as integers).
        normals: ``(V, 3)`` unit vertex normals.
        faces: ``(F, 3)`` vertex indices.
        position_bits: Precision the positions were stored with.
    """

    vertices: "NDArray[np.float64]"
    normals: "NDArray[np.float32]"
    faces: "NDArray[np.int64]"
    position_bits: int


def vertex_normals(mesh: "Mesh", chunk_faces: int) -> "NDArray[np.float32]":
    """Compute smooth vertex normals: the area-weighted mean of the adjacent face normals.

    Args:
        mesh: Mesh (counter-clockwise triangles seen from outside).
        chunk_faces: Triangles processed at a time (bounds temporary memory).

    Returns:
        ``(V, 3)`` unit normals (zero for a vertex of degenerate triangles only).
    """
    n_vertices = len(mesh.vertices)
    sums = np.zeros((n_vertices, _XYZ), dtype=np.float64)
    for start in range(0, mesh.n_faces, chunk_faces):
        tri = mesh.triangles(start, start + chunk_faces).astype(np.float64)
        # The cross product's length is twice the triangle area: area weighting for free.
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        corners = mesh.faces[start : start + chunk_faces]
        for axis in range(_XYZ):
            weights = np.repeat(cross[:, axis], _XYZ)
            sums[:, axis] += np.bincount(corners.ravel(), weights, minlength=n_vertices)
    norm = np.linalg.norm(sums, axis=1, keepdims=True)
    _ = np.divide(sums, norm, out=sums, where=norm > 0)
    return sums.astype(np.float32)


def _padding(size: int) -> int:
    """Count the bytes that bring ``size`` to the next 4-byte boundary.

    Args:
        size: Byte count.

    Returns:
        Padding length in ``[0, 4)``.
    """
    return -size % ALIGNMENT


def _pad_rows(values: "NDArray[Any]", stride: int) -> "NDArray[np.uint8]":
    """Lay out ``(N, 3)`` values with ``stride`` bytes per row (trailing bytes zero).

    Args:
        values: Small-integer rows.
        stride: Bytes per row (at least the row size).

    Returns:
        ``(N, stride)`` bytes.
    """
    row_bytes = values.dtype.itemsize * values.shape[1]
    out = np.zeros((len(values), stride), dtype=np.uint8)
    out[:, :row_bytes] = np.ascontiguousarray(values).view(np.uint8).reshape(len(values), row_bytes)
    return out


@dataclass(frozen=True, slots=True)
class _Encoded:
    """The binary arrays of a mesh and the JSON fields that describe them.

    Attributes:
        positions: Position bytes.
        normals: Normal bytes.
        indices: Index bytes.
        position_accessor: Position accessor fields (without the buffer view).
        normal_accessor: Normal accessor fields (without the buffer view).
        index_accessor: Index accessor fields (without the buffer view).
        position_stride: ``byteStride`` of the positions (``None``: tightly packed).
        normal_stride: ``byteStride`` of the normals (``None``: tightly packed).
        node: Node fields (de-quantization transform, if any).
        extensions: glTF extensions the file requires.
    """

    positions: "NDArray[Any]"
    normals: "NDArray[Any]"
    indices: "NDArray[Any]"
    position_accessor: dict[str, Any]
    normal_accessor: dict[str, Any]
    index_accessor: dict[str, Any]
    position_stride: int | None
    normal_stride: int | None
    node: dict[str, Any]
    extensions: list[str]


def _encode(mesh: "Mesh", position_bits: int, chunk_faces: int) -> _Encoded:
    """Convert a mesh into the arrays and descriptions of a ``.glb`` file.

    Args:
        mesh: Mesh to encode.
        position_bits: 16 (quantized positions, 8-bit normals) or 32 (floats).
        chunk_faces: Triangles processed at a time for the normals.

    Returns:
        The encoded mesh.

    Raises:
        ValueError: If ``position_bits`` is neither 16 nor 32.
    """
    if position_bits not in (QUANTIZED_POSITION_BITS, FLOAT_POSITION_BITS):
        raise ValueError(f"position_bits must be 16 or 32, not {position_bits}")
    vertices = mesh.vertices.astype(np.float64)
    normals = vertex_normals(mesh, chunk_faces)
    count = len(vertices)
    indices = mesh.faces.astype(np.uint32).ravel()
    index_accessor: dict[str, Any] = {
        "componentType": UNSIGNED_INT,
        "count": int(indices.size),
        "type": "SCALAR",
    }
    if position_bits == FLOAT_POSITION_BITS:
        positions = vertices.astype(np.float32)
        return _Encoded(
            positions=positions,
            normals=normals,
            indices=indices,
            position_accessor={
                "componentType": FLOAT,
                "count": count,
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            normal_accessor={"componentType": FLOAT, "count": count, "type": "VEC3"},
            index_accessor=index_accessor,
            position_stride=None,
            normal_stride=None,
            node={},
            extensions=[],
        )
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = high - low
    step = np.where(extent > 0, extent / _POSITION_LEVELS, 1.0)
    quantized = np.round((vertices - low) / step).astype(np.uint16)
    packed_normals = np.round(normals * _NORMAL_LEVELS).astype(np.int8)
    return _Encoded(
        positions=_pad_rows(quantized, _QUANTIZED_POSITION_STRIDE),
        normals=_pad_rows(packed_normals, _QUANTIZED_NORMAL_STRIDE),
        indices=indices,
        position_accessor={
            "componentType": UNSIGNED_SHORT,
            "count": count,
            "type": "VEC3",
            "min": quantized.min(axis=0).tolist(),
            "max": quantized.max(axis=0).tolist(),
        },
        normal_accessor={
            "componentType": BYTE,
            "normalized": True,
            "count": count,
            "type": "VEC3",
        },
        index_accessor=index_accessor,
        position_stride=_QUANTIZED_POSITION_STRIDE,
        normal_stride=_QUANTIZED_NORMAL_STRIDE,
        node={"translation": low.tolist(), "scale": step.tolist()},
        extensions=[QUANTIZATION_EXTENSION],
    )


def _document(encoded: _Encoded, name: str) -> tuple[dict[str, Any], int]:
    """Describe the scene and lay out the binary chunk.

    Args:
        encoded: Encoded mesh.
        name: Mesh and node name.

    Returns:
        The glTF JSON document, and the length of the binary data (before padding).
    """
    views: list[dict[str, Any]] = []
    offset = 0
    for data, target, stride in (
        (encoded.positions, ARRAY_BUFFER, encoded.position_stride),
        (encoded.normals, ARRAY_BUFFER, encoded.normal_stride),
        (encoded.indices, ELEMENT_ARRAY_BUFFER, None),
    ):
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": int(data.nbytes),
            "target": target,
        }
        if stride is not None:
            view["byteStride"] = stride
        views.append(view)
        offset += int(data.nbytes) + _padding(int(data.nbytes))
    accessors = [
        {**encoded.position_accessor, "bufferView": 0},
        {**encoded.normal_accessor, "bufferView": 1},
        {**encoded.index_accessor, "bufferView": 2},
    ]
    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": f"pybones {__version__}",
            "extras": {"unit": UNIT},
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name, **encoded.node}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": TRIANGLES}
                ],
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    if encoded.extensions:
        document["extensionsUsed"] = encoded.extensions
        document["extensionsRequired"] = encoded.extensions
    return document, offset


def write_glb(
    mesh: "Mesh", path: "Path", name: str, position_bits: int, chunk_faces: int
) -> "Path":
    """Write a mesh as a binary glTF 2.0 file (one node, one indexed triangle primitive).

    Args:
        mesh: Mesh in micrometres, counter-clockwise triangles seen from outside.
        path: Destination file.
        name: Mesh and node name (e.g. the label key).
        position_bits: 16 (quantized positions, see the module notes) or 32 (floats).
        chunk_faces: Triangles processed at a time for the normals.

    Returns:
        The written path.
    """
    encoded = _encode(mesh, position_bits, chunk_faces)
    document, bin_length = _document(encoded, name)
    json_bytes = json.dumps(document, separators=(",", ":")).encode(_JSON_ENCODING)
    json_bytes += _JSON_PAD * _padding(len(json_bytes))
    chunk_header = _CHUNK_HEADER_WORDS * _U32.itemsize
    total = _HEADER_WORDS * _U32.itemsize + chunk_header + len(json_bytes) + chunk_header
    total += bin_length
    with path.open("wb") as fh:
        _ = fh.write(GLB_MAGIC)
        _ = fh.write(np.array([GLB_VERSION, total], dtype=_U32).tobytes())
        _ = fh.write(np.array([len(json_bytes)], dtype=_U32).tobytes() + CHUNK_JSON)
        _ = fh.write(json_bytes)
        _ = fh.write(np.array([bin_length], dtype=_U32).tobytes() + CHUNK_BIN)
        for data in (encoded.positions, encoded.normals, encoded.indices):
            np.ascontiguousarray(data).tofile(fh)
            _ = fh.write(_BIN_PAD * _padding(int(data.nbytes)))
    return path


def _accessor_array(document: dict[str, Any], binary: bytes, index: int) -> "NDArray[Any]":
    """Read an accessor written by :func:`write_glb` (strided views included).

    Args:
        document: glTF JSON document.
        binary: Binary chunk.
        index: Accessor index.

    Returns:
        ``(count, components)`` values, or ``(count,)`` for scalars.
    """
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = {BYTE: np.int8, UNSIGNED_SHORT: np.uint16, UNSIGNED_INT: np.uint32, FLOAT: np.float32}
    component = np.dtype(dtype[accessor["componentType"]]).newbyteorder("<")
    components = _XYZ if accessor["type"] == "VEC3" else 1
    count = accessor["count"]
    raw = np.frombuffer(binary, np.uint8, view["byteLength"], view["byteOffset"])
    stride = view.get("byteStride", component.itemsize * components)
    rows = raw.reshape(count, stride)[:, : component.itemsize * components]
    values = np.ascontiguousarray(rows).view(component).reshape(count, components)
    return values if components > 1 else values.ravel()


def read_glb(path: "Path") -> GlbMesh:
    """Read back a mesh written by :func:`write_glb`.

    Args:
        path: ``.glb`` file.

    Returns:
        The mesh, de-quantized to micrometres.

    Raises:
        ValueError: If the file is not a binary glTF 2.0 file.
    """
    data = path.read_bytes()
    header = np.frombuffer(data, _U32, _HEADER_WORDS - 1, len(GLB_MAGIC))
    if data[: len(GLB_MAGIC)] != GLB_MAGIC or int(header[0]) != GLB_VERSION:
        raise ValueError(f"{path} is not a binary glTF 2.0 file")
    offset = _HEADER_WORDS * _U32.itemsize
    chunks: dict[bytes, bytes] = {}
    while offset < len(data):
        length = int(np.frombuffer(data, _U32, 1, offset)[0])
        kind = data[offset + _U32.itemsize : offset + _CHUNK_HEADER_WORDS * _U32.itemsize]
        start = offset + _CHUNK_HEADER_WORDS * _U32.itemsize
        chunks[kind] = data[start : start + length]
        offset = start + length
    document = cast("dict[str, Any]", json.loads(chunks[CHUNK_JSON]))
    binary = chunks[CHUNK_BIN]
    positions = _accessor_array(document, binary, 0).astype(np.float64)
    normals = _accessor_array(document, binary, 1).astype(np.float32)
    node = document["nodes"][0]
    quantized = QUANTIZATION_EXTENSION in document.get("extensionsRequired", [])
    if quantized:
        positions = positions * np.array(node["scale"]) + np.array(node["translation"])
        normals /= _NORMAL_LEVELS
    faces = _accessor_array(document, binary, 2).astype(np.int64).reshape(-1, _XYZ)
    bits = QUANTIZED_POSITION_BITS if quantized else FLOAT_POSITION_BITS
    return GlbMesh(vertices=positions, normals=normals, faces=faces, position_bits=bits)
