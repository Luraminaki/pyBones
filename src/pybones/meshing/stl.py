"""Binary STL writing, chunked and vectorized."""

from typing import TYPE_CHECKING

import numpy as np

from pybones.meshing.mesh import face_normals

if TYPE_CHECKING:
    from pathlib import Path

    from pybones.meshing.mesh import Mesh

#: Binary STL layout (fixed by the format specification).
STL_HEADER_BYTES = 80
STL_COUNT_DTYPE = np.dtype("<u4")
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
_HEADER_ENCODING = "ascii"
_HEADER_PAD = b"\0"


def write_stl(mesh: "Mesh", path: "Path", header: str, chunk_faces: int) -> "Path":
    """Write a binary STL file in chunks (bounded memory, no per-triangle loop).

    Args:
        mesh: Mesh to write.
        path: Destination file.
        header: Text stored in the 80-byte header (truncated, non-ASCII replaced).
        chunk_faces: Triangles converted per chunk.

    Returns:
        The written path.
    """
    records = np.zeros(min(mesh.n_faces, chunk_faces), dtype=STL_RECORD)
    with path.open("wb") as fh:
        _ = fh.write(
            header.encode(_HEADER_ENCODING, "replace")[:STL_HEADER_BYTES].ljust(
                STL_HEADER_BYTES, _HEADER_PAD
            )
        )
        _ = fh.write(STL_COUNT_DTYPE.type(mesh.n_faces).tobytes())
        for start in range(0, mesh.n_faces, chunk_faces):
            tri = mesh.triangles(start, start + chunk_faces)
            chunk = records[: len(tri)]
            chunk["vertices"] = tri
            chunk["normal"] = face_normals(tri)
            chunk.tofile(fh)
    return path
