"""Triangle mesh container and chunked geometric measures."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_TETRA_VOLUME_FACTOR = 6.0


@dataclass(frozen=True, slots=True)
class Mesh:
    """Triangle mesh in physical ``(x, y, z)`` coordinates.

    Attributes:
        vertices: ``(V, 3)`` float32 vertex positions.
        faces: ``(F, 3)`` int32 vertex indices, counter-clockwise seen from outside.
    """

    vertices: "NDArray[np.float32]"
    faces: "NDArray[np.int32]"

    @property
    def n_faces(self) -> int:
        """Number of triangles."""
        return int(self.faces.shape[0])

    def triangles(self, start: int, stop: int) -> "NDArray[np.float32]":
        """Gather the corner coordinates of the triangles ``start:stop``.

        Args:
            start: First face index.
            stop: Face index after the last one (clipped to the face count).

        Returns:
            ``(F, 3, 3)`` corner coordinates.
        """
        return self.vertices[self.faces[start:stop]]

    def volume(self, chunk_faces: int) -> float:
        """Compute the enclosed volume (divergence theorem).

        Args:
            chunk_faces: Triangles processed at a time (bounds temporary memory).

        Returns:
            Volume in cubed coordinate units; positive for outward-facing triangles.
        """
        total = 0.0
        for start in range(0, self.n_faces, chunk_faces):
            tri = self.triangles(start, start + chunk_faces).astype(np.float64)
            total += float(np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])))
        return total / _TETRA_VOLUME_FACTOR

    def flipped(self) -> "Mesh":
        """Reverse every triangle's winding (and so its normal).

        Returns:
            A new mesh sharing the vertices.
        """
        return Mesh(vertices=self.vertices, faces=np.ascontiguousarray(self.faces[:, ::-1]))


def face_normals(triangles: "NDArray[np.float32]") -> "NDArray[np.float32]":
    """Compute the unit normal of each triangle (zero for degenerate ones).

    Args:
        triangles: ``(F, 3, 3)`` corner coordinates.

    Returns:
        ``(F, 3)`` unit normals following the right-hand rule.
    """
    # numpy's stubs type the cross product of float32 arrays as float16; the array is
    # float32 already, so astype(copy=False) only fixes the static type.
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    n = np.cross(edges_a, edges_b).astype(np.float32, copy=False)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    _ = np.divide(n, norm, out=n, where=norm > 0)
    return n
