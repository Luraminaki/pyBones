"""Taubin (λ|μ) mesh smoothing, vectorized with ``np.bincount``.

Marching cubes on voxel data produces "staircase" surfaces that overestimate
the surface area (+8.5 % on a voxelised sphere). Plain Laplacian smoothing
removes them but shrinks the mesh, which would erase thin vessels; Taubin's
alternating positive (λ) / negative (μ) steps act as a low-pass filter that
removes the staircase while preserving volume (sphere: area error +0.8 %
after 10 iterations, a 2-voxel-thin tube keeps ~92 % of its volume).

Reference: G. Taubin, "A signal processing approach to fair surface design",
SIGGRAPH 1995.
"""

from typing import TYPE_CHECKING

import numpy as np

from pybones.meshing.mesh import Mesh
from pybones.segmentation.stages import Stage, no_progress

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.config.schema import TaubinConfig
    from pybones.segmentation.stages import ProgressCallback

_COORDS = 3
#: A λ (shrinking) pass then a μ (inflating) pass per iteration.
_PASSES_PER_ITERATION = 2


def taubin_smooth(
    mesh: Mesh,
    taubin: "TaubinConfig",
    chunk_faces: int,
    progress: "ProgressCallback" = no_progress,
) -> Mesh:
    """Smooth a mesh with Taubin's λ|μ algorithm (topology unchanged).

    The umbrella operator (mean of the neighbours) is accumulated face chunk by
    face chunk: each face contributes both directions of its three edges, i.e.
    every interior edge is counted once per adjacent face, the same multiplicity
    for every neighbour of a closed mesh. Temporary memory is bounded by
    ``chunk_faces`` instead of growing with the mesh.

    Args:
        mesh: Input mesh.
        taubin: Number of iterations and the λ (> 0) / μ (< -λ) factors.
        chunk_faces: Faces processed at a time.
        progress: Progress callback, reported as :attr:`Stage.SMOOTHING` after each λ or μ
            pass (``2 x iterations`` in all).

    Returns:
        A new mesh with moved vertices and the same faces (``mesh`` itself if
        ``taubin.iterations`` is 0).
    """
    if taubin.iterations == 0:
        return mesh
    n_vertices = len(mesh.vertices)
    degree = np.zeros(n_vertices)
    for start in range(0, mesh.n_faces, chunk_faces):
        src, _ = _chunk_edges(mesh.faces[start : start + chunk_faces])
        degree += np.bincount(src, minlength=n_vertices)
    inv_degree = (1.0 / np.maximum(degree, 1))[:, None]
    vertices = mesh.vertices.astype(np.float64)
    neighbour_sum = np.empty_like(vertices)
    passes = _PASSES_PER_ITERATION * taubin.iterations
    done = 0
    progress(Stage.SMOOTHING, done, passes)
    for _ in range(taubin.iterations):
        for factor in (taubin.lambda_factor, taubin.mu_factor):
            neighbour_sum.fill(0.0)
            for start in range(0, mesh.n_faces, chunk_faces):
                src, dst = _chunk_edges(mesh.faces[start : start + chunk_faces])
                for c in range(_COORDS):
                    neighbour_sum[:, c] += np.bincount(
                        src, weights=vertices[dst, c], minlength=n_vertices
                    )
            neighbour_sum *= inv_degree
            neighbour_sum -= vertices
            vertices += factor * neighbour_sum
            done += 1
            progress(Stage.SMOOTHING, done, passes)
    return Mesh(vertices=vertices.astype(np.float32), faces=mesh.faces)


def _chunk_edges(faces: "NDArray[np.int32]") -> "tuple[NDArray[np.intp], NDArray[np.intp]]":
    """List both directions of every edge of ``faces`` as vertex pairs.

    Args:
        faces: ``(F, 3)`` vertex indices.

    Returns:
        ``(src, dst)`` vertex indices, ``6 F`` pairs.
    """
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.intp)
    return np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]])
