"""Surface extraction of one label of a volume with marching cubes."""

from typing import TYPE_CHECKING

import numpy as np
from skimage.measure import marching_cubes as skimage_marching_cubes

from pybones.meshing.mesh import Mesh
from pybones.meshing.smoothing import taubin_smooth
from pybones.segmentation.stages import Stage, no_progress

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.config.schema import MeshConfig
    from pybones.segmentation.labels import Label
    from pybones.segmentation.stages import ProgressCallback

#: Empty border (voxels) so that surfaces touching the volume edge are closed.
_PAD = 1


def occupancy(
    volume: "NDArray[np.uint8]",
    label: "Label",
    factor: int,
    progress: "ProgressCallback" = no_progress,
) -> "NDArray[np.float32]":
    """Measure the fraction of ``label`` voxels in each ``factor³`` block, with an empty border.

    Streams over z, so only ``factor`` slices of ``volume`` (which can be a
    memmap) are touched at a time. The exact block mean keeps sub-voxel
    information, which makes the downsampled surface smoother than subsampling.

    Args:
        volume: ``(Z, Y, X)`` label volume.
        label: Label to extract.
        factor: Integer downsampling factor (1 keeps full resolution).
        progress: Progress callback, reported as :attr:`Stage.OCCUPANCY` after each block row.

    Returns:
        ``float32`` occupancy volume in ``[0, 1]`` padded by one empty voxel.
    """
    nz, ny, nx = (s // factor for s in volume.shape)
    out = np.zeros((nz + 2 * _PAD, ny + 2 * _PAD, nx + 2 * _PAD), dtype=np.float32)
    acc = np.empty((ny, nx), dtype=np.uint16)
    inv_block = np.float32(1.0 / factor**3)
    for k in range(nz):
        _ = acc.fill(0)
        for z in range(k * factor, (k + 1) * factor):
            hit: NDArray[np.uint8] = (volume[z, : ny * factor, : nx * factor] == label).view(
                np.uint8
            )
            # Exact in-plane block sum via a reshape (no interpolation rounding).
            acc += hit.reshape(ny, factor, nx, factor).sum(axis=(1, 3), dtype=np.uint16)
        _ = np.multiply(acc, inv_block, out=out[k + _PAD, _PAD:-_PAD, _PAD:-_PAD])
        progress(Stage.OCCUPANCY, k + 1, nz)
    return out


def extract_surface(
    volume: "NDArray[np.uint8]",
    label: "Label",
    mesh_cfg: "MeshConfig",
    voxel_size_um: tuple[float, float, float],
    progress: "ProgressCallback" = no_progress,
) -> Mesh:
    """Extract the closed surface of one label as a triangle mesh, Taubin-smoothed.

    Peak RAM is roughly ``4 x voxels / factor³`` bytes for the occupancy grid,
    then scikit-image's vertex/face arrays, then ``~100 x faces`` bytes for the
    half-edge arrays of the Taubin smoothing.

    Args:
        volume: ``(Z, Y, X)`` label volume (memmap welcome).
        label: Label to mesh.
        mesh_cfg: Mesh parameters (per-label downsampling, smoothing, ...).
        voxel_size_um: Voxel size along ``(z, y, x)`` in µm.
        progress: Progress callback (:attr:`Stage.OCCUPANCY`, :attr:`Stage.SURFACE`,
            :attr:`Stage.SMOOTHING`); a cancelled tracker stops the work at its next update.

    Returns:
        The mesh in µm, in ``(x, y, z)`` order, triangles facing outwards.

    Raises:
        ValueError: If the label is absent from the volume.
    """
    factor = mesh_cfg.factor_for(label)
    occ = occupancy(volume, label, factor, progress)
    if not occ.any():
        raise ValueError(f"Label {label.key!r} is absent from the volume")
    spacing = tuple(s * factor for s in voxel_size_um)
    progress(Stage.SURFACE, 0, 1)  # one scikit-image call: no progress inside
    verts, faces, _, _ = skimage_marching_cubes(
        occ, level=mesh_cfg.iso_level, spacing=spacing, step_size=mesh_cfg.step_size
    )
    del occ
    progress(Stage.SURFACE, 1, 1)
    # Undo the padding offset, then reorder (z, y, x) -> (x, y, z). Swapping two axes is a
    # reflection, so the winding is reversed to keep normals pointing outwards.
    verts -= np.asarray(spacing, dtype=np.float32) * _PAD
    mesh = Mesh(
        vertices=np.ascontiguousarray(verts[:, ::-1], dtype=np.float32),
        faces=np.ascontiguousarray(faces[:, ::-1], dtype=np.int32),
    )
    mesh = mesh if mesh.volume(mesh_cfg.chunk_faces) >= 0 else mesh.flipped()
    return taubin_smooth(mesh, mesh_cfg.taubin, mesh_cfg.chunk_faces, progress)
