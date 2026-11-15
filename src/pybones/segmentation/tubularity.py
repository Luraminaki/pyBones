"""Multi-scale 3D tubularity (tube vs plate), streamed slice by slice.

For each scale, the six Hessian components of slice ``z`` are obtained by first
combining the raw slices ``z - h .. z + h`` along z with the Gaussian, first-
and second-derivative kernels (three weighted sums), then filtering those three
images in-plane with separable kernels. Only a ring buffer of ``2 h + 1`` raw
slices is kept (h = 3 sigma of the largest scale), never six float volumes.

Hessians are expressed in physical units and scale-normalised (``sigma² H``),
so responses are comparable across scales and anisotropic voxels. Only the
requested voxels are analysed, in fixed-size chunks, so temporary memory stays
small whatever the slice size.

The pipeline uses the contrast-free tube-vs-plate ratio ``|l2| / |l3|``: Frangi's
full vesselness cannot tell trabecular rods from vessels on this data (both are
bright tubes), while the ratio tells plates from tubes.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.imaging.hessian import (
    gaussian_derivative_kernels,
    kernel_half_width,
    shape_measures,
    symmetric_eigenvalues,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.segmentation.geometry import VoxelGeometry


#: Tubularity (``[0, 1]``) is stored as uint8 in the scratch volume read by the refinement.
TUBULARITY_LEVELS = 255
#: Voxels analysed per eigenvalue batch (bounds the float64 temporaries to a few MB).
CHUNK_VOXELS = 1 << 18


def quantize_tubularity(tubularity: "NDArray[np.float32]") -> "NDArray[np.uint8]":
    """Quantize a ``[0, 1]`` tubularity map for the uint8 scratch volume.

    Args:
        tubularity: Tube-vs-plate ratio map.

    Returns:
        ``round(clip(t, 0, 1) * TUBULARITY_LEVELS)`` as uint8.
    """
    return np.rint(np.clip(tubularity, 0.0, 1.0) * TUBULARITY_LEVELS).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class ScaleKernels:
    """Derivative kernels of one scale along each axis (pixel units).

    Attributes:
        sigma_um: Physical scale.
        half_z: Half width along z (slices).
        z: ``(g0, g1, g2)`` kernels along z.
        y: ``(g0, g1, g2)`` kernels along y.
        x: ``(g0, g1, g2)`` kernels along x.
        norm: Factors turning pixel-unit second derivatives into scale-normalised
            physical ones, for ``(zz, yy, xx, yz, xz, xy)``.
    """

    sigma_um: float
    half_z: int
    z: "tuple[NDArray[np.float32], ...]"
    y: "tuple[NDArray[np.float32], ...]"
    x: "tuple[NDArray[np.float32], ...]"
    norm: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class TubularityParams:
    """Tubularity settings resolved for a scan.

    Attributes:
        scales: Kernels of each scale.
    """

    scales: tuple[ScaleKernels, ...]

    @property
    def half_z(self) -> int:
        """Slices needed on each side of the current one."""
        return max(s.half_z for s in self.scales)


def scale_kernels(sigma_um: float, geo: "VoxelGeometry") -> ScaleKernels:
    """Build the per-axis derivative kernels of one physical scale.

    Args:
        sigma_um: Gaussian scale in µm.
        geo: Voxel size of the scan.

    Returns:
        The kernels and the factors that scale-normalise each Hessian component.
    """
    sigmas_px = [sigma_um / spacing for spacing in (geo.z, geo.y, geo.x)]
    halves = [kernel_half_width(s) for s in sigmas_px]
    kernels = [gaussian_derivative_kernels(s, h) for s, h in zip(sigmas_px, halves, strict=True)]
    s2 = sigma_um**2
    norm = (
        s2 / geo.z**2,
        s2 / geo.y**2,
        s2 / geo.x**2,
        s2 / (geo.y * geo.z),
        s2 / (geo.x * geo.z),
        s2 / (geo.x * geo.y),
    )
    return ScaleKernels(sigma_um, halves[0], kernels[0], kernels[1], kernels[2], norm)


def _masked_hessian(
    window: "list[NDArray[np.float32]]", sk: ScaleKernels, offset: int, mask: "NDArray[np.bool_]"
) -> "tuple[NDArray[np.float32], ...]":
    """Hessian ``(zz, yy, xx, yz, xz, xy)`` of the centre slice of ``window``, at ``mask``.

    Each full-slice component is masked as soon as it is computed, so only one of
    them exists at a time.

    Args:
        window: Raw slices as float32; ``window[offset + j]`` is slice ``z - half_z + j``
            of this scale (mirrored at the stack ends).
        sk: Kernels of the scale.
        offset: Index in ``window`` of this scale's first slice (the window is sized
            for the largest scale).
        mask: Voxels of the centre slice to keep.

    Returns:
        The six scale-normalised components, each a 1D array over ``mask``.
    """
    g0z, g1z, g2z = sk.z
    g0y, g1y, g2y = sk.y
    g0x, g1x, g2x = sk.x
    shape = window[0].shape
    z0, z1, z2 = (np.zeros(shape, np.float32) for _ in range(3))
    for j in range(2 * sk.half_z + 1):
        sl = window[offset + j]
        _ = cv2.scaleAdd(sl, float(g0z[j]), z0, dst=z0)
        _ = cv2.scaleAdd(sl, float(g1z[j]), z1, dst=z1)
        _ = cv2.scaleAdd(sl, float(g2z[j]), z2, dst=z2)
    # cv2.sepFilter2D and the z sums are correlations: each first derivative changes sign,
    # which cancels in the mixed terms (two first derivatives each).
    specs = (
        (z2, g0x, g0y),  # zz
        (z0, g0x, g2y),  # yy
        (z0, g2x, g0y),  # xx
        (z1, g0x, g1y),  # yz
        (z1, g1x, g0y),  # xz
        (z0, g1x, g1y),  # xy
    )
    return tuple(
        cv2.sepFilter2D(src, cv2.CV_32F, kx, ky, borderType=cv2.BORDER_REFLECT)[mask] * norm
        for (src, kx, ky), norm in zip(specs, sk.norm, strict=True)
    )


def slice_tubularity(
    window: "list[NDArray[np.float32]]", mask: "NDArray[np.bool_]", params: TubularityParams
) -> "NDArray[np.float32]":
    """Tube-vs-plate ratio of the centre slice of a ``2 * half_z + 1`` window, at ``mask``.

    Args:
        window: Raw slices ``z - half_z .. z + half_z`` as float32 (mirrored at the ends).
        mask: Voxels of the centre slice to analyse.
        params: Tubularity settings.

    Returns:
        ``|l2| / |l3|`` (~1 tube, ~0 plate) at the scale where the structure is
        strongest (largest ``S² = sum l²``); 0 outside ``mask``.
    """
    tubularity = np.zeros(mask.shape, np.float32)
    n = int(mask.sum())
    if n == 0:
        return tubularity
    best_s2, tube = np.full(n, -1.0), np.zeros(n)
    for sk in params.scales:
        h = _masked_hessian(window, sk, params.half_z - sk.half_z, mask)
        for start in range(0, n, CHUNK_VOXELS):
            part = slice(start, start + CHUNK_VOXELS)
            hzz, hyy, hxx, hyz, hxz, hxy = (c[part] for c in h)
            shape = shape_measures(symmetric_eigenvalues(hxx, hyy, hzz, hxy, hxz, hyz))
            stronger = shape.structure2 > best_s2[part]
            tube[part][stronger] = shape.tube_ratio[stronger]
            best_s2[part][stronger] = shape.structure2[stronger]
    tubularity[mask] = tube
    return tubularity
