"""Gaussian-derivative kernels, closed-form 3x3 eigenvalues and Hessian shape measures.

All operations are vectorized over voxels; no Python loop runs per voxel.

Reference: A. F. Frangi et al., "Multiscale vessel enhancement filtering",
MICCAI 1998 (eigenvalue ordering and the ``R_A`` and ``S`` measures).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Gaussian kernels are truncated at this many sigmas.
TRUNCATE_SIGMAS = 3.0
_THIRD = 1.0 / 3.0
_TWO_PI_OVER_3 = 2.0 * np.pi / 3.0
_EPS = 1e-12


def kernel_half_width(sigma_px: float) -> int:
    """Half width of a Gaussian kernel truncated at :data:`TRUNCATE_SIGMAS`.

    Args:
        sigma_px: Gaussian sigma in pixels.

    Returns:
        Half width in pixels (at least 1).
    """
    return max(1, int(np.ceil(TRUNCATE_SIGMAS * sigma_px)))


def gaussian_derivative_kernels(
    sigma_px: float, half: int
) -> "tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]":
    """Sampled Gaussian and its first and second derivatives (in pixel units).

    Args:
        sigma_px: Gaussian sigma in pixels.
        half: Half width of the kernels (length ``2 * half + 1``).

    Returns:
        ``(g0, g1, g2)``: smoothing, first- and second-derivative kernels. ``g0``
        sums to 1; ``g1`` and ``g2`` are the analytic derivatives of the same
        sampled Gaussian (``g2`` is corrected to sum to 0, so flat regions give a
        zero Hessian).
    """
    x = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / sigma_px) ** 2)
    g0: NDArray = g / g.sum()
    g1 = -x / sigma_px**2 * g0
    g2 = (x**2 / sigma_px**4 - 1.0 / sigma_px**2) * g0
    g2 -= g2.mean()
    return g0.astype(np.float32), g1.astype(np.float32), g2.astype(np.float32)


def symmetric_eigenvalues(
    xx: "NDArray[np.floating]",
    yy: "NDArray[np.floating]",
    zz: "NDArray[np.floating]",
    xy: "NDArray[np.floating]",
    xz: "NDArray[np.floating]",
    yz: "NDArray[np.floating]",
) -> "NDArray[np.float64]":
    """Eigenvalues of many symmetric 3x3 matrices, closed form (Smith, 1961).

    Args:
        xx: Entry (x, x); every argument is an array of the same shape.
        yy: Entry (y, y).
        zz: Entry (z, z).
        xy: Entry (x, y).
        xz: Entry (x, z).
        yz: Entry (y, z).

    Returns:
        ``(3, *shape)`` eigenvalues sorted in decreasing order.
    """
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (xx, yy, zz))
    d, e, f = (np.asarray(v, dtype=np.float64) for v in (xy, xz, yz))
    q = (a + b + c) * _THIRD
    a0, b0, c0 = a - q, b - q, c - q
    p2 = a0**2 + b0**2 + c0**2 + 2.0 * (d**2 + e**2 + f**2)
    p = np.sqrt(p2 / 6.0)
    det = a0 * (b0 * c0 - f**2) - d * (d * c0 - f * e) + e * (d * f - b0 * e)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(p > _EPS, det / (2.0 * p**3), 0.0)
    phi = np.arccos(np.clip(r, -1.0, 1.0)) * _THIRD
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + _TWO_PI_OVER_3)
    e2 = 3.0 * q - e1 - e3
    return np.stack([e1, e2, e3])


@dataclass(frozen=True, slots=True)
class ShapeMeasures:
    """Frangi shape descriptors, from eigenvalues sorted by magnitude ``|l1| <= |l2| <= |l3|``.

    A tube has ``l1 ~ 0`` and ``|l2| ~ |l3|``; a plate ``l1 ~ l2 ~ 0`` and ``|l3|`` large.

    Attributes:
        tube_ratio: ``R_A = |l2| / |l3|``: ~1 for tubes, ~0 for plates (contrast-free).
        structure2: ``S² = l1² + l2² + l3²`` (grows with contrast; picks the best scale).
    """

    tube_ratio: "NDArray[np.float64]"
    structure2: "NDArray[np.float64]"


def shape_measures(eigenvalues: "NDArray[np.floating]") -> ShapeMeasures:
    """Compute the shape descriptors of Hessian eigenvalues.

    Args:
        eigenvalues: ``(3, ...)`` Hessian eigenvalues (any order).

    Returns:
        The descriptors.
    """
    order = np.argsort(np.abs(eigenvalues), axis=0)
    l1, l2, l3 = np.take_along_axis(eigenvalues, order, axis=0)
    a2, a3 = np.abs(l2), np.abs(l3)
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = np.where(a3 > _EPS, a2 / a3, 0.0)
    return ShapeMeasures(tube_ratio=ra, structure2=l1**2 + l2**2 + l3**2)
