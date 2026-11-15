"""Binary morphology helpers on boolean masks, backed by OpenCV.

All functions take and return ``bool`` arrays. Conversions to/from OpenCV's
``uint8`` are zero-copy views (a numpy bool is one byte holding 0 or 1).

OpenCV's stubs type every returned image as ``MatLike`` (any numeric dtype); the
results are narrowed with :func:`typing.cast`, which costs nothing at run time.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EIGHT_CONNECTED = 8
_FOUR_CONNECTED = 4
_KERNEL_CACHE_SIZE = 64
_HALF_PIXEL = 0.5


#: Kernel radii ``(ry, rx)`` in pixels: a disk for square pixels, an ellipse otherwise, so
#: that a physical radius stays isotropic whatever the pixel aspect ratio.
Radius = tuple[int, int]


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def ellipse(radius: Radius) -> "NDArray[np.uint8]":
    """Elliptic structuring element of radii ``(ry, rx)``.

    Built analytically (pixel centres inside the ellipse enlarged by half a pixel)
    rather than with ``cv2.MORPH_ELLIPSE``, whose disks carry one-pixel spikes at
    their tips and whose one-row ellipses degenerate to a single pixel.

    Args:
        radius: Radii in pixels (the kernel is ``2 * ry + 1`` by ``2 * rx + 1``);
            a zero radius gives a straight line along the other axis.

    Returns:
        A read-only ``uint8`` kernel, cached per radius.

    Raises:
        ValueError: If a radius is negative.
    """
    ry, rx = radius
    if ry < 0 or rx < 0:
        raise ValueError(f"radii must be >= 0, got {radius}")
    yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
    inside = (yy / (ry + _HALF_PIXEL)) ** 2 + (xx / (rx + _HALF_PIXEL)) ** 2 <= 1.0
    kernel = inside.astype(np.uint8)
    kernel.setflags(write=False)
    return kernel


@lru_cache(maxsize=2)
def neighbourhood(connectivity: int) -> "NDArray[np.uint8]":
    """3x3 kernel of the direct neighbours.

    Args:
        connectivity: 4 (cross) or 8 (square).

    Returns:
        A read-only ``uint8`` kernel, cached per connectivity.
    """
    shape = cv2.MORPH_RECT if connectivity == _EIGHT_CONNECTED else cv2.MORPH_CROSS
    kernel = cast("NDArray[np.uint8]", cv2.getStructuringElement(shape, (3, 3)))
    kernel.setflags(write=False)
    return kernel


def as_uint8(mask: "NDArray[np.bool_]") -> "NDArray[np.uint8]":
    """Zero-copy ``uint8`` view of a boolean mask (copies only if not C-contiguous).

    Args:
        mask: Binary image.

    Returns:
        The same bytes viewed as ``uint8`` (0 or 1).
    """
    return np.ascontiguousarray(mask).view(np.uint8)


def dilate_neighbours(mask: "NDArray[np.bool_]", connectivity: int) -> "NDArray[np.bool_]":
    """Grow ``mask`` by its direct neighbours (one step of ``connectivity``).

    Args:
        mask: Binary image.
        connectivity: 4 or 8.

    Returns:
        The dilated mask.
    """
    grown = cast("NDArray[np.uint8]", cv2.dilate(as_uint8(mask), neighbourhood(connectivity)))
    return grown.view(np.bool_)


def label(mask: "NDArray[np.bool_]", connectivity: int) -> "tuple[int, NDArray[np.int32]]":
    """Label connected components.

    Args:
        mask: Binary image.
        connectivity: 4 or 8.

    Returns:
        ``(n_labels, labels)`` where label 0 is the background.
    """
    n, labels = cv2.connectedComponents(as_uint8(mask), connectivity=connectivity, ltype=cv2.CV_32S)
    return n, cast("NDArray[np.int32]", labels)


def label_with_areas(
    mask: "NDArray[np.bool_]", connectivity: int
) -> "tuple[NDArray[np.int32], NDArray[np.int32]]":
    """Label connected components and measure their areas.

    Args:
        mask: Binary image.
        connectivity: 4 or 8.

    Returns:
        ``(labels, areas)`` where ``areas[k]`` is the pixel count of label ``k``
        (``areas[0]`` is the background).
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        as_uint8(mask), connectivity=connectivity, ltype=cv2.CV_32S
    )
    return cast("NDArray[np.int32]", labels), cast("NDArray[np.int32]", stats[:, cv2.CC_STAT_AREA])


def area_open(mask: "NDArray[np.bool_]", min_area: int, connectivity: int) -> "NDArray[np.bool_]":
    """Remove connected components smaller than ``min_area`` pixels (Matlab ``bwareaopen``).

    Args:
        mask: Binary image.
        min_area: Minimum component area to keep, in pixels.
        connectivity: 4 or 8.

    Returns:
        The filtered mask.
    """
    labels, areas = label_with_areas(mask, connectivity)
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def reconstruct(
    seed: "NDArray[np.bool_]", mask: "NDArray[np.bool_]", connectivity: int
) -> "NDArray[np.bool_]":
    """Binary morphological reconstruction by dilation (Matlab ``imreconstruct``).

    For binary images this is exactly "keep the connected components of
    ``mask`` that contain at least one ``seed`` pixel", which a single
    labelling pass computes without any iteration.

    Args:
        seed: Marker image; pixels outside ``mask`` are ignored.
        mask: Binary image to reconstruct within.
        connectivity: 4 or 8.

    Returns:
        The reconstructed mask (a subset of ``mask``).
    """
    n, labels = label(mask, connectivity)
    hit = np.zeros(n, dtype=np.bool_)
    hit[labels[seed]] = True
    hit[0] = False
    return hit[labels]


def fill_holes(mask: "NDArray[np.bool_]") -> "NDArray[np.bool_]":
    """Fill the holes of ``mask``: background regions not connected to the image border.

    Purely topological (no radius): unlike a closing, it never changes the outer
    boundary nor merges separate objects.

    Args:
        mask: Binary image.

    Returns:
        ``mask`` with its enclosed holes filled.
    """
    background = ~mask
    # 4-connected background pairs with 8-connected objects (no leak through diagonals).
    _, labels = label(background, _FOUR_CONNECTED)
    reaches_border = np.zeros(int(labels.max()) + 1, dtype=np.bool_)
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        reaches_border[edge] = True
    reaches_border[0] = False
    return ~reaches_border[labels]
