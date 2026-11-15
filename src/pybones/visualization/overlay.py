"""2D visualisation helpers: contrast stretching and label overlays."""

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from pybones.segmentation.labels import Label
from pybones.utils.constants import BGR_CHANNELS, UINT8_LEVELS, UINT8_MAX

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.config.schema import RenderConfig

_MIN_RANGE = 1e-12


def stretch_to_uint8(img: "NDArray", low_pct: float, high_pct: float) -> "NDArray[np.uint8]":
    """Contrast-stretch an image to ``uint8`` between two of its percentiles.

    Args:
        img: Grayscale image (any dtype).
        low_pct: Percentile mapped to 0.
        high_pct: Percentile mapped to 255 (brighter pixels saturate).

    Returns:
        The stretched image.
    """
    lo, hi = np.percentile(img, (low_pct, high_pct))
    scale = UINT8_MAX / max(float(hi - lo), _MIN_RANGE)
    return cast("NDArray[np.uint8]", cv2.convertScaleAbs(img, alpha=scale, beta=-lo * scale))


def to_uint8(img: "NDArray", render: "RenderConfig") -> "NDArray[np.uint8]":
    """Contrast-stretch an image to ``uint8`` between the configured percentiles.

    Args:
        img: Grayscale image (any dtype).
        render: Contrast settings.

    Returns:
        The stretched image.
    """
    return stretch_to_uint8(img, render.low_percentile, render.high_percentile)


def color_lut(render: "RenderConfig") -> "NDArray[np.uint8]":
    """Build the lookup table mapping label values to their configured colours.

    Args:
        render: Colours of the labels.

    Returns:
        ``(256, 3)`` BGR table (unconfigured values are black).
    """
    lut = np.zeros((UINT8_LEVELS, BGR_CHANNELS), dtype=np.uint8)
    for key, color in render.colors_bgr.items():
        lut[Label.from_key(key)] = color
    return lut


def overlay(
    img: "NDArray", labels: "NDArray[np.uint8]", render: "RenderConfig"
) -> "NDArray[np.uint8]":
    """Blend label colours over a grayscale slice.

    Args:
        img: Grayscale slice (any dtype; contrast-stretched if not ``uint8``).
        labels: Label image of the same shape.
        render: Colours, opacity and contrast settings.

    Returns:
        BGR ``uint8`` image.
    """
    gray = img if img.dtype == np.uint8 else to_uint8(img, render)
    base = cast("NDArray[np.uint8]", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    blended = cv2.addWeighted(
        base, 1.0 - render.alpha, color_lut(render)[labels], render.alpha, 0.0
    )
    mask = labels != Label.BACKGROUND
    base[mask] = blended[mask]
    return base
