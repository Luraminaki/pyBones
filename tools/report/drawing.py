"""Image helpers of the report figures: file output, conversions, titles and scale bars."""

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from numpy.typing import NDArray

    from tools.report.settings import FigureSettings

_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_THICKNESS = 2
_TEXT_MARGIN = 10
_SCALE_BAR_HEIGHT_PX = 8
_FILLED = -1
_UM_PER_MM = 1000


def write_image(path: "Path", image: "NDArray[np.uint8]", jpeg_quality: int) -> None:
    """Write an image file (the JPEG quality is ignored by other formats).

    Args:
        path: Output file; its extension selects the format.
        image: Image to write.
        jpeg_quality: JPEG quality (1-100).

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise OSError(f"Could not write {path}")


def to_bgr(gray: "NDArray[np.uint8]") -> "NDArray[np.uint8]":
    """Convert a grayscale image to BGR.

    Args:
        gray: ``uint8`` grayscale image.

    Returns:
        The three-channel image.
    """
    return cast("NDArray[np.uint8]", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


def downscale(image: "NDArray[np.uint8]", scale: float) -> "NDArray[np.uint8]":
    """Shrink an image by area averaging.

    Args:
        image: Image to shrink.
        scale: Scale factor (<= 1).

    Returns:
        The resized image.
    """
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cast("NDArray[np.uint8]", resized)


def _put_text(image: "NDArray[np.uint8]", text: str, origin: tuple[int, int], scale: float) -> None:
    """Write white text on an image (in place).

    Args:
        image: BGR image.
        text: Text to write.
        origin: Bottom-left corner of the text, ``(x, y)``.
        scale: OpenCV font scale.
    """
    _ = cv2.putText(image, text, origin, _FONT, scale, _WHITE, _TEXT_THICKNESS, cv2.LINE_AA)


def compose(
    panels: "list[NDArray[np.uint8]]", titles: "Iterable[str]", fig: "FigureSettings"
) -> "NDArray[np.uint8]":
    """Place panels side by side, with a title strip above each.

    Args:
        panels: BGR panels (heights may differ; shorter ones are padded).
        titles: One title per panel.
        fig: Layout settings (title height, font, gap).

    Returns:
        The composed BGR image.
    """
    height = max(p.shape[0] for p in panels)

    def titled(panel: "NDArray[np.uint8]", title: str) -> "NDArray[np.uint8]":
        """Put a title strip above a panel, padded to the common height.

        Args:
            panel: BGR panel.
            title: Its title.

        Returns:
            The titled column.
        """
        strip = np.zeros((fig.title_height_px, panel.shape[1], 3), np.uint8)
        _put_text(
            strip, title, (_TEXT_MARGIN, fig.title_height_px - _TEXT_MARGIN), fig.title_font_scale
        )
        body = np.zeros((height, panel.shape[1], 3), np.uint8)
        body[: panel.shape[0]] = panel
        return np.vstack([strip, body])

    gap = np.zeros((height + fig.title_height_px, fig.panel_gap_px, 3), np.uint8)
    columns = [titled(p, t) for p, t in zip(panels, titles, strict=True)]
    return np.hstack([part for column in columns for part in (gap, column)][1:])


def add_scale_bar(image: "NDArray[np.uint8]", pixels_per_um: float, fig: "FigureSettings") -> None:
    """Draw a labelled scale bar in the lower-left corner (in place).

    Args:
        image: BGR image, modified in place.
        pixels_per_um: Image scale.
        fig: Figure settings (bar length, font).
    """
    length = round(fig.scale_bar_um * pixels_per_um)
    y = image.shape[0] - 2 * _TEXT_MARGIN
    top_left = (_TEXT_MARGIN, y - _SCALE_BAR_HEIGHT_PX)
    _ = cv2.rectangle(image, top_left, (_TEXT_MARGIN + length, y), _WHITE, _FILLED)
    label = (
        f"{fig.scale_bar_um / _UM_PER_MM:g} mm"
        if fig.scale_bar_um >= _UM_PER_MM
        else f"{fig.scale_bar_um:g} um"
    )
    origin = (_TEXT_MARGIN, y - _SCALE_BAR_HEIGHT_PX - _TEXT_MARGIN)
    _put_text(image, label, origin, fig.title_font_scale)
