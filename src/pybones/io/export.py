"""Export label volumes to formats used in the lab (ImageJ / CTan image sequences)."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.config.schema import IOConfig
    from pybones.segmentation.labels import Label

_MASK_ON = 255


def export_mask_sequence(
    volume: "NDArray[np.uint8]", label: "Label", out_dir: "Path", io_config: "IOConfig"
) -> list["Path"]:
    """Write one binary PNG (0/255) per slice for ``label``, streaming over z.

    The result opens in ImageJ with *File > Import > Image Sequence*.

    Args:
        volume: ``(Z, Y, X)`` label volume (memmap welcome).
        label: Label to export.
        out_dir: Destination directory (created if needed).
        io_config: Provides the file name template.

    Returns:
        Written paths, in z order.

    Raises:
        OSError: If a file cannot be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    buf = np.empty(volume.shape[1:], dtype=np.uint8)
    for z in range(volume.shape[0]):
        _ = np.multiply(volume[z] == label, _MASK_ON, out=buf, casting="unsafe")
        path = out_dir / io_config.mask_png_template.format(label=label.key, index=z)
        if not cv2.imwrite(str(path), buf):
            raise OSError(f"Could not write {path}")
        paths.append(path)
    return paths
