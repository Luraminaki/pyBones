"""Saving and loading a rendered phantom (16-bit PNG slices + compressed ground truth)."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from tools.phantom.generator import Phantom

if TYPE_CHECKING:
    from pathlib import Path

    from tools.phantom.settings import PhantomSettings


def save_phantom(phantom: Phantom, directory: "Path", settings: "PhantomSettings") -> None:
    """Write the slices (readable by ``pybones segment``) and the ground truth.

    Args:
        phantom: Rendered phantom.
        directory: Output directory.
        settings: Phantom settings (file names).

    Raises:
        OSError: If a slice cannot be written.
    """
    slices_dir = directory / settings.files.slices_dir
    slices_dir.mkdir(parents=True, exist_ok=True)
    for z, img in enumerate(phantom.image):
        path = slices_dir / settings.files.slice_template.format(index=z)
        if not cv2.imwrite(str(path), img):
            raise OSError(f"Could not write {path}")
    np.savez_compressed(
        directory / settings.files.truth,
        truth=phantom.truth,
        vessel_radius_um=phantom.vessel_radius_um,
        clotted=phantom.clotted,
    )


def load_phantom(directory: "Path", settings: "PhantomSettings") -> Phantom:
    """Read the ground truth of a saved phantom (the image is not loaded).

    Args:
        directory: Phantom directory.
        settings: Phantom settings (file names).

    Returns:
        The phantom, with an empty image.
    """
    data = np.load(directory / settings.files.truth)
    return Phantom(
        image=np.empty(0, np.uint16),
        truth=data["truth"],
        vessel_radius_um=data["vessel_radius_um"],
        clotted=data["clotted"],
    )
