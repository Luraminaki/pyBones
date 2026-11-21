"""Saving and loading a rendered phantom (16-bit PNG slices + compressed ground truth)."""

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from tools.phantom.generator import Phantom, VesselNetwork

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from tools.phantom.settings import PhantomSettings


#: Prefix of the network arrays inside the truth file.
NETWORK_PREFIX = "network_"
#: Names of the network's arrays (its dataclass fields).
_NETWORK_FIELDS = tuple(f.name for f in fields(VesselNetwork))


def _network_arrays(network: VesselNetwork) -> "list[tuple[str, NDArray[Any]]]":
    """List the arrays of a network with their names.

    Args:
        network: Vessel network.

    Returns:
        ``(field name, array)`` pairs.
    """
    return [(name, getattr(network, name)) for name in _NETWORK_FIELDS]


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
    arrays: dict[str, Any] = {
        "truth": phantom.truth,
        "vessel_radius_um": phantom.vessel_radius_um,
        "clotted": phantom.clotted,
    }
    arrays.update(
        (f"{NETWORK_PREFIX}{name}", value) for name, value in _network_arrays(phantom.network)
    )
    np.savez_compressed(directory / settings.files.truth, **arrays)


def load_phantom(directory: "Path", settings: "PhantomSettings") -> Phantom:
    """Read the ground truth of a saved phantom (the image is not loaded).

    Args:
        directory: Phantom directory.
        settings: Phantom settings (file names).

    Returns:
        The phantom, with an empty image.

    Raises:
        ValueError: If the phantom predates the recorded vessel network.
    """
    data = np.load(directory / settings.files.truth)
    missing = [n for n in _NETWORK_FIELDS if f"{NETWORK_PREFIX}{n}" not in data.files]
    if missing:
        raise ValueError(
            f"{directory} was generated before the phantom recorded its vessel network: "
            + "generate it again (python -m tools.phantom generate), the image does not change"
        )
    return Phantom(
        image=np.empty(0, np.uint16),
        truth=data["truth"],
        vessel_radius_um=data["vessel_radius_um"],
        clotted=data["clotted"],
        network=VesselNetwork(
            **{name: data[f"{NETWORK_PREFIX}{name}"] for name in _NETWORK_FIELDS}
        ),
    )
