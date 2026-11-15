"""Small, versioned extracts of each version's segmentation: what the report figures need.

A run of the real stack weighs hundreds of MB and can only be reproduced from the
version's commit. The figures only use a few slices of it and the burst area of
each slice, so ``tools.report history`` keeps exactly that in
``<extracts_dir>/<key>.npz`` (a few hundred KB, compressed). The figures read the
extracts only, so the report can be rebuilt where the past commits do not exist.
"""

import logging
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from pybones.segmentation.labels import Label

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from tools.report.settings import ReportSettings

EXTRACT_SUFFIX = ".npz"
#: Layout of the 2018 Matlab masks: ``<series>/<series>_<z>.png``.
_LEGACY_BONE_SERIES = "travee"
_LEGACY_VESSEL_SERIES = "vaisseaux"
_LEGACY_MASK_TEMPLATE = "{name}_{z:03d}.png"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Extract:
    """The part of a segmentation the report uses.

    Attributes:
        z: Indices of the stored slices.
        slices: ``(len(z), Y, X)`` label slices.
        burst_fraction: Burst area of every slice of the stack, as a fraction.
    """

    z: "NDArray[np.int64]"
    slices: "NDArray[np.uint8]"
    burst_fraction: "NDArray[np.float64]"

    def labels(self, z: int) -> "NDArray[np.uint8]":
        """Return the label slice ``z``.

        Args:
            z: Slice index.

        Returns:
            The label image.

        Raises:
            KeyError: If slice ``z`` was not extracted (a figure setting changed:
                re-run ``tools.report history``).
        """
        hits = np.flatnonzero(self.z == z)
        if hits.size == 0:
            raise KeyError(f"slice {z} is not in the extract (stored: {self.z.tolist()})")
        return self.slices[int(hits[0])]


def needed_slices(settings: "ReportSettings") -> tuple[int, ...]:
    """List the slices the figures show for a version.

    Args:
        settings: Report settings (overview, profile and crops).

    Returns:
        Sorted slice indices.
    """
    fig = settings.figures
    return tuple(sorted({fig.overview_slice, fig.profile.z, *(c.z for c in fig.crops.values())}))


def extract_path(settings: "ReportSettings", repo: "Path", key: str) -> "Path":
    """Locate the extract of a version.

    Args:
        settings: Report settings (extracts directory).
        repo: Repository root.
        key: Version key (or the legacy key).

    Returns:
        Path of the ``.npz`` file.
    """
    return repo / settings.extracts_dir / f"{key}{EXTRACT_SUFFIX}"


def extract_volume(volume: "NDArray[np.uint8]", slices: tuple[int, ...]) -> Extract:
    """Extract the needed slices and the burst area per slice of a label volume.

    Args:
        volume: ``(Z, Y, X)`` label volume (memory-mapped welcome).
        slices: Slice indices to keep.

    Returns:
        The extract.
    """
    burst = np.array(
        [float((np.asarray(volume[z]) == Label.BURST).mean()) for z in range(len(volume))]
    )
    return Extract(
        z=np.array(slices, dtype=np.int64),
        slices=np.stack([np.asarray(volume[z]) for z in slices]),
        burst_fraction=burst,
    )


def _read_mask(path: "Path") -> "NDArray[np.bool_]":
    """Read a binary mask image.

    Args:
        path: Grayscale image file.

    Returns:
        ``True`` where the image is non-zero.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cast("NDArray[np.uint8] | None", cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
    if image is None:
        raise FileNotFoundError(path)
    return image > 0


def extract_legacy_masks(masks_dir: "Path", slices: tuple[int, ...], depth: int) -> Extract:
    """Build the extract of the 2018 Matlab masks (bone and vessels; bursts were not labelled).

    Args:
        masks_dir: Directory holding the ``travee`` and ``vaisseaux`` mask series.
        slices: Slice indices to keep.
        depth: Number of slices of the stack.

    Returns:
        The extract (zero burst area everywhere).
    """

    def labels(z: int) -> "NDArray[np.uint8]":
        """Combine the two 2018 masks of slice ``z`` into a label image.

        Args:
            z: Slice index.

        Returns:
            The label image.
        """
        bone, vessel = (
            _read_mask(masks_dir / name / _LEGACY_MASK_TEMPLATE.format(name=name, z=z))
            for name in (_LEGACY_BONE_SERIES, _LEGACY_VESSEL_SERIES)
        )
        image = np.zeros(bone.shape, np.uint8)
        image[bone] = Label.BONE
        image[vessel] = Label.VESSEL
        return image

    return Extract(
        z=np.array(slices, dtype=np.int64),
        slices=np.stack([labels(z) for z in slices]),
        burst_fraction=np.zeros(depth),
    )


def save_extract(extract: Extract, path: "Path") -> None:
    """Write an extract (compressed).

    Args:
        extract: Extract to save.
        path: Output ``.npz`` file (its directory is created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **asdict(extract))
    logger.info("Wrote %s (%d slices)", path, len(extract.z))


def load_extract(path: "Path") -> Extract:
    """Read an extract.

    Args:
        path: ``.npz`` file written by :func:`save_extract`.

    Returns:
        The extract.

    Raises:
        FileNotFoundError: If the extract does not exist (run ``tools.report history``).
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `python -m tools.report history` first")
    with np.load(path) as data:
        return Extract(**{f.name: data[f.name] for f in fields(Extract)})
