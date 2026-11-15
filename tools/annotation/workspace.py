"""Annotation workspace on disk: manifest, crop images, pre-fills and painted labels.

Layout (names from the ``files`` settings)::

    annotations/<name>/
        manifest.json      # crops (position, category) + provenance   -> versioned
        labels/<id>.png    # painted labels (uint8, pybones values)    -> versioned
        crops/<id>_<o>.png # 8-bit raw crop at z + o                  -> regenerable
        prefill/<id>.png   # segmentation used as starting point     -> regenerable
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from pybones.segmentation.labels import FINAL_LABELS
from pybones.utils.constants import JSON_INDENT, UINT8_LEVELS

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from tools.annotation.settings import AnnotationSettings, FilesSettings

#: Notes are plain UTF-8 text.
_NOTE_ENCODING = "utf-8"


class CropCategory(StrEnum):
    """Why a crop was selected."""

    BURST = "burst"
    VESSEL = "vessel"
    RANDOM = "random"


class CropSpec(BaseModel):
    """One annotation crop: the square ``[y, y + size) x [x, x + size)`` of slice ``z``.

    Attributes:
        crop_id: Unique name of the crop (file names, URLs).
        z: Slice index.
        y: Top row.
        x: Left column.
        category: Why the crop was selected.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    crop_id: str
    z: int
    y: int
    x: int
    category: CropCategory


class Manifest(BaseModel):
    """Content of ``manifest.json``.

    Attributes:
        source: Raw slice stack (relative to the repository when inside it).
        segmentation: Segmentation the crops were chosen from.
        segmentation_version: pyBones version of that segmentation.
        crop_size: Side of the square crops, in pixels.
        context_slices: Neighbouring slices shown on each side.
        prefilled: Whether crops start from the segmentation.
        unsure_value: Label value of pixels left undecided (ignored by the scores).
        created: Creation time (ISO 8601, UTC).
        crops: The crops.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    source: str
    segmentation: str
    segmentation_version: str
    crop_size: int
    context_slices: int
    prefilled: bool
    unsure_value: int
    created: str
    crops: list[CropSpec]


def now_iso() -> str:
    """Format the current UTC time.

    Returns:
        ISO 8601 time, to the second.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


class Workspace:
    """Access to one annotation workspace directory.

    Attributes:
        root: Workspace directory.
        unsure_value: Label value of undecided pixels.
        max_note_chars: Longest note accepted.
    """

    def __init__(self, root: "Path", settings: "AnnotationSettings") -> None:
        """Bind to ``root`` (not created until something is written).

        Args:
            root: Workspace directory, e.g. ``annotations/102OAcentri8sDXZ08``.
            settings: Tool settings (file names, unsure value).
        """
        self.root: Path = root
        self._files: FilesSettings = settings.files
        self.unsure_value: int = settings.unsure_value
        self.max_note_chars: int = settings.max_note_chars
        self._allowed: NDArray[np.bool_] = np.zeros(UINT8_LEVELS, dtype=np.bool_)
        self._allowed[[int(lab) for lab in FINAL_LABELS]] = True
        self._allowed[settings.unsure_value] = True

    @property
    def manifest_path(self) -> "Path":
        """Path of ``manifest.json``."""
        return self.root / self._files.manifest

    def save_manifest(self, manifest: Manifest) -> None:
        """Write the manifest (creating the workspace directories).

        Args:
            manifest: Manifest to save.
        """
        for sub in (self._files.labels_dir, self._files.crops_dir, self._files.prefill_dir):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        _ = self.manifest_path.write_text(manifest.model_dump_json(indent=JSON_INDENT))

    def load_manifest(self) -> Manifest:
        """Read and validate the manifest.

        Returns:
            The manifest.
        """
        return Manifest.model_validate_json(self.manifest_path.read_text())

    def crop_image_path(self, crop_id: str, offset: int) -> "Path":
        """Locate the 8-bit raw image of a crop at slice ``z + offset``.

        Args:
            crop_id: Crop id.
            offset: Slice offset from the crop's slice.

        Returns:
            Path of the PNG image.
        """
        name = self._files.crop_png_template.format(crop_id=crop_id, offset=offset)
        return self.root / self._files.crops_dir / name

    def label_path(self, crop_id: str) -> "Path":
        """Locate the painted labels of a crop.

        Args:
            crop_id: Crop id.

        Returns:
            Path of the PNG label image.
        """
        name = self._files.label_png_template.format(crop_id=crop_id)
        return self.root / self._files.labels_dir / name

    def prefill_path(self, crop_id: str) -> "Path":
        """Locate the segmentation used as the starting point of a crop.

        Args:
            crop_id: Crop id.

        Returns:
            Path of the PNG label image.
        """
        name = self._files.label_png_template.format(crop_id=crop_id)
        return self.root / self._files.prefill_dir / name

    def note_path(self, crop_id: str) -> "Path":
        """Locate the note of a crop.

        Args:
            crop_id: Crop id.

        Returns:
            Path of the text file.
        """
        name = self._files.note_template.format(crop_id=crop_id)
        return self.root / self._files.notes_dir / name

    def read_note(self, crop_id: str) -> str:
        """Read the note of a crop.

        Args:
            crop_id: Crop id.

        Returns:
            The note (empty if there is none).
        """
        path = self.note_path(crop_id)
        return path.read_text(encoding=_NOTE_ENCODING) if path.is_file() else ""

    def write_note(self, crop_id: str, text: str) -> None:
        """Save the note of a crop (an empty note deletes the file).

        Args:
            crop_id: Crop id.
            text: The note.

        Raises:
            ValueError: If the note is longer than ``max_note_chars``.
        """
        if len(text) > self.max_note_chars:
            raise ValueError(f"note longer than {self.max_note_chars} characters")
        path = self.note_path(crop_id)
        if not text.strip():
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(text, encoding=_NOTE_ENCODING)

    def is_annotated(self, crop_id: str) -> bool:
        """Tell whether labels were saved for a crop.

        Args:
            crop_id: Crop id.

        Returns:
            ``True`` once the crop has been annotated.
        """
        return self.label_path(crop_id).exists()

    def read_labels(self, crop_id: str, size: int) -> "NDArray[np.uint8]":
        """Read the labels of a crop: saved ones, else the pre-fill, else all "unsure".

        Args:
            crop_id: Crop id.
            size: Crop side (for the all-"unsure" image).

        Returns:
            ``(size, size)`` uint8 labels.

        Raises:
            OSError: If an existing label image cannot be decoded.
        """
        for path in (self.label_path(crop_id), self.prefill_path(crop_id)):
            if path.exists():
                labels = cast(
                    "NDArray[np.uint8] | None", cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                )
                if labels is None:
                    raise OSError(f"Could not read {path}")
                return labels
        return np.full((size, size), self.unsure_value, dtype=np.uint8)

    def write_labels(self, crop_id: str, labels: "NDArray[np.uint8]") -> None:
        """Validate and save painted labels.

        Args:
            crop_id: Crop id.
            labels: ``(size, size)`` uint8 labels.

        Raises:
            ValueError: If the image holds values other than pybones labels and "unsure".
            OSError: If the image cannot be written.
        """
        if not self._allowed[labels].all():
            raise ValueError(f"Unexpected label values {sorted(set(np.unique(labels).tolist()))}")
        path = self.label_path(crop_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), labels):
            raise OSError(f"Could not write {path}")
