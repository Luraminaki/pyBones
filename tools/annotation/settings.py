"""Configuration of the annotation tool (``config.json`` validated by pydantic)."""

import json
from pathlib import Path
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pybones.segmentation.labels import FINAL_LABELS
from pybones.utils.constants import UINT8_MAX

#: The tool's own configuration file, next to this module.
DEFAULT_SETTINGS_PATH = Path(__file__).with_name("config.json")
_MAX_PORT = 65535

Channel = Annotated[int, Field(ge=0, le=UINT8_MAX)]
FileName = Annotated[str, Field(min_length=1, pattern=r"^[^/\\]+$")]


class _Section(BaseModel):
    """Base of every section: unknown keys are rejected and instances are immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class FilesSettings(_Section):
    """Names inside a workspace directory.

    Attributes:
        manifest: Manifest file.
        labels_dir: Directory of the painted labels.
        crops_dir: Directory of the crop images.
        prefill_dir: Directory of the pre-fills.
        label_png_template: Label image name (``{crop_id}``).
        crop_png_template: Crop image name (``{crop_id}``, ``{offset}``).
        notes_dir: Directory of the annotators' notes.
        note_template: Note file name (``{crop_id}``).
    """

    manifest: Annotated[FileName, Field(pattern=r"\.json$")]
    labels_dir: FileName
    crops_dir: FileName
    prefill_dir: FileName
    label_png_template: Annotated[FileName, Field(pattern=r"\{crop_id\}.*\.png$")]
    crop_png_template: Annotated[FileName, Field(pattern=r"\{crop_id\}.*\{offset[^}]*\}.*\.png$")]
    notes_dir: FileName
    note_template: Annotated[FileName, Field(pattern=r"\{crop_id\}.*\.txt$")]


class CropSettings(_Section):
    """How crops are chosen (see the field descriptions)."""

    size: int = Field(ge=32, le=2048, description="Square crop side in pixels.")
    context_slices: int = Field(ge=0, le=50, description="Neighbouring slices on each side.")
    z_samples: int = Field(ge=1, le=1000, description="Slices sampled for candidate crops.")
    min_tissue_fraction: float = Field(
        ge=0.0, le=1.0, description="Minimum non-background fraction of a candidate."
    )
    n_burst: int = Field(ge=0, le=1000, description="Crops richest in burst.")
    n_vessel: int = Field(ge=0, le=1000, description="Crops richest in vessels.")
    n_random: int = Field(ge=0, le=1000, description="Random tissue crops.")
    seed: int = Field(ge=0, description="Random seed (reproducible selection).")


class ServerSettings(_Section):
    """Local web server.

    Attributes:
        host: Interface to listen on.
        port: Port (0 = any free port).
    """

    host: str = Field(min_length=1)
    port: int = Field(ge=0, le=_MAX_PORT, description="0 = any free port.")


class DisplaySettings(_Section):
    """Annotation page rendering.

    Attributes:
        scale: Screen pixels per image pixel.
        overlay_alpha: Opacity of the painted labels.
        unsure_rgb: Colour of undecided pixels.
        undo_depth: Undo history length.
        max_brush_radius: Largest brush, in image pixels.
        max_zoom: Largest magnification of the view (mouse wheel), on top of ``scale``.
        zoom_step: Magnification factor of one wheel notch.
        autosave_seconds: Labels are saved this long after the last edit (0 = only on demand).
        wand_tolerance: Initial grey-level tolerance of the wand (8-bit levels).
        max_wand_tolerance: Largest tolerance the wand slider offers.
    """

    scale: int = Field(ge=1, le=16, description="Screen pixels per image pixel.")
    overlay_alpha: float = Field(ge=0.0, le=1.0)
    unsure_rgb: tuple[Channel, Channel, Channel]
    undo_depth: int = Field(ge=1, le=1000)
    max_brush_radius: int = Field(ge=1, le=200)
    max_zoom: float = Field(ge=1.0, le=64.0)
    zoom_step: float = Field(gt=1.0, le=4.0)
    autosave_seconds: float = Field(ge=0.0, le=3600.0)
    wand_tolerance: int = Field(ge=0, le=UINT8_MAX)
    max_wand_tolerance: int = Field(ge=1, le=UINT8_MAX)


class AnnotationSettings(_Section):
    """Root of ``tools/annotation/config.json``.

    Attributes:
        workspace_root: Directory holding the workspaces.
        files: Names inside a workspace.
        crops: Crop selection.
        prefill: Start from the current segmentation (faster, biased).
        unsure_value: Label value excluded from the scores.
        max_note_chars: Longest note an annotator can attach to a crop.
        server: Web server.
        display: Page rendering.
    """

    workspace_root: Annotated[str, Field(min_length=1)]
    files: FilesSettings
    crops: CropSettings
    prefill: bool = Field(description="Start from the current segmentation (faster, biased).")
    unsure_value: int = Field(ge=0, le=UINT8_MAX, description="Label value excluded from scores.")
    max_note_chars: int = Field(ge=1, le=100_000)
    server: ServerSettings
    display: DisplaySettings

    @model_validator(mode="after")
    def _unsure_is_not_a_label(self) -> Self:
        """Check that the "unsure" value cannot be mistaken for a label.

        Returns:
            The validated settings.

        Raises:
            ValueError: If ``unsure_value`` is a final label value.
        """
        if self.unsure_value in {int(lab) for lab in FINAL_LABELS}:
            raise ValueError(f"unsure_value {self.unsure_value} collides with a pybones label")
        return self

    @property
    def total_crops(self) -> int:
        """Number of crops requested over all categories."""
        return self.crops.n_burst + self.crops.n_vessel + self.crops.n_random


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> AnnotationSettings:
    """Read and validate the tool configuration.

    Args:
        path: JSON settings file (default: the one next to this module).

    Returns:
        The validated settings.
    """
    return AnnotationSettings.model_validate(json.loads(path.read_text()))
