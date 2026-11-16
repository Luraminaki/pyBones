"""Segmentation output directory: label volume + typed metadata."""

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from pybones import __version__
from pybones.config.schema import PyBonesConfig
from pybones.io.volume_store import open_volume
from pybones.segmentation.classifier import Thresholds
from pybones.segmentation.refinement import RefinementStats
from pybones.utils.constants import JSON_INDENT

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from pybones.config.schema import IOConfig

#: Key of the writing version inside ``meta.json``.
_VERSION_KEY = "pybones_version"
_UNKNOWN_VERSION = "an unknown version"


class SegmentationMeta(BaseModel):
    """Content of ``meta.json``: provenance, thresholds, statistics and the full config.

    Attributes:
        pybones_version: Version of pyBones that wrote the segmentation.
        source: Absolute path of the source stack.
        shape_zyx: Shape of the label volume.
        thresholds: Global thresholds and class means.
        label_fractions: Fraction of the volume per final label key.
        refinement: What the 3D refinement changed (``None`` if disabled).
        timing_s: Duration of each stage, in seconds.
        config: Full effective configuration.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    pybones_version: str
    source: str
    shape_zyx: tuple[int, int, int]
    thresholds: Thresholds
    label_fractions: dict[str, float]
    refinement: RefinementStats | None
    timing_s: dict[str, float]
    config: PyBonesConfig


class IncompatibleOutputError(ValueError):
    """A segmentation was written by a pyBones whose metadata this version cannot read."""


def labels_path(out_dir: "Path", io_config: "IOConfig") -> "Path":
    """Locate the label volume inside an output directory.

    Args:
        out_dir: Segmentation output directory.
        io_config: File names.

    Returns:
        Path of the label volume.
    """
    return out_dir / io_config.labels_file


def scratch_path(out_dir: "Path", io_config: "IOConfig", stage: str) -> "Path":
    """Locate the intermediate (scratch) volume of a stage.

    Args:
        out_dir: Segmentation output directory.
        io_config: File names.
        stage: Stage name, inserted into the scratch file template.

    Returns:
        Path of the scratch volume.
    """
    return out_dir / io_config.scratch_file_template.format(stage=stage)


def save_meta(meta: SegmentationMeta, out_dir: "Path", io_config: "IOConfig") -> "Path":
    """Write ``meta`` as JSON into ``out_dir``.

    Args:
        meta: Metadata to save.
        out_dir: Segmentation output directory.
        io_config: File names.

    Returns:
        Path of the written file.
    """
    path = out_dir / io_config.meta_file
    _ = path.write_text(meta.model_dump_json(indent=JSON_INDENT))
    return path


def load_meta(out_dir: "Path", io_config: "IOConfig") -> SegmentationMeta:
    """Read and validate the metadata of a segmentation output.

    Only metadata of this version's format is read: pyBones is alpha, and an older
    output is cheaper to segment again than to keep compatible.

    Args:
        out_dir: Segmentation output directory.
        io_config: File names.

    Returns:
        The validated metadata.

    Raises:
        IncompatibleOutputError: If the metadata does not match this version's format.
    """
    text = (out_dir / io_config.meta_file).read_text()
    try:
        return SegmentationMeta.model_validate_json(text)
    except ValidationError as exc:
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError:
            data = None
        written_by = data.get(_VERSION_KEY) if isinstance(data, dict) else None
        version = f"pyBones {written_by}" if written_by else _UNKNOWN_VERSION
        raise IncompatibleOutputError(
            f"{out_dir} was written by {version} and cannot be read by pyBones {__version__} "
            + f"({exc.error_count()} differences): segment the stack again"
        ) from exc


def load_labels(out_dir: "Path", io_config: "IOConfig") -> "NDArray[np.uint8]":
    """Memory-map the label volume of a segmentation output (read-only).

    Args:
        out_dir: Segmentation output directory.
        io_config: File names.

    Returns:
        The ``(depth, height, width)`` label volume.
    """
    return open_volume(labels_path(out_dir, io_config))
