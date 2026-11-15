"""Segmentation output directory: label volume + typed metadata."""

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

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

    Args:
        out_dir: Segmentation output directory.
        io_config: File names.

    Returns:
        The validated metadata.
    """
    return SegmentationMeta.model_validate_json((out_dir / io_config.meta_file).read_text())


def load_labels(out_dir: "Path", io_config: "IOConfig") -> "NDArray[np.uint8]":
    """Memory-map the label volume of a segmentation output (read-only).

    Args:
        out_dir: Segmentation output directory.
        io_config: File names.

    Returns:
        The ``(depth, height, width)`` label volume.
    """
    return open_volume(labels_path(out_dir, io_config))
