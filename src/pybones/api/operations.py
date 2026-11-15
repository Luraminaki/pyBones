"""The operations of pyBones as plain functions: what the command line and a web front end call.

Each function takes paths and the configuration, returns typed results, and never
prints: presenting the result (terminal, JSON, page) is the caller's job. Long
operations accept a :data:`~pybones.segmentation.stages.ProgressCallback`; to run
them in the background, see :class:`pybones.api.jobs.JobManager`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from pybones.analysis.metrics import compare_stacks
from pybones.io.export import export_mask_sequence
from pybones.io.slices import SliceStack
from pybones.meshing.marching import extract_surface
from pybones.meshing.stl import write_stl
from pybones.segmentation.pipeline import run_segmentation
from pybones.segmentation.results import load_labels, load_meta
from pybones.segmentation.stages import no_progress
from pybones.visualization.overlay import overlay

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.analysis.metrics import Agreement
    from pybones.config.schema import PyBonesConfig
    from pybones.meshing.mesh import Mesh
    from pybones.segmentation.labels import Label
    from pybones.segmentation.results import SegmentationMeta
    from pybones.segmentation.stages import ProgressCallback

#: Prefix of the 80-byte STL header (followed by the label key).
STL_HEADER_PREFIX = "pybones"
UM3_PER_MM3 = 1e9

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MeshResult:
    """Surface of one label.

    Attributes:
        mesh: Triangle mesh in µm, ``(x, y, z)`` order.
        volume_mm3: Enclosed volume.
    """

    mesh: "Mesh"
    volume_mm3: float


def segment(
    source: Path,
    out_dir: Path,
    config: "PyBonesConfig",
    progress: "ProgressCallback" = no_progress,
) -> "SegmentationMeta":
    """Segment a stack into a label volume and its metadata.

    Args:
        source: Directory or zip archive of slices.
        out_dir: Output directory (created if needed).
        config: Configuration.
        progress: Progress callback, e.g. a :class:`~pybones.segmentation.stages.ProgressTracker`
            (which can also cancel the run).

    Returns:
        The metadata written next to the label volume.
    """
    return run_segmentation(source, out_dir, config, progress)


def preview(
    segmentation: Path,
    z: int,
    config: "PyBonesConfig",
    source: Path | None = None,
    scale: float | None = None,
) -> "NDArray[np.uint8]":
    """Render one slice with its labels overlaid.

    Args:
        segmentation: Segmentation output directory.
        z: Slice index.
        config: Configuration (file names, colours).
        source: Raw stack (default: the one recorded in the segmentation metadata).
        scale: Output scale factor (default: full size).

    Returns:
        BGR image.

    Raises:
        IndexError: If ``z`` is outside the stack.
    """
    labels = load_labels(segmentation, config.io)
    if not 0 <= z < labels.shape[0]:
        raise IndexError(f"slice {z} is outside the stack (0..{labels.shape[0] - 1})")
    stack_path = source or Path(load_meta(segmentation, config.io).source)
    with SliceStack(stack_path, config.io) as stack:
        img = stack.read(z)
    out = overlay(img, np.asarray(labels[z]), config.render)
    if scale is not None:
        out = cast(
            "NDArray[np.uint8]",
            cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
        )
    return out


def mesh_label(segmentation: Path, label: "Label", config: "PyBonesConfig") -> MeshResult:
    """Extract the smoothed surface of one label.

    Args:
        segmentation: Segmentation output directory.
        label: Foreground label to mesh.
        config: Configuration (mesh settings, voxel size).

    Returns:
        The mesh and its volume.
    """
    labels = load_labels(segmentation, config.io)
    mesh = extract_surface(labels, label, config.mesh, config.acquisition.voxel_size_um)
    volume_mm3 = mesh.volume(config.mesh.chunk_faces) / UM3_PER_MM3
    logger.info("Meshed %s: %d triangles, %.4f mm3", label.key, mesh.n_faces, volume_mm3)
    return MeshResult(mesh=mesh, volume_mm3=volume_mm3)


def save_mesh(result: MeshResult, label: "Label", path: Path, config: "PyBonesConfig") -> Path:
    """Write a mesh as a binary STL file (coordinates in µm).

    Args:
        result: Mesh to write.
        label: Its label (recorded in the STL header).
        path: Output file.
        config: Configuration (write chunk size).

    Returns:
        The written path.
    """
    header = f"{STL_HEADER_PREFIX} {label.key}"
    return write_stl(result.mesh, path, header, config.mesh.chunk_faces)


def export_masks(
    segmentation: Path, label: "Label", out_dir: Path, config: "PyBonesConfig"
) -> list[Path]:
    """Export one label as a binary PNG sequence (ImageJ: File > Import > Image Sequence).

    Args:
        segmentation: Segmentation output directory.
        label: Label to export.
        out_dir: Output directory.
        config: Configuration (file names).

    Returns:
        Written paths, in z order.
    """
    paths = export_mask_sequence(load_labels(segmentation, config.io), label, out_dir, config.io)
    logger.info("Exported %d %s masks to %s", len(paths), label.key, out_dir)
    return paths


def compare_label(
    segmentation: Path, label: "Label", reference: Path, config: "PyBonesConfig"
) -> "Agreement":
    """Compare one label with a reference sequence of binary masks.

    Args:
        segmentation: Segmentation output directory.
        label: Label to compare.
        reference: Directory or zip of binary masks, one per slice.
        config: Configuration (file names).

    Returns:
        Dice, precision and recall counts.

    Raises:
        ValueError: If the reference and the segmentation differ in slice count.
    """
    labels = load_labels(segmentation, config.io)
    with SliceStack(reference, config.io) as ref:
        if len(ref) != labels.shape[0]:
            raise ValueError(f"Reference has {len(ref)} slices, volume has {labels.shape[0]}")
        return compare_stacks((labels[z] == label for z in range(len(ref))), (m > 0 for m in ref))
