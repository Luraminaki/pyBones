"""Implementation of the ``pybones`` sub-commands (one function per command)."""

import logging
import sys
import time
from typing import TYPE_CHECKING

import cv2

from pybones.api.operations import (
    compare_label,
    export_masks,
    mesh_label,
    preview,
    save_mesh,
    segment,
)
from pybones.segmentation.pipeline import planned_stages
from pybones.segmentation.stages import ProgressTracker
from pybones.utils.constants import JSON_INDENT

if TYPE_CHECKING:
    from pybones.cli.arguments import CompareArgs, ExportArgs, MeshArgs, PreviewArgs, SegmentArgs
    from pybones.config.schema import PyBonesConfig
    from pybones.segmentation.stages import Progress

_PERCENT = 100

logger = logging.getLogger(__name__)


def progress_line(progress: "Progress") -> str:
    """Describe a run's progress on one line.

    Args:
        progress: Snapshot from a :class:`~pybones.segmentation.stages.ProgressTracker`.

    Returns:
        E.g. ``[2/5] classify   37/100   42%  ~12 s left``.
    """
    eta = "" if progress.remaining_s is None else f"  ~{progress.remaining_s:.0f} s left"
    return (
        f"[{progress.stage_index}/{progress.stage_count}] {progress.stage!s:<10} "
        + f"{progress.done:>4}/{progress.total:<4} {_PERCENT * progress.fraction:4.0f}%{eta}"
    )


def _print_progress(progress: "Progress") -> None:
    """Print the progress line in place on stderr (a new line when a stage ends).

    Args:
        progress: Snapshot of the run.
    """
    end = "\n" if progress.done == progress.total else ""
    print(f"\r  {progress_line(progress)}", end=end, file=sys.stderr, flush=True)


def cmd_config(config: "PyBonesConfig") -> None:
    """Print the effective configuration (defaults + user file + overrides) as JSON.

    Args:
        config: Effective configuration.
    """
    print(config.model_dump_json(indent=JSON_INDENT))


def cmd_segment(args: "SegmentArgs", config: "PyBonesConfig") -> None:
    """Segment a stack into a label volume and print a summary.

    Args:
        args: Source stack and output directory.
        config: Effective configuration.
    """
    tracker = ProgressTracker(planned_stages(config), config.pipeline.stage_weights)
    tracker.add_listener(_print_progress)
    meta = segment(args.source, args.output, config, tracker)
    fractions = ", ".join(f"{k}={v:.2%}" for k, v in meta.label_fractions.items())
    timings = ", ".join(f"{k}={v:.1f}s" for k, v in meta.timing_s.items())
    print(f"Wrote {args.output}\n  thresholds: {meta.thresholds}\n  fractions: {fractions}")
    print(f"  timings: {timings}")
    if meta.refinement is not None:
        print(f"  refinement: {meta.refinement}")


def cmd_preview(args: "PreviewArgs", config: "PyBonesConfig") -> None:
    """Render one slice with its labels overlaid and save it as an image.

    Args:
        args: Segmentation, slice index, output image, optional raw stack and scale factor.
        config: Effective configuration.

    Raises:
        OSError: If the image cannot be written.
    """
    image = preview(args.segmentation, args.z, config, args.source, args.scale)
    if not cv2.imwrite(str(args.output), image):
        raise OSError(f"Could not write {args.output}")
    logger.info("Wrote preview of slice %d to %s", args.z, args.output)
    print(f"Wrote {args.output}")


def cmd_mesh(args: "MeshArgs", config: "PyBonesConfig") -> None:
    """Mesh one label and save it as a binary STL file (coordinates in µm).

    Args:
        args: Segmentation, label and output file.
        config: Effective configuration.
    """
    t0 = time.perf_counter()
    result = mesh_label(args.segmentation, args.label, config)
    _ = save_mesh(result, args.label, args.output, config)
    print(
        f"Wrote {args.output}: {result.mesh.n_faces:,} triangles, "
        + f"volume {result.volume_mm3:.4f} mm³ in {time.perf_counter() - t0:.1f}s"
    )


def cmd_export(args: "ExportArgs", config: "PyBonesConfig") -> None:
    """Export one label as a PNG image sequence (one binary mask per slice).

    Args:
        args: Segmentation, label and output directory.
        config: Effective configuration.
    """
    paths = export_masks(args.segmentation, args.label, args.output, config)
    print(f"Wrote {len(paths)} slices to {args.output}")


def cmd_compare(args: "CompareArgs", config: "PyBonesConfig") -> None:
    """Compare one label with a reference sequence of binary masks and print the scores.

    Args:
        args: Segmentation, label and reference masks (directory or zip, one per slice).
        config: Effective configuration.
    """
    result = compare_label(args.segmentation, args.label, args.reference, config)
    print(
        f"{args.label.key}: dice={result.dice:.3f} "
        + f"precision={result.precision:.3f} recall={result.recall:.3f}"
    )
