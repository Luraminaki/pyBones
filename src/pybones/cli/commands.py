"""Implementation of the ``pybones`` sub-commands (one function per command)."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import uvicorn

from pybones.api.operations import (
    compare_label,
    export_masks,
    preview,
    write_mesh_file,
)
from pybones.config.loader import load_server_config
from pybones.quantification.pipeline import run_quantification
from pybones.segmentation.pipeline import planned_stages, run_segmentation
from pybones.segmentation.stages import MESH_STAGES, QUANTIFY_STAGES, ProgressTracker
from pybones.server.app import DOCS_PATH, create_app, server_url
from pybones.utils.constants import JSON_INDENT

if TYPE_CHECKING:
    from pybones.cli.arguments import (
        CompareArgs,
        ExportArgs,
        MeshArgs,
        PreviewArgs,
        QuantifyArgs,
        SegmentArgs,
        ServeArgs,
    )
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
    """Write the effective configuration (defaults + user file + overrides) as JSON on stdout.

    This is the command's output, not a log message: it is meant to be redirected to a
    file (``pybones config > my_config.json``).

    Args:
        config: Effective configuration.
    """
    print(config.model_dump_json(indent=JSON_INDENT))


def cmd_segment(args: "SegmentArgs", config: "PyBonesConfig") -> None:
    """Segment a stack into a label volume and log the label fractions.

    The pipeline itself logs the thresholds, the refinement and the timings.

    Args:
        args: Source stack and output directory.
        config: Effective configuration.
    """
    tracker = ProgressTracker(planned_stages(config), config.pipeline.stage_weights)
    tracker.add_listener(_print_progress)
    meta = run_segmentation(args.source, args.output, config, tracker)
    fractions = ", ".join(f"{k}={v:.2%}" for k, v in meta.label_fractions.items())
    logger.info("Label fractions of %s: %s", args.output, fractions)


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


def cmd_mesh(args: "MeshArgs", config: "PyBonesConfig") -> None:
    """Mesh one label and save it as an STL or GLB file (coordinates in µm), with progress.

    Args:
        args: Segmentation, label and output file (its extension sets the format).
        config: Effective configuration.
    """
    t0 = time.perf_counter()
    tracker = ProgressTracker(MESH_STAGES, config.mesh.stage_weights)
    tracker.add_listener(_print_progress)
    result = write_mesh_file(args.segmentation, args.label, args.output, config, tracker)
    logger.info(
        "Wrote %s: %s triangles, volume %.4f mm³ in %.1f s",
        result.path,
        f"{result.n_faces:,}",
        result.volume_mm3,
        time.perf_counter() - t0,
    )


def cmd_export(args: "ExportArgs", config: "PyBonesConfig") -> None:
    """Export one label as a PNG image sequence (one binary mask per slice).

    Args:
        args: Segmentation, label and output directory.
        config: Effective configuration.
    """
    _ = export_masks(args.segmentation, args.label, args.output, config)


def cmd_compare(args: "CompareArgs", config: "PyBonesConfig") -> None:
    """Compare one label with a reference sequence of binary masks and log the scores.

    Args:
        args: Segmentation, label and reference masks (directory or zip, one per slice).
        config: Effective configuration.
    """
    result = compare_label(args.segmentation, args.label, args.reference, config)
    logger.info(
        "%s: dice=%.3f precision=%.3f recall=%.3f",
        args.label.key,
        result.dice,
        result.precision,
        result.recall,
    )


def cmd_quantify(args: "QuantifyArgs", config: "PyBonesConfig") -> None:
    """Measure the vessel network of a segmentation, with progress (the run logs its summary).

    Args:
        args: Segmentation output directory (the files are written into it).
        config: Effective configuration (quantification settings).
    """
    tracker = ProgressTracker(QUANTIFY_STAGES, config.quantification.stage_weights)
    tracker.add_listener(_print_progress)
    _ = run_quantification(args.segmentation, config, tracker)


def server_overrides(args: "ServeArgs") -> list[str]:
    """Express the ``serve`` options given on the command line as server config overrides.

    Args:
        args: ``serve`` arguments.

    Returns:
        ``section.key=<JSON value>`` strings (JSON-encoded, so no value is misread).
    """
    given = {
        "http.host": args.host,
        "http.port": args.port,
        "library.data_dir": None if args.data_dir is None else str(args.data_dir),
        "library.output_dir": None if args.output_dir is None else str(args.output_dir),
    }
    return [f"{key}={json.dumps(value)}" for key, value in given.items() if value is not None]


def cmd_serve(args: "ServeArgs", config: "PyBonesConfig") -> None:
    """Run the local web server until interrupted (Ctrl+C).

    Args:
        args: Server configuration file and option overrides.
        config: Base configuration of the segmentations.
    """
    server = load_server_config(args.server_config, server_overrides(args))
    app = create_app(config, server)
    url = server_url(server)
    logger.info(
        "pyBones web server on %s (API documentation: %s%s); data: %s, output: %s; "
        + "Ctrl+C to stop",
        url,
        url,
        DOCS_PATH,
        Path(server.library.data_dir).resolve(),
        Path(server.library.output_dir).resolve(),
    )
    uvicorn.run(
        app,
        host=server.http.host,
        port=server.http.port,
        log_config=None,
        timeout_graceful_shutdown=server.http.shutdown_timeout_s,
    )
