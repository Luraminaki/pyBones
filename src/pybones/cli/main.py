"""Entry point of the ``pybones`` command: argument parsing and dispatch."""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from pydantic import ValidationError

from pybones import __version__
from pybones.cli.arguments import (
    CompareArgs,
    ConfigArgs,
    ExportArgs,
    Invocation,
    MeshArgs,
    PreviewArgs,
    SegmentArgs,
    ServeArgs,
)
from pybones.cli.commands import (
    cmd_compare,
    cmd_config,
    cmd_export,
    cmd_mesh,
    cmd_preview,
    cmd_segment,
    cmd_serve,
)
from pybones.config.loader import load_config
from pybones.segmentation.labels import FOREGROUND_LABELS, label_keys
from pybones.utils.logs import configure_logging

#: Program name: console script, log file stem.
PROGRAM_NAME = "pybones"

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pybones.cli.arguments import CommandArgs
    from pybones.config.schema import PyBonesConfig


def _add_label(parser: argparse.ArgumentParser) -> None:
    """Add the required ``--label`` option (a foreground label key) to a sub-command.

    Args:
        parser: Sub-command parser to extend.
    """
    _ = parser.add_argument(
        "--label", required=True, help=f"one of {label_keys(FOREGROUND_LABELS)}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (global options and one sub-parser per command).

    Returns:
        The ``pybones`` argument parser.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Segment and model the intra-osseous vascular network of µCT stacks.",
    )
    _ = parser.add_argument(
        "--config", type=Path, help="JSON file merged over the packaged default configuration"
    )
    _ = parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one configuration value (repeatable), e.g. pipeline.workers=1",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _ = sub.add_parser("config", help="print the effective configuration as JSON")

    p = sub.add_parser("segment", help="segment a slice stack into a label volume")
    _ = p.add_argument("source", type=Path, help="directory or zip archive of slices")
    _ = p.add_argument("-o", "--output", type=Path, required=True, help="output directory")

    p = sub.add_parser("preview", help="overlay the labels of one slice on the raw image")
    _ = p.add_argument("segmentation", type=Path, help="segmentation output directory")
    _ = p.add_argument("-z", type=int, required=True, help="slice index")
    _ = p.add_argument("-o", "--output", type=Path, required=True, help="output image")
    _ = p.add_argument("--source", type=Path, help="raw stack (default: the one in the metadata)")
    _ = p.add_argument("--scale", type=float, help="output scale factor")

    p = sub.add_parser("mesh", help="mesh one label to a binary STL file (µm)")
    _ = p.add_argument("segmentation", type=Path, help="segmentation output directory")
    _add_label(p)
    _ = p.add_argument("-o", "--output", type=Path, required=True, help="output .stl file")

    p = sub.add_parser("export", help="export one label as a PNG sequence (ImageJ)")
    _ = p.add_argument("segmentation", type=Path, help="segmentation output directory")
    _add_label(p)
    _ = p.add_argument("-o", "--output", type=Path, required=True, help="output directory")

    p = sub.add_parser("compare", help="Dice/precision/recall against reference masks")
    _ = p.add_argument("segmentation", type=Path, help="segmentation output directory")
    _ = p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="directory or zip of binary reference masks, one per slice",
    )
    _add_label(p)

    p = sub.add_parser("serve", help="run the local web server (HTTP API)")
    _ = p.add_argument(
        "--server-config", type=Path, help="JSON file merged over the packaged server.json"
    )
    _ = p.add_argument("--host", help="interface to listen on")
    _ = p.add_argument("--port", type=int, help="TCP port")
    _ = p.add_argument("--data-dir", type=Path, help="folder of the slice stacks")
    _ = p.add_argument("--output-dir", type=Path, help="folder of the segmentations")
    return parser


def run_command(args: "CommandArgs", config: "PyBonesConfig") -> None:
    """Run the sub-command matching the type of ``args``.

    The ``match`` is exhaustive: a new argument model without a case here is a type error.

    Args:
        args: Validated arguments of one sub-command.
        config: Effective configuration.
    """
    match args:
        case ConfigArgs():
            cmd_config(config)
        case SegmentArgs():
            cmd_segment(args, config)
        case PreviewArgs():
            cmd_preview(args, config)
        case MeshArgs():
            cmd_mesh(args, config)
        case ExportArgs():
            cmd_export(args, config)
        case CompareArgs():
            cmd_compare(args, config)
        case ServeArgs():
            cmd_serve(args, config)
        case _:
            assert_never(args)


def main(argv: "Sequence[str] | None" = None) -> None:
    """Parse and validate the arguments, load the configuration, run the command.

    This is the ``pybones`` console script (``[project.scripts]`` in pyproject.toml),
    also reachable as ``python -m pybones``. Invalid arguments or configuration exit
    with argparse's usage message (status 2).

    Every run logs to its own file (see :mod:`pybones.utils.logs`).

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Raises:
        SystemExit: If the command fails (the error and its traceback are logged).
    """
    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        invocation = Invocation.from_namespace(vars(namespace))
    except ValidationError as exc:
        parser.error(f"invalid arguments:\n{exc}")
    try:
        config = load_config(invocation.config, invocation.overrides)
    except (ValidationError, ValueError) as exc:
        parser.error(f"invalid configuration:\n{exc}")
    log_file = configure_logging(config.logging, PROGRAM_NAME)
    logger.info(
        "pybones %s: %s (config %s, overrides %s); log %s",
        __version__,
        invocation.args.command,
        invocation.config,
        list(invocation.overrides),
        log_file,
    )
    try:
        run_command(invocation.args, config)
    except Exception as exc:
        logger.exception("pybones %s failed", invocation.args.command)
        raise SystemExit(
            f"pybones {invocation.args.command} failed: {exc} (see {log_file})"
        ) from exc


if __name__ == "__main__":
    main()
