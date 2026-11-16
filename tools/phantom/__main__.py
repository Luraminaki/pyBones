"""``python -m tools.phantom {generate,score}`` (run from the repository root)."""

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import numpy as np
from pydantic import ValidationError

from pybones.config.loader import load_config
from pybones.io.volume_store import open_volume
from pybones.segmentation.labels import FINAL_LABELS
from pybones.segmentation.results import labels_path
from pybones.utils.constants import UINT8_LEVELS
from pybones.utils.logs import configure_logging
from tools.phantom.arguments import Command, GenerateArgs, PhantomInvocation, ScoreArgs
from tools.phantom.generator import render_phantom
from tools.phantom.scoring import score_phantom
from tools.phantom.settings import DEFAULT_SETTINGS_PATH, load_settings
from tools.phantom.storage import load_phantom, save_phantom

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pybones.config.schema import PyBonesConfig
    from tools.phantom.settings import PhantomSettings


#: Program name of the log files.
PROGRAM_NAME = "tools-phantom"


logger = logging.getLogger(__name__)


def cmd_generate(output: Path, settings: "PhantomSettings", config: "PyBonesConfig") -> None:
    """Render the phantom: 16-bit PNG slices + compressed ground truth.

    Args:
        output: Output directory.
        settings: Phantom settings.
        config: pyBones configuration (voxel size).
    """
    t0 = time.perf_counter()
    phantom = render_phantom(settings, config.acquisition.voxel_size_um)
    save_phantom(phantom, output, settings)
    counts = np.bincount(phantom.truth.ravel(), minlength=UINT8_LEVELS)
    fractions = ", ".join(
        f"{lab.key}={counts[lab] / phantom.truth.size:.2%}" for lab in FINAL_LABELS
    )
    logger.info(
        "Phantom %s in %.1f s -> %s; truth: %s, clotted vessel voxels=%d",
        phantom.image.shape,
        time.perf_counter() - t0,
        output,
        fractions,
        int(phantom.clotted.sum()),
    )


def cmd_score(
    segmentation: Path, phantom_dir: Path, settings: "PhantomSettings", config: "PyBonesConfig"
) -> None:
    """Score a segmentation of the phantom against its ground truth and log the scores.

    Args:
        segmentation: pyBones segmentation output directory.
        phantom_dir: Phantom directory.
        settings: Phantom settings.
        config: pyBones configuration (file names).
    """
    phantom = load_phantom(phantom_dir, settings)
    prediction = np.asarray(open_volume(labels_path(segmentation, config.io)))
    score = score_phantom(
        prediction, phantom, settings.excluded_value, settings.score.radius_bins_um
    )
    labels = score.labels
    lines = [
        f"accuracy {labels.accuracy:.4f} over {labels.voxels:,} voxels",
        *labels.report_lines("truth"),
        "  vessel recall by radius:",
        *(
            f"    {r.low_um:>4.0f}-{r.high_um:<4.0f} µm  {r.recall:.3f}  ({r.voxels:,} voxels)"
            for r in score.vessel_recall_by_radius
        ),
        f"  vessel recall on clotted segments: {score.clotted_recall:.3f}",
    ]
    logger.info("Phantom scores of %s:\n%s", segmentation, "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The ``python -m tools.phantom`` parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tools.phantom",
        description="Synthetic µCT phantom with exact ground truth.",
    )
    _ = parser.add_argument(
        "--settings", type=Path, default=DEFAULT_SETTINGS_PATH, help="phantom configuration"
    )
    _ = parser.add_argument("--config", type=Path, help="pybones configuration (voxel size)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(Command.GENERATE, help="render the phantom slices and ground truth")
    _ = p.add_argument("-o", "--output", type=Path, required=True, help="output directory")

    p = sub.add_parser(Command.SCORE, help="score a segmentation of the phantom")
    _ = p.add_argument("segmentation", type=Path, help="pybones segmentation output directory")
    _ = p.add_argument("--phantom", type=Path, required=True, help="phantom directory")
    return parser


def main(argv: "Sequence[str] | None" = None) -> None:
    """Parse and validate the arguments, then run the command (and log it to its own file).

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).
    """
    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        invocation = PhantomInvocation.from_namespace(vars(namespace))
    except ValidationError as exc:
        parser.error(f"invalid arguments:\n{exc}")
    settings, config = load_settings(invocation.settings), load_config(invocation.config)
    _ = configure_logging(config.logging, PROGRAM_NAME)
    match invocation.args:
        case GenerateArgs():
            cmd_generate(invocation.args.output, settings, config)
        case ScoreArgs():
            cmd_score(invocation.args.segmentation, invocation.args.phantom, settings, config)
        case _:
            assert_never(invocation.args)


if __name__ == "__main__":
    main()
