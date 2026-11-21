"""Progress stage weights checked against the time each stage really takes.

A progress bar and its estimate of the time left are only right if each stage's weight
matches its share of the run. :func:`stage_weights` runs the three long operations on the
sample (segmentation, the vessel mesh, the quantification) with a stopwatch on their progress
reports, and writes each stage's configured weight next to its measured share
(``stage_weights.csv`` and a table for the report). A stage further off than
``stage_weight_tolerance`` is logged with the weight to use instead; the configuration is
never changed here.
"""

import csv
import functools
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pybones.api.operations import write_mesh_file
from pybones.quantification.pipeline import measure_network
from pybones.segmentation.pipeline import planned_stages, run_segmentation
from pybones.segmentation.results import load_labels, load_meta
from pybones.segmentation.stages import MESH_STAGES, QUANTIFY_STAGES, ProgressTracker
from tools.report.meshes import COMPARED_LABEL
from tools.report.runs import REAL_RUN, run_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pybones.segmentation.stages import Progress, Stage
    from tools.report.context import Context

STAGE_WEIGHTS_CSV = "stage_weights.csv"
STAGE_WEIGHTS_TABLE = "stage_weights.tex"
_COLUMNS: tuple[str, ...] = ("operation", "stage", "weight", "measured")
_MESH_FILE = "mesh.glb"
_SEGMENTATION_DIR = "segmentation"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StageClock:
    """Listens to a run's progress and notes when each stage last reported.

    A stage's time runs from the previous stage's last report to its own last one, so it
    includes the work done before its first report.

    Attributes:
        clock: Monotonic clock in seconds (injectable for tests).
        start: When the run started (created with the clock).
        last: Time of the last report of each stage.
    """

    clock: "Callable[[], float]" = time.perf_counter
    start: float = field(init=False)
    last: "dict[Stage, float]" = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Start the stopwatch."""
        self.start = self.clock()

    def __call__(self, progress: "Progress") -> None:
        """Note a progress report.

        Args:
            progress: The report.
        """
        if progress.stage is not None:
            self.last[progress.stage] = self.clock()

    def shares(self, plan: "tuple[Stage, ...]") -> "dict[Stage, float]":
        """Share of the run's time each planned stage took.

        Args:
            plan: The stages, in order (each must have reported).

        Returns:
            Share per stage, summing to 1.
        """
        ends = [self.last[stage] for stage in plan]
        durations = [end - begin for begin, end in zip([self.start, *ends[:-1]], ends, strict=True)]
        total = sum(durations)
        return {stage: seconds / total for stage, seconds in zip(plan, durations, strict=True)}


def _measure(
    operation: str,
    run: "Callable[[ProgressTracker], object]",
    plan: "tuple[Stage, ...]",
    weights: "Mapping[Stage, float]",
) -> list[dict[str, str]]:
    """Run an operation with a stopwatch on its stages.

    Args:
        operation: Its name in the table.
        run: Runs it, reporting to the tracker it is given.
        plan: Its stages, in order.
        weights: Their configured weights (normalised over ``plan`` here, as the tracker does).

    Returns:
        One row per stage: configured weight and measured share.
    """
    tracker = ProgressTracker(plan, weights)
    clock = StageClock()
    tracker.add_listener(clock)
    _ = run(tracker)
    measured = clock.shares(plan)
    total = sum(weights[stage] for stage in plan)
    return [
        {
            "operation": operation,
            "stage": str(stage),
            "weight": f"{weights[stage] / total:.3f}",
            "measured": f"{measured[stage]:.3f}",
        }
        for stage in plan
    ]


def _table(rows: list[dict[str, str]], path: Path) -> None:
    """Write the comparison as a LaTeX table.

    Args:
        rows: Measured rows.
        path: Output ``.tex`` file.
    """
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Operation & Stage & Weight & Measured share \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        operation = row["operation"] if row["operation"] != previous else ""
        previous = row["operation"]
        stage = row["stage"].replace("_", r"\_")
        lines.append(f"{operation} & {stage} & {row['weight']} & {row['measured']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _ = path.write_text("\n".join(lines) + "\n")


def stage_weights(ctx: "Context") -> None:
    """Measure each stage's share of the three long operations and compare it with its weight.

    Args:
        ctx: Figure context.
    """
    config = ctx.config
    real = run_dir(ctx.settings, ctx.repo, ctx.settings.figures.reference_key, REAL_RUN)
    voxel = load_meta(real, config.io).config.acquisition.voxel_size_um
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rows = [
            *_measure(
                "segmentation",
                functools.partial(
                    run_segmentation,
                    ctx.repo / ctx.settings.dataset,
                    work / _SEGMENTATION_DIR,
                    config,
                ),
                planned_stages(config),
                config.pipeline.stage_weights,
            ),
            *_measure(
                "mesh",
                functools.partial(write_mesh_file, real, COMPARED_LABEL, work / _MESH_FILE, config),
                MESH_STAGES,
                config.mesh.stage_weights,
            ),
            *_measure(
                "quantification",
                functools.partial(
                    measure_network, load_labels(real, config.io), voxel, config.quantification
                ),
                QUANTIFY_STAGES,
                config.quantification.stage_weights,
            ),
        ]
    with (ctx.data / STAGE_WEIGHTS_CSV).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _table(rows, ctx.data / STAGE_WEIGHTS_TABLE)
    tolerance = ctx.settings.stage_weight_tolerance
    for row in rows:
        if abs(float(row["weight"]) - float(row["measured"])) > tolerance:
            logger.warning(
                "%s stage %s weighs %s but takes %s of the time: set its weight to %s "
                + "(weights are relative: the measured shares can be used as they are)",
                row["operation"],
                row["stage"],
                row["weight"],
                row["measured"],
                row["measured"],
            )
