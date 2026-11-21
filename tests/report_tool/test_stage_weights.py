"""Tests of the stage stopwatch of the report tool (:mod:`tools.report.stages`)."""

import pytest

from pybones.segmentation.stages import STAGES_2D, Progress
from tests.helpers import FakeClock
from tools.report.stages import StageClock


def report(clock: FakeClock, stopwatch: StageClock, at: float, stage: Progress) -> None:
    """Move the clock, then report progress to the stopwatch.

    Args:
        clock: The fake clock.
        stopwatch: The stopwatch listening.
        at: Time of the report.
        stage: The progress reported.
    """
    clock.now = at
    stopwatch(stage)


def progress(index: int, done: int) -> Progress:
    """Build a progress report of a stage of :data:`STAGES_2D`.

    Args:
        index: Index of the stage in the plan.
        done: Items done.

    Returns:
        The report.
    """
    return Progress(STAGES_2D[index], index + 1, len(STAGES_2D), done, 1, 0.0, 0.0, None)


def test_a_stage_lasts_from_the_previous_last_report_to_its_own() -> None:
    """Work before a stage's first report counts for that stage; shares sum to 1."""
    clock = FakeClock()
    stopwatch = StageClock(clock)
    report(clock, stopwatch, 1.0, progress(0, 0))
    report(clock, stopwatch, 2.0, progress(0, 1))  # first stage: 0 to 2 s
    report(clock, stopwatch, 7.0, progress(1, 0))  # silent work: counts for the second stage
    report(clock, stopwatch, 8.0, progress(1, 1))  # second stage: 2 to 8 s
    shares = stopwatch.shares(STAGES_2D)
    assert shares[STAGES_2D[0]] == pytest.approx(0.25)
    assert shares[STAGES_2D[1]] == pytest.approx(0.75)
