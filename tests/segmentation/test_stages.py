"""Tests of progress tracking and cancellation (:mod:`pybones.segmentation.stages`)."""

from typing import TYPE_CHECKING

import pytest

from pybones.cli.commands import progress_line
from pybones.segmentation.pipeline import planned_stages, run_segmentation
from pybones.segmentation.stages import (
    STAGES_2D,
    STAGES_3D,
    CancelledError,
    ProgressTracker,
    Stage,
)
from tests.helpers import small_config

if TYPE_CHECKING:
    from pathlib import Path

    from pybones.segmentation.stages import (
        Progress,
    )

#: Equal weights make the expected fractions easy to read.
EQUAL = dict.fromkeys(Stage, 1.0)


class FakeClock:
    """A clock that only moves when told to.

    Attributes:
        now: Current time, in seconds.
    """

    def __init__(self) -> None:
        """Start at time 0."""
        self.now: float = 0.0

    def __call__(self) -> float:
        """Read the clock.

        Returns:
            The current time.
        """
        return self.now


def test_snapshot_before_any_update() -> None:
    """Before the first update, nothing is running and no estimate exists."""
    snapshot = ProgressTracker(STAGES_3D, EQUAL).snapshot()
    assert (snapshot.stage, snapshot.stage_index, snapshot.stage_count) == (None, 0, 5)
    assert snapshot.fraction == 0.0
    assert snapshot.remaining_s is None


def test_fraction_is_weighted_over_the_planned_stages() -> None:
    """Finished stages count fully, the current one by its share of items."""
    clock = FakeClock()
    tracker = ProgressTracker(STAGES_2D, {**EQUAL, Stage.CLASSIFY: 3.0}, clock)
    tracker(Stage.THRESHOLDS, 10, 10)
    assert tracker.snapshot().fraction == pytest.approx(0.25)
    clock.now = 30.0
    tracker(Stage.CLASSIFY, 50, 100)
    snapshot = tracker.snapshot()
    assert (snapshot.stage, snapshot.stage_index, snapshot.stage_count) == (Stage.CLASSIFY, 2, 2)
    assert snapshot.stage_fraction == pytest.approx(0.5)
    assert snapshot.fraction == pytest.approx(0.25 + 0.75 * 0.5)
    assert snapshot.remaining_s == pytest.approx(30.0 * (1 - 0.625) / 0.625)


def test_listeners_receive_every_update() -> None:
    """Listeners get one snapshot per update, in order."""
    seen: list[Progress] = []
    tracker = ProgressTracker(STAGES_2D, EQUAL)
    tracker.add_listener(seen.append)
    tracker(Stage.THRESHOLDS, 1, 2)
    tracker(Stage.THRESHOLDS, 2, 2)
    assert [p.done for p in seen] == [1, 2]
    assert "[1/2] thresholds" in progress_line(seen[-1])


def test_cancel_raises_at_the_next_update() -> None:
    """A cancelled tracker stops the run at its next update."""
    tracker = ProgressTracker(STAGES_2D, EQUAL)
    tracker(Stage.THRESHOLDS, 1, 2)
    tracker.cancel()
    assert tracker.cancelled
    with pytest.raises(CancelledError):
        tracker(Stage.THRESHOLDS, 2, 2)


@pytest.mark.parametrize("stage", [Stage.CLASSIFY, Stage.RESOLVE])
def test_cancelled_run_leaves_no_volume_behind(
    slice_dir: "Path", tmp_path: "Path", stage: Stage
) -> None:
    """Cancelling during classification or refinement removes every volume it started."""
    config = small_config()
    tracker = ProgressTracker(planned_stages(config), config.pipeline.stage_weights)

    def cancel_in_stage(progress: "Progress") -> None:
        """Cancel as soon as the chosen stage starts.

        Args:
            progress: Current snapshot.
        """
        if progress.stage == stage:
            tracker.cancel()

    tracker.add_listener(cancel_in_stage)
    out_dir = tmp_path / "out"
    with pytest.raises(CancelledError):
        _ = run_segmentation(slice_dir, out_dir, config, tracker)
    assert list(out_dir.iterdir()) == []
