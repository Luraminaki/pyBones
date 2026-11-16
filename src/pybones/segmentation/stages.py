"""Pipeline stages, progress reporting and cancellation.

The pipeline reports progress through a plain callback, ``(stage, done, total)``,
after each processed slice. :class:`ProgressTracker` is such a callback that also
keeps a thread-safe snapshot of the whole run (:class:`Progress`: current stage,
stage index, overall fraction, elapsed and remaining time), notifies listeners,
and can cancel the run. A front end (CLI line, web page) polls
:meth:`ProgressTracker.snapshot` or listens to it, while the pipeline runs in
another thread.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class Stage(StrEnum):
    """Stages of the jobs: the segmentation pipeline's, in execution order, then meshing."""

    THRESHOLDS = "thresholds"
    CLASSIFY = "classify"
    ANALYSE = "analyse"
    RESOLVE = "resolve"
    CLEAN = "clean"
    #: Surface extraction of one label (a job of its own, after a segmentation).
    MESH = "mesh"


#: Stages of a run without / with the 3D refinement.
STAGES_2D = (Stage.THRESHOLDS, Stage.CLASSIFY)
STAGES_3D = (*STAGES_2D, Stage.ANALYSE, Stage.RESOLVE, Stage.CLEAN)

#: Called after each processed item with ``(stage, done, total)``.
ProgressCallback = Callable[[Stage, int, int], None]


class CancelledError(Exception):
    """Raised inside the pipeline when its run is cancelled (see :meth:`ProgressTracker.cancel`)."""


def no_progress(_stage: Stage, _done: int, _total: int) -> None:
    """Ignore a progress update (the default :data:`ProgressCallback`).

    Args:
        _stage: Pipeline stage (unused).
        _done: Items processed so far (unused).
        _total: Items in the stage (unused).
    """


@dataclass(frozen=True, slots=True)
class Progress:
    """State of a run at one moment.

    Attributes:
        stage: Current stage (``None`` before the first update).
        stage_index: 1-based position of the stage in the plan (0 before the first update).
        stage_count: Number of planned stages.
        done: Items processed in the current stage.
        total: Items in the current stage.
        fraction: Overall progress in ``[0, 1]``, weighted by the stage durations.
        elapsed_s: Seconds since the tracker was created.
        remaining_s: Estimated seconds left (``None`` until there is something to extrapolate).
    """

    stage: Stage | None
    stage_index: int
    stage_count: int
    done: int
    total: int
    fraction: float
    elapsed_s: float
    remaining_s: float | None

    @property
    def stage_fraction(self) -> float:
        """Progress within the current stage, in ``[0, 1]``."""
        return self.done / self.total if self.total else 0.0


#: Called with each new snapshot (from the thread that runs the pipeline).
ProgressListener = Callable[[Progress], None]


class ProgressTracker:
    """Thread-safe progress of a run, usable as its :data:`ProgressCallback`.

    Example:
        >>> tracker = ProgressTracker(STAGES_3D, config.pipeline.stage_weights)
        >>> worker = threading.Thread(target=run_segmentation, args=(src, out, config, tracker))
        >>> worker.start()
        >>> tracker.snapshot().fraction   # poll from the front end
        >>> tracker.cancel()              # the run raises CancelledError at its next update

    Attributes:
        plan: Stages expected, in order.
    """

    def __init__(
        self,
        plan: tuple[Stage, ...],
        weights: "Mapping[Stage, float]",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start tracking a run.

        Args:
            plan: Stages that will run, in order.
            weights: Relative duration of each stage (normalised over ``plan``).
            clock: Monotonic clock in seconds (injectable for tests).
        """
        self.plan: tuple[Stage, ...] = plan
        total = sum(weights[s] for s in plan)
        self._weights: dict[Stage, float] = {s: weights[s] / total for s in plan}
        self._clock: Callable[[], float] = clock
        self._start: float = clock()
        self._lock: threading.Lock = threading.Lock()
        self._cancelled: threading.Event = threading.Event()
        self._listeners: list[ProgressListener] = []
        self._state: tuple[Stage | None, int, int] = (None, 0, 0)

    def add_listener(self, listener: ProgressListener) -> None:
        """Call ``listener`` with every new snapshot.

        Args:
            listener: Function receiving each :class:`Progress`.
        """
        with self._lock:
            self._listeners.append(listener)

    def cancel(self) -> None:
        """Ask the run to stop: its next progress update raises :class:`CancelledError`."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        """Whether :meth:`cancel` was called."""
        return self._cancelled.is_set()

    def __call__(self, stage: Stage, done: int, total: int) -> None:
        """Record a progress update from the pipeline.

        Args:
            stage: Stage being run.
            done: Items processed in it so far.
            total: Items in the stage.

        Raises:
            CancelledError: If the run was cancelled.
        """
        if self._cancelled.is_set():
            raise CancelledError(f"cancelled during {stage}")
        with self._lock:
            self._state = (stage, done, total)
            listeners = tuple(self._listeners)
        if listeners:
            snapshot = self.snapshot()
            for listener in listeners:
                listener(snapshot)

    def snapshot(self) -> Progress:
        """Describe the run now (safe to call from any thread).

        Returns:
            The current progress.
        """
        with self._lock:
            stage, done, total = self._state
        elapsed = self._clock() - self._start
        if stage is None:
            return Progress(None, 0, len(self.plan), 0, 0, 0.0, elapsed, None)
        index = self.plan.index(stage)
        finished = sum(self._weights[s] for s in self.plan[:index])
        within = done / total if total else 0.0
        fraction = min(1.0, finished + self._weights[stage] * within)
        remaining = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
        return Progress(stage, index + 1, len(self.plan), done, total, fraction, elapsed, remaining)
