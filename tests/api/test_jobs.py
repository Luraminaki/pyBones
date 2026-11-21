"""Tests of the background jobs (:mod:`pybones.api.jobs`)."""

import json
import queue
import threading
import time
from typing import TYPE_CHECKING

import pytest

from pybones.api.job_status import MESH_JOB, JobState, UnknownJobError
from pybones.api.jobs import JobManager, put_latest
from pybones.api.operations import MeshFile
from pybones.config.loader import load_config
from pybones.segmentation.labels import Label
from pybones.segmentation.pipeline import run_segmentation
from pybones.segmentation.results import SegmentationMeta
from pybones.segmentation.stages import STAGES_2D, CancelledError, Stage
from tests.helpers import small_config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from pybones.api.job_status import JobStatus
    from pybones.segmentation.stages import ProgressTracker

WEIGHTS = dict.fromkeys(Stage, 1.0)
WAIT_S = 30.0
FAILURE = "boom"
#: Wait long enough for a running clock to move.
SETTLE_S = 0.05
#: How long the second job waits in the queue.
QUEUE_WAIT_S = 0.3


@pytest.fixture
def manager() -> "Iterator[JobManager]":
    """Run a job manager with one worker and a short history for the duration of a test.

    Yields:
        The manager (shut down afterwards).
    """
    config = load_config(overrides=["jobs.max_concurrent=1", "jobs.history_size=2"]).jobs
    jobs = JobManager(config)
    yield jobs
    jobs.shutdown()


def blocked_job(
    release: threading.Event, started: threading.Event
) -> "Callable[[ProgressTracker], str]":
    """Build a job that reports progress, then waits until released.

    Args:
        release: Set to let the job finish.
        started: Set by the job once it runs.

    Returns:
        The job's work function.
    """

    def work(tracker: "ProgressTracker") -> str:
        """Report one update, wait, report again (the second update honours cancellation).

        Args:
            tracker: Progress tracker of the job.

        Returns:
            ``"done"``.
        """
        tracker(Stage.THRESHOLDS, 1, 2)
        started.set()
        _ = release.wait(WAIT_S)
        tracker(Stage.THRESHOLDS, 2, 2)
        return "done"

    return work


def test_segmentation_job_runs_to_completion(
    manager: JobManager, slice_dir: "Path", tmp_path: "Path"
) -> None:
    """A segmentation job succeeds, reports full progress and returns the metadata."""
    job_id = manager.submit_segmentation(slice_dir, tmp_path / "out", small_config())
    meta = manager.result(job_id, WAIT_S)
    status = manager.status(job_id)
    assert status.state == JobState.SUCCEEDED
    assert status.progress is not None
    assert status.progress.fraction == 1.0
    assert status.error is None
    assert isinstance(meta, SegmentationMeta)
    assert meta.shape_zyx[0] == len(list(slice_dir.iterdir()))
    assert json.loads(status.model_dump_json())["state"] == "succeeded"


def test_queued_job_can_be_cancelled_before_it_starts(manager: JobManager) -> None:
    """With one worker busy, a queued job is cancelled without ever running."""
    release, started = threading.Event(), threading.Event()
    first = manager.submit("test", blocked_job(release, started), STAGES_2D, WEIGHTS)
    assert started.wait(WAIT_S)
    second = manager.submit(
        "test", blocked_job(threading.Event(), threading.Event()), STAGES_2D, WEIGHTS
    )
    assert manager.status(second).state == JobState.QUEUED
    assert manager.status(second).progress is None
    manager.cancel(second)
    release.set()
    assert manager.result(first, WAIT_S) == "done"
    with pytest.raises(CancelledError):
        _ = manager.result(second, WAIT_S)
    status = manager.status(second)
    assert status.state == JobState.CANCELLED
    assert status.progress is not None
    elapsed = status.progress.elapsed_s
    time.sleep(SETTLE_S)
    later = manager.status(second).progress
    assert later is not None
    assert later.elapsed_s == elapsed  # frozen like any finished job, never growing


def test_a_queued_job_counts_its_time_from_its_start(manager: JobManager) -> None:
    """The wait for a worker is not part of a job's elapsed time (nor of its time left)."""
    release, started = threading.Event(), threading.Event()
    first = manager.submit("test", blocked_job(release, started), STAGES_2D, WEIGHTS)
    assert started.wait(WAIT_S)
    second_release, second_started = threading.Event(), threading.Event()
    second = manager.submit("test", blocked_job(second_release, second_started), STAGES_2D, WEIGHTS)
    submitted = time.monotonic()
    time.sleep(QUEUE_WAIT_S)
    release.set()
    assert second_started.wait(WAIT_S)
    progress = manager.status(second).progress
    second_release.set()
    assert progress is not None
    # Since its submission more than QUEUE_WAIT_S passed; since its start, far less.
    assert progress.elapsed_s < time.monotonic() - submitted - QUEUE_WAIT_S / 2
    assert manager.result(first, WAIT_S) == "done"
    assert manager.result(second, WAIT_S) == "done"


def test_running_job_stops_at_its_next_update(manager: JobManager) -> None:
    """A running job is cancelled at its next progress update."""
    release, started = threading.Event(), threading.Event()
    job_id = manager.submit("test", blocked_job(release, started), STAGES_2D, WEIGHTS)
    assert started.wait(WAIT_S)
    status = manager.status(job_id)
    assert status.state == JobState.RUNNING
    assert status.progress is not None
    assert status.progress.stage == Stage.THRESHOLDS
    manager.cancel(job_id)
    release.set()
    with pytest.raises(CancelledError):
        _ = manager.result(job_id, WAIT_S)
    assert manager.status(job_id).state == JobState.CANCELLED


def test_failed_job_reports_its_error(manager: JobManager) -> None:
    """An exception in a job is recorded in its status and re-raised by ``result``."""

    def fail(_tracker: "ProgressTracker") -> None:
        """Fail immediately.

        Args:
            _tracker: Progress tracker (unused).

        Raises:
            ValueError: Always.
        """
        raise ValueError(FAILURE)

    job_id = manager.submit("test", fail, STAGES_2D, WEIGHTS)
    with pytest.raises(ValueError, match=FAILURE):
        _ = manager.result(job_id, WAIT_S)
    status = manager.status(job_id)
    assert status.state == JobState.FAILED
    assert status.error == f"ValueError: {FAILURE}"


def test_only_recent_finished_jobs_are_remembered(manager: JobManager) -> None:
    """Finished jobs beyond the history size are forgotten, oldest first."""

    def quick(_tracker: "ProgressTracker") -> int:
        """Finish at once.

        Args:
            _tracker: Progress tracker (unused).

        Returns:
            0.
        """
        return 0

    ids = []
    for _ in range(4):
        ids.append(manager.submit("test", quick, STAGES_2D, WEIGHTS))
        _ = manager.result(ids[-1], WAIT_S)
    remembered = [s.job_id for s in manager.statuses()]
    assert ids[0] not in remembered
    assert ids[-1] in remembered
    with pytest.raises(UnknownJobError):
        _ = manager.status(ids[0])


def test_events_push_every_state_change_in_order(
    manager: JobManager, slice_dir: "Path", tmp_path: "Path"
) -> None:
    """Following a job yields its states in order up to the final one; progress never drops."""
    job_id = manager.submit_segmentation(slice_dir, tmp_path / "out", small_config())
    statuses = list(manager.events(job_id))
    states = [s.state for s in statuses]
    order = [JobState.QUEUED, JobState.RUNNING, JobState.SUCCEEDED]
    assert states[-1] == JobState.SUCCEEDED
    assert [order.index(s) for s in states] == sorted(order.index(s) for s in states)
    fractions = [s.progress.fraction for s in statuses if s.progress is not None]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_progress_events_are_throttled(slice_dir: "Path", tmp_path: "Path") -> None:
    """With a long interval, progress events are skipped; state changes still arrive."""
    config = load_config(overrides=["jobs.event_interval_s=60"]).jobs
    jobs = JobManager(config)
    try:
        job_id = jobs.submit_segmentation(slice_dir, tmp_path / "out", small_config())
        statuses = list(jobs.events(job_id))
    finally:
        jobs.shutdown()
    running = [s for s in statuses if s.state == JobState.RUNNING]
    assert len(running) <= 2  # the start, and at most one progress event in the first minute
    assert statuses[-1].state == JobState.SUCCEEDED


def test_events_of_a_cancelled_job_end_with_cancelled(manager: JobManager) -> None:
    """A follower learns that the job was cancelled, then the stream ends."""
    release, started = threading.Event(), threading.Event()
    job_id = manager.submit("test", blocked_job(release, started), STAGES_2D, WEIGHTS)
    assert started.wait(WAIT_S)
    stream = manager.events(job_id)
    assert next(stream).state == JobState.RUNNING
    manager.cancel(job_id)
    release.set()
    assert [s.state for s in stream][-1] == JobState.CANCELLED


def test_slow_subscribers_lose_the_oldest_snapshots(manager: JobManager) -> None:
    """A full queue drops its oldest status to keep the newest."""
    release, started = threading.Event(), threading.Event()
    job_id = manager.submit("test", blocked_job(release, started), STAGES_2D, WEIGHTS)
    assert started.wait(WAIT_S)
    statuses = [manager.status(job_id).model_copy(update={"error": str(i)}) for i in range(3)]
    events: queue.Queue[JobStatus] = queue.Queue(maxsize=2)
    for status in statuses:
        put_latest(events, status)
    assert [events.get_nowait().error for _ in range(2)] == ["1", "2"]
    release.set()


def test_mesh_job_writes_the_stl_file(
    manager: JobManager, slice_dir: "Path", tmp_path: "Path"
) -> None:
    """A mesh job reports the mesh stage, succeeds and returns the written file."""
    config = small_config()
    seg = tmp_path / "seg"
    _ = run_segmentation(slice_dir, seg, config)
    path = tmp_path / "bone.stl"
    job_id = manager.submit_mesh(seg, Label.BONE, path, config)
    result = manager.result(job_id, WAIT_S)
    status = manager.status(job_id)
    assert isinstance(result, MeshFile)
    assert (result.path, path.is_file(), result.n_faces > 0) == (path, True, True)
    assert (status.kind, status.state) == (MESH_JOB, JobState.SUCCEEDED)
    assert status.progress is not None
    assert (status.progress.stage, status.progress.fraction) == (Stage.WRITING, 1.0)


def test_events_repeat_the_status_when_idle(manager: JobManager) -> None:
    """With a heartbeat, a quiet job's status is repeated instead of blocking forever."""
    release, started = threading.Event(), threading.Event()
    job_id = manager.submit("test", blocked_job(release, started), STAGES_2D, WEIGHTS)
    assert started.wait(WAIT_S)
    stream = manager.events(job_id, heartbeat_s=0.01)
    first, repeated = next(stream), next(stream)
    assert first.state == repeated.state == JobState.RUNNING
    release.set()
    assert [s.state for s in stream][-1] == JobState.SUCCEEDED


def test_a_finished_job_keeps_its_duration(manager: JobManager) -> None:
    """The elapsed time of a finished job stops at its end instead of following the clock."""

    def quick(tracker: "ProgressTracker") -> int:
        """Report one update and finish.

        Args:
            tracker: Progress tracker of the job.

        Returns:
            0.
        """
        tracker(Stage.THRESHOLDS, 1, 1)
        return 0

    job_id = manager.submit("test", quick, STAGES_2D, WEIGHTS)
    _ = manager.result(job_id, WAIT_S)
    first = manager.status(job_id).progress
    time.sleep(SETTLE_S)
    later = manager.status(job_id).progress
    assert first is not None
    assert later is not None
    assert later.elapsed_s == first.elapsed_s
    assert later.remaining_s == 0.0
