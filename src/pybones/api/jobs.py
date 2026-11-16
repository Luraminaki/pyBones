r"""Background jobs: run long operations in worker threads, report their progress, cancel them.

A front end (web server, GUI) submits a job and gets an id, then follows it with
:meth:`JobManager.events` (pushed :class:`JobStatus` snapshots, made for Server-Sent
Events: no polling), cancels it with :meth:`JobManager.cancel`, and collects the
result with :meth:`JobManager.result`::

    manager = JobManager(config.jobs)
    job_id = manager.submit_segmentation(Path("data_set.zip"), Path("output/run"), config)
    for status in manager.events(job_id):      # an SSE endpoint yields these as they come
        send(f"data: {status.model_dump_json()}\n\n")
    manager.cancel(job_id)                     # stops at the next slice, leaves no partial files
    meta = manager.result(job_id)              # blocks until done; raises if it failed

Events: every state change is sent; progress at most every ``jobs.event_interval_s``
seconds. Each subscriber has a bounded queue (``jobs.event_queue_size``): a client
too slow to keep up loses its oldest snapshots, never the newest (each one is a
complete status). :meth:`JobManager.status` remains for one-off queries.

At most ``jobs.max_concurrent`` jobs run at once; the others wait in order. Only
the ``jobs.history_size`` most recent finished jobs are remembered.
"""

import contextlib
import itertools
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

from pybones.api.operations import segment, write_mesh_file
from pybones.segmentation.pipeline import planned_stages
from pybones.segmentation.stages import CancelledError, ProgressTracker, Stage

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from concurrent.futures import Future
    from pathlib import Path

    from pybones.api.operations import MeshFile
    from pybones.config.schema import JobsConfig, PyBonesConfig
    from pybones.segmentation.labels import Label
    from pybones.segmentation.results import SegmentationMeta
    from pybones.segmentation.stages import Progress

#: Name prefix of the job threads, as shown in the logs.
JOB_THREAD_PREFIX = "pybones-job"
SEGMENTATION_JOB = "segmentation"
MESH_JOB = "mesh"
#: Stages of a mesh job (a single one: its weight is irrelevant).
MESH_PLAN = (Stage.MESH,)
MESH_WEIGHTS = {Stage.MESH: 1.0}

logger = logging.getLogger(__name__)


class JobState(StrEnum):
    """Life cycle of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: States after which a job does not change any more.
FINISHED_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class JobProgress(BaseModel):
    """Progress of a job (see :class:`pybones.segmentation.stages.Progress`).

    Attributes:
        stage: Current stage (``None`` before the first update).
        stage_index: 1-based position of the stage in the plan.
        stage_count: Number of planned stages.
        done: Items processed in the current stage.
        total: Items in the current stage.
        fraction: Overall progress in ``[0, 1]``.
        elapsed_s: Seconds since the job started.
        remaining_s: Estimated seconds left, if known.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    stage: Stage | None
    stage_index: int
    stage_count: int
    done: int
    total: int
    fraction: float
    elapsed_s: float
    remaining_s: float | None


class JobStatus(BaseModel):
    """What a front end shows about a job (JSON-ready: ``status.model_dump_json()``).

    Attributes:
        job_id: Identifier returned by :meth:`JobManager.submit`.
        kind: What the job does, e.g. ``segmentation``.
        state: Where it is in its life cycle.
        progress: Its progress (``None`` while queued).
        error: Why it failed, if it did.
        submitted_at: Submission time (seconds since the epoch).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    job_id: str
    kind: str
    state: JobState
    progress: JobProgress | None
    error: str | None
    submitted_at: float


class _Job:
    """Mutable record of one job, guarded by the manager's lock.

    Attributes:
        job_id: Identifier.
        kind: What the job does.
        tracker: Its progress tracker (also its cancellation switch).
        future: Its result, once submitted to the pool.
        state: Current state.
        error: Failure message, if any.
        submitted_at: Submission time (seconds since the epoch).
        subscribers: Event queues of the clients following the job.
        last_event: Monotonic time of the last progress event sent.
    """

    def __init__(self, job_id: str, kind: str, tracker: ProgressTracker) -> None:
        """Create a queued job.

        Args:
            job_id: Identifier.
            kind: What the job does.
            tracker: Progress tracker the job reports to.
        """
        self.job_id: str = job_id
        self.kind: str = kind
        self.tracker: ProgressTracker = tracker
        self.future: Future[object] | None = None
        self.state: JobState = JobState.QUEUED
        self.error: str | None = None
        self.submitted_at: float = time.time()
        self.subscribers: list[queue.Queue[JobStatus]] = []
        self.last_event: float = 0.0


def put_latest(events: "queue.Queue[JobStatus]", status: JobStatus) -> None:
    """Queue a status, dropping the oldest one if the subscriber is too slow.

    Args:
        events: A subscriber's queue.
        status: Status to send.
    """
    while True:
        try:
            events.put_nowait(status)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                _ = events.get_nowait()
        else:
            return


class JobManager:
    """Runs jobs in a bounded thread pool and keeps their status."""

    def __init__(self, config: "JobsConfig") -> None:
        """Start an idle manager.

        Args:
            config: Concurrency and history settings.
        """
        self._config: JobsConfig = config
        self._pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=config.max_concurrent, thread_name_prefix=JOB_THREAD_PREFIX
        )
        self._lock: threading.Lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._ids: itertools.count[int] = itertools.count(1)

    def submit(
        self,
        kind: str,
        work: "Callable[[ProgressTracker], object]",
        plan: tuple[Stage, ...],
        weights: "Mapping[Stage, float]",
    ) -> str:
        """Queue a job.

        Args:
            kind: What the job does (shown in its status).
            work: The job: receives its progress tracker (to report progress and be
                cancelled) and returns its result.
            plan: Stages the job will report.
            weights: Relative duration of each stage.

        Returns:
            The job id.
        """
        tracker = ProgressTracker(plan, weights)
        with self._lock:
            job = _Job(f"{kind}-{next(self._ids)}", kind, tracker)
            self._jobs[job.job_id] = job
            self._forget_old_jobs()

        def on_progress(_progress: "Progress") -> None:
            """Publish the job's progress (throttled).

            Args:
                _progress: New snapshot (the status is rebuilt from the job).
            """
            self._publish(job, state_changed=False)

        tracker.add_listener(on_progress)
        job.future = self._pool.submit(self._run, job, work)
        logger.info("Queued job %s", job.job_id)
        return job.job_id

    def submit_segmentation(self, source: "Path", out_dir: "Path", config: "PyBonesConfig") -> str:
        """Queue the segmentation of a stack.

        Args:
            source: Directory or zip archive of slices.
            out_dir: Output directory.
            config: Configuration.

        Returns:
            The job id; its result is the :class:`~pybones.segmentation.results.SegmentationMeta`.
        """

        def work(tracker: ProgressTracker) -> "SegmentationMeta":
            """Segment, reporting to the job's tracker.

            Args:
                tracker: Progress tracker of the job.

            Returns:
                The segmentation metadata.
            """
            return segment(source, out_dir, config, tracker)

        return self.submit(
            SEGMENTATION_JOB, work, planned_stages(config), config.pipeline.stage_weights
        )

    def submit_mesh(
        self, segmentation: "Path", label: "Label", path: "Path", config: "PyBonesConfig"
    ) -> str:
        """Queue the meshing of one label into a binary STL file.

        Args:
            segmentation: Segmentation output directory.
            label: Foreground label to mesh.
            path: Output STL file (appears only once complete).
            config: Configuration.

        Returns:
            The job id; its result is the :class:`~pybones.api.operations.MeshFile`.
        """

        def work(tracker: ProgressTracker) -> "MeshFile":
            """Mesh, reporting to the job's tracker.

            Args:
                tracker: Progress tracker of the job.

            Returns:
                The written mesh file.
            """
            return write_mesh_file(segmentation, label, path, config, tracker)

        return self.submit(MESH_JOB, work, MESH_PLAN, MESH_WEIGHTS)

    def _run(self, job: _Job, work: "Callable[[ProgressTracker], object]") -> object:
        """Run a job in a worker thread and record how it ended.

        Args:
            job: The job.
            work: Its work.

        Returns:
            The work's result.

        Raises:
            CancelledError: If the job was cancelled (also recorded in its state).
            Exception: Whatever the work raised (also recorded in its state).
        """
        with self._lock:
            cancelled = job.tracker.cancelled
            job.state = JobState.CANCELLED if cancelled else JobState.RUNNING
        self._publish(job, state_changed=True)
        if cancelled:
            raise CancelledError(f"{job.job_id} cancelled before it started")
        logger.info("Started job %s", job.job_id)
        try:
            result = work(job.tracker)
        except CancelledError:
            self._finish(job, JobState.CANCELLED, None)
            raise
        except Exception as exc:
            logger.exception("Job %s failed", job.job_id)
            self._finish(job, JobState.FAILED, f"{type(exc).__name__}: {exc}")
            raise
        self._finish(job, JobState.SUCCEEDED, None)
        return result

    def _finish(self, job: _Job, state: JobState, error: str | None) -> None:
        """Record the end of a job.

        Args:
            job: The job.
            state: Final state.
            error: Failure message, if any.
        """
        with self._lock:
            job.state, job.error = state, error
        self._publish(job, state_changed=True)
        logger.info("Job %s %s", job.job_id, state)

    def _publish(self, job: _Job, state_changed: bool) -> None:
        """Push the job's status to its subscribers.

        Args:
            job: The job.
            state_changed: State changes are always sent; progress updates are throttled.
        """
        now = time.monotonic()
        with self._lock:
            if not state_changed and now - job.last_event < self._config.event_interval_s:
                return
            job.last_event = now
            subscribers = tuple(job.subscribers)
        if not subscribers:
            return
        status = self._describe(job)
        for events in subscribers:
            put_latest(events, status)

    def events(
        self, job_id: str, heartbeat_s: float | None = None
    ) -> "Generator[JobStatus, None, None]":
        """Follow a job: its current status, then every change, until it finishes.

        Made for Server-Sent Events: the caller sends each status as it comes. The
        iterator blocks between events; closing it (client gone) unsubscribes.

        Args:
            job_id: Job id.
            heartbeat_s: If set, the current status is repeated whenever nothing was
                sent for this long, so a server notices a departed client even while
                the job is quiet (``None``: wait for changes only).

        Yields:
            Status snapshots; the last one has a finished state.
        """
        job = self._job(job_id)
        events: queue.Queue[JobStatus] = queue.Queue(maxsize=self._config.event_queue_size)
        with self._lock:
            job.subscribers.append(events)
        try:
            status = self._describe(job)
            yield status
            while status.state not in FINISHED_STATES:
                try:
                    status = events.get(timeout=heartbeat_s)
                except queue.Empty:
                    status = self._describe(job)
                yield status
        finally:
            with self._lock:
                job.subscribers.remove(events)

    def _forget_old_jobs(self) -> None:
        """Drop the oldest finished jobs beyond the history size (caller holds the lock)."""
        finished = [j for j in self._jobs.values() if j.state in FINISHED_STATES]
        for job in finished[: max(0, len(finished) - self._config.history_size)]:
            del self._jobs[job.job_id]

    def _job(self, job_id: str) -> _Job:
        """Find a job.

        Args:
            job_id: Job id.

        Returns:
            The job record.

        Raises:
            KeyError: If no job has this id (or it was forgotten).
        """
        with self._lock:
            return self._jobs[job_id]

    def status(self, job_id: str) -> JobStatus:
        """Describe a job now (safe to call from any thread).

        Args:
            job_id: Job id.

        Returns:
            Its status.
        """
        return self._describe(self._job(job_id))

    def _describe(self, job: _Job) -> JobStatus:
        """Build the status of a job.

        Args:
            job: The job.

        Returns:
            Its status now.
        """
        with self._lock:
            state, error = job.state, job.error
        progress = None
        if state != JobState.QUEUED:
            p = job.tracker.snapshot()
            progress = JobProgress(
                stage=p.stage,
                stage_index=p.stage_index,
                stage_count=p.stage_count,
                done=p.done,
                total=p.total,
                fraction=1.0 if state == JobState.SUCCEEDED else p.fraction,
                elapsed_s=p.elapsed_s,
                remaining_s=p.remaining_s,
            )
        return JobStatus(
            job_id=job.job_id,
            kind=job.kind,
            state=state,
            progress=progress,
            error=error,
            submitted_at=job.submitted_at,
        )

    def statuses(self) -> list[JobStatus]:
        """Describe every remembered job, oldest first.

        Returns:
            Their statuses.
        """
        with self._lock:
            ids = list(self._jobs)
        return [self.status(job_id) for job_id in ids]

    def cancel(self, job_id: str) -> None:
        """Ask a job to stop: a queued job never starts, a running one stops at its next update.

        Args:
            job_id: Job id.
        """
        job = self._job(job_id)
        job.tracker.cancel()
        logger.info("Cancelling job %s", job_id)

    def result(self, job_id: str, timeout: float | None = None) -> object:
        """Wait for a job and return its result.

        Args:
            job_id: Job id.
            timeout: Seconds to wait at most (``None``: no limit).

        Returns:
            The job's result.

        Raises:
            CancelledError: If the job was cancelled.
            TimeoutError: If it did not finish in time.
            RuntimeError: If the job was never handed to the pool.
            Exception: Whatever the job raised, if it failed.
        """
        future = self._job(job_id).future
        if future is None:
            raise RuntimeError(f"{job_id} was not submitted")
        return future.result(timeout)

    def shutdown(self) -> None:
        """Cancel every job and wait for the running ones to stop."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.tracker.cancel()
        self._pool.shutdown(wait=True)
