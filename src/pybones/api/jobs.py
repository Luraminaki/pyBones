r"""Background jobs: run long operations in worker threads, report their progress, cancel them.

A front end (web server, GUI) submits a job and gets an id, then follows it with
:meth:`JobManager.events` (pushed :class:`JobStatus` snapshots, made for Server-Sent
Events: no polling), cancels it with :meth:`JobManager.cancel`, and collects the
result with :meth:`JobManager.result`::

    manager = JobManager(config.jobs)
    job_id = manager.submit_segmentation(Path("data/data_set.zip"), Path("output/run"), config)
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
import dataclasses
import functools
import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from pybones.api.job_status import (
    FINISHED_STATES,
    MESH_JOB,
    QUANTIFICATION_JOB,
    SEGMENTATION_JOB,
    JobState,
    JobStatus,
    UnknownJobError,
)
from pybones.api.operations import write_mesh_file
from pybones.quantification.pipeline import run_quantification
from pybones.segmentation.pipeline import planned_stages, run_segmentation
from pybones.segmentation.stages import (
    MESH_STAGES,
    QUANTIFY_STAGES,
    CancelledError,
    ProgressTracker,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from concurrent.futures import Future
    from pathlib import Path

    from pybones.config.schema import JobsConfig, PyBonesConfig
    from pybones.segmentation.labels import Label
    from pybones.segmentation.stages import Progress, Stage

#: Name prefix of the job threads, as shown in the logs.
JOB_THREAD_PREFIX = "pybones-job"
logger = logging.getLogger(__name__)


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
        subscribers: Listeners of the clients following the job (each gets every status sent).
        last_event: Monotonic time of the last progress event sent.
        last_stage: Stage of the last status sent (``None`` before any).
        final_elapsed_s: How long the job ran, frozen when it finished (``None`` before).
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
        self.subscribers: list[StatusListener] = []
        self.last_event: float = 0.0
        self.last_stage: Stage | None = None
        self.final_elapsed_s: float | None = None


#: Receives the status of a followed job; called from the job's thread, so it must not block.
StatusListener = Callable[[JobStatus], None]


def put_latest(events: "queue.Queue[JobStatus]", status: JobStatus) -> None:
    """Queue a status, dropping the oldest one if the subscriber is too slow.

    Only the job's thread puts statuses, so once room is made the put cannot fail; the
    follower's thread takes them, and may empty the queue between the check and the drop.

    Args:
        events: A subscriber's queue.
        status: Status to send.
    """
    if events.full():
        with contextlib.suppress(queue.Empty):  # the follower took it first: room made anyway
            _ = events.get_nowait()
    events.put_nowait(status)


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
        plan: "tuple[Stage, ...]",
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

        def on_progress(progress: "Progress") -> None:
            """Publish the job's progress (throttled within a stage).

            Args:
                progress: New snapshot (its stage; the status is rebuilt from the job).
            """
            self._publish(job, state_changed=False, stage=progress.stage)

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
        return self.submit(
            SEGMENTATION_JOB,
            functools.partial(run_segmentation, source, out_dir, config),
            planned_stages(config),
            config.pipeline.stage_weights,
        )

    def submit_mesh(
        self, segmentation: "Path", label: "Label", path: "Path", config: "PyBonesConfig"
    ) -> str:
        """Queue the meshing of one label into a mesh file (format from its extension).

        Args:
            segmentation: Segmentation output directory.
            label: Foreground label to mesh.
            path: Output ``.stl`` or ``.glb`` file (appears only once complete).
            config: Configuration.

        Returns:
            The job id; its result is the :class:`~pybones.api.operations.MeshFile`.
        """
        return self.submit(
            MESH_JOB,
            functools.partial(write_mesh_file, segmentation, label, path, config),
            MESH_STAGES,
            config.mesh.stage_weights,
        )

    def submit_quantification(self, segmentation: "Path", config: "PyBonesConfig") -> str:
        """Queue the quantification of a segmentation's vessel network.

        Args:
            segmentation: Segmentation output directory (the files are written into it).
            config: Configuration (quantification settings).

        Returns:
            The job id; its result is the
            :class:`~pybones.quantification.measures.NetworkSummary`.
        """
        return self.submit(
            QUANTIFICATION_JOB,
            functools.partial(run_quantification, segmentation, config),
            QUANTIFY_STAGES,
            config.quantification.stage_weights,
        )

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
        # Time counts from here: a job waiting for a worker is not slow, and its estimate of
        # the time left must not include the wait.
        job.tracker.restart_clock()
        if job.tracker.cancelled:
            self._finish(job, JobState.CANCELLED, None)
            raise CancelledError(f"{job.job_id} cancelled before it started")
        with self._lock:
            job.state = JobState.RUNNING
        self._publish(job, state_changed=True)
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
            job.final_elapsed_s = job.tracker.snapshot().elapsed_s
        self._publish(job, state_changed=True)
        logger.info("Job %s %s", job.job_id, state)

    def _publish(self, job: _Job, state_changed: bool, stage: "Stage | None" = None) -> None:
        """Push the job's status to its subscribers.

        A new state or a new stage is always sent: a stage starting right after another
        would otherwise be held back until its first progress, which a long single-step
        stage (a skeleton, a surface) never sends. Progress within a stage is throttled.

        Args:
            job: The job.
            state_changed: Whether the job's state changed.
            stage: Stage of the progress being published (``None`` for a state change).
        """
        now = time.monotonic()
        with self._lock:
            new_stage = stage is not None and stage != job.last_stage
            recent = now - job.last_event < self._config.event_interval_s
            if not state_changed and not new_stage and recent:
                return
            job.last_event = now
            if stage is not None:
                job.last_stage = stage
            subscribers = tuple(job.subscribers)
        if not subscribers:
            return
        status = self._describe(job)
        for listener in subscribers:
            try:
                listener(status)
            except Exception:  # a broken follower must not break the job
                logger.exception("Listener of job %s failed", job.job_id)

    def subscribe(
        self, job_id: str, listener: StatusListener
    ) -> "tuple[JobStatus, Callable[[], None]]":
        """Follow a job with a callback: every status sent from now on reaches ``listener``.

        The listener runs in the job's thread: it must hand the status over quickly
        (to a queue, an event loop...) and never block. An async server uses this to
        follow jobs without holding a thread per follower.

        Args:
            job_id: Job id.
            listener: Called with each status sent (state changes, throttled progress).

        Returns:
            The job's status now, and a function that stops the listener.

        Raises:
            UnknownJobError: If no job has this id (or it was forgotten).
        """
        job = self._job(job_id)
        with self._lock:
            job.subscribers.append(listener)

        def unsubscribe() -> None:
            """Stop sending statuses to the listener (calling it twice is harmless)."""
            with self._lock:
                if listener in job.subscribers:
                    job.subscribers.remove(listener)

        return self._describe(job), unsubscribe

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
        status, unsubscribe = self.subscribe(job_id, functools.partial(put_latest, events))
        try:
            yield status
            while status.state not in FINISHED_STATES:
                try:
                    status = events.get(timeout=heartbeat_s)
                except queue.Empty:
                    status = self._describe(job)
                yield status
        finally:
            unsubscribe()

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
            UnknownJobError: If no job has this id (or it was forgotten).
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise UnknownJobError(f"no job {job_id!r}")
        return job

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
            state, error, final_elapsed = job.state, job.error, job.final_elapsed_s
        progress = None
        if state != JobState.QUEUED:
            progress = job.tracker.snapshot()
            if final_elapsed is not None:
                # A finished job's duration stops at its end, however late it is asked for.
                progress = dataclasses.replace(
                    progress,
                    fraction=1.0 if state == JobState.SUCCEEDED else progress.fraction,
                    elapsed_s=final_elapsed,
                    remaining_s=0.0,
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
