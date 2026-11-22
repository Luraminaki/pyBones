"""The record of one background job, and how a status reaches a slow follower.

:class:`~pybones.api.jobs.JobManager` keeps one :class:`JobRecord` per job, guarded by its lock;
:func:`put_latest` queues a status for a follower without ever blocking the job's thread.
"""

import contextlib
import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from pybones.api.job_status import JobState, JobStatus

if TYPE_CHECKING:
    from concurrent.futures import Future

    from pybones.segmentation.stages import ProgressTracker, Stage


class JobRecord:
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

    def __init__(self, job_id: str, kind: str, tracker: "ProgressTracker") -> None:
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
