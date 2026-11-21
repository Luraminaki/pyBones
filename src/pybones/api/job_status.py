"""What a front end sees of a job: its kind, state, progress and error."""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

# Pydantic reads the type of `JobStatus.progress` at runtime.
from pybones.segmentation.stages import Progress

SEGMENTATION_JOB = "segmentation"
MESH_JOB = "mesh"
QUANTIFICATION_JOB = "quantification"


class UnknownJobError(LookupError):
    """No job has this id (never submitted, or forgotten since it finished)."""


class JobState(StrEnum):
    """Life cycle of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: States after which a job does not change any more.
FINISHED_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class JobStatus(BaseModel):
    """What a front end shows about a job (JSON-ready: ``status.model_dump_json()``).

    Attributes:
        job_id: Identifier returned by :meth:`JobManager.submit`.
        kind: What the job does, e.g. ``segmentation``.
        state: Where it is in its life cycle.
        progress: Its progress (``None`` while queued); a finished job's is frozen at its
            end (time left 0, and 100 % if it succeeded).
        error: Why it failed, if it did.
        submitted_at: Submission time (seconds since the epoch).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    job_id: str
    kind: str
    state: JobState
    progress: Progress | None
    error: str | None
    submitted_at: float
