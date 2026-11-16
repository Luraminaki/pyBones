"""State shared by the routes: configuration, library and jobs, and how jobs are started.

One :class:`ServerContext` is built per application (see
:func:`pybones.server.app.create_app`) and reaches the routes through the
:data:`Context` dependency, so tests build their own application instead of patching
module globals.
"""

import contextlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, cast

# FastAPI reads the annotations of dependencies at runtime (`get_context` below).
from fastapi import Depends, Request  # noqa: TC002

from pybones.api.jobs import FINISHED_STATES, SEGMENTATION_JOB
from pybones.api.operations import segment
from pybones.config.loader import apply_overrides
from pybones.segmentation.pipeline import planned_stages
from pybones.server.library import ConflictError, NotFoundError
from pybones.server.schemas import JobView

if TYPE_CHECKING:
    from pybones.api.jobs import JobManager, JobStatus
    from pybones.config.schema import PyBonesConfig, ServerConfig
    from pybones.meshing.formats import MeshFormat
    from pybones.segmentation.labels import Label
    from pybones.segmentation.results import SegmentationMeta
    from pybones.segmentation.stages import ProgressTracker
    from pybones.server.library import Library
    from pybones.server.schemas import SegmentationRequest


@dataclass(frozen=True, slots=True)
class JobTarget:
    """What a job works on.

    Attributes:
        segmentation: Segmentation id.
        label: Label being meshed (``None`` for a segmentation job).
        mesh_format: Format of the mesh (``None`` for a segmentation job).
    """

    segmentation: str
    label: "Label | None"
    mesh_format: "MeshFormat | None"


@dataclass(slots=True)
class ServerContext:
    """Everything the routes need.

    Attributes:
        config: Base configuration of the segmentations (a request may override parts).
        server: Server configuration.
        library: Datasets and segmentations.
        jobs: Background jobs.
        targets: What each remembered job works on, by job id.
        lock: Guards :attr:`targets`.
        submit_lock: Makes "check that no job works on this target, then start one" atomic.
    """

    config: "PyBonesConfig"
    server: "ServerConfig"
    library: "Library"
    jobs: "JobManager"
    targets: dict[str, JobTarget] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    submit_lock: threading.Lock = field(default_factory=threading.Lock)

    def view(self, status: "JobStatus") -> JobView:
        """Add a job's target to its status.

        Args:
            status: Job status.

        Returns:
            The status and the segmentation (and label) it concerns.
        """
        with self.lock:
            target = self.targets[status.job_id]
        return JobView(
            **status.model_dump(),
            segmentation=target.segmentation,
            label=None if target.label is None else target.label.key,
            mesh_format=None if target.mesh_format is None else target.mesh_format.value,
        )

    def job(self, job_id: str) -> JobView:
        """Describe a job.

        Args:
            job_id: Job id.

        Returns:
            Its status and target.

        Raises:
            NotFoundError: If no job has this id (or it was forgotten).
        """
        try:
            return self.view(self.jobs.status(job_id))
        except KeyError:
            raise NotFoundError(f"no job {job_id!r}") from None

    def all_jobs(self) -> list[JobView]:
        """Describe every remembered job, oldest first.

        Returns:
            Their statuses and targets (a job submitted this very moment may be missing).
        """
        statuses = self.jobs.statuses()
        with self.lock:
            recorded = set(self.targets)
        return [self.view(status) for status in statuses if status.job_id in recorded]

    def active_job(self, target: JobTarget) -> JobView | None:
        """Find the unfinished job working on a target, if any.

        Args:
            target: Segmentation (and label) to look for.

        Returns:
            The queued or running job, or ``None``.
        """
        with self.lock:
            same = [job_id for job_id, t in self.targets.items() if t == target]
        for job_id in same:
            with contextlib.suppress(NotFoundError):
                current = self.job(job_id)
                if current.state not in FINISHED_STATES:
                    return current
        return None

    def _record(self, job_id: str, target: JobTarget) -> None:
        """Remember a new job's target, forgetting those of the jobs the manager dropped.

        Args:
            job_id: New job.
            target: What it works on.
        """
        known = {status.job_id for status in self.jobs.statuses()}
        with self.lock:
            self.targets[job_id] = target
            for old in [j for j in self.targets if j not in known]:
                del self.targets[old]

    def start_segmentation(self, request: "SegmentationRequest") -> JobView:
        """Queue the segmentation of a dataset into a new output folder.

        The name is taken as long as its folder exists or an unfinished job will write
        it, so a job cancelled while queued frees it without leaving anything behind.
        The folder is created when the job starts, and removed if the run fails or
        is cancelled (the pipeline leaves it empty then).

        Args:
            request: Dataset, new name and configuration overrides.

        Returns:
            The queued job.

        Raises:
            ConflictError: If the name is taken.
        """
        config = apply_overrides(self.config, request.overrides)
        source = self.library.dataset_path(request.dataset)
        out_dir = self.library.new_segmentation_path(request.name)
        target = JobTarget(request.name, None, None)

        def work(tracker: "ProgressTracker") -> "SegmentationMeta":
            """Segment, removing the output folder if the run does not finish.

            Args:
                tracker: Progress tracker of the job.

            Returns:
                The segmentation metadata.
            """
            try:
                return segment(source, out_dir, config, tracker)
            except BaseException:
                with contextlib.suppress(OSError):
                    out_dir.rmdir()
                raise

        with self.submit_lock:
            if self.active_job(target) is not None:
                raise ConflictError(f"a job is already writing {request.name!r}")
            job_id = self.jobs.submit(
                SEGMENTATION_JOB, work, planned_stages(config), config.pipeline.stage_weights
            )
            self._record(job_id, target)
        return self.job(job_id)

    def start_mesh(
        self, segmentation_id: str, label: "Label", mesh_format: "MeshFormat"
    ) -> JobView:
        """Queue the meshing of a label in one format, or return the job already doing it.

        The geometry (voxel size) is the one the segmentation was made with; the mesh
        settings (grid, smoothing, GLB precision) are the server's. Asking again once a mesh
        exists rebuilds it, e.g. after changing those settings.

        Args:
            segmentation_id: Segmentation id.
            label: Foreground label.
            mesh_format: File format.

        Returns:
            The queued (or already running) job.
        """
        path = self.library.mesh_path(segmentation_id, label, mesh_format)
        stored = self.library.meta(segmentation_id).config
        config = stored.model_copy(update={"mesh": self.config.mesh})
        segmentation = self.library.segmentation_path(segmentation_id)
        target = JobTarget(segmentation_id, label, mesh_format)
        with self.submit_lock:
            if (current := self.active_job(target)) is not None:
                return current
            job_id = self.jobs.submit_mesh(segmentation, label, path, config)
            self._record(job_id, target)
        return self.job(job_id)


def get_context(request: Request) -> ServerContext:
    """Fetch the context of the application serving a request (FastAPI dependency).

    Args:
        request: Current request.

    Returns:
        The application's context.
    """
    return cast("ServerContext", request.app.state.context)


#: Route parameter receiving the :class:`ServerContext`.
Context = Annotated[ServerContext, Depends(get_context)]
