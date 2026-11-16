"""Background jobs: list, inspect, follow (Server-Sent Events) and cancel."""

from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from pybones.server.context import Context
from pybones.server.library import NotFoundError
from pybones.server.schemas import JobView

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

    from pybones.api.jobs import JobStatus
    from pybones.server.context import ServerContext

SSE_MEDIA_TYPE = "text/event-stream"
#: Headers of an event stream: never cached, never buffered by a proxy.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

router = APIRouter(prefix="/jobs", tags=["jobs"])


def next_status(events: "Generator[JobStatus, None, None]") -> "JobStatus | None":
    """Wait for the next status of a job (blocking: run it in a worker thread).

    Args:
        events: The job's event iterator.

    Returns:
        The next status, or ``None`` once the job has finished.
    """
    return next(events, None)


async def sse_stream(
    context: "ServerContext", events: "Generator[JobStatus, None, None]", request: Request
) -> "AsyncIterator[str]":
    """Turn a job's statuses into Server-Sent Events, until it finishes or the client leaves.

    Each wait runs in a worker thread and lasts at most ``api.event_heartbeat_s``, so
    a departed client is noticed (and unsubscribed) even while the job is quiet.

    Args:
        context: Server state.
        events: The job's event iterator (closed at the end, which unsubscribes).
        request: The streaming request (to detect a departed client).

    Yields:
        One ``data: <JobView JSON>`` event per status.
    """
    try:
        while (status := await run_in_threadpool(next_status, events)) is not None:
            yield f"data: {context.view(status).model_dump_json()}\n\n"
            if await request.is_disconnected():
                return
    finally:
        events.close()


@router.get("")
def list_jobs(context: Context) -> list[JobView]:
    """List the remembered jobs, oldest first.

    Args:
        context: Server state.

    Returns:
        Their statuses and targets.
    """
    return context.all_jobs()


@router.get("/{job_id}")
def get_job(context: Context, job_id: str) -> JobView:
    """Describe a job now.

    Args:
        context: Server state.
        job_id: Job id.

    Returns:
        Its status and target.
    """
    return context.job(job_id)


@router.get(
    "/{job_id}/events",
    response_class=StreamingResponse,
    responses={HTTPStatus.OK: {"content": {SSE_MEDIA_TYPE: {}}}},
)
async def follow_job(context: Context, job_id: str, request: Request) -> StreamingResponse:
    """Follow a job with Server-Sent Events: its status now, then every change until it ends.

    Progress events are throttled (``jobs.event_interval_s``); state changes always
    come through, the last event has a finished state. Use ``EventSource`` in a browser.

    Args:
        context: Server state.
        job_id: Job id.
        request: The streaming request.

    Returns:
        The event stream.

    Raises:
        NotFoundError: If no job has this id.
    """
    try:
        _ = context.jobs.status(job_id)
    except KeyError:
        raise NotFoundError(f"no job {job_id!r}") from None
    events = context.jobs.events(job_id, context.server.api.event_heartbeat_s)
    return StreamingResponse(
        sse_stream(context, events, request), media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS
    )


@router.delete("/{job_id}", status_code=HTTPStatus.ACCEPTED)
def cancel_job(context: Context, job_id: str) -> JobView:
    """Ask a job to stop: a queued job never starts, a running one stops at its next update.

    Args:
        context: Server state.
        job_id: Job id.

    Returns:
        Its status now (follow it to see it end as ``cancelled``).

    Raises:
        NotFoundError: If no job has this id.
    """
    try:
        context.jobs.cancel(job_id)
    except KeyError:
        raise NotFoundError(f"no job {job_id!r}") from None
    return context.job(job_id)
