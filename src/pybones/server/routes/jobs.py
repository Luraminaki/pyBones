"""Background jobs: list, inspect, follow (Server-Sent Events) and cancel."""

import asyncio
import contextlib
from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from pybones.api.job_status import FINISHED_STATES, UnknownJobError
from pybones.server.context import Context
from pybones.server.schemas import JobView

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from pybones.api.job_status import JobStatus
    from pybones.server.context import ServerContext

SSE_MEDIA_TYPE = "text/event-stream"
#: Headers of an event stream: never cached, never buffered by a proxy.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

router = APIRouter(prefix="/jobs", tags=["jobs"])


def offer_latest(inbox: "asyncio.Queue[JobStatus]", status: "JobStatus") -> None:
    """Queue a status, dropping the oldest one if the client lags.

    Runs in the event loop, the only thread touching ``inbox``: nothing can change it
    between the check and the put.

    Args:
        inbox: The stream's queue.
        status: Status to send.
    """
    if inbox.full():
        _ = inbox.get_nowait()
    inbox.put_nowait(status)


def sse_event(context: "ServerContext", status: "JobStatus") -> str:
    """Format a status as one Server-Sent Event.

    Args:
        context: Server state (adds the job's target).
        status: Job status.

    Returns:
        ``data: <JobView JSON>`` and the blank line ending the event.
    """
    return f"data: {context.view(status).model_dump_json()}\n\n"


async def sse_stream(
    context: "ServerContext",
    job_id: str,
    first: "JobStatus",
    inbox: "asyncio.Queue[JobStatus]",
    unsubscribe: "Callable[[], None]",
    request: Request,
) -> "AsyncIterator[str]":
    """Send a job's statuses as Server-Sent Events, until it finishes or the client leaves.

    Statuses arrive in ``inbox`` from the job's thread through the event loop, so a
    waiting stream holds no worker thread. When nothing came for ``api.event_heartbeat_s``,
    the current status is sent again, which also reveals a departed client.

    Args:
        context: Server state.
        job_id: Job id.
        first: The job's status when the stream opened.
        inbox: Statuses sent since.
        unsubscribe: Stops the job from feeding ``inbox`` (called at the end).
        request: The streaming request (to detect a departed client).

    Yields:
        One event per status; the last one has a finished state.
    """
    heartbeat = context.server.api.event_heartbeat_s
    status = first
    try:
        yield sse_event(context, status)
        while status.state not in FINISHED_STATES:
            try:
                status = await asyncio.wait_for(inbox.get(), heartbeat)
            except TimeoutError:
                try:
                    status = context.jobs.status(job_id)
                except UnknownJobError:  # forgotten meanwhile by another thread: nothing to say
                    return
            yield sse_event(context, status)
            if await request.is_disconnected():
                return
    finally:
        unsubscribe()


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
        UnknownJobError: If no job has this id (answered 404).
    """
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue[JobStatus] = asyncio.Queue(maxsize=context.config.jobs.event_queue_size)

    def deliver(status: "JobStatus") -> None:
        """Hand a status from the job's thread to the event loop (never blocks).

        Args:
            status: Status sent.
        """
        with contextlib.suppress(RuntimeError):  # the loop is closing: nobody to tell
            _ = loop.call_soon_threadsafe(offer_latest, inbox, status)

    first, unsubscribe = context.jobs.subscribe(job_id, deliver)
    return StreamingResponse(
        sse_stream(context, job_id, first, inbox, unsubscribe, request),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
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
        UnknownJobError: If no job has this id (answered 404).
    """
    context.jobs.cancel(job_id)
    return context.job(job_id)
