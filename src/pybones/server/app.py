"""The FastAPI application: routers, error mapping and the job manager's life cycle.

Routes (all under ``/api``; interactive documentation at ``/docs``)::

    GET    /version                               version of pyBones
    GET    /config                                base configuration of new segmentations
    GET    /config/schema                         its JSON schema (types, bounds, descriptions)
    GET    /datasets                              slice stacks of the data folder
    GET    /segmentations                         finished segmentations of the output folder
    POST   /segmentations                         segment a dataset (202 + job)
    GET    /segmentations/{id}                    full metadata
    GET    /segmentations/{id}/slices/{z}         slice with its labels overlaid (PNG)
    POST   /segmentations/{id}/meshes/{label}     build a label's mesh (202 + job)
    GET    /segmentations/{id}/meshes/{label}     download it (binary STL, µm)
    GET    /jobs                                  remembered jobs
    GET    /jobs/{id}                             one job
    GET    /jobs/{id}/events                      follow it (Server-Sent Events)
    DELETE /jobs/{id}                             cancel it (202)

``GET /`` (outside ``/api``) redirects a browser to the documentation.

Errors use HTTP status codes with FastAPI's ``{"detail": ...}`` body: 404 for an
unknown id, 409 for a name already taken or a segmentation written by an older pyBones, 422 for
invalid input.
"""

from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse

from pybones import __version__
from pybones.api.jobs import JobManager
from pybones.segmentation.results import IncompatibleOutputError
from pybones.server.context import ServerContext
from pybones.server.library import ConflictError, Library, NotFoundError
from pybones.server.routes import datasets, jobs, segmentations, system

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import Request

    from pybones.config.schema import PyBonesConfig, ServerConfig

API_PREFIX = "/api"
#: Interactive API documentation (Swagger UI).
DOCS_PATH = "/docs"
#: Listening addresses meaning "every interface": a browser reaches them as localhost.
ANY_HOSTS = frozenset({"0.0.0.0", "::"})
LOCALHOST = "localhost"
TITLE = "pyBones"
DESCRIPTION = "Segmentation and 3D modelling of the intra-osseous vascular network (µCT)."


def _error(status: HTTPStatus, exc: Exception) -> JSONResponse:
    """Build an error response in FastAPI's format.

    Args:
        status: HTTP status.
        exc: The error (its message becomes the detail).

    Returns:
        ``{"detail": message}`` with ``status``.
    """
    return JSONResponse({"detail": str(exc)}, status_code=status)


async def not_found(_request: "Request", exc: Exception) -> JSONResponse:
    """Answer 404 to a :class:`NotFoundError`.

    Args:
        _request: Failed request (unused).
        exc: The error.

    Returns:
        The 404 response.
    """
    return _error(HTTPStatus.NOT_FOUND, exc)


async def conflict(_request: "Request", exc: Exception) -> JSONResponse:
    """Answer 409 to a :class:`ConflictError` or an output this version cannot read.

    Args:
        _request: Failed request (unused).
        exc: The error.

    Returns:
        The 409 response.
    """
    return _error(HTTPStatus.CONFLICT, exc)


def server_url(server: "ServerConfig") -> str:
    """Address to type in a browser on this machine.

    Args:
        server: Server configuration.

    Returns:
        E.g. ``http://127.0.0.1:8000``.
    """
    host = LOCALHOST if server.http.host in ANY_HOSTS else server.http.host
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    return f"http://{host}:{server.http.port}"


def create_app(config: "PyBonesConfig", server: "ServerConfig") -> FastAPI:
    """Build the application and its state (relative folders resolve against the working dir).

    The job manager is shut down when the application stops: running jobs are
    cancelled and their partial outputs removed.

    Args:
        config: Base configuration of the segmentations.
        server: Server configuration.

    Returns:
        The application, ready for uvicorn or a test client.
    """
    manager = JobManager(config.jobs)
    library = Library.from_config(server.library, config.io)
    context = ServerContext(config=config, server=server, library=library, jobs=manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> "AsyncGenerator[None]":
        """Run the application, then stop its jobs.

        Args:
            _app: The application (unused).

        Yields:
            Control while the application serves.
        """
        try:
            yield
        finally:
            await run_in_threadpool(manager.shutdown)

    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url=DOCS_PATH,
    )
    app.state.context = context
    app.add_exception_handler(NotFoundError, not_found)
    app.add_exception_handler(ConflictError, conflict)
    app.add_exception_handler(IncompatibleOutputError, conflict)
    for module in (system, datasets, segmentations, jobs):
        app.include_router(module.router, prefix=API_PREFIX)

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        """Send a browser to the API documentation (until the web page exists).

        Returns:
            A redirection to :data:`DOCS_PATH`.
        """
        return RedirectResponse(DOCS_PATH)

    return app
