"""Local web server of the annotation page (standard library only).

Routes::

    GET  /                               the page (static/index.html)
    GET  /static/<file>                  page assets (whitelisted directory)
    GET  /api/session                    labels, colours, display settings, crop list
    GET  /api/crops/<id>/image/<offset>  8-bit raw crop at z + offset (PNG)
    GET  /api/crops/<id>/labels          current labels (raw uint8 bytes, row-major)
    PUT  /api/crops/<id>/labels          save labels (same raw format)

Only crop ids listed in the manifest are served, so no request can reach other files.
"""

import json
from enum import StrEnum
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import numpy as np

from pybones.segmentation.labels import FINAL_LABELS

if TYPE_CHECKING:
    from pybones.config.schema import PyBonesConfig
    from tools.annotation.settings import AnnotationSettings
    from tools.annotation.workspace import CropSpec, Manifest, Workspace

STATIC_DIR = Path(__file__).with_name("static")
INDEX_FILE = "index.html"
TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"
JSON_CONTENT_TYPE = "application/json"
RAW_CONTENT_TYPE = "application/octet-stream"
#: Content type of the static files, by extension (others are served as raw bytes).
STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".json": JSON_CONTENT_TYPE,
}
_QUERY_SEPARATOR = "?"
_PATH_SEPARATOR = "/"
_SIGNS = "+-"
_NOT_FOUND = "not found"
_TEXT_ENCODING = "utf-8"
#: A character takes at most this many bytes in UTF-8.
_MAX_UTF8_BYTES = 4


class Header(StrEnum):
    """HTTP headers read or written by the handler."""

    CONTENT_TYPE = "Content-Type"
    CONTENT_LENGTH = "Content-Length"
    CACHE_CONTROL = "Cache-Control"


#: Responses are never cached: labels change while the page is open.
NO_STORE = "no-store"


class Route(StrEnum):
    """URL path segments (``match`` compares them by value)."""

    API = "api"
    STATIC = "static"
    SESSION = "session"
    CROPS = "crops"
    IMAGE = "image"
    LABELS = "labels"
    NOTE = "note"


class AnnotationApp:
    """State shared by all requests: workspace, manifest and display settings.

    Attributes:
        workspace: Annotation workspace.
        settings: Tool settings.
        config: pybones configuration (label colours).
        manifest: Manifest of the workspace.
        crops: Crop id -> crop, for the crops of the manifest.
    """

    def __init__(
        self,
        workspace: "Workspace",
        settings: "AnnotationSettings",
        config: "PyBonesConfig",
    ) -> None:
        """Load the manifest of ``workspace``.

        Args:
            workspace: Annotation workspace (must hold a manifest).
            settings: Tool settings.
            config: pybones configuration (label colours).
        """
        self.workspace: Workspace = workspace
        self.settings: AnnotationSettings = settings
        self.config: PyBonesConfig = config
        self.manifest: Manifest = workspace.load_manifest()
        self.crops: dict[str, CropSpec] = {crop.crop_id: crop for crop in self.manifest.crops}

    def session(self) -> dict[str, Any]:
        """Describe everything the page needs to start.

        Returns:
            JSON-serialisable labels, colours, display settings and crop list.
        """
        colors = self.config.render.colors_bgr  # background has no colour: drawn transparent
        labels = [
            {
                "value": int(lab),
                "key": lab.key,
                "rgb": list(reversed(colors[lab.key])) if lab.key in colors else None,
            }
            for lab in FINAL_LABELS
        ]
        display = self.settings.display
        return {
            "labels": labels,
            "unsure": {"value": self.manifest.unsure_value, "rgb": list(display.unsure_rgb)},
            "crop_size": self.manifest.crop_size,
            "context_slices": self.manifest.context_slices,
            "prefilled": self.manifest.prefilled,
            "display": display.model_dump(),
            "max_note_chars": self.settings.max_note_chars,
            "crops": [
                {
                    **crop.model_dump(),
                    "saved": self.workspace.is_annotated(crop.crop_id),
                    "note": bool(self.workspace.read_note(crop.crop_id)),
                }
                for crop in self.manifest.crops
            ],
        }


class AnnotationHandler(BaseHTTPRequestHandler):
    """Request handler bound to an :class:`AnnotationApp`.

    Attributes:
        app: Shared state of the server.
    """

    def __init__(self, *args: Any, app: AnnotationApp, **kwargs: Any) -> None:
        """Keep a reference to the shared app state (see :func:`make_server`).

        Args:
            *args: Positional arguments of :class:`BaseHTTPRequestHandler`.
            app: Shared state of the server.
            **kwargs: Keyword arguments of :class:`BaseHTTPRequestHandler`.
        """
        self.app: AnnotationApp = app
        super().__init__(*args, **kwargs)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging.

        Args:
            format: printf-style message format (ignored).
            *args: Message arguments (ignored).
        """

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        """Send a complete, non-cacheable response.

        Args:
            status: HTTP status.
            body: Response body.
            content_type: Value of the ``Content-Type`` header.
        """
        self.send_response(status)
        self.send_header(Header.CONTENT_TYPE, content_type)
        self.send_header(Header.CONTENT_LENGTH, str(len(body)))
        self.send_header(Header.CACHE_CONTROL, NO_STORE)
        self.end_headers()
        _ = self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        """Send an error response with a plain-text message.

        Args:
            status: HTTP status.
            message: Explanation for the client.
        """
        self._send(status, message.encode(), TEXT_CONTENT_TYPE)

    def _file(self, path: Path) -> None:
        """Send a static file (404 if it does not exist).

        Args:
            path: File to send.
        """
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, _NOT_FOUND)
            return
        content_type = STATIC_CONTENT_TYPES.get(path.suffix, RAW_CONTENT_TYPE)
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _parts(self) -> list[str]:
        """Split the request path (query string dropped) into its segments.

        Returns:
            Non-empty path segments.
        """
        path = self.path.split(_QUERY_SEPARATOR, 1)[0]
        return [p for p in path.split(_PATH_SEPARATOR) if p]

    def do_GET(self) -> None:
        """Serve the page, its assets, the session and crop data."""
        match self._parts():
            case []:
                self._file(STATIC_DIR / INDEX_FILE)
            case [Route.STATIC, name]:
                target = (STATIC_DIR / name).resolve()
                if target.parent != STATIC_DIR.resolve():
                    self._error(HTTPStatus.NOT_FOUND, _NOT_FOUND)
                    return
                self._file(target)
            case [Route.API, Route.SESSION]:
                body = json.dumps(self.app.session()).encode()
                self._send(HTTPStatus.OK, body, JSON_CONTENT_TYPE)
            case [Route.API, Route.CROPS, crop_id, Route.IMAGE, offset] if (
                crop_id in self.app.crops and offset.lstrip(_SIGNS).isdigit()
            ):
                self._file(self.app.workspace.crop_image_path(crop_id, int(offset)))
            case [Route.API, Route.CROPS, crop_id, Route.LABELS] if crop_id in self.app.crops:
                labels = self.app.workspace.read_labels(crop_id, self.app.manifest.crop_size)
                self._send(HTTPStatus.OK, labels.tobytes(), RAW_CONTENT_TYPE)
            case [Route.API, Route.CROPS, crop_id, Route.NOTE] if crop_id in self.app.crops:
                note = self.app.workspace.read_note(crop_id).encode(_TEXT_ENCODING)
                self._send(HTTPStatus.OK, note, TEXT_CONTENT_TYPE)
            case _:
                self._error(HTTPStatus.NOT_FOUND, _NOT_FOUND)

    def do_PUT(self) -> None:
        """Save the labels or the note of one crop."""
        match self._parts():
            case [Route.API, Route.CROPS, crop_id, Route.LABELS] if crop_id in self.app.crops:
                self._save_labels(crop_id)
            case [Route.API, Route.CROPS, crop_id, Route.NOTE] if crop_id in self.app.crops:
                self._save_note(crop_id)
            case _:
                self._error(HTTPStatus.NOT_FOUND, "unknown crop")

    def _save_labels(self, crop_id: str) -> None:
        """Validate and save the labels sent in the request body (400 if invalid).

        Args:
            crop_id: Crop whose labels are saved.
        """
        size = self.app.manifest.crop_size
        length = int(self.headers.get(Header.CONTENT_LENGTH, "0"))
        if length != size * size:
            self._error(HTTPStatus.BAD_REQUEST, f"expected {size * size} bytes, got {length}")
            return
        labels = np.frombuffer(self.rfile.read(length), dtype=np.uint8).reshape(size, size)
        try:
            self.app.workspace.write_labels(crop_id, labels)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send(HTTPStatus.NO_CONTENT, b"", JSON_CONTENT_TYPE)

    def _save_note(self, crop_id: str) -> None:
        """Save the note sent in the request body as UTF-8 text (400 if invalid or too long).

        Args:
            crop_id: Crop whose note is saved.
        """
        length = int(self.headers.get(Header.CONTENT_LENGTH, "0"))
        if length > self.app.settings.max_note_chars * _MAX_UTF8_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "note too long")
            return
        try:
            text = self.rfile.read(length).decode(_TEXT_ENCODING)
            self.app.workspace.write_note(crop_id, text)
        except (UnicodeDecodeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send(HTTPStatus.NO_CONTENT, b"", TEXT_CONTENT_TYPE)


def make_server(app: AnnotationApp, host: str, port: int) -> ThreadingHTTPServer:
    """Create (not start) the HTTP server.

    Args:
        app: Shared state of the server.
        host: Interface to listen on.
        port: Port to listen on (0 picks a free port).

    Returns:
        The server, bound but not serving yet.
    """
    return ThreadingHTTPServer((host, port), partial(AnnotationHandler, app=app))
