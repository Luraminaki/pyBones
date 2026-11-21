"""Segmentations: list, create (as a job), inspect, slice images and meshes."""

from http import HTTPStatus
from typing import TYPE_CHECKING, Annotated, cast

import cv2
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse, Response

from pybones.api.operations import preview
from pybones.meshing.formats import MeshFormat
from pybones.quantification.measures import NetworkSummary
from pybones.segmentation.labels import (
    FOREGROUND_LABELS,
    Label,
    foreground_from_key,
    label_keys,
)
from pybones.segmentation.results import SegmentationMeta
from pybones.server.context import Context
from pybones.server.library import NotFoundError
from pybones.server.schemas import JobView, SegmentationRequest, SegmentationSummary

if TYPE_CHECKING:
    import pathlib

    import numpy as np

PNG_EXTENSION = ".png"
PNG_MEDIA_TYPE = "image/png"
#: Slices are rendered on demand and can weigh megabytes: browsers must not keep them (a
#: page scrolling through a stack would otherwise fill its cache with every slice seen).
SLICE_HEADERS = {"Cache-Control": "no-store"}
#: Meshes and centrelines may be kept, but checked on every use: a rebuilt file replaces the
#: old one at the same address, and without this the browser could keep showing the old one.
REVALIDATE_CACHE_CONTROL = "no-cache"
ETAG_HEADER = "ETag"
IF_NONE_MATCH_HEADER = "if-none-match"
#: Centreline points: raw little-endian float32 ``(x, y, z)`` triples in µm.
CENTRELINES_MEDIA_TYPE = "application/octet-stream"
#: Media type of each mesh format.
MESH_MEDIA_TYPES = {MeshFormat.STL: "model/stl", MeshFormat.GLB: "model/gltf-binary"}
#: Query parameter choosing the mesh format.
FORMAT_PARAMETER = "format"

router = APIRouter(prefix="/segmentations", tags=["segmentations"])


def cache_headers(path: "pathlib.Path") -> dict[str, str]:
    """Build the headers letting a browser keep a file but check it on every use.

    Args:
        path: Served file (its modification time and size make the ETag).

    Returns:
        ``Cache-Control`` and ``ETag`` headers.
    """
    stat = path.stat()
    return {
        "Cache-Control": REVALIDATE_CACHE_CONTROL,
        ETAG_HEADER: f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"',
    }


def unchanged(headers: dict[str, str], request: Request) -> bool:
    """Tell whether the browser already holds this version of the file.

    Args:
        headers: From :func:`cache_headers`.
        request: The request (a browser echoes the ETag in ``If-None-Match``).

    Returns:
        Whether to answer ``304 Not Modified``.
    """
    # The substring test also covers a list of ETags.
    return headers[ETAG_HEADER] in request.headers.get(IF_NONE_MATCH_HEADER, "")


def foreground_label(label: str) -> Label:
    """Parse the ``{label}`` path parameter (FastAPI dependency).

    Args:
        label: Label key from the path.

    Returns:
        The foreground label.

    Raises:
        HTTPException: 422 if it names no foreground label.
    """
    keys = label_keys(FOREGROUND_LABELS)
    if label.lower() not in keys:
        detail = f"choose one of {keys}, not {label!r}"
        raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, detail)
    return foreground_from_key(label)


ForegroundLabel = Annotated[Label, Depends(foreground_label)]


def mesh_format(
    context: Context,
    requested: Annotated[MeshFormat | None, Query(alias=FORMAT_PARAMETER)] = None,
) -> MeshFormat:
    """Resolve the ``?format=`` query parameter (FastAPI dependency).

    Args:
        context: Server state.
        requested: Format named in the request, if any.

    Returns:
        That format, or the server's ``api.default_mesh_format``.
    """
    return context.server.api.default_mesh_format if requested is None else requested


RequestedFormat = Annotated[MeshFormat, Depends(mesh_format)]


@router.get("")
def list_segmentations(context: Context) -> list[SegmentationSummary]:
    """List the finished segmentations of the output folder.

    Args:
        context: Server state.

    Returns:
        One summary per segmentation, sorted by id.
    """
    return context.library.segmentations()


@router.post("", status_code=HTTPStatus.ACCEPTED)
def create_segmentation(context: Context, request: SegmentationRequest) -> JobView:
    """Start segmenting a dataset into a new segmentation (follow the returned job).

    Args:
        context: Server state.
        request: Dataset, new segmentation id and configuration overrides.

    Returns:
        The queued job.

    Raises:
        HTTPException: 422 if an override names an unknown key or an invalid value.
    """
    try:
        return context.start_segmentation(request)
    except ValueError as exc:  # pydantic's ValidationError included
        raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from None


@router.get("/{segmentation_id}")
def get_segmentation(context: Context, segmentation_id: str) -> SegmentationMeta:
    """Return the full metadata of a segmentation (thresholds, fractions, configuration).

    Args:
        context: Server state.
        segmentation_id: Segmentation id.

    Returns:
        Its metadata.
    """
    return context.library.meta(segmentation_id)


@router.get(
    "/{segmentation_id}/slices/{z}",
    response_class=Response,
    responses={HTTPStatus.OK: {"content": {PNG_MEDIA_TYPE: {}}}},
)
def get_slice(
    context: Context,
    segmentation_id: str,
    z: Annotated[int, Path(ge=0)],
    scale: Annotated[float | None, Query(gt=0.0)] = None,
) -> Response:
    """Render one slice with its labels overlaid, as a PNG image.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.
        z: Slice index.
        scale: Image scale factor (default: full size), up to ``api.max_preview_scale``.

    Returns:
        The PNG image.

    Raises:
        HTTPException: 422 if ``scale`` is too large.
        NotFoundError: If the slice is outside the stack or the source stack is gone.
        RuntimeError: If the image cannot be encoded.
    """
    if scale is not None and scale > context.server.api.max_preview_scale:
        detail = f"scale must be <= {context.server.api.max_preview_scale}"
        raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, detail)
    depth = context.library.meta(segmentation_id).shape_zyx[0]
    if z >= depth:
        raise NotFoundError(f"slice {z} is outside the stack (0..{depth - 1})")
    segmentation = context.library.segmentation_path(segmentation_id)
    try:  # the stack is read from where it was when segmented: it may have moved
        image = preview(segmentation, z, context.config, scale=scale)
    except FileNotFoundError as exc:
        raise NotFoundError(f"source stack of {segmentation_id!r} not found: {exc}") from None
    ok, encoded = cv2.imencode(PNG_EXTENSION, image)
    if not ok:
        raise RuntimeError(f"could not encode slice {z} of {segmentation_id!r}")
    return Response(
        cast("np.ndarray", encoded).tobytes(), media_type=PNG_MEDIA_TYPE, headers=SLICE_HEADERS
    )


@router.post("/{segmentation_id}/meshes/{label}", status_code=HTTPStatus.ACCEPTED)
def create_mesh(
    context: Context, segmentation_id: str, label: ForegroundLabel, fmt: RequestedFormat
) -> JobView:
    """Start meshing a label (or return the job already doing it); follow the returned job.

    ``?format=glb`` (for the browser) or ``?format=stl`` (for other software); the
    default is the server's ``api.default_mesh_format``. Asking again rebuilds the mesh.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.
        label: Foreground label.
        fmt: File format.

    Returns:
        The meshing job.
    """
    return context.start_mesh(segmentation_id, label, fmt)


@router.get(
    "/{segmentation_id}/meshes/{label}",
    response_class=FileResponse,
    responses={HTTPStatus.OK: {"content": {t: {} for t in MESH_MEDIA_TYPES.values()}}},
)
def get_mesh(
    context: Context,
    segmentation_id: str,
    label: ForegroundLabel,
    fmt: RequestedFormat,
    request: Request,
) -> Response:
    """Download the mesh of a label (coordinates in µm), once it was built.

    The browser may keep it but must ask again each time (``Cache-Control: no-cache``):
    an unchanged mesh is answered ``304 Not Modified`` (no download), a rebuilt one in
    full, since its ETag follows the file's modification time and size.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.
        label: Foreground label.
        fmt: File format (``?format=``, default ``api.default_mesh_format``).
        request: The request (its ``If-None-Match`` header).

    Returns:
        The mesh file, or an empty 304 if the browser's copy is current.

    Raises:
        NotFoundError: If the mesh was not built yet (``POST`` the same path first).
    """
    path = context.library.mesh_path(segmentation_id, label, fmt)
    if not path.is_file():
        raise NotFoundError(f"no {label.key} {fmt.value} mesh for {segmentation_id!r} yet")
    headers = cache_headers(path)
    if unchanged(headers, request):
        return Response(status_code=HTTPStatus.NOT_MODIFIED, headers=headers)
    return FileResponse(path, media_type=MESH_MEDIA_TYPES[fmt], filename=path.name, headers=headers)


@router.post("/{segmentation_id}/quantification", status_code=HTTPStatus.ACCEPTED)
def create_quantification(context: Context, segmentation_id: str) -> JobView:
    """Start measuring the vessel network (or return the job doing it); follow the returned job.

    Asking again measures it again, e.g. after changing the server's quantification settings.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.

    Returns:
        The quantification job.
    """
    return context.start_quantification(segmentation_id)


@router.get("/{segmentation_id}/quantification")
def get_quantification(context: Context, segmentation_id: str) -> NetworkSummary:
    """Return the summary of the vessel network, once measured.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.

    Returns:
        Branches, junctions, ends, lengths and medians for the whole stack.

    Raises:
        NotFoundError: If it was not quantified yet (``POST`` the same path first).
    """
    return context.library.network(segmentation_id)


@router.get(
    "/{segmentation_id}/centrelines",
    response_class=Response,
    responses={HTTPStatus.OK: {"content": {CENTRELINES_MEDIA_TYPE: {}}}},
)
def get_centrelines(context: Context, segmentation_id: str, request: Request) -> Response:
    """Download the centreline voxels as points in µm, for drawing next to the meshes.

    The body is ``N`` little-endian float32 ``(x, y, z)`` triples (``12 N`` bytes). Cached
    like the meshes: an unchanged file is answered ``304 Not Modified``.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.
        request: The request (its ``If-None-Match`` header).

    Returns:
        The points, or an empty 304 if the browser's copy is current.

    Raises:
        NotFoundError: If it was not quantified yet.
    """
    headers = cache_headers(context.library.centreline_path(segmentation_id))
    if unchanged(headers, request):
        return Response(status_code=HTTPStatus.NOT_MODIFIED, headers=headers)
    points = context.library.centreline_points_um(segmentation_id)
    return Response(
        points.astype("<f4").tobytes(), media_type=CENTRELINES_MEDIA_TYPE, headers=headers
    )
