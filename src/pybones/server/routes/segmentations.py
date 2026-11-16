"""Segmentations: list, create (as a job), inspect, slice images and meshes."""

from http import HTTPStatus
from typing import TYPE_CHECKING, Annotated, cast

import cv2
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import FileResponse, Response

from pybones.api.operations import preview
from pybones.segmentation.labels import Label, foreground_from_key
from pybones.segmentation.results import SegmentationMeta
from pybones.server.context import Context
from pybones.server.library import NotFoundError
from pybones.server.schemas import JobView, SegmentationRequest, SegmentationSummary

if TYPE_CHECKING:
    import numpy as np

PNG_EXTENSION = ".png"
PNG_MEDIA_TYPE = "image/png"
STL_MEDIA_TYPE = "model/stl"

router = APIRouter(prefix="/segmentations", tags=["segmentations"])


def foreground_label(label: str) -> Label:
    """Parse the ``{label}`` path parameter (FastAPI dependency).

    Args:
        label: Label key from the path.

    Returns:
        The foreground label.

    Raises:
        HTTPException: 422 if it names no foreground label.
    """
    try:
        return foreground_from_key(label)
    except ValueError as exc:
        raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from None


ForegroundLabel = Annotated[Label, Depends(foreground_label)]


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
    segmentation = context.library.segmentation_path(segmentation_id)
    try:
        image = preview(segmentation, z, context.config, scale=scale)
    except IndexError as exc:
        raise NotFoundError(str(exc)) from None
    except FileNotFoundError as exc:
        raise NotFoundError(f"source stack of {segmentation_id!r} not found: {exc}") from None
    ok, encoded = cv2.imencode(PNG_EXTENSION, image)
    if not ok:
        raise RuntimeError(f"could not encode slice {z} of {segmentation_id!r}")
    return Response(cast("np.ndarray", encoded).tobytes(), media_type=PNG_MEDIA_TYPE)


@router.post("/{segmentation_id}/meshes/{label}", status_code=HTTPStatus.ACCEPTED)
def create_mesh(context: Context, segmentation_id: str, label: ForegroundLabel) -> JobView:
    """Start meshing a label (or return the job already doing it); follow the returned job.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.
        label: Foreground label.

    Returns:
        The meshing job.
    """
    return context.start_mesh(segmentation_id, label)


@router.get(
    "/{segmentation_id}/meshes/{label}",
    response_class=FileResponse,
    responses={HTTPStatus.OK: {"content": {STL_MEDIA_TYPE: {}}}},
)
def get_mesh(context: Context, segmentation_id: str, label: ForegroundLabel) -> FileResponse:
    """Download the mesh of a label (binary STL, coordinates in µm), once it was built.

    Args:
        context: Server state.
        segmentation_id: Segmentation id.
        label: Foreground label.

    Returns:
        The STL file.

    Raises:
        NotFoundError: If the mesh was not built yet (``POST`` the same path first).
    """
    path = context.library.mesh_path(segmentation_id, label)
    if not path.is_file():
        raise NotFoundError(f"no {label.key} mesh for {segmentation_id!r} yet")
    return FileResponse(path, media_type=STL_MEDIA_TYPE, filename=path.name)
