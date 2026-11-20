"""Version and configuration of the server."""

from typing import Any

from fastapi import APIRouter

from pybones import __version__
from pybones.config.schema import PyBonesConfig
from pybones.server.context import Context
from pybones.server.schemas import OVERRIDABLE_SECTIONS, ServerInfo, VersionInfo

router = APIRouter(tags=["system"])


@router.get("/version")
def get_version() -> VersionInfo:
    """Report the version of pyBones serving the API.

    Returns:
        The version.
    """
    return VersionInfo(version=__version__)


@router.get("/server")
def get_server(context: Context) -> ServerInfo:
    """Describe this server: version, folders and limits the web page adapts to.

    Args:
        context: Server state.

    Returns:
        The server description.
    """
    return ServerInfo(
        version=__version__,
        data_dir=str(context.library.data_dir),
        output_dir=str(context.library.output_dir),
        max_preview_scale=context.server.api.max_preview_scale,
        default_mesh_format=context.server.api.default_mesh_format.value,
        overridable_sections=sorted(OVERRIDABLE_SECTIONS),
    )


@router.get("/config")
def get_config(context: Context) -> PyBonesConfig:
    """Return the base configuration of new segmentations.

    Args:
        context: Server state.

    Returns:
        The configuration (a segmentation request may override parts of it).
    """
    return context.config


@router.get("/config/schema")
def get_config_schema() -> dict[str, Any]:
    """Return the JSON schema of the configuration (types, bounds, descriptions).

    Returns:
        The JSON schema, for a front end to build its forms from.
    """
    return PyBonesConfig.model_json_schema()
