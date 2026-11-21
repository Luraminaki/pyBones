"""Schema of the web server configuration (``server.json``), kept apart from the results'."""

from typing import Annotated

from pydantic import Field

from pybones.config.fields import FileName, Section
from pybones.meshing.formats import MeshFormat


class HttpConfig(Section):
    """Address of the web server (local by default: no authentication)."""

    host: Annotated[str, Field(min_length=1)] = Field(description="Interface to listen on.")
    port: int = Field(ge=1, le=65535, description="TCP port.")
    shutdown_timeout_s: int = Field(
        ge=0, le=3600, description="Grace period for open connections when stopping."
    )


class LibraryConfig(Section):
    """Folders the web server reads stacks from and writes segmentations to."""

    data_dir: Annotated[str, Field(min_length=1)] = Field(
        description="Folder of the slice stacks (directories or zip archives)."
    )
    output_dir: Annotated[str, Field(min_length=1)] = Field(
        description="Folder of the segmentation outputs (one sub-folder each)."
    )
    mesh_file_template: Annotated[FileName, Field(pattern=r"\{label\}.*\.\{format\}$")] = Field(
        description="Name of a label's cached mesh inside its segmentation folder; "
        + "{format} is the extension (stl, glb)."
    )


class ApiConfig(Section):
    """Limits and timings of the web API."""

    max_preview_scale: float = Field(
        gt=0.0,
        le=1.0,
        description="Largest slice image scale (1 = full size: a browser zooms on its own, and an "
        + "enlarged slice would be built in server memory for nothing).",
    )
    default_mesh_format: MeshFormat = Field(
        description="Mesh format when a request names none (glb for the browser, stl to export)."
    )
    event_heartbeat_s: float = Field(
        gt=0.0, le=3600.0, description="Status repeated on an idle event stream after this time."
    )


class ServerConfig(Section):
    """Root of the web server configuration (``server.json``).

    Kept apart from :class:`PyBonesConfig` on purpose: every segmentation stores its
    :class:`PyBonesConfig`, and deployment settings are neither results nor
    reproducibility information.
    """

    http: HttpConfig
    library: LibraryConfig
    api: ApiConfig
