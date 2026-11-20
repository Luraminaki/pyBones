"""Request and response bodies of the HTTP API (JSON, validated by pydantic)."""

from enum import StrEnum
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pybones.api.jobs import JobStatus
from pybones.config.loader import override_section

#: Ids of datasets and segmentations: the name of an entry directly inside the data or
#: output folder. No separator and no leading dot, so an id can never leave its folder.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
MAX_ID_LENGTH = 128
EntryId = Annotated[str, Field(pattern=ID_PATTERN, max_length=MAX_ID_LENGTH)]

#: Configuration sections a segmentation request may override: those that shape the
#: result. File names, logging, jobs, meshing and rendering stay the server's own.
OVERRIDABLE_SECTIONS = frozenset(
    {
        "acquisition",
        "pipeline",
        "thresholds",
        "segmentation",
        "tubularity",
        "z_smoothing",
        "refinement",
    }
)


class Body(BaseModel):
    """Base of the API bodies: unknown fields are rejected and instances are immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class VersionInfo(Body):
    """``GET /api/version``.

    Attributes:
        version: Version of pyBones serving the API.
    """

    version: str


class ServerInfo(Body):
    """``GET /api/server``: what a front end needs to know about this server.

    Attributes:
        version: Version of pyBones serving the API.
        data_dir: Absolute folder the stacks are read from.
        output_dir: Absolute folder the segmentations are written to.
        max_preview_scale: Largest slice image scale.
        default_mesh_format: Mesh format when a request names none.
        overridable_sections: Configuration sections a segmentation request may override.
    """

    version: str
    data_dir: str
    output_dir: str
    max_preview_scale: float
    default_mesh_format: str
    overridable_sections: list[str]


class DatasetKind(StrEnum):
    """How a stack is stored."""

    DIRECTORY = "directory"
    ZIP = "zip"


class DatasetInfo(Body):
    """A slice stack of the data folder.

    Attributes:
        id: Name of the directory or archive in the data folder.
        kind: Directory or zip archive.
        slices: Number of slices.
        height: Slice height in pixels.
        width: Slice width in pixels.
    """

    id: str
    kind: DatasetKind
    slices: int
    height: int
    width: int


class SegmentationSummary(Body):
    """A finished segmentation of the output folder.

    Attributes:
        id: Name of its folder in the output folder.
        source: Path of the stack it was made from.
        pybones_version: Version of pyBones that made it.
        shape_zyx: Shape of the label volume.
        voxel_size_um: Voxel size it was made with, ``(x, y, z)``.
        label_fractions: Fraction of the volume per label key.
        meshes: Formats ready to download, by label key (labels without a mesh omitted).
    """

    id: str
    source: str
    pybones_version: str
    shape_zyx: tuple[int, int, int]
    voxel_size_um: tuple[float, float, float]
    label_fractions: dict[str, float]
    meshes: dict[str, list[str]]


class SegmentationRequest(Body):
    """``POST /api/segmentations``: segment a dataset into a new segmentation.

    Attributes:
        dataset: Id of the stack to segment.
        name: Id of the new segmentation (must not exist yet).
        overrides: ``section.key=value`` changes to the server's configuration, limited
            to :data:`OVERRIDABLE_SECTIONS`.
    """

    dataset: EntryId
    name: EntryId
    overrides: tuple[str, ...] = ()

    @field_validator("overrides")
    @classmethod
    def _overridable(cls, overrides: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse overrides of sections that are the server's own.

        Args:
            overrides: Requested overrides.

        Returns:
            ``overrides``, unchanged.

        Raises:
            ValueError: If an override targets another section.
        """
        refused = [item for item in overrides if override_section(item) not in OVERRIDABLE_SECTIONS]
        if refused:
            raise ValueError(f"only {sorted(OVERRIDABLE_SECTIONS)} can be overridden: {refused}")
        return overrides


class JobView(JobStatus):
    """A job's status and what it works on.

    Attributes:
        segmentation: Id of the segmentation the job writes or reads.
        label: Key of the label being meshed (mesh jobs only).
        mesh_format: Format of the mesh being written (mesh jobs only).
    """

    segmentation: str
    label: str | None
    mesh_format: str | None
