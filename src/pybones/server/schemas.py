"""Request and response bodies of the HTTP API (JSON, validated by pydantic)."""

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pybones.api.job_status import MESH_JOB, QUANTIFICATION_JOB, SEGMENTATION_JOB, JobStatus
from pybones.config.loader import override_path, override_section
from pybones.config.quantification import QuantificationConfig
from pybones.quantification.measures import NetworkSummary

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        "thin_vessels",
    }
)
#: The section a mesh job may change, and the one a quantification job may change.
MESH_SECTIONS = frozenset({"mesh"})
QUANTIFICATION_SECTIONS = frozenset({"quantification"})


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
        settings_sections: Configuration sections each kind of job may change, by job kind
            (the sections of its settings form).
        settings_kept: Paths inside those sections that stay the server's (left out of the
            forms).
    """

    version: str
    data_dir: str
    output_dir: str
    max_preview_scale: float
    default_mesh_format: str
    settings_sections: dict[str, list[str]]
    settings_kept: list[str]


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
        voxel_size_um: Voxel size it was made with, ``(z, y, x)`` in µm.
        label_fractions: Fraction of the volume per label key.
        meshes: Formats ready to download, by label key (labels without a mesh omitted).
        quantified: Whether its vessel network was measured (summary and centrelines ready).
    """

    id: str
    source: str
    pybones_version: str
    shape_zyx: tuple[int, int, int]
    voxel_size_um: tuple[float, float, float]
    label_fractions: dict[str, float]
    meshes: dict[str, list[str]]
    quantified: bool


class SettingsRequest(Body):
    """Base of the requests that may change settings for one job.

    Attributes:
        overrides: ``section.key=value`` changes to the server's configuration, limited to
            :attr:`SECTIONS`, outside :attr:`KEPT`.
    """

    #: Configuration sections this kind of job may change.
    SECTIONS: ClassVar[frozenset[str]] = frozenset()
    #: Paths inside them that stay the server's (file names the server must find again).
    KEPT: ClassVar[frozenset[str]] = frozenset()

    overrides: tuple[str, ...] = ()

    @field_validator("overrides")
    @classmethod
    def _overridable(cls, overrides: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse overrides of sections this kind of job does not use, or of kept paths.

        Args:
            overrides: Requested overrides.

        Returns:
            ``overrides``, unchanged.

        Raises:
            ValueError: If an override targets another section, or a kept path.
        """
        refused = [item for item in overrides if override_section(item) not in cls.SECTIONS]
        if refused:
            raise ValueError(f"only {sorted(cls.SECTIONS)} can be overridden: {refused}")
        kept = [item for item in overrides if is_kept(override_path(item), cls.KEPT)]
        if kept:
            raise ValueError(f"{sorted(cls.KEPT)} stay the server's: {kept}")
        return overrides


def is_kept(path: str, kept: "Iterable[str]") -> bool:
    """Tell whether a setting is, or lies inside, a kept path.

    Args:
        path: Dotted path of a setting.
        kept: Kept dotted paths.

    Returns:
        Whether it is one of them or below one.
    """
    return any(path == k or path.startswith(f"{k}.") for k in kept)


class SegmentationRequest(SettingsRequest):
    """``POST /api/segmentations``: segment a dataset into a new segmentation.

    Attributes:
        dataset: Id of the stack to segment.
        name: Id of the new segmentation (must not exist yet).
    """

    SECTIONS: ClassVar[frozenset[str]] = OVERRIDABLE_SECTIONS

    dataset: EntryId
    name: EntryId


class MeshRequest(SettingsRequest):
    """``POST .../meshes/{label}`` (optional body): mesh settings for this mesh only."""

    SECTIONS: ClassVar[frozenset[str]] = MESH_SECTIONS


class QuantificationRequest(SettingsRequest):
    """``POST .../quantification`` (optional body): quantification settings for this run only."""

    SECTIONS: ClassVar[frozenset[str]] = QUANTIFICATION_SECTIONS
    KEPT: ClassVar[frozenset[str]] = frozenset({"quantification.files"})


#: The request of each kind of job (``GET /api/server`` describes their sections).
SETTINGS_REQUESTS: dict[str, type[SettingsRequest]] = {
    SEGMENTATION_JOB: SegmentationRequest,
    MESH_JOB: MeshRequest,
    QUANTIFICATION_JOB: QuantificationRequest,
}


class NetworkView(Body):
    """``GET .../quantification``: the network's summary and the settings it was measured with.

    Attributes:
        summary: Branches, junctions, ends, lengths and medians for the whole stack.
        settings: The quantification settings of that measurement.
    """

    summary: NetworkSummary
    settings: QuantificationConfig


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
