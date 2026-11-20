"""Pydantic schema of the pyBones JSON configuration.

The schema defines types and allowed ranges only; every value lives in the JSON
file (``default.json`` ships with the package). All sections are strict: unknown
keys are rejected and models are immutable.
"""

from typing import Annotated, ClassVar, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from pybones.meshing.formats import MeshFormat
from pybones.segmentation.labels import FOREGROUND_LABELS, Label, label_keys
from pybones.segmentation.stages import MESH_STAGES, STAGES_3D, Stage
from pybones.utils.constants import UINT8_MAX

#: Largest mesh downsampling factor: factor³ must fit the uint16 block counters of the
#: occupancy grid.
_MAX_DOWNSAMPLE = 40
_COLORED_LABELS = tuple(lab for lab in Label if lab != Label.BACKGROUND)


def _keyed_by(labels: tuple[Label, ...]) -> AfterValidator:
    """Validator requiring a mapping to have exactly one entry per label of ``labels``.

    Args:
        labels: Labels whose keys the mapping must hold, no more, no less.

    Returns:
        A pydantic ``AfterValidator`` for ``dict[str, ...]`` fields.
    """

    def check(mapping: dict[str, object]) -> dict[str, object]:
        """Check the keys of ``mapping``.

        Args:
            mapping: Validated mapping.

        Returns:
            ``mapping``, unchanged.

        Raises:
            ValueError: If the keys differ from the expected label keys.
        """
        expected = set(label_keys(labels))
        if set(mapping) != expected:
            raise ValueError(f"keys must be exactly {sorted(expected)}, got {sorted(mapping)}")
        return mapping

    return AfterValidator(check)


#: Physical quantities: generous but finite ranges (1 m, 1 m², 1 m³ as upper bounds).
Micrometres = Annotated[float, Field(gt=0.0, le=1e6)]
Length = Annotated[float, Field(ge=0.0, le=1e6)]
Area = Annotated[float, Field(ge=0.0, le=1e12)]
Volume = Annotated[float, Field(ge=0.0, le=1e18)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Channel = Annotated[int, Field(ge=0, le=UINT8_MAX)]
ColorBGR = tuple[Channel, Channel, Channel]
#: Characters that would turn a file name into a path.
PATH_SEPARATORS = ("/", "\\")


def _plain_name(name: str) -> str:
    """Check that a configured file name stays inside its folder.

    A validator rather than a ``pattern``: a field adding its own ``pattern`` (e.g. an
    extension) would replace this one instead of adding to it.

    Args:
        name: Configured file name.

    Returns:
        ``name``, unchanged.

    Raises:
        ValueError: If it is empty or holds a path separator.
    """
    if not name or any(sep in name for sep in PATH_SEPARATORS):
        raise ValueError(f"must be a plain file name (no {' or '.join(PATH_SEPARATORS)}): {name!r}")
    return name


FileName = Annotated[str, AfterValidator(_plain_name)]


class Section(BaseModel):
    """Base of every configuration section: strict and immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class IOConfig(Section):
    """File names and slice discovery rules."""

    slice_extensions: tuple[Annotated[str, Field(pattern=r"^\.\w+$")], ...] = Field(min_length=1)
    ignored_name_prefixes: tuple[Annotated[str, Field(min_length=1)], ...]
    ignored_directories: tuple[Annotated[str, Field(min_length=1)], ...]
    labels_file: Annotated[FileName, Field(pattern=r"\.npy$")]
    meta_file: Annotated[FileName, Field(pattern=r"\.json$")]
    scratch_file_template: Annotated[FileName, Field(pattern=r"\{stage\}.*\.npy$")]
    mask_png_template: Annotated[FileName, Field(pattern=r"\{label\}.*\{index[^}]*\}.*\.png$")]


class PipelineConfig(Section):
    """Parallelism of the per-slice stage, and progress estimation."""

    workers: int = Field(ge=1, le=256, description="Threads (1 = sequential).")
    extra_in_flight: int = Field(ge=0, le=256, description="Queued slices beyond `workers`.")
    stage_weights: dict[Stage, Annotated[float, Field(gt=0.0, le=1.0)]] = Field(
        description="Relative duration of each stage, for the overall progress fraction "
        + "(measured on the reference stack; normalised over the stages that run)."
    )

    @model_validator(mode="after")
    def _every_stage_weighted(self) -> Self:
        """Check that every segmentation stage has a weight (other jobs bring their own).

        Returns:
            The validated section.

        Raises:
            ValueError: If a segmentation stage has no weight.
        """
        missing = sorted(set(STAGES_3D) - set(self.stage_weights))
        if missing:
            raise ValueError(f"stage_weights misses {missing}")
        return self


#: Names of the standard logging levels accepted in the configuration.
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(Section):
    """Log files (one per process, rotated by size) and console output."""

    directory: Annotated[str, Field(min_length=1)] = Field(
        description="Directory of the log files (relative to the working directory)."
    )
    file_level: LogLevel
    console_level: LogLevel
    max_bytes: int = Field(ge=1024, le=1 << 30, description="Size at which a log file rotates.")
    backup_count: int = Field(ge=0, le=100, description="Rotated files kept per process.")
    max_files: int = Field(
        ge=1, le=10_000, description="Log files kept per program; older runs' files are deleted."
    )


class ThresholdConfig(Section):
    """Global multi-Otsu threshold estimation."""

    histogram_stride: int = Field(ge=1, le=10_000, description="Use one slice every N.")
    histogram_bins: int = Field(ge=16, le=4096, description="Otsu search resolution.")


class JobsConfig(Section):
    """Background jobs (see :mod:`pybones.api.jobs`)."""

    max_concurrent: int = Field(
        ge=1, le=64, description="Jobs run at once (each one also uses `pipeline.workers` threads)."
    )
    history_size: int = Field(ge=1, le=10_000, description="Finished jobs remembered.")
    event_interval_s: float = Field(
        ge=0.0, le=60.0, description="Least time between two progress events of a job."
    )
    event_queue_size: int = Field(
        ge=1, le=10_000, description="Events buffered per subscriber (oldest dropped first)."
    )


class AcquisitionConfig(Section):
    """Scan geometry shared by segmentation (µm -> pixels) and meshing."""

    voxel_size_um: tuple[Micrometres, Micrometres, Micrometres] = Field(
        description="Voxel size along (z, y, x) in µm."
    )


class VesselConfig(Section):
    """Vessel detection (physical units)."""

    rim_occupancy: Fraction = Field(
        description="Contrast-agent fraction at which a partial-volume voxel (agent + soft "
        + "tissue) counts as vessel; 0.5 = half-maximum edge."
    )
    rim_um: Length = Field(description="Max rim growth around bright cores (partial volume).")
    min_area_um2: Area = Field(description="Smaller 2D vessel blobs are noise.")


class BoneConfig(Section):
    """Bone detection (physical units)."""

    min_area_um2: Area = Field(description="Smaller 2D bone blobs are noise.")


class BurstConfig(Section):
    """Burst-vessel (leaked contrast agent) detection (physical units)."""

    window_xy_um: Length = Field(description="In-plane side of the bright-density box.")
    density_grid_um: Micrometres = Field(
        description="Block size of the grid the density is stored on (much smaller than "
        + "the box; saves memory along the 3D window)."
    )
    window_z_um: Length = Field(description="z extent of the box (<= one slice: 2D only).")
    density: float = Field(gt=0.0, le=1.0, description="Min bright-pixel density of a core.")
    min_area_um2: Area = Field(description="Min area of a burst core.")
    boundary_sigma_um: Length = Field(
        description="Scale of the fine bright-pixel density that places the cluster boundary."
    )
    boundary_fraction: Fraction = Field(
        description="The boundary is where the fine density falls to this fraction of the "
        + "core density (0.5 = half-maximum)."
    )


class SegmentationConfig(Section):
    """Per-slice classification (physical units)."""

    smoothing_sigma_um: Length = Field(description="Gaussian denoising sigma (0 = off).")
    connectivity: Literal[4, 8] = Field(description="In-plane pixel connectivity.")
    small_blob_max_area_um2: Area = Field(
        description="Bone-intensity blobs smaller than this are ambiguous in 2D (rod "
        + "cross-section or dim vessel; decided in 3D); larger ones are bone.",
    )
    vessel: VesselConfig
    bone: BoneConfig
    burst: BurstConfig


class TubularityConfig(Section):
    """Multi-scale 3D Hessian tube-vs-plate ratio, used to veto plate-like vessels."""

    enabled: bool
    scales_um: tuple[Annotated[float, Field(gt=0.0, le=1e4)], ...] = Field(
        min_length=1, description="Gaussian scales (about structure radius / sqrt 2)."
    )


class ZSmoothingConfig(Section):
    """Sliding majority vote along z."""

    window_um: Length = Field(description="z extent of the vote (<= one slice: off).")


class RefinementConfig(Section):
    """3D connectivity refinement (physical units)."""

    enabled: bool
    vessel_contact_ratio: float = Field(
        gt=0.0,
        le=1000.0,
        description="Ambiguous component becomes vessel if "
        + "vessel contacts > ratio x bone contacts.",
    )
    min_tubularity: Fraction = Field(
        description="Ambiguous components flatter than this mean tube-vs-plate ratio "
        + "(|l2|/|l3|: ~1 tube, ~0 plate) cannot be vessels (needs tubularity.enabled)."
    )
    min_isolated_vessel_um3: Volume = Field(
        description="Isolated ambiguous components smaller than this are noise."
    )
    min_vessel_um3: Volume = Field(description="Smaller 3D vessels are noise.")
    min_bone_um3: Volume = Field(description="Smaller 3D bone pieces are noise.")


class TaubinConfig(Section):
    """Taubin λ|μ smoothing of extracted surfaces (removes the voxel staircase)."""

    iterations: int = Field(ge=0, le=1000, description="λ|μ passes (0 = no smoothing).")
    lambda_factor: float = Field(gt=0.0, le=1.0, description="Shrinking step (λ > 0).")
    mu_factor: float = Field(ge=-1.0, lt=0.0, description="Inflating step (μ < -λ).")

    @model_validator(mode="after")
    def _taubin_condition(self) -> Self:
        """Check Taubin's non-shrinking condition.

        Returns:
            The validated section.

        Raises:
            ValueError: If ``mu_factor >= -lambda_factor``.
        """
        if self.mu_factor >= -self.lambda_factor:
            raise ValueError("Taubin smoothing needs mu_factor < -lambda_factor")
        return self


class MeshConfig(Section):
    """Surface extraction and mesh files.

    ``downsample`` sets the grid the surface is built on: a factor of 2 averages blocks
    of 2x2x2 voxels (a 6 µm grid at 3 µm voxels), so details smaller than a block are
    smoothed; 1 is full resolution, with about 4 times more triangles.
    """

    downsample: Annotated[
        dict[str, Annotated[int, Field(ge=1, le=_MAX_DOWNSAMPLE)]],
        _keyed_by(FOREGROUND_LABELS),
    ]
    step_size: int = Field(ge=1, le=50, description="Marching-cubes step (coarser if larger).")
    iso_level: float = Field(gt=0.0, lt=1.0, description="Occupancy level of the surface.")
    chunk_faces: int = Field(ge=1024, le=10**9, description="Triangles per write chunk.")
    glb_position_bits: Literal[16, 32] = Field(
        description="GLB positions: 16-bit integers over the bounding box (error <= half of "
        + "its size / 65535, 8-bit normals) or 32-bit floats. STL is always 32-bit."
    )
    stage_weights: dict[Stage, Annotated[float, Field(gt=0.0, le=1.0)]] = Field(
        description="Relative duration of each meshing stage, for the progress of a mesh job "
        + "(measured on the vessels and the bone of the reference stack)."
    )
    taubin: TaubinConfig

    @model_validator(mode="after")
    def _mesh_stages_weighted(self) -> Self:
        """Check that the weights are exactly those of the meshing stages.

        Returns:
            The validated section.

        Raises:
            ValueError: If a meshing stage has no weight, or another stage has one.
        """
        if set(self.stage_weights) != set(MESH_STAGES):
            raise ValueError(f"stage_weights must weight exactly {[s.value for s in MESH_STAGES]}")
        return self

    def factor_for(self, label: Label) -> int:
        """Downsampling factor configured for ``label``.

        Args:
            label: Foreground label to mesh.

        Returns:
            Integer block size of the occupancy grid.

        Raises:
            KeyError: If no factor is configured for ``label``.
        """
        return self.downsample[label.key]


class RenderConfig(Section):
    """2D overlays."""

    alpha: Fraction
    low_percentile: float = Field(ge=0.0, le=100.0)
    high_percentile: float = Field(ge=0.0, le=100.0)
    colors_bgr: Annotated[dict[str, ColorBGR], _keyed_by(_COLORED_LABELS)]

    @model_validator(mode="after")
    def _ordered_percentiles(self) -> Self:
        """Check that the contrast-stretch percentiles are ordered.

        Returns:
            The validated section.

        Raises:
            ValueError: If ``low_percentile >= high_percentile``.
        """
        if self.low_percentile >= self.high_percentile:
            raise ValueError("low_percentile must be < high_percentile")
        return self


class PyBonesConfig(Section):
    """Root of the configuration file."""

    io: IOConfig
    logging: LoggingConfig
    jobs: JobsConfig
    acquisition: AcquisitionConfig
    pipeline: PipelineConfig
    thresholds: ThresholdConfig
    segmentation: SegmentationConfig
    tubularity: TubularityConfig
    z_smoothing: ZSmoothingConfig
    refinement: RefinementConfig
    mesh: MeshConfig
    render: RenderConfig


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
