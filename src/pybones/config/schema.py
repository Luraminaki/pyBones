"""Pydantic schema of the pyBones JSON configuration.

The schema defines types and allowed ranges only; every value lives in the JSON
file (``default.json`` ships with the package). All sections are strict: unknown
keys are rejected and models are immutable.
"""

from typing import Annotated, ClassVar, Literal, Self

from pydantic import AfterValidator, ConfigDict, Field, model_validator

from pybones.config.fields import (
    EFFECT_TEXT,
    EFFECTS_KEY,
    ColorBGR,
    Effect,
    FileName,
    Fraction,
    Level,
    Micrometres,
    Section,
    advice,
    section_config,
)
from pybones.config.quantification import QuantificationConfig
from pybones.config.segmentation import (
    RefinementConfig,
    SegmentationConfig,
    ThinVesselsConfig,
    TubularityConfig,
    ZSmoothingConfig,
)
from pybones.segmentation.labels import FOREGROUND_LABELS, Label, label_keys
from pybones.segmentation.stages import MESH_STAGES, STAGES_3D, THIN_STAGES, Stage

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


class IOConfig(Section):
    """File names and slice discovery rules."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.NONE, Level.ADVANCED)

    slice_extensions: tuple[Annotated[str, Field(pattern=r"^\.\w+$")], ...] = Field(
        min_length=1, description="Slice file extensions, in lower case (files may use upper case)."
    )
    ignored_name_prefixes: tuple[Annotated[str, Field(min_length=1)], ...] = Field(
        description="Files whose name starts with one of these are skipped (e.g. macOS `._`)."
    )
    ignored_directories: tuple[Annotated[str, Field(min_length=1)], ...] = Field(
        description="Folders skipped inside a stack (e.g. `__MACOSX` in a zip)."
    )
    labels_file: Annotated[FileName, Field(pattern=r"\.npy$")] = Field(
        description="Label volume of a segmentation (uint8 NumPy array): a file name ending "
        + "in `.npy`."
    )
    meta_file: Annotated[FileName, Field(pattern=r"\.json$")] = Field(
        description="Metadata of a segmentation (source, thresholds, timings, configuration): a "
        + "file name ending in `.json`."
    )
    scratch_file_template: Annotated[FileName, Field(pattern=r"\{stage\}.*\.npy$")] = Field(
        description="Intermediate volumes of a run, one per `{stage}` (deleted at the end): a file "
        + "name ending in `.npy`."
    )
    mask_png_template: Annotated[FileName, Field(pattern=r"\{label\}.*\{index[^}]*\}.*\.png$")] = (
        Field(
            description="Image sequence written by `pybones export`, one PNG per slice: a file "
            + "name with `{label}` and `{index}` (a format, e.g. `{index:04d}`), ending in `.png`."
        )
    )


class PipelineConfig(Section):
    """Parallelism of the per-slice stage, and progress estimation."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.NONE, Level.ADVANCED)

    workers: int = Field(
        ge=1,
        le=256,
        description="Threads (1 = sequential).",
        json_schema_extra=advice(
            "Lower it to use less memory or leave cores to other work; more than the cores "
            + "gains nothing."
        ),
    )
    extra_in_flight: int = Field(
        ge=0,
        le=256,
        description="Queued slices beyond `workers`.",
        json_schema_extra=advice("Rarely: more keeps the threads busier and uses more memory."),
    )
    stage_weights: dict[Stage, Annotated[float, Field(gt=0.0, le=1.0)]] = Field(
        description="Relative duration of each stage, for the overall progress fraction "
        + "(measured on the reference stack; normalised over the stages that run).",
        json_schema_extra=advice(
            "Only if the progress bar runs unevenly on another machine; "
            + "`python -m tools.report figures` measures them."
        ),
    )

    @model_validator(mode="after")
    def _every_stage_weighted(self) -> Self:
        """Check that every segmentation stage has a weight (other jobs bring their own).

        Returns:
            The validated section.

        Raises:
            ValueError: If a segmentation stage has no weight.
        """
        missing = sorted({*STAGES_3D, *THIN_STAGES} - set(self.stage_weights))
        if missing:
            raise ValueError(f"stage_weights misses {missing}")
        return self


#: Names of the standard logging levels accepted in the configuration.
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(Section):
    """Log files (one per process, rotated by size) and console output."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.NONE, Level.ADVANCED)

    directory: Annotated[str, Field(min_length=1)] = Field(
        description="Directory of the log files (relative to the working directory)."
    )
    file_level: LogLevel = Field(description="Lowest level written to the log file.")
    console_level: LogLevel = Field(description="Lowest level shown in the terminal.")
    max_bytes: int = Field(ge=1024, le=1 << 30, description="Size at which a log file rotates.")
    backup_count: int = Field(ge=0, le=100, description="Rotated files kept per process.")
    max_files: int = Field(
        ge=1, le=10_000, description="Log files kept per program; older runs' files are deleted."
    )


class ThresholdConfig(Section):
    """Global multi-Otsu threshold estimation."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS, Level.ADVANCED)

    histogram_stride: int = Field(
        ge=1,
        le=10_000,
        description="Use one slice every N.",
        json_schema_extra=advice(
            "Lower it if the stack's brightness varies along z and few slices are sampled; "
            + "raise it to start faster on a deep stack."
        ),
    )
    histogram_bins: int = Field(
        ge=16,
        le=4096,
        description="Otsu search resolution.",
        json_schema_extra=advice("Rarely: finer bins place the thresholds more precisely."),
    )


class JobsConfig(Section):
    """Background jobs (see :mod:`pybones.api.jobs`)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.NONE, Level.ADVANCED)

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

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS)

    voxel_size_um: tuple[Micrometres, Micrometres, Micrometres] = Field(
        description="Voxel size along (z, y, x) in µm.",
        json_schema_extra=advice(
            "Set it to the scan's real voxel size: every size in µm, and the meshes, follow it."
        ),
    )


class TaubinConfig(Section):
    """Taubin λ|μ smoothing of extracted surfaces (removes the voxel staircase)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.MESHES)

    iterations: int = Field(
        ge=0,
        le=1000,
        description="λ|μ passes (0 = no smoothing).",
        json_schema_extra=advice(
            "0 shows the raw voxel staircase; more smooths more, and rounds small details."
        ),
    )
    lambda_factor: float = Field(
        gt=0.0,
        le=1.0,
        description="Shrinking step (λ > 0).",
        json_schema_extra=advice("Rarely: λ and μ together keep the volume (Taubin)."),
    )
    mu_factor: float = Field(
        ge=-1.0,
        lt=0.0,
        description="Inflating step (μ < -λ).",
        json_schema_extra=advice("Rarely: keep it slightly beyond -λ."),
    )

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
    """Surface extraction and mesh files."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.MESHES)

    downsample: Annotated[
        dict[str, Annotated[int, Field(ge=1, le=_MAX_DOWNSAMPLE)]],
        _keyed_by(FOREGROUND_LABELS),
    ] = Field(
        description="Grid of the surface, per label: 2 averages 2x2x2 voxel blocks (details "
        + "smaller than a block are smoothed); 1 is full resolution, about 4 times more triangles.",
        json_schema_extra=advice(
            "1 for the finest surface of one label (files 4 times larger, a slower browser); 3 "
            + "or 4 for a quick look at the bone."
        ),
    )
    step_size: int = Field(
        ge=1,
        le=50,
        description="Marching-cubes step (coarser if larger).",
        json_schema_extra=advice("Rarely: the grid above coarsens more smoothly."),
    )
    iso_level: float = Field(
        gt=0.0,
        lt=1.0,
        description="Occupancy level of the surface.",
        json_schema_extra=advice(
            "Rarely: 0.5 puts the surface halfway between label and background; lower draws "
            + "the surface outside, higher inside."
        ),
    )
    chunk_faces: int = Field(
        ge=1024,
        le=10**9,
        description="Triangles per write chunk.",
        json_schema_extra=advice(
            "Lower it if writing a large STL runs short of memory.", Effect.NONE
        ),
    )
    glb_position_bits: Literal[16, 32] = Field(
        description="GLB positions: 16-bit integers over the bounding box (error <= half of "
        + "its size / 65535, 8-bit normals) or 32-bit floats. STL is always 32-bit.",
        json_schema_extra=advice(
            "32 for exact positions in other software (files a third larger); 16 is far below "
            + "a voxel."
        ),
    )
    stage_weights: dict[Stage, Annotated[float, Field(gt=0.0, le=1.0)]] = Field(
        description="Relative duration of each meshing stage, for the progress of a mesh job "
        + "(measured on the vessels and the bone of the reference stack).",
        json_schema_extra=advice(
            "Only if the progress bar runs unevenly; `python -m tools.report figures` measures "
            + "them.",
            Effect.NONE,
        ),
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

    model_config: ClassVar[ConfigDict] = section_config(Effect.PREVIEWS)

    alpha: Fraction = Field(description="Opacity of the label colours over the slice.")
    low_percentile: float = Field(
        ge=0.0, le=100.0, description="Intensity percentile shown black (contrast stretch)."
    )
    high_percentile: float = Field(
        ge=0.0, le=100.0, description="Intensity percentile shown white (above low_percentile)."
    )
    colors_bgr: Annotated[dict[str, ColorBGR], _keyed_by(_COLORED_LABELS)] = Field(
        description="Colour of each label, as (blue, green, red) 0-255."
    )

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
    """Root of the configuration (``default.json``), recorded by every segmentation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={EFFECTS_KEY: {e.value: text for e, text in EFFECT_TEXT.items()}},
    )

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
    thin_vessels: ThinVesselsConfig
    mesh: MeshConfig
    quantification: QuantificationConfig
    render: RenderConfig
