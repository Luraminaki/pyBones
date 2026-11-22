"""Pydantic schema of the pyBones JSON configuration.

The schema defines types and allowed ranges only; every value lives in the JSON
file (``default.json`` ships with the package). All sections are strict: unknown
keys are rejected and models are immutable.
"""

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator

from pybones.config.fields import (
    Area,
    ColorBGR,
    FileName,
    Fraction,
    Length,
    Micrometres,
    Section,
    Volume,
)
from pybones.config.quantification import QuantificationConfig
from pybones.segmentation.labels import FOREGROUND_LABELS, Label, label_keys
from pybones.segmentation.stages import MESH_STAGES, STAGES_3D, Stage

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
    file_level: LogLevel = Field(description="Lowest level written to the log file.")
    console_level: LogLevel = Field(description="Lowest level shown in the terminal.")
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


class BoneSideConfig(Section):
    """Vessel edges where a vessel touches bone (off by default: unconfirmed on real scans)."""

    enabled: bool = Field(
        description="Measure a vessel's half-maximum edge against bone on the bone side of a "
        + "contact, instead of against the darkest material around the vessel. Better on the "
        + "phantom; on a real scan it may turn thin dim vessels next to bone into bone, which no "
        + "annotation can yet confirm or refute."
    )
    barrier_um: Length = Field(
        description="Only bone that holds a disc of this radius counts as the bone side (thin "
        + "foreground, such as a dim vessel, does not)."
    )


class VesselConfig(Section):
    """Vessel detection (physical units)."""

    rim_occupancy: Fraction = Field(
        description="Contrast-agent fraction at which a partial-volume voxel (agent + soft "
        + "tissue) counts as vessel; 0.5 = half-maximum edge."
    )
    rim_um: Length = Field(description="Max rim growth around bright cores (partial volume).")
    min_area_um2: Area = Field(description="Smaller 2D vessel blobs are noise.")
    bone_side: BoneSideConfig


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
        description="In-plane Gaussian scale of the fine bright-pixel density that places the "
        + "cluster boundary; along z it is averaged over the box's slices (a 3D density: a "
        + "vessel lying in the slice is a cylinder, not a dense strip)."
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

    enabled: bool = Field(
        description="Compute it (only with the refinement; off: no plate veto, faster)."
    )
    scales_um: tuple[Annotated[float, Field(gt=0.0, le=1e4)], ...] = Field(
        min_length=1, description="Gaussian scales (about structure radius / sqrt 2)."
    )


class ZSmoothingConfig(Section):
    """Sliding majority vote along z."""

    window_um: Length = Field(description="z extent of the vote (<= one slice: off).")


class RefinementConfig(Section):
    """3D connectivity refinement (physical units)."""

    enabled: bool = Field(
        description="Decide the ambiguous blobs in 3D (off: they become bone, slice by slice)."
    )
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
    """Surface extraction and mesh files."""

    downsample: Annotated[
        dict[str, Annotated[int, Field(ge=1, le=_MAX_DOWNSAMPLE)]],
        _keyed_by(FOREGROUND_LABELS),
    ] = Field(
        description="Grid of the surface, per label: 2 averages 2x2x2 voxel blocks (details "
        + "smaller than a block are smoothed); 1 is full resolution, about 4 times more triangles."
    )
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
    quantification: QuantificationConfig
    render: RenderConfig
