"""Phantom configuration (``config.json`` validated by pydantic)."""

import json
from pathlib import Path
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pybones.segmentation.labels import FINAL_LABELS
from pybones.utils.constants import UINT8_MAX, UINT16_MAX

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("config.json")
_PAIR = 2

Positive = Annotated[float, Field(gt=0.0, le=1e5)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Section(BaseModel):
    """Base of every section: unknown keys are rejected and instances are immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class _Range(_Section):
    """Base for sections holding ``[low, high]`` ranges (validated as ordered)."""

    @model_validator(mode="after")
    def _ordered_ranges(self) -> Self:
        """Check that every ``(low, high)`` pair is ordered.

        Returns:
            The validated section.

        Raises:
            ValueError: If a range has ``low > high``.
        """
        for name, value in self:
            if isinstance(value, tuple) and len(value) == _PAIR and value[0] > value[1]:
                raise ValueError(f"{name}: low > high")
        return self


class IntensitySettings(_Section):
    """Grey levels, calibrated on the reference scan (class medians, noise)."""

    background: int = Field(ge=0, le=UINT16_MAX)
    bone: int = Field(ge=0, le=UINT16_MAX)
    vessel: int = Field(ge=0, le=UINT16_MAX)
    vessel_spread: Unit = Field(description="Relative spread of per-vessel contrast filling.")
    noise_std: float = Field(ge=0.0, le=UINT16_MAX, description="Noise std at background level.")
    psf_sigma_um: float = Field(ge=0.0, le=1000.0, description="Scanner blur (Gaussian sigma).")


class CortexSettings(_Range):
    """Cortical walls: thick slabs spanning the volume."""

    count: int = Field(ge=0, le=10)
    thickness_um: tuple[Positive, Positive]


class TrabeculaeSettings(_Range):
    """Trabecular plates (discs) and rods (cylinders)."""

    plates: int = Field(ge=0, le=10_000)
    plate_thickness_um: tuple[Positive, Positive]
    plate_radius_um: tuple[Positive, Positive]
    rods: int = Field(ge=0, le=10_000)
    rod_radius_um: tuple[Positive, Positive]
    rod_length_um: tuple[Positive, Positive]


class VesselSettings(_Range):
    """Barium-filled vessels: smooth random walks of tube segments."""

    count: int = Field(ge=0, le=100_000)
    radius_um: tuple[Positive, Positive]
    segments: tuple[int, int]
    segment_length_um: Positive
    turn_std_deg: float = Field(ge=0.0, le=180.0)
    clot_probability: Unit = Field(description="Segments only partially filled (dim).")
    clot_fill: tuple[Unit, Unit] = Field(description="Contrast fraction of clotted segments.")
    gap_probability: Unit = Field(description="Segments with no contrast agent at all.")
    branch_probability: Unit = Field(
        description="Chance that a segment ends in a bifurcation (0: unbranched walks, the "
        + "segmentation phantom; > 0: vessel trees, the network phantom)."
    )
    branch_angle_deg: tuple[Positive, Positive] = Field(
        description="Angle between a branch and its parent's direction."
    )
    branch_radius_ratio: tuple[Unit, Unit] = Field(
        description="Radius of a branch relative to its parent (branches thinner than the "
        + "smallest vessel radius are not drawn)."
    )
    branch_depth: int = Field(ge=0, le=10, description="Generations of branches below a vessel.")


class BurstSettings(_Range):
    """Burst clusters: speckled spheres of leaked contrast agent."""

    count: int = Field(ge=0, le=1000)
    radius_um: tuple[Positive, Positive]
    speckle_radius_um: tuple[Positive, Positive]
    speckle_fill: Unit = Field(description="Fraction of the sphere covered by speckles.")
    haze_fill: Unit = Field(description="Diffuse contrast inside the sphere (fraction).")


class FilesSettings(_Section):
    """Output names inside the phantom directory."""

    slices_dir: Annotated[str, Field(min_length=1, pattern=r"^[^/\\]+$")]
    slice_template: Annotated[str, Field(pattern=r"\{index[^}]*\}.*\.png$")]
    truth: Annotated[str, Field(pattern=r"\.npz$")]


class ScoreSettings(_Section):
    """Evaluation against the ground truth."""

    radius_bins_um: tuple[Annotated[float, Field(ge=0.0)], ...] = Field(
        min_length=2, description="Vessel recall is reported per radius bin."
    )

    @model_validator(mode="after")
    def _increasing(self) -> Self:
        """Check that the radius bin edges are strictly increasing.

        Returns:
            The validated section.

        Raises:
            ValueError: If the edges are not strictly increasing.
        """
        if list(self.radius_bins_um) != sorted(set(self.radius_bins_um)):
            raise ValueError("radius_bins_um must be strictly increasing")
        return self


class PhantomSettings(_Section):
    """Root of ``tools/phantom/config.json``."""

    shape_zyx: tuple[int, int, int] = Field(description="Volume shape (slices, rows, columns).")
    seed: int = Field(ge=0)
    intensity: IntensitySettings
    cortex: CortexSettings
    trabeculae: TrabeculaeSettings
    vessels: VesselSettings
    bursts: BurstSettings
    excluded_value: int = Field(ge=0, le=UINT8_MAX, description="Truth value ignored in scores.")
    files: FilesSettings
    score: ScoreSettings

    @model_validator(mode="after")
    def _excluded_is_not_a_label(self) -> Self:
        """Check that the excluded truth value cannot be mistaken for a label.

        Returns:
            The validated settings.

        Raises:
            ValueError: If ``excluded_value`` is a final label value.
        """
        if self.excluded_value in {int(lab) for lab in FINAL_LABELS}:
            raise ValueError("excluded_value collides with a pybones label")
        return self


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> PhantomSettings:
    """Read and validate the phantom configuration.

    Args:
        path: JSON settings file (default: the one next to this module).

    Returns:
        The validated settings.
    """
    return PhantomSettings.model_validate(json.loads(path.read_text()))
