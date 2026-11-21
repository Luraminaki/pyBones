"""Schema of the ``quantification`` section: backends, pruning, files, progress weights."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from pybones.config.fields import FileName, Length, Micrometres, Section
from pybones.quantification.protocols import CavityBackend, DistanceBackend, SkeletonBackend
from pybones.segmentation.stages import QUANTIFY_STAGES, Stage


class QuantificationBackends(Section):
    """Implementation of each step of the quantification (interchangeable, same results)."""

    cavities: CavityBackend = Field(description="Filling the cavities of the vessel mask.")
    skeleton: SkeletonBackend = Field(description="Thinning the vessels to centrelines.")
    distances: DistanceBackend = Field(description="Distance to the wall (the calibre).")


class PruningConfig(Section):
    """When a piece of the centreline network is an artefact (see quantification.pruning)."""

    spur_radius_factor: float = Field(
        ge=0.0,
        le=100.0,
        description="A free-end branch shorter than this many vessel radii "
        + "at its fork is a bump of the surface.",
    )
    min_spur_um: Length = Field(description="... and never shorter than this.")
    min_fragment_um: Length = Field(description="Isolated branches shorter than this are specks.")
    loop_factor: float = Field(
        ge=0.0,
        le=100.0,
        description="A loop shorter than this many vessel circumferences is a hole or a fold "
        + "of the vessel mask: its longest branch is removed (0: loops kept).",
    )
    junction_merge_factor: float = Field(
        ge=0.0,
        le=100.0,
        description="Two forks joined by a branch shorter than this many "
        + "times their radii together are one fork.",
    )


class QuantificationFiles(Section):
    """Files the quantification writes into the segmentation folder."""

    network_file: Annotated[FileName, Field(pattern=r"\.json$")] = Field(
        description="Summary and node table."
    )
    branches_file: Annotated[FileName, Field(pattern=r"\.csv$")] = Field(
        description="One row per branch."
    )
    centrelines_file: Annotated[FileName, Field(pattern=r"\.npz$")] = Field(
        description="Centreline voxels, their branch, node and wall distance."
    )


class QuantificationConfig(Section):
    """Vascular quantification: centrelines, network, per-branch measures."""

    backends: QuantificationBackends
    pruning: PruningConfig
    anchor_margin_um: Length = Field(
        description="An end this much beyond its own radius from a face of the stack, or "
        + "from a burst, continues there (it is not a free end)."
    )
    files: QuantificationFiles
    stage_weights: dict[Stage, Annotated[float, Field(gt=0.0, le=1.0)]] = Field(
        description="Relative duration of each quantification stage (progress bar)."
    )
    max_radius_um: Micrometres = Field(
        description="Largest vessel radius measured (distances are capped there)."
    )
    distance_tile_slices: int = Field(
        ge=1, le=10_000, description="Depth of a tile of the scipy distance backend (memory)."
    )
    distance_tile_pixels: int = Field(
        ge=8, le=100_000, description="In-plane side of a tile of the scipy distance backend."
    )

    @model_validator(mode="after")
    def _quantify_stages_weighted(self) -> Self:
        """Check that the weights are exactly those of the quantification stages.

        Returns:
            The validated section.

        Raises:
            ValueError: If a stage has no weight, or another stage has one.
        """
        if set(self.stage_weights) != set(QUANTIFY_STAGES):
            names = [s.value for s in QUANTIFY_STAGES]
            raise ValueError(f"stage_weights must weight exactly {names}")
        return self
