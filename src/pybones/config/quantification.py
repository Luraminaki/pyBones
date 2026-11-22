"""Schema of the ``quantification`` section: backends, pruning, files, progress weights."""

from typing import Annotated, ClassVar, Self

from pydantic import ConfigDict, Field, model_validator

from pybones.config.fields import (
    Effect,
    FileName,
    Length,
    Level,
    Micrometres,
    Section,
    advice,
    section_config,
)
from pybones.quantification.protocols import CavityBackend, DistanceBackend, SkeletonBackend
from pybones.segmentation.stages import QUANTIFY_STAGES, Stage


class QuantificationBackends(Section):
    """Implementation of each step of the quantification (interchangeable, same results)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.NONE, Level.ADVANCED)

    cavities: CavityBackend = Field(
        description="Filling the cavities of the vessel mask.",
        json_schema_extra=advice("Rarely: both give the same result; `scipy` cross-checks."),
    )
    skeleton: SkeletonBackend = Field(
        description="Thinning the vessels to centrelines.",
        json_schema_extra=advice("Rarely: one backend today."),
    )
    distances: DistanceBackend = Field(
        description="Distance to the wall (the calibre).",
        json_schema_extra=advice(
            "Use `scipy` for voxels that are not square in-plane, or to cross-check (much slower)."
        ),
    )


class PruningConfig(Section):
    """When a piece of the centreline network is an artefact (see quantification.pruning)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.NETWORK)

    spur_radius_factor: float = Field(
        ge=0.0,
        le=100.0,
        description="A free-end branch shorter than this many vessel radii "
        + "at its fork is a bump of the surface.",
        json_schema_extra=advice(
            "Raise it if bumps of thick vessels count as branches; lower it to keep short real "
            + "side branches."
        ),
    )
    min_spur_um: Length = Field(
        description="... and never shorter than this.",
        json_schema_extra=advice("Raise it on a noisier mask, where thin vessels have bumps too."),
    )
    min_fragment_um: Length = Field(
        description="Isolated branches shorter than this are specks.",
        json_schema_extra=advice("Lower it to count short isolated pieces (vessels cut by clots)."),
    )
    loop_factor: float = Field(
        ge=0.0,
        le=100.0,
        description="A loop shorter than this many vessel circumferences is a hole or a fold "
        + "of the vessel mask: its longest branch is removed (0: loops kept).",
        json_schema_extra=advice(
            "Raise it if junctions that are not forks remain around holes; 0 shows the raw loops."
        ),
    )
    junction_merge_factor: float = Field(
        ge=0.0,
        le=100.0,
        description="Two forks joined by a branch shorter than this many "
        + "times their radii together are one fork.",
        json_schema_extra=advice(
            "Raise it to merge more close forks (fewer false junctions, but real forks merged "
            + "too); lower it to keep close forks apart."
        ),
    )


class QuantificationFiles(Section):
    """Files the quantification writes into the segmentation folder."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.NONE, Level.ADVANCED)

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

    model_config: ClassVar[ConfigDict] = section_config(Effect.NETWORK)

    backends: QuantificationBackends
    pruning: PruningConfig
    anchor_margin_um: Length = Field(
        description="An end this much beyond its own radius from a face of the stack, or "
        + "from a burst, continues there (it is not a free end).",
        json_schema_extra=advice("Raise it if ends just short of a face are counted as free."),
    )
    files: QuantificationFiles
    stage_weights: dict[Stage, Annotated[float, Field(gt=0.0, le=1.0)]] = Field(
        description="Relative duration of each quantification stage (progress bar).",
        json_schema_extra=advice(
            "Only if the progress bar runs unevenly; `python -m tools.report figures` measures "
            + "them.",
            Effect.NONE,
        ),
    )
    max_radius_um: Micrometres = Field(
        description="Largest vessel radius measured (distances are capped there).",
        json_schema_extra=advice("Raise it for a scan with thicker vessels than this one."),
    )
    distance_tile_slices: int = Field(
        ge=1,
        le=10_000,
        description="Depth of a tile of the scipy distance backend (memory).",
        json_schema_extra=advice(
            "Lower it if the `scipy` backend runs out of memory.", Effect.NONE
        ),
    )
    distance_tile_pixels: int = Field(
        ge=8,
        le=100_000,
        description="In-plane side of a tile of the scipy distance backend.",
        json_schema_extra=advice(
            "Lower it if the `scipy` backend runs out of memory.", Effect.NONE
        ),
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
