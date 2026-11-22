"""Configuration schema of the segmentation: per-slice classes, tubularity, 3D refinement.

Every section here changes the labels (:attr:`Effect.LABELS`) unless a setting says
otherwise; each setting carries *advice*, when to change it, for the web page's form.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field

from pybones.config.fields import (
    Area,
    Effect,
    Fraction,
    Length,
    Level,
    Micrometres,
    Section,
    Volume,
    advice,
    section_config,
)


class BoneSideConfig(Section):
    """Vessel edges where a vessel touches bone (off by default: unconfirmed on real scans)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS, Level.EXPERIMENTAL)

    enabled: bool = Field(
        description="Measure a vessel's half-maximum edge against bone on the bone side of a "
        + "contact, instead of against the darkest material around the vessel. Better on the "
        + "phantom; on a real scan it may turn thin dim vessels next to bone into bone, which no "
        + "annotation can yet confirm or refute.",
        json_schema_extra=advice(
            "To compare with a run without it, where vessels press on bone: false vessel along "
            + "bone surfaces is what it removes. Keep it off for measurements until an expert "
            + "has checked it."
        ),
    )
    barrier_um: Length = Field(
        description="Only bone that holds a disc of this radius counts as the bone side (thin "
        + "foreground, such as a dim vessel, does not).",
        json_schema_extra=advice(
            "Raise it if thin vessels lying on bone are lost to bone; lower it if thin bone "
            + "plates should count as the bone side too."
        ),
    )


class VesselConfig(Section):
    """Vessel detection (physical units)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS)

    rim_occupancy: Fraction = Field(
        description="Contrast-agent fraction at which a partial-volume voxel (agent + soft "
        + "tissue) counts as vessel; 0.5 = half-maximum edge.",
        json_schema_extra=advice(
            "Rarely: 0.5 is the physical edge. Lower it to draw vessels wider, raise it to "
            + "draw them narrower (diameters move with it)."
        ),
    )
    rim_um: Length = Field(
        description="Max rim growth around bright cores (partial volume).",
        json_schema_extra=advice(
            "Raise it with a wider scanner blur (another scanner or resolution), so that the "
            + "rim of thick vessels is reached."
        ),
    )
    min_area_um2: Area = Field(
        description="Smaller 2D vessel blobs are noise.",
        json_schema_extra=advice(
            "Raise it on a noisier scan (speckle read as vessel); lower it to keep the "
            + "smallest vessel sections."
        ),
    )
    bone_side: BoneSideConfig


class BoneConfig(Section):
    """Bone detection (physical units)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS)

    min_area_um2: Area = Field(
        description="Smaller 2D bone blobs are noise.",
        json_schema_extra=advice("Raise it on a noisier scan; lower it to keep thin trabeculae."),
    )


class BurstConfig(Section):
    """Burst-vessel (leaked contrast agent) detection (physical units)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS)

    window_xy_um: Length = Field(
        description="In-plane side of the bright-density box.",
        json_schema_extra=advice(
            "Scale it with the bursts of the scan: about half of their usual size."
        ),
    )
    density_grid_um: Micrometres = Field(
        description="Block size of the grid the density is stored on (much smaller than "
        + "the box; saves memory along the 3D window).",
        json_schema_extra=advice(
            "Raise it to save memory on a very large stack; results move slightly."
        ),
    )
    window_z_um: Length = Field(
        description="z extent of the box (<= one slice: 2D only).",
        json_schema_extra=advice(
            "Lower it if bursts spread into neighbouring slices where there are none; one "
            + "slice or less makes the detection 2D."
        ),
    )
    density: float = Field(
        gt=0.0,
        le=1.0,
        description="Min bright-pixel density of a core.",
        json_schema_extra=advice(
            "Raise it if dense vessel networks are taken for bursts; lower it if real bursts "
            + "are missed."
        ),
    )
    min_area_um2: Area = Field(
        description="Min area of a burst core.",
        json_schema_extra=advice("Raise it if small vessel clusters become bursts."),
    )
    boundary_sigma_um: Length = Field(
        description="In-plane Gaussian scale of the fine bright-pixel density that places the "
        + "cluster boundary; along z it is averaged over the box's slices (a 3D density: a "
        + "vessel lying in the slice is a cylinder, not a dense strip).",
        json_schema_extra=advice(
            "Lower it for tighter burst edges, which give thick vessels near bursts back; "
            + "too low and the burst's bright haze becomes vessel."
        ),
    )
    boundary_fraction: Fraction = Field(
        description="The boundary is where the fine density falls to this fraction of the "
        + "core density (0.5 = half-maximum).",
        json_schema_extra=advice("Raise it to shrink bursts, lower it to grow them."),
    )


class SegmentationConfig(Section):
    """Per-slice classification (physical units)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS)

    smoothing_sigma_um: Length = Field(
        description="Gaussian denoising sigma (0 = off).",
        json_schema_extra=advice(
            "Raise it on a noisier scan; it also blurs thin vessels, so keep it small."
        ),
    )
    connectivity: Literal[4, 8] = Field(
        description="In-plane pixel connectivity.",
        json_schema_extra=advice(
            "Rarely: 8 links diagonal pixels (thin oblique vessels stay whole); 4 splits more."
        ),
    )
    small_blob_max_area_um2: Area = Field(
        description="Bone-intensity blobs smaller than this are ambiguous in 2D (rod "
        + "cross-section or dim vessel; decided in 3D); larger ones are bone.",
        json_schema_extra=advice(
            "Raise it if dim vessels wider than usual end up as bone; lower it if bone rods "
            + "are taken for vessels."
        ),
    )
    vessel: VesselConfig
    bone: BoneConfig
    burst: BurstConfig


class TubularityConfig(Section):
    """Multi-scale 3D Hessian tube-vs-plate ratio, used to veto plate-like vessels."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS, Level.ADVANCED)

    enabled: bool = Field(
        description="Compute it (only with the refinement; off: no plate veto, faster).",
        json_schema_extra=advice(
            "Turn it off for a faster run when bone plates are not mistaken for vessels."
        ),
    )
    scales_um: tuple[Annotated[float, Field(gt=0.0, le=1e4)], ...] = Field(
        min_length=1,
        description="Gaussian scales (about structure radius / sqrt 2).",
        json_schema_extra=advice(
            "Match them to the radii of the vessels to judge: add a larger scale for thicker "
            + "vessels (each scale costs time)."
        ),
    )


class ZSmoothingConfig(Section):
    """Sliding majority vote along z."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS, Level.ADVANCED)

    window_um: Length = Field(
        description="z extent of the vote (<= one slice: off).",
        json_schema_extra=advice(
            "Raise it if labels flicker from slice to slice; lower it if thin structures "
            + "crossing the slices are erased."
        ),
    )


class RefinementConfig(Section):
    """3D connectivity refinement (physical units)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS)

    enabled: bool = Field(
        description="Decide the ambiguous blobs in 3D (off: they become bone, slice by slice).",
        json_schema_extra=advice(
            "Leave it on: off is the 2D result, for comparison only (many vessels become bone)."
        ),
    )
    vessel_contact_ratio: float = Field(
        gt=0.0,
        le=1000.0,
        description="Ambiguous component becomes vessel if "
        + "vessel contacts > ratio x bone contacts.",
        json_schema_extra=advice(
            "Raise it if bone pieces touching vessels become vessel; lower it if vessel "
            + "pieces touching bone become bone."
        ),
    )
    min_tubularity: Fraction = Field(
        description="Ambiguous components flatter than this mean tube-vs-plate ratio "
        + "(|l2|/|l3|: ~1 tube, ~0 plate) cannot be vessels (needs tubularity.enabled).",
        json_schema_extra=advice(
            "Raise it if bone plates become vessel; lower it if flattened vessels become bone."
        ),
    )
    min_isolated_vessel_um3: Volume = Field(
        description="Isolated ambiguous components smaller than this are noise.",
        json_schema_extra=advice("Raise it on a noisier scan; lower it to keep isolated specks."),
    )
    min_vessel_um3: Volume = Field(
        description="Smaller 3D vessels are noise.",
        json_schema_extra=advice(
            "Raise it if the network has many tiny isolated pieces; lower it to keep short "
            + "vessel fragments (cut by clots)."
        ),
    )
    min_bone_um3: Volume = Field(
        description="Smaller 3D bone pieces are noise.",
        json_schema_extra=advice("Raise it on a noisier scan; lower it to keep small bone pieces."),
    )


class ThinVesselsConfig(Section):
    """Thin vessels the blur hides, found on a deconvolved image (off by default: unconfirmed)."""

    model_config: ClassVar[ConfigDict] = section_config(Effect.LABELS, Level.EXPERIMENTAL)

    enabled: bool = Field(
        description="After the other stages, deconvolve the stack and turn thin bright pieces "
        + "labelled background or bone into vessel. Finds about half of the thin vessels the "
        + "blur hides on the phantom; on a real scan the pieces it adds sit in or along bone, "
        + "where only an expert can tell a vessel from dense bone.",
        json_schema_extra=advice(
            "To compare with a run without it, where thin vessels matter (marrow networks, "
            + "vessel lengths). Keep it off for measurements until an expert has checked it. "
            + "It adds about 70 % to the segmentation's time."
        ),
    )
    psf_sigma_um: Annotated[float, Field(gt=0.0, le=1000.0)] = Field(
        description="Scanner blur the deconvolution undoes (Gaussian sigma), as measured on the "
        + "scan's bone edges.",
        json_schema_extra=advice(
            "Set it to another scanner's blur; the result holds with a value 20 % off."
        ),
    )
    iterations: int = Field(
        ge=1,
        le=100,
        description="Richardson-Lucy iterations: more sharpens more, and amplifies the noise.",
        json_schema_extra=advice(
            "Rarely: 3 was the best trade-off on the phantom; more finds more and invents more."
        ),
    )
    min_volume_um3: Volume = Field(
        description="Smaller added pieces are noise.",
        json_schema_extra=advice("Raise it if specks of noise are added on a noisier scan."),
    )
    max_half_thickness_um: Length = Field(
        description="A piece holding a ball of this radius is not thin: it is not added.",
        json_schema_extra=advice(
            "Lower it if dense bone is added; raise it to add slightly thicker vessels."
        ),
    )
    block_slices: int = Field(
        ge=1,
        le=100_000,
        description="Slices of a deconvolution block: larger blocks repeat their margins less "
        + "(faster) and take more memory per worker.",
        json_schema_extra=advice("Lower it if memory runs short.", Effect.NONE),
    )
    block_px: int = Field(
        ge=1,
        le=100_000,
        description="In-plane side of a deconvolution block, in pixels (same trade-off).",
        json_schema_extra=advice("Lower it if memory runs short.", Effect.NONE),
    )
