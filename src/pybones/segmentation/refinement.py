"""3D connectivity refinement of a classified label volume.

Three streaming passes over on-disk volumes (a few slices in memory at a time):

1. **Analyse** the ``AMBIGUOUS`` voxels: their 3D connected components, and how
   many of their voxels touch (26-neighbourhood) bone versus vessels/bursts.
2. **Resolve** each ambiguous component, write the resolved volume, and track
   the 3D components of vessels and bone:

   * touching vessels more than ``vessel_contact_ratio`` x bone -> vessel
     (partially filled / clotted vessel continuing a perfused one);
   * otherwise touching bone -> bone (trabecula joining the bone network);
   * isolated: bone cannot float, so a large enough isolated piece is a
     vessel fragment; small ones are noise -> background.
3. **Clean**: remove vessel and bone 3D components smaller than their minimum
   voxel counts (specks).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from pybones.imaging import morphology as morph
from pybones.io.volume_store import LabelVolumeWriter, open_volume
from pybones.segmentation.connectivity import StreamingComponents
from pybones.segmentation.labels import Label
from pybones.segmentation.stages import Stage
from pybones.segmentation.tubularity import TUBULARITY_LEVELS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.segmentation.scales import PixelParams
    from pybones.segmentation.stages import ProgressCallback


class RefinementStats(BaseModel):
    """What the refinement changed.

    Attributes:
        ambiguous_components: 3D components of ambiguous voxels.
        ambiguous_to_bone: Ambiguous components labelled bone.
        ambiguous_to_vessel: Ambiguous components labelled vessel.
        ambiguous_to_background: Ambiguous components labelled background (noise, plates).
        vessel_components: 3D vessel components before cleaning.
        vessel_components_removed: Vessel components below the minimum volume.
        bone_components: 3D bone components before cleaning.
        bone_components_removed: Bone components below the minimum volume.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    ambiguous_components: int
    ambiguous_to_bone: int
    ambiguous_to_vessel: int
    ambiguous_to_background: int
    vessel_components: int
    vessel_components_removed: int
    bone_components: int
    bone_components_removed: int


@dataclass(frozen=True, slots=True)
class _AmbiguousAnalysis:
    """Result of pass 1.

    Attributes:
        tracker: Connectivity of the ambiguous voxels (to re-label slices in pass 2).
        component_of: Global id -> 3D component.
        sizes: Voxels per component.
        bone_contacts: Voxels per component touching bone.
        vessel_contacts: Voxels per component touching vessels or bursts.
        tubularity: Mean tube-vs-plate ratio per component (``None`` if not computed).
    """

    tracker: StreamingComponents
    component_of: "NDArray[np.int64]"
    sizes: "NDArray[np.int64]"
    bone_contacts: "NDArray[np.int64]"
    vessel_contacts: "NDArray[np.int64]"
    tubularity: "NDArray[np.float64] | None"


@dataclass(frozen=True, slots=True)
class _Removal:
    """Pass 3 plan for one label: which of its 3D components are too small.

    Attributes:
        label: Label being cleaned.
        tracker: Connectivity of that label (to re-label slices).
        remove_id: Per global id, whether its component is removed.
        too_small: Per 3D component, whether it is removed (index 0 = background).
    """

    label: Label
    tracker: StreamingComponents
    remove_id: "NDArray[np.bool_]"
    too_small: "NDArray[np.bool_]"


def _plan_removal(label: Label, tracker: StreamingComponents, min_voxels: int) -> _Removal:
    """Find the 3D components of one label that are too small to keep.

    Args:
        label: Label being cleaned.
        tracker: Connectivity of that label.
        min_voxels: Smaller components are removed.

    Returns:
        The removal plan of the label.
    """
    component_of = tracker.resolve()
    too_small = tracker.component_sizes(component_of) < min_voxels
    too_small[0] = False
    return _Removal(label, tracker, too_small[component_of], too_small)


def _neighbourhood_slab(volume: "NDArray[np.uint8]", z: int) -> "NDArray[np.uint8]":
    """Read slices ``z - 1 .. z + 1`` (clipped at the volume ends).

    Args:
        volume: Label volume (memory-mapped).
        z: Centre slice.

    Returns:
        The slab, loaded in memory.
    """
    return np.asarray(volume[max(0, z - 1) : z + 2])


def _analyse_ambiguous(
    volume: "NDArray[np.uint8]",
    tubularity: "NDArray[np.uint8] | None",
    connectivity: int,
    progress: "ProgressCallback",
) -> _AmbiguousAnalysis:
    """Pass 1: 3D components of ambiguous voxels, their contacts and mean tubularity.

    Args:
        volume: Classified volume (``AMBIGUOUS`` included).
        tubularity: Quantised tubularity volume, if computed.
        connectivity: In-plane connectivity (4 or 8).
        progress: Progress callback (reported as :attr:`Stage.ANALYSE`).

    Returns:
        The components and their measurements.
    """
    tracker = StreamingComponents(connectivity)
    bone_contacts = [np.zeros(1, dtype=np.int64)]
    vessel_contacts = [np.zeros(1, dtype=np.int64)]
    tube_sums = [np.zeros(1, dtype=np.int64)]
    depth = volume.shape[0]
    for z in range(depth):
        cur = np.asarray(volume[z])
        ambiguous = cur == Label.AMBIGUOUS
        local = tracker.add_slice(ambiguous)
        n_new = tracker.n_ids - 1 - tracker.offsets[-1]
        slab = _neighbourhood_slab(volume, z)
        # Dilating the union of the 3 slices by the in-plane neighbourhood gives the
        # 26-neighbourhood (8-connectivity) of every voxel of slice z.
        near_bone = morph.dilate_neighbours((slab == Label.BONE).any(axis=0), connectivity)
        vessel_like = (slab == Label.VESSEL) | (slab == Label.BURST)
        near_vessel = morph.dilate_neighbours(vessel_like.any(axis=0), connectivity)
        ids = local[ambiguous]
        for contacts, near in ((bone_contacts, near_bone), (vessel_contacts, near_vessel)):
            counts = np.bincount(ids[near[ambiguous]], minlength=n_new + 1)
            contacts.append(counts[1:].astype(np.int64))
        if tubularity is not None:
            weights = np.asarray(tubularity[z])[ambiguous]
            tube_sums.append(
                np.bincount(ids, weights=weights, minlength=n_new + 1)[1:].astype(np.int64)
            )
        progress(Stage.ANALYSE, z + 1, depth)
    component_of = tracker.resolve()
    sizes = tracker.component_sizes(component_of)
    mean_tube = None
    if tubularity is not None:
        tube_total = tracker.per_component(component_of, np.concatenate(tube_sums))
        mean_tube = tube_total / np.maximum(sizes, 1) / TUBULARITY_LEVELS
    return _AmbiguousAnalysis(
        tracker=tracker,
        component_of=component_of,
        sizes=sizes,
        bone_contacts=tracker.per_component(component_of, np.concatenate(bone_contacts)),
        vessel_contacts=tracker.per_component(component_of, np.concatenate(vessel_contacts)),
        tubularity=mean_tube,
    )


def _decide(analysis: _AmbiguousAnalysis, params: "PixelParams") -> "NDArray[np.uint8]":
    """Decide the label of each ambiguous 3D component.

    * small isolated components: noise -> background;
    * plate-like components (mean tubularity below ``min_tubularity``) cannot be
      vessels: bone if they touch bone, else background (e.g. the dim sheath of a
      vessel, or a fascia, in soft tissue);
    * otherwise vessel when touching vessels more than ``vessel_contact_ratio`` x bone,
      or when isolated (bone cannot float); bone else.

    Args:
        analysis: Result of pass 1.
        params: Refinement parameters in voxel units.

    Returns:
        One :class:`Label` per component (index 0 = background).
    """
    bone_c, vessel_c, sizes = analysis.bone_contacts, analysis.vessel_contacts, analysis.sizes
    isolated = (bone_c == 0) & (vessel_c == 0)
    tubular = (
        np.ones(sizes.shape, dtype=np.bool_)
        if analysis.tubularity is None
        else analysis.tubularity >= params.min_tubularity
    )
    vessel_like = (vessel_c > params.vessel_contact_ratio * bone_c) | isolated
    decision = np.full(sizes.shape, Label.BONE, dtype=np.uint8)
    decision[tubular & vessel_like] = Label.VESSEL
    decision[~tubular & (bone_c == 0)] = Label.BACKGROUND
    decision[isolated & (sizes < params.min_isolated_vessel_voxels)] = Label.BACKGROUND
    decision[0] = Label.BACKGROUND
    return decision


def _iter_resolved(
    volume: "NDArray[np.uint8]",
    analysis: _AmbiguousAnalysis,
    decision: "NDArray[np.uint8]",
) -> "Iterator[NDArray[np.uint8]]":
    """Pass 2 stream: replace the ambiguous voxels by their component's decision.

    Args:
        volume: Classified volume.
        analysis: Result of pass 1.
        decision: Label of each ambiguous component (see :func:`_decide`).

    Yields:
        Each resolved slice, in z order.
    """
    label_of_id = decision[analysis.component_of]
    for z in range(volume.shape[0]):
        cur = np.array(volume[z])
        ambiguous = cur == Label.AMBIGUOUS
        local = analysis.tracker.local_labels(ambiguous)
        cur[ambiguous] = label_of_id[local[ambiguous] + analysis.tracker.offsets[z]]
        yield cur


def refine_volume(
    classified_path: "Path",
    resolved_path: "Path",
    final_path: "Path",
    params: "PixelParams",
    progress: "ProgressCallback",
    tubularity_path: "Path | None" = None,
) -> "tuple[RefinementStats, NDArray[np.int64]]":
    """Run the three refinement passes.

    Args:
        classified_path: Input volume, ``AMBIGUOUS`` labels included.
        resolved_path: Scratch file for the resolved volume (deleted afterwards).
        final_path: Output volume (final labels only).
        params: Segmentation / refinement parameters in pixel units.
        progress: Called with ``(stage, done, total)`` after each slice.
        tubularity_path: Quantised tubularity volume (enables the plate veto), if any.

    Returns:
        Statistics on what was changed, and the voxel count of each label value
        in the final volume.
    """
    conn = params.connectivity
    classified = open_volume(classified_path)
    shape = classified.shape
    tubularity = open_volume(tubularity_path) if tubularity_path is not None else None
    analysis = _analyse_ambiguous(classified, tubularity, conn, progress)
    del tubularity
    decision = _decide(analysis, params)

    vessels, bones = StreamingComponents(conn), StreamingComponents(conn)
    with LabelVolumeWriter(resolved_path, shape) as writer:
        for z, sl in enumerate(_iter_resolved(classified, analysis, decision)):
            writer.write(sl)
            _ = vessels.add_slice(sl == Label.VESSEL)
            _ = bones.add_slice(sl == Label.BONE)
            progress(Stage.RESOLVE, z + 1, shape[0])
    del classified

    removals = [
        _plan_removal(Label.VESSEL, vessels, params.min_vessel_voxels),
        _plan_removal(Label.BONE, bones, params.min_bone_voxels),
    ]

    resolved = open_volume(resolved_path)
    with LabelVolumeWriter(final_path, shape) as final_writer:
        for z in range(shape[0]):
            cur = np.array(resolved[z])
            for removal in removals:
                mask = cur == removal.label
                local = removal.tracker.local_labels(mask)
                drop = np.zeros_like(mask)
                drop[mask] = removal.remove_id[local[mask] + removal.tracker.offsets[z]]
                cur[drop] = Label.BACKGROUND
            final_writer.write(cur)
            progress(Stage.CLEAN, z + 1, shape[0])
    del resolved

    n_ambiguous = len(decision) - 1
    vessel_small, bone_small = (removal.too_small for removal in removals)
    stats = RefinementStats(
        ambiguous_components=n_ambiguous,
        ambiguous_to_bone=int(np.count_nonzero(decision[1:] == Label.BONE)),
        ambiguous_to_vessel=int(np.count_nonzero(decision[1:] == Label.VESSEL)),
        ambiguous_to_background=int(np.count_nonzero(decision[1:] == Label.BACKGROUND)),
        vessel_components=len(vessel_small) - 1,
        vessel_components_removed=int(vessel_small.sum()),
        bone_components=len(bone_small) - 1,
        bone_components_removed=int(bone_small.sum()),
    )
    return stats, final_writer.counts
