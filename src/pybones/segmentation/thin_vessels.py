"""Thin vessels the scanner blur hides, found on a deconvolved image (optional last pass).

A vessel thinner than the blur reads dim: often no brighter than bone, so the segmentation
labels it bone or background. Deconvolving the stack (Richardson-Lucy, the measured blur)
brings such vessels back above the vessel-core threshold. Not everything that does is a vessel:
noise, and dense bone, sharpen too. So only *thin* pieces are added:

1. **sharpen**: the stack is deconvolved block by block; candidates are the voxels labelled
   background or bone where the deconvolved image reaches the vessel-core threshold. A
   candidate voxel is *thick* where a ball of ``max_half_thickness_um`` around it holds only
   candidates. Candidates are linked into 3D pieces as they stream past
   (:class:`StreamingComponents`), counting each piece's voxels and thick voxels;
2. **thin**: pieces large enough to be more than noise and without any thick voxel become
   vessel; the labels are rewritten slice by slice.

Phantom: about half of the thin vessels the blur hides are found, nothing else changes. Real
scan: the pieces sit in or along bone, where only an expert can tell a vessel from dense bone,
hence the switch, off by default.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from pybones.imaging import morphology as morph
from pybones.imaging.deconvolution import GaussianPsf, iter_deconvolved_above
from pybones.io.volume_store import LabelVolumeWriter, open_volume
from pybones.segmentation.connectivity import StreamingComponents
from pybones.segmentation.labels import Label
from pybones.segmentation.scales import voxel_geometry
from pybones.segmentation.stages import Stage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.config.schema import PyBonesConfig
    from pybones.io.slices import SliceStack
    from pybones.segmentation.geometry import Ball
    from pybones.segmentation.stages import ProgressCallback

#: Labels a thin vessel can hide under.
HIDING_LABELS = (Label.BACKGROUND, Label.BONE)


class ThinVesselStats(BaseModel):
    """What the thin-vessel pass added (recorded in ``meta.json``).

    Attributes:
        candidate_pieces: 3D pieces of candidates (bright once deconvolved, not yet vessel).
        pieces_added: Pieces that were thin and large enough, now vessel.
        from_background: Voxels added that were background.
        from_bone: Voxels added that were bone.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    candidate_pieces: int
    pieces_added: int
    from_background: int
    from_bone: int


@dataclass(frozen=True, slots=True)
class ThinVesselParams:
    """The thin-vessel settings in voxel units.

    Attributes:
        psf: The scanner blur, in voxels.
        iterations: Richardson-Lucy iterations.
        min_voxels: Smaller pieces are noise.
        thick_ball: A candidate voxel whose ball is all candidate is thick.
        block: Core of a deconvolution block, ``(slices, pixels)``.
        connectivity: In-plane connectivity of the pieces (8: 26-connected in 3D).
    """

    psf: GaussianPsf
    iterations: int
    min_voxels: int
    thick_ball: "Ball"
    block: tuple[int, int]
    connectivity: int


def resolve_thin_vessels(config: "PyBonesConfig") -> ThinVesselParams:
    """Convert the thin-vessel settings to voxel units.

    Args:
        config: Effective configuration.

    Returns:
        The settings of this scan.
    """
    geo = voxel_geometry(config)
    tv = config.thin_vessels
    sigma = tv.psf_sigma_um
    return ThinVesselParams(
        psf=GaussianPsf((sigma / geo.z, sigma / geo.y, sigma / geo.x)),
        iterations=tv.iterations,
        min_voxels=geo.volume_voxels(tv.min_volume_um3),
        thick_ball=geo.ball(tv.max_half_thickness_um),
        block=(tv.block_slices, tv.block_px),
        connectivity=config.segmentation.connectivity,
    )


def iter_thick(
    masks: "Iterable[NDArray[np.bool_]]", ball: "Ball"
) -> "Iterator[tuple[NDArray[np.bool_], NDArray[np.bool_]]]":
    """Stream a mask volume with, for each slice, its voxels whose whole ball is in the mask.

    Beyond the stack's faces counts as inside, as for a distance map: a piece cut by a face is
    judged on its visible part.

    Args:
        masks: The mask, slice by slice.
        ball: The ball, as per-slice ellipses (:meth:`VoxelGeometry.ball`).

    Yields:
        ``(mask, thick)`` for each slice, in order.
    """
    reach = max(abs(dz) for dz, _ in ball)
    kernels = tuple((dz, morph.ellipse(radius)) for dz, radius in ball)
    window: dict[int, NDArray[np.bool_]] = {}

    def thick(z: int) -> "NDArray[np.bool_]":
        """Erode slice ``z`` by the ball, from the slices around it.

        Args:
            z: Slice index (its neighbours within the ball's reach are in the window).

        Returns:
            Its thick voxels.
        """
        out = window[z].copy()
        for dz, kernel in kernels:
            other = window.get(z + dz)
            if other is not None:  # missing beyond the stack: inside, as said above
                out &= cv2.erode(morph.as_uint8(other), kernel).astype(bool)
        return out

    last = -1
    for last, mask in enumerate(masks):
        window[last] = mask
        done = last - reach
        if done >= 0:
            yield window[done], thick(done)
            _ = window.pop(done - reach, None)
    for done in range(max(0, last - reach + 1), last + 1):
        yield window[done], thick(done)


def store_raw(stack: "SliceStack", path: "Path", progress: "ProgressCallback") -> None:
    """Copy the raw slices into a ``.npy`` volume the deconvolution reads block by block.

    Args:
        stack: Source slices.
        path: Scratch file (overwritten).
        progress: Reported as :attr:`Stage.SHARPEN`, over twice the depth (copy, then
            deconvolution).
    """
    depth = len(stack)
    raw = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=stack.shape)
    for z in range(depth):
        raw[z] = stack.read(z)
        progress(Stage.SHARPEN, z + 1, 2 * depth)
    raw.flush()
    del raw


def add_thin_vessels(
    paths: "tuple[Path, Path, Path, Path]",
    level: float,
    params: ThinVesselParams,
    workers: int,
    progress: "ProgressCallback",
) -> "tuple[ThinVesselStats, NDArray[np.int64]]":
    """Add the thin vessels to a label volume.

    Args:
        paths: ``(raw, labels, candidates, final)``: the raw volume (:func:`store_raw`), the
            labels so far, a scratch file for the candidates, the output volume.
        level: Vessel-core threshold (grey level).
        params: Thin-vessel settings in voxel units.
        workers: Threads deconvolving blocks.
        progress: Called with ``(stage, done, total)`` after each slice.

    Returns:
        What was added, and the voxel count of each label value in the output.
    """
    raw_path, labels_path, candidates_path, final_path = paths
    raw, labels = np.load(raw_path, mmap_mode="r"), open_volume(labels_path)
    depth, shape = labels.shape[0], labels.shape
    tracker = StreamingComponents(params.connectivity)
    thick_per_id = [np.zeros(1, np.int64)]
    bright = iter_deconvolved_above(
        raw, params.psf, params.iterations, level, params.block, workers
    )

    with LabelVolumeWriter(candidates_path, shape) as writer:

        def candidates() -> "Iterator[NDArray[np.bool_]]":
            """Keep the bright voxels a thin vessel could hide under, storing them.

            Yields:
                The candidates of each slice.
            """
            for z, mask in enumerate(bright):
                cand = mask & np.isin(labels[z], HIDING_LABELS)
                writer.write(morph.as_uint8(cand))
                progress(Stage.SHARPEN, depth + z + 1, 2 * depth)
                yield cand

        for cand, thick in iter_thick(candidates(), params.thick_ball):
            local = tracker.add_slice(cand)
            thick_per_id.append(np.bincount(local[thick], minlength=int(local.max()) + 1)[1:])

    component_of = tracker.resolve()
    sizes = tracker.component_sizes(component_of)
    thick_voxels = tracker.per_component(component_of, np.concatenate(thick_per_id))
    keep = (sizes >= params.min_voxels) & (thick_voxels == 0)
    keep[0] = False

    stored = open_volume(candidates_path)
    added_from = dict.fromkeys(HIDING_LABELS, 0)
    with LabelVolumeWriter(final_path, shape) as writer:
        for z in range(depth):
            out, cand = np.array(labels[z]), stored[z].astype(bool)
            local = tracker.local_labels(cand)
            add = np.zeros_like(cand)
            add[cand] = keep[component_of[local[cand] + tracker.offsets[z]]]
            for label in HIDING_LABELS:
                added_from[label] += int(np.count_nonzero(out[add] == label))
            out[add] = Label.VESSEL
            writer.write(out)
            progress(Stage.THIN, z + 1, depth)
    stats = ThinVesselStats(
        candidate_pieces=len(sizes) - 1,
        pieces_added=int(keep.sum()),
        from_background=added_from[Label.BACKGROUND],
        from_bone=added_from[Label.BONE],
    )
    return stats, writer.counts
