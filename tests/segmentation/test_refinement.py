"""Tests of the 3D connectivity refinement (:mod:`pybones.segmentation.refinement`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.io.volume_store import LabelVolumeWriter, open_volume
from pybones.segmentation.labels import FINAL_LABELS, Label
from pybones.segmentation.refinement import RefinementStats, refine_volume
from pybones.segmentation.stages import no_progress
from pybones.segmentation.tubularity import TUBULARITY_LEVELS
from tests.helpers import small_params

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

PARAMS = small_params(
    "refinement.min_isolated_vessel_um3=540",  # 20 voxels
    "refinement.min_vessel_um3=135",  # 5 voxels
    "refinement.min_bone_um3=1350",  # 50 voxels
)
B, V, A = Label.BONE, Label.VESSEL, Label.AMBIGUOUS
CLASSIFIED, RESOLVED, FINAL, TUBULARITY = "c.npy", "r.npy", "f.npy", "t.npy"
PLATE_LIKE, TUBE_LIKE = round(0.2 * TUBULARITY_LEVELS), round(0.8 * TUBULARITY_LEVELS)

#: Output of :func:`refine`: final volume, statistics and label counts.
type Refined = tuple[NDArray[np.uint8], RefinementStats, NDArray[np.int64]]


def write_volume(path: "Path", volume: "NDArray[np.uint8]") -> None:
    """Store a volume the way the pipeline does.

    Args:
        path: Output ``.npy`` file.
        volume: ``(Z, Y, X)`` uint8 volume.
    """
    with LabelVolumeWriter(path, (volume.shape[0], volume.shape[1], volume.shape[2])) as writer:
        for sl in volume:
            writer.write(sl)


def refine(
    directory: "Path", volume: "NDArray[np.uint8]", tubularity: "NDArray[np.uint8] | None" = None
) -> Refined:
    """Run the refinement on in-memory volumes.

    Args:
        directory: Scratch directory.
        volume: Classified volume (``AMBIGUOUS`` included).
        tubularity: Quantised tubularity volume (enables the plate veto), if any.

    Returns:
        The final volume, the statistics and the label counts.
    """
    write_volume(directory / CLASSIFIED, volume)
    tubularity_path = None
    if tubularity is not None:
        tubularity_path = directory / TUBULARITY
        write_volume(tubularity_path, tubularity)
    stats, counts = refine_volume(
        directory / CLASSIFIED,
        directory / RESOLVED,
        directory / FINAL,
        PARAMS,
        no_progress,
        tubularity_path,
    )
    return np.array(open_volume(directory / FINAL)), stats, counts


@pytest.fixture(scope="module")
def refined(
    tmp_path_factory: pytest.TempPathFactory,
) -> Refined:
    """Refine a volume holding one example of every decision.

    Args:
        tmp_path_factory: Temporary directory factory.

    Returns:
        The final volume, the statistics and the label counts.
    """
    vol = np.zeros((6, 40, 40), np.uint8)
    vol[:, 0:5, :] = B  # bone plate through every slice
    vol[:, 5:15, 10] = A  # rod attached to the plate -> bone
    vol[:, 20, 20:30] = V  # perfused vessel...
    vol[:, 20, 30:36] = A  # ...continued by a dim (clotted) segment -> vessel
    vol[2:4, 30, 5:7] = A  # isolated, 4 voxels -> noise
    vol[:, 35, 10:15] = A  # isolated, 30 voxels -> vessel fragment
    vol[3, 37, 37] = V  # 1-voxel vessel speck -> removed
    vol[3, 30:33, 30:33] = B  # 9-voxel bone speck -> removed
    return refine(tmp_path_factory.mktemp("refine"), vol)


def test_only_final_labels_remain(refined: Refined) -> None:
    """No ``AMBIGUOUS`` voxel survives the refinement."""
    out, _, _ = refined
    assert set(np.unique(out)) <= {int(lab) for lab in FINAL_LABELS}


def test_rod_touching_bone_becomes_bone(refined: Refined) -> None:
    """A blob touching only bone joins the bone."""
    assert (refined[0][:, 5:15, 10] == B).all()


def test_dim_segment_continuing_a_vessel_becomes_vessel(refined: Refined) -> None:
    """A blob continuing a vessel becomes vessel."""
    assert (refined[0][:, 20, 30:36] == V).all()


def test_isolated_components_by_size(refined: Refined) -> None:
    """Isolated blobs are noise when small, vessel fragments otherwise."""
    out, _, _ = refined
    assert (out[2:4, 30, 5:7] == Label.BACKGROUND).all()
    assert (out[:, 35, 10:15] == V).all()


def test_small_3d_specks_are_removed(refined: Refined) -> None:
    """Vessel and bone components below their minimum volume are removed, not the rest."""
    out, _, _ = refined
    assert out[3, 37, 37] == Label.BACKGROUND
    assert (out[3, 30:33, 30:33] == Label.BACKGROUND).all()
    assert (out[:, 0:5, :] == B).all()


def test_stats_and_counts(refined: Refined) -> None:
    """The statistics count every decision and the counts match the output volume."""
    out, stats, counts = refined
    assert stats.ambiguous_components == 4
    assert (stats.ambiguous_to_bone, stats.ambiguous_to_vessel) == (1, 2)
    assert stats.ambiguous_to_background == 1
    assert stats.vessel_components_removed == 1
    assert stats.bone_components_removed == 1
    np.testing.assert_array_equal(
        counts[: len(Label)], np.bincount(out.ravel(), minlength=len(Label))
    )


def test_plate_veto_uses_tubularity(tmp_path: "Path") -> None:
    """A plate-like blob touching a vessel is not a vessel; a tube-like one is."""
    vol = np.zeros((4, 30, 30), np.uint8)
    vol[:, 2, 2:28] = V  # a vessel...
    vol[:, 3:9, 0:4] = B  # (bone next to the first blob)
    vol[:, 3:9, 4:10] = A  # ...touched by a blob that is plate-like (144 voxels)
    vol[:, 3:9, 16:22] = A  # ...and by one that is tube-like
    tub = np.zeros_like(vol)
    tub[:, 3:9, 4:10] = PLATE_LIKE
    tub[:, 3:9, 16:22] = TUBE_LIKE
    out, _, _ = refine(tmp_path, vol, tub)
    assert (out[:, 3:9, 4:10] == B).all()  # plate-like, touching bone: bone despite the vessel
    assert (out[:, 3:9, 16:22] == V).all()


def test_plate_like_blob_away_from_bone_is_background(tmp_path: "Path") -> None:
    """A plate-like blob touching only a vessel (a sheath, a fascia...) is background."""
    vol = np.zeros((4, 30, 30), np.uint8)
    vol[:, 2, 2:28] = V
    vol[:, 3:9, 4:10] = A
    tub = np.where(vol == A, PLATE_LIKE, 0).astype(np.uint8)
    out, _, _ = refine(tmp_path, vol, tub)
    assert (out[:, 3:9, 4:10] == Label.BACKGROUND).all()
