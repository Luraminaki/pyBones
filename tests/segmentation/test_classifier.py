"""Tests of the per-slice classification (:mod:`pybones.segmentation.classifier`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.segmentation.classifier import Thresholds, estimate_thresholds, extract_features
from pybones.segmentation.labels import Label
from tests.helpers import SLICE_SHAPE, classify_slice, small_config, small_params, synthetic_slice

if TYPE_CHECKING:
    from numpy.typing import NDArray

CONFIG = small_config()
#: Centre ``(y, x)`` of the synthetic vessel and of the small dim blob (see ``synthetic_slice``).
VESSEL_CENTRE = (150, 60)
DIM_BLOB = (120, 120)
#: Hand-set thresholds of the drawn slices below, and their intensity levels.
THRESHOLDS = Thresholds(
    bone=9_000, vessel=20_000, background_mean=4_000, bone_mean=13_000, vessel_mean=30_000
)
SOFT, BONE, CORE = 4_000, 13_000, 30_000
#: Bone holding a quarter of a vessel's blur: under the half-maximum against bone (21,500),
#: over the one against soft tissue (17,000).
BLURRED_BONE = 17_250
#: A vessel's edge against soft tissue, just over its half-maximum.
SOFT_RIM = 17_500
#: A partly filled vessel: bone-like intensity.
DIM_VESSEL = 14_000
ENABLE_BONE_SIDE = "segmentation.vessel.bone_side.enabled=true"


@pytest.fixture(scope="module")
def labels() -> "NDArray[np.uint8]":
    """Classify the first synthetic slice with thresholds from three slices.

    Returns:
        The label image.
    """
    slices = [synthetic_slice(seed=s) for s in range(3)]
    params = small_params()
    thresholds = estimate_thresholds(slices, params, CONFIG.thresholds.histogram_bins)
    return classify_slice(slices[0], thresholds, params)


def test_output_is_uint8_label_image(labels: "NDArray[np.uint8]") -> None:
    """The output has the slice shape and only label values."""
    assert labels.dtype == np.uint8
    assert labels.shape == SLICE_SHAPE
    assert set(np.unique(labels)) <= {int(lab) for lab in Label}


def test_bone_band_is_bone(labels: "NDArray[np.uint8]") -> None:
    """The cortical band is bone."""
    assert (labels[25:55, 10:240] == Label.BONE).mean() > 0.98


def test_vessel_core_is_vessel_and_its_halo_is_not_bone(labels: "NDArray[np.uint8]") -> None:
    """A bright core is vessel; its dim ring (below half-maximum) is halo, not bone."""
    yy, xx = np.ogrid[: SLICE_SHAPE[0], : SLICE_SHAPE[1]]
    r2 = (yy - VESSEL_CENTRE[0]) ** 2 + (xx - VESSEL_CENTRE[1]) ** 2
    assert (labels[r2 <= 3**2] == Label.VESSEL).all()
    ring = labels[(r2 > 5**2) & (r2 <= 6**2)]
    assert not np.isin(ring, [Label.BONE, Label.AMBIGUOUS]).any()


def test_small_dim_blob_is_ambiguous(labels: "NDArray[np.uint8]") -> None:
    """A small bone-intensity blob without a core is left to the 3D refinement."""
    assert labels[DIM_BLOB] == Label.AMBIGUOUS


def test_dense_cluster_is_burst(labels: "NDArray[np.uint8]") -> None:
    """The dense speckle cluster is burst, and nothing else is."""
    assert (labels[185:215, 185:215] == Label.BURST).mean() > 0.95
    assert not (labels[:150] == Label.BURST).any()


def test_background_stays_background(labels: "NDArray[np.uint8]") -> None:
    """Soft tissue away from every structure stays background."""
    assert (labels[80:110, 150:250] == Label.BACKGROUND).all()


def test_z_blur_of_a_vessel_in_the_next_slice_is_halo_not_bone() -> None:
    """The z blur of a vessel lying in a neighbouring slice is its halo in 3D, not bone."""
    params = small_params()
    vessel_slice = np.full((64, 64), SOFT, np.uint16)
    vessel_slice[30:34, 8:56] = CORE
    blur_slice = np.full((64, 64), SOFT, np.uint16)
    blur_slice[30:34, 8:56] = 12_000  # bone-like intensity, but only the vessel's z blur
    window = [blur_slice, vessel_slice, blur_slice]
    in_3d = extract_features(window, 0, THRESHOLDS, params).masks
    in_2d = extract_features([blur_slice], 0, THRESHOLDS, params).masks
    assert in_2d.foreground[32, 30]  # 2D: looks like bone
    assert not in_3d.foreground[32, 30]  # 3D: halo of the vessel next door
    assert not in_3d.vessel[32, 30]


@pytest.mark.parametrize(("bone_side", "blurred_bone_is_vessel"), [(False, True), (True, False)])
def test_bone_side_edge_keeps_blurred_bone_out_of_the_vessel(
    bone_side: bool, blurred_bone_is_vessel: bool
) -> None:
    """At a contact, bone with some vessel blur is vessel against soft tissue, bone against bone."""
    params = small_params(*([ENABLE_BONE_SIDE] if bone_side else []))
    img = np.full((64, 64), SOFT, np.uint16)
    img[0:30] = BONE  # a thick plate
    img[30:33, 8:56] = CORE  # a thin vessel pressing on it: soft tissue just past it
    img[29, 8:56] = BLURRED_BONE
    img[30:33, 56] = SOFT_RIM  # the vessel's end, in soft tissue
    vessel = extract_features([img], 0, THRESHOLDS, params).masks.vessel
    assert vessel[29, 30] == blurred_bone_is_vessel
    assert vessel[31, 30]  # the core
    assert vessel[31, 56]  # the edge against soft tissue is unchanged


@pytest.mark.parametrize(("barrier_um", "edge_is_vessel"), [(12, True), (3, False)])
def test_thin_foreground_is_no_bone_barrier(barrier_um: int, edge_is_vessel: bool) -> None:
    """A dim vessel continuing a bright one is not bone: thinner than the barrier disc."""
    params = small_params(
        ENABLE_BONE_SIDE, f"segmentation.vessel.bone_side.barrier_um={barrier_um}"
    )
    img = np.full((64, 64), SOFT, np.uint16)
    img[30:33, 4:30] = CORE
    img[30:33, 30:60] = DIM_VESSEL  # 9 µm thick: holds a 3 µm disc, not a 12 µm one
    img[30:33, 30] = SOFT_RIM
    vessel = extract_features([img], 0, THRESHOLDS, params).masks.vessel
    assert vessel[31, 30] == edge_is_vessel
