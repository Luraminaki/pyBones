"""Tests of the per-slice classification (:mod:`pybones.segmentation.classifier`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.segmentation.classifier import (
    Thresholds,
    classify_slice,
    estimate_thresholds,
    extract_features,
)
from pybones.segmentation.labels import Label
from tests.helpers import SLICE_SHAPE, small_config, small_params, synthetic_slice

if TYPE_CHECKING:
    from numpy.typing import NDArray

CONFIG = small_config()
#: Centre ``(y, x)`` of the synthetic vessel and of the small dim blob (see ``synthetic_slice``).
VESSEL_CENTRE = (150, 60)
DIM_BLOB = (120, 120)


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
    thresholds = Thresholds(
        bone=9_000, vessel=20_000, background_mean=4_000, bone_mean=13_000, vessel_mean=30_000
    )
    vessel_slice = np.full((64, 64), 4_000, np.uint16)
    vessel_slice[30:34, 8:56] = 30_000
    blur_slice = np.full((64, 64), 4_000, np.uint16)
    blur_slice[30:34, 8:56] = 12_000  # bone-like intensity, but only the vessel's z blur
    window = [blur_slice, vessel_slice, blur_slice]
    in_3d = extract_features(window, 0, thresholds, params).masks
    in_2d = extract_features([blur_slice], 0, thresholds, params).masks
    assert in_2d.foreground[32, 30]  # 2D: looks like bone
    assert not in_3d.foreground[32, 30]  # 3D: halo of the vessel next door
    assert not in_3d.vessel[32, 30]
