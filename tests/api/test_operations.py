"""Tests of the operation functions (:mod:`pybones.api.operations`)."""

from typing import TYPE_CHECKING

import pytest

from pybones.api.operations import (
    compare_label,
    export_masks,
    mesh_label,
    preview,
    save_mesh,
    segment,
)
from pybones.meshing.stl import read_stl
from pybones.segmentation.labels import Label
from pybones.utils.constants import BGR_CHANNELS
from tests.helpers import N_SYNTHETIC_SLICES, SLICE_SHAPE, small_config

if TYPE_CHECKING:
    from pathlib import Path

PREVIEW_SCALE = 0.5


@pytest.fixture
def segmentation(slice_dir: "Path", tmp_path: "Path") -> "Path":
    """Segment the synthetic stack.

    Args:
        slice_dir: Synthetic slices.
        tmp_path: Temporary directory of the test.

    Returns:
        The segmentation output directory.
    """
    out = tmp_path / "seg"
    _ = segment(slice_dir, out, small_config())
    return out


def test_preview_renders_and_checks_the_slice(segmentation: "Path") -> None:
    """The preview is a scaled BGR image; a slice outside the stack is refused."""
    image = preview(segmentation, 3, small_config(), scale=PREVIEW_SCALE)
    expected = (*(round(s * PREVIEW_SCALE) for s in SLICE_SHAPE), BGR_CHANNELS)
    assert image.shape == expected
    with pytest.raises(IndexError):
        _ = preview(segmentation, N_SYNTHETIC_SLICES, small_config())


def test_mesh_is_saved_as_stl(segmentation: "Path", tmp_path: "Path") -> None:
    """A label's mesh has a positive volume and round-trips through STL."""
    config = small_config()
    result = mesh_label(segmentation, Label.BONE, config)
    assert result.volume_mm3 > 0
    path = save_mesh(result, Label.BONE, tmp_path / "bone.stl", config)
    assert len(read_stl(path)) == result.mesh.n_faces


def test_export_and_compare_agree_with_themselves(segmentation: "Path", tmp_path: "Path") -> None:
    """A label compared with its own export is a perfect match; a short reference is refused."""
    config = small_config()
    paths = export_masks(segmentation, Label.VESSEL, tmp_path / "png", config)
    assert len(paths) == N_SYNTHETIC_SLICES
    agreement = compare_label(segmentation, Label.VESSEL, tmp_path / "png", config)
    assert agreement.dice == 1.0
    paths[-1].unlink()
    with pytest.raises(ValueError, match="slices"):
        _ = compare_label(segmentation, Label.VESSEL, tmp_path / "png", config)
