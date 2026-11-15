"""Tests of slice reading, label volumes and exports (:mod:`pybones.io`)."""

import zipfile
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import pytest

from pybones.config.loader import load_config
from pybones.io.export import export_mask_sequence
from pybones.io.slices import SliceStack
from pybones.io.volume_store import LabelVolumeWriter, open_volume
from pybones.segmentation.labels import Label
from tests.helpers import N_SYNTHETIC_SLICES, SLICE_SHAPE

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

IO = load_config().io
_MASK_ON = 255


def test_slice_stack_natural_order_and_zip(slice_dir: "Path", tmp_path: "Path") -> None:
    """Slices are naturally sorted, read from directories or zips, and junk entries skipped."""
    with SliceStack(slice_dir, IO) as stack:
        assert stack.names == [f"img_{z}.png" for z in range(N_SYNTHETIC_SLICES)]
        assert stack.shape == (N_SYNTHETIC_SLICES, *SLICE_SHAPE)
        assert stack.dtype == np.uint16
        first = stack.read(0)
    archive = tmp_path / "stack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for z in (10, 2, 1):  # out of order, with a 2-digit index
            zf.write(slice_dir / "img_0.png", f"data/img_{z}.png")
        zf.writestr("__MACOSX/data/img_1.png", b"junk")
        zf.writestr("data/._img_3.png", b"junk")
    with SliceStack(archive, IO) as stack:
        assert stack.names == ["data/img_1.png", "data/img_2.png", "data/img_10.png"]
        np.testing.assert_array_equal(stack.read(-1), first)


def test_slice_stack_errors(tmp_path: "Path") -> None:
    """A missing source and a source without slices are reported."""
    with pytest.raises(FileNotFoundError):
        _ = SliceStack(tmp_path / "missing", IO)
    with pytest.raises(ValueError, match="No slice"):
        _ = SliceStack(tmp_path, IO)


def test_volume_writer_roundtrip_and_counts(tmp_path: "Path") -> None:
    """A written volume reads back identically, with per-value counts."""
    path = tmp_path / "v.npy"
    slices = [np.full((4, 5), z, dtype=np.uint8) for z in range(3)]
    with LabelVolumeWriter(path, (3, 4, 5)) as writer:
        for sl in slices:
            writer.write(sl)
    np.testing.assert_array_equal(open_volume(path), np.stack(slices))
    assert writer.counts[:3].tolist() == [20, 20, 20]


def test_volume_writer_checks_slices(tmp_path: "Path") -> None:
    """Wrong slice shapes, missing slices and writes outside the block are rejected."""
    with pytest.raises(ValueError), LabelVolumeWriter(tmp_path / "v.npy", (1, 4, 5)) as writer:
        writer.write(np.zeros((5, 4), np.uint8))
    with pytest.raises(RuntimeError), LabelVolumeWriter(tmp_path / "v.npy", (2, 4, 5)) as writer:
        writer.write(np.zeros((4, 5), np.uint8))
    with pytest.raises(RuntimeError, match="outside"):
        LabelVolumeWriter(tmp_path / "v.npy", (1, 4, 5)).write(np.zeros((4, 5), np.uint8))


def test_export_mask_sequence(tmp_path: "Path") -> None:
    """One 0/255 PNG per slice, named by the configured template."""
    vol = np.zeros((3, 5, 6), np.uint8)
    vol[1, 2, 3] = Label.VESSEL
    paths = export_mask_sequence(vol, Label.VESSEL, tmp_path, IO)
    assert [p.name for p in paths] == ["vessel_0000.png", "vessel_0001.png", "vessel_0002.png"]
    img = cast("NDArray[np.uint8]", cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED))
    assert img.dtype == np.uint8
    assert img[2, 3] == _MASK_ON
    assert img.sum() == _MASK_ON
