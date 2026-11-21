"""Tests of the operation functions (:mod:`pybones.api.operations`)."""

from typing import TYPE_CHECKING

import pytest

from pybones.api.operations import (
    compare_label,
    export_masks,
    mesh_label,
    preview,
    save_mesh,
    write_mesh_file,
)
from pybones.meshing.gltf import read_glb
from pybones.segmentation.labels import Label
from pybones.segmentation.pipeline import run_segmentation
from pybones.segmentation.stages import MESH_STAGES, CancelledError, ProgressTracker
from pybones.utils.constants import BGR_CHANNELS, PARTIAL_SUFFIX
from tests.helpers import N_SYNTHETIC_SLICES, SLICE_SHAPE, cancel_in, read_stl, small_config

if TYPE_CHECKING:
    from pathlib import Path

    from pybones.segmentation.stages import Stage

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
    _ = run_segmentation(slice_dir, out, small_config())
    return out


def test_preview_renders_and_checks_the_slice(segmentation: "Path") -> None:
    """The preview is a scaled BGR image; a slice outside the stack is refused."""
    image = preview(segmentation, 3, small_config(), scale=PREVIEW_SCALE)
    expected = (*(round(s * PREVIEW_SCALE) for s in SLICE_SHAPE), BGR_CHANNELS)
    assert image.shape == expected
    with pytest.raises(IndexError):
        _ = preview(segmentation, N_SYNTHETIC_SLICES, small_config())


def test_mesh_is_saved_in_the_format_of_its_extension(
    segmentation: "Path", tmp_path: "Path"
) -> None:
    """A label's mesh has a positive volume and round-trips through STL and GLB."""
    config = small_config()
    result = mesh_label(segmentation, Label.BONE, config)
    assert result.volume_mm3 > 0
    stl = save_mesh(result, Label.BONE, tmp_path / "bone.stl", config)
    assert len(read_stl(stl)) == result.mesh.n_faces
    glb = save_mesh(result, Label.BONE, tmp_path / "bone.glb", config)
    back = read_glb(glb)
    assert len(back.faces) == result.mesh.n_faces
    assert back.position_bits == config.mesh.glb_position_bits
    with pytest.raises(ValueError, match="unsupported mesh format"):
        _ = write_mesh_file(segmentation, Label.BONE, tmp_path / "bone.obj", config)


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


def test_mesh_file_reports_every_stage_and_appears_only_when_complete(
    segmentation: "Path", tmp_path: "Path"
) -> None:
    """Meshing reports its four stages in order; a cancelled run leaves no file at all."""
    config = small_config()
    path = tmp_path / "vessel.stl"
    seen: list[Stage] = []

    def record(stage: "Stage", done: int, total: int) -> None:
        """Record the stages reported, in order, and check each update.

        Args:
            stage: Reported stage.
            done: Steps done.
            total: Steps in the stage.
        """
        assert 0 <= done <= total
        if not seen or seen[-1] != stage:
            seen.append(stage)

    result = write_mesh_file(segmentation, Label.VESSEL, path, config, record)
    assert seen == list(MESH_STAGES)
    assert len(read_stl(path)) == result.n_faces
    # Cancelled at any stage up to the file being written: no file, no leftover.
    for stage in MESH_STAGES:
        tracker = ProgressTracker(MESH_STAGES, config.mesh.stage_weights)
        tracker.add_listener(cancel_in(tracker, stage))
        cancelled = tmp_path / f"cancelled_{stage}.stl"
        with pytest.raises(CancelledError):
            _ = write_mesh_file(segmentation, Label.VESSEL, cancelled, config, tracker)
        assert not cancelled.exists()
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}")) == []
