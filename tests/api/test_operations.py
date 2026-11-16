"""Tests of the operation functions (:mod:`pybones.api.operations`)."""

from typing import TYPE_CHECKING

import pytest

from pybones.api.operations import (
    MESH_STEPS,
    PARTIAL_SUFFIX,
    compare_label,
    export_masks,
    mesh_label,
    preview,
    save_mesh,
    segment,
    write_mesh_file,
)
from pybones.meshing.stl import read_stl
from pybones.segmentation.labels import Label
from pybones.segmentation.stages import CancelledError, ProgressTracker, Stage
from pybones.utils.constants import BGR_CHANNELS
from tests.helpers import N_SYNTHETIC_SLICES, SLICE_SHAPE, small_config

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pybones.segmentation.stages import Progress

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


def test_mesh_file_appears_only_when_complete(segmentation: "Path", tmp_path: "Path") -> None:
    """The STL is written in steps of the mesh stage; a cancelled run leaves no file at all."""
    config = small_config()
    path = tmp_path / "vessel.stl"
    steps: list[int] = []

    def record(stage: Stage, done: int, total: int) -> None:
        """Record a progress update.

        Args:
            stage: Reported stage.
            done: Steps done.
            total: Steps in the stage.
        """
        assert (stage, total) == (Stage.MESH, MESH_STEPS)
        steps.append(done)

    result = write_mesh_file(segmentation, Label.VESSEL, path, config, record)
    assert steps == list(range(MESH_STEPS + 1))
    assert len(read_stl(path)) == result.n_faces
    for cancel_at in range(MESH_STEPS):  # after the last step, the file stays
        tracker = ProgressTracker((Stage.MESH,), {Stage.MESH: 1.0})
        tracker.add_listener(cancel_after(tracker, cancel_at))
        cancelled = tmp_path / f"cancelled_{cancel_at}.stl"
        with pytest.raises(CancelledError):
            _ = write_mesh_file(segmentation, Label.VESSEL, cancelled, config, tracker)
        assert not cancelled.exists()
    assert list(tmp_path.glob(f"*{PARTIAL_SUFFIX}")) == []


def cancel_after(tracker: ProgressTracker, step: int) -> "Callable[[Progress], None]":
    """Build a listener cancelling a run once it reported ``step``.

    Args:
        tracker: Tracker to cancel.
        step: Step after which to cancel.

    Returns:
        The listener.
    """

    def listener(progress: "Progress") -> None:
        """Cancel the run at the chosen step.

        Args:
            progress: Reported progress.
        """
        if progress.done == step:
            tracker.cancel()

    return listener
