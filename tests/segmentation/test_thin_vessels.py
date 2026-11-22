"""Tests of the thin-vessel pass (:mod:`pybones.segmentation.thin_vessels`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import ndimage

from pybones.imaging import morphology as morph
from pybones.imaging.deconvolution import GaussianPsf, gaussian_blur, richardson_lucy
from pybones.segmentation.geometry import VoxelGeometry
from pybones.segmentation.labels import Label
from pybones.segmentation.pipeline import planned_stages, run_segmentation
from pybones.segmentation.stages import THIN_STAGES, no_progress
from pybones.segmentation.thin_vessels import ThinVesselParams, add_thin_vessels, iter_thick
from tests.helpers import small_config

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.segmentation.geometry import Ball

GEOMETRY = VoxelGeometry(z=3.0, y=3.0, x=3.0)
BALL = GEOMETRY.ball(6.0)  # 2 voxels
PSF = GaussianPsf((1.47, 1.47, 1.47))  # 4.4 µm at 3 µm voxels
ITERATIONS = 3
SHAPE = (12, 64, 64)
BACKGROUND_LEVEL, BRIGHT_LEVEL = 3000.0, 30000.0
TUBE = (slice(None), 20, slice(8, 56))  # one voxel thin, along z and x
PLATE = (slice(None), slice(38, 50), slice(8, 56))  # 12 voxels thick
SPECK = (6, 58, 4)


def brute_thick(mask: "NDArray[np.bool_]", ball: "Ball") -> "NDArray[np.bool_]":
    """Erode a mask by the ball in 3D with SciPy, beyond the faces counting as inside.

    Args:
        mask: ``(Z, Y, X)`` mask.
        ball: Per-slice ellipses.

    Returns:
        The voxels whose whole ball is in the mask.
    """
    reach = max(abs(dz) for dz, _ in ball)
    kernels = {dz: morph.ellipse(radius) for dz, radius in ball}
    side_y = max(k.shape[0] for k in kernels.values())
    side_x = max(k.shape[1] for k in kernels.values())
    structure = np.zeros((2 * reach + 1, side_y, side_x), bool)
    for dz, kernel in kernels.items():
        oy, ox = (side_y - kernel.shape[0]) // 2, (side_x - kernel.shape[1]) // 2
        structure[dz + reach, oy : oy + kernel.shape[0], ox : ox + kernel.shape[1]] = kernel > 0
    return ndimage.binary_erosion(mask, structure, border_value=1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_thick_voxels_match_a_3d_erosion(seed: int) -> None:
    """Streaming the erosion slice by slice equals SciPy's 3D erosion, faces included."""
    rng = np.random.default_rng(seed)
    mask = ndimage.binary_dilation(rng.random((9, 30, 30)) > 0.97, iterations=2)
    streamed = list(iter_thick(iter(mask), BALL))
    assert np.array_equal(np.stack([m for m, _ in streamed]), mask)
    assert np.array_equal(np.stack([t for _, t in streamed]), brute_thick(mask, BALL))


def test_a_stack_shorter_than_the_ball_is_still_judged() -> None:
    """Fewer slices than the ball's reach: every slice comes out, eroded within the stack."""
    mask = np.ones((2, 12, 12), bool)
    mask[:, 5, 5] = False
    thick = np.stack([t for _, t in iter_thick(iter(mask), BALL)])
    assert np.array_equal(thick, brute_thick(mask, BALL))


@pytest.fixture
def scene(tmp_path: "Path") -> "tuple[tuple[Path, Path, Path, Path], float]":
    """Write a blurred scene: a thin tube, a thick plate and a speck, all labelled bone.

    Args:
        tmp_path: Test folder.

    Returns:
        The pass's paths ``(raw, labels, candidates, final)`` and a vessel-core level that
        only the deconvolved image reaches in the tube.
    """
    truth = np.full(SHAPE, BACKGROUND_LEVEL, np.float32)
    truth[TUBE] = truth[PLATE] = BRIGHT_LEVEL
    truth[SPECK] = BRIGHT_LEVEL
    raw = gaussian_blur(truth, PSF)
    sharp = richardson_lucy(raw, PSF, ITERATIONS)
    level = float(raw[TUBE].max() + sharp[TUBE].max()) / 2
    labels = np.full(SHAPE, Label.BACKGROUND, np.uint8)
    labels[raw > (BACKGROUND_LEVEL + level) / 2] = Label.BONE
    paths = (
        tmp_path / "raw.npy",
        tmp_path / "labels.npy",
        tmp_path / "cand.npy",
        tmp_path / "out.npy",
    )
    np.save(paths[0], raw.astype(np.uint16))
    np.save(paths[1], labels)
    return paths, level


def thin_params(min_voxels: int) -> ThinVesselParams:
    """Settings for the scene, with small blocks (several of them per axis).

    Args:
        min_voxels: Smallest piece added.

    Returns:
        The settings.
    """
    return ThinVesselParams(PSF, ITERATIONS, min_voxels, BALL, (5, 24), connectivity=8)


def test_a_thin_tube_becomes_vessel_but_not_a_plate_or_a_speck(
    scene: "tuple[tuple[Path, Path, Path, Path], float]",
) -> None:
    """The tube hidden under bone is added; the thick plate and the lone speck stay as they were."""
    paths, level = scene
    stats, counts = add_thin_vessels(paths, level, thin_params(min_voxels=10), 2, no_progress)
    before, after = np.load(paths[1]), np.load(paths[3])
    assert (after[TUBE] == Label.VESSEL).mean() > 0.9
    assert np.array_equal(after[PLATE], before[PLATE])
    assert after[SPECK] == before[SPECK]
    assert stats.pieces_added == 1
    assert stats.from_bone + stats.from_background == int((after != before).sum())
    assert int(counts[Label.VESSEL]) == int((after == Label.VESSEL).sum())


def test_a_piece_smaller_than_the_minimum_is_noise(
    scene: "tuple[tuple[Path, Path, Path, Path], float]",
) -> None:
    """With a minimum above the tube's size, nothing is added."""
    paths, level = scene
    stats, _ = add_thin_vessels(paths, level, thin_params(min_voxels=10**6), 1, no_progress)
    assert stats.pieces_added == 0
    assert np.array_equal(np.load(paths[3]), np.load(paths[1]))


def test_the_pipeline_runs_the_pass_when_enabled(slice_dir: "Path", tmp_path: "Path") -> None:
    """Its stages are reported, its stats recorded, and no scratch file stays behind."""
    config = small_config("thin_vessels.enabled=true")
    assert planned_stages(config)[-len(THIN_STAGES) :] == THIN_STAGES
    assert not set(planned_stages(small_config())) & set(THIN_STAGES)
    seen: set[str] = set()

    def record(stage: str, _done: int, _total: int) -> None:
        """Note each stage reported.

        Args:
            stage: Stage reported.
            _done: Items done (unused).
            _total: Items in the stage (unused).
        """
        seen.add(stage)

    out = tmp_path / "out"
    meta = run_segmentation(slice_dir, out, config, record)
    assert meta.thin_vessels is not None
    assert set(THIN_STAGES) <= seen
    assert sorted(p.name for p in out.iterdir()) == sorted(
        (config.io.labels_file, config.io.meta_file)
    )
