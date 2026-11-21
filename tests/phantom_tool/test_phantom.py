"""Tests of the synthetic phantom and its scorer (:mod:`tools.phantom`)."""

import json
import logging
from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from pybones.cli.main import main as pybones_main
from pybones.segmentation.labels import FINAL_LABELS, Label
from tests.helpers import cli_overrides
from tools.phantom.__main__ import main as phantom_main
from tools.phantom.generator import render_phantom
from tools.phantom.scoring import score_phantom
from tools.phantom.settings import PhantomSettings, load_settings
from tools.phantom.storage import load_phantom, save_phantom
from tools.phantom.vessels import SegmentState, VesselNetwork

if TYPE_CHECKING:
    from pathlib import Path

    from tools.phantom.generator import Phantom

VOXEL = (3.0, 3.0, 3.0)
#: Logger of the phantom command line (``tools/phantom/__main__.py``).
PHANTOM_CLI_LOGGER = "tools.phantom.__main__"
SHAPE = (24, 96, 96)


def small_settings(**updates: object) -> PhantomSettings:
    """Shrink the default phantom so that it renders in a fraction of a second.

    Args:
        **updates: Top-level settings to replace.

    Returns:
        The validated settings.
    """
    data = json.loads(load_settings().model_dump_json())
    data["shape_zyx"] = list(SHAPE)
    data["trabeculae"].update(plates=3, rods=4)
    data["vessels"].update(count=6)
    data["bursts"].update(count=1, radius_um=[40, 50])
    data.update(updates)
    return PhantomSettings.model_validate(data)


@pytest.fixture(scope="module")
def phantom() -> "Phantom":
    """Render the small phantom once for the module.

    Returns:
        The phantom.
    """
    return render_phantom(small_settings(), VOXEL)


def test_phantom_is_reproducible_and_complete(phantom: "Phantom") -> None:
    """The same seed gives the same image, and every final label is present."""
    again = render_phantom(small_settings(), VOXEL)
    np.testing.assert_array_equal(phantom.image, again.image)
    assert phantom.image.dtype == np.uint16
    assert phantom.image.shape == phantom.truth.shape == SHAPE
    present = set(np.unique(phantom.truth).tolist())
    assert {int(lab) for lab in FINAL_LABELS} <= present


def test_truth_metadata_is_consistent(phantom: "Phantom") -> None:
    """Radii and clot flags exist exactly on truth-vessel voxels."""
    vessel = phantom.truth == Label.VESSEL
    assert (phantom.vessel_radius_um[vessel] > 0).all()
    assert (phantom.vessel_radius_um[~vessel] == 0).all()
    assert not (phantom.clotted & ~vessel).any()


def test_intensities_follow_the_classes(phantom: "Phantom") -> None:
    """Background and bone intensities match the configured levels."""
    ints = load_settings().intensity
    background = np.median(phantom.image[phantom.truth == Label.BACKGROUND])
    bone = np.median(phantom.image[phantom.truth == Label.BONE])
    assert abs(background - ints.background) < 3 * ints.noise_std
    assert abs(bone - ints.bone) < 0.15 * ints.bone  # partial volume lowers bone a bit


def test_scoring_the_truth_is_perfect(phantom: "Phantom") -> None:
    """Scoring the truth against itself gives perfect scores."""
    settings = small_settings()
    excluded = phantom.truth == settings.excluded_value
    pred = np.where(excluded, Label.BACKGROUND, phantom.truth).astype(np.uint8)
    score = score_phantom(pred, phantom, settings.excluded_value, settings.score.radius_bins_um)
    assert all(a.dice == 1.0 for a in score.labels.per_label.values() if a.tp)
    assert all(r.recall == 1.0 for r in score.vessel_recall_by_radius if r.voxels)


def test_invalid_ranges_are_rejected() -> None:
    """A ``(low, high)`` range with low > high fails validation."""
    with pytest.raises(ValidationError, match="low > high"):
        _ = small_settings(cortex={"count": 1, "thickness_um": [200, 100]})


def test_storage_roundtrip(phantom: "Phantom", tmp_path: "Path") -> None:
    """Saved slices are readable images and the ground truth reads back identically."""
    settings = small_settings()
    save_phantom(phantom, tmp_path, settings)
    assert len(list((tmp_path / settings.files.slices_dir).iterdir())) == SHAPE[0]
    loaded = load_phantom(tmp_path, settings)
    np.testing.assert_array_equal(loaded.truth, phantom.truth)
    np.testing.assert_array_equal(loaded.vessel_radius_um, phantom.vessel_radius_um)
    np.testing.assert_array_equal(loaded.clotted, phantom.clotted)
    for name in (f.name for f in fields(VesselNetwork)):
        np.testing.assert_array_equal(getattr(loaded.network, name), getattr(phantom.network, name))


def test_generate_segment_and_score_cli(tmp_path: "Path", caplog: pytest.LogCaptureFixture) -> None:
    """The CLI renders a phantom that pybones segments and the scorer evaluates."""
    settings_path = tmp_path / "phantom.json"
    _ = settings_path.write_text(small_settings().model_dump_json())
    phantom_dir, seg = tmp_path / "phantom", tmp_path / "seg"
    phantom_main(["--settings", str(settings_path), "generate", "-o", str(phantom_dir)])
    slices = phantom_dir / small_settings().files.slices_dir
    pybones_main([*cli_overrides(), "segment", str(slices), "-o", str(seg)])
    caplog.clear()
    with caplog.at_level(logging.INFO):
        phantom_main(
            ["--settings", str(settings_path), "score", str(seg), "--phantom", str(phantom_dir)]
        )
    scores = [r.getMessage() for r in caplog.records if r.name == PHANTOM_CLI_LOGGER]
    assert len(scores) == 1
    assert "\naccuracy " in scores[0]
    assert "vessel recall by radius" in scores[0]


#: Settings that make every vessel segment likely to fork.
BRANCHING = {"branch_probability": 0.5, "branch_depth": 2}


def vessel_settings(**vessels: object) -> PhantomSettings:
    """Build the small phantom settings with some vessel settings replaced.

    Args:
        **vessels: Vessel settings to replace.

    Returns:
        The validated settings.
    """
    data = json.loads(small_settings().model_dump_json())
    data["vessels"].update(vessels)
    return PhantomSettings.model_validate(data)


def test_unbranched_walks_record_their_segments(phantom: "Phantom") -> None:
    """Without branching, each vessel is one branch of segments of the configured length."""
    settings = small_settings()
    network = phantom.network
    assert len(network.branch_parent) == settings.vessels.count
    assert (network.branch_parent == -1).all()
    assert len(network.junctions_um) == 0
    lengths = np.linalg.norm(network.segments_um[:, 1] - network.segments_um[:, 0], axis=1)
    np.testing.assert_allclose(lengths, settings.vessels.segment_length_um, rtol=1e-5)
    low, high = settings.vessels.radius_um
    assert ((network.radius_um >= low) & (network.radius_um <= high)).all()
    assert set(np.unique(network.state)) <= {int(s) for s in SegmentState}


def test_branching_off_does_not_change_the_phantom(phantom: "Phantom") -> None:
    """Branch settings are ignored at probability 0: the image is the same, draw for draw."""
    other = render_phantom(
        vessel_settings(branch_probability=0.0, branch_depth=5, branch_angle_deg=[10, 20]), VOXEL
    )
    np.testing.assert_array_equal(other.image, phantom.image)


def test_branching_draws_trees_and_records_them() -> None:
    """Each fork is a junction at the end of a drawn parent segment; branches are thinner."""
    network = render_phantom(vessel_settings(**BRANCHING), VOXEL).network
    parents = network.branch_parent
    side = np.flatnonzero(parents >= 0)
    assert len(side) == len(network.junctions_um) > 0
    assert (parents[side] < side).all()  # a parent exists before its branches
    # Junctions are recorded in the order the side branches start.
    for branch, junction in zip(side, network.junctions_um, strict=True):
        parent_ends = network.segments_um[network.branch == parents[branch]][:, 1]
        assert np.isclose(parent_ends, junction, atol=1e-3).all(axis=1).any()
    radius = {
        int(b): float(network.radius_um[network.branch == b][0]) for b in np.unique(network.branch)
    }
    for branch in side:
        if int(branch) in radius and int(parents[branch]) in radius:
            assert radius[int(branch)] <= radius[int(parents[branch])]


def test_an_old_phantom_says_to_generate_it_again(phantom: "Phantom", tmp_path: "Path") -> None:
    """A truth file without the vessel network is refused with what to do."""
    settings = small_settings()
    save_phantom(phantom, tmp_path, settings)
    path = tmp_path / settings.files.truth
    with np.load(path) as data:
        kept = {k: data[k] for k in ("truth", "vessel_radius_um", "clotted")}
    np.savez_compressed(path, **kept)
    with pytest.raises(ValueError, match="generate it again"):
        _ = load_phantom(tmp_path, settings)
