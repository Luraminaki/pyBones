"""Tests of configuration loading and validation (:mod:`pybones.config`)."""

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from pybones.config.loader import default_config_text, load_config
from pybones.segmentation.labels import Label

if TYPE_CHECKING:
    from pathlib import Path


def test_default_config_is_valid_and_frozen() -> None:
    """The packaged configuration validates and cannot be modified."""
    config = load_config()
    assert config.segmentation.connectivity in (4, 8)
    assert config.mesh.factor_for(Label.VESSEL) >= 1
    with pytest.raises(ValidationError):
        config.pipeline.workers = 3


def test_user_file_is_deep_merged(tmp_path: "Path") -> None:
    """A user file only needs the values it changes."""
    user = tmp_path / "user.json"
    _ = user.write_text(json.dumps({"segmentation": {"burst": {"density": 0.5}}}))
    config = load_config(user)
    assert config.segmentation.burst.density == 0.5
    default = json.loads(default_config_text())
    window = default["segmentation"]["burst"]["window_xy_um"]
    assert config.segmentation.burst.window_xy_um == window


def test_overrides_parse_json_values() -> None:
    """Override values are parsed as JSON."""
    config = load_config(overrides=["pipeline.workers=1", "refinement.enabled=false"])
    assert config.pipeline.workers == 1
    assert config.refinement.enabled is False


@pytest.mark.parametrize(
    "override",
    [
        "acquisition.voxel_size_um=[0, 3, 3]",  # voxel size must be > 0
        "segmentation.vessel.rim_occupancy=1.5",  # > 1
        "pipeline.workers=0",  # < 1
        "segmentation.connectivity=6",  # not 4/8
        'mesh.downsample={"bone": 2}',  # missing labels
        "render.low_percentile=99.9",  # >= high percentile
        'io.labels_file="labels.txt"',  # wrong extension
        "mesh.taubin.mu_factor=-0.4",  # Taubin needs mu < -lambda
    ],
)
def test_out_of_range_values_are_rejected(override: str) -> None:
    """Types, bounds and cross-field rules are enforced."""
    with pytest.raises(ValidationError):
        _ = load_config(overrides=[override])


def test_unknown_keys_are_rejected(tmp_path: "Path") -> None:
    """Unknown keys fail, whether given as an override or in a user file."""
    with pytest.raises(ValueError, match="unknown key"):
        _ = load_config(overrides=["pipeline.threads=2"])
    user = tmp_path / "user.json"
    _ = user.write_text(json.dumps({"pipeline": {"threads": 2}}))
    with pytest.raises(ValidationError):
        _ = load_config(user)
