"""Tests of configuration loading and validation (:mod:`pybones.config`)."""

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from pybones.config.loader import (
    apply_overrides,
    default_config_text,
    load_config,
    load_server_config,
    override_section,
)
from pybones.segmentation.labels import Label
from pybones.segmentation.stages import MESH_STAGES, STAGES_3D

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
        'io.labels_file="sub/labels.npy"',  # a path, not a file name
        'io.meta_file="..\\\\meta.json"',  # a Windows path
        'io.mask_png_template="png/{label}_{index:04d}.png"',  # a path, not a file name
        "mesh.taubin.mu_factor=-0.4",  # Taubin needs mu < -lambda
        "mesh.glb_position_bits=24",  # 16 or 32
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


def test_stage_weights_cover_the_segmentation_stages_only() -> None:
    """Segmentation and meshing weigh their own stages (mesh jobs have their own weights)."""
    weights = load_config().pipeline.stage_weights
    assert set(weights) == set(STAGES_3D)
    assert not set(weights) & set(MESH_STAGES)
    assert set(load_config().mesh.stage_weights) == set(MESH_STAGES)
    with pytest.raises(ValidationError, match="weight exactly"):
        _ = load_config(overrides=['mesh.stage_weights={"occupancy": 0.5, "surface": 0.5}'])
    partial = json.dumps({s.value: 1.0 for s in STAGES_3D[1:]})
    with pytest.raises(ValidationError, match="misses"):
        _ = load_config(overrides=[f"pipeline.stage_weights={partial}"])


def test_apply_overrides_derives_a_new_config() -> None:
    """Overrides apply on top of an existing configuration, which stays unchanged."""
    base = load_config(overrides=["pipeline.workers=3"])
    derived = apply_overrides(base, ["refinement.enabled=false"])
    assert (derived.pipeline.workers, derived.refinement.enabled) == (3, False)
    assert base.refinement.enabled is True
    with pytest.raises(ValueError, match="unknown key"):
        _ = apply_overrides(base, ["refinement.threads=2"])


def test_override_section_names_the_top_level_key() -> None:
    """The section of an override is its first dotted key."""
    assert override_section("segmentation.burst.density=0.3") == "segmentation"
    assert override_section("io") == "io"


def test_server_config_loads_and_merges(tmp_path: "Path") -> None:
    """``server.json`` validates; a user file and overrides change only their values."""
    default = load_server_config()
    assert default.http.host == "127.0.0.1"
    user = tmp_path / "server.json"
    _ = user.write_text(json.dumps({"library": {"data_dir": "/srv/stacks"}}))
    server = load_server_config(user, ["http.port=8080"])
    assert (server.http.port, server.library.data_dir) == (8080, "/srv/stacks")
    assert server.library.output_dir == default.library.output_dir


@pytest.mark.parametrize(
    "override",
    [
        "http.port=0",  # < 1
        "http.port=70000",  # > 65535
        'library.mesh_file_template="mesh_{label}.stl"',  # no {format} extension
        'library.mesh_file_template="mesh.{format}"',  # no {label}
        'library.mesh_file_template="meshes/{label}.{format}"',  # not a plain file name
        'api.default_mesh_format="obj"',  # not a mesh format
        "api.max_preview_scale=0",  # must be > 0
        "api.max_preview_scale=2",  # enlarging is the browser's job
        "api.event_heartbeat_s=0",  # must be > 0
    ],
)
def test_invalid_server_values_are_rejected(override: str) -> None:
    """The server configuration has bounds like the main one."""
    with pytest.raises(ValidationError):
        _ = load_server_config(overrides=[override])
