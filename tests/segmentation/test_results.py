"""Tests of the segmentation metadata (:mod:`pybones.segmentation.results`)."""

import json
from typing import TYPE_CHECKING, Any

import pytest

from pybones import __version__
from pybones.segmentation.pipeline import run_segmentation
from pybones.segmentation.results import IncompatibleOutputError, load_meta
from tests.helpers import small_config

if TYPE_CHECKING:
    from pathlib import Path

    from pybones.config.schema import PyBonesConfig

OLD_VERSION = "0.2.0"


def metadata_without_a_section(
    slice_dir: "Path", tmp_path: "Path", version: str
) -> "tuple[Path, PyBonesConfig]":
    """Segment, then rewrite the metadata as an older format would: one section missing.

    Args:
        slice_dir: Synthetic slices.
        tmp_path: Temporary directory of the test.
        version: Version number to record in the metadata.

    Returns:
        The segmentation folder and its configuration.
    """
    config = small_config()
    out = tmp_path / "seg"
    _ = run_segmentation(slice_dir, out, config)
    path = out / config.io.meta_file
    data: dict[str, Any] = json.loads(path.read_text())
    del data["config"]["jobs"]  # a section an older format did not have
    data["pybones_version"] = version
    _ = path.write_text(json.dumps(data))
    return out, config


def test_older_metadata_is_refused_with_a_clear_message(
    slice_dir: "Path", tmp_path: "Path"
) -> None:
    """Metadata in an older format names both versions and says what to do."""
    out, config = metadata_without_a_section(slice_dir, tmp_path, OLD_VERSION)
    with pytest.raises(IncompatibleOutputError, match="segment the stack again") as caught:
        _ = load_meta(out, config.io)
    assert f"pyBones {OLD_VERSION}" in str(caught.value)
    assert f"pyBones {__version__}" in str(caught.value)


def test_an_earlier_build_of_the_same_version_is_named_as_such(
    slice_dir: "Path", tmp_path: "Path"
) -> None:
    """Same version number, older configuration format: the message says it was an earlier build."""
    out, config = metadata_without_a_section(slice_dir, tmp_path, __version__)
    with pytest.raises(IncompatibleOutputError, match="an earlier build of pyBones") as caught:
        _ = load_meta(out, config.io)
    assert "segment the stack again" in str(caught.value)
