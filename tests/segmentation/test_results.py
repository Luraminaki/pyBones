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

OLD_VERSION = "0.2.0"


def test_older_metadata_is_refused_with_a_clear_message(
    slice_dir: "Path", tmp_path: "Path"
) -> None:
    """Metadata in an older format names both versions and says what to do."""
    config = small_config()
    out = tmp_path / "seg"
    _ = run_segmentation(slice_dir, out, config)
    path = out / config.io.meta_file
    data: dict[str, Any] = json.loads(path.read_text())
    del data["config"]["jobs"]  # a section an older version did not have
    data["pybones_version"] = OLD_VERSION
    _ = path.write_text(json.dumps(data))
    with pytest.raises(IncompatibleOutputError, match="segment the stack again") as caught:
        _ = load_meta(out, config.io)
    assert f"pyBones {OLD_VERSION}" in str(caught.value)
    assert f"pyBones {__version__}" in str(caught.value)
