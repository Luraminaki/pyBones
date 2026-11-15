"""Tests of the report tool (:mod:`tools.report`)."""

import csv
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import ValidationError

from tools.report.drawing import add_scale_bar, compose
from tools.report.history import write_csv
from tools.report.latex import latex_text
from tools.report.machine import current_machine
from tools.report.runs import WORKING_TREE_REVISION, RunStats, read_stats
from tools.report.settings import ReportSettings, load_settings

if TYPE_CHECKING:
    from pathlib import Path


def settings_data() -> dict[str, Any]:
    """Dump the default settings as plain JSON data, ready to be modified.

    Returns:
        The settings as a dict.
    """
    return json.loads(load_settings().model_dump_json())


def test_default_settings_are_valid() -> None:
    """The shipped settings validate; the reference version runs from the working tree."""
    settings = load_settings()
    reference = settings.version(settings.figures.reference_key)
    assert reference.commit == WORKING_TREE_REVISION
    with pytest.raises(KeyError):
        _ = settings.version("nope")


def test_duplicate_version_keys_are_rejected() -> None:
    """Version keys must be unique."""
    data = settings_data()
    data["versions"].append(data["versions"][0])
    with pytest.raises(ValidationError, match="unique"):
        _ = ReportSettings.model_validate(data)


def test_figures_must_name_known_versions_and_crops() -> None:
    """Figures may only show configured versions (or the legacy masks) and crops."""
    data = settings_data()
    data["figures"]["burst_keys"] = ["nope"]
    with pytest.raises(ValidationError, match="unknown versions"):
        _ = ReportSettings.model_validate(data)
    data = settings_data()
    data["figures"]["evolution_screenshot"]["crop"] = "nope"
    with pytest.raises(ValidationError, match="unknown crop"):
        _ = ReportSettings.model_validate(data)


def test_write_csv_unions_columns_and_rounds(tmp_path: "Path") -> None:
    """Columns are the union of the rows' keys; floats are rounded."""
    path = tmp_path / "h.csv"
    write_csv([{"key": "a", "x": 0.123456}, {"key": "b", "y": 2.0}], path)
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0] == {"key": "a", "x": "0.1235", "y": ""}
    assert rows[1]["y"] == "2.0000"


def test_compose_and_scale_bar() -> None:
    """Composed panels are padded to the tallest; the scale bar is drawn at the bottom."""
    fig = load_settings().figures
    panels = [np.zeros((40, 30, 3), np.uint8), np.zeros((50, 20, 3), np.uint8)]
    strip = compose(panels, ["a", "b"], fig)
    assert strip.shape == (50 + fig.title_height_px, 30 + fig.panel_gap_px + 20, 3)
    image = np.zeros((200, 400, 3), np.uint8)
    add_scale_bar(image, 0.1, fig)  # 1 mm at 0.1 px/µm = 100 px
    assert image[-30:].any()


def test_stats_record_the_machine_and_old_stats_are_rerun(tmp_path: "Path") -> None:
    """Stats round-trip with their machine; stats without one (older format) count as missing."""
    path = tmp_path / "stats.json"
    assert read_stats(path) is None
    _ = path.write_text(json.dumps({"seconds": 1.0, "peak_rss_mb": 2.0}))
    assert read_stats(path) is None
    stats = RunStats(seconds=1.0, peak_rss_mb=2.0, machine=current_machine())
    _ = path.write_text(stats.model_dump_json())
    assert read_stats(path) == stats
    assert stats.machine.logical_cores >= 1


def test_latex_text_escapes_special_characters() -> None:
    """Free text (CPU or system names) cannot break the LaTeX build."""
    assert latex_text("Intel_Core #1 & 50% ~$") == r"Intel\_Core \#1 \& 50\% \textasciitilde{}\$"
