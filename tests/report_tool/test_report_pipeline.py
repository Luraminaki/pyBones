"""End-to-end test of the report tool on synthetic data: history, extracts, figures, values."""

import json
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from pybones.segmentation.pipeline import run_segmentation
from tests.helpers import small_config, small_phantom_settings, synthetic_slice
from tools.phantom.generator import render_phantom
from tools.phantom.settings import PhantomSettings
from tools.phantom.storage import save_phantom
from tools.report import history as history_module
from tools.report import runs as runs_module
from tools.report.__main__ import FIGURE_STEPS
from tools.report.context import Context
from tools.report.extracts import extract_path, load_extract, needed_slices
from tools.report.history import KEY_COLUMN, history_rows, read_history, write_csv
from tools.report.latex import README_SCORES_END, README_SCORES_START
from tools.report.machine import current_machine
from tools.report.meshes import README_MESHES_END, README_MESHES_START
from tools.report.network import NETWORK_JSON, NETWORK_TABLE, REAL_KEY
from tools.report.runs import RunStats
from tools.report.settings import ReportSettings
from tools.report.settings import load_settings as load_report_settings

if TYPE_CHECKING:
    from pathlib import Path

    from tools.phantom.generator import Phantom
    from tools.report.settings import Version

N_SLICES = 16
CROP = {"z": 7, "y": 150, "x": 30, "size": 64}
LEGACY_SERIES = ("travee", "vaisseaux")
VOXEL = (3.0, 3.0, 3.0)
PHANTOM_SHAPE = (12, 96, 96)
#: Rows of the network score table: the true mask and the segmentation.
SCORED_ROWS = ("true mask", "segmentation")


def report_settings() -> ReportSettings:
    """Adapt the default report settings to a synthetic repository of 256 px slices.

    Returns:
        Settings with one real version (today's code) and small crops.
    """
    data: dict[str, Any] = json.loads(load_report_settings().model_dump_json())
    data["dataset"] = "stack"
    data["versions"] = [{"key": "current", "commit": "HEAD", "title": "Today", "real": True}]
    fig = data["figures"]
    fig["crops"] = {"region": CROP}
    fig["comparisons"] = {"cmp": {"crop": "region", "keys": ["legacy", "current"]}}
    fig["evolution_screenshot"] = {
        "crop": "region",
        "keys": ["legacy", "current"],
        "titles": ["raw", "2018", "today"],
    }
    fig["tubularity_crop"] = CROP
    fig["phantom_crop"] = {"z": 5, "y": 0, "x": 0, "size": 64}
    fig["profile"] = {"z": 0, "y": 150, "x0": 40, "x1": 80, "keys": ["current"]}
    fig["burst_keys"] = ["current"]
    fig["overview_slice"] = 0
    return ReportSettings.model_validate(data)


def phantom_settings() -> PhantomSettings:
    """Shrink the default phantom so that it renders in a fraction of a second.

    Returns:
        The validated settings.
    """
    return small_phantom_settings(PHANTOM_SHAPE)


def network_phantom_settings() -> PhantomSettings:
    """Shrink the default phantom and let its vessels branch, for the network step.

    Returns:
        The validated settings.
    """
    data = json.loads(phantom_settings().model_dump_json())
    data["vessels"].update(count=4, branch_probability=0.2, branch_depth=1)
    return PhantomSettings.model_validate(data)


def fake_segment(
    _settings: ReportSettings, _project: "Path", source: "Path", out_dir: "Path"
) -> RunStats:
    """Segment in-process with today's code instead of a ``uv run`` subprocess.

    Args:
        _settings: Report settings (unused).
        _project: Project to run (unused: today's code runs).
        source: Slices to segment.
        out_dir: Output directory.

    Returns:
        Fixed run stats on this machine.
    """
    _ = run_segmentation(source, out_dir, small_config())
    return RunStats(seconds=1.0, peak_rss_mb=1.0, machine=current_machine())


@pytest.fixture
def repo(tmp_path: "Path", monkeypatch: pytest.MonkeyPatch) -> "Path":
    """Build a synthetic repository: stack, phantom and 2018 masks.

    Args:
        tmp_path: Temporary directory of the test.
        monkeypatch: Replaces the subprocess segmentation.

    Returns:
        The repository root.
    """
    root = tmp_path / "repo"
    settings = report_settings()
    stack = root / settings.dataset
    stack.mkdir(parents=True)
    for z in range(N_SLICES):
        _ = cv2.imwrite(str(stack / f"img_{z:02d}.png"), synthetic_slice(seed=z))
    ph_settings = phantom_settings()
    save_phantom(render_phantom(ph_settings, VOXEL), root / settings.phantom_dir, ph_settings)
    network_settings = root / settings.network.phantom_settings
    network_settings.parent.mkdir(parents=True, exist_ok=True)
    _ = network_settings.write_text(network_phantom_settings().model_dump_json())
    masks = root / settings.legacy_masks
    for name in LEGACY_SERIES:
        (masks / name).mkdir(parents=True)
        for z in range(N_SLICES):
            mask = np.zeros((256, 256), np.uint8)
            mask[20:60] = 255 if name == LEGACY_SERIES[0] else 0
            _ = cv2.imwrite(str(masks / name / f"{name}_{z:03d}.png"), mask)
    _ = (root / settings.readme).write_text(
        f"# Scores\n{README_SCORES_START}\nold table\n{README_SCORES_END}\n"
        + f"# Meshes\n{README_MESHES_START}\nold meshes\n{README_MESHES_END}\nafter\n"
    )
    monkeypatch.setattr(runs_module, "segment", fake_segment)
    return root


def run_history(repo: "Path") -> list[dict[str, str | float]]:
    """Run the history step on the synthetic repository and write its CSV.

    Args:
        repo: Synthetic repository root.

    Returns:
        The history rows.
    """
    settings = report_settings()
    ph_settings = phantom_settings()
    phantom: Phantom = render_phantom(ph_settings, VOXEL)
    rows = history_rows(settings, phantom, ph_settings, repo, force=False)
    write_csv(rows, repo / settings.data_dir / settings.history_csv)
    return rows


def test_history_writes_scores_and_extracts(repo: "Path") -> None:
    """Every version gets a row and an extract holding the slices the figures need."""
    settings = report_settings()
    rows = run_history(repo)
    assert [row[KEY_COLUMN] for row in rows] == ["current"]
    extract = load_extract(extract_path(settings, repo, "current"))
    assert tuple(extract.z.tolist()) == needed_slices(settings)
    assert extract.burst_fraction.shape == (N_SLICES,)
    legacy = load_extract(extract_path(settings, repo, settings.figures.legacy_key))
    assert legacy.labels(0)[30, 30] == 1  # bone band of the synthetic 2018 masks


def test_every_figure_step_runs_on_the_extracts(repo: "Path") -> None:
    """The figures, plot data, table and values are written from the history's outputs."""
    settings = report_settings()
    _ = run_history(repo)
    ctx = Context(
        settings=settings, config=small_config(), phantom_settings=phantom_settings(), repo=repo
    )
    for step in FIGURE_STEPS:
        step(ctx)
    figures = {p.stem for p in ctx.figures.iterdir()}
    assert {
        "overview_current",
        "cmp_raw",
        "cmp_legacy",
        "vessel_network",
        "phantom_truth",
    } <= figures
    values = (ctx.data / "values.tex").read_text()
    assert "\\pbValueCpu" in values
    assert "\\pbValueVesselDice" in values
    assert "\\pbValueMeshVesselGlbMB" in values
    assert "\\pbValueMeshErrorVoxelRatio" in values
    assert "\\pbValueNetBranches" in values
    assert "\\pbValueNetSegRadiusThick" in values
    record = json.loads((ctx.data / NETWORK_JSON).read_text())
    assert record[REAL_KEY]["branches"] > 0
    assert (ctx.data / NETWORK_TABLE).read_text().count("\\\\") == len(SCORED_ROWS) + 1
    readme = (repo / settings.readme).read_text()
    assert "old table" not in readme
    assert "| Vessel |" in readme
    assert "old meshes" not in readme
    assert "| GLB, 16-bit |" in readme
    assert readme.endswith("after\n")


def test_unavailable_versions_keep_their_archived_rows(
    repo: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version whose commit is gone is not re-run: its archived row is kept, or it is an error."""
    settings = report_settings()
    rows = run_history(repo)

    def unavailable(_version: "Version", _repo: "Path") -> bool:
        """Pretend no commit is available.

        Args:
            _version: Version (unused).
            _repo: Repository (unused).

        Returns:
            Always ``False``.
        """
        return False

    monkeypatch.setattr(history_module, "is_available", unavailable)
    archived = run_history(repo)
    csv_path = repo / settings.data_dir / settings.history_csv
    assert read_history(csv_path)["current"] == archived[0]
    assert float(archived[0]["vessel_dice"]) == pytest.approx(
        float(rows[0]["vessel_dice"]), abs=1e-4
    )
    csv_path.unlink()
    with pytest.raises(RuntimeError, match="no archived row"):
        _ = run_history(repo)
