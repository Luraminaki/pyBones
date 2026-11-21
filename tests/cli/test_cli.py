"""End-to-end tests of the ``pybones`` command (:mod:`pybones.cli.main`)."""

import json
from typing import TYPE_CHECKING, cast

import cv2
import pytest

from pybones.cli import main as cli_main
from pybones.cli.main import INTERRUPTED_EXIT_STATUS, main
from pybones.utils.constants import BGR_CHANNELS
from tests.helpers import N_SYNTHETIC_SLICES, SLICE_SHAPE, cli_overrides

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

PREVIEW_SCALE = 0.5


def run(*args: str) -> None:
    """Run ``pybones`` with the synthetic-slice overrides.

    Args:
        *args: Command and arguments.
    """
    main([*cli_overrides(), *args])


def test_config_command_prints_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``pybones config`` prints the effective configuration, overrides applied."""
    main(["--set", "pipeline.workers=1", "config"])
    assert json.loads(capsys.readouterr().out)["pipeline"]["workers"] == 1


def test_invalid_config_exits_with_usage_error() -> None:
    """An out-of-range override exits with argparse's usage error."""
    with pytest.raises(SystemExit):
        main(["--set", "segmentation.vessel.rim_um=-1", "config"])


def test_full_cli_workflow(slice_dir: "Path", tmp_path: "Path") -> None:
    """Segment, preview, mesh, export and compare, end to end."""
    seg = tmp_path / "seg"
    preview = tmp_path / "p.png"
    run("segment", str(slice_dir), "-o", str(seg))
    run("preview", str(seg), "-z", "3", "-o", str(preview), "--scale", str(PREVIEW_SCALE))
    image = cast("NDArray[np.uint8]", cv2.imread(str(preview)))
    expected = (*(round(s * PREVIEW_SCALE) for s in SLICE_SHAPE), BGR_CHANNELS)
    assert image.shape == expected
    run("mesh", str(seg), "--label", "bone", "-o", str(tmp_path / "bone.stl"))
    assert (tmp_path / "bone.stl").stat().st_size > 0
    run("export", str(seg), "--label", "vessel", "-o", str(tmp_path / "png"))
    assert len(list((tmp_path / "png").iterdir())) == N_SYNTHETIC_SLICES
    run("compare", str(seg), "--label", "vessel", "--reference", str(tmp_path / "png"))


def test_background_is_not_a_valid_label(tmp_path: "Path") -> None:
    """Only foreground labels can be meshed."""
    with pytest.raises(SystemExit):
        main(["mesh", str(tmp_path), "--label", "background", "-o", "x.stl"])


def interrupt(*_args: object) -> None:
    """Stand in for a command stopped by Ctrl+C.

    Args:
        _args: The command's arguments (ignored).

    Raises:
        KeyboardInterrupt: Always.
    """
    raise KeyboardInterrupt


def test_ctrl_c_exits_quietly_with_status_130(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted command exits with the shell's status for Ctrl+C, not a traceback."""
    monkeypatch.setattr(cli_main, "run_command", interrupt)
    with pytest.raises(SystemExit) as exit_info:
        main(["config"])
    assert exit_info.value.code == INTERRUPTED_EXIT_STATUS
    assert exit_info.value.__cause__ is None
