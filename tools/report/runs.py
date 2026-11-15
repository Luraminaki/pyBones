"""Running a version of pyBones: git worktrees, timed segmentations and their stored outputs.

Each version is checked out in its own git worktree, with its own virtual
environment (``uv run --project``), so its code runs exactly as it was. Outputs
are cached under ``runs_dir/<key>/<run>/`` (not versioned: they weigh hundreds of
MB); what the report needs from them is kept as small extracts (see
:mod:`tools.report.extracts`).
"""

import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from pybones.utils.constants import KIB_PER_MIB
from tools.report.machine import Machine, current_machine

if TYPE_CHECKING:
    from pathlib import Path

    from tools.report.settings import ReportSettings, Version

#: Segmentation outputs of every version share this layout (unchanged since 0.1).
LABELS_FILE = "labels.npy"
META_FILE = "meta.json"
#: Cost of a run, written next to its outputs (see :class:`RunStats`).
STATS_FILE = "stats.json"
#: Sub-directories of a version's runs.
PHANTOM_RUN = "phantom"
REAL_RUN = "real"
#: A version at this revision runs from the repository itself (uncommitted changes included).
WORKING_TREE_REVISION = "HEAD"

_SEGMENT_COMMAND = "segment"
_OUTPUT_OPTION = "-o"

logger = logging.getLogger(__name__)


class RunStats(BaseModel):
    """Cost of one segmentation run (the whole child process tree), and where it was measured.

    Attributes:
        seconds: Wall time.
        peak_rss_mb: Peak resident memory, in MiB.
        machine: Machine the run was timed on.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    seconds: float
    peak_rss_mb: float
    machine: Machine


def read_stats(path: "Path") -> RunStats | None:
    """Read the stats stored next to a run.

    Args:
        path: :data:`STATS_FILE` of the run.

    Returns:
        The stats, or ``None`` if the file is missing or predates the current format
        (e.g. without the machine description): the run must then be timed again.
    """
    if not path.is_file():
        return None
    try:
        return RunStats.model_validate_json(path.read_text())
    except ValidationError:
        logger.warning("Ignoring outdated run stats %s", path)
        return None


def run_dir(settings: "ReportSettings", repo: "Path", key: str, run: str) -> "Path":
    """Locate the cached outputs of one run.

    Args:
        settings: Report settings (runs directory).
        repo: Repository root.
        key: Version key.
        run: :data:`PHANTOM_RUN` or :data:`REAL_RUN`.

    Returns:
        The run's output directory.
    """
    return repo / settings.runs_dir / key / run


def is_available(version: "Version", repo: "Path") -> bool:
    """Tell whether a version's code can still be checked out.

    A fresh clone does not hold the commits of the private development history:
    such versions keep their archived scores and extracts instead of being re-run.

    Args:
        version: Version to check.
        repo: Repository root.

    Returns:
        ``True`` for the working tree, or for a commit known to this repository.
    """
    if version.commit == WORKING_TREE_REVISION:
        return True
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{version.commit}^{{commit}}",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def worktree(settings: "ReportSettings", version: "Version", repo: "Path") -> "Path":
    """Check ``version`` out in its own worktree (reused if present).

    Args:
        settings: Report settings (worktree directory).
        version: Version to check out.
        repo: Repository root.

    Returns:
        The worktree directory.

    Raises:
        subprocess.CalledProcessError: If git cannot create the worktree.
    """
    path = repo / settings.worktrees_dir / version.key
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Checking out %s (%s) in %s", version.key, version.commit, path)
        _ = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), version.commit],
            check=True,
            capture_output=True,
        )
    return path


def segment(
    settings: "ReportSettings", project: "Path", source: "Path", out_dir: "Path"
) -> RunStats:
    """Segment ``source`` with the pybones of ``project`` (its own environment).

    Args:
        settings: Report settings (uv executable, console script name).
        project: Project directory whose pybones is run.
        source: Slices to segment.
        out_dir: Output directory of the segmentation.

    Returns:
        Wall time and peak resident memory of the run, on this machine.

    Raises:
        RuntimeError: If the segmentation fails.
    """
    command = [
        settings.uv_executable,
        "run",
        "--quiet",
        "--project",
        str(project),
        settings.cli_name,
        _SEGMENT_COMMAND,
        str(source),
        _OUTPUT_OPTION,
        str(out_dir),
    ]
    logger.info("Running %s", " ".join(command))
    t0 = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, usage = os.wait4(proc.pid, 0)
    elapsed = time.perf_counter() - t0
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f"Segmentation failed: {' '.join(command)}")
    stats = RunStats(
        seconds=elapsed, peak_rss_mb=usage.ru_maxrss / KIB_PER_MIB, machine=current_machine()
    )
    logger.info("Segmented %s in %.1f s, peak %.0f MiB", source, stats.seconds, stats.peak_rss_mb)
    return stats


def run_version(
    settings: "ReportSettings",
    version: "Version",
    phantom_slices: "Path",
    repo: "Path",
    force: bool,
) -> dict[str, RunStats]:
    """Segment the phantom (and the real data if requested) with one version.

    Existing outputs are reused unless ``force``, or unless their :data:`STATS_FILE`
    is missing or outdated (see :func:`read_stats`).

    Args:
        settings: Report settings.
        version: Version to run (must be available, see :func:`is_available`).
        phantom_slices: Slices of the phantom.
        repo: Repository root.
        force: Re-run even if outputs exist.

    Returns:
        Run name (:data:`PHANTOM_RUN`, :data:`REAL_RUN`) -> cost of the run.
    """
    project = worktree(settings, version, repo) if version.commit != WORKING_TREE_REVISION else repo
    inputs = {PHANTOM_RUN: phantom_slices}
    if version.real:
        inputs[REAL_RUN] = repo / settings.dataset
    stats: dict[str, RunStats] = {}
    for name, source in inputs.items():
        out_dir = run_dir(settings, repo, version.key, name)
        stats_file = out_dir / STATS_FILE
        run = None if force or not (out_dir / LABELS_FILE).exists() else read_stats(stats_file)
        if run is None:
            run = segment(settings, project, source, out_dir)
            _ = stats_file.write_text(run.model_dump_json())
        else:
            logger.debug("Reusing the %s run of %s", name, version.key)
        stats[name] = run
    return stats
