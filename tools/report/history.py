"""Score every version of pyBones on the same phantom, and keep what the report needs.

Every version segments the *current* phantom and is scored with today's scorer,
which makes the scores comparable. Versions whose commit is not in the repository
(a fresh clone does not hold the private development history) are not re-run:
their row of the history CSV and their extract are kept as archived.
"""

import csv
import logging
from typing import TYPE_CHECKING

import numpy as np

from pybones.segmentation.labels import FINAL_LABELS, Label
from tools.phantom.scoring import score_phantom
from tools.report.extracts import (
    extract_legacy_masks,
    extract_path,
    extract_volume,
    needed_slices,
    save_extract,
)
from tools.report.runs import (
    LABELS_FILE,
    PHANTOM_RUN,
    REAL_RUN,
    is_available,
    run_dir,
    run_version,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from tools.phantom.generator import Phantom
    from tools.phantom.scoring import PhantomScore
    from tools.phantom.settings import PhantomSettings
    from tools.report.runs import RunStats
    from tools.report.settings import ReportSettings, Version

#: Columns of the history CSV (also read by the figures and by pgfplots).
KEY_COLUMN = "key"
ACCURACY_COLUMN = "accuracy"
CLOTTED_RECALL_COLUMN = "clotted_recall"
REAL_SECONDS_COLUMN = "real_seconds"
REAL_PEAK_RSS_COLUMN = "real_peak_rss_mb"
METRICS = ("dice", "precision", "recall")

_CSV_DECIMALS = 4
_MASK_GLOB = "*.png"

Row = dict[str, str | float]

logger = logging.getLogger(__name__)


def metric_column(label: Label, metric: str) -> str:
    """Name the history column of one per-label metric.

    Args:
        label: Scored label.
        metric: One of :data:`METRICS`.

    Returns:
        The column name, e.g. ``vessel_dice``.
    """
    return f"{label.key}_{metric}"


def radius_column(low_um: float, high_um: float) -> str:
    """Name the history column of the vessel recall of one radius band.

    Args:
        low_um: Lower radius of the band.
        high_um: Upper radius of the band.

    Returns:
        The column name, e.g. ``vessel_recall_0_6um``.
    """
    return f"vessel_recall_{low_um:.0f}_{high_um:.0f}um"


def score_columns(score: "PhantomScore") -> dict[str, float]:
    """Flatten a phantom score into the numeric history columns.

    Args:
        score: Score of a segmentation of the phantom.

    Returns:
        Column name -> value (accuracy, per-label metrics, recall per radius band and on clots).
    """
    columns = {ACCURACY_COLUMN: score.labels.accuracy}
    for lab in FINAL_LABELS[1:]:
        agreement = score.labels.per_label[int(lab)]
        for metric in METRICS:
            columns[metric_column(lab, metric)] = float(getattr(agreement, metric))
    for band in score.vessel_recall_by_radius:
        columns[radius_column(band.low_um, band.high_um)] = band.recall
    columns[CLOTTED_RECALL_COLUMN] = score.clotted_recall
    return columns


def score_row(
    version: "Version",
    prediction: "NDArray[np.uint8]",
    phantom: "Phantom",
    phantom_settings: "PhantomSettings",
    real_stats: "RunStats | None",
) -> Row:
    """Score a version's segmentation of the phantom.

    Args:
        version: Scored version.
        prediction: Its label volume of the phantom.
        phantom: Phantom with its ground truth.
        phantom_settings: Phantom settings (excluded value, radius bands).
        real_stats: Cost of its run on the real stack, if any.

    Returns:
        The version's row: its key and numeric columns only (titles live in the config).
    """
    score = score_phantom(
        prediction,
        phantom,
        phantom_settings.excluded_value,
        phantom_settings.score.radius_bins_um,
    )
    row: Row = {KEY_COLUMN: version.key, **score_columns(score)}
    if real_stats is not None:
        row[REAL_SECONDS_COLUMN] = real_stats.seconds
        row[REAL_PEAK_RSS_COLUMN] = real_stats.peak_rss_mb
    return row


def run_and_score(
    settings: "ReportSettings",
    version: "Version",
    phantom: "Phantom",
    phantom_settings: "PhantomSettings",
    repo: "Path",
    force: bool,
) -> Row:
    """Run one available version, score it, and refresh its extract.

    Args:
        settings: Report settings.
        version: Version to run.
        phantom: Phantom with its ground truth.
        phantom_settings: Phantom settings.
        repo: Repository root.
        force: Re-run even if outputs exist.

    Returns:
        The version's row.
    """
    phantom_slices = repo / settings.phantom_dir / phantom_settings.files.slices_dir
    stats = run_version(settings, version, phantom_slices, repo, force)
    prediction = np.load(run_dir(settings, repo, version.key, PHANTOM_RUN) / LABELS_FILE)
    if version.real:
        real = np.load(run_dir(settings, repo, version.key, REAL_RUN) / LABELS_FILE, mmap_mode="r")
        extract = extract_volume(real, needed_slices(settings))
        save_extract(extract, extract_path(settings, repo, version.key))
    return score_row(version, prediction, phantom, phantom_settings, stats.get(REAL_RUN))


def refresh_legacy_extract(settings: "ReportSettings", repo: "Path") -> None:
    """Rebuild the extract of the 2018 masks when they are present (else keep the archived one).

    Args:
        settings: Report settings (masks directory, legacy key).
        repo: Repository root.
    """
    masks = repo / settings.legacy_masks
    if not masks.is_dir():
        logger.info("No 2018 masks in %s: keeping the archived extract", masks)
        return
    depth = sum(1 for _ in next(p for p in masks.iterdir() if p.is_dir()).glob(_MASK_GLOB))
    extract = extract_legacy_masks(masks, needed_slices(settings), depth)
    save_extract(extract, extract_path(settings, repo, settings.figures.legacy_key))


def read_history(path: "Path") -> dict[str, Row]:
    """Read an existing history CSV (the archive of versions that cannot be re-run).

    Args:
        path: History CSV.

    Returns:
        Version key -> row (values as text); empty if the file does not exist.
    """
    if not path.is_file():
        return {}
    with path.open() as fh:
        return {row[KEY_COLUMN]: dict(row) for row in csv.DictReader(fh)}


def history_rows(
    settings: "ReportSettings",
    phantom: "Phantom",
    phantom_settings: "PhantomSettings",
    repo: "Path",
    force: bool,
) -> list[Row]:
    """Run and score every available version; keep the archived rows of the others.

    Args:
        settings: Report settings (versions, directories).
        phantom: Phantom with its ground truth.
        phantom_settings: Phantom settings (directories, scoring).
        repo: Repository root.
        force: Re-run versions whose outputs exist.

    Returns:
        One row per version, in the configured order.

    Raises:
        RuntimeError: If a version can neither be run nor be found in the archive.
    """
    archive = read_history(repo / settings.data_dir / settings.history_csv)
    refresh_legacy_extract(settings, repo)
    rows: list[Row] = []
    for version in settings.versions:
        if is_available(version, repo):
            row = run_and_score(settings, version, phantom, phantom_settings, repo, force)
            state = "run"
        elif version.key in archive:
            row = archive[version.key]
            state = "archived"
        else:
            raise RuntimeError(
                f"{version.key}: commit {version.commit} is not in this repository and has no "
                + "archived row"
            )
        rows.append(row)
        dice = METRICS[0]
        vessel = float(row[metric_column(Label.VESSEL, dice)])
        burst = float(row[metric_column(Label.BURST, dice)])
        logger.info("%s (%s): vessel Dice %.3f, burst Dice %.3f", version.key, state, vessel, burst)
        print(f"  {version.key:<10} {state:<8} vessel Dice {vessel:.3f}  burst Dice {burst:.3f}")
    return rows


def write_csv(rows: list[Row], path: "Path") -> None:
    """Write rows (union of keys, in first-seen order) with floats rounded.

    Args:
        rows: Rows from :func:`history_rows`.
        path: Output CSV file.
    """
    columns: list[str] = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.{_CSV_DECIMALS}f}" if isinstance(v, float) else v for k, v in row.items()}
            )
