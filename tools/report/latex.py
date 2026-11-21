"""Plot data (CSV for pgfplots), the history table and the measured values (LaTeX macros).

Measured numbers are written as LaTeX macros (``values.tex``) so that the text
of the report never carries a hand-copied value.
"""

import csv
import itertools
from typing import TYPE_CHECKING

import numpy as np

from pybones.io.slices import SliceStack
from pybones.segmentation.classifier import smooth
from pybones.segmentation.labels import FOREGROUND_LABELS, Label
from pybones.segmentation.scales import resolve_pixel_params
from pybones.utils.constants import MIB_PER_GIB
from tools.report.history import (
    ACCURACY_COLUMN,
    CLOTTED_RECALL_COLUMN,
    METRICS,
    REAL_PEAK_RSS_COLUMN,
    REAL_SECONDS_COLUMN,
    metric_column,
    radius_column,
)
from tools.report.meshes import mesh_values
from tools.report.network import network_values
from tools.report.readme import replace_section
from tools.report.runs import META_FILE, STATS_FILE, WORKING_TREE_REVISION, read_stats

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from tools.report.context import Context

#: Output files read by the LaTeX sources.
HISTOGRAM_CSV = "histogram.csv"
PROFILE_CSV = "profile.csv"
BURSTS_CSV = "bursts_by_slice.csv"
HISTORY_TABLE = "history_table.tex"
VALUES_TEX = "values.tex"
#: Words naming the vessel radius bands in LaTeX macros (macro names are letters only).
RADIUS_BAND_NAMES = ("Thin", "Medium", "Large")

_PERCENT = 100
#: Commit cell of today's code (its commit does not exist yet when the report is built).
_NO_COMMIT = "--"
#: The README's score table is written between these two lines.
README_SCORES_START = "<!-- scores:start (written by python -m tools.report figures) -->"
README_SCORES_END = "<!-- scores:end -->"
_PROFILE_SUFFIX = "_vessel"
#: Characters with a special meaning in LaTeX text, and their escaped form.
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_text(text: str) -> str:
    """Escape free text (CPU or system names) for LaTeX.

    Args:
        text: Raw text.

    Returns:
        The text with LaTeX special characters escaped.
    """
    return "".join(_LATEX_SPECIAL.get(c, c) for c in text)


def _write_csv(path: "Path", header: list[str], rows: "Iterable[Iterable[object]]") -> None:
    """Write a CSV file for pgfplots.

    Args:
        path: Output file (its directory is created if needed).
        header: Column names.
        rows: One iterable of values per row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def plot_data(ctx: "Context") -> None:
    """Write the CSV data of the pgfplots figures: histogram, vessel profile, bursts per slice.

    Args:
        ctx: Figure context.
    """
    fig = ctx.settings.figures
    params = resolve_pixel_params(ctx.config)
    edges = np.linspace(0, fig.histogram_max_intensity, fig.histogram_bins + 1)
    hist = np.zeros(fig.histogram_bins)
    with SliceStack(ctx.repo / ctx.settings.dataset, ctx.config.io) as stack:
        for z in range(0, len(stack), fig.histogram_stride):
            hist += np.histogram(smooth(stack.read(z), params.smoothing_sigma), bins=edges)[0]
    _write_csv(
        ctx.data / HISTOGRAM_CSV,
        ["intensity", "fraction"],
        zip((edges[:-1] + edges[1:]) / 2, hist / hist.sum(), strict=True),
    )

    prof = fig.profile
    raw = ctx.raw_slice(prof.z)
    sm = smooth(raw, params.smoothing_sigma)
    keys = (fig.legacy_key, *prof.keys)
    vessels = [ctx.slice_labels(k, prof.z)[prof.y, prof.x0 : prof.x1] == Label.VESSEL for k in keys]
    voxel_um = ctx.config.acquisition.voxel_size_um[2]
    _write_csv(
        ctx.data / PROFILE_CSV,
        ["x_um", "raw", "smoothed", *(f"{k}{_PROFILE_SUFFIX}" for k in keys)],
        (
            (
                (x - prof.x0) * voxel_um,
                int(raw[prof.y, x]),
                int(sm[prof.y, x]),
                *(int(v[i]) for v in vessels),
            )
            for i, x in enumerate(range(prof.x0, prof.x1))
        ),
    )

    fractions = [ctx.extract(k).burst_fraction for k in fig.burst_keys]
    _write_csv(
        ctx.data / BURSTS_CSV,
        ["z", *fig.burst_keys],
        ((z, *(float(f[z]) for f in fractions)) for z in range(len(fractions[0]))),
    )


def history_table(ctx: "Context") -> None:
    """Write the LaTeX table of the version history (titles from the config, scores from the CSV).

    Args:
        ctx: Figure context.
    """
    rows = ctx.history()
    lines = [
        r"\begin{tabular}{@{}l l r r r r r r@{}}",
        r"\toprule",
        r"Version & Commit & Bone & Vessel & (prec.) & (recall) & Burst & Real stack \\",
        r"\midrule",
    ]
    columns = (
        metric_column(Label.BONE, "dice"),
        metric_column(Label.VESSEL, "dice"),
        metric_column(Label.VESSEL, "precision"),
        metric_column(Label.VESSEL, "recall"),
        metric_column(Label.BURST, "dice"),
    )
    for version in ctx.settings.versions:
        row = rows[version.key]
        title = ctx.title(version.title)
        commit = (
            _NO_COMMIT
            if version.commit == WORKING_TREE_REVISION
            else f"\\texttt{{{version.commit}}}"
        )
        cost = "--"
        if row.get(REAL_SECONDS_COLUMN):
            seconds = float(row[REAL_SECONDS_COLUMN])
            gigabytes = float(row[REAL_PEAK_RSS_COLUMN]) / MIB_PER_GIB
            cost = f"{seconds:.0f}\\,s / {gigabytes:.1f}\\,GB"
        bone, vessel, precision, recall, burst = (f"{float(row[c]):.3f}" for c in columns)
        lines.append(
            f"{title} & {commit} & {bone} & \\textbf{{{vessel}}} & "
            + f"{precision} & {recall} & {burst} & {cost} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _ = (ctx.data / HISTORY_TABLE).write_text("\n".join(lines) + "\n")


def values(ctx: "Context") -> None:
    """Write the measured numbers as LaTeX macros (``pbValue...``, ``pbNum...``) for the text.

    Args:
        ctx: Figure context.

    Raises:
        ValueError: If the phantom's radius bands do not match :data:`RADIUS_BAND_NAMES`,
            or if the reference run has no up-to-date stats (run ``history`` first).
    """
    key = ctx.settings.figures.reference_key
    meta = ctx.reference_json(META_FILE)
    stats = read_stats(ctx.reference_file(STATS_FILE))
    if stats is None:
        raise ValueError(f"no up-to-date {STATS_FILE} for {key!r}: run `history` first")
    machine = stats.machine
    workers = int(meta["config"]["pipeline"]["workers"])
    th = meta["thresholds"]
    fractions = meta["label_fractions"]
    with (ctx.data / PROFILE_CSV).open() as fh:
        profile_core = max(float(row["smoothed"]) for row in csv.DictReader(fh))
    occupancy = ctx.config.segmentation.vessel.rim_occupancy
    half_max = th["background_mean"] + occupancy * (profile_core - th["background_mean"])
    entries = {
        "ThrBone": f"{th['bone']:,}",
        "ThrVessel": f"{th['vessel']:,}",
        "MeanBackground": f"{th['background_mean']:,.0f}",
        "MeanBone": f"{th['bone_mean']:,.0f}",
        "MeanVessel": f"{th['vessel_mean']:,.0f}",
        "FracBone": f"{_PERCENT * fractions[Label.BONE.key]:.1f}",
        "FracVessel": f"{_PERCENT * fractions[Label.VESSEL.key]:.2f}",
        "FracBurst": f"{_PERCENT * fractions[Label.BURST.key]:.2f}",
        "Seconds": f"{stats.seconds:.0f}",
        "PeakGB": f"{stats.peak_rss_mb / MIB_PER_GIB:.1f}",
        "Cpu": latex_text(machine.cpu),
        "Cores": f"{machine.logical_cores}",
        "MemoryGB": f"{machine.memory_gib:.0f}",
        "System": latex_text(machine.system),
        "Python": machine.python,
        "PowerProfile": latex_text(machine.power_profile),
        "StageWeightTolerance": f"{ctx.settings.stage_weight_tolerance:.2f}",
        "Workers": f"{min(workers, machine.logical_cores)}",
        "Version": meta["pybones_version"],
        "ProfileCore": f"{profile_core:,.0f}",
        "ProfileHalfMax": f"{half_max:,.0f}",
    }
    current = ctx.history()[key]
    for label in FOREGROUND_LABELS:
        for metric in METRICS:
            name = f"{label.key.capitalize()}{metric.capitalize()}"
            entries[name] = f"{float(current[metric_column(label, metric)]):.3f}"
    entries["Accuracy"] = f"{float(current[ACCURACY_COLUMN]):.3f}"
    bands = list(itertools.pairwise(ctx.phantom_settings.score.radius_bins_um))
    if len(bands) != len(RADIUS_BAND_NAMES):
        raise ValueError(f"expected {len(RADIUS_BAND_NAMES)} radius bands, got {len(bands)}")
    for (low, high), name in zip(bands, RADIUS_BAND_NAMES, strict=True):
        entries[f"VesselRecall{name}"] = f"{float(current[radius_column(low, high)]):.2f}"
    entries["ClottedRecall"] = f"{float(current[CLOTTED_RECALL_COLUMN]):.2f}"
    entries.update(mesh_values(ctx))
    entries.update(network_values(ctx))
    numbers = {  # unformatted twins, for pgfplots coordinates
        "ThrBone": th["bone"],
        "ThrVessel": th["vessel"],
        "MeanBackground": th["background_mean"],
        "MeanBone": th["bone_mean"],
        "MeanVessel": th["vessel_mean"],
        "ProfileHalfMax": half_max,
    }
    lines = [f"\\newcommand{{\\pbValue{k}}}{{{v.replace(',', '{,}')}}}" for k, v in entries.items()]
    lines += [f"\\newcommand{{\\pbNum{k}}}{{{v:.0f}}}" for k, v in numbers.items()]
    _ = (ctx.data / VALUES_TEX).write_text("\n".join(lines) + "\n")


def readme_scores(ctx: "Context") -> None:
    """Write today's phantom scores into the README, between the score markers.

    Args:
        ctx: Figure context.

    Raises:
        ValueError: If the README lacks the markers.
    """
    current = ctx.history()[ctx.settings.figures.reference_key]
    lines = [
        f"| Phantom ({ctx.version}) | Dice | Precision | Recall |",
        "|---|---|---|---|",
        *(
            f"| {label.key.capitalize()} | "
            + " | ".join(f"{float(current[metric_column(label, m)]):.3f}" for m in METRICS)
            + " |"
            for label in FOREGROUND_LABELS
        ),
    ]
    replace_section(ctx.repo / ctx.settings.readme, README_SCORES_START, README_SCORES_END, lines)
