"""The vascular network of the report: the real sample's measures and the phantom's scores.

:func:`network_files` quantifies the reference run's real segmentation, then generates the
branching phantom (``network.phantom_settings``) afresh, segments it, and scores the network
measured on its true mask and on its segmentation against the true one. Everything measured
lands in ``network.json`` and two LaTeX tables; :func:`network_values` turns the file into
the report's macros, so that no network figure is typed by hand.
"""

import itertools
import json
import math
import shutil
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from pybones.quantification.measures import NetworkSummary
from pybones.quantification.pipeline import measure_network, run_quantification
from pybones.segmentation.pipeline import run_segmentation
from pybones.segmentation.results import load_labels
from pybones.utils.constants import JSON_INDENT
from tools.phantom.generator import render_phantom
from tools.phantom.network_scoring import NetworkScore, score_network
from tools.phantom.settings import load_settings as load_phantom_settings
from tools.phantom.storage import save_phantom
from tools.report.runs import META_FILE, REAL_RUN, run_dir

if TYPE_CHECKING:
    from tools.phantom.generator import Phantom
    from tools.report.context import Context

NETWORK_JSON = "network.json"
NETWORK_TABLE = "network_scores.tex"
NETWORK_BANDS_TABLE = "network_bands.tex"
#: Keys of ``network.json``.
REAL_KEY = "real"
REAL_SECONDS_KEY = "real_seconds"
TRUE_MASK_KEY = "true_mask"
SEGMENTED_KEY = "segmented"
BINS_KEY = "radius_bins_um"
#: Keys of a segmentation's ``meta.json`` read here (stack depth = slices x voxel z).
SHAPE_KEY = "shape_zyx"
CONFIG_KEY = "config"
ACQUISITION_KEY = "acquisition"
VOXEL_KEY = "voxel_size_um"
#: Rows of the score table: key of ``network.json`` and its caption.
_SCORED = ((TRUE_MASK_KEY, "True mask"), (SEGMENTED_KEY, "Segmentation"))
_PHANTOM_DIR = "phantom"
_SEGMENTATION_DIR = "segmentation"
_PERCENT = 100.0


def _score(
    ctx: "Context", labels: Any, phantom: "Phantom", bins: tuple[float, ...]
) -> NetworkScore:
    """Measure the network of a label volume of the phantom and score it.

    Args:
        ctx: Figure context (configuration, network settings).
        labels: ``(z, y, x)`` label volume of the phantom (truth or segmentation).
        phantom: The phantom (its true network).
        bins: Edges of the true-radius bands.

    Returns:
        The score.
    """
    voxel = ctx.config.acquisition.voxel_size_um
    measured = measure_network(labels, voxel, ctx.config.quantification)
    shape = (int(labels.shape[0]), int(labels.shape[1]), int(labels.shape[2]))
    return score_network(
        measured.network,
        measured.distances_um,
        phantom.network,
        shape,
        voxel,
        ctx.settings.network.tolerance_um,
        ctx.settings.network.junction_tolerance_um,
        bins,
    )


def _signed(value: float) -> str:
    """Format a radius offset for a table (a dash where nothing was measured).

    Args:
        value: Offset in µm, or NaN.

    Returns:
        LaTeX text.
    """
    return "--" if math.isnan(value) else f"${value:+.2f}$"


def _tables(ctx: "Context", record: dict[str, Any]) -> None:
    """Write the score table and the radius-band table of the phantom.

    Args:
        ctx: Figure context.
        record: Content of ``network.json``.
    """
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"On & Recall & Precision & Forks & Not forks & Crossings & Length (mm) "
        + r"& Radius ($\mu$m) \\",
        r"\midrule",
    ]
    for key, caption in _SCORED:
        s = NetworkScore(**record[key])
        lines.append(
            f"{caption} & {s.centreline_recall:.2f} & {s.centreline_precision:.2f} & "
            + f"{s.matched_junctions} / {s.true_junctions} & {s.spurious_junctions} & "
            + f"{s.crossing_junctions} & "
            + f"{s.measured_length_mm:.2f} / {s.true_length_mm:.2f} & "
            + f"{_signed(s.radius_offset_um)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _ = (ctx.data / NETWORK_TABLE).write_text("\n".join(lines) + "\n")

    bins = record[BINS_KEY]
    bands = [f"{low:g}--{high:g}" for low, high in itertools.pairwise(bins)]
    lines = [
        r"\begin{tabular}{l" + "r" * len(bands) + "}",
        r"\toprule",
        r"True radius ($\mu$m) & " + " & ".join(bands) + r" \\",
        r"\midrule",
    ]
    for key, caption in _SCORED:
        offsets = NetworkScore(**record[key]).radius_offset_by_band_um
        lines.append(f"{caption} & " + " & ".join(_signed(v) for v in offsets) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _ = (ctx.data / NETWORK_BANDS_TABLE).write_text("\n".join(lines) + "\n")


def network_files(ctx: "Context") -> None:
    """Measure the real sample's network and score the branching phantom's (see the module).

    Args:
        ctx: Figure context.
    """
    real = run_dir(ctx.settings, ctx.repo, ctx.settings.figures.reference_key, REAL_RUN)
    t0 = time.perf_counter()
    summary = run_quantification(real, ctx.config)
    real_seconds = time.perf_counter() - t0

    settings = load_phantom_settings(ctx.repo / ctx.settings.network.phantom_settings)
    work = ctx.repo / ctx.settings.network.work_dir
    shutil.rmtree(work, ignore_errors=True)  # afresh: a cache would outlive code changes
    phantom = render_phantom(settings, ctx.config.acquisition.voxel_size_um)
    save_phantom(phantom, work / _PHANTOM_DIR, settings)
    segmentation = work / _SEGMENTATION_DIR
    _ = run_segmentation(work / _PHANTOM_DIR / settings.files.slices_dir, segmentation, ctx.config)
    bins = settings.score.radius_bins_um
    record = {
        REAL_KEY: summary.model_dump(mode="json"),
        REAL_SECONDS_KEY: real_seconds,
        BINS_KEY: list(bins),
        TRUE_MASK_KEY: asdict(_score(ctx, phantom.truth, phantom, bins)),
        SEGMENTED_KEY: asdict(_score(ctx, load_labels(segmentation, ctx.config.io), phantom, bins)),
    }
    _ = (ctx.data / NETWORK_JSON).write_text(json.dumps(record, indent=JSON_INDENT) + "\n")
    _tables(ctx, record)


def network_values(ctx: "Context") -> dict[str, str]:
    """Read ``network.json`` as the report's macros (``pbValueNet...``).

    Args:
        ctx: Figure context.

    Returns:
        Macro name (after ``pbValue``) to its text.
    """
    record = json.loads((ctx.data / NETWORK_JSON).read_text())
    real = NetworkSummary.model_validate(record[REAL_KEY])
    seg = NetworkScore(**record[SEGMENTED_KEY])
    true = NetworkScore(**record[TRUE_MASK_KEY])
    bands = seg.radius_offset_by_band_um
    length_error = seg.measured_length_mm / seg.true_length_mm - 1
    phantom = load_phantom_settings(ctx.repo / ctx.settings.network.phantom_settings)
    report = ctx.settings.network
    quantification = ctx.config.quantification
    meta = ctx.reference_json(META_FILE)
    depth_um = meta[SHAPE_KEY][0] * meta[CONFIG_KEY][ACQUISITION_KEY][VOXEL_KEY][0]
    return {
        "NetSeconds": f"{record[REAL_SECONDS_KEY]:.0f}",
        "NetStackDepthUm": f"{depth_um:.0f}",
        "NetBranches": f"{real.branches:,}",
        "NetTruncated": f"{real.truncated_branches:,}",
        "NetJunctions": f"{real.junctions:,}",
        "NetLoops": f"{real.loops:,}",
        "NetFreeEnds": f"{real.free_ends:,}",
        "NetFaceEnds": f"{real.face_ends:,}",
        "NetBurstEnds": f"{real.burst_ends:,}",
        "NetLengthMm": f"{real.total_length_mm:.0f}",
        "NetDensity": f"{real.length_density_mm_per_mm3:.1f}",
        "NetMedianBranchUm": f"{real.median_branch_length_um:.0f}",
        "NetMedianRadiusUm": f"{real.median_radius_um:.1f}",
        "NetMedianTortuosity": f"{real.median_tortuosity:.2f}",
        "NetPhantomVessels": f"{phantom.vessels.count}",
        "NetPhantomDepth": f"{phantom.vessels.branch_depth}",
        "NetTolerance": f"{report.tolerance_um:g}",
        "NetJunctionTolerance": f"{report.junction_tolerance_um:g}",
        "NetAnchorMargin": f"{quantification.anchor_margin_um:g}",
        "NetSpurFactor": f"{quantification.pruning.spur_radius_factor:g}",
        "NetTrueForks": f"{seg.true_junctions}",
        "NetTrueMaskMissed": f"{true.true_junctions - true.matched_junctions}",
        "NetTrueMaskForks": f"{true.matched_junctions}",
        "NetTrueMaskSpurious": f"{true.spurious_junctions}",
        "NetSegForks": f"{seg.matched_junctions}",
        "NetSegSpurious": f"{seg.spurious_junctions}",
        "NetSegRecall": f"{seg.centreline_recall:.2f}",
        "NetSegPrecision": f"{seg.centreline_precision:.2f}",
        "NetSegLengthErrorPct": f"{_PERCENT * abs(length_error):.1f}",
        # Signed values in math mode: a real minus sign, never split from its number.
        "NetSegRadiusOffset": f"${seg.radius_offset_um:+.2f}$",
        "NetSegRadiusThin": f"${bands[0]:+.1f}$",
        "NetSegRadiusThick": f"${bands[-1]:+.1f}$",
        "NetTrueRadiusThick": f"${true.radius_offset_by_band_um[-1]:+.1f}$",
    }
