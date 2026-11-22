"""The experimental switches measured: bone-side vessel edges and thin vessels, on and off.

Both are off by default: they improve the phantoms, but what they change on a real scan no
annotation can confirm yet. :func:`experimental_files` segments the standard phantom and the
branching phantom (the one :func:`~tools.report.network.network_files` generated) with each
combination of the two switches, scores them, and segments the real sample with each switch
alone to measure how much vessel it moves. Everything lands in ``experimental.json`` and a
LaTeX table; :func:`experimental_values` turns the file into the report's macros
(``pbValueExp...``).
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pybones.config.loader import apply_overrides
from pybones.segmentation.labels import Label
from pybones.segmentation.pipeline import THIN_TIMING, run_segmentation
from pybones.segmentation.results import load_labels
from pybones.utils.constants import JSON_INDENT
from tools.phantom.network_scoring import NetworkScore
from tools.phantom.scoring import score_phantom
from tools.phantom.settings import load_settings as load_phantom_settings
from tools.phantom.storage import load_phantom
from tools.report.network import PHANTOM_DIR, length_error_pct, score_phantom_network
from tools.report.runs import REAL_RUN, run_dir

if TYPE_CHECKING:
    from tools.report.context import Context

EXPERIMENTAL_JSON = "experimental.json"
EXPERIMENTAL_TABLE = "experimental.tex"
#: The two switches, as overrides.
BONE_SIDE_ON = "segmentation.vessel.bone_side.enabled=true"
THIN_VESSELS_ON = "thin_vessels.enabled=true"
#: Setups measured: macro name, table caption, overrides.
SETUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Default", "Both off (default)", ()),
    ("BoneSide", "Bone-side edges", (BONE_SIDE_ON,)),
    ("Thin", "Thin vessels", (THIN_VESSELS_ON,)),
    ("Both", "Both on", (BONE_SIDE_ON, THIN_VESSELS_ON)),
)
#: Keys of ``experimental.json``.
STANDARD_KEY, NETWORK_KEY, REAL_KEY = "standard", "network", "real"
DICE_KEY, PRECISION_KEY, THIN_RECALL_KEY = "vessel_dice", "vessel_precision", "thin_recall"
VESSEL_KEY, THIN_SECONDS_KEY = "vessel_voxels", "thin_seconds"
_PERCENT = 100.0


def _vessel_voxels(labels: Any) -> int:
    """Count the vessel voxels of a label volume.

    Args:
        labels: ``(z, y, x)`` labels (memory-mapped: read slice by slice).

    Returns:
        Their number.
    """
    return sum(int(np.count_nonzero(np.asarray(sl) == Label.VESSEL)) for sl in labels)


def experimental_files(ctx: "Context") -> None:
    """Segment and score the phantoms and the real sample with each switch (see the module).

    Args:
        ctx: Figure context.
    """
    standard_dir = ctx.repo / ctx.settings.phantom_dir
    standard = load_phantom(standard_dir, ctx.phantom_settings)
    network_settings = load_phantom_settings(ctx.repo / ctx.settings.network.phantom_settings)
    network_dir = ctx.repo / ctx.settings.network.work_dir / PHANTOM_DIR
    network = load_phantom(network_dir, network_settings)
    real = run_dir(ctx.settings, ctx.repo, ctx.settings.figures.reference_key, REAL_RUN)
    record: dict[str, Any] = {
        REAL_KEY: {"Default": {VESSEL_KEY: _vessel_voxels(load_labels(real, ctx.config.io))}}
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, _, overrides in SETUPS:
            config = apply_overrides(ctx.config, overrides)
            out = Path(tmp) / name
            _ = run_segmentation(
                standard_dir / ctx.phantom_settings.files.slices_dir, out / "std", config
            )
            score = score_phantom(
                load_labels(out / "std", config.io),
                standard,
                ctx.phantom_settings.excluded_value,
                ctx.phantom_settings.score.radius_bins_um,
            )
            vessel = score.labels.per_label[Label.VESSEL]
            record.setdefault(STANDARD_KEY, {})[name] = {
                DICE_KEY: vessel.dice,
                PRECISION_KEY: vessel.precision,
                THIN_RECALL_KEY: score.vessel_recall_by_radius[0].recall,
            }
            _ = run_segmentation(
                network_dir / network_settings.files.slices_dir, out / "net", config
            )
            labels = load_labels(out / "net", config.io)
            bins = network_settings.score.radius_bins_um
            record.setdefault(NETWORK_KEY, {})[name] = asdict(
                score_phantom_network(ctx, labels, network, bins)
            )
            if len(overrides) == 1:  # each switch alone on the real sample
                meta = run_segmentation(ctx.repo / ctx.settings.dataset, out / "real", config)
                record[REAL_KEY][name] = {
                    VESSEL_KEY: _vessel_voxels(load_labels(out / "real", config.io)),
                    THIN_SECONDS_KEY: meta.timing_s.get(THIN_TIMING, 0.0),
                }
    _ = (ctx.data / EXPERIMENTAL_JSON).write_text(json.dumps(record, indent=JSON_INDENT) + "\n")
    _table(ctx, record)


def _table(ctx: "Context", record: dict[str, Any]) -> None:
    """Write the comparison of the setups as a LaTeX table.

    Args:
        ctx: Figure context.
        record: Content of ``experimental.json``.
    """
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Standard phantom (vessels)}"
        + r" & \multicolumn{4}{c}{Branching phantom} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r"Setup & Dice & Precision & Recall $<6$\um & Forks & Other & Loops & Length \\",
        r"\midrule",
    ]
    for name, caption, _ in SETUPS:
        std = record[STANDARD_KEY][name]
        net = NetworkScore(**record[NETWORK_KEY][name])
        lines.append(
            f"{caption} & {std[DICE_KEY]:.3f} & {std[PRECISION_KEY]:.3f} & "
            + f"{std[THIN_RECALL_KEY]:.2f} & {net.matched_junctions}/{net.true_junctions} & "
            + f"{net.spurious_junctions} & {net.independent_cycles} & "
            + f"{length_error_pct(net)}\\,\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _ = (ctx.data / EXPERIMENTAL_TABLE).write_text("\n".join(lines) + "\n")


def experimental_values(ctx: "Context") -> dict[str, str]:
    """Read ``experimental.json`` as the report's macros (``pbValueExp...``).

    Args:
        ctx: Figure context.

    Returns:
        Macro name (after ``pbValue``) to its text.
    """
    record = json.loads((ctx.data / EXPERIMENTAL_JSON).read_text())
    values: dict[str, str] = {}
    for name, _, _ in SETUPS:
        std = record[STANDARD_KEY][name]
        net = NetworkScore(**record[NETWORK_KEY][name])
        values |= {
            f"Exp{name}Dice": f"{std[DICE_KEY]:.3f}",
            f"Exp{name}Precision": f"{std[PRECISION_KEY]:.3f}",
            f"Exp{name}ThinRecall": f"{std[THIN_RECALL_KEY]:.2f}",
            f"Exp{name}Forks": f"{net.matched_junctions}",
            f"Exp{name}Spurious": f"{net.spurious_junctions}",
            f"Exp{name}Loops": f"{net.independent_cycles}",
            f"Exp{name}LengthErrorPct": length_error_pct(net),
        }
    real = record[REAL_KEY]
    before = real["Default"][VESSEL_KEY]
    for name in ("BoneSide", "Thin"):
        change = _PERCENT * (real[name][VESSEL_KEY] / before - 1)
        values[f"ExpReal{name}VesselPct"] = f"${change:+.1f}$"
    values["ExpRealThinSeconds"] = f"{real['Thin'][THIN_SECONDS_KEY]:.0f}"
    thin = ctx.config.thin_vessels
    values |= {
        "ExpThinSigma": f"{thin.psf_sigma_um:g}",
        "ExpThinIterations": f"{thin.iterations}",
        "ExpThinMinVolume": f"{thin.min_volume_um3:g}",
        "ExpThinMaxHalf": f"{thin.max_half_thickness_um:g}",
        "ExpThinBlockSlices": f"{thin.block_slices}",
        "ExpThinBlockPx": f"{thin.block_px}",
        "ExpBoneSideBarrier": f"{ctx.config.segmentation.vessel.bone_side.barrier_um:g}",
    }
    return values
