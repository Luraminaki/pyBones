"""Golden regression test: today's scores on a fixed phantom may only change on purpose.

The whole pipeline (default configuration) segments a reduced phantom rendered from
a fixed seed, and its scores are compared with ``golden_phantom.json``:

* a score lower than the golden one by more than :data:`TOLERANCE` fails (regression);
* a score higher by more than :data:`TOLERANCE` fails too, so that improvements are
  recorded on purpose and the bar only moves forward;
* a change of the phantom or of the segmentation settings fails with a request to
  refresh the golden file, rather than comparing unlike things.

To record a new baseline after a deliberate change::

    PYBONES_UPDATE_GOLDEN=1 uv run pytest tests/regression
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from pybones.config.fields import EFFECT_KEY, Effect
from pybones.config.loader import load_config
from pybones.config.schema import PyBonesConfig
from pybones.segmentation.pipeline import run_segmentation
from pybones.segmentation.results import load_labels
from pybones.utils.constants import JSON_INDENT
from tools.phantom.generator import render_phantom
from tools.phantom.scoring import score_phantom
from tools.phantom.settings import PhantomSettings, load_settings
from tools.phantom.storage import save_phantom
from tools.report.history import score_columns

GOLDEN_FILE = Path(__file__).with_name("golden_phantom.json")
#: Set to 1 to rewrite the golden file from the current scores.
UPDATE_ENV = "PYBONES_UPDATE_GOLDEN"
#: Largest accepted change of a score (the pipeline is deterministic: this only absorbs
#: floating-point differences between platforms and library versions).
TOLERANCE = 0.002
#: Reduced phantom: a quarter of the default volume, same calibration, scaled counts.
SHAPE = (64, 320, 320)
COUNTS = {
    "trabeculae": {"plates": 5, "rods": 12},
    "vessels": {"count": 25},
    "bursts": {"count": 1, "radius_um": [110, 150]},
}
#: Configuration sections that change the segmentation: those whose declared effect is the
#: labels (logging, meshing, rendering... are not), so a new one cannot be forgotten.


def changes_labels(section: object) -> bool:
    """Tell whether a root section declares that a change of it changes the labels.

    Args:
        section: A field's annotation.

    Returns:
        Whether it is a section whose declared effect is :attr:`Effect.LABELS`.
    """
    if not (isinstance(section, type) and issubclass(section, BaseModel)):
        return False
    extra = section.model_config.get("json_schema_extra")
    return isinstance(extra, dict) and extra.get(EFFECT_KEY) == Effect.LABELS


SEGMENTATION_SECTIONS = tuple(
    name for name, field in PyBonesConfig.model_fields.items() if changes_labels(field.annotation)
)
_SCORES_KEY = "scores"
_PHANTOM_KEY = "phantom_digest"
_CONFIG_KEY = "config_digest"


def golden_phantom_settings() -> PhantomSettings:
    """Reduce the default phantom settings to the golden phantom.

    Returns:
        The validated settings.
    """
    data: dict[str, Any] = json.loads(load_settings().model_dump_json())
    data["shape_zyx"] = list(SHAPE)
    for section, values in COUNTS.items():
        data[section].update(values)
    return PhantomSettings.model_validate(data)


def digest(data: object) -> str:
    """Fingerprint JSON-serialisable settings.

    Args:
        data: Settings as plain data.

    Returns:
        SHA-256 of their canonical JSON form.
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def config_digest(config: "PyBonesConfig") -> str:
    """Fingerprint the configuration sections that change the segmentation.

    Args:
        config: Configuration.

    Returns:
        SHA-256 of those sections.
    """
    data = json.loads(config.model_dump_json())
    return digest({section: data[section] for section in SEGMENTATION_SECTIONS})


def current_scores(tmp_path: Path) -> dict[str, float]:
    """Render the golden phantom, segment it with the default configuration, score it.

    Args:
        tmp_path: Scratch directory.

    Returns:
        History columns (accuracy, per-label metrics, recall per band and on clots).
    """
    settings = golden_phantom_settings()
    config = load_config()
    phantom = render_phantom(settings, config.acquisition.voxel_size_um)
    save_phantom(phantom, tmp_path, settings)
    out_dir = tmp_path / "segmentation"
    _ = run_segmentation(tmp_path / settings.files.slices_dir, out_dir, config)
    prediction = np.asarray(load_labels(out_dir, config.io))
    score = score_phantom(
        prediction, phantom, settings.excluded_value, settings.score.radius_bins_um
    )
    return score_columns(score)


def differences(current: dict[str, float], golden: dict[str, float]) -> list[str]:
    """List the scores that moved by more than the tolerance (``nan`` equals ``nan``).

    Args:
        current: Today's scores.
        golden: Recorded scores.

    Returns:
        One message per changed, missing or new score.
    """
    messages = [f"{name}: new score" for name in current.keys() - golden.keys()]
    messages += [f"{name}: no longer measured" for name in golden.keys() - current.keys()]
    for name in sorted(current.keys() & golden.keys()):
        now, then = current[name], golden[name]
        if math.isnan(now) and math.isnan(then):
            continue
        if math.isnan(now) or math.isnan(then) or abs(now - then) > TOLERANCE:
            direction = "improved" if now > then else "REGRESSED"
            messages.append(f"{name}: {direction} {then:.4f} -> {now:.4f}")
    return messages


def test_scores_match_the_golden_baseline(tmp_path: Path) -> None:
    """Today's phantom scores equal the recorded baseline, within the tolerance."""
    scores = current_scores(tmp_path)
    fingerprints = {
        _PHANTOM_KEY: digest(json.loads(golden_phantom_settings().model_dump_json())),
        _CONFIG_KEY: config_digest(load_config()),
    }
    if os.environ.get(UPDATE_ENV) == "1":
        record = {**fingerprints, _SCORES_KEY: {k: round(v, 6) for k, v in scores.items()}}
        _ = GOLDEN_FILE.write_text(json.dumps(record, indent=JSON_INDENT, sort_keys=True) + "\n")
        pytest.skip(f"golden baseline rewritten: {GOLDEN_FILE}")
    golden = json.loads(GOLDEN_FILE.read_text())
    refresh = f"if intended, refresh it with {UPDATE_ENV}=1 uv run pytest tests/regression"
    for key, value in fingerprints.items():
        assert golden[key] == value, f"{key} changed since the baseline was recorded; {refresh}"
    changes = differences(scores, golden[_SCORES_KEY])
    assert not changes, "phantom scores changed:\n  " + "\n  ".join(changes) + f"\n{refresh}"


def test_differences_flags_moves_both_ways() -> None:
    """Changes beyond the tolerance are reported in both directions; ``nan`` matches ``nan``."""
    golden = {"a": 0.5, "b": 0.5, "c": float("nan"), "d": 0.5}
    current = {"a": 0.5 + TOLERANCE / 2, "b": 0.49, "c": float("nan"), "e": 0.1}
    messages = differences(current, golden)
    assert messages == [
        "e: new score",
        "d: no longer measured",
        "b: REGRESSED 0.5000 -> 0.4900",
    ]
