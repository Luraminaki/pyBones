"""Tests of the whole-stack segmentation (:mod:`pybones.segmentation.pipeline`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.io.slices import SliceStack
from pybones.segmentation.classifier import estimate_thresholds
from pybones.segmentation.labels import FINAL_LABELS, Label
from pybones.segmentation.pipeline import iter_classified, planned_stages, run_segmentation
from pybones.segmentation.results import load_labels, load_meta
from pybones.segmentation.scales import resolve_pixel_params, resolve_tubularity
from tests.helpers import N_SYNTHETIC_SLICES, SLICE_SHAPE, classify_slice, small_config

if TYPE_CHECKING:
    from pathlib import Path

    from pybones.segmentation.stages import Stage


@pytest.mark.parametrize("refine", [True, False])
def test_run_segmentation_end_to_end(slice_dir: "Path", tmp_path: "Path", refine: bool) -> None:
    """A run writes the labels and metadata only, reports its stages and finds the burst."""
    config = small_config(f"refinement.enabled={str(refine).lower()}")
    out_dir = tmp_path / "out"
    seen: list[Stage] = []

    def record(stage: "Stage", _done: int, _total: int) -> None:
        """Record the stage of a progress update.

        Args:
            stage: Reported stage.
            _done: Items processed (unused).
            _total: Items in the stage (unused).
        """
        seen.append(stage)

    meta = run_segmentation(slice_dir, out_dir, config, record)
    labels = load_labels(out_dir, config.io)
    assert labels.shape == (N_SYNTHETIC_SLICES, *SLICE_SHAPE)
    assert set(np.unique(labels)) <= {int(lab) for lab in FINAL_LABELS}
    assert load_meta(out_dir, config.io) == meta
    assert meta.config == config
    assert sum(meta.label_fractions.values()) == pytest.approx(1.0)
    assert (meta.refinement is not None) == refine
    # Every planned stage is reported, in the planned order.
    assert list(dict.fromkeys(seen)) == list(planned_stages(config))
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        [config.io.labels_file, config.io.meta_file]
    )
    assert (labels == Label.BURST).any()


def test_threaded_classification_matches_sequential(slice_dir: "Path") -> None:
    """Threads change nothing in the labels or the tubularity maps."""
    threaded_cfg = small_config("pipeline.workers=3")
    sequential_cfg = small_config("pipeline.workers=1")
    with SliceStack(slice_dir, threaded_cfg.io) as stack:
        params = resolve_pixel_params(threaded_cfg)
        thresholds = estimate_thresholds(
            iter(stack), params, threaded_cfg.thresholds.histogram_bins
        )
        tubes = resolve_tubularity(threaded_cfg)
        threaded = list(iter_classified(stack, thresholds, params, threaded_cfg.pipeline, tubes))
        sequential = list(
            iter_classified(stack, thresholds, params, sequential_cfg.pipeline, tubes)
        )
    for a, b in zip(threaded, sequential, strict=True):
        np.testing.assert_array_equal(a.labels, b.labels)
        np.testing.assert_array_equal(a.tubularity, b.tubularity)


def test_window_z_1_reproduces_the_2d_classifier(slice_dir: "Path") -> None:
    """With a one-slice burst window, the streamed pipeline equals the 2D classifier."""
    config = small_config("segmentation.burst.window_z_um=3")  # one slice
    params = resolve_pixel_params(config)
    assert params.burst_window_z == 1
    with SliceStack(slice_dir, config.io) as stack:
        thresholds = estimate_thresholds(iter(stack), params, config.thresholds.histogram_bins)
        streamed = [c.labels for c in iter_classified(stack, thresholds, params, config.pipeline)]
        per_slice = [classify_slice(img, thresholds, params) for img in stack]
    for a, b in zip(streamed, per_slice, strict=True):
        np.testing.assert_array_equal(a, b)
