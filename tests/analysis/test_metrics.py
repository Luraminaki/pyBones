"""Tests of the agreement metrics (:mod:`pybones.analysis.metrics`)."""

import numpy as np
import pytest

from pybones.analysis.metrics import compare_stacks


def test_dice_precision_recall() -> None:
    """Counts accumulate over slices and give the expected scores."""
    pred = np.array([[1, 1, 0, 0]], bool)
    ref = np.array([[1, 0, 1, 0]], bool)
    agreement = compare_stacks([pred, pred], [ref, ref])
    assert (agreement.tp, agreement.fp, agreement.fn) == (2, 2, 2)
    assert agreement.dice == pytest.approx(0.5)
    assert agreement.precision == pytest.approx(0.5)
    assert agreement.recall == pytest.approx(0.5)


def test_compare_rejects_mismatches() -> None:
    """Streams of different lengths or slice shapes are rejected."""
    a = np.zeros((2, 2), bool)
    with pytest.raises(ValueError):
        _ = compare_stacks([a], [a, a])
    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = compare_stacks([a], [np.zeros((3, 2), bool)])
