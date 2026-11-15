"""Agreement metrics between segmentations."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pybones.segmentation.labels import FINAL_LABELS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


@dataclass(slots=True)
class Agreement:
    """Running confusion counts between a prediction and a reference mask.

    Attributes:
        tp: Voxels set in both.
        fp: Voxels set in the prediction only.
        fn: Voxels set in the reference only.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, pred: "NDArray[np.bool_]", ref: "NDArray[np.bool_]") -> None:
        """Accumulate counts for one pair of same-shaped masks.

        Args:
            pred: Predicted mask.
            ref: Reference mask.
        """
        inter = int(np.count_nonzero(pred & ref))
        self.tp += inter
        self.fp += int(np.count_nonzero(pred)) - inter
        self.fn += int(np.count_nonzero(ref)) - inter

    @property
    def dice(self) -> float:
        """Sørensen-Dice coefficient (1 = identical, ``nan`` if both empty)."""
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else float("nan")

    @property
    def precision(self) -> float:
        """Fraction of predicted voxels present in the reference."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        """Fraction of reference voxels found by the prediction."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")


def compare_stacks(
    pred: "Iterable[NDArray[np.bool_]]", ref: "Iterable[NDArray[np.bool_]]"
) -> Agreement:
    """Compare two streams of 2D masks slice by slice.

    Args:
        pred: Predicted masks, in z order.
        ref: Reference masks, in the same order and shapes.

    Returns:
        Accumulated agreement over all slices.

    Raises:
        ValueError: If the streams differ in length or slice shape.
    """
    agreement = Agreement()
    for p, r in zip(pred, ref, strict=True):
        if p.shape != r.shape:
            raise ValueError(f"Shape mismatch: {p.shape} vs {r.shape}")
        agreement.update(p, r)
    return agreement


_N_LABELS = len(FINAL_LABELS)
_KEY_WIDTH = 11
_COUNT_WIDTH = 12


def _empty_agreements() -> dict[int, Agreement]:
    """Create one empty agreement per final label.

    Returns:
        Label value -> agreement.
    """
    return {int(lab): Agreement() for lab in FINAL_LABELS}


def _empty_confusion() -> "NDArray[np.int64]":
    """Create an empty confusion matrix over the final labels.

    Returns:
        Zero ``(labels, labels)`` matrix.
    """
    return np.zeros((_N_LABELS, _N_LABELS), dtype=np.int64)


@dataclass(slots=True)
class LabelScores:
    """Agreement of a label image with a reference, for every final label, accumulated over updates.

    Attributes:
        per_label: Label value -> Dice / precision / recall counts.
        confusion: ``confusion[reference, prediction]`` voxel counts.
    """

    per_label: dict[int, Agreement] = field(default_factory=_empty_agreements)
    confusion: "NDArray[np.int64]" = field(default_factory=_empty_confusion)

    def update(
        self,
        prediction: "NDArray[np.uint8]",
        reference: "NDArray[np.uint8]",
        valid: "NDArray[np.bool_]",
    ) -> None:
        """Accumulate one pair of same-shaped label images.

        Args:
            prediction: Labels to evaluate.
            reference: Reference labels (final labels wherever ``valid``).
            valid: Voxels to compare (e.g. not "unsure" or excluded).
        """
        for lab, agreement in self.per_label.items():
            agreement.update((prediction == lab) & valid, (reference == lab) & valid)
        pairs = reference[valid].astype(np.int64) * _N_LABELS + prediction[valid]
        self.confusion += np.bincount(pairs, minlength=_N_LABELS**2).reshape(_N_LABELS, -1)

    @property
    def voxels(self) -> int:
        """Number of voxels compared."""
        return int(self.confusion.sum())

    @property
    def accuracy(self) -> float:
        """Fraction of compared voxels with the right label (``nan`` if none)."""
        total = self.voxels
        return float(np.trace(self.confusion)) / total if total else float("nan")

    def report_lines(self, reference_name: str) -> list[str]:
        """Format the per-label scores and the confusion matrix for a terminal.

        Args:
            reference_name: What the rows of the confusion matrix are (``truth``,
                ``annotation``...).

        Returns:
            One line per label, then the confusion matrix.
        """
        keys = [lab.key for lab in FINAL_LABELS]
        lines = [
            f"  {lab.key:<10} dice {a.dice:.3f}  precision {a.precision:.3f}  recall {a.recall:.3f}"
            for lab, a in ((lab, self.per_label[int(lab)]) for lab in FINAL_LABELS)
        ]
        lines.append(f"  confusion (rows: {reference_name}, columns: segmentation)")
        lines.append("  " + " " * _KEY_WIDTH + "".join(f"{k:>{_COUNT_WIDTH}}" for k in keys))
        lines += [
            f"  {key:<{_KEY_WIDTH}}" + "".join(f"{v:>{_COUNT_WIDTH},}" for v in row)
            for key, row in zip(keys, self.confusion, strict=True)
        ]
        return lines
