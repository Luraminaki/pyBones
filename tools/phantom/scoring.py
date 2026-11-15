"""Comparison of a segmentation with the phantom's exact ground truth."""

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pybones.analysis.metrics import LabelScores
from pybones.segmentation.labels import Label

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from tools.phantom.generator import Phantom


@dataclass(frozen=True, slots=True)
class RadiusRecall:
    """Fraction of truth-vessel voxels labelled vessel, for vessels of radius in [low, high).

    Attributes:
        low_um: Lower radius of the band.
        high_um: Upper radius of the band.
        voxels: Truth-vessel voxels in the band.
        recall: Fraction of them labelled vessel (``nan`` if none).
    """

    low_um: float
    high_um: float
    voxels: int
    recall: float


@dataclass(frozen=True, slots=True)
class PhantomScore:
    """Segmentation quality against the phantom truth (excluded voxels ignored).

    Attributes:
        labels: Per-label scores and confusion (rows: truth).
        vessel_recall_by_radius: Vessel recall per radius bin.
        clotted_recall: Vessel recall on clotted (dim) segments only.
    """

    labels: LabelScores
    vessel_recall_by_radius: list[RadiusRecall]
    clotted_recall: float


def _recall(hits: "NDArray[np.bool_]", mask: "NDArray[np.bool_]") -> float:
    """Measure the fraction of ``mask`` covered by ``hits``.

    Args:
        hits: Predicted voxels.
        mask: Reference voxels.

    Returns:
        The recall (``nan`` for an empty mask).
    """
    total = int(mask.sum())
    return float(hits[mask].sum()) / total if total else float("nan")


def score_phantom(
    prediction: "NDArray[np.uint8]",
    phantom: "Phantom",
    excluded_value: int,
    radius_bins_um: tuple[float, ...],
) -> PhantomScore:
    """Score a label volume against the phantom truth.

    Args:
        prediction: ``(Z, Y, X)`` segmentation of the phantom image.
        phantom: The phantom (truth, vessel radii, clot map).
        excluded_value: Truth value ignored by every metric.
        radius_bins_um: Bin edges for the vessel recall by radius.

    Returns:
        The score.
    """
    labels = LabelScores()
    labels.update(prediction, phantom.truth, phantom.truth != excluded_value)

    found = prediction == Label.VESSEL
    radius = phantom.vessel_radius_um

    def band(low: float, high: float) -> RadiusRecall:
        """Measure the vessel recall for vessels of radius in ``[low, high)``.

        Args:
            low: Lower radius of the band.
            high: Upper radius of the band.

        Returns:
            The recall of the band.
        """
        in_band = (radius > 0) & (radius >= low) & (radius < high)
        return RadiusRecall(low, high, int(in_band.sum()), _recall(found, in_band))

    by_radius = [band(low, high) for low, high in itertools.pairwise(radius_bins_um)]
    return PhantomScore(labels, by_radius, _recall(found, phantom.clotted))
