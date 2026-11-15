"""Score a segmentation against the painted crops ("unsure" pixels are ignored)."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pybones.analysis.metrics import LabelScores

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tools.annotation.workspace import Manifest, Workspace


@dataclass(slots=True)
class Score:
    """Agreement between a segmentation and the annotations.

    Attributes:
        labels: Per-label scores and confusion over the annotated pixels.
        crops_scored: Number of crops that had saved labels.
    """

    labels: LabelScores = field(default_factory=LabelScores)
    crops_scored: int = 0


def score_segmentation(
    volume: "NDArray[np.uint8]", manifest: "Manifest", workspace: "Workspace"
) -> Score:
    """Compare a label volume with every saved crop annotation.

    Args:
        volume: ``(Z, Y, X)`` segmentation to evaluate (same stack as the manifest).
        manifest: Workspace manifest (crop positions).
        workspace: Workspace holding the painted labels.

    Returns:
        The accumulated score.
    """
    score = Score()
    size = manifest.crop_size
    for crop in manifest.crops:
        if not workspace.is_annotated(crop.crop_id):
            continue
        ref = workspace.read_labels(crop.crop_id, size)
        pred = np.asarray(volume[crop.z, crop.y : crop.y + size, crop.x : crop.x + size])
        score.labels.update(pred, ref, ref != manifest.unsure_value)
        score.crops_scored += 1
    return score
