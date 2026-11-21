"""Thinning a 3D mask to centrelines (implementations of :class:`~.protocols.Skeletonizer`)."""

from typing import TYPE_CHECKING

import numpy as np
from skimage.morphology import skeletonize

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ScikitImageSkeletonizer:
    """Lee's 3D thinning (1994), as implemented by :func:`skimage.morphology.skeletonize`.

    It keeps the topology of the mask: a solid tube becomes a curve, a cavity a closed
    surface (hence the cavity filling first), a loop stays a loop.

    Pros: tested library implementation; ~9 s on the sample.
    Cons: the whole volume in memory (scikit-image copies and pads it: ~1.1 GB peak on the
    sample); the thinning removes voxels in a fixed order, so a centreline can sit up to
    one voxel off the true axis, and diagonal steps leave staircase corners and short spurs
    that the network step has to merge and prune.
    """

    def skeletonize(self, mask: "NDArray[np.bool_]") -> "NDArray[np.bool_]":
        """Thin a mask.

        Args:
            mask: ``(Z, Y, X)`` mask (not modified).

        Returns:
            The centreline voxels.
        """
        return np.asarray(skeletonize(mask), dtype=np.bool_)
