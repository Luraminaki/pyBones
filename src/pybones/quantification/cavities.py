"""Filling the cavities of a 3D mask (implementations of :class:`~.protocols.CavityFiller`).

A cavity is a 6-connected background region that never reaches a face of the volume, the
complement convention of a 26-connected foreground. Both implementations agree on it.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from pybones.segmentation.connectivity import StreamingComponents

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: In-plane 4-connectivity: background components 6-connected in 3D.
_BACKGROUND_CONNECTIVITY = 4
#: The 6-neighbourhood as a SciPy structuring element.
_SIX_CONNECTED = ndimage.generate_binary_structure(3, 1)


def _border_ids(local: "NDArray[np.int32]") -> "NDArray[np.int32]":
    """List the local labels touching the edge of a slice.

    Args:
        local: Local labels of a slice.

    Returns:
        Distinct non-zero labels on its first and last rows and columns.
    """
    edge = np.concatenate([local[0], local[-1], local[:, 0], local[:, -1]])
    return np.unique(edge[edge > 0])


class StreamingCavityFiller:
    """Fills cavities slice by slice with pyBones' streaming 3D connectivity.

    The background is labelled in 2D, linked across slices and resolved once; a second
    pass relabels only the slices that hold a cavity and fills it.

    Pros: no extra dependency; memory is the mask, its filled copy and one label slice.
    Cons: two passes over the slices; the first one (labelling the whole background of
    every slice and linking it across slices) is most of the time, ~6 s on the sample.
    """

    def fill(self, mask: "NDArray[np.bool_]") -> "NDArray[np.bool_]":
        """Fill the cavities of a mask.

        Args:
            mask: ``(Z, Y, X)`` mask (not modified).

        Returns:
            The mask with its cavities set.
        """
        depth = mask.shape[0]
        tracker = StreamingComponents(connectivity=_BACKGROUND_CONNECTIVITY)
        open_ids: list[NDArray[np.int64]] = []
        for z in range(depth):
            local = tracker.add_slice(~mask[z])
            offset = tracker.offsets[-1]
            # Whole first and last slices are on a face; elsewhere, only the slice edges.
            touching = np.unique(local[local > 0]) if z in (0, depth - 1) else _border_ids(local)
            open_ids.append(touching.astype(np.int64) + offset)
        component_of = tracker.resolve()
        open_components = np.zeros(int(component_of.max()) + 1, dtype=np.bool_)
        open_components[component_of[np.concatenate(open_ids)]] = True
        open_components[0] = True  # the foreground itself
        is_cavity = ~open_components[component_of]  # per global id
        # Slice z owns the global ids (offsets[z], offsets[z + 1]]: relabel only the slices
        # holding a cavity (few: most of the background reaches a face).
        bounds = [*tracker.offsets, tracker.n_ids - 1]
        filled = mask.copy()
        for z in range(depth):
            if not is_cavity[bounds[z] + 1 : bounds[z + 1] + 1].any():
                continue
            local = tracker.local_labels(~mask[z])
            ids = np.where(local > 0, local.astype(np.int64) + tracker.offsets[z], 0)
            filled[z] |= is_cavity[ids]
        return filled


class ScipyCavityFiller:
    """Fills cavities with :func:`scipy.ndimage.binary_fill_holes` (6-connected background).

    Pros: one call to a mature library; ~6 s on the sample.
    Cons: it propagates the outside inwards by repeated dilation until nothing changes, so
    its time grows with the length of the background's paths, not just its size; it holds
    several full-size temporary masks.
    """

    def fill(self, mask: "NDArray[np.bool_]") -> "NDArray[np.bool_]":
        """Fill the cavities of a mask.

        Args:
            mask: ``(Z, Y, X)`` mask (not modified).

        Returns:
            The mask with its cavities set.
        """
        return np.asarray(ndimage.binary_fill_holes(mask, structure=_SIX_CONNECTED), dtype=np.bool_)
