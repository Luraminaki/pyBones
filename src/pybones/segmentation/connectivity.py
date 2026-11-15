"""Streaming 3D connected components of a mask volume, slice by slice.

A full ``int32`` label volume of the dataset would weigh 1.4 GB. Instead:

1. each slice is labelled in 2D (OpenCV); slice ``z``'s local label ``k``
   gets the global id ``k + offsets[z]``;
2. ids of consecutive slices that touch (6- or 26-neighbourhood) are recorded
   as graph edges, using only the previous slice;
3. the graph is solved once with a vectorized union-find (hooking + pointer
   jumping), giving a ``global id -> 3D component`` lookup table.

Callers re-label any slice later (2D labelling is deterministic) and map the
selected voxels through the table, so only one or two label slices live in
memory and offsets are only added where needed.
"""

from typing import TYPE_CHECKING

import numpy as np

from pybones.imaging import morphology as morph

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EIGHT_CONNECTED = 8
_ID_BITS = 32
_ID_MASK = (1 << _ID_BITS) - 1
#: Diagonal / side offsets towards the previous slice, used for 26-connectivity on top of
#: the straight (0, 0) link. 4-connected slices use the straight link only (6-connectivity).
_SIDE_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))


def _pack(cur_ids: "NDArray[np.int64]", prev_ids: "NDArray[np.int64]") -> "NDArray[np.int64]":
    """Pack id pairs into single int64 keys (so that :func:`numpy.unique` deduplicates them).

    Args:
        cur_ids: Global ids in slice ``z`` (high 32 bits).
        prev_ids: Global ids in slice ``z - 1`` (low 32 bits).

    Returns:
        One key per pair.
    """
    return (cur_ids << _ID_BITS) | prev_ids


def cross_slice_pairs(
    prev: "NDArray[np.int32]",
    prev_offset: int,
    cur: "NDArray[np.int32]",
    cur_offset: int,
    connectivity: int,
) -> "NDArray[np.int64]":
    """List the unique ``(cur_id, prev_id)`` global id pairs of touching voxels in adjacent slices.

    Distinct 8-connected components of one slice are never adjacent, so a side
    (diagonal) link can only add information where the voxel straight below is
    background: only that thin band next to ``prev``'s components is scanned
    for the 8 side offsets.

    Args:
        prev: Local labels of slice ``z - 1`` (0 = background).
        prev_offset: Global id offset of slice ``z - 1``.
        cur: Local labels of slice ``z``.
        cur_offset: Global id offset of slice ``z``.
        connectivity: In-plane connectivity (8 -> 26-connected in 3D, 4 -> 6-connected).

    Returns:
        ``(k, 2)`` int64 array of global id pairs.
    """
    cur_fg, prev_fg = cur > 0, prev > 0
    straight = cur_fg & prev_fg
    keys = [
        _pack(
            cur[straight].astype(np.int64) + cur_offset,
            prev[straight].astype(np.int64) + prev_offset,
        )
    ]
    if connectivity == _EIGHT_CONNECTED:
        band = cur_fg & ~prev_fg & morph.dilate_neighbours(prev_fg, connectivity)
        ys, xs = np.nonzero(band)
        if ys.size:
            height, width = cur.shape
            cur_ids = cur[ys, xs].astype(np.int64) + cur_offset
            for dy, dx in _SIDE_OFFSETS:
                yy, xx = ys + dy, xs + dx
                inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
                other = np.zeros(ys.size, dtype=np.int64)
                other[inside] = prev[yy[inside], xx[inside]]
                touching = other > 0
                keys.append(_pack(cur_ids[touching], other[touching] + prev_offset))
    unique = np.unique(np.concatenate(keys))
    return np.stack([unique >> _ID_BITS, unique & _ID_MASK], axis=1)


def connected_roots(n_nodes: int, edges: "NDArray[np.int64]") -> "NDArray[np.int64]":
    """Find the connected components of an undirected graph, fully vectorized.

    Union-find by hooking every root onto the smallest neighbouring root, then
    pointer jumping to flatten the forest; repeated until no edge spans two trees.

    Args:
        n_nodes: Number of nodes (ids ``0 .. n_nodes - 1``).
        edges: ``(k, 2)`` node id pairs.

    Returns:
        Component index per node, compacted to ``0 .. K-1`` in order of each
        component's smallest node id (so node 0 always maps to 0).
    """
    parent = np.arange(n_nodes, dtype=np.int64)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        ra, rb = parent[a], parent[b]
        spanning: NDArray[np.bool] = ra != rb
        if not spanning.any():
            break
        ra, rb = ra[spanning], rb[spanning]
        low = np.minimum(ra, rb)
        np.minimum.at(parent, ra, low)
        np.minimum.at(parent, rb, low)
        while True:  # pointer jumping until every node points at its root
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    _, compact = np.unique(parent, return_inverse=True)
    return compact


class StreamingComponents:
    """Accumulate the 3D connectivity of a mask streamed slice by slice.

    Example:
        >>> tracker = StreamingComponents(connectivity=8)
        >>> for mask in masks:
        ...     local = tracker.add_slice(mask)
        >>> component_of = tracker.resolve()   # global id -> 3D component
        >>> sizes = tracker.component_sizes(component_of)

    Attributes:
        connectivity: In-plane connectivity (4 or 8).
        offsets: Global id offset of each slice (``global = local + offset``).
    """

    def __init__(self, connectivity: int) -> None:
        """Start an empty tracker.

        Args:
            connectivity: In-plane connectivity (8 -> 26-connected in 3D, 4 -> 6-connected).
        """
        self.connectivity: int = connectivity
        self.offsets: list[int] = []
        self._n_ids: int = 1  # id 0 is the background
        self._areas: list[NDArray[np.int64]] = [np.zeros(1, dtype=np.int64)]
        self._edges: list[NDArray[np.int64]] = []
        self._prev: NDArray[np.int32] | None = None

    @property
    def n_ids(self) -> int:
        """Number of global ids allocated so far (background included)."""
        return self._n_ids

    def local_labels(self, mask: "NDArray[np.bool_]") -> "NDArray[np.int32]":
        """Label ``mask`` in 2D again.

        Labelling is deterministic, so for the mask of slice ``z`` this reproduces
        what :meth:`add_slice` returned; add ``offsets[z]`` to get global ids.

        Args:
            mask: Binary slice.

        Returns:
            Local labels (0 = background).
        """
        _, local = morph.label(mask, self.connectivity)
        return local

    def add_slice(self, mask: "NDArray[np.bool_]") -> "NDArray[np.int32]":
        """Label the next slice and link it to the previous one.

        Args:
            mask: Binary slice.

        Returns:
            Local labels of the slice (0 = background); global ids are
            ``local + offsets[-1]`` for non-zero labels.
        """
        offset = self._n_ids - 1
        local, areas = morph.label_with_areas(mask, self.connectivity)
        if self._prev is not None:
            self._edges.append(
                cross_slice_pairs(self._prev, self.offsets[-1], local, offset, self.connectivity)
            )
        self.offsets.append(offset)
        self._areas.append(areas[1:].astype(np.int64))
        self._n_ids += len(areas) - 1
        self._prev = local
        return local

    def resolve(self) -> "NDArray[np.int64]":
        """Solve the accumulated graph.

        Returns:
            Lookup table ``global id -> 3D component`` (component 0 = background).
        """
        edges = np.concatenate(self._edges) if self._edges else np.empty((0, 2), np.int64)
        return connected_roots(self._n_ids, edges)

    def component_sizes(self, component_of: "NDArray[np.int64]") -> "NDArray[np.int64]":
        """Count the voxels of each 3D component.

        Args:
            component_of: Lookup table from :meth:`resolve`.

        Returns:
            Voxels per component (index 0 = background, always 0).
        """
        return self.per_component(component_of, np.concatenate(self._areas))

    @staticmethod
    def per_component(
        component_of: "NDArray[np.int64]", per_id: "NDArray[np.int64]"
    ) -> "NDArray[np.int64]":
        """Sum a per-global-id quantity over each 3D component.

        Args:
            component_of: Lookup table from :meth:`resolve`.
            per_id: Quantity per global id.

        Returns:
            Sum per component.
        """
        n_components = int(component_of.max()) + 1
        return np.bincount(component_of, weights=per_id, minlength=n_components).astype(np.int64)
