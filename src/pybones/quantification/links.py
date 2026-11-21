"""Links between touching centreline voxels, without the staircase corners.

1. **Links**: skeleton voxels that touch (26-neighbourhood) are linked; a link's length is
   the µm distance between the two voxel centres.
2. **Corners**: where three voxels touch each other, the longest link of the triangle is
   dropped (a diagonal step is not a fork), but only if its ends stay joined through the
   triangle's two other links: dropping never splits the skeleton.

Vectorized over voxels and links.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Half of the 26-neighbourhood: each link is found once, from its lexicographically
#: smaller voxel.
FORWARD_OFFSETS = np.asarray(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) > (0, 0, 0)
    ],
    dtype=np.int64,
)
_PAIR = 2


@dataclass(frozen=True, slots=True)
class Links:
    """Links between touching centreline voxels.

    Attributes:
        pairs: ``(M, 2)`` voxel indices.
        lengths_um: ``(M,)`` distance between the two voxel centres.
    """

    pairs: "NDArray[np.int64]"
    lengths_um: "NDArray[np.float64]"


def _keys(points: "NDArray[np.int64]", shape: tuple[int, int, int]) -> "NDArray[np.int64]":
    """Flatten voxel indices into single integers (row-major order).

    Args:
        points: ``(N, 3)`` voxel indices.
        shape: Volume shape.

    Returns:
        ``(N,)`` flat indices.
    """
    return (points[:, 0] * shape[1] + points[:, 1]) * shape[2] + points[:, 2]


def find_links(
    points: "NDArray[np.int32]",
    shape: tuple[int, int, int],
    voxel_um: tuple[float, float, float],
) -> Links:
    """Link every pair of touching voxels (26-neighbourhood).

    Args:
        points: ``(N, 3)`` centreline voxels.
        shape: Volume shape.
        voxel_um: Voxel size ``(z, y, x)`` in µm.

    Returns:
        The links, each found once.
    """
    if len(points) == 0:
        return Links(np.empty((0, _PAIR), np.int64), np.empty(0))
    pts = points.astype(np.int64)
    keys = _keys(pts, shape)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    bounds = np.asarray(shape, dtype=np.int64)
    pairs: list[NDArray[np.int64]] = []
    lengths: list[NDArray[np.float64]] = []
    for offset in FORWARD_OFFSETS:
        other = pts + offset
        valid = np.all((other >= 0) & (other < bounds), axis=1)
        other_keys = _keys(other, shape)
        pos = np.minimum(np.searchsorted(sorted_keys, other_keys), len(keys) - 1)
        hit = valid & (sorted_keys[pos] == other_keys)
        found = np.flatnonzero(hit)
        pairs.append(np.stack([found, order[pos[hit]]], axis=1))
        step = float(np.linalg.norm(offset * np.asarray(voxel_um)))
        lengths.append(np.full(len(found), step))
    return Links(np.concatenate(pairs), np.concatenate(lengths))


def _link_lookup(
    pairs: "NDArray[np.int64]", n_points: int
) -> "tuple[NDArray[np.int64], NDArray[np.int64]]":
    """Index the links by their (smaller, larger) voxel pair.

    Args:
        pairs: ``(M, 2)`` links.
        n_points: Number of voxels.

    Returns:
        Sorted pair keys and, for each, the link index.
    """
    keys = np.minimum(pairs[:, 0], pairs[:, 1]) * n_points + np.maximum(pairs[:, 0], pairs[:, 1])
    order = np.argsort(keys)
    return keys[order], order


def _find_link(
    a: "NDArray[np.int64]",
    b: "NDArray[np.int64]",
    lookup: "tuple[NDArray[np.int64], NDArray[np.int64]]",
    n_points: int,
) -> "NDArray[np.int64]":
    """Find the links between voxel pairs.

    Args:
        a: First voxels.
        b: Second voxels.
        lookup: From :func:`_link_lookup`.
        n_points: Number of voxels.

    Returns:
        Link index per pair, -1 where the voxels are not linked.
    """
    sorted_keys, order = lookup
    if len(sorted_keys) == 0:
        return np.full(len(a), -1, np.int64)
    keys = np.minimum(a, b) * n_points + np.maximum(a, b)
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return np.where(sorted_keys[pos] == keys, order[pos], -1)


def drop_corner_links(links: Links, n_points: int) -> Links:
    """Drop the longest link of each triangle of touching voxels, keeping connectivity.

    Args:
        links: All links.
        n_points: Number of voxels.

    Returns:
        The links kept.
    """
    pairs = links.pairs
    if len(pairs) == 0:
        return links
    # Neighbour lists (both directions), grouped by voxel.
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    degree = np.bincount(src, minlength=n_points)
    start = np.concatenate([[0], np.cumsum(degree)])[:-1]
    lookup = _link_lookup(pairs, n_points)
    triangles: list[NDArray[np.int64]] = []
    for p in range(int(degree.max()) - 1):
        for q in range(p + 1, int(degree.max())):
            centre = np.flatnonzero(degree > q)
            a, b = dst[start[centre] + p], dst[start[centre] + q]
            closing = _find_link(a, b, lookup, n_points)
            keep = closing >= 0
            if keep.any():
                centre, a, b, closing = centre[keep], a[keep], b[keep], closing[keep]
                triangles.append(
                    np.stack(
                        [
                            _find_link(centre, a, lookup, n_points),
                            _find_link(centre, b, lookup, n_points),
                            closing,
                        ],
                        axis=1,
                    )
                )
    if not triangles:
        return links
    tri = np.concatenate(triangles)
    # Longest link of each triangle; between equal lengths, the largest index (deterministic).
    # Lengths are compared first and exactly: folding both into one number would let an
    # index difference outweigh a length difference.
    tri_lengths = links.lengths_um[tri]
    is_longest = tri_lengths == tri_lengths.max(axis=1, keepdims=True)
    longest = np.where(is_longest, tri, -1).max(axis=1)
    dropped = np.zeros(len(pairs), dtype=np.bool_)
    dropped[longest] = True
    # Keep a dropped link unless some triangle joins its ends through two kept links.
    others = tri[tri != longest[:, None]].reshape(len(tri), _PAIR)
    bypassed = np.zeros(len(pairs), dtype=np.bool_)
    bypassed[longest[~dropped[others].any(axis=1)]] = True
    drop = dropped & bypassed
    return Links(pairs[~drop], links.lengths_um[~drop])
