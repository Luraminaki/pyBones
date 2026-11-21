"""From centreline voxels to a vessel network: nodes, branches, spur pruning.

1. **Links**: skeleton voxels that touch (26-neighbourhood) are linked; a link's length is
   the µm distance between the two voxel centres.
2. **Corners**: where three voxels touch each other, the longest link of the triangle is
   dropped (a diagonal step is not a fork), but only if its ends stay joined through the
   triangle's two other links: dropping never splits the skeleton.
3. **Nodes**: a voxel with one link is an end, with three or more a junction; touching
   junction voxels form one node.
4. **Branches**: runs of two-link voxels between nodes (or closed on themselves), plus
   direct node-to-node links. A branch's length is the sum of its links, the links to its
   end nodes included.
5. **Pruning**, one kind of operation at a time until none applies:

   * a branch from a junction to a free end, shorter than the vessel's width there, is a
     bump of the surface and goes; so does an isolated fragment shorter than
     ``min_fragment_um``, and a loop from a junction back to itself shorter than the
     circumference there (a ring around the vessel, left by a cavity);
   * two junctions joined by a branch shorter than their radii together are one fork
     (a thick vessel often thins into two nearby forks): they merge;
   * the two branches meeting at a node left with only them become one.

Ends against a face of the stack or a burst are *anchored*: the vessel goes on beyond
them, so they are never pruned, and the branches reaching them are flagged.

Every step is vectorized over voxels, links or branches; only the pruning passes loop.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from pybones.segmentation.connectivity import connected_roots

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.config.schema import PruningConfig

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
_DIMS = 3
_PAIR = 2
#: A loop around the vessel itself is shorter than its circumference, 2 pi r.
TAU = 2 * np.pi
#: Degrees of the voxel classes.
_END_DEGREE = 1
_PATH_DEGREE = 2
NO_NODE = -1
NO_BRANCH = -1


class EndKind(IntEnum):
    """What lies beyond a centreline end."""

    FREE = 0
    FACE = 1
    BURST = 2


class NodeKind(IntEnum):
    """Role of a node in the network."""

    JUNCTION = 0
    END = 1
    FACE = 2
    BURST = 3


@dataclass(frozen=True, slots=True)
class Links:
    """Links between touching centreline voxels.

    Attributes:
        pairs: ``(M, 2)`` voxel indices.
        lengths_um: ``(M,)`` distance between the two voxel centres.
    """

    pairs: "NDArray[np.int64]"
    lengths_um: "NDArray[np.float64]"


@dataclass(frozen=True, slots=True)
class Network:
    """A vessel network.

    Attributes:
        points: ``(N, 3)`` centreline voxels ``(z, y, x)``.
        point_branch: ``(N,)`` branch of each voxel (:data:`NO_BRANCH` for node voxels and
            removed ones).
        point_node: ``(N,)`` node of each voxel (:data:`NO_NODE` if none).
        node_position_um: ``(K, 3)`` centre of each node's voxels, ``(z, y, x)`` µm.
        node_kind: ``(K,)`` :class:`NodeKind`.
        node_degree: ``(K,)`` branch ends at each node (a loop counts twice).
        node_length_um: ``(K,)`` length of the short branches contracted into each node (the
            vessel between two forks merged into one): part of the network's total length.
        branch_start: ``(B,)`` first node of each branch (:data:`NO_NODE` for a closed loop).
        branch_end: ``(B,)`` last node of each branch.
        branch_length_um: ``(B,)`` length along the centreline.
    """

    points: "NDArray[np.int32]"
    point_branch: "NDArray[np.int32]"
    point_node: "NDArray[np.int32]"
    node_position_um: "NDArray[np.float64]"
    node_kind: "NDArray[np.uint8]"
    node_degree: "NDArray[np.int32]"
    node_length_um: "NDArray[np.float64]"
    branch_start: "NDArray[np.int32]"
    branch_end: "NDArray[np.int32]"
    branch_length_um: "NDArray[np.float64]"


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


@dataclass(slots=True)
class _State:
    """The network while it is pruned (arrays replaced, never shared with the caller).

    Attributes:
        start: First node per branch (:data:`NO_NODE` for a closed loop).
        end: Last node per branch.
        length: Length per branch (µm).
        kind: :class:`NodeKind` per node.
        node_length: Length of the branches contracted into each node (µm).
        point_branch: Branch per voxel (:data:`NO_BRANCH` if none).
        point_node: Node per voxel (:data:`NO_NODE` if none).
    """

    start: "NDArray[np.int64]"
    end: "NDArray[np.int64]"
    length: "NDArray[np.float64]"
    kind: "NDArray[np.uint8]"
    node_length: "NDArray[np.float64]"
    point_branch: "NDArray[np.int64]"
    point_node: "NDArray[np.int64]"


@dataclass(frozen=True, slots=True)
class _Geometry:
    """Where the nodes are and how wide the vessel is there (derived from their voxels).

    Attributes:
        position: Centre of each node's voxels, ``(z, y, x)`` µm.
        radius: Largest wall distance among each node's voxels (µm).
    """

    position: "NDArray[np.float64]"
    radius: "NDArray[np.float64]"


def _geometry(
    state: _State, points_um: "NDArray[np.float64]", distances_um: "NDArray[np.float64]"
) -> _Geometry:
    """Compute the nodes' geometry from the voxels they own.

    Args:
        state: Current network.
        points_um: Voxel centres in µm.
        distances_um: Wall distance per voxel.

    Returns:
        Position and radius per node (zero for nodes without voxels).
    """
    n_nodes = len(state.kind)
    own = state.point_node >= 0
    node = state.point_node[own]
    count = np.maximum(np.bincount(node, minlength=n_nodes), 1)
    position = (
        np.stack(
            [np.bincount(node, points_um[own, axis], n_nodes) for axis in range(_DIMS)], axis=1
        )
        / count[:, None]
    )
    radius = np.zeros(n_nodes)
    np.maximum.at(radius, node, distances_um[own])
    return _Geometry(position=position, radius=radius)


def _initial_graph(
    points: "NDArray[np.int32]", links: Links, end_kind: "NDArray[np.uint8]"
) -> _State:
    """Group the voxels into nodes and branches, before pruning.

    Args:
        points: ``(N, 3)`` centreline voxels.
        links: Links kept after :func:`drop_corner_links`.
        end_kind: ``(N,)`` :class:`EndKind` of each voxel (read at ends only).

    Returns:
        The unpruned network.
    """
    n = len(points)
    pairs, lengths = links.pairs, links.lengths_um
    degree = np.bincount(pairs.ravel(), minlength=n)
    is_junction = degree > _PATH_DEGREE
    is_end = degree == _END_DEGREE
    is_path = degree == _PATH_DEGREE
    a, b = pairs[:, 0], pairs[:, 1]

    # Nodes: each end alone; touching junction voxels together.
    node_voxels = np.flatnonzero(is_junction | is_end)
    compact = np.full(n, -1, np.int64)
    compact[node_voxels] = np.arange(len(node_voxels))
    both_junction = is_junction[a] & is_junction[b]
    node_of_compact = connected_roots(len(node_voxels), compact[pairs[both_junction]])
    point_node = np.full(n, NO_NODE, np.int64)
    point_node[node_voxels] = node_of_compact
    n_nodes = int(node_of_compact.max()) + 1 if len(node_voxels) else 0
    kind = np.full(n_nodes, NodeKind.JUNCTION, np.uint8)
    ends = np.flatnonzero(is_end)
    kind[point_node[ends]] = np.where(
        end_kind[ends] == EndKind.FACE,
        NodeKind.FACE,
        np.where(end_kind[ends] == EndKind.BURST, NodeKind.BURST, NodeKind.END),
    )

    # Runs of path voxels: the branches' interiors.
    path_voxels = np.flatnonzero(is_path)
    run_compact = np.full(n, -1, np.int64)
    run_compact[path_voxels] = np.arange(len(path_voxels))
    both_path = is_path[a] & is_path[b]
    run_of_compact = connected_roots(len(path_voxels), run_compact[pairs[both_path]])
    n_runs = int(run_of_compact.max()) + 1 if len(path_voxels) else 0
    point_branch = np.full(n, NO_BRANCH, np.int64)
    point_branch[path_voxels] = run_of_compact
    run_length = np.bincount(point_branch[a[both_path]], lengths[both_path], n_runs)

    # Where runs meet nodes: two attachments per open run, none for a closed loop.
    attach = is_path[a] != is_path[b]
    run_side = np.where(is_path[a[attach]], a[attach], b[attach])
    node_side = np.where(is_path[a[attach]], b[attach], a[attach])
    att_run, att_node = point_branch[run_side], point_node[node_side]
    run_length += np.bincount(att_run, lengths[attach], n_runs)
    start, end = _two_ends(att_run, att_node, n_runs)

    # Node-to-node links between different nodes: branches without interior voxels.
    direct = ~is_path[a] & ~is_path[b] & (point_node[a] != point_node[b])
    return _State(
        start=np.concatenate([start, point_node[a[direct]]]),
        end=np.concatenate([end, point_node[b[direct]]]),
        length=np.concatenate([run_length, lengths[direct]]),
        kind=kind,
        node_length=np.zeros(n_nodes),
        point_branch=point_branch,
        point_node=point_node,
    )


def _two_ends(
    owner: "NDArray[np.int64]", node: "NDArray[np.int64]", n_owners: int
) -> "tuple[NDArray[np.int64], NDArray[np.int64]]":
    """Gather the (up to two) end nodes listed for each branch.

    Args:
        owner: Branch of each listed end.
        node: Node of each listed end.
        n_owners: Number of branches.

    Returns:
        First and last listed node per branch (:data:`NO_NODE` when none is listed).
    """
    order = np.argsort(owner, kind="stable")
    owner, node = owner[order], node[order]
    first = np.searchsorted(owner, np.arange(n_owners))
    count = np.bincount(owner, minlength=n_owners)
    start = np.full(n_owners, NO_NODE, np.int64)
    end = np.full(n_owners, NO_NODE, np.int64)
    listed = count > 0
    start[listed] = node[first[listed]]
    end[listed] = node[first[listed] + count[listed] - 1]
    return start, end


def _node_degree(state: _State) -> "NDArray[np.int64]":
    """Count the branch ends at each node.

    Args:
        state: Current network.

    Returns:
        Degree per node (a loop at a node counts twice).
    """
    ends = np.concatenate([state.start, state.end])
    return np.bincount(ends[ends >= 0], minlength=len(state.kind))


def _remove(state: _State, removed: "NDArray[np.bool_]") -> _State:
    """Drop branches; their voxels leave the network (their nodes stay, maybe orphaned).

    Args:
        state: Current network.
        removed: Branches to drop.

    Returns:
        The network without them.
    """
    keep = np.flatnonzero(~removed)
    renumber = np.full(len(removed) + 1, NO_BRANCH, np.int64)  # last slot: NO_BRANCH -> itself
    renumber[keep] = np.arange(len(keep))
    return _State(
        start=state.start[keep],
        end=state.end[keep],
        length=state.length[keep],
        kind=state.kind,
        node_length=state.node_length,
        point_branch=renumber[state.point_branch],
        point_node=state.point_node,
    )


def _contract(state: _State, short: "NDArray[np.bool_]") -> _State:
    """Merge the junctions joined by short branches into single nodes.

    Args:
        state: Current network.
        short: Junction-to-junction branches to contract.

    Returns:
        The network with those branches' ends merged and their voxels given to the node.
    """
    pairs = np.stack([state.start[short], state.end[short]], axis=1)
    group = connected_roots(len(state.kind), pairs)
    n_groups = int(group.max()) + 1
    kind = np.empty(n_groups, np.uint8)
    kind[group] = state.kind  # a merged group holds junctions only
    group_ext = np.append(group, NO_NODE)  # NO_NODE (-1) reads the last slot
    point_node = group_ext[state.point_node]
    on_short = np.append(short, False)[state.point_branch]  # NO_BRANCH reads the last slot
    point_node[on_short] = group[state.start[state.point_branch[on_short]]]
    # The contracted branches are vessel between the two forks: their length stays, in the node.
    node_length = np.bincount(group, state.node_length, n_groups)
    node_length += np.bincount(group[state.start[short]], state.length[short], n_groups)
    removed = _remove(state, short)
    return _State(
        start=group_ext[removed.start],
        end=group_ext[removed.end],
        length=removed.length,
        kind=kind,
        node_length=node_length,
        point_branch=removed.point_branch,
        point_node=point_node,
    )


def _merge_pairs(state: _State, dissolve: "NDArray[np.bool_]") -> _State:
    """Join the two branches meeting at each dissolved node (chains join transitively).

    Args:
        state: Current network.
        dissolve: Nodes of degree two to remove.

    Returns:
        The network with each chain of branches made one; the dissolved nodes' voxels join it.
    """
    n_branches = len(state.length)
    both = np.arange(n_branches)
    end_branch = np.concatenate([both, both])
    end_node = np.concatenate([state.start, state.end])
    dissolve_ext = np.append(dissolve, False)
    at = dissolve_ext[end_node]
    order = np.argsort(end_node[at], kind="stable")
    group = connected_roots(n_branches, end_branch[at][order].reshape(-1, _PAIR))
    n_groups = int(group.max()) + 1
    length = np.bincount(group, state.length, n_groups)
    # The merged branch's ends: its members' ends that were not dissolved.
    outer = ~at & (end_node >= 0)
    start, end = _two_ends(group[end_branch[outer]], end_node[outer], n_groups)
    point_branch = np.append(group, NO_BRANCH)[state.point_branch]
    node_to_branch = np.full(len(dissolve) + 1, NO_BRANCH, np.int64)
    node_to_branch[end_node[at]] = group[end_branch[at]]
    absorbed = dissolve_ext[state.point_node]
    point_branch[absorbed] = node_to_branch[state.point_node[absorbed]]
    point_node = state.point_node.copy()
    point_node[absorbed] = NO_NODE
    return _State(start, end, length, state.kind, state.node_length, point_branch, point_node)


def _prune(
    state: _State,
    points_um: "NDArray[np.float64]",
    distances_um: "NDArray[np.float64]",
    rules: "PruningConfig",
) -> _State:
    """Apply the cleaning operations, one kind at a time, until none applies.

    In order of priority: remove spurs, specks and rings; contract short links between
    junctions; join the branches meeting two by two. Each operation acts on every
    instance at once; the loop ends when the network no longer changes.

    Args:
        state: Unpruned network.
        points_um: Voxel centres in µm.
        distances_um: Wall distance per voxel.
        rules: Thresholds.

    Returns:
        The pruned network.
    """
    while True:
        degree = _node_degree(state)
        # A junction left with a single branch is that branch's free end now.
        state.kind[(state.kind == NodeKind.JUNCTION) & (degree == 1)] = NodeKind.END
        geometry = _geometry(state, points_um, distances_um)
        kind_ext = np.append(state.kind, np.uint8(NodeKind.JUNCTION))  # NO_NODE -> last slot
        radius_ext = np.append(geometry.radius, 0.0)
        s, e = state.start, state.end
        s_free, e_free = kind_ext[s] == NodeKind.END, kind_ext[e] == NodeKind.END
        s_fork = (s >= 0) & (kind_ext[s] == NodeKind.JUNCTION)
        e_fork = (e >= 0) & (kind_ext[e] == NodeKind.JUNCTION)
        fork_radius = np.where(s_fork, radius_ext[s], radius_ext[e])
        spur_limit = np.maximum(rules.min_spur_um, rules.spur_radius_factor * fork_radius)
        spur = ((s_free & e_fork) | (e_free & s_fork)) & (state.length < spur_limit)
        speck = s_free & e_free & (state.length < rules.min_fragment_um)
        ring = s_fork & (s == e) & (state.length < TAU * radius_ext[s])
        removable = spur | speck | ring
        if removable.any():
            state = _remove(state, removable)
            continue
        short = (
            s_fork
            & e_fork
            & (s != e)
            & (state.length < rules.junction_merge_factor * (radius_ext[s] + radius_ext[e]))
        )
        if short.any():
            state = _contract(state, short)
            continue
        dissolve = (state.kind == NodeKind.JUNCTION) & (degree == _PATH_DEGREE)
        if dissolve.any():
            state = _merge_pairs(state, dissolve)
            continue
        return state


def build_network(
    points: "NDArray[np.int32]",
    distances_um: "NDArray[np.float32]",
    end_kind: "NDArray[np.uint8]",
    shape: tuple[int, int, int],
    voxel_um: tuple[float, float, float],
    rules: "PruningConfig",
) -> Network:
    """Build the vessel network of a skeleton.

    Args:
        points: ``(N, 3)`` centreline voxels ``(z, y, x)``.
        distances_um: ``(N,)`` distance of each voxel to the vessel wall.
        end_kind: ``(N,)`` :class:`EndKind` of each voxel (what lies beyond it, read at ends).
        shape: Volume shape.
        voxel_um: Voxel size ``(z, y, x)`` in µm.
        rules: Pruning thresholds.

    Returns:
        The pruned network; nodes without branches are dropped and the rest renumbered.
    """
    links = drop_corner_links(find_links(points, shape, voxel_um), len(points))
    points_um = points.astype(np.float64) * np.asarray(voxel_um)
    distances = distances_um.astype(np.float64)
    state = _prune(_initial_graph(points, links, end_kind), points_um, distances, rules)
    degree = _node_degree(state)
    geometry = _geometry(state, points_um, distances)
    kept = np.flatnonzero(degree > 0)
    renumber = np.full(len(state.kind) + 1, NO_NODE, np.int64)  # last slot: NO_NODE -> itself
    renumber[kept] = np.arange(len(kept))
    return Network(
        points=points.astype(np.int32),
        point_branch=state.point_branch.astype(np.int32),
        point_node=renumber[state.point_node].astype(np.int32),
        node_position_um=geometry.position[kept],
        node_kind=state.kind[kept],
        node_degree=degree[kept].astype(np.int32),
        node_length_um=state.node_length[kept],
        branch_start=renumber[state.start].astype(np.int32),
        branch_end=renumber[state.end].astype(np.int32),
        branch_length_um=state.length,
    )
