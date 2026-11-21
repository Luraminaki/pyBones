"""Pruning a vessel network: one kind of operation at a time, until none applies.

* a branch from a junction to a free end, shorter than the vessel's width there, is a bump of
  the surface and goes; so does an isolated fragment shorter than ``min_fragment_um``, and a
  loop from a junction back to itself shorter than the circumference there (a ring around the
  vessel, left by a cavity);
* two junctions joined by a branch shorter than their radii together are one fork (a thick
  vessel often thins into two nearby forks): they merge, the branch's length kept in the node;
* the two branches meeting at a node left with only them become one.

Each operation acts on every instance at once; only the passes loop.
"""

from typing import TYPE_CHECKING

import numpy as np

from pybones.quantification.network import (
    NO_BRANCH,
    NO_NODE,
    PATH_DEGREE,
    GraphState,
    NodeKind,
    node_degree,
    node_geometry,
    two_ends,
)
from pybones.segmentation.connectivity import connected_roots

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.config.quantification import PruningConfig

_PAIR = 2
#: A loop around the vessel itself is shorter than its circumference, 2 pi r.
TAU = 2 * np.pi


def _remove(state: GraphState, removed: "NDArray[np.bool_]") -> GraphState:
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
    return GraphState(
        start=state.start[keep],
        end=state.end[keep],
        length=state.length[keep],
        kind=state.kind,
        node_length=state.node_length,
        point_branch=renumber[state.point_branch],
        point_node=state.point_node,
    )


def _contract(state: GraphState, short: "NDArray[np.bool_]") -> GraphState:
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
    return GraphState(
        start=group_ext[removed.start],
        end=group_ext[removed.end],
        length=removed.length,
        kind=kind,
        node_length=node_length,
        point_branch=removed.point_branch,
        point_node=point_node,
    )


def _merge_pairs(state: GraphState, dissolve: "NDArray[np.bool_]") -> GraphState:
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
    start, end = two_ends(group[end_branch[outer]], end_node[outer], n_groups)
    point_branch = np.append(group, NO_BRANCH)[state.point_branch]
    node_to_branch = np.full(len(dissolve) + 1, NO_BRANCH, np.int64)
    node_to_branch[end_node[at]] = group[end_branch[at]]
    absorbed = dissolve_ext[state.point_node]
    point_branch[absorbed] = node_to_branch[state.point_node[absorbed]]
    point_node = state.point_node.copy()
    point_node[absorbed] = NO_NODE
    return GraphState(start, end, length, state.kind, state.node_length, point_branch, point_node)


def _bubbles(
    state: GraphState, between_forks: "NDArray[np.bool_]", radius: "NDArray[np.float64]"
) -> "NDArray[np.bool_]":
    """Find the extra strands of bubbles: branches joining the same two junctions as a shorter one.

    Where the vessel mask has a hole or a fold, its skeleton splits into two strands for a short
    way: two branches between the same two junctions. When the loop they make is shorter than the
    vessel's circumference there, it is one vessel, and the longer strand goes.

    Args:
        state: Current network.
        between_forks: Branches joining two different junctions.
        radius: Larger radius of each branch's two end nodes.

    Returns:
        The branches to remove.
    """
    candidates = np.flatnonzero(between_forks)
    if len(candidates) == 0:
        return np.zeros(len(state.length), np.bool_)
    n_nodes = len(state.kind)
    low = np.minimum(state.start[candidates], state.end[candidates])
    high = np.maximum(state.start[candidates], state.end[candidates])
    pair = low * n_nodes + high
    order = np.lexsort((state.length[candidates], pair))  # by pair, shortest first
    ranked, pair = candidates[order], pair[order]
    first = np.concatenate([[True], pair[1:] != pair[:-1]])
    shortest = np.maximum.accumulate(np.where(first, np.arange(len(pair)), 0))
    loop = state.length[ranked] + state.length[ranked[shortest]]
    extra = ~first & (loop < TAU * radius[ranked])
    removed = np.zeros(len(state.length), np.bool_)
    removed[ranked[extra]] = True
    return removed


def prune(
    state: GraphState,
    points_um: "NDArray[np.float64]",
    distances_um: "NDArray[np.float64]",
    rules: "PruningConfig",
) -> GraphState:
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
        degree = node_degree(state)
        # A junction left with a single branch is that branch's free end now.
        state.kind[(state.kind == NodeKind.JUNCTION) & (degree == 1)] = NodeKind.END
        geometry = node_geometry(state, points_um, distances_um)
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
        bubble = _bubbles(
            state, s_fork & e_fork & (s != e), np.maximum(radius_ext[s], radius_ext[e])
        )
        removable = spur | speck | ring | bubble
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
        dissolve = (state.kind == NodeKind.JUNCTION) & (degree == PATH_DEGREE)
        if dissolve.any():
            state = _merge_pairs(state, dissolve)
            continue
        return state
