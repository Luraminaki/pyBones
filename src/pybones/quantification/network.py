"""The vessel network: its types, and how centreline voxels become nodes and branches.

* **Nodes**: a voxel with one link is an end, with three or more a junction; touching
  junction voxels form one node.
* **Branches**: runs of two-link voxels between nodes (or closed on themselves), plus
  direct node-to-node links. A branch's length is the sum of its links, the links to its
  end nodes included.

Ends against a face of the stack or a burst are *anchored* (:class:`EndKind`): the vessel
goes on beyond them, so pruning never removes them and the branches reaching them are flagged.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from pybones.segmentation.connectivity import connected_roots

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.quantification.links import Links

_DIMS = 3
#: Degrees of the voxel classes (links per voxel).
_END_DEGREE = 1
PATH_DEGREE = 2
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


@dataclass(slots=True)
class GraphState:
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
class NodeGeometry:
    """Where the nodes are and how wide the vessel is there (derived from their voxels).

    Attributes:
        position: Centre of each node's voxels, ``(z, y, x)`` µm.
        radius: Largest wall distance among each node's voxels (µm).
    """

    position: "NDArray[np.float64]"
    radius: "NDArray[np.float64]"


def node_geometry(
    state: GraphState, points_um: "NDArray[np.float64]", distances_um: "NDArray[np.float64]"
) -> NodeGeometry:
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
    return NodeGeometry(position=position, radius=radius)


def initial_graph(
    points: "NDArray[np.int32]", links: "Links", end_kind: "NDArray[np.uint8]"
) -> GraphState:
    """Group the voxels into nodes and branches, before pruning.

    Args:
        points: ``(N, 3)`` centreline voxels.
        links: Links kept after :func:`~pybones.quantification.links.drop_corner_links`.
        end_kind: ``(N,)`` :class:`EndKind` of each voxel (read at ends only).

    Returns:
        The unpruned network.
    """
    n = len(points)
    pairs, lengths = links.pairs, links.lengths_um
    degree = np.bincount(pairs.ravel(), minlength=n)
    is_junction = degree > PATH_DEGREE
    is_end = degree == _END_DEGREE
    is_path = degree == PATH_DEGREE
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
    start, end = two_ends(att_run, att_node, n_runs)

    # Node-to-node links between different nodes: branches without interior voxels.
    direct = ~is_path[a] & ~is_path[b] & (point_node[a] != point_node[b])
    return GraphState(
        start=np.concatenate([start, point_node[a[direct]]]),
        end=np.concatenate([end, point_node[b[direct]]]),
        length=np.concatenate([run_length, lengths[direct]]),
        kind=kind,
        node_length=np.zeros(n_nodes),
        point_branch=point_branch,
        point_node=point_node,
    )


def two_ends(
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


def node_degree(state: GraphState) -> "NDArray[np.int64]":
    """Count the branch ends at each node.

    Args:
        state: Current network.

    Returns:
        Degree per node (a loop at a node counts twice).
    """
    ends = np.concatenate([state.start, state.end])
    return np.bincount(ends[ends >= 0], minlength=len(state.kind))
