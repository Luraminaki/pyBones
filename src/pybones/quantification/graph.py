"""From centreline voxels to a pruned vessel network.

The steps live in their own modules: :mod:`~pybones.quantification.links` (touching voxels,
staircase corners), :mod:`~pybones.quantification.network` (nodes and branches) and
:mod:`~pybones.quantification.pruning` (artefacts removed).
"""

from typing import TYPE_CHECKING

import numpy as np

from pybones.quantification.links import drop_corner_links, find_links
from pybones.quantification.network import (
    NO_NODE,
    Network,
    initial_graph,
    node_degree,
    node_geometry,
)
from pybones.quantification.pruning import prune

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.config.quantification import PruningConfig


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
        end_kind: ``(N,)`` :class:`~pybones.quantification.network.EndKind` of each voxel
            (what lies beyond it, read at ends).
        shape: Volume shape.
        voxel_um: Voxel size ``(z, y, x)`` in µm.
        rules: Pruning thresholds.

    Returns:
        The pruned network; nodes without branches are dropped and the rest renumbered.
    """
    links = drop_corner_links(find_links(points, shape, voxel_um), len(points))
    points_um = points.astype(np.float64) * np.asarray(voxel_um)
    distances = distances_um.astype(np.float64)
    state = prune(initial_graph(points, links, end_kind), points_um, distances, rules)
    degree = node_degree(state)
    geometry = node_geometry(state, points_um, distances)
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
