"""What lies beyond each centreline end, and the measures of a vessel network.

Radii are the wall distances of the centreline voxels, uncorrected: on the network phantom
they are within half a voxel of the true radius, thin vessels of a segmentation coming out
slightly wide and thick ones slightly narrow (see the report for the measured bias).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from pybones.quantification.network import NO_BRANCH, NO_NODE, EndKind, NodeKind
from pybones.utils.constants import UM_PER_MM

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.quantification.network import Network

_DIMS = 3


def end_kinds(
    points: "NDArray[np.int32]",
    distances_um: "NDArray[np.float32]",
    burst_distances_um: "NDArray[np.float32]",
    shape: tuple[int, int, int],
    voxel_um: tuple[float, float, float],
    margin_um: float,
) -> "NDArray[np.uint8]":
    """Say, for each centreline voxel, whether a vessel ending there would go on beyond it.

    A centreline stops short of a face of the stack, or of a burst, by about the vessel's
    radius: an end within its own radius plus ``margin_um`` of a face continues outside the
    stack (:attr:`~.network.EndKind.FACE`, which wins), of a burst inside it
    (:attr:`~.network.EndKind.BURST`).

    Args:
        points: ``(N, 3)`` centreline voxels.
        distances_um: ``(N,)`` wall distance of each.
        burst_distances_um: ``(N,)`` distance of each to the nearest burst voxel.
        shape: Volume shape.
        voxel_um: Voxel size ``(z, y, x)`` in µm.
        margin_um: Extra reach beyond the radius.

    Returns:
        ``(N,)`` :class:`~.network.EndKind` per voxel.
    """
    voxel = np.asarray(voxel_um)
    position = points.astype(np.float64) * voxel
    far = (np.asarray(shape) - 1) * voxel
    to_face = np.minimum(position, far - position).min(axis=1)
    kind = np.full(len(points), EndKind.FREE, np.uint8)
    kind[within_reach(distances_um, burst_distances_um, margin_um)] = EndKind.BURST
    kind[within_reach(distances_um, to_face, margin_um)] = EndKind.FACE
    return kind


def within_reach(
    distances_um: "NDArray[np.float32]", target_um: "NDArray[np.floating]", margin_um: float
) -> "NDArray[np.bool_]":
    """Tell which centreline voxels have something within their vessel's radius plus a margin.

    A centreline stops short of what it touches by about the vessel's radius: a face of the
    stack, a burst, the bone.

    Args:
        distances_um: ``(N,)`` wall distance of each voxel (the vessel's radius there).
        target_um: ``(N,)`` distance of each voxel to the nearest thing of interest.
        margin_um: Extra reach beyond the radius.

    Returns:
        ``(N,)`` whether the thing is within reach.
    """
    return target_um <= distances_um.astype(np.float64) + margin_um


@dataclass(frozen=True, slots=True)
class BranchMeasures:
    """Measures of each branch of a network (one entry per branch).

    Attributes:
        length_um: Length along the centreline.
        chord_um: Straight distance between its end nodes (0 for a loop).
        tortuosity: Length / chord (NaN for a loop).
        radius_mean_um: Mean wall distance of its voxels (NaN without interior voxels).
        radius_median_um: Median wall distance.
        radius_min_um: Smallest wall distance.
        radius_max_um: Largest wall distance.
        voxels: Interior centreline voxels.
        truncated: An end continues beyond a face of the stack.
        at_burst: An end continues into a burst.
    """

    length_um: "NDArray[np.float64]"
    chord_um: "NDArray[np.float64]"
    tortuosity: "NDArray[np.float64]"
    radius_mean_um: "NDArray[np.float64]"
    radius_median_um: "NDArray[np.float64]"
    radius_min_um: "NDArray[np.float64]"
    radius_max_um: "NDArray[np.float64]"
    voxels: "NDArray[np.int64]"
    truncated: "NDArray[np.bool_]"
    at_burst: "NDArray[np.bool_]"


def _group_median(
    groups: "NDArray[np.int64]", values: "NDArray[np.float64]", n: int
) -> "NDArray[np.float64]":
    """Median of ``values`` within each group, vectorized (NaN for an empty group).

    Args:
        groups: Group of each value.
        values: Values.
        n: Number of groups.

    Returns:
        Median per group.
    """
    order = np.lexsort((values, groups))
    sorted_values = values[order]
    count = np.bincount(groups, minlength=n)
    first = np.concatenate([[0], np.cumsum(count)])[:-1]
    out = np.full(n, np.nan)
    has = count > 0
    low = first[has] + (count[has] - 1) // 2
    high = first[has] + count[has] // 2
    out[has] = (sorted_values[low] + sorted_values[high]) / 2
    return out


def branch_measures(network: "Network", distances_um: "NDArray[np.float32]") -> BranchMeasures:
    """Measure every branch of a network.

    Args:
        network: The network.
        distances_um: ``(N,)`` wall distance of each centreline voxel.

    Returns:
        The measures, branch by branch.
    """
    n = len(network.branch_length_um)
    on = network.point_branch != NO_BRANCH
    branch = network.point_branch[on].astype(np.int64)
    dist = distances_um[on].astype(np.float64)
    voxels = np.bincount(branch, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(branch, dist, n) / voxels
    low = np.full(n, np.inf)
    high = np.full(n, -np.inf)
    np.minimum.at(low, branch, dist)
    np.maximum.at(high, branch, dist)
    empty = voxels == 0
    low[empty], high[empty] = np.nan, np.nan
    start, end = network.branch_start.astype(np.int64), network.branch_end.astype(np.int64)
    position = np.vstack([network.node_position_um, np.zeros((1, _DIMS))])  # NO_NODE -> last row
    chord = np.where(
        (start == NO_NODE) | (start == end),
        0.0,
        np.linalg.norm(position[start] - position[end], axis=1),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        tortuosity = np.where(chord > 0, network.branch_length_um / chord, np.nan)
    kind = np.append(network.node_kind, np.uint8(NodeKind.JUNCTION))
    ends = np.stack([kind[start], kind[end]])
    return BranchMeasures(
        length_um=network.branch_length_um,
        chord_um=chord,
        tortuosity=tortuosity,
        radius_mean_um=mean,
        radius_median_um=_group_median(branch, dist, n),
        radius_min_um=low,
        radius_max_um=high,
        voxels=voxels,
        truncated=(ends == NodeKind.FACE).any(axis=0),
        at_burst=(ends == NodeKind.BURST).any(axis=0),
    )


class NetworkSummary(BaseModel):
    """The network of a segmentation in a few numbers (JSON-ready).

    Attributes:
        branches: Branches.
        truncated_branches: Branches cut by a face of the stack.
        loops: Branches closed on themselves (alone, or from a fork back to it).
        junctions: Forks (nodes of three branches or more).
        free_ends: Ends of vessels inside the stack.
        face_ends: Ends where a vessel leaves the stack.
        burst_ends: Ends where a vessel runs into a burst.
        total_length_mm: Total centreline length (forks merged into one node included).
        stack_volume_mm3: Volume of the whole stack.
        length_density_mm_per_mm3: Total length per stack volume.
        median_branch_length_um: Median branch length.
        median_radius_um: Median wall distance, over the centreline voxels.
        median_tortuosity: Median tortuosity (loops excluded).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    branches: int
    truncated_branches: int
    loops: int
    junctions: int
    free_ends: int
    face_ends: int
    burst_ends: int
    total_length_mm: float
    stack_volume_mm3: float
    length_density_mm_per_mm3: float
    median_branch_length_um: float
    median_radius_um: float
    median_tortuosity: float


def _median(values: "NDArray[np.float64]") -> float:
    """Median of the finite values (NaN if none).

    Args:
        values: Values.

    Returns:
        Their median.
    """
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def summarize(
    network: "Network",
    measures: BranchMeasures,
    distances_um: "NDArray[np.float32]",
    shape: tuple[int, int, int],
    voxel_um: tuple[float, float, float],
) -> NetworkSummary:
    """Summarize a network.

    Args:
        network: The network.
        measures: Its branch measures.
        distances_um: Wall distance of each centreline voxel.
        shape: Volume shape.
        voxel_um: Voxel size ``(z, y, x)`` in µm.

    Returns:
        The summary.
    """
    kind = network.node_kind
    kept = (network.point_branch != NO_BRANCH) | (network.point_node != NO_NODE)
    total = float(network.branch_length_um.sum() + network.node_length_um.sum()) / UM_PER_MM
    volume = float(np.prod(np.asarray(shape) * np.asarray(voxel_um))) / UM_PER_MM**3
    return NetworkSummary(
        branches=len(network.branch_length_um),
        truncated_branches=int(measures.truncated.sum()),
        loops=int((network.branch_start == network.branch_end).sum()),
        junctions=int((kind == NodeKind.JUNCTION).sum()),
        free_ends=int((kind == NodeKind.END).sum()),
        face_ends=int((kind == NodeKind.FACE).sum()),
        burst_ends=int((kind == NodeKind.BURST).sum()),
        total_length_mm=total,
        stack_volume_mm3=volume,
        length_density_mm_per_mm3=total / volume if volume else float("nan"),
        median_branch_length_um=_median(measures.length_um),
        median_radius_um=_median(distances_um[kept].astype(np.float64)),
        median_tortuosity=_median(measures.tortuosity),
    )
