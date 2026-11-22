"""Scoring a measured vessel network against the phantom's true one.

A true vessel is *visible* where it lies inside the volume, holds contrast agent (not a gap)
and is not hidden: inside a burst, the truth labels its voxels burst, so no vessel mask can
contain it.

* **Centrelines**: the visible true centrelines are sampled every voxel; *recall* is the
  share of samples within ``tolerance_um`` of a measured centreline voxel, *precision* the
  share of measured voxels within ``tolerance_um`` of a true centreline.
* **Junctions**: each visible true bifurcation (on a visible voxel, its side branch visible
  for longer than its parent's radius) is matched to the nearest unmatched measured junction
  within ``junction_tolerance_um`` plus the parent's radius; unmatched measured junctions
  are split into *crossings* (where two different vessels of the phantom pass close to each
  other: a real junction of the mask, not a bifurcation) and *spurious* ones.
* **Calibre**: at the measured voxels near a true centreline, the measured wall distance is
  compared with the true radius there, overall and per band of true radius.
* **Length**: measured total length against the true one.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pybones.quantification.network import NO_BRANCH, NO_NODE, NodeKind
from pybones.segmentation.connectivity import connected_roots
from pybones.utils.constants import UM_PER_MM
from tools.phantom.vessels import SegmentState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.quantification.network import Network
    from tools.phantom.vessels import VesselNetwork

_CHUNK = 2048
#: Coordinates compared at once by the brute-force searches (8 bytes each: ~64 MB).
_PAIR_BUDGET = 8_000_000
_DIMS = 3


@dataclass(frozen=True, slots=True)
class NetworkScore:
    """How a measured network compares with the true one.

    Attributes:
        centreline_recall: Share of the true centreline found.
        centreline_precision: Share of the measured centreline that is true.
        true_junctions: Visible bifurcations (inside the volume, side branch visible).
        matched_junctions: Those found.
        crossing_junctions: Measured junctions where two different true vessels meet.
        spurious_junctions: Other unmatched measured junctions.
        independent_cycles: Independent loops of the measured network (branches - nodes +
            connected pieces): a tree has none; crossings and holes in the mask each add one.
        radius_offset_um: Median of (measured wall distance - true radius).
        radius_abs_error_um: Median absolute error of the radius after removing the offset.
        radius_offset_by_band_um: Median of (measured - true radius) per true-radius band
            (NaN for a band with no measured voxel).
        true_length_mm: Visible true centreline length.
        measured_length_mm: Measured total length.
    """

    centreline_recall: float
    centreline_precision: float
    true_junctions: int
    matched_junctions: int
    crossing_junctions: int
    spurious_junctions: int
    independent_cycles: int
    radius_offset_um: float
    radius_abs_error_um: float
    radius_offset_by_band_um: tuple[float, ...]
    true_length_mm: float
    measured_length_mm: float


def _point_segment_distances(
    points: "NDArray[np.float64]", a: "NDArray[np.float64]", b: "NDArray[np.float64]"
) -> "tuple[NDArray[np.float64], NDArray[np.int64]]":
    """Find, for each point, the nearest segment and its distance (chunked brute force).

    Args:
        points: ``(N, 3)`` points.
        a: ``(S, 3)`` segment starts.
        b: ``(S, 3)`` segment ends.

    Returns:
        Distance to the nearest segment, and its index, per point.
    """
    ab = b - a
    ab2 = np.maximum((ab**2).sum(axis=1), 1e-12)
    best = np.full(len(points), np.inf)
    which = np.zeros(len(points), np.int64)
    for start in range(0, len(points), _CHUNK):
        p = points[start : start + _CHUNK, None, :]
        t = np.clip(((p - a) * ab).sum(axis=2) / ab2, 0.0, 1.0)
        d = np.linalg.norm(p - (a + t[..., None] * ab), axis=2)
        which[start : start + _CHUNK] = d.argmin(axis=1)
        best[start : start + _CHUNK] = d.min(axis=1)
    return best, which


def _to_segments(
    point: "NDArray[np.float64]", a: "NDArray[np.float64]", b: "NDArray[np.float64]"
) -> "NDArray[np.float64]":
    """Measure the distance from one point to every segment.

    Args:
        point: ``(3,)`` point.
        a: ``(S, 3)`` segment starts.
        b: ``(S, 3)`` segment ends.

    Returns:
        ``(S,)`` distances.
    """
    ab = b - a
    t = np.clip(((point - a) * ab).sum(axis=1) / np.maximum((ab**2).sum(axis=1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(point - (a + t[:, None] * ab), axis=1)


def _visible_points(
    points_um: "NDArray[np.float64]",
    hidden: "NDArray[np.bool_]",
    voxel: "NDArray[np.float64]",
) -> "NDArray[np.bool_]":
    """Tell which points lie inside the volume, on a voxel where a vessel can be seen.

    Args:
        points_um: ``(N, 3)`` points in µm.
        hidden: ``(Z, Y, X)`` voxels where no vessel can be seen (bursts).
        voxel: Voxel size ``(z, y, x)`` in µm.

    Returns:
        ``(N,)`` visibility of each point.
    """
    extent = (np.asarray(hidden.shape) - 1) * voxel
    inside = np.all((points_um >= 0) & (points_um <= extent), axis=1)
    index = np.clip(np.rint(points_um / voxel).astype(np.int64), 0, np.asarray(hidden.shape) - 1)
    return inside & ~hidden[tuple(index.T)]


def _visible_length(
    truth: "VesselNetwork", hidden: "NDArray[np.bool_]", voxel: "NDArray[np.float64]"
) -> "NDArray[np.float64]":
    """Measure how much of each true branch is visible (see the module).

    Args:
        truth: True network.
        hidden: Voxels where no vessel can be seen.
        voxel: Voxel size in µm.

    Returns:
        Visible length per branch (µm), counted by segment midpoints.
    """
    a, b = truth.segments_um[:, 0].astype(np.float64), truth.segments_um[:, 1].astype(np.float64)
    seen = _visible_points((a + b) / 2, hidden, voxel) & (truth.state != SegmentState.GAP)
    length = np.linalg.norm(b - a, axis=1) * seen
    return np.bincount(truth.branch, length, minlength=len(truth.branch_parent))


def _visible_junctions(
    truth: "VesselNetwork", hidden: "NDArray[np.bool_]", voxel: "NDArray[np.float64]"
) -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
    """Keep the bifurcations an image can show, with how far from them a fork may appear.

    A bifurcation counts if it lies on a visible voxel and its side branch is visible for
    longer than its parent's radius (otherwise no fork is drawn). The fork of a thick
    parent is found anywhere across it: the reach is the parent's radius.

    Args:
        truth: True network.
        hidden: Voxels where no vessel can be seen.
        voxel: Voxel size in µm.

    Returns:
        Visible bifurcations ``(J, 3)`` µm, and the parent's radius at each.
    """
    side = np.flatnonzero(truth.branch_parent >= 0)  # in the order the junctions were drawn
    radius = np.zeros(len(truth.branch_parent))
    np.maximum.at(radius, truth.branch, truth.radius_um.astype(np.float64))
    parent_radius = radius[truth.branch_parent[side]]
    points = truth.junctions_um.astype(np.float64).reshape(-1, _DIMS)
    side_length = _visible_length(truth, hidden, voxel)[side]
    visible = _visible_points(points, hidden, voxel) & (side_length > parent_radius)
    return points[visible], parent_radius[visible]


def _root_vessel(truth: "VesselNetwork") -> "NDArray[np.int64]":
    """Find the root vessel (tree) of each branch of the true network.

    Args:
        truth: True network.

    Returns:
        Root branch per branch.
    """
    root = np.arange(len(truth.branch_parent), dtype=np.int64)
    parent = truth.branch_parent.astype(np.int64)
    while True:
        up = parent[root]
        moved = up >= 0
        if not moved.any():
            return root
        root = np.where(moved, up, root)


def independent_cycles(network: "Network") -> int:
    """Count the independent loops of a network (its circuit rank).

    Args:
        network: The network.

    Returns:
        Branches - nodes + connected pieces, over the branches joining two nodes (a branch
        closed on itself, with no node, is one loop).
    """
    start, end = network.branch_start.astype(np.int64), network.branch_end.astype(np.int64)
    linked = (start != NO_NODE) & (end != NO_NODE)
    closed = int(((start == NO_NODE) & (end == NO_NODE)).sum())
    nodes = np.unique(np.concatenate([start[linked], end[linked]]))
    if len(nodes) == 0:
        return closed
    index = np.searchsorted(nodes, np.stack([start[linked], end[linked]], axis=1))
    pieces = len(np.unique(connected_roots(len(nodes), index)))
    return int(linked.sum()) - len(nodes) + pieces + closed


def score_network(
    measured: "Network",
    distances_um: "NDArray[np.float32]",
    truth: "VesselNetwork",
    hidden: "NDArray[np.bool_]",
    voxel_um: tuple[float, float, float],
    tolerance_um: float,
    junction_tolerance_um: float,
    radius_bins_um: tuple[float, ...],
) -> NetworkScore:
    """Score a measured network against the phantom's true network.

    Args:
        measured: Network built from a segmentation of the phantom.
        distances_um: Wall distance of each of its voxels.
        truth: The phantom's true network.
        hidden: ``(Z, Y, X)`` voxels of the volume where no vessel can be seen (the true
            bursts): the true network is not scored there.
        voxel_um: Voxel size ``(z, y, x)`` in µm.
        tolerance_um: Distance under which a centreline voxel and a true centreline agree.
        junction_tolerance_um: Distance under which a measured junction matches a true one.
        radius_bins_um: Edges of the true-radius bands of the calibre error (increasing).

    Returns:
        The scores.
    """
    voxel = np.asarray(voxel_um)
    a = truth.segments_um[:, 0].astype(np.float64)
    b = truth.segments_um[:, 1].astype(np.float64)
    filled = truth.state != SegmentState.GAP

    # True centreline samples, every voxel, where visible.
    length = np.linalg.norm(b - a, axis=1)
    steps = np.maximum(np.ceil(length / voxel.min()).astype(int), 1)
    seg_of = np.repeat(np.arange(len(a)), steps)
    t = (np.arange(len(seg_of)) - np.repeat(np.cumsum(steps) - steps, steps) + 0.5) / steps[seg_of]
    samples = a[seg_of] + t[:, None] * (b - a)[seg_of]
    inside = _visible_points(samples, hidden, voxel) & filled[seg_of]
    sample_step = length[seg_of] / steps[seg_of]
    true_length = float(sample_step[inside].sum())

    kept = (measured.point_branch != NO_BRANCH) | (measured.point_node != NO_NODE)
    voxels = measured.points[kept].astype(np.float64) * voxel
    # Precision: measured voxels near any true centreline (gaps included: they are vessels).
    to_truth, nearest = _point_segment_distances(voxels, a, b)
    precision = float((to_truth <= tolerance_um).mean()) if len(voxels) else 0.0
    # Recall: true samples near a measured voxel (same brute force, roles swapped).
    found = np.zeros(int(inside.sum()), np.bool_)
    true_points = samples[inside]
    if len(voxels):
        # Rows per chunk so that the (rows x voxels x 3) temporary stays within the budget.
        rows = max(1, _PAIR_BUDGET // (len(voxels) * _DIMS))
        for start in range(0, len(true_points), rows):
            chunk = true_points[start : start + rows, None, :]
            found[start : start + rows] = (
                np.linalg.norm(chunk - voxels[None, :, :], axis=2).min(axis=1) <= tolerance_um
            )
    recall = float(found.mean()) if len(found) else 0.0

    # Calibre at the measured voxels near a true centreline.
    near = to_truth <= tolerance_um
    true_radius = truth.radius_um[nearest[near]].astype(np.float64)
    error = distances_um[kept][near].astype(np.float64) - true_radius
    offset = float(np.median(error)) if len(error) else 0.0
    abs_error = float(np.median(np.abs(error - offset))) if len(error) else 0.0
    band = np.digitize(true_radius, radius_bins_um) - 1  # -1 and len(bins) - 1: outside
    by_band = tuple(
        float(np.median(error[band == k])) if (band == k).any() else math.nan
        for k in range(len(radius_bins_um) - 1)
    )

    # Junctions: greedy nearest matching, then classify the leftovers.
    true_j, reach = _visible_junctions(truth, hidden, voxel)
    reach = reach + junction_tolerance_um
    measured_j = measured.node_position_um[measured.node_kind == NodeKind.JUNCTION]
    matched = np.zeros(len(measured_j), np.bool_)
    n_matched = 0
    if len(true_j) and len(measured_j):
        d = np.linalg.norm(true_j[:, None, :] - measured_j[None, :, :], axis=2)
        for k in np.argsort(d.min(axis=1)):
            candidates = np.flatnonzero(~matched & (d[k] <= reach[k]))
            if len(candidates):
                matched[candidates[np.argmin(d[k, candidates])]] = True
                n_matched += 1
    leftovers = measured_j[~matched]
    root = _root_vessel(truth)[truth.branch]
    crossings = sum(
        int(len(np.unique(root[_to_segments(point, a, b) <= junction_tolerance_um])) > 1)
        for point in leftovers
    )
    return NetworkScore(
        centreline_recall=recall,
        centreline_precision=precision,
        true_junctions=len(true_j),
        matched_junctions=n_matched,
        crossing_junctions=crossings,
        spurious_junctions=len(leftovers) - crossings,
        independent_cycles=independent_cycles(measured),
        radius_offset_um=offset,
        radius_abs_error_um=abs_error,
        radius_offset_by_band_um=by_band,
        true_length_mm=true_length / UM_PER_MM,
        measured_length_mm=float(measured.branch_length_um.sum() + measured.node_length_um.sum())
        / UM_PER_MM,
    )
