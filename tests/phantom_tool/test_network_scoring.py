"""Tests of network scoring against the phantom (:mod:`tools.phantom.network_scoring`)."""

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.quantification.graph import build_network
from pybones.quantification.network import EndKind
from tests.helpers import KEEP_ALL, line
from tools.phantom.network_scoring import independent_cycles, score_network
from tools.phantom.vessels import SegmentState, VesselNetwork

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.quantification.network import Network

SHAPE = (20, 60, 60)
VOXEL = (2.0, 2.0, 2.0)
RADIUS = 4.0
TOLERANCE = 4.0
JUNCTION_TOLERANCE = 6.0
#: Radius bands of the calibre error: the true radius (4 µm) falls in the first.
RADIUS_BINS = (0.0, 6.0, 30.0)
#: No burst anywhere: every true vessel of the volume can be seen.
NOTHING_HIDDEN = np.zeros(SHAPE, np.bool_)
#: Length of the parent left visible by a burst from voxel x = 35: from 4 to 69 µm.
VISIBLE_LENGTH_MM = 0.065


def true_network(paths: list[list[tuple[int, int, int]]], parents: list[int]) -> VesselNetwork:
    """Describe polylines of voxels as a true network (one branch each), in µm.

    A branch with a parent starts at its first point, which is recorded as a junction.

    Args:
        paths: Corner voxels of each branch.
        parents: Parent branch of each (-1 for a root).

    Returns:
        The network.
    """
    voxel = np.asarray(VOXEL)
    segments, branch = [], []
    for index, corners in enumerate(paths):
        for a, b in itertools.pairwise(corners):
            segments.append([np.asarray(a) * voxel, np.asarray(b) * voxel])
            branch.append(index)
    junctions = [np.asarray(paths[i][0]) * voxel for i, p in enumerate(parents) if p >= 0]
    return VesselNetwork(
        segments_um=np.asarray(segments, np.float32),
        radius_um=np.full(len(segments), RADIUS, np.float32),
        branch=np.asarray(branch, np.int32),
        state=np.full(len(segments), SegmentState.FILLED, np.uint8),
        branch_parent=np.asarray(parents, np.int32),
        junctions_um=np.asarray(junctions, np.float32).reshape(-1, 3),
    )


def measured(paths: list[list[tuple[int, int, int]]]) -> "tuple[Network, NDArray[np.float32]]":
    """Build the network of voxels laid exactly on polylines, with the true radius.

    Args:
        paths: Corner voxels of each polyline.

    Returns:
        The network and the wall distance of each of its voxels.
    """
    voxels = sorted(
        {v for corners in paths for a, b in itertools.pairwise(corners) for v in line(a, b)}
    )
    points = np.asarray(voxels, np.int32)
    distances = np.full(len(points), RADIUS, np.float32)
    kinds = np.full(len(points), EndKind.FREE, np.uint8)
    return build_network(points, distances, kinds, SHAPE, VOXEL, KEEP_ALL), distances


def test_a_network_laid_on_the_truth_scores_perfectly() -> None:
    """Centrelines found and true, the fork matched, the radius exact, the length right."""
    paths = [[(10, 30, 2), (10, 30, 57)], [(10, 30, 30), (10, 55, 30)]]
    net, distances = measured(paths)
    score = score_network(
        net,
        distances,
        true_network(paths, [-1, 0]),
        NOTHING_HIDDEN,
        VOXEL,
        TOLERANCE,
        JUNCTION_TOLERANCE,
        RADIUS_BINS,
    )
    assert score.centreline_recall == pytest.approx(1.0)
    assert score.centreline_precision == pytest.approx(1.0)
    assert (score.true_junctions, score.matched_junctions) == (1, 1)
    assert (score.crossing_junctions, score.spurious_junctions) == (0, 0)
    assert score.independent_cycles == 0  # a tree
    assert (score.radius_offset_um, score.radius_abs_error_um) == (0.0, 0.0)
    assert score.radius_offset_by_band_um[0] == 0.0
    assert math.isnan(score.radius_offset_by_band_um[1])  # no vessel that thick
    assert score.measured_length_mm == pytest.approx(score.true_length_mm, rel=0.05)


def test_a_crossing_of_two_vessels_is_not_a_spurious_fork() -> None:
    """Two separate vessels crossing: their meeting is a crossing, no bifurcation is missed."""
    paths = [[(10, 30, 2), (10, 30, 57)], [(10, 2, 30), (10, 57, 30)]]
    net, distances = measured(paths)
    score = score_network(
        net,
        distances,
        true_network(paths, [-1, -1]),
        NOTHING_HIDDEN,
        VOXEL,
        TOLERANCE,
        JUNCTION_TOLERANCE,
        RADIUS_BINS,
    )
    assert score.true_junctions == 0
    assert (score.crossing_junctions, score.spurious_junctions) == (1, 0)
    assert score.independent_cycles == 0  # a cross is a node of degree four, not a loop


def test_missing_and_extra_centrelines_lower_recall_and_precision() -> None:
    """A true vessel never measured lowers recall; a measured one not in the truth, precision."""
    truth = true_network([[(10, 30, 2), (10, 30, 57)], [(10, 10, 2), (10, 10, 57)]], [-1, -1])
    net, distances = measured([[(10, 30, 2), (10, 30, 57)], [(10, 50, 2), (10, 50, 57)]])
    score = score_network(
        net, distances, truth, NOTHING_HIDDEN, VOXEL, TOLERANCE, JUNCTION_TOLERANCE, RADIUS_BINS
    )
    assert score.centreline_recall == pytest.approx(0.5, abs=0.05)
    assert score.centreline_precision == pytest.approx(0.5, abs=0.05)


def test_a_loop_is_counted_once_alone_or_on_a_vessel() -> None:
    """A ring with no node, or a ring hanging from a vessel, is one independent loop."""
    ring = [(10, 20, 20), (10, 20, 40), (10, 40, 40), (10, 40, 20), (10, 20, 20)]
    alone, _ = measured([ring])
    assert independent_cycles(alone) == 1
    on_vessel, _ = measured([ring, [(10, 20, 20), (10, 5, 5)]])
    assert independent_cycles(on_vessel) == 1


def test_vessels_hidden_by_a_burst_are_not_scored() -> None:
    """A true vessel inside a burst is expected neither as a centreline nor as a fork."""
    paths = [[(10, 30, 2), (10, 30, 57)], [(10, 30, 40), (10, 55, 40)]]
    hidden = np.zeros(SHAPE, np.bool_)
    hidden[:, :, 35:] = True  # a burst over the fork and the far half of the parent
    net, distances = measured([[(10, 30, 2), (10, 30, 34)]])
    score = score_network(
        net,
        distances,
        true_network(paths, [-1, 0]),
        hidden,
        VOXEL,
        TOLERANCE,
        JUNCTION_TOLERANCE,
        RADIUS_BINS,
    )
    assert score.true_junctions == 0
    assert score.centreline_recall == pytest.approx(1.0)
    assert score.true_length_mm == pytest.approx(VISIBLE_LENGTH_MM, rel=0.05)
