"""Tests of the network built from centreline voxels (:mod:`pybones.quantification.graph`)."""

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.config.quantification import PruningConfig
from pybones.quantification.graph import build_network
from pybones.quantification.links import drop_corner_links, find_links
from pybones.quantification.network import NO_BRANCH, NO_NODE, EndKind, NodeKind
from tests.helpers import KEEP_ALL, line

if TYPE_CHECKING:
    from pybones.quantification.network import Network

SHAPE = (20, 40, 40)
UNIT = (1.0, 1.0, 1.0)
#: Rules that prune nothing (to test the grouping alone).
RULES = PruningConfig(
    spur_radius_factor=1.5, min_spur_um=3.0, min_fragment_um=5.0, junction_merge_factor=1.0
)
RADIUS = 2.0


def network(
    voxels: list[tuple[int, int, int]],
    rules: "PruningConfig" = KEEP_ALL,
    kinds: "dict[tuple[int, int, int], EndKind] | None" = None,
) -> "Network":
    """Build the network of hand-placed centreline voxels (duplicates merged).

    Args:
        voxels: Centreline voxels.
        rules: Pruning thresholds.
        kinds: What lies beyond some voxels (default: free).

    Returns:
        The network.
    """
    unique = sorted(set(voxels))
    points = np.asarray(unique, dtype=np.int32)
    end_kind = np.full(len(points), EndKind.FREE, np.uint8)
    for voxel, kind in (kinds or {}).items():
        end_kind[unique.index(voxel)] = kind
    radius = np.full(len(points), RADIUS, np.float32)
    return build_network(points, radius, end_kind, SHAPE, UNIT, rules)


def kinds_of(net: "Network") -> list[int]:
    """List the node kinds, sorted.

    Args:
        net: Network.

    Returns:
        Sorted :class:`NodeKind` values.
    """
    return sorted(int(k) for k in net.node_kind)


def test_a_straight_line_is_one_branch_between_two_ends() -> None:
    """Twenty-one voxels in a row: one branch, 20 µm long, two free ends."""
    net = network(line((5, 5, 5), (5, 5, 25)))
    assert len(net.branch_length_um) == 1
    assert net.branch_length_um[0] == pytest.approx(20.0)
    assert kinds_of(net) == [NodeKind.END, NodeKind.END]
    assert (net.node_degree == 1).all()


def test_a_diagonal_line_counts_its_diagonal_steps() -> None:
    """A line along the space diagonal: steps of sqrt(3) each."""
    net = network(line((2, 2, 2), (12, 12, 12)))
    assert net.branch_length_um[0] == pytest.approx(10 * math.sqrt(3))


def test_staircase_corners_are_not_forks() -> None:
    """A 2D staircase (every voxel touches the next two): no junction, one branch."""
    stairs = [(5, 5 + k // 2 + k % 2, 5 + k // 2) for k in range(12)]
    links = find_links(np.asarray(stairs, np.int32), SHAPE, UNIT)
    kept = drop_corner_links(links, len(stairs))
    degree = np.bincount(kept.pairs.ravel(), minlength=len(stairs))
    assert degree.max() == 2  # every corner triangle lost its diagonal
    net = network(stairs)
    assert len(net.branch_length_um) == 1
    assert kinds_of(net) == [NodeKind.END, NodeKind.END]
    # Walking the staircase: one axis step per voxel (the digitization bias of such chains).
    assert net.branch_length_um[0] == pytest.approx(len(stairs) - 1)


def test_dropping_corners_never_splits_the_skeleton() -> None:
    """A solid 2x2x2 block and a 3x3 plate stay one piece after the corner links go."""
    block = [(z, y, x) for z in (4, 5) for y in (4, 5) for x in (4, 5)]
    plate = [(10, y, x) for y in (10, 11, 12) for x in (10, 11, 12)]
    for voxels in (block, plate):
        points = np.asarray(voxels, np.int32)
        kept = drop_corner_links(find_links(points, SHAPE, UNIT), len(voxels))
        reached = {0}
        for _ in range(len(voxels)):
            for a, b in kept.pairs:
                if a in reached or b in reached:
                    reached |= {int(a), int(b)}
        assert reached == set(range(len(voxels)))


def test_a_fork_is_one_junction_with_three_branches() -> None:
    """A Y: three arms meeting at one node of degree 3."""
    centre = (5, 20, 20)
    voxels = line(centre, (5, 20, 5)) + line(centre, (5, 32, 30)) + line(centre, (5, 8, 30))
    net = network(voxels)
    assert len(net.branch_length_um) == 3
    assert kinds_of(net) == [NodeKind.JUNCTION, NodeKind.END, NodeKind.END, NodeKind.END]
    junction = int(np.flatnonzero(net.node_kind == NodeKind.JUNCTION)[0])
    assert net.node_degree[junction] == 3
    ends = np.concatenate([net.branch_start, net.branch_end])
    assert (ends == junction).sum() == 3
    assert net.branch_length_um.sum() == pytest.approx(
        15 + 2 * math.dist((20, 20), (32, 30)), rel=0.15
    )


def test_a_cross_is_one_node_of_degree_four() -> None:
    """Two lines crossing: their touching junction voxels make a single node."""
    voxels = line((5, 20, 5), (5, 20, 35)) + line((5, 5, 20), (5, 35, 20))
    net = network(voxels)
    junctions = np.flatnonzero(net.node_kind == NodeKind.JUNCTION)
    assert len(junctions) == 1
    assert net.node_degree[junctions[0]] == 4
    assert len(net.branch_length_um) == 4


def test_a_ring_is_a_closed_branch_without_nodes() -> None:
    """A square ring: one branch, no node, as long as its perimeter."""
    ring = line((5, 10, 10), (5, 10, 20)) + line((5, 10, 20), (5, 20, 20))
    ring += line((5, 20, 20), (5, 20, 10)) + line((5, 20, 10), (5, 10, 10))
    net = network(ring)
    assert len(net.branch_length_um) == 1
    assert net.branch_start[0] == net.branch_end[0] == NO_NODE
    assert len(net.node_kind) == 0
    assert net.branch_length_um[0] == pytest.approx(40.0)


def test_a_short_spur_is_pruned_and_its_vessel_rejoined() -> None:
    """A two-voxel bump on a line goes; the line becomes one branch again, its voxels kept."""
    main = line((5, 20, 5), (5, 20, 35))
    bump = [(5, 21, 20), (5, 22, 20)]
    net = network(main + bump, RULES)
    assert len(net.branch_length_um) == 1
    assert kinds_of(net) == [NodeKind.END, NodeKind.END]
    assert net.branch_length_um[0] == pytest.approx(30.0)
    removed = [
        tuple(p)
        for p, b in zip(net.points, net.point_branch, strict=True)
        if b == NO_BRANCH and tuple(p) in bump
    ]
    assert set(removed) == set(bump)


def test_a_long_side_branch_is_kept() -> None:
    """A side branch longer than the spur limit is a vessel: two forks' worth of branches stay."""
    main = line((5, 20, 5), (5, 20, 35))
    side = line((5, 21, 20), (5, 32, 20))
    net = network(main + side, RULES)
    assert len(net.branch_length_um) == 3


def test_a_short_branch_against_a_face_is_kept() -> None:
    """A stub ending at a face of the stack is a cut vessel, not a bump: never pruned."""
    main = line((5, 20, 5), (5, 20, 35))
    stub = [(4, 20, 20), (3, 20, 20)]
    net = network(main + stub, RULES, kinds={(3, 20, 20): EndKind.FACE})
    assert len(net.branch_length_um) == 3
    assert NodeKind.FACE in kinds_of(net)


def test_specks_are_removed() -> None:
    """An isolated fragment shorter than the minimum disappears with its nodes."""
    net = network(line((5, 5, 5), (5, 5, 8)), RULES)
    assert len(net.branch_length_um) == 0
    assert len(net.node_kind) == 0
    assert (net.point_branch == NO_BRANCH).all()
    assert (net.point_node == NO_NODE).all()


def test_every_voxel_belongs_to_one_branch_or_one_node() -> None:
    """After pruning, kept voxels sit on exactly one branch or one node; references are valid."""
    centre = (5, 20, 20)
    voxels = line(centre, (5, 20, 2)) + line(centre, (5, 35, 30)) + line(centre, (5, 5, 30))
    voxels += [(5, 21, 10), (5, 22, 10)]  # a spur to prune
    net = network(voxels, RULES)
    on_branch, on_node = net.point_branch >= 0, net.point_node >= 0
    assert not (on_branch & on_node).any()
    assert net.point_branch.max() < len(net.branch_length_um)
    assert net.point_node.max() < len(net.node_kind)
    ends = np.concatenate([net.branch_start, net.branch_end])
    assert ends.max() < len(net.node_kind)
    np.testing.assert_array_equal(
        np.bincount(ends[ends >= 0], minlength=len(net.node_kind)), net.node_degree
    )


def test_two_close_forks_become_one_node() -> None:
    """Two junctions three voxels apart on a thick vessel (radius 2 each) are one fork."""
    main = line((5, 20, 2), (5, 20, 38))
    up, down = line((5, 21, 18), (5, 32, 18)), line((5, 19, 21), (5, 8, 21))
    net = network(main + up + down, RULES)
    junctions = np.flatnonzero(net.node_kind == NodeKind.JUNCTION)
    assert len(junctions) == 1
    assert net.node_degree[junctions[0]] == 4
    assert len(net.branch_length_um) == 4
    kept = network(main + up + down, KEEP_ALL)
    assert (kept.node_kind == NodeKind.JUNCTION).sum() == 2  # without the rule: two forks
    total = net.branch_length_um.sum() + net.node_length_um.sum()
    assert total == pytest.approx(kept.branch_length_um.sum())  # the length between them stays


def test_a_small_ring_at_a_fork_is_removed() -> None:
    """A tiny loop from a junction back to itself (shorter than 2 pi r) is a cavity's ring."""
    main = line((5, 20, 2), (5, 20, 38))
    loop = [(5, 21, 19), (5, 22, 20), (5, 21, 21)]  # leaves the line at x=19, back at x=21
    net = network(main + loop, RULES)
    assert len(net.branch_length_um) == 1
    assert kinds_of(net) == [NodeKind.END, NodeKind.END]
