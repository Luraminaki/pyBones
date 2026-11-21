"""Vessels of the phantom: smooth random walks of capsules, and the true network they draw.

Each vessel is a walk of capsules whose radius drifts; some segments are clotted (dim: only
a fraction of the contrast agent) or empty (gaps: invisible, excluded from scores). A walk
may branch, starting a thinner child at its current point. Every segment drawn is recorded
in a :class:`VesselNetwork`, the ground truth of the quantification.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from tools.phantom.canvas import paint, unit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tools.phantom.canvas import Canvas
    from tools.phantom.settings import PhantomSettings

_HALF = 0.5
_DIMS = 3


class SegmentState(IntEnum):
    """What a recorded vessel segment holds."""

    FILLED = 0
    CLOTTED = 1
    GAP = 2


@dataclass(frozen=True, slots=True)
class VesselNetwork:
    """The true vessel network of a phantom, as drawn (coordinates ``(z, y, x)`` in µm).

    Attributes:
        segments_um: ``(N, 2, 3)`` start and end of every drawn segment (partly outside
            the volume for some).
        radius_um: ``(N,)`` radius of each segment.
        branch: ``(N,)`` branch each segment belongs to.
        state: ``(N,)`` :class:`SegmentState` of each segment.
        branch_parent: ``(B,)`` parent branch of each branch (-1 for a vessel's root).
        junctions_um: ``(J, 3)`` bifurcation points (where a branch leaves its parent).
    """

    segments_um: "NDArray[np.float32]"
    radius_um: "NDArray[np.float32]"
    branch: "NDArray[np.int32]"
    state: "NDArray[np.uint8]"
    branch_parent: "NDArray[np.int32]"
    junctions_um: "NDArray[np.float32]"

    @classmethod
    def empty(cls) -> "VesselNetwork":
        """Build a network without any vessel.

        Returns:
            Zero-length arrays of the right shapes.
        """
        return cls(
            segments_um=np.zeros((0, 2, _DIMS), np.float32),
            radius_um=np.zeros(0, np.float32),
            branch=np.zeros(0, np.int32),
            state=np.zeros(0, np.uint8),
            branch_parent=np.zeros(0, np.int32),
            junctions_um=np.zeros((0, _DIMS), np.float32),
        )


@dataclass(slots=True)
class _NetworkRecorder:
    """Collects the segments, branches and junctions of :class:`VesselNetwork` as drawn.

    Attributes:
        segments: Start and end of each segment.
        radius: Radius of each segment.
        branch: Branch of each segment.
        state: State of each segment.
        parents: Parent of each branch.
        junctions: Bifurcation points.
    """

    segments: "list[tuple[NDArray[np.float64], NDArray[np.float64]]]" = field(default_factory=list)
    radius: list[float] = field(default_factory=list)
    branch: list[int] = field(default_factory=list)
    state: list[SegmentState] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    junctions: "list[NDArray[np.float64]]" = field(default_factory=list)

    def new_branch(self, parent: int) -> int:
        """Start a branch.

        Args:
            parent: Parent branch (-1 for a vessel's root).

        Returns:
            The new branch's index.
        """
        self.parents.append(parent)
        return len(self.parents) - 1

    def network(self) -> VesselNetwork:
        """Freeze what was recorded.

        Returns:
            The network.
        """
        if not self.segments:
            return VesselNetwork.empty()
        return VesselNetwork(
            segments_um=np.asarray([np.stack(ab) for ab in self.segments], np.float32),
            radius_um=np.asarray(self.radius, np.float32),
            branch=np.asarray(self.branch, np.int32),
            state=np.asarray(self.state, np.uint8),
            branch_parent=np.asarray(self.parents, np.int32),
            junctions_um=np.asarray(self.junctions, np.float32).reshape(-1, _DIMS),
        )


@dataclass(frozen=True, slots=True)
class _Walk:
    """A vessel branch still to draw.

    Attributes:
        point: Start ``(z, y, x)`` in µm.
        direction: Unit direction.
        radius: Radius in µm.
        depth: Generation (0 for a vessel's root).
        branch: Index of the branch in the recorder.
    """

    point: "NDArray[np.float64]"
    direction: "NDArray[np.float64]"
    radius: float
    depth: int
    branch: int


def _branch_direction(
    direction: "NDArray[np.float64]", angle_deg: float, rng: np.random.Generator
) -> "NDArray[np.float64]":
    """Turn a direction by an angle, around a random axis perpendicular to it.

    Args:
        direction: Unit direction of the parent.
        angle_deg: Angle between parent and branch.
        rng: Random generator.

    Returns:
        The branch's unit direction.
    """
    side = rng.normal(size=_DIMS)
    side = unit(side - np.dot(side, direction) * direction)
    angle = np.radians(angle_deg)
    return unit(np.cos(angle) * direction + np.sin(angle) * side)


def draw_vessels(
    canvas: "Canvas", settings: "PhantomSettings", rng: np.random.Generator
) -> VesselNetwork:
    """Draw the vessels: random walks of capsules, some segments clotted or empty, branching.

    With ``branch_probability`` 0 the random draws are exactly those of unbranched walks,
    so the segmentation phantom does not change when the network phantom is not asked for.

    Args:
        canvas: Canvas to draw on.
        settings: Phantom settings.
        rng: Random generator.

    Returns:
        The network as drawn.
    """
    ves, spread = settings.vessels, settings.intensity.vessel_spread
    extent = np.asarray(settings.shape_zyx) * canvas.voxel
    turn = np.sin(np.radians(ves.turn_std_deg))
    record = _NetworkRecorder()
    for _ in range(ves.count):
        radius = rng.uniform(*ves.radius_um)
        level = rng.uniform(1 - spread, 1 + spread)  # vessel-to-vessel filling variation
        point = rng.uniform(0, 1, _DIMS) * extent
        direction = unit(rng.normal(size=_DIMS))
        walks = [_Walk(point, direction, radius, 0, record.new_branch(-1))]
        while walks:
            walk = walks.pop()
            point, direction = walk.point, walk.direction
            for _ in range(rng.integers(ves.segments[0], ves.segments[1] + 1)):
                direction = unit(direction + rng.normal(0, turn, _DIMS))
                nxt = point + direction * ves.segment_length_um
                shape = canvas.capsule(point, nxt, walk.radius)
                start, point = point, nxt
                if shape is None:
                    continue
                slices, occ = shape
                draw = rng.uniform()
                inside = occ >= _HALF
                record.segments.append((start, nxt))
                record.radius.append(walk.radius)
                record.branch.append(walk.branch)
                if draw < ves.gap_probability:
                    record.state.append(SegmentState.GAP)
                    paint(canvas.gap, slices, occ)
                else:
                    clotted = draw < ves.gap_probability + ves.clot_probability
                    record.state.append(SegmentState.CLOTTED if clotted else SegmentState.FILLED)
                    fill = rng.uniform(*ves.clot_fill) if clotted else 1.0
                    paint(canvas.signal, slices, occ * (fill * level))
                    paint(canvas.vessel, slices, occ)
                    canvas.radius[slices][inside] = np.maximum(
                        canvas.radius[slices][inside], walk.radius
                    )
                    if clotted:
                        canvas.clot[slices] |= inside
                # Drawn only when branching is on: the unbranched draws stay identical.
                if (
                    ves.branch_probability > 0
                    and walk.depth < ves.branch_depth
                    and rng.uniform() < ves.branch_probability
                ):
                    child = walk.radius * rng.uniform(*ves.branch_radius_ratio)
                    angle = rng.uniform(*ves.branch_angle_deg)
                    if child >= ves.radius_um[0]:
                        turned = _branch_direction(direction, angle, rng)
                        walks.append(
                            _Walk(
                                point, turned, child, walk.depth + 1, record.new_branch(walk.branch)
                            )
                        )
                        record.junctions.append(point)
    return record.network()
