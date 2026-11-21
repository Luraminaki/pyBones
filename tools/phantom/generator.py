"""Rendering of a synthetic barium-perfused bone µCT volume with exact ground truth.

Shapes are rendered as partial-volume occupancies (linear fall-off across one
voxel at their surface), each only inside its own bounding box:

* cortex (thick slabs) and trabeculae (disc plates, cylindrical rods): bone;
* vessels: smooth random walks of capsules, some segments clotted (dim: only a
  fraction of the contrast agent) or empty (gaps: invisible, excluded from scores);
* bursts: spheres of leaked contrast agent (dense speckles + diffuse haze).

The image is the bone and contrast signals composed by maximum, blurred by a
Gaussian PSF and corrupted by intensity-dependent Gaussian noise. Ground truth
is each class's occupancy >= 1/2 (priority: excluded > burst > vessel > bone).
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.segmentation.labels import Label
from pybones.utils.constants import UINT16_MAX

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tools.phantom.settings import PhantomSettings

_HALF = 0.5
_DIMS = 3
#: Cortex walls sit at 8 % and 92 % of the width, slightly tilted.
_CORTEX_X = 0.08
_CORTEX_TILT = 0.1
#: Candidate rod ends drawn on each bone surface.
_ANCHORS_PER_SURFACE = 60
#: Burst centres are drawn in the central part of the volume (fraction of each extent).
_BURST_CENTRE_RANGE = (0.25, 0.75)
#: Gaussian blur kernels along z are truncated at this many sigmas.
_BLUR_TRUNCATE_SIGMAS = 3
_EPS = 1e-12

#: A shape rendered into a bounding box: voxel slices of the box and occupancy inside it.
Shape = tuple[tuple[slice, ...], "NDArray[np.float32]"]


@dataclass(frozen=True, slots=True)
class Phantom:
    """A rendered phantom.

    Attributes:
        image: ``(Z, Y, X)`` ``uint16`` µCT-like volume.
        truth: ``(Z, Y, X)`` ``uint8`` labels (pybones values, plus the excluded value).
        vessel_radius_um: Radius of the vessel each truth-vessel voxel belongs to (0 elsewhere).
        clotted: Truth-vessel voxels of clotted (dim) segments.
        network: The vessel network as drawn (centrelines, radii, bifurcations).
    """

    image: "NDArray[np.uint16]"
    truth: "NDArray[np.uint8]"
    vessel_radius_um: "NDArray[np.float32]"
    clotted: "NDArray[np.bool_]"
    network: "VesselNetwork"


def _paint(field: "NDArray[np.float32]", slices: tuple[slice, ...], values: "NDArray") -> None:
    """Compose a shape into an occupancy field by maximum (in place).

    Args:
        field: Occupancy field.
        slices: Bounding box of the shape in ``field``.
        values: Occupancy of the shape inside the box.
    """
    _ = np.maximum(field[slices], values, out=field[slices])


class _Canvas:
    """Occupancy fields and the partial-volume shape primitives drawing into them.

    Attributes:
        shape: ``(Z, Y, X)`` volume shape.
        voxel: Voxel size ``(z, y, x)`` in µm.
        pv_width: Width of the partial-volume fall-off at shape surfaces, in µm.
        bone: Bone occupancy (image and truth).
        signal: Contrast-agent signal (image only: clotted vessels are dimmer).
        vessel: Vessel occupancy (truth).
        gap: Occupancy of empty vessel segments (excluded from the scores).
        burst: Burst-sphere occupancy (truth).
        radius: Radius of the vessel covering each voxel, in µm.
        clot: Voxels of clotted vessel segments.
    """

    def __init__(self, shape: tuple[int, int, int], voxel_um: tuple[float, float, float]) -> None:
        """Allocate empty fields.

        Args:
            shape: ``(Z, Y, X)`` volume shape.
            voxel_um: Voxel size ``(z, y, x)`` in µm.
        """
        self.shape: tuple[int, int, int] = shape
        self.voxel: NDArray[np.float64] = np.asarray(voxel_um, dtype=np.float64)
        self.pv_width: float = float(self.voxel.min())
        self.bone: NDArray[np.float32] = self._zeros()
        self.signal: NDArray[np.float32] = self._zeros()
        self.vessel: NDArray[np.float32] = self._zeros()
        self.gap: NDArray[np.float32] = self._zeros()
        self.burst: NDArray[np.float32] = self._zeros()
        self.radius: NDArray[np.float32] = self._zeros()
        self.clot: NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)

    def _zeros(self) -> "NDArray[np.float32]":
        """Allocate an empty field.

        Returns:
            A zero float32 volume of :attr:`shape`.
        """
        return np.zeros(self.shape, dtype=np.float32)

    def _box(
        self, lo_um: "NDArray[np.float64]", hi_um: "NDArray[np.float64]"
    ) -> "tuple[tuple[slice, ...], list[NDArray[np.float64]]] | None":
        """Find the voxels covering the box ``[lo, hi]``.

        Args:
            lo_um: Lower corner ``(z, y, x)`` in µm.
            hi_um: Upper corner in µm.

        Returns:
            Voxel slices of the box and broadcastable µm coordinates of its voxels, or
            ``None`` if the box lies outside the volume.
        """
        lo = np.maximum(np.floor(lo_um / self.voxel).astype(int), 0)
        hi = np.minimum(np.ceil(hi_um / self.voxel).astype(int) + 1, self.shape)
        if np.any(hi <= lo):
            return None
        slices = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        coords = np.ogrid[slices]
        return slices, [c * v for c, v in zip(coords, self.voxel, strict=True)]

    def _occupancy(self, signed_inside_um: "NDArray[np.float64]") -> "NDArray[np.float32]":
        """Convert a signed distance to the surface into a partial-volume occupancy.

        Args:
            signed_inside_um: Distance in µm, positive inside the shape.

        Returns:
            Occupancy in ``[0, 1]`` (1/2 on the surface).
        """
        return np.clip(signed_inside_um / self.pv_width + _HALF, 0.0, 1.0).astype(np.float32)

    def capsule(
        self, a: "NDArray[np.float64]", b: "NDArray[np.float64]", radius: float
    ) -> "Shape | None":
        """Render a capsule (segment ``a``-``b`` swept by a sphere of ``radius``).

        Args:
            a: First end ``(z, y, x)`` in µm.
            b: Second end in µm.
            radius: Radius in µm.

        Returns:
            The rendered shape, or ``None`` if it lies outside the volume.
        """
        margin = radius + self.pv_width
        box = self._box(np.minimum(a, b) - margin, np.maximum(a, b) + margin)
        if box is None:
            return None
        slices, (zz, yy, xx) = box
        ab = b - a
        rel = (zz - a[0], yy - a[1], xx - a[2])
        t = np.clip((rel[0] * ab[0] + rel[1] * ab[1] + rel[2] * ab[2]) / max(ab @ ab, _EPS), 0, 1)
        dist2 = sum((r - t * c) ** 2 for r, c in zip(rel, ab, strict=True))
        return slices, self._occupancy(radius - np.sqrt(dist2))

    def disc(
        self,
        centre: "NDArray[np.float64]",
        normal: "NDArray[np.float64]",
        thickness: float,
        radius: float,
    ) -> "Shape | None":
        """Render a flat disc (plate).

        Args:
            centre: Centre ``(z, y, x)`` in µm.
            normal: Unit normal of the disc.
            thickness: Thickness in µm.
            radius: Radius in µm.

        Returns:
            The rendered shape, or ``None`` if it lies outside the volume.
        """
        margin = radius + thickness + self.pv_width
        box = self._box(centre - margin, centre + margin)
        if box is None:
            return None
        slices, (zz, yy, xx) = box
        rel = (zz - centre[0], yy - centre[1], xx - centre[2])
        axial = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2]
        radial = np.sqrt(np.maximum(sum(r * r for r in rel) - axial**2, 0.0))
        occ = np.minimum(
            self._occupancy(thickness / 2 - np.abs(axial)), self._occupancy(radius - radial)
        )
        return slices, occ

    def sphere(self, centre: "NDArray[np.float64]", radius: float) -> "Shape | None":
        """Render a ball.

        Args:
            centre: Centre ``(z, y, x)`` in µm.
            radius: Radius in µm.

        Returns:
            The rendered shape, or ``None`` if it lies outside the volume.
        """
        return self.capsule(centre, centre, radius)


def _unit(v: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """Normalise a vector.

    Args:
        v: Non-zero vector.

    Returns:
        ``v / |v|``.
    """
    return v / np.linalg.norm(v)


def _in_plane_offsets(
    normal: "NDArray[np.float64]", radius: float, n: int, rng: np.random.Generator
) -> "NDArray[np.float64]":
    """Draw random points of a disc orthogonal to ``normal`` (centred at 0).

    Args:
        normal: Unit normal of the disc.
        radius: Radius of the disc.
        n: Number of points.
        rng: Random generator.

    Returns:
        ``(n, 3)`` points, uniform over the disc.
    """
    raw = rng.normal(size=(n, _DIMS))
    in_plane = raw - np.outer(raw @ normal, normal)
    in_plane /= np.linalg.norm(in_plane, axis=1, keepdims=True)
    return in_plane * (radius * np.sqrt(rng.uniform(size=(n, 1))))


def _draw_bone(canvas: _Canvas, settings: "PhantomSettings", rng: np.random.Generator) -> None:
    """Draw the cortex walls and plates, then rods bridging two different bone surfaces.

    Real trabecular rods connect plates (or the cortex); a rod floating alone
    would be indistinguishable from a dim vessel, which is not what the data shows.

    Args:
        canvas: Canvas to draw on.
        settings: Phantom settings.
        rng: Random generator.
    """
    extent = np.asarray(settings.shape_zyx) * canvas.voxel
    trab = settings.trabeculae
    # (centre, normal, usable radius) of every plate-like structure
    surfaces: list[tuple[NDArray[np.float64], NDArray[np.float64], float]] = []
    for i in range(settings.cortex.count):
        # Walls near the x edges, slightly tilted, spanning the whole volume.
        x = extent[2] * (_CORTEX_X if i % 2 == 0 else 1 - _CORTEX_X)
        centre = np.array([extent[0] / 2, extent[1] / 2, x])
        normal = _unit(np.array([0.0, rng.normal(0, _CORTEX_TILT), 1.0]))
        thickness = rng.uniform(*settings.cortex.thickness_um)
        shape = canvas.disc(centre, normal, thickness, float(extent.max()))
        if shape:
            _paint(canvas.bone, *shape)
        surfaces.append(
            (centre + normal * thickness / 2 * (1 if i % 2 else -1), normal, extent[1] / 2)
        )
    for _ in range(trab.plates):
        centre = rng.uniform(0, 1, _DIMS) * extent
        normal = _unit(rng.normal(size=_DIMS))
        radius = rng.uniform(*trab.plate_radius_um)
        shape = canvas.disc(centre, normal, rng.uniform(*trab.plate_thickness_um), radius)
        if shape:
            _paint(canvas.bone, *shape)
        surfaces.append((centre, normal, radius))

    points = [
        centre + _in_plane_offsets(normal, radius, _ANCHORS_PER_SURFACE, rng)
        for centre, normal, radius in surfaces
    ]
    inside = [np.all((p >= 0) & (p <= extent), axis=1) for p in points]
    anchors_arr = np.concatenate([p[i] for p, i in zip(points, inside, strict=True)])
    owner_arr = np.concatenate([np.full(int(i.sum()), k) for k, i in enumerate(inside)])
    low, high = trab.rod_length_um
    for _ in range(trab.rods):
        i = rng.integers(len(anchors_arr))
        dist = np.linalg.norm(anchors_arr - anchors_arr[i], axis=1)
        ok = np.flatnonzero((owner_arr != owner_arr[i]) & (dist >= low) & (dist <= high))
        if ok.size == 0:
            continue
        j = rng.choice(ok)
        shape = canvas.capsule(anchors_arr[i], anchors_arr[j], rng.uniform(*trab.rod_radius_um))
        if shape:
            _paint(canvas.bone, *shape)


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
    side = _unit(side - np.dot(side, direction) * direction)
    angle = np.radians(angle_deg)
    return _unit(np.cos(angle) * direction + np.sin(angle) * side)


def _draw_vessels(
    canvas: _Canvas, settings: "PhantomSettings", rng: np.random.Generator
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
        direction = _unit(rng.normal(size=_DIMS))
        walks = [_Walk(point, direction, radius, 0, record.new_branch(-1))]
        while walks:
            walk = walks.pop()
            point, direction = walk.point, walk.direction
            for _ in range(rng.integers(ves.segments[0], ves.segments[1] + 1)):
                direction = _unit(direction + rng.normal(0, turn, _DIMS))
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
                    _paint(canvas.gap, slices, occ)
                else:
                    clotted = draw < ves.gap_probability + ves.clot_probability
                    record.state.append(SegmentState.CLOTTED if clotted else SegmentState.FILLED)
                    fill = rng.uniform(*ves.clot_fill) if clotted else 1.0
                    _paint(canvas.signal, slices, occ * (fill * level))
                    _paint(canvas.vessel, slices, occ)
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


def _draw_bursts(canvas: _Canvas, settings: "PhantomSettings", rng: np.random.Generator) -> None:
    """Draw the bursts: spheres of diffuse haze filled with dense speckles.

    Args:
        canvas: Canvas to draw on.
        settings: Phantom settings.
        rng: Random generator.
    """
    bur, spread = settings.bursts, settings.intensity.vessel_spread
    extent = np.asarray(settings.shape_zyx) * canvas.voxel
    for _ in range(bur.count):
        radius = rng.uniform(*bur.radius_um)
        centre = rng.uniform(*_BURST_CENTRE_RANGE, _DIMS) * extent
        level = rng.uniform(1 - spread, 1 + spread)
        shape = canvas.sphere(centre, radius)
        if shape is None:
            continue
        _paint(canvas.burst, *shape)
        _paint(canvas.signal, shape[0], shape[1] * (bur.haze_fill * level))
        mean_speckle = np.mean(bur.speckle_radius_um)
        n_speckles = int(bur.speckle_fill * (radius / mean_speckle) ** _DIMS)
        for _ in range(n_speckles):
            offset = _unit(rng.normal(size=_DIMS)) * radius * rng.uniform() ** (1 / _DIMS)
            speckle = canvas.sphere(centre + offset, rng.uniform(*bur.speckle_radius_um))
            if speckle:
                _paint(canvas.signal, speckle[0], speckle[1] * level)


def _blur(volume: "NDArray[np.float32]", sigma_voxels: "NDArray[np.float64]") -> None:
    """Blur a volume in place with a separable 3D Gaussian (OpenCV in-plane, then along z).

    Args:
        volume: ``(Z, Y, X)`` float32 volume, modified in place.
        sigma_voxels: Sigmas ``(z, y, x)`` in voxels.
    """
    sz, sy, sx = sigma_voxels
    for z in range(volume.shape[0]):
        volume[z] = cv2.GaussianBlur(volume[z], (0, 0), sigmaX=sx, sigmaY=sy)
    if sz > 0:
        flat = volume.reshape(volume.shape[0], -1)
        kernel = cv2.getGaussianKernel(2 * int(np.ceil(_BLUR_TRUNCATE_SIGMAS * sz)) + 1, sz)
        flat[:] = cv2.sepFilter2D(flat, -1, np.ones((1, 1)), kernel, borderType=cv2.BORDER_REFLECT)


def render_phantom(settings: "PhantomSettings", voxel_um: tuple[float, float, float]) -> Phantom:
    """Render the phantom described by ``settings`` at the given voxel size.

    Args:
        settings: Phantom settings.
        voxel_um: Voxel size ``(z, y, x)`` in µm (the pybones acquisition voxel size).

    Returns:
        The rendered image and its ground truth.
    """
    rng = np.random.default_rng(settings.seed)
    canvas = _Canvas(settings.shape_zyx, voxel_um)
    _draw_bone(canvas, settings, rng)
    network = _draw_vessels(canvas, settings, rng)
    _draw_bursts(canvas, settings, rng)

    ints = settings.intensity
    bg = np.float32(ints.background)
    image = np.maximum(
        bg + np.float32(ints.bone - ints.background) * canvas.bone,
        bg + np.float32(ints.vessel - ints.background) * canvas.signal,
    )
    _blur(image, ints.psf_sigma_um / canvas.voxel)
    noise_scale = np.sqrt(np.maximum(image, 1.0) / bg)  # photon-noise-like growth with intensity
    image += rng.normal(0.0, ints.noise_std, image.shape).astype(np.float32) * noise_scale
    image_u16 = np.clip(image, 0, UINT16_MAX).astype(np.uint16)

    truth = np.full(settings.shape_zyx, Label.BACKGROUND, dtype=np.uint8)
    truth[canvas.bone >= _HALF] = Label.BONE
    truth[canvas.vessel >= _HALF] = Label.VESSEL
    truth[canvas.burst >= _HALF] = Label.BURST
    truth[(canvas.gap >= _HALF) & (truth != Label.VESSEL)] = settings.excluded_value
    vessel_truth = truth == Label.VESSEL
    return Phantom(
        image=image_u16,
        truth=truth,
        vessel_radius_um=np.where(vessel_truth, canvas.radius, 0).astype(np.float32),
        clotted=canvas.clot & vessel_truth,
        network=network,
    )
