"""Drawing shapes into the phantom's volumes: partial-volume occupancies, box by box.

Each shape is rendered as an occupancy with a linear fall-off across one voxel at its
surface, computed only inside its own bounding box and combined by maximum.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HALF = 0.5
_DIMS = 3
_EPS = 1e-12

#: A shape rendered into a bounding box: voxel slices of the box and occupancy inside it.
Shape = tuple[tuple[slice, ...], "NDArray[np.float32]"]


def paint(field: "NDArray[np.float32]", slices: tuple[slice, ...], values: "NDArray") -> None:
    """Compose a shape into an occupancy field by maximum (in place).

    Args:
        field: Occupancy field.
        slices: Bounding box of the shape in ``field``.
        values: Occupancy of the shape inside the box.
    """
    _ = np.maximum(field[slices], values, out=field[slices])


class Canvas:
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


def unit(v: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """Normalise a vector.

    Args:
        v: Non-zero vector.

    Returns:
        ``v / |v|``.
    """
    return v / np.linalg.norm(v)


def in_plane_offsets(
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
