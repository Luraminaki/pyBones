"""Voxel geometry: conversions of physical sizes (µm, µm², µm³) into pixel / voxel units."""

from dataclasses import dataclass

#: A 3D ball as ``((dz, (ry, rx)), ...)``: in-plane ellipse radii per slice offset.
Ball = tuple[tuple[int, tuple[int, int]], ...]


def _odd(pixels: float) -> int:
    """Round to the nearest odd integer >= 1 (windows need a centre pixel).

    Args:
        pixels: Size in pixels or slices.

    Returns:
        The odd window size.
    """
    n = max(1, round(pixels))
    return n if n % 2 else n + 1


@dataclass(frozen=True, slots=True)
class VoxelGeometry:
    """Voxel size and µm -> pixel conversions.

    Attributes:
        z: Slice spacing in µm.
        y: Pixel height in µm.
        x: Pixel width in µm.
    """

    z: float
    y: float
    x: float

    def radius_yx(self, um: float) -> tuple[int, int]:
        """Convert a physical radius into in-plane kernel radii.

        Args:
            um: Radius in µm.

        Returns:
            ``(ry, rx)`` in pixels.
        """
        return round(um / self.y), round(um / self.x)

    def sigma_yx(self, um: float) -> tuple[float, float]:
        """Convert a physical Gaussian sigma into in-plane sigmas.

        Args:
            um: Sigma in µm.

        Returns:
            ``(sy, sx)`` in pixels.
        """
        return um / self.y, um / self.x

    def block_yx(self, um: float) -> tuple[int, int]:
        """Convert a physical grid block side into in-plane block sides.

        Args:
            um: Block side in µm.

        Returns:
            ``(gy, gx)`` in pixels, each at least 1.
        """
        return max(1, round(um / self.y)), max(1, round(um / self.x))

    def window_z(self, um: float) -> int:
        """Convert a physical z extent into a window of slices.

        Args:
            um: Extent in µm.

        Returns:
            Odd number of slices (at least 1).
        """
        return _odd(um / self.z)

    def ball(self, um: float) -> Ball:
        """Describe a 3D ball of physical radius ``um`` as per-slice ellipses.

        Args:
            um: Radius in µm.

        Returns:
            ``((dz, (ry, rx)), ...)`` for every slice offset ``dz`` within the
            radius, the in-plane radii shrinking with ``|dz|`` (sphere section).
        """
        reach = int(um // self.z)
        return tuple(
            (dz, self.radius_yx((um**2 - (dz * self.z) ** 2) ** 0.5))
            for dz in range(-reach, reach + 1)
        )

    def area_px(self, um2: float) -> int:
        """Convert a physical area into a pixel count.

        Args:
            um2: Area in µm².

        Returns:
            Pixel count (at least 1).
        """
        return max(1, round(um2 / (self.y * self.x)))

    def volume_voxels(self, um3: float) -> int:
        """Convert a physical volume into a voxel count.

        Args:
            um3: Volume in µm³.

        Returns:
            Voxel count (at least 1).
        """
        return max(1, round(um3 / (self.z * self.y * self.x)))
