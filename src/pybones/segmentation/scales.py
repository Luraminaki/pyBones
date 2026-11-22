"""Conversion of physical segmentation parameters (µm) into pixel / voxel units.

The configuration states every size in µm, µm² or µm³ so that it keeps its
anatomical meaning whatever the scan resolution: change
``acquisition.voxel_size_um`` and every kernel, window and area threshold
follows. Anisotropic voxels are supported: in-plane kernels become ellipses and
windows rectangles; z extents use the slice spacing.

The algorithms receive a :class:`PixelParams` computed once per run.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pybones.segmentation.geometry import VoxelGeometry
from pybones.segmentation.tubularity import TubularityParams, scale_kernels

if TYPE_CHECKING:
    from pybones.config.schema import PyBonesConfig
    from pybones.imaging.morphology import Radius
    from pybones.segmentation.geometry import Ball


@dataclass(frozen=True, slots=True)
class PixelParams:
    """Segmentation and refinement parameters in pixel / voxel units.

    In-plane radii, sigmas, grids and windows are ``(y, x)`` pairs; z windows are odd
    slice counts; areas are pixel counts and volumes voxel counts.

    Attributes:
        connectivity: In-plane pixel connectivity (4 or 8).
        smoothing_sigma: Gaussian denoising sigma.
        small_blob_max_area: Bone-intensity blobs up to this area are ambiguous in 2D.
        vessel_rim_occupancy: Contrast-agent fraction of a vessel edge (0.5 = half-maximum).
        vessel_core_ball: Ball in which a voxel looks for the vessel core it borders.
        vessel_surround_ball: Ball in which a voxel looks for the material around a vessel.
        vessel_bone_side: Whether a vessel's edge is measured against bone on the bone side of
            a contact (see :func:`pybones.segmentation.classifier.vessel_edges`).
        vessel_bone_barrier: Radius of the disc bone must hold to count as the bone side.
        vessel_min_area: Smaller 2D vessel blobs are noise.
        bone_min_area: Smaller 2D bone blobs are noise.
        burst_grid: Block size of the coarse bright-density grid.
        burst_window_blocks: In-plane side of the density box, in grid blocks.
        burst_window_z: z extent of the density box (1 = 2D only).
        burst_density: Minimum bright-pixel density of a burst core.
        burst_min_area: Minimum area of a burst core.
        burst_boundary_sigma: In-plane sigma of the fine density that places the burst
            boundary, in blocks of the coarse grid.
        burst_boundary_fraction: Fraction of the core density where the boundary lies.
        z_window: Slices of the z majority vote (1 = off).
        vessel_contact_ratio: Ambiguous component becomes vessel if its vessel contacts
            exceed this ratio times its bone contacts.
        min_tubularity: Ambiguous components flatter than this cannot be vessels.
        min_isolated_vessel_voxels: Smaller isolated ambiguous components are noise.
        min_vessel_voxels: Smaller 3D vessels are noise.
        min_bone_voxels: Smaller 3D bone pieces are noise.
    """

    connectivity: int
    smoothing_sigma: tuple[float, float]
    small_blob_max_area: int
    vessel_rim_occupancy: float
    vessel_core_ball: "Ball"
    vessel_surround_ball: "Ball"
    vessel_bone_side: bool
    vessel_bone_barrier: "Radius"
    vessel_min_area: int
    bone_min_area: int
    burst_grid: tuple[int, int]
    burst_window_blocks: tuple[int, int]
    burst_window_z: int
    burst_density: float
    burst_min_area: int
    burst_boundary_sigma: tuple[float, float]
    burst_boundary_fraction: float
    z_window: int
    vessel_contact_ratio: float
    min_tubularity: float
    min_isolated_vessel_voxels: int
    min_vessel_voxels: int
    min_bone_voxels: int

    @property
    def edge_half_z(self) -> int:
        """Slices needed on each side of a slice to find its vessel edges in 3D."""
        return max(abs(dz) for dz, _ in self.vessel_core_ball + self.vessel_surround_ball)


def voxel_geometry(config: "PyBonesConfig") -> VoxelGeometry:
    """Build the voxel geometry of the configured acquisition.

    Args:
        config: Effective configuration.

    Returns:
        Voxel size of the scan.
    """
    z, y, x = config.acquisition.voxel_size_um
    return VoxelGeometry(z=z, y=y, x=x)


def resolve_pixel_params(config: "PyBonesConfig") -> PixelParams:
    """Convert the physical parameters of ``config`` to pixel / voxel units.

    Args:
        config: Effective configuration.

    Returns:
        The parameters the segmentation and refinement algorithms work with.
    """
    geo = voxel_geometry(config)
    seg, burst, ref = config.segmentation, config.segmentation.burst, config.refinement
    grid = geo.block_yx(burst.density_grid_um)
    boundary_sy, boundary_sx = geo.sigma_yx(burst.boundary_sigma_um)
    params = PixelParams(
        connectivity=seg.connectivity,
        smoothing_sigma=geo.sigma_yx(seg.smoothing_sigma_um),
        small_blob_max_area=geo.area_px(seg.small_blob_max_area_um2),
        vessel_rim_occupancy=seg.vessel.rim_occupancy,
        vessel_core_ball=geo.ball(seg.vessel.rim_um),
        # The surround is sought one voxel beyond the rim.
        vessel_surround_ball=geo.ball(seg.vessel.rim_um + max(geo.y, geo.x)),
        vessel_bone_side=seg.vessel.bone_side.enabled,
        vessel_bone_barrier=geo.radius_yx(seg.vessel.bone_side.barrier_um),
        vessel_min_area=geo.area_px(seg.vessel.min_area_um2),
        bone_min_area=geo.area_px(seg.bone.min_area_um2),
        burst_grid=grid,
        burst_window_blocks=(
            max(1, round(burst.window_xy_um / (grid[0] * geo.y))),
            max(1, round(burst.window_xy_um / (grid[1] * geo.x))),
        ),
        burst_window_z=geo.window_z(burst.window_z_um),
        burst_density=burst.density,
        burst_min_area=geo.area_px(burst.min_area_um2),
        burst_boundary_sigma=(boundary_sy / grid[0], boundary_sx / grid[1]),
        burst_boundary_fraction=burst.boundary_fraction,
        z_window=geo.window_z(config.z_smoothing.window_um),
        vessel_contact_ratio=ref.vessel_contact_ratio,
        min_tubularity=ref.min_tubularity,
        min_isolated_vessel_voxels=geo.volume_voxels(ref.min_isolated_vessel_um3),
        min_vessel_voxels=geo.volume_voxels(ref.min_vessel_um3),
        min_bone_voxels=geo.volume_voxels(ref.min_bone_um3),
    )
    return params


def resolve_tubularity(config: "PyBonesConfig") -> TubularityParams:
    """Build the tubularity settings of this scan (derivative kernels from the voxel size).

    Args:
        config: Effective configuration.

    Returns:
        Kernels of every configured scale.
    """
    geo = voxel_geometry(config)
    return TubularityParams(
        scales=tuple(scale_kernels(s, geo) for s in config.tubularity.scales_um)
    )
