"""Per-slice classification into background / bone / vessel / burst / ambiguous.

Physical model (barium-sulfate perfused µCT):

* background / soft tissue: darkest class;
* bone (trabeculae and cortex): intermediate intensity;
* vessels: brightest (radio-opaque contrast agent). Thin or partially filled
  vessels look dimmer (partial-volume effect, clots), so vessel rims are grown
  from bright cores;
* burst vessels: contrast agent that leaked out of ruptured vessels (e.g.
  freezing damage), forming dense bright clusters. Detected as regions where
  the *local density* of bright pixels is high;
* ambiguous: small bone-intensity blobs without a bright core. In 2D they can be
  trabecular cross-sections or dim vessels; the 3D refinement decides.

The output is a single ``uint8`` label image (see :class:`Label`), so classes
are disjoint by construction.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from pybones.imaging import morphology as morph
from pybones.imaging.threshold import accumulate_histogram, class_means, two_level_otsu
from pybones.segmentation.labels import Label

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

    from pybones.segmentation.scales import PixelParams


class Thresholds(BaseModel):
    """Intensity thresholds shared by every slice of a stack.

    Attributes:
        bone: Pixels ``>= bone`` are foreground (bone or vessel).
        vessel: Pixels ``>= vessel`` are bright vessel cores.
        background_mean: Mean intensity of the background class (soft tissue level,
            reference of the vessel-rim half-maximum).
        bone_mean: Mean intensity of the bone class.
        vessel_mean: Mean intensity of the vessel class.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    bone: int
    vessel: int
    background_mean: float
    bone_mean: float
    vessel_mean: float


def smooth(img: "NDArray[np.uint16]", sigma: tuple[float, float]) -> "NDArray[np.uint16]":
    """Gaussian denoising that keeps the ``uint16`` dtype (no float copy).

    Args:
        img: Input slice.
        sigma: Gaussian sigmas ``(sy, sx)`` in pixels; zeros return ``img`` unchanged.

    Returns:
        The smoothed slice.
    """
    sy, sx = sigma
    if sy == 0 and sx == 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REFLECT)
    return cast("NDArray[np.uint16]", blurred)


def estimate_thresholds(
    slices: "Iterable[NDArray[np.uint16]]", params: "PixelParams", n_bins: int
) -> Thresholds:
    """Estimate the global thresholds from the (smoothed) histogram of a set of slices.

    Using one set of thresholds for the whole stack keeps the classification
    consistent along z, unlike per-slice thresholds.

    Args:
        slices: Slices to sample (typically every ``histogram_stride``-th one).
        params: Segmentation parameters in pixel units.
        n_bins: Resolution of the multi-Otsu search.

    Returns:
        The estimated thresholds.
    """
    hist = accumulate_histogram(smooth(s, params.smoothing_sigma) for s in slices)
    bone, vessel = two_level_otsu(hist, n_bins)
    bg_mean, bone_mean, vessel_mean = class_means(hist, (bone, vessel))
    return Thresholds(
        bone=bone,
        vessel=vessel,
        background_mean=bg_mean,
        bone_mean=bone_mean,
        vessel_mean=vessel_mean,
    )


@dataclass(frozen=True, slots=True)
class SliceMasks:
    """Per-slice masks that the final labelling (:func:`finalize_slice`) needs.

    Attributes:
        foreground: Pixels at or above the bone threshold (vessel halo excluded).
        bright: Bright vessel cores.
        vessel: Vessels (cores + rims + small blobs holding a core), noise removed.
        tubularity: Quantised tube-vs-plate ratio (see
            :func:`pybones.segmentation.tubularity.quantize_tubularity`), if computed.
    """

    foreground: "NDArray[np.bool_]"
    bright: "NDArray[np.bool_]"
    vessel: "NDArray[np.bool_]"
    tubularity: "NDArray[np.uint8] | None"


@dataclass(frozen=True, slots=True)
class SliceFeatures:
    """Everything computed on a slice before the (3D) burst detection.

    Attributes:
        masks: Masks kept until the slice is finalised.
        burst_density: Fraction of bright pixels in the burst window around each cell
            of the coarse density grid (see :func:`coarse_density`); averaged over
            slices for the 3D burst density.
    """

    masks: SliceMasks
    burst_density: "NDArray[np.float32]"


@dataclass(frozen=True, slots=True)
class ClassifiedSlice:
    """Output of the per-slice classification.

    Attributes:
        labels: :class:`Label` image (``AMBIGUOUS`` included).
        tubularity: Quantised tubularity map, if computed.
    """

    labels: "NDArray[np.uint8]"
    tubularity: "NDArray[np.uint8] | None"


def extract_features(
    smoothed: "Sequence[NDArray[np.uint16]]",
    centre: int,
    thresholds: Thresholds,
    params: "PixelParams",
    tubularity: "NDArray[np.uint8] | None" = None,
) -> SliceFeatures:
    """Compute the masks of a slice that do not depend on the burst detection.

    Args:
        smoothed: Smoothed slices around the current one (see :func:`vessel_edges`).
        centre: Index of the current slice in ``smoothed``.
        thresholds: Global thresholds (see :func:`estimate_thresholds`).
        params: Segmentation parameters in pixel units.
        tubularity: Quantised tubularity map of the slice, carried along if given.

    Returns:
        The slice features.
    """
    conn = params.connectivity
    sm = smoothed[centre]
    foreground = sm >= thresholds.bone
    bright = sm >= thresholds.vessel

    rim, halo = vessel_edges(smoothed, centre, thresholds, params)
    vessel = morph.area_open(bright | rim, params.vessel_min_area, conn)
    # The halo is a vessel / soft-tissue mix below half-maximum: background, not bone.
    foreground &= ~halo

    masks = SliceMasks(foreground, bright, vessel, tubularity)
    return SliceFeatures(masks, coarse_density(bright, params))


def coarse_density(bright: "NDArray[np.bool_]", params: "PixelParams") -> "NDArray[np.float32]":
    """Measure the bright-pixel density in the burst window, on a coarse grid of blocks.

    The density is a mean over a window an order of magnitude larger than a grid
    block, so storing it per block loses nothing that matters while using 16x
    less memory than per pixel (it is buffered over the whole 3D window).

    Args:
        bright: Bright vessel-core mask.
        params: Pixel-unit parameters (grid block and window size in blocks).

    Returns:
        ``float32`` density in ``[0, 1]``, one value per grid block.
    """
    gy, gx = params.burst_grid
    h, w = bright.shape
    ny, nx = -(-h // gy), -(-w // gx)
    padded = np.zeros((ny * gy, nx * gx), np.float32)
    padded[:h, :w] = bright
    blocks = padded.reshape(ny, gy, nx, gx).mean(axis=(1, 3), dtype=np.float32)
    ky, kx = params.burst_window_blocks
    density = cv2.boxFilter(blocks, -1, (kx, ky), normalize=True, borderType=cv2.BORDER_CONSTANT)
    return cast("NDArray[np.float32]", density)


def upsample_grid(
    mask: "NDArray[np.bool_]", params: "PixelParams", shape: tuple[int, int]
) -> "NDArray[np.bool_]":
    """Expand a coarse-grid mask back to pixels (each block to its pixels).

    Args:
        mask: Mask on the coarse density grid.
        params: Pixel-unit parameters (grid block size).
        shape: Pixel shape of the slice (the last blocks may overhang it).

    Returns:
        Pixel mask of shape ``shape``.
    """
    gy, gx = params.burst_grid
    return np.repeat(np.repeat(mask, gy, axis=0), gx, axis=1)[: shape[0], : shape[1]]


def ambiguous_candidates(masks: SliceMasks, params: "PixelParams") -> "NDArray[np.bool_]":
    """Find the pixels that may end up ``AMBIGUOUS``: small bone-intensity blobs outside vessels.

    Mirrors the bone / ambiguous split of :func:`finalize_slice` before bursts are
    removed, so the (costly) tubularity is only computed where it can be used.

    Args:
        masks: Masks of the slice.
        params: Segmentation parameters in pixel units.

    Returns:
        Candidate pixels.
    """
    conn = params.connectivity
    dim = morph.area_open(masks.foreground & ~masks.vessel, params.bone_min_area, conn)
    return dim & ~morph.area_open(dim, params.small_blob_max_area, conn)


def vessel_edges(
    smoothed: "Sequence[NDArray[np.uint16]]",
    centre: int,
    thresholds: Thresholds,
    params: "PixelParams",
) -> "tuple[NDArray[np.bool_], NDArray[np.bool_]]":
    """Find the partial-volume edge of the bright vessel cores, at a local half-maximum, in 3D.

    A voxel holding a fraction ``f`` of contrast agent, the rest being the
    surrounding material, reads ``surround + f * (core - surround)``:

    * ``core``: the brightest core within the rim radius (grey dilation), i.e. the
      intensity of a fully filled voxel of *that* vessel, whatever its filling;
    * ``surround``: the material just beyond the rim, from the darkest intensity
      there (grey erosion). Below the bone threshold it is soft tissue (the vessel's
      own blur lifts that minimum, so the soft-tissue mean is used); otherwise bone,
      so a vessel running along or inside bone is measured against bone and cannot
      eat into it.

    Pixels near a core but below the half-maximum form the halo. Where the
    surround is soft tissue, the halo is a vessel / soft-tissue mix whose
    intensity may reach bone level: it must not be mistaken for bone.

    Cores and surround are sought in a 3D ball, so the z blur of a vessel lying
    in a neighbouring slice is recognised as its halo too. Offsets missing from
    ``smoothed`` (single-slice use) are skipped, which gives the 2D behaviour.

    Args:
        smoothed: Smoothed slices around the current one (consecutive z).
        centre: Index of the current slice in ``smoothed``.
        thresholds: Global thresholds and class means.
        params: Pixel-unit parameters (rim balls, occupancy fraction).

    Returns:
        ``(rim, halo)``: pixels at or above the local half-maximum, and pixels near a
        core, below it, with a soft-tissue surround.
    """
    sm = smoothed[centre]
    available = range(-centre, len(smoothed) - centre)
    core_level = np.zeros(sm.shape, np.uint16)
    for dz, radius in params.vessel_core_ball:
        if dz in available:
            other = smoothed[centre + dz]
            cores = np.where(other >= thresholds.vessel, other, 0).astype(np.uint16)
            _ = np.maximum(core_level, cv2.dilate(cores, morph.ellipse(radius)), out=core_level)
    darkest = sm.copy()
    for dz, radius in params.vessel_surround_ball:
        if dz in available:
            _ = np.minimum(
                darkest, cv2.erode(smoothed[centre + dz], morph.ellipse(radius)), out=darkest
            )
    core_level = core_level.astype(np.float32)
    soft_tissue = darkest < thresholds.bone
    surround = np.where(
        soft_tissue,
        np.float32(thresholds.background_mean),
        np.minimum(darkest, thresholds.bone_mean).astype(np.float32),
    )
    threshold = surround + params.vessel_rim_occupancy * (core_level - surround)
    near = core_level > 0
    rim = near & (sm >= threshold)
    return rim, near & ~rim & soft_tissue


def burst_region(
    core: "NDArray[np.bool_]", bright: "NDArray[np.bool_]", params: "PixelParams"
) -> "NDArray[np.bool_]":
    """Turn a raw burst core (dense bright area) into the final burst region.

    Args:
        core: Pixels whose bright-pixel density reaches the burst density.
        bright: Bright vessel-core mask of the slice.
        params: Segmentation parameters in pixel units.

    Returns:
        Burst mask: the cores grown, at a fine scale, wherever the bright-pixel
        density stays above a fraction of the core density (a half-maximum on the
        density, like the vessel edges), enclosed holes filled.
    """
    conn = params.connectivity
    core = morph.area_open(core, params.burst_min_area, conn)
    if not core.any():
        return core
    sy, sx = params.burst_boundary_sigma
    fine = cv2.GaussianBlur(
        bright.astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_CONSTANT
    )
    dense = fine >= params.burst_density * params.burst_boundary_fraction
    return morph.fill_holes(morph.reconstruct(core, dense | core, conn))


def finalize_slice(
    masks: SliceMasks, burst_core: "NDArray[np.bool_]", params: "PixelParams"
) -> "NDArray[np.uint8]":
    """Build the :class:`Label` image of a slice (``AMBIGUOUS`` included).

    Args:
        masks: Masks of the slice (see :func:`extract_features`).
        burst_core: Pixel mask of the dense burst cores of this slice (see
            :func:`pybones.segmentation.bursts.iter_burst_cores`).
        params: Segmentation parameters in pixel units.

    Returns:
        ``uint8`` label image.
    """
    conn = params.connectivity
    burst = burst_region(burst_core, masks.bright, params)
    # Remaining foreground: large blobs are bone, small ones are undecidable in 2D.
    remaining = masks.foreground & ~masks.vessel & ~burst
    dim = morph.area_open(remaining, params.bone_min_area, conn)
    bone = morph.area_open(dim, params.small_blob_max_area, conn)

    labels = np.zeros(masks.foreground.shape, dtype=np.uint8)
    labels[dim] = Label.AMBIGUOUS
    labels[bone] = Label.BONE
    labels[masks.vessel] = Label.VESSEL
    labels[burst] = Label.BURST
    return labels
