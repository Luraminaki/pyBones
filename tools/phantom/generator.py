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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.segmentation.labels import Label
from pybones.utils.constants import UINT16_MAX
from tools.phantom.canvas import Canvas, in_plane_offsets, paint, unit
from tools.phantom.vessels import draw_vessels

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tools.phantom.settings import PhantomSettings
    from tools.phantom.vessels import VesselNetwork

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


def _draw_bone(canvas: Canvas, settings: "PhantomSettings", rng: np.random.Generator) -> None:
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
        normal = unit(np.array([0.0, rng.normal(0, _CORTEX_TILT), 1.0]))
        thickness = rng.uniform(*settings.cortex.thickness_um)
        shape = canvas.disc(centre, normal, thickness, float(extent.max()))
        if shape:
            paint(canvas.bone, *shape)
        surfaces.append(
            (centre + normal * thickness / 2 * (1 if i % 2 else -1), normal, extent[1] / 2)
        )
    for _ in range(trab.plates):
        centre = rng.uniform(0, 1, _DIMS) * extent
        normal = unit(rng.normal(size=_DIMS))
        radius = rng.uniform(*trab.plate_radius_um)
        shape = canvas.disc(centre, normal, rng.uniform(*trab.plate_thickness_um), radius)
        if shape:
            paint(canvas.bone, *shape)
        surfaces.append((centre, normal, radius))

    points = [
        centre + in_plane_offsets(normal, radius, _ANCHORS_PER_SURFACE, rng)
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
            paint(canvas.bone, *shape)


def _draw_bursts(canvas: Canvas, settings: "PhantomSettings", rng: np.random.Generator) -> None:
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
        paint(canvas.burst, *shape)
        paint(canvas.signal, shape[0], shape[1] * (bur.haze_fill * level))
        mean_speckle = np.mean(bur.speckle_radius_um)
        n_speckles = int(bur.speckle_fill * (radius / mean_speckle) ** _DIMS)
        for _ in range(n_speckles):
            offset = unit(rng.normal(size=_DIMS)) * radius * rng.uniform() ** (1 / _DIMS)
            speckle = canvas.sphere(centre + offset, rng.uniform(*bur.speckle_radius_um))
            if speckle:
                paint(canvas.signal, speckle[0], speckle[1] * level)


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
    canvas = Canvas(settings.shape_zyx, voxel_um)
    _draw_bone(canvas, settings, rng)
    network = draw_vessels(canvas, settings, rng)
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
