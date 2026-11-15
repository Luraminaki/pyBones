"""Image figures of the report and README screenshots.

Panels go to the report's figure directory; composed strips go to the README
screenshots. Past versions are shown from their extracts, today's version from
its runs (see :class:`tools.report.context.Context`).
"""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.imaging import morphology as morph
from pybones.io.slices import SliceStack
from pybones.segmentation.classifier import (
    Thresholds,
    ambiguous_candidates,
    extract_features,
    smooth,
)
from pybones.segmentation.labels import Label
from pybones.segmentation.scales import resolve_pixel_params, resolve_tubularity
from pybones.segmentation.tubularity import slice_tubularity
from pybones.utils.constants import UINT8_MAX
from pybones.visualization.overlay import overlay, to_uint8
from tools.phantom.storage import load_phantom
from tools.report.drawing import add_scale_bar, compose, downscale, to_bgr
from tools.report.runs import META_FILE, PHANTOM_RUN

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tools.report.context import Context
    from tools.report.settings import Comparison

#: Panel name of the raw crop (``<figure>_raw``).
RAW_PANEL = "raw"
#: README screenshots (stems; the extension comes from the settings).
SCREENSHOT_SEGMENTATION = "01-segmentation"
SCREENSHOT_NETWORK = "02-vessel-network"
SCREENSHOT_EVOLUTION = "03-evolution"
SCREENSHOT_PHANTOM = "04-phantom"

_DEPTH_COLORMAP = cv2.COLORMAP_TURBO
_TUBE_COLORMAP = cv2.COLORMAP_JET
#: Brightness of the bone context in the vessel projection.
_CONTEXT_GREY = 0.35


def crop_panels(ctx: "Context", comparison: "Comparison") -> "list[NDArray[np.uint8]]":
    """Render the raw crop followed by the overlay of each version of a comparison.

    Args:
        ctx: Figure context.
        comparison: Crop name and versions.

    Returns:
        BGR panels: raw crop, then one per version key.
    """
    crop = ctx.settings.figures.crops[comparison.crop]
    img = ctx.raw_slice(crop.z)
    render = ctx.config.render
    panels = [overlay(img, np.zeros(img.shape, np.uint8), render)[crop.window]]
    panels += [
        overlay(img, ctx.slice_labels(key, crop.z), render)[crop.window] for key in comparison.keys
    ]
    return panels


def overview(ctx: "Context") -> None:
    """Render the full slice: raw with scale bar, 2018 masks, current segmentation.

    Args:
        ctx: Figure context.
    """
    fig = ctx.settings.figures
    z = fig.overview_slice
    img = ctx.raw_slice(z)
    render = ctx.config.render
    pixels_per_um = fig.overview_scale / ctx.config.acquisition.voxel_size_um[2]

    def shown(key: str) -> "NDArray[np.uint8]":
        """Overlay the labels of one version on the slice, downscaled.

        Args:
            key: Version key or legacy key.

        Returns:
            The BGR panel.
        """
        return downscale(overlay(img, ctx.slice_labels(key, z), render), fig.overview_scale)

    raw = downscale(to_bgr(to_uint8(img, render)), fig.overview_scale)
    add_scale_bar(raw, pixels_per_um, fig)
    current = shown(fig.reference_key)
    ctx.save(ctx.figures, "overview_raw", raw)
    ctx.save(ctx.figures, "overview_legacy", shown(fig.legacy_key))
    ctx.save(ctx.figures, "overview_current", current)
    screenshot = current.copy()
    add_scale_bar(screenshot, pixels_per_um, fig)
    ctx.save(ctx.screenshots, SCREENSHOT_SEGMENTATION, screenshot)


def comparisons(ctx: "Context") -> None:
    """Render the before/after crops along the project's history, and the evolution strip.

    Args:
        ctx: Figure context.
    """
    fig = ctx.settings.figures
    for name, comparison in fig.comparisons.items():
        panels = crop_panels(ctx, comparison)
        for label, panel in zip((RAW_PANEL, *comparison.keys), panels, strict=True):
            ctx.save(ctx.figures, f"{name}_{label}", panel)
    evolution = fig.evolution_screenshot
    titles = [ctx.title(t) for t in evolution.titles]
    strip = compose(crop_panels(ctx, evolution), titles, fig)
    ctx.save(ctx.screenshots, SCREENSHOT_EVOLUTION, strip)


def vessel_network(ctx: "Context") -> None:
    """Render a top view of the vessel labels, coloured by depth, over a faint bone projection.

    Args:
        ctx: Figure context.
    """
    labels = ctx.reference_labels()
    depth = np.full(labels.shape[1:], -1, np.int32)
    bone_fraction = np.zeros(labels.shape[1:], np.float32)
    for z in range(labels.shape[0]):
        sl = np.asarray(labels[z])
        first = (sl == Label.VESSEL) & (depth < 0)
        depth[first] = z
        bone_fraction += sl == Label.BONE
    bone_fraction /= labels.shape[0]
    shade = (_CONTEXT_GREY * UINT8_MAX * np.sqrt(bone_fraction)).astype(np.uint8)
    image = to_bgr(shade)
    scaled = (depth.clip(0) * (UINT8_MAX / max(labels.shape[0] - 1, 1))).astype(np.uint8)
    colours = cv2.applyColorMap(scaled, _DEPTH_COLORMAP)
    image[depth >= 0] = colours[depth >= 0]
    fig = ctx.settings.figures
    image = downscale(image, fig.overview_scale)
    add_scale_bar(image, fig.overview_scale / ctx.config.acquisition.voxel_size_um[2], fig)
    ctx.save(ctx.figures, "vessel_network", image)
    ctx.save(ctx.screenshots, SCREENSHOT_NETWORK, image)


def tubularity_map(ctx: "Context") -> None:
    """Render the mean tube-vs-plate ratio of each ambiguous candidate blob (blue plate, red tube).

    Args:
        ctx: Figure context.
    """
    crop = ctx.settings.figures.tubularity_crop
    params = resolve_pixel_params(ctx.config)
    tubes = resolve_tubularity(ctx.config)
    thresholds = Thresholds.model_validate(ctx.reference_json(META_FILE)["thresholds"])
    half = max(tubes.half_z, params.edge_half_z)
    with SliceStack(ctx.repo / ctx.settings.dataset, ctx.config.io) as stack:
        raws = [stack.read(z) for z in range(crop.z - half, crop.z + half + 1)]
    smoothed = [smooth(r, params.smoothing_sigma) for r in raws]
    eh, th = params.edge_half_z, tubes.half_z
    features = extract_features(smoothed[half - eh : half + eh + 1], eh, thresholds, params)
    candidates = ambiguous_candidates(features.masks, params)
    floats = [r.astype(np.float32) for r in raws[half - th : half + th + 1]]
    tub = slice_tubularity(floats, candidates, tubes)
    raw = to_bgr(to_uint8(raws[half], ctx.config.render))
    # The refinement decides on the mean over each (3D) component: show the per-blob mean.
    n, blobs = morph.label(candidates, params.connectivity)
    sums = np.bincount(blobs.ravel(), weights=tub.ravel(), minlength=n)
    sizes = np.bincount(blobs.ravel(), minlength=n)
    blob_mean = (sums / np.maximum(sizes, 1))[blobs]
    colours = cv2.applyColorMap((blob_mean * UINT8_MAX).astype(np.uint8), _TUBE_COLORMAP)
    coloured = raw.copy()
    coloured[candidates] = colours[candidates]
    ctx.save(ctx.figures, "tubularity_raw", raw[crop.window])
    ctx.save(ctx.figures, "tubularity_map", coloured[crop.window])


def phantom_figure(ctx: "Context") -> None:
    """Render a phantom slice, its exact ground truth and the current segmentation.

    Args:
        ctx: Figure context.
    """
    fig = ctx.settings.figures
    crop = fig.phantom_crop
    phantom_settings = ctx.phantom_settings
    phantom_dir = ctx.repo / ctx.settings.phantom_dir
    phantom = load_phantom(phantom_dir, phantom_settings)
    with SliceStack(phantom_dir / phantom_settings.files.slices_dir, ctx.config.io) as stack:
        img = stack.read(crop.z)
    truth = phantom.truth[crop.z]
    truth = np.where(truth == phantom_settings.excluded_value, Label.BACKGROUND, truth)
    render = ctx.config.render
    seg = np.asarray(ctx.reference_labels(PHANTOM_RUN)[crop.z])
    panels = [
        overlay(img, np.zeros(img.shape, np.uint8), render)[crop.window],
        overlay(img, truth.astype(np.uint8), render)[crop.window],
        overlay(img, seg, render)[crop.window],
    ]
    for name, panel in zip(("raw", "truth", "segmentation"), panels, strict=True):
        ctx.save(ctx.figures, f"phantom_{name}", panel)
    strip = compose(panels, [ctx.title(t) for t in fig.phantom_titles], fig)
    ctx.save(ctx.screenshots, SCREENSHOT_PHANTOM, strip)
