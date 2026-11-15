"""Choice of annotation crops and extraction of their images."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybones.io.slices import SliceStack
from pybones.segmentation.labels import Label
from pybones.visualization.overlay import stretch_to_uint8
from tools.annotation.workspace import CropCategory, CropSpec

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.config.schema import PyBonesConfig
    from tools.annotation.settings import CropSettings
    from tools.annotation.workspace import Workspace

CROP_ID_TEMPLATE = "z{z:03d}_y{y:04d}_x{x:04d}"
#: Columns of the candidate table built by :func:`select_crops`.
_Z, _TILE_Y, _TILE_X, _BURST, _VESSEL = range(5)


def _tile_fractions(labels: "NDArray[np.uint8]", size: int) -> "dict[Label, NDArray[np.float64]]":
    """Measure the fraction of each label in every ``size x size`` tile of a slice (vectorized).

    Args:
        labels: Label slice.
        size: Tile side (the last partial row and column of tiles are dropped).

    Returns:
        Label -> ``(rows, columns)`` fraction per tile.
    """
    ny, nx = labels.shape[0] // size, labels.shape[1] // size
    tiles = labels[: ny * size, : nx * size].reshape(ny, size, nx, size)
    return {lab: (tiles == lab).mean(axis=(1, 3)) for lab in Label}


def _candidates(volume: "NDArray[np.uint8]", z: int, crops: "CropSettings") -> "NDArray":
    """Tabulate the candidate tiles of one slice (vectorized).

    Args:
        volume: Label volume.
        z: Slice index.
        crops: Crop settings (tile size, minimum tissue fraction).

    Returns:
        ``(n, 5)`` float rows: z, tile row, tile column, burst and vessel fractions
        (columns ``_Z`` .. ``_VESSEL``).
    """
    frac = _tile_fractions(np.asarray(volume[z]), crops.size)
    ty, tx = np.nonzero(1.0 - frac[Label.BACKGROUND] >= crops.min_tissue_fraction)
    return np.column_stack(
        [np.full(ty.size, z), ty, tx, frac[Label.BURST][ty, tx], frac[Label.VESSEL][ty, tx]]
    ).astype(np.float64)


def select_crops(volume: "NDArray[np.uint8]", crops: "CropSettings") -> list[CropSpec]:
    """Pick burst-rich, vessel-rich and random tissue crops from a label volume.

    Candidates are the non-overlapping tiles of ``z_samples`` evenly spaced slices
    (leaving room for the context slices) with enough tissue. Each (y, x) tile is
    used at most once, so crops show different places.

    Args:
        volume: ``(Z, Y, X)`` label volume of the current segmentation.
        crops: Crop settings.

    Returns:
        The chosen crops (fewer than requested if not enough candidates qualify).
    """
    depth = volume.shape[0]
    ctx = crops.context_slices
    zs = np.unique(np.linspace(ctx, depth - 1 - ctx, crops.z_samples).round().astype(int))
    table = np.concatenate([_candidates(volume, int(z), crops) for z in zs])
    if table.size == 0:
        return []
    used: set[tuple[int, int]] = set()
    picks: list[CropSpec] = []

    def take(order: "NDArray[np.intp]", category: CropCategory, count: int) -> None:
        """Pick up to ``count`` candidates in ``order``, skipping tiles already used.

        Args:
            order: Candidate rows, best first.
            category: Category of the picked crops.
            count: Crops to pick.
        """
        for i in order:
            if count == 0:
                return
            z, ty, tx = int(table[i, _Z]), int(table[i, _TILE_Y]), int(table[i, _TILE_X])
            if (ty, tx) in used:
                continue
            used.add((ty, tx))
            y, x = ty * crops.size, tx * crops.size
            crop_id = CROP_ID_TEMPLATE.format(z=z, y=y, x=x)
            picks.append(CropSpec(crop_id=crop_id, z=z, y=y, x=x, category=category))
            count -= 1

    has_burst = np.flatnonzero(table[:, _BURST] > 0)
    take(has_burst[np.argsort(-table[has_burst, _BURST])], CropCategory.BURST, crops.n_burst)
    take(np.argsort(-table[:, _VESSEL]), CropCategory.VESSEL, crops.n_vessel)
    rng = np.random.default_rng(crops.seed)
    take(rng.permutation(len(table)), CropCategory.RANDOM, crops.n_random)
    return picks


def extract_crops(
    source: "Path",
    volume: "NDArray[np.uint8]",
    crop_specs: list[CropSpec],
    crops: "CropSettings",
    config: "PyBonesConfig",
    workspace: "Workspace",
    prefill: bool,
) -> None:
    """Write the 8-bit context images (and pre-fills) of every crop.

    Each needed slice is decoded once, whatever the number of crops using it.

    Args:
        source: Raw slice stack (directory or zip).
        volume: Label volume used for the pre-fills.
        crop_specs: Crops to extract.
        crops: Crop settings (size, context).
        config: pybones configuration (I/O rules, display percentiles).
        workspace: Destination workspace.
        prefill: Also write the segmentation of each crop as its starting labels.
    """
    size, ctx = crops.size, crops.context_slices
    by_z: dict[int, list[tuple[CropSpec, int]]] = {}
    for spec in crop_specs:
        for offset in range(-ctx, ctx + 1):
            by_z.setdefault(spec.z + offset, []).append((spec, offset))
        if prefill:
            window = np.asarray(volume[spec.z, spec.y : spec.y + size, spec.x : spec.x + size])
            _ = cv2.imwrite(str(workspace.prefill_path(spec.crop_id)), window)
    render = config.render
    with SliceStack(source, config.io) as stack:
        for z in sorted(by_z):
            display = stretch_to_uint8(stack.read(z), render.low_percentile, render.high_percentile)
            for spec, offset in by_z[z]:
                window = display[spec.y : spec.y + size, spec.x : spec.x + size]
                _ = cv2.imwrite(str(workspace.crop_image_path(spec.crop_id, offset)), window)
