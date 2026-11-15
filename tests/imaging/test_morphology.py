"""Tests of the binary morphology helpers (:mod:`pybones.imaging.morphology`)."""

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from pybones.imaging import morphology as morph

if TYPE_CHECKING:
    from numpy.typing import NDArray

CONN = 8


def iterative_reconstruct(
    seed: "NDArray[np.bool_]", mask: "NDArray[np.bool_]"
) -> "NDArray[np.bool_]":
    """Reconstruct by geodesic dilation until stability, 8-connected (reference implementation).

    Args:
        seed: Marker image.
        mask: Image to reconstruct within.

    Returns:
        The reconstructed mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    cur = (seed & mask).astype(np.uint8)
    inside = mask.astype(np.uint8)
    while True:
        nxt = cast("NDArray[np.uint8]", cv2.dilate(cur, kernel)) & inside
        if np.array_equal(nxt, cur):
            return cur.astype(bool)
        cur = nxt


def test_reconstruct_matches_iterative_reference() -> None:
    """The one-pass labelling reconstruction equals iterated geodesic dilation."""
    rng = np.random.default_rng(0)
    mask = rng.random((120, 120)) > 0.45
    seed = rng.random((120, 120)) > 0.995
    np.testing.assert_array_equal(
        morph.reconstruct(seed, mask, CONN), iterative_reconstruct(seed, mask)
    )


def test_area_open_removes_only_small_components() -> None:
    """Components below the minimum area go; larger ones stay intact."""
    mask = np.zeros((50, 50), bool)
    mask[5:10, 5:10] = True  # 25 px
    mask[30:32, 30:32] = True  # 4 px
    out = morph.area_open(mask, 10, CONN)
    assert out[5:10, 5:10].all()
    assert not out[30:32, 30:32].any()


def test_label_with_areas() -> None:
    """Areas are reported per label, background first."""
    mask = np.zeros((10, 10), bool)
    mask[0:2, 0:3] = True
    mask[8, 8] = True
    labels, areas = morph.label_with_areas(mask, CONN)
    assert sorted(areas[1:].tolist()) == [1, 6]
    assert labels[8, 8] > 0


def test_kernels_are_cached_and_read_only() -> None:
    """Kernels are shared per radius, sized ``2 r + 1`` per axis and immutable."""
    k = morph.ellipse((3, 3))
    assert k is morph.ellipse((3, 3))
    assert k.shape == (7, 7)
    assert morph.ellipse((1, 3)).shape == (3, 7)  # anisotropic pixels: (ry, rx)
    assert not k.flags.writeable
    assert morph.neighbourhood(8).sum() == 9
    assert morph.neighbourhood(4).sum() == 5


def test_dilate_neighbours() -> None:
    """One 8-connected step grows a 4x4 square to 6x6."""
    mask = np.zeros((40, 40), bool)
    mask[18:22, 18:22] = True
    assert morph.dilate_neighbours(mask, CONN).sum() == 36


def test_ellipse_kernels_are_symmetric_and_degenerate_gracefully() -> None:
    """Disks are symmetric without spikes; a zero radius gives a straight line."""
    disk = morph.ellipse((3, 3))
    assert (disk == disk.T).all()
    assert (disk == disk[::-1]).all()
    assert disk[0].sum() == 3  # rounded tip, no one-pixel spike
    assert morph.ellipse((0, 3)).tolist() == [[1] * 7]  # one-row ellipse = full line
    assert (morph.ellipse((3, 0)) == 1).all()


def test_fill_holes_is_topological() -> None:
    """Enclosed holes are filled; notches open to the border are not."""
    mask = np.zeros((20, 20), bool)
    mask[2:12, 2:12] = True
    mask[5:9, 5:9] = False  # enclosed hole
    mask[14:18, 0:6] = True
    mask[15:17, 0:3] = False  # notch open to the border: not a hole
    out = morph.fill_holes(mask)
    assert out[5:9, 5:9].all()
    assert not out[15:17, 0:3].any()
    assert (out & ~mask).sum() == 16  # nothing else changed
