"""Tests of the interchangeable quantification steps (:mod:`pybones.quantification`)."""

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from pybones.config.loader import load_config
from pybones.quantification.cavities import ScipyCavityFiller, StreamingCavityFiller
from pybones.quantification.distances import ScipyDistanceProbe, SliceDistanceProbe
from pybones.quantification.protocols import CavityBackend, DistanceBackend, SkeletonBackend
from pybones.quantification.registry import (
    CAVITY_FILLERS,
    DISTANCE_PROBES,
    SKELETONIZERS,
)
from pybones.quantification.skeleton import ScikitImageSkeletonizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.quantification.protocols import CavityFiller, DistanceProbe

SHAPE = (14, 24, 26)
CAP_UM = 9.0
CHUNK = 3
TILE = 9
ISOTROPIC = (2.0, 1.5, 1.5)
ANISOTROPIC = (2.0, 1.0, 1.5)
FILLERS: "list[CavityFiller]" = [StreamingCavityFiller(), ScipyCavityFiller()]


def backend_name(backend: object) -> str:
    """Name a backend in the test ids.

    Args:
        backend: Implementation of a step.

    Returns:
        Its class name.
    """
    return type(backend).__name__


def blobs(seed: int, density: float = 0.55) -> "NDArray[np.bool_]":
    """Draw a random blobby mask (smoothed noise), with cavities and tunnels of all kinds.

    Args:
        seed: Random seed.
        density: Fraction of noise kept before smoothing.

    Returns:
        A ``SHAPE`` mask.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(size=SHAPE)
    smooth = (noise + np.roll(noise, 1, 0) + np.roll(noise, 1, 1) + np.roll(noise, 1, 2)) / 4
    return smooth < density


def hollow_ball(outer: float, inner: float) -> "NDArray[np.bool_]":
    """Draw a spherical shell (a sealed cavity inside).

    Args:
        outer: Outer radius in voxels.
        inner: Inner radius in voxels.

    Returns:
        A ``SHAPE`` mask.
    """
    zz, yy, xx = np.indices(SHAPE)
    r = np.sqrt((zz - SHAPE[0] / 2) ** 2 + (yy - SHAPE[1] / 2) ** 2 + (xx - SHAPE[2] / 2) ** 2)
    return (r <= outer) & (r > inner)


def brute_distances(
    mask: "NDArray[np.bool_]", points: "NDArray[np.intp]", voxel: tuple[float, float, float]
) -> "NDArray[np.float64]":
    """Compute distances to the nearest background voxel by checking every one of them.

    Args:
        mask: Mask.
        points: ``(N, 3)`` voxel indices.
        voxel: Voxel size ``(z, y, x)``.

    Returns:
        Distances in µm, capped at :data:`CAP_UM` (no background beyond the volume).
    """
    background = np.argwhere(~mask) * np.asarray(voxel)
    out = np.full(len(points), CAP_UM)
    if len(background):
        here = points * np.asarray(voxel)
        dists = np.sqrt(((here[:, None, :] - background[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        out = np.minimum(dists, CAP_UM)
    return out


@pytest.mark.parametrize("filler", FILLERS, ids=backend_name)
def test_a_sealed_cavity_is_filled(filler: "CavityFiller") -> None:
    """The inside of a closed shell is filled; nothing else changes."""
    shell = hollow_ball(6, 3)
    filled = filler.fill(shell)
    np.testing.assert_array_equal(filled, hollow_ball(6, -1))
    assert not shell[SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2]  # the input is not modified


@pytest.mark.parametrize("filler", FILLERS, ids=backend_name)
def test_a_tunnel_to_a_face_is_not_a_cavity(filler: "CavityFiller") -> None:
    """Background reaching a face of the volume stays, whatever the face (z, y or x)."""
    for axis in range(3):
        shell = hollow_ball(6, 3)
        tunnel = [
            slice(SHAPE[0] // 2, SHAPE[0] // 2 + 1),
            slice(SHAPE[1] // 2, SHAPE[1] // 2 + 1),
            slice(SHAPE[2] // 2, SHAPE[2] // 2 + 1),
        ]
        tunnel[axis] = slice(0, (SHAPE[axis] + 1) // 2 + 1)
        shell[tuple(tunnel)] = False
        np.testing.assert_array_equal(filler.fill(shell), shell)


@pytest.mark.parametrize("seed", range(4))
def test_both_cavity_fillers_agree(seed: int) -> None:
    """On random masks full of cavities and tunnels, both backends fill the same voxels."""
    mask = blobs(seed)
    filled = [filler.fill(mask) for filler in FILLERS]
    assert (filled[0] & ~mask).any()  # the masks do have cavities
    np.testing.assert_array_equal(filled[0], filled[1])


def test_a_solid_tube_thins_to_its_axis() -> None:
    """A straight cylinder along x becomes a one-voxel line near its axis."""
    zz, yy, _ = np.indices(SHAPE)
    centre = (SHAPE[0] // 2, SHAPE[1] // 2)
    tube = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= 9
    skeleton = ScikitImageSkeletonizer().skeletonize(tube)
    points = np.argwhere(skeleton)
    assert len(points) > 0
    assert (np.abs(points[:, 0] - centre[0]) <= 1).all()
    assert (np.abs(points[:, 1] - centre[1]) <= 1).all()
    assert len(np.unique(points[:, 2])) == len(points)  # one voxel per x: a line


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "probe",
    [SliceDistanceProbe(CAP_UM), ScipyDistanceProbe(CAP_UM, CHUNK, TILE)],
    ids=["pybones", "scipy"],
)
def test_distances_match_a_brute_force_search(seed: int, probe: "DistanceProbe") -> None:
    """Every probe finds the exact distance to the nearest background voxel, up to the cap."""
    mask = blobs(seed, density=0.6)
    points = np.argwhere(mask)
    expected = brute_distances(mask, points, ISOTROPIC)
    np.testing.assert_allclose(probe.distances(mask, points, ISOTROPIC), expected, rtol=1e-5)


def test_the_scipy_probe_handles_rectangular_pixels() -> None:
    """Anisotropic in-plane voxels: SciPy measures them; the OpenCV probe says why it cannot."""
    mask = blobs(5, density=0.6)
    points = np.argwhere(mask)
    expected = brute_distances(mask, points, ANISOTROPIC)
    got = ScipyDistanceProbe(CAP_UM, CHUNK, TILE).distances(mask, points, ANISOTROPIC)
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    with pytest.raises(ValueError, match="square pixels"):
        _ = SliceDistanceProbe(CAP_UM).distances(mask, points, ANISOTROPIC)


def test_a_mask_filling_the_volume_has_no_wall() -> None:
    """With no background at all, every distance is the cap (the faces are not walls)."""
    mask = np.ones(SHAPE, dtype=np.bool_)
    points = np.argwhere(mask)[::37]
    for probe in (SliceDistanceProbe(CAP_UM), ScipyDistanceProbe(CAP_UM, CHUNK, TILE)):
        np.testing.assert_allclose(probe.distances(mask, points, ISOTROPIC), CAP_UM)


def test_every_backend_name_builds_its_step() -> None:
    """Each configurable name has a factory, and the configuration builds each step."""
    assert set(CAVITY_FILLERS) == set(CavityBackend)
    assert set(SKELETONIZERS) == set(SkeletonBackend)
    assert set(DISTANCE_PROBES) == set(DistanceBackend)
    for cavities, distances in itertools.product(CavityBackend, DistanceBackend):
        config = load_config(
            overrides=[
                f'quantification.backends.cavities="{cavities.value}"',
                f'quantification.backends.distances="{distances.value}"',
            ]
        ).quantification
        mask = hollow_ball(6, 3)
        assert (
            CAVITY_FILLERS[config.backends.cavities](config).fill(mask).sum()
            == hollow_ball(6, -1).sum()
        )
        _ = SKELETONIZERS[config.backends.skeleton](config).skeletonize(mask)
        probe = DISTANCE_PROBES[config.backends.distances](config)
        assert probe.distances(mask, np.argwhere(mask), ISOTROPIC).shape == (int(mask.sum()),)


def test_unknown_backends_are_rejected() -> None:
    """A backend name the registry does not know fails when the configuration loads."""
    with pytest.raises(ValidationError):
        _ = load_config(overrides=['quantification.backends.distances="magic"'])
