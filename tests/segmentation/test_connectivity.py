"""Tests of the streaming 3D connected components (:mod:`pybones.segmentation.connectivity`)."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.segmentation.connectivity import StreamingComponents, connected_roots
from tests.helpers import NEIGHBOURS_6, NEIGHBOURS_26, flood_fill_components

if TYPE_CHECKING:
    from numpy.typing import NDArray


def stream(
    volume: "NDArray[np.bool_]", connectivity: int
) -> "tuple[StreamingComponents, NDArray[np.int64]]":
    """Track the components of a volume slice by slice.

    Args:
        volume: Binary volume.
        connectivity: In-plane connectivity (4 or 8).

    Returns:
        The tracker and its ``global id -> component`` table.
    """
    tracker = StreamingComponents(connectivity)
    for sl in volume:
        _ = tracker.add_slice(sl)
    return tracker, tracker.resolve()


@pytest.mark.parametrize(("connectivity", "neighbours"), [(8, NEIGHBOURS_26), (4, NEIGHBOURS_6)])
@pytest.mark.parametrize("density", [0.2, 0.35])
def test_component_count_matches_flood_fill(
    connectivity: int, neighbours: tuple[tuple[int, int, int], ...], density: float
) -> None:
    """Component count and sizes match a breadth-first flood fill on random volumes."""
    rng = np.random.default_rng(int(density * 100) + connectivity)
    volume = rng.random((8, 24, 24)) < density
    tracker, component_of = stream(volume, connectivity)
    assert component_of.max() == flood_fill_components(volume, neighbours)
    assert tracker.component_sizes(component_of).sum() == volume.sum()


def test_diagonal_link_across_slices_is_26_connected_only() -> None:
    """Voxels touching by a corner across slices are linked in 26- but not 6-connectivity."""
    volume = np.zeros((2, 5, 5), bool)
    volume[0, 1, 1] = volume[1, 2, 2] = True
    assert stream(volume, 8)[1].max() == 1
    assert stream(volume, 4)[1].max() == 2


def test_relabelling_maps_voxels_to_their_component() -> None:
    """Re-labelling a slice and adding its offset finds each voxel's component."""
    volume = np.zeros((3, 6, 6), bool)
    volume[:, 1, 1] = True  # a rod through the 3 slices
    volume[1, 4, 4] = True  # an isolated voxel
    tracker, component_of = stream(volume, 8)
    rod = {
        int(component_of[tracker.local_labels(volume[z])[1, 1] + tracker.offsets[z]])
        for z in range(3)
    }
    dot = int(component_of[tracker.local_labels(volume[1])[4, 4] + tracker.offsets[1]])
    assert len(rod) == 1
    assert dot not in rod
    sizes = tracker.component_sizes(component_of)
    assert sizes[rod.pop()] == 3
    assert sizes[dot] == 1


def test_connected_roots_on_a_chain_and_isolated_nodes() -> None:
    """A chain, a pair and isolated nodes give four components, node 0 first."""
    edges = np.array([[5, 4], [4, 3], [1, 2]], dtype=np.int64)
    roots = connected_roots(7, edges)
    assert roots[0] == 0
    assert roots[3] == roots[4] == roots[5]
    assert roots[1] == roots[2]
    assert len(set(roots.tolist())) == 4  # {0}, {1,2}, {3,4,5}, {6}
