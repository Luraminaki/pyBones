"""The interchangeable steps of the quantification (Strategy pattern) and their names.

Each 3D operation the quantification needs is described by a small interface
(:class:`CavityFiller`, :class:`Skeletonizer`, :class:`DistanceProbe`). Several
implementations can sit behind one interface; the configuration chooses one by name
(``quantification.backends``), and :mod:`pybones.quantification.registry` builds it.
Adding a backend means adding a class and a registry entry: the code using the step
does not change.

This module imports no implementation, so the configuration schema can use the names.

Choosing a backend (timings on the sample stack, 100 x 2300 x 1487 voxels, 2.2 % vessel):

=============  ===========================  ==================================================
Step           Backend                      Pros / cons
=============  ===========================  ==================================================
cavities       ``pybones`` (default)        no extra dependency; one label slice at a time;
                                            6 s. Two passes over the slices.
cavities       ``scipy``                    one library call; 6 s here, but iterative
                                            propagation (time grows with the background's
                                            paths) and several full-size temporaries.
skeleton       ``scikit-image``             Lee's topology-preserving thinning; 9 s, ~1.1 GB
                                            peak (whole volume in memory).
distances      ``pybones`` (default)        exact, 0.4 s, one slice map at a time; square
                                            pixels only.
distances      ``scipy``                    exact for any voxel shape, independent
                                            implementation (cross-check); ~100x slower (39 s).
=============  ===========================  ==================================================

Both backends of a step give the same result (tested on random volumes against each other
and against brute force; identical on the sample, distances within 4e-6 µm).

Limits shared by every backend:

* cavities are background regions that never reach a face: a hollow vessel cut by a face of
  the stack stays open there, and can still thin into a small surface near that face;
* the skeleton of a 26-connected mask has staircase corners (false junctions of a few voxels)
  and short spurs from surface bumps: the network step must merge and prune them;
* distances run between voxel centres, so they exceed a true radius by up to about half a
  voxel, and are capped at ``max_radius_um``; the faces of the stack are not walls.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class CavityBackend(StrEnum):
    """Implementations of :class:`CavityFiller`."""

    PYBONES = "pybones"
    SCIPY = "scipy"


class SkeletonBackend(StrEnum):
    """Implementations of :class:`Skeletonizer`."""

    SCIKIT_IMAGE = "scikit-image"


class DistanceBackend(StrEnum):
    """Implementations of :class:`DistanceProbe`."""

    PYBONES = "pybones"
    SCIPY = "scipy"


class CavityFiller(Protocol):
    """Fills the cavities of a 3D mask: background regions that do not reach the volume's faces.

    A vessel with a dim, unfilled core is segmented as a shell around a cavity; thinning a
    shell gives a surface, not a centreline. Filling the cavity first restores the tube.
    """

    def fill(self, mask: "NDArray[np.bool_]") -> "NDArray[np.bool_]":
        """Fill the cavities of a mask.

        Args:
            mask: ``(Z, Y, X)`` mask (not modified).

        Returns:
            The mask with its cavities set (26-connected foreground, 6-connected background).
        """
        ...


class Skeletonizer(Protocol):
    """Thins a 3D mask down to one-voxel-wide centrelines, keeping its topology."""

    def skeletonize(self, mask: "NDArray[np.bool_]") -> "NDArray[np.bool_]":
        """Thin a mask.

        Args:
            mask: ``(Z, Y, X)`` mask (not modified).

        Returns:
            The centreline voxels.
        """
        ...


class DistanceProbe(Protocol):
    """Measures, at chosen voxels of a mask, the distance to the nearest voxel outside it."""

    def distances(
        self,
        mask: "NDArray[np.bool_]",
        points: "NDArray[np.integer]",
        voxel_um: tuple[float, float, float],
    ) -> "NDArray[np.float32]":
        """Measure the distance from each point to the nearest background voxel.

        Distances are between voxel centres. Beyond the volume there is no background:
        a vessel cut by a face of the stack is not shrunk by it.

        Args:
            mask: ``(Z, Y, X)`` mask.
            points: ``(N, 3)`` voxel indices ``(z, y, x)``, inside the mask.
            voxel_um: Voxel size ``(z, y, x)`` in µm.

        Returns:
            ``(N,)`` distances in µm, capped at the probe's largest radius.
        """
        ...
