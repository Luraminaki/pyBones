"""Build the configured implementation of each quantification step (Strategy pattern).

One table per step maps a backend name to a factory taking the quantification settings,
e.g. ``CAVITY_FILLERS[config.backends.cavities](config)``. A new implementation is a class
satisfying the step's protocol, a member of its name enum, and one line here (a test checks
that every name has its factory).
"""

from typing import TYPE_CHECKING

from pybones.quantification.cavities import ScipyCavityFiller, StreamingCavityFiller
from pybones.quantification.distances import ScipyDistanceProbe, SliceDistanceProbe
from pybones.quantification.protocols import CavityBackend, DistanceBackend, SkeletonBackend
from pybones.quantification.skeleton import ScikitImageSkeletonizer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pybones.config.schema import QuantificationConfig
    from pybones.quantification.protocols import CavityFiller, DistanceProbe, Skeletonizer

CAVITY_FILLERS: "Mapping[CavityBackend, Callable[[QuantificationConfig], CavityFiller]]" = {
    CavityBackend.PYBONES: lambda _config: StreamingCavityFiller(),
    CavityBackend.SCIPY: lambda _config: ScipyCavityFiller(),
}
SKELETONIZERS: "Mapping[SkeletonBackend, Callable[[QuantificationConfig], Skeletonizer]]" = {
    SkeletonBackend.SCIKIT_IMAGE: lambda _config: ScikitImageSkeletonizer(),
}
DISTANCE_PROBES: "Mapping[DistanceBackend, Callable[[QuantificationConfig], DistanceProbe]]" = {
    DistanceBackend.PYBONES: lambda config: SliceDistanceProbe(config.max_radius_um),
    DistanceBackend.SCIPY: lambda config: ScipyDistanceProbe(
        config.max_radius_um, config.distance_tile_slices, config.distance_tile_pixels
    ),
}
