"""Quantify the vessels of a segmentation: from its label volume to network files.

Stages (each reported to the progress callback, which can cancel between them):

1. **cavities**: the vessel mask with its cavities filled;
2. **skeleton**: its centreline voxels;
3. **calibre**: each voxel's distance to the wall, and to the nearest burst;
4. **network**: nodes, branches, pruning, measures, files.

The voxel size is the one the segmentation was made with (its metadata); the
quantification settings are today's. The three files appear together, once complete.
"""

import csv
import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pybones import __version__
from pybones.quantification.graph import build_network
from pybones.quantification.measures import branch_measures, end_kinds, summarize
from pybones.quantification.registry import CAVITY_FILLERS, DISTANCE_PROBES, SKELETONIZERS
from pybones.segmentation.labels import Label
from pybones.segmentation.results import load_labels, load_meta
from pybones.segmentation.stages import Stage, no_progress
from pybones.utils.constants import JSON_INDENT
from pybones.utils.files import complete_files

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.config.quantification import QuantificationConfig
    from pybones.config.schema import PyBonesConfig
    from pybones.quantification.measures import BranchMeasures, NetworkSummary
    from pybones.quantification.network import Network
    from pybones.segmentation.stages import ProgressCallback

#: Columns of the branch table, in order.
BRANCH_COLUMNS = (
    "branch",
    "start_node",
    "end_node",
    "length_um",
    "chord_um",
    "tortuosity",
    "radius_mean_um",
    "radius_median_um",
    "radius_min_um",
    "radius_max_um",
    "voxels",
    "truncated",
    "at_burst",
)
#: Decimals written for µm values and ratios.
_DECIMALS = 3

logger = logging.getLogger(__name__)


def _cell(value: object) -> str:
    """Format a table cell (empty for NaN, rounded floats, 0/1 booleans).

    Args:
        value: Cell value.

    Returns:
        Its text.
    """
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "" if math.isnan(value) else f"{value:.{_DECIMALS}f}"
    return str(value)


def _write_branches(path: "Path", network: "Network", measures: "BranchMeasures") -> None:
    """Write the branch table as CSV.

    Args:
        path: Output file.
        network: The network (end nodes).
        measures: Its branch measures.
    """
    columns = (
        np.arange(len(measures.length_um)),
        network.branch_start,
        network.branch_end,
        measures.length_um,
        measures.chord_um,
        measures.tortuosity,
        measures.radius_mean_um,
        measures.radius_median_um,
        measures.radius_min_um,
        measures.radius_max_um,
        measures.voxels,
        measures.truncated,
        measures.at_burst,
    )
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(BRANCH_COLUMNS)
        writer.writerows([_cell(v) for v in row] for row in zip(*columns, strict=True))


def _write_files(
    segmentation: "Path",
    config: "QuantificationConfig",
    measured: "MeasuredNetwork",
    progress: "ProgressCallback",
) -> None:
    """Write the three result files under temporary names, then rename them together.

    Args:
        segmentation: Segmentation folder.
        config: Quantification settings (file names, recorded with the summary).
        measured: The network and its measures.
        progress: Progress callback (the last cancellation point is before the renames).
    """
    network, distances_um = measured.network, measured.distances_um
    measures, summary = measured.measures, measured.summary
    files = config.files
    finals = (
        segmentation / name
        for name in (files.network_file, files.branches_file, files.centrelines_file)
    )
    with complete_files(*finals) as (network_path, branches_path, centrelines_path):
        record = {
            "pybones_version": __version__,
            "summary": summary.model_dump(mode="json"),
            "settings": config.model_dump(mode="json"),
            "nodes": {
                "position_um": network.node_position_um.round(_DECIMALS).tolist(),
                "kind": network.node_kind.tolist(),
                "degree": network.node_degree.tolist(),
                "length_um": network.node_length_um.round(_DECIMALS).tolist(),
            },
        }
        _ = network_path.write_text(json.dumps(record, indent=JSON_INDENT))
        _write_branches(branches_path, network, measures)
        with centrelines_path.open("wb") as fh:
            np.savez_compressed(
                fh,
                points=network.points,
                point_branch=network.point_branch,
                point_node=network.point_node,
                distance_um=distances_um,
            )
        progress(Stage.NETWORK, 1, 1)  # the last moment a cancellation leaves nothing behind


@dataclass(frozen=True, slots=True)
class MeasuredNetwork:
    """A vessel network and its measures.

    Attributes:
        network: Nodes and branches.
        distances_um: Wall distance of each centreline voxel.
        measures: Per-branch measures.
        summary: Whole-stack measures.
    """

    network: "Network"
    distances_um: "NDArray[np.float32]"
    measures: "BranchMeasures"
    summary: "NetworkSummary"


def measure_network(
    labels: "NDArray[np.uint8]",
    voxel_um: tuple[float, float, float],
    config: "QuantificationConfig",
    progress: "ProgressCallback" = no_progress,
) -> MeasuredNetwork:
    """Measure the vessel network of a label volume (nothing is written).

    Args:
        labels: ``(z, y, x)`` label volume (a memory map is read, never copied whole).
        voxel_um: Voxel size ``(z, y, x)`` in µm.
        config: Quantification settings.
        progress: Progress callback (quantification stages; cancels between them).

    Returns:
        The network and its measures.
    """
    shape = (int(labels.shape[0]), int(labels.shape[1]), int(labels.shape[2]))

    progress(Stage.CAVITIES, 0, 1)
    vessels = CAVITY_FILLERS[config.backends.cavities](config).fill(
        np.asarray(labels) == Label.VESSEL
    )
    progress(Stage.CAVITIES, 1, 1)

    progress(Stage.SKELETON, 0, 1)
    skeleton = SKELETONIZERS[config.backends.skeleton](config).skeletonize(vessels)
    points = np.argwhere(skeleton).astype(np.int32)
    del skeleton
    progress(Stage.SKELETON, 1, 1)

    progress(Stage.CALIBRE, 0, 1)
    probe = DISTANCE_PROBES[config.backends.distances](config)
    distances = probe.distances(vessels, points, voxel_um)
    del vessels
    burst_distances = probe.distances(np.asarray(labels) != Label.BURST, points, voxel_um)
    progress(Stage.CALIBRE, 1, 1)

    progress(Stage.NETWORK, 0, 1)
    kinds = end_kinds(points, distances, burst_distances, shape, voxel_um, config.anchor_margin_um)
    network = build_network(points, distances, kinds, shape, voxel_um, config.pruning)
    measures = branch_measures(network, distances)
    summary = summarize(network, measures, distances, shape, voxel_um)
    return MeasuredNetwork(network, distances, measures, summary)


def run_quantification(
    segmentation: "Path", config: "PyBonesConfig", progress: "ProgressCallback" = no_progress
) -> "NetworkSummary":
    """Quantify the vessels of a segmentation and write the network files next to it.

    Args:
        segmentation: Segmentation folder (labels and metadata).
        config: Configuration (quantification settings, file names).
        progress: Progress callback (quantification stages); a cancelled tracker stops the
            run between stages, leaving no file.

    Returns:
        The network summary.
    """
    voxel = load_meta(segmentation, config.io).config.acquisition.voxel_size_um
    labels = load_labels(segmentation, config.io)
    measured = measure_network(labels, voxel, config.quantification, progress)
    _write_files(segmentation, config.quantification, measured, progress)
    summary = measured.summary
    logger.info(
        "Quantified %s: %d branches (%d truncated), %d junctions, %.1f mm of vessels "
        + "(%.2f mm/mm³), median radius %.1f µm",
        segmentation,
        summary.branches,
        summary.truncated_branches,
        summary.junctions,
        summary.total_length_mm,
        summary.length_density_mm_per_mm3,
        summary.median_radius_um,
    )
    return summary
