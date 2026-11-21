"""Tests of the anchoring, measures and pipeline of the quantification."""

import csv
import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pybones.quantification.graph import build_network
from pybones.quantification.measures import branch_measures, end_kinds, summarize
from pybones.quantification.network import NO_BRANCH, EndKind, NodeKind
from pybones.quantification.pipeline import BRANCH_COLUMNS, run_quantification
from pybones.segmentation.pipeline import run_segmentation
from pybones.segmentation.stages import QUANTIFY_STAGES, CancelledError, ProgressTracker
from tests.helpers import KEEP_ALL, cancel_in, line, small_config

if TYPE_CHECKING:
    from pathlib import Path

    from pybones.segmentation.stages import Stage

SHAPE = (20, 40, 40)
UNIT = (1.0, 1.0, 1.0)
MARGIN = 1.0


def test_ends_near_a_face_or_a_burst_are_anchored() -> None:
    """Within its radius plus the margin of a face it is a face end; of a burst, a burst end."""
    points = np.asarray([[10, 20, 2], [10, 20, 20], [10, 20, 30], [0, 20, 20]], np.int32)
    radius = np.asarray([2.0, 2.0, 2.0, 5.0], np.float32)
    burst = np.asarray([50.0, 50.0, 2.5, 1.0], np.float32)
    kinds = end_kinds(points, radius, burst, SHAPE, UNIT, MARGIN)
    # x=2 is 2 µm from the face (radius 2 + margin 1 reaches it); the last one is on a face
    # and near a burst: the face wins.
    assert kinds.tolist() == [EndKind.FACE, EndKind.FREE, EndKind.BURST, EndKind.FACE]


def test_branch_measures_follow_the_voxels() -> None:
    """Length, chord, tortuosity, radius statistics (median of an even count) per branch."""
    voxels = line((5, 20, 5), (5, 20, 15)) + line((5, 20, 15), (5, 28, 23))
    points = np.asarray(sorted(set(voxels)), np.int32)
    radius = np.linspace(1.0, 4.0, len(points)).astype(np.float32)
    kinds = np.full(len(points), EndKind.FREE, np.uint8)
    net = build_network(points, radius, kinds, SHAPE, UNIT, KEEP_ALL)
    m = branch_measures(net, radius)
    assert len(m.length_um) == 1
    on = net.point_branch == 0
    assert m.radius_median_um[0] == pytest.approx(float(np.median(radius[on])))
    assert m.radius_mean_um[0] == pytest.approx(float(radius[on].mean()))
    assert (m.radius_min_um[0], m.radius_max_um[0]) == (radius[on].min(), radius[on].max())
    expected_chord = math.dist((5, 20, 5), (5, 28, 23))
    assert m.chord_um[0] == pytest.approx(expected_chord, rel=1e-6)
    assert m.tortuosity[0] == pytest.approx(net.branch_length_um[0] / expected_chord)
    assert not m.truncated[0] and not m.at_burst[0]


def test_loops_have_no_tortuosity_and_flags_follow_the_ends() -> None:
    """A ring's tortuosity is undefined; a branch to a face end is truncated."""
    ring = line((5, 10, 10), (5, 10, 20)) + line((5, 10, 20), (5, 20, 20))
    ring += line((5, 20, 20), (5, 20, 10)) + line((5, 20, 10), (5, 10, 10))
    stub = line((5, 30, 5), (5, 30, 15))
    points = np.asarray(sorted(set(ring + stub)), np.int32)
    kinds = np.full(len(points), EndKind.FREE, np.uint8)
    kinds[np.flatnonzero((points == (5, 30, 5)).all(axis=1))] = EndKind.FACE
    radius = np.full(len(points), 2.0, np.float32)
    net = build_network(points, radius, kinds, SHAPE, UNIT, KEEP_ALL)
    m = branch_measures(net, radius)
    loop = int(np.flatnonzero(net.branch_start == net.branch_end)[0])
    assert math.isnan(m.tortuosity[loop])
    assert m.truncated.tolist().count(True) == 1
    summary = summarize(net, m, radius, SHAPE, UNIT)
    assert (summary.branches, summary.loops, summary.truncated_branches) == (2, 1, 1)
    assert (summary.face_ends, summary.free_ends, summary.junctions) == (1, 1, 0)
    assert summary.total_length_mm == pytest.approx(net.branch_length_um.sum() / 1000)
    assert summary.stack_volume_mm3 == pytest.approx(np.prod(SHAPE) / 1e9)


@pytest.fixture
def segmentation(slice_dir: "Path", tmp_path: "Path") -> "Path":
    """Segment the synthetic stack.

    Args:
        slice_dir: Synthetic slices.
        tmp_path: Temporary directory of the test.

    Returns:
        The segmentation folder.
    """
    out = tmp_path / "seg"
    _ = run_segmentation(slice_dir, out, small_config())
    return out


def test_the_pipeline_writes_consistent_files(segmentation: "Path") -> None:
    """Summary, branch table and centrelines agree with each other."""
    config = small_config()
    files = config.quantification.files
    summary = run_quantification(segmentation, config)
    record = json.loads((segmentation / files.network_file).read_text())
    assert record["summary"] == summary.model_dump(mode="json")
    assert (
        len(record["nodes"]["kind"])
        == summary.junctions + summary.free_ends + summary.face_ends + summary.burst_ends
    )
    with (segmentation / files.branches_file).open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == BRANCH_COLUMNS
    assert len(rows) == summary.branches
    with np.load(segmentation / files.centrelines_file) as data:
        assert set(data.files) == {"points", "point_branch", "point_node", "distance_um"}
        assert data["point_branch"].max(initial=NO_BRANCH) < summary.branches
    total = sum(float(r["length_um"]) for r in rows) + sum(record["nodes"]["length_um"])
    assert total / 1000 == pytest.approx(summary.total_length_mm, rel=1e-3)
    assert sum(k == NodeKind.JUNCTION for k in record["nodes"]["kind"]) == summary.junctions


@pytest.mark.parametrize("stage", QUANTIFY_STAGES)
def test_a_cancelled_quantification_leaves_no_file(segmentation: "Path", stage: "Stage") -> None:
    """Cancelled at any stage, the run raises and writes none of its three files."""
    config = small_config()
    tracker = ProgressTracker(QUANTIFY_STAGES, config.quantification.stage_weights)
    tracker.add_listener(cancel_in(tracker, stage))
    with pytest.raises(CancelledError):
        _ = run_quantification(segmentation, config, tracker)
    files = config.quantification.files
    for name in (files.network_file, files.branches_file, files.centrelines_file):
        assert not (segmentation / name).exists()
    assert not list(segmentation.glob("*.partial"))


def test_no_centreline_gives_an_empty_network() -> None:
    """Without any skeleton voxel, every step returns empty results instead of failing."""
    points = np.zeros((0, 3), np.int32)
    empty = np.zeros(0, np.float32)
    kinds = end_kinds(points, empty, empty, SHAPE, UNIT, MARGIN)
    net = build_network(points, empty, kinds, SHAPE, UNIT, KEEP_ALL)
    assert (len(net.branch_length_um), len(net.node_kind)) == (0, 0)
    summary = summarize(net, branch_measures(net, empty), empty, SHAPE, UNIT)
    assert (summary.branches, summary.total_length_mm) == (0, 0.0)
    assert math.isnan(summary.median_radius_um)


def test_a_segmentation_without_vessels_is_quantified(segmentation: "Path") -> None:
    """A stack with no vessel label gives an empty network and still writes its files."""
    config = small_config()
    labels_file = segmentation / config.io.labels_file
    shape = np.load(labels_file, mmap_mode="r").shape
    np.save(labels_file, np.zeros(shape, np.uint8))
    summary = run_quantification(segmentation, config)
    assert (summary.branches, summary.junctions) == (0, 0)
    assert (segmentation / config.quantification.files.branches_file).is_file()
