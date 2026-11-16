"""Tests of the stacks and segmentations the server reaches (:mod:`pybones.server.library`)."""

from typing import TYPE_CHECKING

import pytest

from pybones.config.loader import load_config
from pybones.meshing.formats import MeshFormat
from pybones.segmentation.labels import Label
from pybones.server.library import ConflictError, Library, NotFoundError, is_entry_id
from pybones.server.schemas import DatasetKind
from tests.helpers import N_SYNTHETIC_SLICES, SEGMENTATION, SLICE_SHAPE, STACK_DIR, STACK_ZIP

if TYPE_CHECKING:
    from tests.helpers import Folders


@pytest.fixture
def library(folders: "Folders") -> Library:
    """Build a library over the fixture folders.

    Args:
        folders: Data and output folders.

    Returns:
        The library.
    """
    return Library(
        data_dir=folders.data,
        output_dir=folders.output,
        io=load_config().io,
        mesh_file_template="mesh_{label}.{format}",
    )


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("stack", True),
        ("data_set.zip", True),
        ("run-2.v1", True),
        ("", False),
        (".hidden", False),
        ("..", False),
        ("a/b", False),
        ("a\\b", False),
        ("with space", False),
        ("x" * 200, False),
    ],
)
def test_ids_are_plain_names(name: str, valid: bool) -> None:
    """An id is a plain entry name: no separator, no leading dot, bounded length."""
    assert is_entry_id(name) is valid


def test_datasets_are_stacks_of_the_data_folder(library: Library) -> None:
    """Directories and zips of slices are listed; other files and hidden entries are not."""
    found = {d.id: d for d in library.datasets()}
    assert set(found) == {STACK_DIR, STACK_ZIP}
    assert (found[STACK_DIR].kind, found[STACK_ZIP].kind) == (
        DatasetKind.DIRECTORY,
        DatasetKind.ZIP,
    )
    zipped = found[STACK_ZIP]
    assert (zipped.slices, (zipped.height, zipped.width)) == (N_SYNTHETIC_SLICES, SLICE_SHAPE)


def test_segmentations_are_finished_and_readable(library: Library) -> None:
    """Only folders with valid metadata are segmentations."""
    found = library.segmentations()
    assert [s.id for s in found] == [SEGMENTATION]
    assert found[0].shape_zyx == (N_SYNTHETIC_SLICES, *SLICE_SHAPE)
    assert found[0].meshes == {}
    with pytest.raises(NotFoundError):
        _ = library.segmentation_path("unfinished")


@pytest.mark.parametrize("bad_id", ["..", "../output", ".hidden", "missing"])
def test_ids_never_leave_their_folder(library: Library, bad_id: str) -> None:
    """Invalid or unknown ids are not found, whatever they point at."""
    with pytest.raises(NotFoundError):
        _ = library.dataset_path(bad_id)
    with pytest.raises(NotFoundError):
        _ = library.segmentation_path(bad_id)


def test_new_segmentations_need_a_free_name(library: Library) -> None:
    """A new segmentation cannot reuse the name of an existing entry."""
    assert not library.new_segmentation_path("fresh").exists()
    for taken in (SEGMENTATION, "unfinished"):
        with pytest.raises(ConflictError):
            _ = library.new_segmentation_path(taken)


def test_mesh_files_are_listed_once_written(library: Library) -> None:
    """A mesh file inside the segmentation shows in its summary."""
    _ = library.mesh_path(SEGMENTATION, Label.VESSEL, MeshFormat.GLB).write_bytes(b"")
    assert library.summary(SEGMENTATION).meshes == {Label.VESSEL.key: [MeshFormat.GLB.value]}
