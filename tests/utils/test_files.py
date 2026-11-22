"""Tests of the files that appear only once complete (:mod:`pybones.utils.files`)."""

from typing import TYPE_CHECKING

import pytest

from pybones.utils import files as files_module
from pybones.utils.files import complete_files

if TYPE_CHECKING:
    from pathlib import Path

OLD = "old"
NEW = "new"


class WriteError(Exception):
    """Raised by a test in the middle of writing."""


def test_finished_writes_replace_the_files_together(tmp_path: "Path") -> None:
    """Both files take their final names at the end; nothing temporary is left."""
    first, second = tmp_path / "a.json", tmp_path / "b.csv"
    _ = first.write_text(OLD)
    with complete_files(first, second) as (a, b):
        _ = a.write_text(NEW)
        _ = b.write_text(NEW)
        assert first.read_text() == OLD  # not before the end
        assert not second.exists()
    assert (first.read_text(), second.read_text()) == (NEW, NEW)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.csv"]


def test_an_interrupted_write_leaves_the_old_files(tmp_path: "Path") -> None:
    """A failure in the block removes the temporary files and keeps the final ones."""
    first, second = tmp_path / "a.json", tmp_path / "b.csv"
    _ = first.write_text(OLD)
    with pytest.raises(WriteError), complete_files(first, second) as (a, b):
        _ = a.write_text(NEW)
        _ = b.write_text(NEW)
        raise WriteError
    assert first.read_text() == OLD
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_files_reach_the_disk_before_their_names(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each file is flushed before any rename, and the folder's names after the renames."""
    events: list[str] = []
    real_file, real_folder = files_module.sync_file, files_module.sync_folder

    def flush_file(path: "Path") -> None:
        """Note the flush (and the final file's state at that time), then flush.

        Args:
            path: The file.
        """
        events.append(f"file {path.name} (finals exist: {first.exists() or second.exists()})")
        real_file(path)

    def flush_folder(folder: "Path") -> None:
        """Note the flush (and whether both files are in place), then flush.

        Args:
            folder: The folder.
        """
        events.append(f"folder (finals exist: {first.exists() and second.exists()})")
        real_folder(folder)

    monkeypatch.setattr(files_module, "sync_file", flush_file)
    monkeypatch.setattr(files_module, "sync_folder", flush_folder)
    first, second = tmp_path / "a.json", tmp_path / "b.csv"
    with complete_files(first, second) as (a, b):
        _ = a.write_text(NEW)
        _ = b.write_text(NEW)
    # The folder is always asked; the real flush does nothing where folders cannot be synced.
    assert events == [
        f"file {a.name} (finals exist: False)",
        f"file {b.name} (finals exist: False)",
        "folder (finals exist: True)",
    ]
    assert first.read_text() == second.read_text() == NEW
