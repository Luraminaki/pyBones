"""Tests of the files that appear only once complete (:mod:`pybones.utils.files`)."""

from typing import TYPE_CHECKING

import pytest

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
