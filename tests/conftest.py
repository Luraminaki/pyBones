"""Fixtures shared by the whole test suite."""

import logging
from typing import TYPE_CHECKING

import cv2
import pytest

from tests.helpers import N_SYNTHETIC_SLICES, synthetic_slice

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_working_directory(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> "Iterator[None]":
    """Run every test from its own directory, and drop the log handlers it installed.

    Entry points write their log files under the working directory, so tests never
    write into the repository.

    Args:
        tmp_path: Temporary directory of the test.
        monkeypatch: Restores the working directory afterwards.

    Yields:
        Control to the test.
    """
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in [h for h in root.handlers if h not in before]:
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def slice_dir(tmp_path: "Path") -> "Path":
    """Write a directory of synthetic 16-bit PNG slices (``img_0.png`` .. ``img_6.png``).

    Args:
        tmp_path: Temporary directory of the test.

    Returns:
        The slice directory.
    """
    d = tmp_path / "stack"
    d.mkdir()
    for z in range(N_SYNTHETIC_SLICES):
        _ = cv2.imwrite(str(d / f"img_{z}.png"), synthetic_slice(seed=z))
    return d
