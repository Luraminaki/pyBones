"""Files that appear only once complete, and stay complete after a power cut."""

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

from pybones.utils.constants import PARTIAL_SUFFIX

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

#: Whether a folder can be opened to flush its entries (POSIX; Windows flushes them itself).
CAN_SYNC_FOLDERS = hasattr(os, "O_DIRECTORY")


def sync_file(path: "Path") -> None:
    """Flush a written file's content to the disk.

    Args:
        path: The file.
    """
    with path.open("r+b") as fh:
        os.fsync(fh.fileno())


def sync_folder(folder: "Path") -> None:
    """Flush a folder's entries (names created or replaced) to the disk, where possible.

    Args:
        folder: The folder.
    """
    if not CAN_SYNC_FOLDERS:
        return
    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def complete_files(*finals: "Path") -> "Generator[tuple[Path, ...]]":
    """Write files under temporary names, renamed to their final names together at the end.

    The block writes the temporary paths it is given. If it finishes, every file is flushed to
    the disk, then renamed (replacing an older one), then the folders' new names are flushed
    too: a power cut never leaves a complete-looking name on an empty file. If the block raises
    (a failure, a cancellation), the temporary files are removed and the final ones are left
    untouched. A reader never sees a partial file.

    Example:
        >>> with complete_files(out / "a.json", out / "b.csv") as (a, b):
        ...     write_a(a)
        ...     write_b(b)
        ...     progress(stage, 1, 1)  # the last moment a cancellation leaves nothing

    Args:
        finals: Final paths.

    Yields:
        The temporary path of each, in the same order.
    """
    partials = tuple(path.with_name(path.name + PARTIAL_SUFFIX) for path in finals)
    try:
        yield partials
        for partial in partials:
            sync_file(partial)
        for partial, final in zip(partials, finals, strict=True):
            _ = partial.replace(final)
        for folder in {final.parent for final in finals}:
            sync_folder(folder)
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)
