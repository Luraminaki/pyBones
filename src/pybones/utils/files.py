"""Files that appear only once complete."""

from contextlib import contextmanager
from typing import TYPE_CHECKING

from pybones.utils.constants import PARTIAL_SUFFIX

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@contextmanager
def complete_files(*finals: "Path") -> "Generator[tuple[Path, ...]]":
    """Write files under temporary names, renamed to their final names together at the end.

    The block writes the temporary paths it is given. If it finishes, every file is renamed
    (replacing an older one); if it raises (a failure, a cancellation), the temporary files
    are removed and the final ones are left untouched. A reader never sees a partial file.

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
        for partial, final in zip(partials, finals, strict=True):
            _ = partial.replace(final)
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)
