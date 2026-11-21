"""The README sections written by the report tool, between two markers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def replace_section(path: "Path", start_marker: str, end_marker: str, lines: list[str]) -> None:
    """Replace what lies between two markers of a text file, the markers kept.

    Args:
        path: File to update (the README).
        start_marker: Line opening the section.
        end_marker: Line closing it.
        lines: New content of the section.

    Raises:
        ValueError: If the file lacks the markers, or has them in the wrong order.
    """
    text = path.read_text()
    start, end = text.find(start_marker), text.find(end_marker)
    if start < 0 or end < start:
        raise ValueError(f"{path} lacks the {start_marker!r} ... {end_marker!r} markers")
    section = "\n".join([start_marker, *lines, end_marker])
    _ = path.write_text(text[:start] + section + text[end + len(end_marker) :])
