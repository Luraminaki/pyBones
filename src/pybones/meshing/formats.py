"""Mesh file formats, chosen from the file extension."""

from enum import StrEnum
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pathlib import Path

_SUFFIX_SEPARATOR = "."


class MeshFormat(StrEnum):
    """File formats a mesh can be written in.

    ``STL`` is what other software (ImageJ, MeshLab, slicers) expects: 32-bit floats,
    every triangle stored with its own corners. ``GLB`` (binary glTF 2.0) is indexed,
    loads straight into a browser's GPU memory, and can store positions as 16-bit
    integers (``mesh.glb_position_bits``).
    """

    STL = "stl"
    GLB = "glb"

    @property
    def suffix(self) -> str:
        """File extension, dot included (e.g. ``.glb``)."""
        return f"{_SUFFIX_SEPARATOR}{self.value}"

    @classmethod
    def from_path(cls, path: "Path") -> Self:
        """Choose the format from a file's extension (case-insensitive).

        Args:
            path: Mesh file.

        Returns:
            The format.

        Raises:
            ValueError: If the extension is not a supported format.
        """
        suffix = path.suffix.lower().removeprefix(_SUFFIX_SEPARATOR)
        if suffix not in {f.value for f in cls}:
            supported = ", ".join(f.suffix for f in cls)
            raise ValueError(f"{path}: unsupported mesh format (use {supported})")
        return cls(suffix)
