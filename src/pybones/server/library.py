"""The stacks and segmentations the web server can reach, addressed by id only.

A client never sends a path: datasets and segmentations are named by their entry in
the data or output folder (:data:`~pybones.server.schemas.ID_PATTERN`), and every
id is resolved here, inside its folder.
"""

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pybones.io.slices import SliceStack
from pybones.segmentation.labels import FOREGROUND_LABELS
from pybones.segmentation.results import IncompatibleOutputError, load_meta
from pybones.server.schemas import (
    ID_PATTERN,
    MAX_ID_LENGTH,
    DatasetInfo,
    DatasetKind,
    SegmentationSummary,
)

if TYPE_CHECKING:
    from pybones.config.schema import IOConfig, LibraryConfig
    from pybones.segmentation.labels import Label
    from pybones.segmentation.results import SegmentationMeta

_ID_RE = re.compile(ID_PATTERN)

logger = logging.getLogger(__name__)


class NotFoundError(LookupError):
    """No dataset, segmentation, slice, mesh or job has this id."""


class ConflictError(Exception):
    """The request clashes with what exists, e.g. a segmentation name already taken."""


def is_entry_id(name: str) -> bool:
    """Tell whether a name is a valid dataset or segmentation id.

    Args:
        name: Entry name.

    Returns:
        Whether it matches :data:`~pybones.server.schemas.ID_PATTERN` and its length limit.
    """
    return len(name) <= MAX_ID_LENGTH and _ID_RE.fullmatch(name) is not None


@dataclass(frozen=True, slots=True)
class Library:
    """Datasets (data folder) and segmentations (output folder) of the server.

    Attributes:
        data_dir: Absolute data folder.
        output_dir: Absolute output folder.
        io: File names of the segmentation outputs and slice discovery rules.
        mesh_file_template: Name of a cached mesh inside its segmentation (``{label}``).
    """

    data_dir: Path
    output_dir: Path
    io: "IOConfig"
    mesh_file_template: str

    @classmethod
    def from_config(cls, config: "LibraryConfig", io: "IOConfig") -> "Library":
        """Build the library, resolving relative folders against the working directory.

        Args:
            config: Folders and mesh file name.
            io: File names of the segmentation outputs.

        Returns:
            The library.
        """
        return cls(
            data_dir=Path(config.data_dir).resolve(),
            output_dir=Path(config.output_dir).resolve(),
            io=io,
            mesh_file_template=config.mesh_file_template,
        )

    @staticmethod
    def _entries(folder: Path) -> list[Path]:
        """List the entries of a folder whose name is a valid id.

        Args:
            folder: Data or output folder.

        Returns:
            Its entries, sorted by name (none if the folder does not exist).
        """
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if is_entry_id(p.name))

    @staticmethod
    def _child(folder: Path, entry_id: str) -> Path:
        """Resolve an id inside a folder.

        Args:
            folder: Data or output folder.
            entry_id: Requested id.

        Returns:
            The entry's path (which may not exist).

        Raises:
            NotFoundError: If ``entry_id`` is not a valid id.
        """
        if not is_entry_id(entry_id):
            raise NotFoundError(f"invalid id {entry_id!r}")
        return folder / entry_id

    def datasets(self) -> list[DatasetInfo]:
        """List the slice stacks of the data folder (entries holding no slice are skipped).

        Returns:
            One description per stack, sorted by id.
        """
        found: list[DatasetInfo] = []
        for path in self._entries(self.data_dir):
            kind = DatasetKind.DIRECTORY if path.is_dir() else DatasetKind.ZIP
            if kind == DatasetKind.ZIP and not zipfile.is_zipfile(path):
                continue
            try:
                with SliceStack(path, self.io) as stack:
                    depth, height, width = stack.shape
            except (OSError, ValueError) as exc:
                logger.debug("Skipping %s: %s", path, exc)
                continue
            found.append(
                DatasetInfo(id=path.name, kind=kind, slices=depth, height=height, width=width)
            )
        return found

    def dataset_path(self, dataset_id: str) -> Path:
        """Locate a dataset.

        Args:
            dataset_id: Dataset id.

        Returns:
            Its directory or archive.

        Raises:
            NotFoundError: If the data folder has no such entry.
        """
        path = self._child(self.data_dir, dataset_id)
        if not path.exists():
            raise NotFoundError(f"no dataset {dataset_id!r}")
        return path

    def segmentation_path(self, segmentation_id: str) -> Path:
        """Locate a finished segmentation (its metadata is written last).

        Args:
            segmentation_id: Segmentation id.

        Returns:
            Its output directory.

        Raises:
            NotFoundError: If there is no finished segmentation with this id.
        """
        path = self._child(self.output_dir, segmentation_id)
        if not (path / self.io.meta_file).is_file():
            raise NotFoundError(f"no segmentation {segmentation_id!r}")
        return path

    def meta(self, segmentation_id: str) -> "SegmentationMeta":
        """Read the metadata of a segmentation.

        Args:
            segmentation_id: Segmentation id.

        Returns:
            Its validated metadata.
        """
        return load_meta(self.segmentation_path(segmentation_id), self.io)

    def mesh_path(self, segmentation_id: str, label: "Label") -> Path:
        """Locate the cached mesh of a label (it may not exist yet).

        Args:
            segmentation_id: Segmentation id.
            label: Foreground label.

        Returns:
            Path of its STL file.
        """
        name = self.mesh_file_template.format(label=label.key)
        return self.segmentation_path(segmentation_id) / name

    def summary(self, segmentation_id: str) -> SegmentationSummary:
        """Describe a segmentation.

        Args:
            segmentation_id: Segmentation id.

        Returns:
            Its summary.
        """
        meta = self.meta(segmentation_id)
        meshes = [
            label.key
            for label in FOREGROUND_LABELS
            if self.mesh_path(segmentation_id, label).is_file()
        ]
        return SegmentationSummary(
            id=segmentation_id,
            source=meta.source,
            pybones_version=meta.pybones_version,
            shape_zyx=meta.shape_zyx,
            voxel_size_um=meta.config.acquisition.voxel_size_um,
            label_fractions=meta.label_fractions,
            meshes=meshes,
        )

    def segmentations(self) -> list[SegmentationSummary]:
        """List the finished segmentations of the output folder.

        Folders without metadata (unfinished, or not segmentations) are skipped, and so
        are those written by an older pyBones (logged).

        Returns:
            One summary per segmentation, sorted by id.
        """
        found: list[SegmentationSummary] = []
        for path in self._entries(self.output_dir):
            if not (path / self.io.meta_file).is_file():
                continue
            try:
                found.append(self.summary(path.name))
            except (OSError, IncompatibleOutputError) as exc:
                logger.warning("Skipping segmentation: %s", exc)
        return found

    def new_segmentation_path(self, name: str) -> Path:
        """Locate the folder of a new segmentation, checking that the name is free on disk.

        Args:
            name: Id of the new segmentation.

        Returns:
            Its future output directory (not created here).

        Raises:
            ConflictError: If an entry with this name exists already.
        """
        path = self._child(self.output_dir, name)
        if path.exists():
            raise ConflictError(f"{name!r} exists already in the output folder")
        return path
