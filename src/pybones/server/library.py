"""The stacks and segmentations the web server can reach, addressed by id only.

A client never sends a path: datasets and segmentations are named by their entry in
the data or output folder (:data:`~pybones.server.schemas.ID_PATTERN`), and every
id is resolved here, inside its folder.
"""

import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ValidationError

from pybones.api.operations import MeshRecord
from pybones.io.slices import SliceStack
from pybones.meshing.formats import MeshFormat
from pybones.quantification.pipeline import POINTS_KEY, SETTINGS_KEY, SUMMARY_KEY
from pybones.segmentation.labels import FOREGROUND_LABELS
from pybones.segmentation.results import IncompatibleOutputError, load_meta
from pybones.server.schemas import (
    ID_PATTERN,
    MAX_ID_LENGTH,
    DatasetInfo,
    DatasetKind,
    NetworkView,
    SegmentationSummary,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pybones.config.quantification import QuantificationFiles
    from pybones.config.schema import IOConfig
    from pybones.config.server import LibraryConfig
    from pybones.segmentation.labels import Label
    from pybones.segmentation.results import SegmentationMeta

_ID_RE = re.compile(ID_PATTERN)
#: Appended to a cached mesh's file name to name the record of its settings.
MESH_SETTINGS_SUFFIX = ".json"

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
        network_files: File names of the quantification outputs.
        mesh_file_template: Name of a cached mesh inside its segmentation (``{label}``,
            ``{format}``).
        shapes: Shape of each stack already read, with the modification time and size it
            was read at (describing a stack decodes a slice: listings happen after every job).
    """

    data_dir: Path
    output_dir: Path
    io: "IOConfig"
    network_files: "QuantificationFiles"
    mesh_file_template: str
    shapes: "dict[Path, tuple[int, int, tuple[int, int, int] | None]]" = field(
        default_factory=dict, compare=False
    )

    @classmethod
    def from_config(
        cls, config: "LibraryConfig", io: "IOConfig", network_files: "QuantificationFiles"
    ) -> "Library":
        """Build the library, resolving relative folders against the working directory.

        Args:
            config: Folders and mesh file name.
            io: File names of the segmentation outputs.
            network_files: File names of the quantification outputs.

        Returns:
            The library.
        """
        return cls(
            data_dir=Path(config.data_dir).resolve(),
            output_dir=Path(config.output_dir).resolve(),
            io=io,
            network_files=network_files,
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
            shape = self._shape(path)
            if shape is None:
                continue
            depth, height, width = shape
            found.append(
                DatasetInfo(id=path.name, kind=kind, slices=depth, height=height, width=width)
            )
        return found

    def _shape(self, path: Path) -> tuple[int, int, int] | None:
        """Read the shape of a stack, again only if the file or folder changed since.

        Args:
            path: Directory or zip archive.

        Returns:
            ``(depth, height, width)``, or ``None`` if it holds no readable slice.
        """
        stat = path.stat()
        known = self.shapes.get(path)
        if known is not None and known[:2] == (stat.st_mtime_ns, stat.st_size):
            return known[2]
        shape: tuple[int, int, int] | None = None
        try:
            with SliceStack(path, self.io) as stack:
                shape = stack.shape
        except (OSError, ValueError) as exc:
            logger.debug("Not a stack: %s (%s)", path, exc)
        self.shapes[path] = (stat.st_mtime_ns, stat.st_size, shape)
        return shape

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

    def mesh_path(self, segmentation_id: str, label: "Label", mesh_format: MeshFormat) -> Path:
        """Locate the cached mesh of a label in one format (it may not exist yet).

        Args:
            segmentation_id: Segmentation id.
            label: Foreground label.
            mesh_format: File format.

        Returns:
            Path of its mesh file.
        """
        name = self.mesh_file_template.format(label=label.key, format=mesh_format.value)
        return self.segmentation_path(segmentation_id) / name

    def mesh_settings_path(
        self, segmentation_id: str, label: "Label", mesh_format: MeshFormat
    ) -> Path:
        """Locate the record of the settings a cached mesh was built with (next to it).

        Args:
            segmentation_id: Segmentation id.
            label: Foreground label.
            mesh_format: File format.

        Returns:
            Path of the record: the mesh's file name plus :data:`MESH_SETTINGS_SUFFIX`.
        """
        mesh = self.mesh_path(segmentation_id, label, mesh_format)
        return mesh.with_name(mesh.name + MESH_SETTINGS_SUFFIX)

    def mesh_settings(
        self, segmentation_id: str, label: "Label", mesh_format: MeshFormat
    ) -> MeshRecord:
        """Read the settings a cached mesh was built with.

        Args:
            segmentation_id: Segmentation id.
            label: Foreground label.
            mesh_format: File format.

        Returns:
            The record.

        Raises:
            NotFoundError: If the mesh has no record (not built, or built before 0.8.1).
            IncompatibleOutputError: If the record is damaged or in another format.
        """
        path = self.mesh_settings_path(segmentation_id, label, mesh_format)
        if not path.is_file():
            raise NotFoundError(f"no record of the {label.key} {mesh_format.value} mesh settings")
        try:  # file content: damaged, or written by another version's settings format
            return MeshRecord.model_validate_json(path.read_text())
        except ValidationError as exc:
            what = f"the {label.key} mesh settings of {segmentation_id!r}"
            raise IncompatibleOutputError(f"{what} cannot be read: build it again") from exc

    def summary(self, segmentation_id: str) -> SegmentationSummary:
        """Describe a segmentation.

        Args:
            segmentation_id: Segmentation id.

        Returns:
            Its summary.
        """
        meta = self.meta(segmentation_id)
        meshes = {
            label.key: [
                fmt.value
                for fmt in MeshFormat
                if self.mesh_path(segmentation_id, label, fmt).is_file()
            ]
            for label in FOREGROUND_LABELS
        }
        meshes = {key: formats for key, formats in meshes.items() if formats}
        folder = self.segmentation_path(segmentation_id)
        return SegmentationSummary(
            id=segmentation_id,
            source=meta.source,
            pybones_version=meta.pybones_version,
            shape_zyx=meta.shape_zyx,
            voxel_size_um=meta.config.acquisition.voxel_size_um,
            label_fractions=meta.label_fractions,
            meshes=meshes,
            quantified=(folder / self.network_files.network_file).is_file(),
        )

    def _quantification_file(self, segmentation_id: str, name: str) -> Path:
        """Locate a file of a segmentation's quantification.

        Args:
            segmentation_id: Segmentation id.
            name: File name (see :attr:`network_files`).

        Returns:
            Its path.

        Raises:
            NotFoundError: If the segmentation was not quantified.
        """
        path = self.segmentation_path(segmentation_id) / name
        if not path.is_file():
            raise NotFoundError(f"{segmentation_id!r} is not quantified yet")
        return path

    def network(self, segmentation_id: str) -> NetworkView:
        """Read the summary of a segmentation's vessel network, and the settings it used.

        Args:
            segmentation_id: Segmentation id.

        Returns:
            The summary and settings written by its last quantification.

        Raises:
            NotFoundError: If the segmentation was not quantified.
            IncompatibleOutputError: If the file is damaged or in another format.
        """
        path = self._quantification_file(segmentation_id, self.network_files.network_file)
        try:  # file content: damaged, or written by an older quantification
            record = json.loads(path.read_text())
            if not isinstance(record, dict):
                record = {}
            return NetworkView.model_validate(
                {key: record.get(key) for key in (SUMMARY_KEY, SETTINGS_KEY)}
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise IncompatibleOutputError(
                f"the network of {segmentation_id!r} cannot be read: quantify again"
            ) from exc

    def centreline_path(self, segmentation_id: str) -> Path:
        """Locate the centreline file of a segmentation.

        Args:
            segmentation_id: Segmentation id.

        Returns:
            Its path.

        Raises:
            NotFoundError: If the segmentation was not quantified.
        """
        return self._quantification_file(segmentation_id, self.network_files.centrelines_file)

    def centreline_points_um(self, segmentation_id: str) -> "NDArray[np.float32]":
        """Read the centreline voxels of a segmentation as points, for drawing.

        Args:
            segmentation_id: Segmentation id.

        Returns:
            ``(N, 3)`` voxel centres in µm, ``(x, y, z)`` order like the meshes.
        """
        path = self.centreline_path(segmentation_id)
        voxel = np.asarray(self.meta(segmentation_id).config.acquisition.voxel_size_um)
        with np.load(path) as data:
            points = data[POINTS_KEY]
        return (points * voxel)[:, ::-1].astype(np.float32)

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
