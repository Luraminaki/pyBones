"""What the report figures read: settings, the real stack, the extracts and the reference runs.

Past versions are only read through their extracts (see :mod:`tools.report.extracts`).
The reference version (today's code) is also read from its full runs, which
``tools.report history`` always regenerates.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from pybones.io.slices import SliceStack
from tools.report.drawing import write_image
from tools.report.extracts import extract_path, load_extract
from tools.report.history import read_history
from tools.report.runs import LABELS_FILE, META_FILE, REAL_RUN, run_dir

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.config.schema import PyBonesConfig
    from tools.phantom.settings import PhantomSettings
    from tools.report.extracts import Extract
    from tools.report.history import Row
    from tools.report.settings import ReportSettings

_EXTRACT_CACHE_SIZE = 32


@lru_cache(maxsize=_EXTRACT_CACHE_SIZE)
def _cached_extract(path: "Path") -> "Extract":
    """Load an extract once per process (figures read the same ones repeatedly).

    Args:
        path: Extract file.

    Returns:
        The extract.
    """
    return load_extract(path)


@dataclass(frozen=True, slots=True)
class Context:
    """Everything the figure functions need.

    Attributes:
        settings: Report settings.
        config: pyBones configuration (rendering, voxel size, file names).
        phantom_settings: Phantom settings (files, scoring bands).
        repo: Repository root.
    """

    settings: "ReportSettings"
    config: "PyBonesConfig"
    phantom_settings: "PhantomSettings"
    repo: "Path"

    @property
    def figures(self) -> "Path":
        """Directory of the report figures."""
        return self.repo / self.settings.figures_dir

    @property
    def data(self) -> "Path":
        """Directory of the report plot data and values."""
        return self.repo / self.settings.data_dir

    @property
    def screenshots(self) -> "Path":
        """Directory of the README screenshots."""
        return self.repo / self.settings.screenshots_dir

    def save(self, directory: "Path", stem: str, image: "NDArray[np.uint8]") -> None:
        """Write an image in the configured format (JPEG: ~5x smaller than PNG for CT).

        Args:
            directory: Output directory (created if needed).
            stem: File name without extension.
            image: BGR or grayscale image.
        """
        fig = self.settings.figures
        write_image(directory / f"{stem}{fig.image_extension}", image, fig.jpeg_quality)

    def extract(self, key: str) -> "Extract":
        """Read the extract of a version (or of the 2018 masks, under the legacy key).

        Args:
            key: Version key or legacy key.

        Returns:
            The extract.
        """
        return _cached_extract(extract_path(self.settings, self.repo, key))

    def slice_labels(self, key: str, z: int) -> "NDArray[np.uint8]":
        """Read the labels of one slice of the real stack, for a version or the 2018 masks.

        Args:
            key: Version key, or the legacy key for the 2018 Matlab masks.
            z: Slice index (one of the extracted slices).

        Returns:
            Label image of the slice.
        """
        return self.extract(key).labels(z)

    def reference_labels(self, run: str = REAL_RUN) -> "NDArray[np.uint8]":
        """Memory-map a full label volume of the reference version (today's code).

        Args:
            run: :data:`~tools.report.runs.REAL_RUN` or :data:`~tools.report.runs.PHANTOM_RUN`.

        Returns:
            The read-only label volume.
        """
        key = self.settings.figures.reference_key
        return np.load(run_dir(self.settings, self.repo, key, run) / LABELS_FILE, mmap_mode="r")

    def reference_file(self, name: str) -> "Path":
        """Locate a file of the reference version's real-stack run.

        Args:
            name: File name (metadata, stats).

        Returns:
            Path of the file.
        """
        key = self.settings.figures.reference_key
        return run_dir(self.settings, self.repo, key, REAL_RUN) / name

    def reference_json(self, name: str) -> dict[str, Any]:
        """Read a JSON file of the reference version's real-stack run.

        Args:
            name: File name.

        Returns:
            The decoded object.
        """
        return json.loads(self.reference_file(name).read_text())

    @property
    def version(self) -> str:
        """Version of pyBones that produced the reference run (today's code)."""
        return str(self.reference_json(META_FILE)["pybones_version"])

    def title(self, template: str) -> str:
        """Fill the ``{version}`` placeholder of a configured title.

        Args:
            template: Title from the settings, e.g. ``"today ({version})"``.

        Returns:
            The title, with today's version.
        """
        return template.format(version=self.version)

    def raw_slice(self, z: int) -> "NDArray[np.uint16]":
        """Read a raw slice of the real dataset.

        Args:
            z: Slice index.

        Returns:
            The slice.
        """
        with SliceStack(self.repo / self.settings.dataset, self.config.io) as stack:
            return stack.read(z)

    def history(self) -> "dict[str, Row]":
        """Read the history CSV written by ``tools.report history``.

        Returns:
            Version key -> row (column -> text).

        Raises:
            FileNotFoundError: If the history has not been written yet.
        """
        path = self.data / self.settings.history_csv
        rows = read_history(path)
        if not rows:
            raise FileNotFoundError(f"{path} is missing: run `python -m tools.report history`")
        return rows
