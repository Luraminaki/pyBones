"""Configuration of the report tool (``config.json`` validated by pydantic)."""

import json
from pathlib import Path
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("config.json")

NonEmpty = Annotated[str, Field(min_length=1)]
#: Version key: also a directory name and a CSV / LaTeX identifier.
Key = Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]


class _Section(BaseModel):
    """Base of every section: unknown keys are rejected and instances are immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class Version(_Section):
    """A past state of the project to benchmark.

    Attributes:
        key: Short identifier (directory and CSV key).
        commit: Git revision (hash, tag or the working tree's ``HEAD``).
        title: What this version introduced.
        real: Also segment the real dataset (needed for before/after figures).
    """

    key: Key
    commit: NonEmpty
    title: NonEmpty
    real: bool


class Crop(_Section):
    """Square region ``[y, y + size) x [x, x + size)`` of slice ``z``.

    Attributes:
        z: Slice index.
        y: Top row.
        x: Left column.
        size: Side in pixels.
    """

    z: int = Field(ge=0)
    y: int = Field(ge=0)
    x: int = Field(ge=0)
    size: int = Field(ge=8, le=4096)

    @property
    def window(self) -> tuple[slice, slice]:
        """In-plane slices of the crop."""
        return slice(self.y, self.y + self.size), slice(self.x, self.x + self.size)


class Profile(_Section):
    """Horizontal intensity profile of row ``y`` of slice ``z``, columns ``[x0, x1)``.

    Attributes:
        z: Slice index.
        y: Row.
        x0: First column.
        x1: Column after the last one.
        keys: Versions whose vessel labels are tabulated along the profile.
    """

    z: int = Field(ge=0)
    y: int = Field(ge=0)
    x0: int = Field(ge=0)
    x1: int = Field(ge=1)
    keys: tuple[Key, ...] = Field(min_length=1)


class Comparison(_Section):
    """The same crop segmented by several versions.

    Attributes:
        crop: Name of the crop (a key of ``figures.crops``).
        keys: Versions shown after the raw crop (or the legacy key, for the 2018 masks).
    """

    crop: NonEmpty
    keys: tuple[Key, ...] = Field(min_length=1)


class TitledComparison(Comparison):
    """A comparison composed into one image with a title above each panel.

    Attributes:
        titles: Title of the raw panel, then of each version panel.
    """

    titles: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def _one_title_per_panel(self) -> Self:
        """Check that every panel has a title.

        Returns:
            The validated comparison.

        Raises:
            ValueError: If the titles do not match the raw panel plus the versions.
        """
        if len(self.titles) != len(self.keys) + 1:
            raise ValueError("titles must name the raw panel, then each key")
        return self


class FigureSettings(_Section):
    """What the figures show and how they are laid out.

    Attributes:
        image_extension: Image format of the figures (``.jpg`` or ``.png``).
        jpeg_quality: JPEG quality, used when ``image_extension`` is ``.jpg``.
        reference_key: Version shown as today's segmentation.
        legacy_key: Pseudo-version standing for the 2018 Matlab masks.
        overview_slice: Slice of the full-slice overview.
        overview_scale: Downscaling of the full-slice images.
        scale_bar_um: Length of the scale bars.
        panel_gap_px: Gap between composed panels.
        title_height_px: Height of the title strip of composed panels.
        title_font_scale: OpenCV font scale of titles and scale-bar labels.
        crops: Named crops of the real stack.
        comparisons: Figure name -> crop and versions; each panel is saved as
            ``<figure>_<key>`` (``<figure>_raw`` for the raw crop).
        evolution_screenshot: Composed README screenshot of the project's evolution.
        tubularity_crop: Crop of the tube-vs-plate figure.
        phantom_crop: Crop of the phantom figure.
        phantom_titles: Titles of the phantom screenshot panels (raw, truth, segmentation).
        profile: Intensity profile across a vessel.
        burst_keys: Versions whose burst fraction per slice is plotted.
        histogram_max_intensity: Upper end of the intensity histogram.
        histogram_bins: Bins of the intensity histogram.
        histogram_stride: Use one slice every N for the histogram.
    """

    image_extension: Annotated[str, Field(pattern=r"^\.(jpg|png)$")]
    jpeg_quality: int = Field(ge=1, le=100)
    reference_key: Key
    legacy_key: Key
    overview_slice: int = Field(ge=0)
    overview_scale: float = Field(gt=0.0, le=1.0)
    scale_bar_um: float = Field(gt=0.0)
    panel_gap_px: int = Field(ge=0, le=100)
    title_height_px: int = Field(ge=0, le=200)
    title_font_scale: float = Field(gt=0.0, le=10.0)
    crops: dict[NonEmpty, Crop]
    comparisons: dict[Key, Comparison]
    evolution_screenshot: TitledComparison
    tubularity_crop: Crop
    phantom_crop: Crop
    phantom_titles: tuple[NonEmpty, NonEmpty, NonEmpty]
    profile: Profile
    burst_keys: tuple[Key, ...] = Field(min_length=1)
    histogram_max_intensity: int = Field(ge=1, le=65535)
    histogram_bins: int = Field(ge=8, le=65536)
    histogram_stride: int = Field(ge=1)

    @model_validator(mode="after")
    def _known_crops(self) -> Self:
        """Check that every comparison names a configured crop.

        Returns:
            The validated settings.

        Raises:
            ValueError: If a comparison names an unknown crop.
        """
        for comparison in (*self.comparisons.values(), self.evolution_screenshot):
            if comparison.crop not in self.crops:
                raise ValueError(f"unknown crop {comparison.crop!r}")
        return self


class NetworkReport(_Section):
    """The vascular network measured for the report: real sample and branching phantom.

    Attributes:
        phantom_settings: Settings of the branching phantom (``tools/phantom`` format).
        work_dir: Where that phantom and its segmentation are generated, afresh each time.
        tolerance_um: Distance under which a centreline voxel and a true centreline agree.
        junction_tolerance_um: Distance under which a measured fork matches a true one (plus
            the parent vessel's radius).
    """

    phantom_settings: NonEmpty
    work_dir: NonEmpty
    tolerance_um: float = Field(gt=0.0, le=1000.0)
    junction_tolerance_um: float = Field(gt=0.0, le=1000.0)


class ReportSettings(_Section):
    """Root of ``tools/report/config.json`` (paths relative to the repository root).

    Attributes:
        uv_executable: uv command used to run past versions.
        cli_name: Console script of pyBones.
        worktrees_dir: Where past versions are checked out.
        runs_dir: Where their outputs are stored (``<key>/<run>/``).
        phantom_dir: Phantom generated by ``tools.phantom``.
        dataset: Real µCT stack.
        legacy_masks: 2018 Matlab masks of the real stack (only needed to rebuild their extract).
        data_dir: Plot data, tables and values of the report.
        extracts_dir: Versioned extracts of every version's segmentation (see
            :mod:`tools.report.extracts`).
        figures_dir: Figures of the report.
        screenshots_dir: README screenshots.
        readme: README whose score table is kept up to date.
        history_csv: Score table, in ``data_dir``.
        versions: Versions to benchmark, oldest first.
        figures: Figure settings.
        network: The vascular network section of the report.
    """

    uv_executable: NonEmpty
    cli_name: NonEmpty
    worktrees_dir: NonEmpty
    runs_dir: NonEmpty
    phantom_dir: NonEmpty
    dataset: NonEmpty
    legacy_masks: NonEmpty
    data_dir: NonEmpty
    extracts_dir: NonEmpty
    figures_dir: NonEmpty
    screenshots_dir: NonEmpty
    readme: NonEmpty
    history_csv: Annotated[str, Field(pattern=r"\.csv$")]
    versions: tuple[Version, ...] = Field(min_length=1)
    figures: FigureSettings
    network: NetworkReport

    @model_validator(mode="after")
    def _consistent_keys(self) -> Self:
        """Check that version keys are unique and that figures only name known versions.

        Returns:
            The validated settings.

        Raises:
            ValueError: If a key is duplicated or a figure names an unknown version.
        """
        keys = [v.key for v in self.versions]
        if len(keys) != len(set(keys)):
            raise ValueError("version keys must be unique")
        fig = self.figures
        if fig.reference_key not in keys:
            raise ValueError(f"reference_key {fig.reference_key!r} is not a version key")
        if fig.legacy_key in keys:
            raise ValueError(f"legacy_key {fig.legacy_key!r} must differ from the version keys")
        shown = (
            *(k for c in fig.comparisons.values() for k in c.keys),
            *fig.evolution_screenshot.keys,
            *fig.profile.keys,
            *fig.burst_keys,
        )
        unknown = sorted({k for k in shown if k not in keys and k != fig.legacy_key})
        if unknown:
            raise ValueError(f"figures name unknown versions: {unknown}")
        return self

    def version(self, key: str) -> Version:
        """Find a version by key.

        Args:
            key: Version key.

        Returns:
            The version.

        Raises:
            KeyError: If no version has this key.
        """
        for v in self.versions:
            if v.key == key:
                return v
        raise KeyError(key)


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> ReportSettings:
    """Read and validate the report tool configuration.

    Args:
        path: JSON settings file (default: the one next to this module).

    Returns:
        The validated settings.
    """
    return ReportSettings.model_validate(json.loads(path.read_text()))
