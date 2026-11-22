"""Whole-stack segmentation: thresholds -> per-slice classification -> 3D refinement.

Memory stays bounded whatever the stack size: slices are decoded and
classified a few at a time, smoothed along z with a sliding window and
streamed to disk; the 3D refinement then works on memory-mapped volumes.

Output directory layout (names from the ``io`` config section)::

    out_dir/
        labels.npy   # uint8 (depth, height, width), values from Label
        meta.json    # source, thresholds, statistics, timings, full config
"""

import contextlib
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from pybones import __version__
from pybones.io.slices import SliceStack
from pybones.io.volume_store import LabelVolumeWriter
from pybones.segmentation.bursts import iter_burst_cores
from pybones.segmentation.classifier import (
    ClassifiedSlice,
    ambiguous_candidates,
    estimate_thresholds,
    extract_features,
    finalize_slice,
    smooth,
)
from pybones.segmentation.labels import FINAL_LABELS, Label
from pybones.segmentation.refinement import refine_volume
from pybones.segmentation.results import (
    SegmentationMeta,
    labels_path,
    save_meta,
    scratch_path,
)
from pybones.segmentation.scales import resolve_pixel_params, resolve_tubularity
from pybones.segmentation.stages import (
    STAGES_2D,
    STAGES_3D,
    CancelledError,
    Stage,
    no_progress,
)
from pybones.segmentation.tubularity import quantize_tubularity, slice_tubularity
from pybones.segmentation.zsmoothing import z_majority
from pybones.utils.parallel import ordered_map
from pybones.utils.windows import sliding_windows

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from pybones.config.schema import PipelineConfig, PyBonesConfig
    from pybones.segmentation.bursts import SliceBursts
    from pybones.segmentation.classifier import SliceFeatures, Thresholds
    from pybones.segmentation.scales import PixelParams
    from pybones.segmentation.stages import ProgressCallback
    from pybones.segmentation.tubularity import TubularityParams

logger = logging.getLogger(__name__)

#: Scratch volume holding the quantised tubularity maps for the refinement.
TUBULARITY_SCRATCH = "tubularity"

#: Keys of ``meta.json``'s ``timing_s`` besides the stage names.
REFINEMENT_TIMING = "refinement"
TOTAL_TIMING = "total"


@dataclass(frozen=True, slots=True)
class _Prepared:
    """A decoded slice ready for the per-slice features.

    Attributes:
        smoothed: Smoothed slice.
        tube_input: Raw slice as float32 for the Hessian, if the tubularity is computed.
    """

    smoothed: "NDArray[np.uint16]"
    tube_input: "NDArray[np.float32] | None"

    def require_tube_input(self) -> "NDArray[np.float32]":
        """Return the Hessian input, which exists whenever the tubularity is computed.

        Returns:
            The raw slice as float32.

        Raises:
            ValueError: If the slice was prepared without it.
        """
        if self.tube_input is None:
            raise ValueError("slice prepared without its tubularity input")
        return self.tube_input


def iter_classified(
    stack: SliceStack,
    thresholds: "Thresholds",
    params: "PixelParams",
    pipeline: "PipelineConfig",
    tube_params: "TubularityParams | None" = None,
) -> "Iterator[ClassifiedSlice]":
    """Classify every slice, yielding label images (and tubularity maps) in z order.

    1. decode + smooth (threaded), then per-slice features from a z window
       (3D vessel edges and, if ``tube_params`` is given, 3D tubularity; threaded);
    2. burst cores from the bright-pixel density in a 3D box (streamed along z);
    3. burst regions + bone / ambiguous split (threaded).

    Threads pay off because OpenCV, zlib and most numpy kernels release the GIL
    (about 2x faster than sequential on the reference dataset); ``workers = 1``
    runs sequentially. Memory is bounded by the in-flight limit and the
    ``window_z`` buffer.

    Args:
        stack: Source slices.
        thresholds: Global thresholds.
        params: Segmentation parameters in pixel units.
        pipeline: Parallelism settings.
        tube_params: Tubularity settings; ``None`` skips the tubularity maps.

    Yields:
        One :class:`ClassifiedSlice` per slice.
    """
    workers = min(pipeline.workers, os.cpu_count() or 1)
    extra = pipeline.extra_in_flight

    edge_half = params.edge_half_z
    tube_half = tube_params.half_z if tube_params is not None else 0
    half = max(edge_half, tube_half)

    def prepare(z: int) -> _Prepared:
        """Decode and smooth one slice (and keep its float copy for the Hessian).

        Args:
            z: Slice index.

        Returns:
            The prepared slice.
        """
        raw = stack.read(z)
        tube_input = raw.astype(np.float32) if tube_params is not None else None
        return _Prepared(smooth(raw, params.smoothing_sigma), tube_input)

    def features_of(window: "list[_Prepared]") -> "SliceFeatures":
        """Compute the features of the centre slice of a z window.

        Args:
            window: ``2 * half + 1`` prepared slices centred on the current one.

        Returns:
            The features of the centre slice, with its tubularity if computed.
        """
        smoothed = [p.smoothed for p in window[half - edge_half : half + edge_half + 1]]
        features = extract_features(smoothed, edge_half, thresholds, params)
        if tube_params is None:
            return features
        floats = [p.require_tube_input() for p in window[half - tube_half : half + tube_half + 1]]
        candidates = ambiguous_candidates(features.masks, params)
        tubularity = quantize_tubularity(slice_tubularity(floats, candidates, tube_params))
        return replace(features, masks=replace(features.masks, tubularity=tubularity))

    def finalize(item: "SliceBursts") -> ClassifiedSlice:
        """Label a slice once its burst core and boundary are known.

        Args:
            item: Masks of the slice and its burst inputs.

        Returns:
            The label image and the tubularity map of the slice.
        """
        labels = finalize_slice(item.masks, item.core, item.boundary, params)
        return ClassifiedSlice(labels, item.masks.tubularity)

    prepared = ordered_map(prepare, range(len(stack)), workers, extra)
    windows = sliding_windows(prepared, len(stack), half)
    features = ordered_map(features_of, windows, workers, extra)
    yield from ordered_map(finalize, iter_burst_cores(features, params), workers, extra)


def _labels_storing_tubularity(
    classified: "Iterable[ClassifiedSlice]", writer: "LabelVolumeWriter | None"
) -> "Iterator[NDArray[np.uint8]]":
    """Yield the labels, streaming the tubularity maps to ``writer`` (if any).

    Args:
        classified: Classified slices in z order.
        writer: Scratch volume of the tubularity maps; ``None`` when not computed.

    Yields:
        The label image of each slice.

    Raises:
        ValueError: If ``writer`` is given but a slice has no tubularity map.
    """
    for item in classified:
        if writer is not None:
            if item.tubularity is None:
                raise ValueError("slice classified without its tubularity map")
            writer.write(item.tubularity)
        yield item.labels


def _ambiguous_as_bone(slices: "Iterable[NDArray[np.uint8]]") -> "Iterator[NDArray[np.uint8]]":
    """Relabel ambiguous blobs as bone (fallback without the 3D refinement).

    Undecided blobs keep their intensity class. Slices are modified in place.

    Args:
        slices: Label images in z order.

    Yields:
        Each label image without ``AMBIGUOUS`` pixels.
    """
    for sl in slices:
        sl[sl == Label.AMBIGUOUS] = Label.BONE
        yield sl


def _write(
    path: "Path",
    shape: tuple[int, int, int],
    slices: "Iterable[NDArray[np.uint8]]",
    progress: "ProgressCallback",
) -> "NDArray[np.int64]":
    """Stream ``slices`` to a label volume.

    Args:
        path: Output ``.npy`` file.
        shape: ``(depth, height, width)`` of the volume.
        slices: Label images in z order.
        progress: Progress callback (reported as :attr:`Stage.CLASSIFY`).

    Returns:
        Voxel count of every uint8 value.
    """
    with LabelVolumeWriter(path, shape) as writer:
        for z, sl in enumerate(slices):
            writer.write(sl)
            progress(Stage.CLASSIFY, z + 1, shape[0])
    return writer.counts


def planned_stages(config: "PyBonesConfig") -> tuple[Stage, ...]:
    """List the stages a run will report, in order (for progress trackers).

    Args:
        config: Configuration (whether the 3D refinement runs).

    Returns:
        :data:`STAGES_3D` with the refinement, :data:`STAGES_2D` without.
    """
    return STAGES_3D if config.refinement.enabled else STAGES_2D


def _estimate_stack_thresholds(
    stack: SliceStack, config: "PyBonesConfig", params: "PixelParams", progress: "ProgressCallback"
) -> "Thresholds":
    """Estimate the global thresholds from every ``histogram_stride``-th slice.

    Args:
        stack: Source slices.
        config: Configuration (stride, histogram bins).
        params: Segmentation parameters in pixel units.
        progress: Progress callback (reported as :attr:`Stage.THRESHOLDS`).

    Returns:
        The thresholds.
    """
    indices = range(0, len(stack), config.thresholds.histogram_stride)

    def sampled() -> "Iterator[NDArray[np.uint16]]":
        """Read the sampled slices, reporting each one.

        Yields:
            Each sampled slice.
        """
        for done, z in enumerate(indices, start=1):
            yield stack.read(z)
            progress(Stage.THRESHOLDS, done, len(indices))

    thresholds = estimate_thresholds(sampled(), params, config.thresholds.histogram_bins)
    logger.info("Thresholds from %d slices: %s", len(indices), thresholds)
    return thresholds


def _remove_files(paths: "Iterable[Path | None]") -> None:
    """Delete scratch files that exist (``None`` entries are skipped).

    Args:
        paths: Files to delete.
    """
    for path in paths:
        if path is not None and path.exists():
            path.unlink()
            logger.debug("Removed %s", path)


def run_segmentation(
    source: "Path",
    out_dir: "Path",
    config: "PyBonesConfig",
    progress: "ProgressCallback" = no_progress,
) -> SegmentationMeta:
    """Segment a full stack into ``out_dir`` (label volume + ``meta.json``).

    Scratch volumes are removed whatever happens; on failure or cancellation
    (:class:`~pybones.segmentation.stages.CancelledError` raised by ``progress``),
    no partial label volume is left behind either, nor ``out_dir`` if this run created it.

    Args:
        source: Directory or zip archive of slices.
        out_dir: Output directory (created if needed).
        config: Configuration.
        progress: Called with ``(stage, done, total)`` after each slice (see
            :func:`planned_stages` for the stages it will see).

    Returns:
        The metadata written next to the volume.
    """
    created = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    io_cfg = config.io
    params = resolve_pixel_params(config)
    final_path = labels_path(out_dir, io_cfg)
    refine = config.refinement.enabled
    use_tubes = refine and config.tubularity.enabled
    # With the refinement, the classified volume is a scratch file it reads back.
    classified_path = scratch_path(out_dir, io_cfg, Stage.CLASSIFY) if refine else final_path
    resolved_path = scratch_path(out_dir, io_cfg, Stage.RESOLVE)
    tubularity_path = scratch_path(out_dir, io_cfg, TUBULARITY_SCRATCH) if use_tubes else None
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    logger.info("Segmenting %s into %s (refinement: %s)", source, out_dir, refine)

    failed = False
    try:
        with SliceStack(source, io_cfg) as stack:
            shape = stack.shape
            logger.info("Stack: %d slices of %d x %d", *shape)
            t0 = time.perf_counter()
            thresholds = _estimate_stack_thresholds(stack, config, params, progress)
            timings[Stage.THRESHOLDS] = time.perf_counter() - t0

            t0 = time.perf_counter()
            tube_params = resolve_tubularity(config) if use_tubes else None
            classified = iter_classified(stack, thresholds, params, config.pipeline, tube_params)
            with ExitStack() as resources:
                tube_writer = (
                    resources.enter_context(LabelVolumeWriter(tubularity_path, shape))
                    if tubularity_path is not None
                    else None
                )
                slices = z_majority(
                    _labels_storing_tubularity(classified, tube_writer), params.z_window
                )
                first_pass = slices if refine else _ambiguous_as_bone(slices)
                counts = _write(classified_path, shape, first_pass, progress)
            timings[Stage.CLASSIFY] = time.perf_counter() - t0
            logger.info("Classified %d slices in %.1f s", shape[0], timings[Stage.CLASSIFY])

        stats = None
        if refine:
            t0 = time.perf_counter()
            stats, counts = refine_volume(
                classified_path, resolved_path, final_path, params, progress, tubularity_path
            )
            timings[REFINEMENT_TIMING] = time.perf_counter() - t0
            logger.info("Refined in %.1f s: %s", timings[REFINEMENT_TIMING], stats)
    except BaseException as exc:
        failed = True
        if isinstance(exc, (CancelledError, KeyboardInterrupt)):
            logger.info("Segmentation of %s cancelled", source)
        else:
            logger.exception("Segmentation of %s failed", source)
        _remove_files((final_path,))
        raise
    finally:
        _remove_files((classified_path if refine else None, resolved_path, tubularity_path))
        if failed and created:  # last, once the scratch files inside it are gone
            with contextlib.suppress(OSError):  # only if empty: never someone else's files
                out_dir.rmdir()

    total = int(counts.sum())
    meta = SegmentationMeta(
        pybones_version=__version__,
        source=str(source.resolve()),
        shape_zyx=shape,
        thresholds=thresholds,
        label_fractions={lab.key: int(counts[lab]) / total for lab in FINAL_LABELS},
        refinement=stats,
        timing_s={**timings, TOTAL_TIMING: time.perf_counter() - t_start},
        config=config,
    )
    meta_path = save_meta(meta, out_dir, io_cfg)
    logger.info("Wrote %s and %s in %.1f s", final_path, meta_path, meta.timing_s[TOTAL_TIMING])
    return meta
