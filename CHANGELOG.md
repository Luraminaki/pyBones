# Changelog

All notable changes to this project are documented here. Versioning loosely
follows [Semantic Versioning](https://semver.org/); the project is still
`-alpha`, so breaking changes (configuration keys, outputs) can happen between
minor versions.

The current version lives in [pyproject.toml](pyproject.toml) and is read
from package metadata at runtime; every segmentation records it in its
`meta.json`. The [report](docs/report/) measures what each step below changed.

## Unreleased

- MIT License for the code; README section on the data (not covered by the license, shared for
  research and education, removable on request).
- Jobs push their status: `JobManager.events(job_id)` yields every state change and throttled
  progress, ready for Server-Sent Events (no polling); bounded per-subscriber queues drop the oldest
  snapshots first (`jobs.event_interval_s`, `jobs.event_queue_size`).

## 0.5.0-alpha

- Programmatic interface `pybones.api`: one function per operation (segment, preview, mesh, export,
  compare), which the command line now wraps, and a `JobManager` running long operations in the
  background (status as JSON-ready models, progress, cancellation, results; `jobs` section of the
  configuration).
- Golden regression test: a small fixed phantom must keep its recorded scores (a deliberate change
  refreshes the baseline with `PYBONES_UPDATE_GOLDEN=1`).
- Annotation page: wand (fills the connected pixels of similar brightness), outline-only overlay,
  a note per crop (saved in `notes/`, flagged in the crop list).
- Report: labels of today's version follow the package version (`{version}` in the report tool's
  titles, `\pbValueVersion` in the text); the README's score table is written by
  `tools.report figures` between two markers instead of being typed by hand.

## 0.4.0-alpha

- Logging: one rotating log file per process (`logs/<program>-<pid>.log`), records with time,
  process, thread, module and function; the oldest runs' files are pruned at start-up
  (`logging` section of the configuration). Library modules only create loggers.
- Progress: `ProgressTracker` gives a thread-safe snapshot of a run (stage k of n, overall fraction
  weighted by measured stage durations, elapsed and remaining time) for a future web interface,
  and can cancel it; a cancelled or failed run leaves no partial volume behind. The thresholds
  stage now reports progress too.
- The tools validate their command-line arguments with pydantic models, like `pybones`.
- Report tool split into small modules (runs, extracts, history, context, drawing, figures,
  latex). Each version's segmentation is kept as a small versioned extract
  (`docs/report/data/runs/`), so the report can be rebuilt and updated from a clone without the
  development history: versions whose commit is missing keep their archived scores. The 2018
  masks are replaced by their extract.
- Annotation page: zoom and pan, brightness and contrast, "paint over" protection, autosave.
- Shared code: format constants (`pybones.utils.constants`), per-label scores and their table
  (`pybones.analysis.metrics.LabelScores`); dead code removed.
- `pytest-cov` added; tool coverage raised (96 % overall).

## 0.3.2-alpha

- Code review pass, no change in results (the real stack segments voxel-identically, the report
  figures are byte-identical):
  - complete Google docstrings on every function and method, private ones included, and on the
    tests; `tests/test_conventions.py` enforces them, and that names are imported from the module
    that defines them;
  - basedpyright (`recommended`) added as a development tool and configured in `pyproject.toml`;
    the code type-checks with no diagnostic;
  - types instead of `None` checks: per-slice `SliceMasks` separated from the burst density,
    commands dispatched by an exhaustive `match`, OpenCV results narrowed with `typing.cast`;
  - `VoxelGeometry` moved to `segmentation/geometry.py` (import cycle removed); unused Frangi
    vesselness removed; one shared contrast stretch (`stretch_to_uint8`);
  - report tool: version keys, crops and panel titles of the figures moved to its validated
    `config.json`; CSV column names shared between the writer and the figures;
  - the tests import shared data from `tests.helpers` (no `sys.path` entry for `tests/`);
  - list-building loops turned into comprehensions; crop candidates tabulated with NumPy.
- Report: written in the first person; no personal names; every timed run records the machine it ran
  on (`tools/report/machine.py`), and the report prints it next to the timings.
- The 2018 masks and the 2018 images the report uses moved to `docs/report/2018/`: nothing depends on
  the 2018 code and documents any more.
- `.gitignore`: `data/` anchored to the root, so the report's generated data (`docs/report/data/`) is
  versioned again and the PDF can be rebuilt from a clone.

## 0.3.1-alpha

- Command-line arguments validated into one typed, immutable pydantic model per
  sub-command; `python -m pybones` works like the `pybones` command.
- Report restyled (formal layout) and compiled; `INSTALL.md` documents the full
  generation sequence.

## 0.3.0-alpha

- Every size is physical (µm, µm², µm³) and converted from a single
  `acquisition.voxel_size_um`; anisotropic voxels supported. Analytic elliptic
  kernels replace OpenCV's (spiky) disks; unused morphology removed.
- Synthetic validation phantom (`tools/phantom`) with exact ground truth,
  calibrated on the reference scan (intensities, noise, measured blur,
  burst sizes), and its scorer (per class, vessel recall by radius, clots).
- Vessel edges at a local half-maximum between the vessel and its surrounding
  material, sought in 3D; the sub-half-maximum halo is background, not bone.
  Vessels were 1.75x too thick before.
- 3D tube-vs-plate ratio (Hessian eigenvalues, streamed): plate-like ambiguous
  blobs can no longer become vessels.
- Burst boundaries at half the core density (no fixed closing / halo radius).
- Optimisation pass: 36 s / 3.3 GB -> 24 s / 1.6 GB on the reference stack.
- Documentation restructured (README, INSTALL, CHANGELOG) and a full report
  (`docs/report/`, `tools/report` to reproduce its measurements and figures).

## 0.2.0-alpha

- Package split into subpackages; every tunable value in a JSON configuration
  validated by pydantic (with bounds); type-only imports under `TYPE_CHECKING`,
  no lazy imports, enforced by Ruff.
- 3D connectivity refinement (streaming union-find): blobs that are ambiguous
  in 2D are decided by what they touch in 3D.
- Burst detection with a 3D bright-density box (2D united with guarded 3D).
- Taubin smoothing of the meshes (removes the voxel staircase).
- Local annotation tool (`tools/annotation`) to build a ground truth.

## 0.1.0-alpha

- First pure-Python port of the 2018 Matlab + PyQt5 project: global multi-Otsu thresholds, per-slice classification into
  bone / vessel / burst, streamed label volume, marching cubes + STL, CLI.
