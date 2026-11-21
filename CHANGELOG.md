# Changelog

All notable changes to this project are documented here. Versioning loosely
follows [Semantic Versioning](https://semver.org/); the project is still
`-alpha`, so breaking changes (configuration keys, outputs) can happen between
minor versions.

The current version lives in [pyproject.toml](pyproject.toml) and is read
from package metadata at runtime; every segmentation records it in its
`meta.json`. The [report](docs/report/) measures what each step below changed.

## Unreleased

- Modules split by responsibility, none over ~460 lines:
  - `quantification.graph` becomes `links` (touching voxels, staircase corners), `network` (types,
    nodes and branches), `pruning` and `graph` (`build_network`);
  - the phantom generator becomes `canvas`, `vessels` (walks and the true network) and `generator`;
  - the configuration schema becomes `config.fields` (shared field types), `config.quantification`,
    `config.server` and `config.schema`;
  - the job status types move to `api.job_status`;
  - the 3D viewer's mesh memory helpers move to `mesh_memory.js`, now tested with Node.
  Names are imported from their new modules (no re-exports).
- MIT License for the code; README section on the data (not covered by the license, shared for
  research and education, removable on request).
- Jobs push their status: `JobManager.events(job_id)` yields every state change and throttled
  progress, ready for Server-Sent Events (no polling); bounded per-subscriber queues drop the oldest
  snapshots first (`jobs.event_interval_s`, `jobs.event_queue_size`).
- Web API: `pybones serve` runs a local FastAPI server (`pybones.server`) listing the stacks of a data
  folder and the segmentations of an output folder, starting segmentations and meshes as jobs
  followed with Server-Sent Events, cancelling them, rendering slices (PNG) and serving meshes
  (binary STL). Clients address everything by id, never by path; errors use HTTP status codes. Its
  settings live in a separate `server.json`, so that deployment settings never enter `meta.json`.
  Interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`; `GET /api/server`
  describes the server (folders, limits) for the page. Meshes take the segmentation's voxel size and the server's mesh settings;
  asking for an existing mesh again rebuilds it.
- Mesh jobs: `JobManager.submit_mesh` and `write_mesh_file` (written under a temporary name, renamed
  once complete; a cancelled job leaves no file). New `mesh` stage; `pipeline.stage_weights` only
  needs the segmentation stages.
- `JobManager.events(job_id, heartbeat_s)` repeats the status of a quiet job, so a server notices a
  departed client.
- New dependencies: `fastapi`, `uvicorn`; `httpx2` for the tests.
- Messages of the `pybones` commands and of the tools go through the logger instead of `print`: the
  terminal shows them in a short `time level message` form (`logging.console_level` is now `INFO`),
  the log file in full. `pybones config` still writes its JSON on stdout, and the progress line still
  updates in place.
- A segmentation written by an older pyBones is refused with a clear message (both versions, and
  "segment the stack again") instead of a validation dump; the web server skips it when listing and
  answers 409 when it is asked for. Outputs are not kept compatible across versions while alpha.
- Configuration file names (`io.*_file`, `io.*_template`) now really refuse paths: a field's own
  pattern (its extension) used to replace the "no `/` or `\`" rule instead of adding to it.
- Web page at `http://127.0.0.1:8000` (`pybones serve`): segmentations and stacks, a form to segment
  a stack (with per-run settings), jobs followed live with progress, time left and cancel, a slice
  viewer with the labels overlaid, and a 3D viewer of the GLB meshes (three.js; bone see-through,
  meshes built on demand). Plain HTML/CSS/JavaScript modules, no build step; light and dark themes.
  three.js 0.187.0 (MIT) is vendored with a manifest of file hashes, checked by the tests, as is
  every import of every module. Checked in Firefox: segmenting, slices, all three meshes.
  *Segment again…* fills the form with the shown segmentation's stack and the settings it was made
  with (those differing from the defaults). The 3D view has a Rotate / Move switch for the left
  button, + / − zoom buttons, zoom towards the cursor and arrow keys, besides right-drag and the wheel.
- Mesh jobs report real progress: four weighted stages (`occupancy`, `surface`, `smoothing`,
  `writing`; `mesh.stage_weights`, measured on the sample) instead of one stage that sat at 0 % until
  the very end; they can now be cancelled in the middle. `pybones mesh` shows the same progress line
  as `pybones segment` and writes its file under a temporary name until complete.
- VS Code settings, launch configurations and recommended extensions are versioned in `.vscode/`
  (with a "serve" launch), with no OS-specific interpreter path: VS Code selects the workspace
  `.venv`, and basedpyright finds it through `pyproject.toml` on Linux, macOS and Windows.
- The dataset now lives in `data/` (`data/data_set.zip`), where the web server looks for stacks:
  commands, report tool, annotation manifest and documentation follow; `data/.gitkeep` keeps the
  folder in the repository, its content stays ignored.
- Review of the server and the page (memory and robustness):
  - event streams no longer hold a worker thread each: jobs push their statuses to the event loop
    (`JobManager.subscribe`), so many open streams cannot starve slice and list requests of threads
    (checked with 60 streams open during a run);
  - the browser keeps meshes in GPU memory only (the CPU copy three.js keeps is released once
    uploaded: ~175 MB less for the bone and vessels of the sample);
  - one slice request at a time, the latest slider position winning: dragging never piles renders
    up on the server; slices are served at most at full size (`api.max_preview_scale` 1: an
    enlarged slice would be built in server memory for nothing, 164 MB at the old maximum of 4);
  - job rows are updated in place (Cancel could miss a click while progress arrived), and a job is
    followed by one event stream however many parts of the page watch it;
  - the name of a new segmentation is checked under the same lock as the running jobs (a run
    finishing at that moment could have let a second one write into its folder);
  - stack descriptions are kept until the stack changes (each listing decoded a slice of every
    stack);
  - slices never accumulate in the browser: the page fetches each one with `cache: 'no-store'`,
    shows it through a temporary `blob:` URL revoked once decoded, and the server sends
    `Cache-Control: no-store` as well;
  - a rebuilt mesh is no longer hidden by the browser's cache: meshes are sent with
    `Cache-Control: no-cache` and an ETag following the file, an unchanged mesh is answered
    `304 Not Modified` (Starlette's `FileResponse` never does), a rebuilt one in full.
- Vascular quantification, first layer: its three 3D steps (filling the cavities of the vessel
  mask, skeletonisation, distance to the wall) are interchangeable backends behind small
  interfaces (Strategy pattern), chosen by name in the new `quantification` section of the
  configuration. Cavities: pyBones' streaming connectivity or SciPy (identical on the sample);
  skeleton: scikit-image; distances: per-slice OpenCV maps combined along z (0.4 s on the sample)
  or SciPy's exact 3D transform on tiles (39 s, for rectangular pixels and as a cross-check; both
  agree to 4e-6 µm). Filling the cavities first matters: without it, thick vessel blobs thinned
  into sheets (junction clusters up to 2,707 voxels; 72 with it).
- Vascular network (`pybones.quantification.graph`, `measures`, `pipeline`): the skeleton becomes a
  graph (26-neighbour links, staircase corners dropped without ever disconnecting, junction clusters
  as single nodes), pruned in µm (`quantification.pruning`: spurs shorter than a multiple of the
  junction's radius, isolated specks, short junction-to-junction branches contracted with their
  length kept). Ends within their radius plus `anchor_margin_um` of a stack face or a burst are
  anchored, their branches flagged truncated. Per branch: length, chord, tortuosity, radius
  (median, mean, min, max); for the stack: branches, junctions, ends by kind, loops, total length,
  length density, medians. `run_quantification` writes `network.json`, `branches.csv` and
  `centrelines.npz` into the segmentation folder (temporary names until complete, cancellable at
  each stage). On the sample: 17.6 s, 1.2 GB peak; 680 mm of vessels in 12,072 branches.
- Phantom network scoring (`tools/phantom/network_scoring.py`): centreline recall and precision,
  junctions matched / crossings / spurious, radius offset and error, total length. On the network
  phantom, segmented: length within 2 % of the truth, 24 of 54 visible bifurcations found, but
  82 spurious junctions (mostly vessel/bone contact bulges), radius offset −0.8 µm (not constant:
  thin vessels read thicker, thick ones thinner, so it is not corrected).
- Code audit (duplication, needless wrappers, reliability):
  - Jobs count their time from their start, not their submission: a mesh waiting behind another one
    no longer shows the wait as elapsed time nor an inflated time left. A job cancelled while queued
    keeps a fixed duration instead of one growing forever.
  - A failed or cancelled segmentation removes the output folder it created (any front end, the
    command line included), never one that existed before; a cancellation is logged without a
    traceback.
  - `try` only where the code cannot check first (I/O, network, outside input, other threads):
    - lookups check membership: `Label.from_key`, `MeshFormat.from_path`, the label parameter
      of the server;
    - the slice route checks `z` against the stack depth before rendering;
    - the job queues check for room instead of catching `Full`;
    - an unknown job raises `UnknownJobError` (a `LookupError`, no longer a `KeyError`), turned
      into a 404 by one handler of the application instead of a `try` in each route;
    - the page's `try` blocks wrap only their network call.
  - One helper writes the files that appear only once complete (`pybones.utils.files.complete_files`),
    for meshes and the quantification files alike.
  - Removed:
    - `api.operations.segment`: use `segmentation.pipeline.run_segmentation`;
    - `loader.default_config_text`;
    - the quantification registry's accessor functions: index `CAVITY_FILLERS` and the other
      tables directly;
    - `graph.PruningRules`: the graph takes `PruningConfig`;
    - `JobProgress`: a job status carries the `Progress` itself;
    - the µm/mm constants: now `UM_PER_MM` in `utils/constants.py`.
  - Web page:
    - a duration of 59.6 s or 119.6 s showed as "60 s" or "1 min 60 s";
    - JSDoc on every function;
    - helpers shared instead of copied: label list, storage access, request code, file names;
    - mesh colours built from the same CSS colour as the swatches (identical values).
  - JavaScript checks with Node.js (skipped without it, no npm): `node --check` on every module,
    `node:test` suites in `tests/js/`, and checks that the page's labels and finished job states
    match Python's. A test also rejects `var`, loose equality, and HTML or code built from strings
    (`innerHTML`, `eval`); the page used none of them.
- Phantom: vessels can branch (`vessels.branch_*`, off in the segmentation phantom, which is
  unchanged pixel for pixel), and the phantom records its true network (centrelines, radii,
  bifurcations) in `truth.npz`. `tools/phantom/network.json` describes a branching network phantom
  for validating the quantification. An older phantom asks to be generated again.
- New dependencies: SciPy (already required by scikit-image), `scipy-stubs` for the type checker.
- The streaming 3D connectivity relabels a slice with the same OpenCV call as its first pass
  (it used another function that happened to number components alike); results unchanged.
- The page read the voxel size's axes in the wrong order (`(z, y, x)` in the configuration):
  invisible on cubic voxels, wrong on others. INSTALL's dependency list now names FastAPI and
  uvicorn.
- Report brought up to date with the web work: dependencies (Chapter 1), architecture with `api/`,
  `server/` and `utils/` (Chapter 3), the memory of the web page and mesh timings (Chapter 6), the
  limits of the server (Chapter 7), the conclusion and the appendix. Every mesh figure (Table 4.1,
  sizes, counts, position error, timings) is now measured by `tools.report figures`, which meshes
  each label in every format (`meshes.csv`); it also writes the README's mesh table between markers.
  Today's version was re-timed in the same power mode as the others (24 s, 1.7 GB).
- A finished job's duration no longer keeps growing when it is asked for later (it showed minutes
  for a 14 s job).
- Mesh files in two formats, chosen by the extension: STL (other software) and GLB (binary glTF, for
  the browser), written with numpy only. GLB stores each point once, with smooth vertex normals, and
  by default 16-bit positions over the mesh's bounding box (`mesh.glb_position_bits`, error at most
  half of its size / 65,535: 0.05 µm on the sample; `32` for floats). The vessel mesh of the sample:
  134.6 MB as STL, 48.5 MB as GLB. The server builds and caches each label per format
  (`?format=glb|stl`, default `api.default_mesh_format`), and lists them per label.
- Documentation: how to start the web server and which address to open (README, INSTALL "WEB
  SERVER"); README "KNOWN LIMITATIONS" grouped into results, other scans (preparation, thresholds,
  fixed rules, values tuned on one sample, format) and web server; the report gains the web server
  (Chapter 3) and the limits for other scans and of the server (Chapter 7).

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
