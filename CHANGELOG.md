# Changelog

All notable changes to this project are documented here. Versioning loosely
follows [Semantic Versioning](https://semver.org/); the project is still
`-alpha`, so breaking changes (configuration keys, outputs) can happen between
minor versions.

The current version lives in [pyproject.toml](pyproject.toml) and is read
from package metadata at runtime; every segmentation records it in its
`meta.json`. The [report](docs/report/) measures what each step below changed.

## Unreleased

- Thin vessels the blur hides, an optional last pass (`thin_vessels.enabled`, **off by default**):
  - What it does: the stack is deconvolved (Richardson-Lucy with the measured blur, σ 4.4 µm,
    3 iterations). Voxels labelled background or bone that then reach the vessel-core threshold
    are candidates, linked into 3D pieces. Pieces of at least 540 µm³ in which no ball of 6 µm
    fits become vessel.
  - Phantom (measured before the stage was built, as a post-process): thin vessels found under
    6 µm radius 43 % -> 61 %; nothing else worse (vessel precision unchanged, 50 false
    junctions instead of 49); network length -8 % -> -3 %. The result holds with the blur wrong
    by ±20 %.
  - Real stack: +96,173 voxels (+1.3 % vessel), 96 % of them labelled bone before; 1,595 pieces.
    Up close they are bright spots and short bars in or along bone, where only an expert can
    tell a vessel from dense bone: hence off by default, like `bone_side`.
  - Cost on the real stack: +17 s (segmentation 24 -> 41 s); peak memory 1.64 -> 2.53 GB.
  - How it streams: the deconvolution runs in blocks whose margins make them equal to the whole
    volume (tested), on the worker threads. The thin test is an erosion by the ball, slice by
    slice. Pieces are linked across slices as the refinement does.
  - New stages `sharpen` and `thin`, with measured progress weights. `meta.json` records what
    the pass added (`thin_vessels`).
  - New modules: `imaging/deconvolution.py` and `segmentation/thin_vessels.py`.
  - The configuration gains a section: outputs of 0.7.0 must be segmented again.
- INSTALL: a LABELLING CROPS section for whoever labels the real scan. It covers starting the
  annotation page (regenerate, serve), painting and the *unsure* label, the neighbouring slices,
  correcting rather than trusting the pre-fill, where the labels are saved, and scoring. The
  README's annotation screenshot links to it.
- Documentation reorganised, each fact in one place:
  - `CONTRIBUTING.md` (new): checking a change, the tests and the rules they enforce, adding a
    setting, VS Code and the web page, the tools, the generated documentation, the report. `INSTALL.md`
    keeps installing and running pyBones (command line, browser, and a FROM PYTHON section).
  - `docs/CONFIGURATION.md` (new): every setting of `default.json` and `server.json`, its default,
    the values it accepts and its meaning, written from the pydantic schemas by `python -m tools.docs`
    (new tool). `tests/docs_tool` fails when the file is out of date or a setting has no description;
    the 16 settings without one got one (the `io` file names, the logging levels, both `enabled`
    switches, `mesh.downsample`, the `render` section).
  - Measured numbers in README, INSTALL and CONTRIBUTING are marked with two comments naming a report
    value, and rewritten by `python -m tools.report figures` (new step `markdown_values`, new setting
    `value_documents`); a name the report does not know stops it. 22 numbers are marked: segmentation
    time and peak memory, mesh and quantification times, the sample's network, the phantom's thin-vessel
    recall and network scores.
  - The bone-side setting's phantom numbers left the README (the 0.7.0 entry below records them); the
    README's project layout no longer says `docs/report/2018/` holds the 2018 masks (gone since 0.4.0).

## 0.7.0-alpha

Fewer false junctions in the vascular network: bursts no longer follow the vessels lying in a slice,
an optional vessel edge against bone (off until the real scan can confirm it), small loops pruned, and
phantom network scores that expect only what an image can show. The causes of the remaining false
junctions are measured and ranked. Outputs of 0.6.0 must be segmented again.

- Bursts no longer follow vessels: the density that places a burst's boundary is now measured in 3D
  (a Gaussian in the slice, on the 12 µm grid, averaged along z over the burst box's 41 slices), like
  the core density. Measured per slice, a vessel lying in the slice plane is a strip of bright pixels,
  as dense as a burst's edge from about 8 µm of radius: a burst touching it grew along its whole
  length and took the core of thick vessels, leaving them hollow (on the branching phantom, 47,000 of
  the 60,000 vessel voxels the bursts took, half of them more than 60 µm away from the burst). In 3D
  the same vessel is a cylinder, as dense only from about 22 µm. Tightening the 2D boundary could not
  do this (it left the bursts' edge speckles out, as false vessel).
  - Standard phantom: vessel Dice 0.851 → 0.862, burst 0.913 → 0.922, bone 0.989 unchanged; golden
    baseline refreshed, every score up (vessel Dice 0.773 → 0.839, burst 0.799 → 0.838).
  - Branching phantom: vessel Dice 0.790 → 0.850; 26 forks found instead of 24, 49 junctions that
    are not forks instead of 59, 7 independent loops instead of 10, centreline precision 0.76 → 0.85.
  - Real stack: bursts more consistent along z (slice-to-slice Dice mean 0.961 → 0.978, min
    0.847 → 0.887), burst +2.2 %, vessel +1.0 %, bone −0.1 %. The network barely changes (3,121
    junctions instead of 3,129): on the sample, a burst's growth past its core mostly covers speckled
    lobes of the burst, not vessels; the phantom, whose bursts sit anywhere among solid vessels,
    exaggerated that error.
  - Same time (~24 s); peak memory +30 MB (a second coarse map in the z window), and the
    full-resolution Gaussian of each slice is gone. `PixelParams.burst_boundary_sigma` is now in grid
    blocks.
- New setting `segmentation.vessel.bone_side` (off by default). Where a vessel presses on bone, its
  half-maximum edge is measured against the soft tissue around it, so a band of bone one to three
  voxels wide joins the vessel: the largest source of junctions that are not forks on the branching
  phantom. With `enabled: true`, each pixel near a vessel is measured against the nearer of soft
  tissue and bone, bone counting only where it holds a disc of `barrier_um` (12 µm; a dim, partly
  filled vessel is foreground too, but thin).
  - Standard phantom: vessel Dice 0.862 → 0.887, bone 0.989 → 0.990. Branching phantom: vessel Dice
    0.850 → 0.870, 28 junctions that are not forks instead of 49, 30 forks instead of 26.
  - Off, because the real stack cannot confirm it: it turns 0.98 million voxels of vessel into bone
    (vessel −4.0 %), largely networks of thin, dim structures next to bone that may be partly filled
    vessels. Against the 2018 masks, vessel precision rises (0.828 → 0.858) but bone precision falls
    (0.931 → 0.916), and no annotation exists to decide. It should become the default once annotated
    crops confirm it.
- Network scores of the phantom: a vessel crossing a burst is no longer expected (the truth labels it
  burst, so no vessel mask can hold it), as gaps already were not: 1.15 of the 22.87 mm and 1 of the 54
  forks of the branching phantom. Centreline recall of the old segmentation: 0.68 → 0.72 from this
  alone. It shows that the measure itself reads lengths 8 % long on the exact mask (a centreline of
  voxels zigzags, each step counted from centre to centre); the report and the README now say so and
  read lengths only with recall and precision (new macro `\pbValueNetTrueLengthErrorPct`).
- Vascular network: loops barely longer than their vessel's circumference are holes or folds of the
  vessel mask, not structures. Every loop shorter than `quantification.pruning.loop_factor` (1.5)
  circumferences loses its longest branch, found with a minimum spanning tree of the network, so
  nothing is ever disconnected (two strands around a small hole, a *bubble*, are the simplest case).
  On the segmented branching phantom: 59 junctions that are not forks instead of 90, 10 independent
  loops instead of 49 (4 in the exact mask, where vessels cross), 2 of its 26 real forks lost; the
  exact mask keeps all its 40 forks. On the sample: 3,129 junctions instead of 4,325, 591 mm of vessels
  instead of 680 (doubled strands gone); the network stage takes 0.8 s instead of 0.3 (the loop
  search runs only once no cheaper removal applies).
- Corrected explanation (report, README, page): most junctions that are not forks come from small
  loops the skeleton draws around holes and folds of the vessel mask. Repairing the segmented mask
  from the truth one defect at a time, the quantification unchanged, ranks their causes: the band of
  bone that joins a vessel where it presses on bone (repaired alone, about half of them go; see
  `segmentation.vessel.bone_side` above), then the bursts hollowing out thick vessels (fixed above),
  vessel drawn over soft tissue, and clotted segments cut off by the halo rule. With every defect
  repaired, the segmentation's network is nearly the exact mask's: the quantification is sound. The
  network length is the net of several errors, not reliable by itself. Phantom scores count each
  network's independent loops. Tried and dropped, measured on the phantom: longer spurs against bone
  and closing the vessel mask before thinning (more real forks and length lost than false junctions
  removed), filling small holes slice by slice (no effect), tightening the 2D edge of bursts (the
  bursts' edge speckles become false vessel; the standard phantom's validated scores drop).
- An output written by an earlier build of the same version, with another configuration format, now
  says so, instead of "written by pyBones 0.6.0 and cannot be read by pyBones 0.6.0".
- Progress you can trust during long silent stages:
  - the server sends every new stage at once (only progress within a stage is throttled): a stage
    starting right after another, like the quantification's skeleton or a mesh's surface, used to be
    held back until its end, the page showing the previous stage's name all along;
  - the page keeps a running job's times moving between two statuses (elapsed up, time left down),
    in the jobs panel, the Network tab and the 3D view.
- Progress stage weights are measured: `python -m tools.report figures` times each stage of the
  segmentation, a mesh and the quantification on the sample (`stage_weights.csv`, a table in the
  performance chapter) and warns when a weight is more than `stage_weight_tolerance` (0.05) off. The
  defaults were set from these measurements (segmentation resolve 0.12 -> 0.17, mesh occupancy
  0.24 -> 0.30, among others); every stage is now within 0.02 of its share.
- `tests/test_unused.py`: no dead code. Every function, class, method and module constant of
  `src/` and `tools/` must be used by them, not by the tests alone; every test helper by a test;
  every JavaScript export by another module or a Node test (checked against planted dead code).
  Its first run moved `read_stl` and `classify_slice`, which only the tests used, to
  `tests/helpers.py`, and un-exported `fitView`.
- Stale or inconsistent documentation fixed: segmentation time (~25 s everywhere), mesh build times
  and a Network step in INSTALL's page walkthrough, the thick vessels' radius bias (1.5 µm) and the
  skeleton time (10 s) in the README, the pipeline figure of the report (the network row).
- `measure_network` reports the end of its last stage itself (its graph work counted for nothing
  when it ran on its own).

## 0.6.0-alpha

Vascular quantification, the original goal of the project: a segmentation's vessel network is
measured from the command line, the API or the page. Outputs of 0.5.0 must be segmented again.

- Report: chapter 6, *The vascular network* (method, branching phantom scores, radius error per
  band, the sample's measures, limits); every number is written by `python -m tools.report figures`
  (new `network` section of its settings: the phantom is generated and segmented afresh each time).
  Limitations and conclusion updated; chapters 7-9 renumbered.
- `measure_network` measures any label volume without writing anything; `run_quantification`
  reads, measures and writes. Phantom network scores give the radius offset per band of true radius.
- README: the quantification step, its backends' pros and cons, its limits, and a screenshot of the
  Network tab and the centrelines.
- Annotation page (`tools/annotation/static/`): split from one 658-line script into ES modules
  (`state`, `view`, `paint`, `session`, `controls`, `app`), every function documented (JSDoc).
  Elements are built without `innerHTML`; failed requests are reported instead of being read as
  data or lost (opening a crop from a click or an arrow key, saving, the note). The same checks as
  the web page now cover it.
- Report timings: every version re-timed in one session, under one power profile, which every
  timed run now records (`Machine.power_profile`, from `/sys/firmware/acpi/platform_profile` or
  `powerprofilesctl`) and the report prints. The archived timings were 5-10 % slower than today's
  re-run (a slower power mode, most likely); the comparison between versions is unchanged.
- INSTALL: a FROM SCRATCH walkthrough (clone, `uv sync`, dataset, command line or browser, report),
  Git and the optional tools (browser, Node.js, LaTeX) in the prerequisites, and the report steps
  corrected: `history` segments the stack itself, `figures` measures the network, timings need one
  power profile, screenshots 05-07 are taken by hand.
- VS Code: a `pybones: quantify` launch configuration; the report's reproduction appendix matches
  INSTALL (no needless segmentation step, the network step, timings under one power profile).
- `python -m tools.check` runs every check in order (formatting, lint, types, tests) and lists the
  failed ones (`--fix` reformats and applies Ruff's safe fixes first). INSTALL's DEVELOPMENT section
  explains how to check a change, run all or part of the tests, and which rule each test enforces.
- `tests/test_duplication.py`: no pure forwarders, no two functions with the same body, no run of
  logic (3 statements) or of JavaScript (4 lines) repeated in two places, variable names aside;
  plain call sequences (entry-point wiring) are allowed. Each rule was checked against planted
  violations. Its first run removed the duplications it found: starting or joining a job
  (`startOrJoin`), freeing a 3D object (`discard`), locating a quantification file, writing a
  README section (`tools/report/readme.py`), validating the command line (`parse_invocation`,
  shared by `pybones` and the three tools), grouping voxels in the network graph, and several test
  setups (`start_blocked`, `small_phantom_settings`, `truncated_window`).
- `tests/test_conventions.py` keeps every module, Python or JavaScript (vendored code aside), under
  500 lines.
- Fixed: on the 3D or Network tab, every finished job reloaded the slice the hidden Slices tab
  showed, and Firefox refused to decode it there ("Slice 50 could not be shown" after each mesh
  build). A segmentation shown again keeps its slice, and slices wait for their load event, which
  fires in a hidden tab too.
- README screenshots: the web page (three tabs, 0.6.0) retaken, the Network tab added.
- Corrected: wall distances read *short* of the true radius on an exact mask (up to a third of a
  voxel, more for thick vessels), not long as the docstring said.
- Vascular network, ready to use:
  - `pybones quantify SEGMENTATION`, with a progress line;
  - API: `POST /api/segmentations/{id}/quantification` (a job; asking again joins it, or measures
    again once done), `GET` the same path for the summary (404 before, 409 for a damaged or older
    file), `GET /api/segmentations/{id}/centrelines` for the points (revalidated like the meshes);
    segmentation summaries say whether they are `quantified`;
  - the page: a Network tab (Quantify / Quantify again, progress, the measures, what they mean
    and their limits) and a centrelines toggle in the 3D view (points over the meshes);
  - job targets carry the job's kind, so a quantification never answers for the segmentation of the
    same name; the page's job kinds (`JOB_KINDS`) are checked against Python's.
- The page's own files are served with `Cache-Control: no-cache`: a browser checks them on every load,
  so new modules never meet old cached ones (which had broken the 3D tab during development).
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
  phantom, segmented: length within 1 % of the truth, 26 of 54 visible bifurcations found, but
  90 spurious junctions (mostly vessel/bone contact bulges), radius offset −0.8 µm (not constant:
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
