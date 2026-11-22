# INSTALL

This guide covers installing pyBones, getting the dataset in place, and running it: from the command
line, from Python or from the browser. Changing the code, the tests, the validation tools and the report
are in [CONTRIBUTING.md](CONTRIBUTING.md); every setting is in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## TABLE OF CONTENT

<!-- TOC -->

- [INSTALL](#install)
  - [TABLE OF CONTENT](#table-of-content)
  - [FROM SCRATCH](#from-scratch)
  - [PREREQUISITES](#prerequisites)
    - [Git](#git)
    - [Python 3](#python-3)
    - [uv (recommended)](#uv-recommended)
    - [Optional](#optional)
  - [INSTALL GUIDE](#install-guide)
  - [DATASET](#dataset)
  - [RUN](#run)
  - [WEB SERVER](#web-server)
  - [FROM PYTHON](#from-python)

<!-- /TOC -->

## FROM SCRATCH

From nothing to a segmented, meshed and measured stack in the browser, then the report. Each step
links to its details.

1. **Prerequisites**: [Git](#git), [Python 3.12+](#python-3) and [uv](#uv-recommended).
2. **Get the code and its environment**:

   ```sh
   git clone https://github.com/Luraminaki/pyBones.git
   cd pyBones
   uv sync                      # creates .venv with the exact locked versions
   uv run python -m tools.check # optional: every check, tests included, no dataset needed
   ```

3. **Get the dataset** into `data/data_set.zip` ([DATASET](#dataset)).
4. **Use it**, from the command line ([RUN](#run)):

   ```sh
   uv run pybones segment data/data_set.zip -o output/run     # labels, about half a minute
   uv run pybones mesh output/run --label vessel -o vessels.glb
   uv run pybones quantify output/run                          # vessel network, about 20 s
   ```

   or from the browser ([WEB SERVER](#web-server)): `uv run pybones serve`, then open
   <http://127.0.0.1:8000>, pick the stack, *Segment*, and explore the Slices, 3D and Network tabs.
5. **Rebuild the report** (optional; the PDF is versioned): install
   [LaTeX](CONTRIBUTING.md#latex), then follow
   [Generating the report](CONTRIBUTING.md#generating-the-report): render the phantom, `history`,
   `figures`, `latexmk`.

## PREREQUISITES

### Git

Needed to clone the project, and by the report tool, which checks past versions out as git worktrees.

- **Debian / Ubuntu**: `sudo apt-get install -y git`
- **Arch Linux**: `sudo pacman -S git`
- **Windows**: `winget install Git.Git`

### Python 3

This project targets **Python 3.12+**.

- **Debian / Ubuntu**

```sh
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

- **Arch Linux**

```sh
sudo pacman -Syu python python-pip
```

- **Windows**

```powershell
winget install Python.Python.3.12
```

Alternatively, download the installer from [python.org](https://www.python.org/downloads/).

### uv (recommended)

[uv](https://docs.astral.sh/uv/) creates the virtual environment, installs the locked dependencies
(`uv.lock`) and runs the commands in one go. Plain `pip` works too (see [below](#install-guide)).

- **Debian / Ubuntu** or **MacOS**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **Arch Linux**

```sh
sudo pacman -Syu uv
```

- **Windows**

```powershell
winget install astral-sh.uv
```

### Optional

- **A recent browser** (Firefox, Chrome, Edge, Safari) for the web page; WebGL 2 for its 3D view.
- **[Node.js](https://nodejs.org/)**, any recent version, only to run the JavaScript checks of the
  test suite ([CONTRIBUTING](CONTRIBUTING.md#setup)); they are skipped without it.
- **LaTeX**, only to rebuild the report ([CONTRIBUTING](CONTRIBUTING.md#latex)).

## INSTALL GUIDE

Clone or download this project, then open a terminal in the `pyBones` directory.

- **With uv** (runtime + development tools, exact locked versions)

```sh
uv sync
```

Every command below can then be prefixed with `uv run` (e.g. `uv run pybones --help`), no
activation needed.

- **With pip**: create and activate a virtual environment, as shown in
  [HowTo](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments), then install
  the project (this reads `pyproject.toml`).

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U .                 # or: pip install -U -e . --group dev  (pip >= 25.1)
```

Dependencies are deliberately few: **numpy** and **OpenCV** for the computations, **pydantic** for
the configuration, **scikit-image** for marching cubes and 3D skeletonisation only, **SciPy** as one
of the interchangeable backends of the vascular quantification (scikit-image requires it anyway),
and **FastAPI** with **uvicorn** for the web server. three.js, for the page's 3D view, is kept in the
repository.

## DATASET

The µCT stack (100 slices, 2300 x 1487, 16-bit TIFF, ~685 MB) is not versioned. Download it
from [Google Drive](https://drive.google.com/file/d/1y0qstBI9n7ujrAe_rba5TdLauAi6bJex/view?usp=drive_link)
and put the archive in the repository's `data/` folder, as `data/data_set.zip`: pyBones reads the
slices straight from the zip, no extraction needed. A directory of slices works as well. `data/` is
where the web server looks for stacks; git ignores its content (only `data/.gitkeep` is versioned).

The dataset is not covered by the MIT License of the code: see
[LICENSE AND DATA](README.md#license-and-data) in the README before reusing it.

## RUN

```sh
pybones segment data/data_set.zip -o output/run                  # see (*) for time and memory
python -m pybones segment data/data_set.zip -o output/run        # same command, as a module
pybones preview output/run -z 50 --scale 0.5 -o slice50.png
pybones mesh output/run --label vessel -o vessels.stl             # µm, Taubin-smoothed
pybones mesh output/run --label vessel -o vessels.glb             # same, glTF (16-bit positions)
pybones export output/run --label vessel -o output/vessel_png     # ImageJ image sequence
pybones quantify output/run                                       # vessel network (3 files in output/run)
pybones config > my_config.json                                   # effective configuration
pybones --config my_config.json --set pipeline.workers=1 segment data/data_set.zip -o output/run
```

Every run of `pybones` or of a tool logs to its own file, `logs/<program>-<pid>.log` (time, process,
thread, module, function, message), and to the terminal in a short form (`time level message`): what
was written where, scores, timings, warnings. Only the most recent runs' files are kept; the levels of
both are in the `logging` section of the configuration (`console_level` `WARNING` keeps the terminal
quiet). `pybones config` writes its JSON on stdout, so that it can be redirected to a file.

(*) The sample takes ~<!-- pb:Seconds -->24<!-- /pb --> s, with a peak of
~<!-- pb:PeakGB -->1.7<!-- /pb --> GB of memory, on an AMD Ryzen AI 9 HX 370 (24 logical cores, 4 worker
threads), 24 GB RAM, Linux; the report lists the machine every timing was measured on.

Every setting, its default, the values it accepts and what it does are listed in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md), with the three ways to change them (a JSON file of
your own, `--set`, the page). A user file only needs the values it changes. Sizes are physical (µm):
if the voxel size of your scan differs, change `acquisition.voxel_size_um` and everything follows.

## WEB SERVER

The server reads its stacks from the `data/` folder (where the [dataset](#dataset) goes) and writes
segmentations to `output/`, both relative to the folder you start it from. Any folder or zip archive
of 16-bit slices put in `data/` shows up on the page. Start it from the repository root:

```sh
pybones serve                                     # or: uv run pybones serve
```

and open **<http://127.0.0.1:8000>** in a web browser on the same machine. The terminal prints that
address (and the folders in use) when the server starts; Ctrl+C stops it, which also cancels the
running jobs and removes their partial outputs.

The page, in a few steps:

1. **New segmentation**: pick a stack of `data/`, keep or change the suggested name, and press
   *Segment*. Each run gets its own name; to run the segmentation you are looking at again (e.g. with
   one setting changed), press *Segment again…* under its title: the form is filled with its stack
   and its settings. Under *Settings for this run*, one `section.key=value` per line changes the
   configuration for this run only, e.g. `refinement.enabled=false`. The *Jobs* panel shows the stage,
   the progress, the time elapsed and the time left (~<!-- pb:Seconds -->24<!-- /pb --> s in all on
   the sample); *Cancel* stops the run and removes its partial files.
2. **Slices**: when the run ends, the new segmentation opens. Drag the slider to move through the
   stack; *Size* sets the image scale.
3. **3D**: tick the labels to show. A mesh that does not exist yet is built first (a job,
   ~<!-- pb:MeshVesselSeconds -->6<!-- /pb --> s for the vessels of the sample,
   ~<!-- pb:MeshBoneSeconds -->13<!-- /pb --> s for the bone), then loaded. Left-drag turns the view, or moves it
   with the *Move* switch (Shift swaps the two); right-drag always moves it; the wheel, a two-finger
   swipe or the + / − buttons zoom; *Reset view* looks down the stack again.
4. **Network**: *Quantify* runs the quantification (a job, ~<!-- pb:NetSeconds -->18<!-- /pb --> s
   on the sample), then shows the
   network's summary; its centrelines can be drawn over the meshes in the 3D tab.

Everything the page does goes through the HTTP API, documented and testable at
**<http://127.0.0.1:8000/docs>** ("Try it out"). The same session by hand:

1. `GET /api/datasets`: the stacks found in `data/` (here `data_set.zip`).
2. `POST /api/segmentations` with `{"dataset": "data_set.zip", "name": "run1"}`: starts a job;
   `"overrides": ["refinement.enabled=false"]` changes settings for this run.
3. `GET /api/jobs/segmentation-1/events`: its progress, pushed as it happens (Server-Sent Events).
4. `GET /api/segmentations/run1/slices/50?scale=0.4`: a slice with its labels, as a PNG image.
5. `POST /api/segmentations/run1/meshes/vessel`, then once its job succeeded,
   `GET /api/segmentations/run1/meshes/vessel`: the vessel surface in µm, as GLB (for the browser);
   add `?format=stl` to both for an STL file. The mesh settings, precision included, are explained in
   the [README](README.md#how-it-works) and listed in [docs/CONFIGURATION.md](docs/CONFIGURATION.md#mesh).
6. `POST /api/segmentations/run1/quantification`, then once its job succeeded,
   `GET /api/segmentations/run1/quantification`: the vessel network's summary (branches, junctions,
   ends, length, length density, medians); `GET /api/segmentations/run1/centrelines`: its centreline
   points as raw float32 `(x, y, z)` µm triples, as drawn in the 3D tab.

Other folders or port:

```sh
pybones serve --port 8080 --data-dir /srv/stacks --output-dir /srv/segmentations
pybones serve --server-config my_server.json      # only the values it changes
```

The server settings (address, folders, limits) live in
[src/pybones/config/server.json](src/pybones/config/server.json), apart from the segmentation
configuration (which every segmentation records in its `meta.json`). Clients only send ids (a name in
the data or output folder), never paths. It listens on `127.0.0.1` without any login: **keep it
local** (see the limitations in the [README](README.md#known-limitations)).

## FROM PYTHON

The operations are also available from Python (`pybones.api`), e.g. for a web front end:

```python
from pathlib import Path
from pybones.api.jobs import JobManager
from pybones.config.loader import load_config

config = load_config()
jobs = JobManager(config.jobs)
job_id = jobs.submit_segmentation(Path("data/data_set.zip"), Path("output/run"), config)
for status in jobs.events(job_id):             # pushed as they happen: no polling
    print(status.model_dump_json())            # state, stage k/n, fraction, remaining time
meta = jobs.result(job_id)                     # or jobs.cancel(job_id) from another thread
```

`events()` is made for Server-Sent Events: a web endpoint sends each status as
`data: <json>\n\n`. Progress events are throttled (`jobs.event_interval_s`); state changes always go
through; a slow client loses its oldest snapshots, never the newest.
