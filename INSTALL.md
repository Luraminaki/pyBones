# INSTALL

This guide covers installing pyBones, getting the dataset in place, running the pipeline and the
tools, and building the report.

## TABLE OF CONTENT

<!-- TOC -->

- [INSTALL](#install)
  - [TABLE OF CONTENT](#table-of-content)
  - [PREREQUISITES](#prerequisites)
    - [Python 3](#python-3)
    - [uv (recommended)](#uv-recommended)
  - [INSTALL GUIDE](#install-guide)
  - [DATASET](#dataset)
  - [RUN](#run)
  - [WEB SERVER](#web-server)
  - [DEVELOPMENT](#development)
  - [TOOLS](#tools)
  - [REPORT (LaTeX)](#report-latex)
    - [LaTeX](#latex)
    - [Generating the report](#generating-the-report)

<!-- /TOC -->

## PREREQUISITES

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
the configuration, **scikit-image** for marching cubes only.

## DATASET

The µCT stack (100 slices, 2300 x 1487, 16-bit TIFF, ~685 MB) is not versioned. Download it
from [Google Drive](https://drive.google.com/file/d/1y0qstBI9n7ujrAe_rba5TdLauAi6bJex/view?usp=drive_link)
and put the archive at the repository root as `data_set.zip`: pyBones reads the slices straight from
the zip, no extraction needed. A directory of slices works as well.

The dataset is not covered by the MIT License of the code: see
[LICENSE AND DATA](README.md#license-and-data) in the README before reusing it.

## RUN

```sh
pybones segment data_set.zip -o output/run                       # ~25 s, ~1.7 GB peak (*)
python -m pybones segment data_set.zip -o output/run             # same command, as a module
pybones preview output/run -z 50 --scale 0.5 -o slice50.png
pybones mesh output/run --label vessel -o vessels.stl             # µm, Taubin-smoothed
pybones mesh output/run --label vessel -o vessels.glb             # same, glTF (16-bit positions)
pybones export output/run --label vessel -o output/vessel_png     # ImageJ image sequence
pybones config > my_config.json                                   # effective configuration
pybones --config my_config.json --set pipeline.workers=1 segment data_set.zip -o output/run
```

Every run of `pybones` or of a tool logs to its own file, `logs/<program>-<pid>.log` (time, process,
thread, module, function, message), and to the terminal in a short form (`time level message`): what
was written where, scores, timings, warnings. Only the most recent runs' files are kept; the levels of
both are in the `logging` section of the configuration (`console_level` `WARNING` keeps the terminal
quiet). `pybones config` writes its JSON on stdout, so that it can be redirected to a file.

(*) On an AMD Ryzen AI 9 HX 370 (24 logical cores, 4 worker threads), 24 GB RAM, Linux; the report
lists the machine every timing was measured on.

Every tunable value lives in [src/pybones/config/default.json](src/pybones/config/default.json) and
is validated (types, bounds) by [src/pybones/config/schema.py](src/pybones/config/schema.py). A user
file only needs the values it changes. Sizes are physical (µm): if the voxel size of your scan
differs, change `acquisition.voxel_size_um` and everything follows.

## WEB SERVER

The server reads its stacks from a `data/` folder and writes segmentations to `output/` (both relative
to the folder you start it from). Put the dataset where it looks, without moving it away from the
repository root where the other commands expect it:

```sh
mkdir -p data && ln -s ../data_set.zip data/      # Linux / macOS: a link, no copy
```

```powershell
mkdir data; copy data_set.zip data\              # Windows: a copy
```

Then start it from the repository root:

```sh
pybones serve                                     # or: uv run pybones serve
```

and open **<http://127.0.0.1:8000>** in a web browser on the same machine. The terminal prints that
address (and the folders in use) when the server starts; Ctrl+C stops it, which also cancels the
running jobs and removes their partial outputs.

There is no web page yet: the address leads to the interactive API documentation (`/docs`), where
each request can be tried with "Try it out". A typical session:

1. `GET /api/datasets`: the stacks found in `data/` (here `data_set.zip`).
2. `POST /api/segmentations` with `{"dataset": "data_set.zip", "name": "run1"}`: starts a job
   (~35 s on the sample); `"overrides": ["refinement.enabled=false"]` changes settings for this run.
3. `GET /api/jobs/segmentation-1/events`: its progress, pushed as it happens (Server-Sent Events).
4. `GET /api/segmentations/run1/slices/50?scale=0.4`: a slice with its labels, as a PNG image.
5. `POST /api/segmentations/run1/meshes/vessel`, then once its job succeeded,
   `GET /api/segmentations/run1/meshes/vessel`: the vessel surface in µm, as GLB (for the browser);
   add `?format=stl` to both for an STL file. The mesh settings, precision included, are explained in
   the [README](README.md#how-it-works).

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

## DEVELOPMENT

```sh
uv run ruff check src tests tools
uv run ruff format src tests tools
uv run basedpyright
uv run pytest
uv run pytest --cov=src/pybones --cov=tools --cov-report=term-missing   # coverage
PYBONES_UPDATE_GOLDEN=1 uv run pytest tests/regression                 # record a new baseline
```

`tests/regression` segments a small fixed phantom and compares its scores with
`tests/regression/golden_phantom.json`: any score moving by more than the tolerance (up or down) fails,
so that a change of results is always deliberate. After an intended change, record the new baseline
with the command above and commit it with the change.

The operations are also available from Python (`pybones.api`), e.g. for a web front end:

```python
from pathlib import Path
from pybones.api.jobs import JobManager
from pybones.config.loader import load_config

config = load_config()
jobs = JobManager(config.jobs)
job_id = jobs.submit_segmentation(Path("data_set.zip"), Path("output/run"), config)
for status in jobs.events(job_id):             # pushed as they happen: no polling
    print(status.model_dump_json())            # state, stage k/n, fraction, remaining time
meta = jobs.result(job_id)                     # or jobs.cancel(job_id) from another thread
```

`events()` is made for Server-Sent Events: a web endpoint sends each status as
`data: <json>\n\n`. Progress events are throttled (`jobs.event_interval_s`); state changes always go
through; a slow client loses its oldest snapshots, never the newest.

[Ruff](https://docs.astral.sh/ruff/) also enforces the project conventions: no magic values, Google
docstrings, type-only imports under `if TYPE_CHECKING:` (quoted annotations), no imports outside the
module top level. [basedpyright](https://docs.basedpyright.com/) checks the types with the settings of
`pyproject.toml` (the VS Code extension reads the same ones). [pytest](https://docs.pytest.org/) runs
the suite in `tests/` (synthetic data, brute-force references; no dataset needed); it includes
`tests/test_conventions.py`, which checks that every function, private ones included, has a complete
Google docstring and that names are imported from the module that defines them.

## TOOLS

Run from the repository root (they are not part of the installed package):

```sh
# Synthetic phantom with exact ground truth: the validation metric
python -m tools.phantom generate -o output/phantom
pybones segment output/phantom/slices -o output/phantom_seg
python -m tools.phantom score output/phantom_seg --phantom output/phantom

# Annotation page (local web server) to paint a reference on real crops
python -m tools.annotation prepare output/run --name 102OAcentri8sDXZ08
python -m tools.annotation serve --name 102OAcentri8sDXZ08 --open
python -m tools.annotation score output/run --name 102OAcentri8sDXZ08

# Report data and figures: runs past versions (git worktrees) on the same inputs
python -m tools.report history
python -m tools.report figures
```

Each tool has its own `config.json`, validated like the main one.

## REPORT (LaTeX)

The report ([docs/report/report.pdf](docs/report/report.pdf), sources in [docs/report/](docs/report/))
is written in LaTeX and built with **XeLaTeX**. The compiled PDF is versioned, so you only need what
follows to regenerate it.

### LaTeX

- **Debian / Ubuntu**

```sh
sudo apt-get install -y texlive-xetex texlive-latex-extra texlive-pictures texlive-fonts-recommended latexmk
```

- **Arch Linux**

```sh
sudo pacman -S --needed texlive-basic texlive-bin texlive-latex texlive-latexrecommended \
    texlive-latexextra texlive-fontsrecommended texlive-pictures texlive-xetex texlive-binextra
```

- **Windows**

```powershell
winget install MiKTeX.MiKTeX
```

### Generating the report

Every figure, plot, table and measured value of the report is generated from the data and from the git
history; nothing is typed by hand. From the repository root, with `data_set.zip` in place:

```sh
# 1. Segment the stack and render the synthetic phantom
uv run pybones segment data_set.zip -o output/latest
uv run python -m tools.phantom generate -o output/phantom

# 2. Run every milestone version listed in tools/report/config.json (each in its own git worktree
#    and environment) on the phantom, and on the real data where figures need it; score them and
#    keep a small extract of each (docs/report/data/runs/). Versions whose commit is not in this
#    repository (the development history is not published) keep their archived scores and extracts.
uv run python -m tools.report history

# 3. Render the figures, plot data, history table, measured values (values.tex) and README screenshots
uv run python -m tools.report figures

# 4. Build the PDF (settings in docs/report/latexmkrc; the PDF is copied to docs/report/report.pdf)
cd docs/report && latexmk report.tex
```

The figures read the extracts, never the full runs of past versions, so the report can be rebuilt
and updated from a fresh clone: only the current code is re-run. Step 2 takes a few minutes the first
time (one environment per version, then cached); later runs
reuse the stored outputs unless `--force` is given. Old worktrees can be removed with
`rm -rf output/report/worktrees && git worktree prune`.
