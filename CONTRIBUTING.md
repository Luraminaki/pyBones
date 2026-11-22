# CONTRIBUTING

This guide is for changing pyBones: the checks every change passes, the tests and the rules they
enforce, the validation tools, the documentation written from the code, and the report. To install
and run pyBones, see [INSTALL.md](INSTALL.md); every setting is described in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## TABLE OF CONTENT

<!-- TOC -->

- [CONTRIBUTING](#contributing)
  - [TABLE OF CONTENT](#table-of-content)
  - [SETUP](#setup)
  - [CHECKING A CHANGE](#checking-a-change)
  - [TESTS](#tests)
    - [Running the tests](#running-the-tests)
    - [What the tests enforce](#what-the-tests-enforce)
  - [CHANGING A SETTING](#changing-a-setting)
  - [VS CODE AND THE WEB PAGE](#vs-code-and-the-web-page)
  - [TOOLS](#tools)
  - [GENERATED DOCUMENTATION](#generated-documentation)
  - [REPORT (LaTeX)](#report-latex)
    - [LaTeX](#latex)
    - [Generating the report](#generating-the-report)

<!-- /TOC -->

## SETUP

Install as in [INSTALL.md](INSTALL.md), with uv: `uv sync` also installs the development tools (Ruff,
basedpyright, pytest). Optional:

- **[Node.js](https://nodejs.org/)**, any recent version, for the JavaScript checks of the test suite;
  they are skipped without it. No `npm install` needed.
- **LaTeX**, to rebuild the report ([LaTeX](#latex)).
- **A browser**, to check the web page by hand (there is no browser test suite).

## CHECKING A CHANGE

One command runs every check, in order, and lists the ones that failed at the end (exit status 1):

```sh
uv run python -m tools.check          # formatting, lint, types, tests (~30 s)
uv run python -m tools.check --fix    # reformat and apply Ruff's safe fixes first, then the same
```

| Step | Tool | What it checks |
|---|---|---|
| format | [Ruff](https://docs.astral.sh/ruff/) | the code is formatted (`ruff format`) |
| lint | Ruff | its rules, the project's among them: no magic values, Google docstrings, type-only imports under `if TYPE_CHECKING:`, no imports inside functions |
| types | [basedpyright](https://docs.basedpyright.com/) | the types, with the settings of `pyproject.toml` (the VS Code extension reads the same ones) |
| tests | [pytest](https://docs.pytest.org/) | the whole suite, below |

Each step can be run alone: `uv run ruff format --check src tests tools`, `uv run ruff check src tests
tools`, `uv run basedpyright`, `uv run pytest`.

## TESTS

### Running the tests

The suite in `tests/` needs no dataset: synthetic stacks, a small phantom and brute-force references
stand in for the real data. It takes about 20 s.

```sh
uv run pytest                                   # everything
uv run pytest -q tests/quantification           # one folder
uv run pytest tests/server/test_app.py          # one file
uv run pytest -k "mesh and not glb"             # tests whose name matches
uv run pytest -x --lf                           # stop at the first failure; rerun the last failures
node --test tests/js/                           # the JavaScript tests alone (Node.js)
uv run pytest --cov=src/pybones --cov=tools --cov-report=term-missing   # coverage
PYBONES_UPDATE_GOLDEN=1 uv run pytest tests/regression                 # record a new baseline
```

`tests/regression` segments a small fixed phantom and compares its scores with
`tests/regression/golden_phantom.json`: any score moving by more than the tolerance (up or down) fails,
so that a change of results is always deliberate. After an intended change, record the new baseline
with the command above and commit it with the change.

The JavaScript checks run inside pytest when [Node.js](https://nodejs.org/) is installed (any recent
version; no `npm install`), and are skipped otherwise. The pages themselves are checked by hand in a
browser.

### What the tests enforce

Besides the behaviour of the code, the suite checks the project's rules, so that a review does not
have to:

| Rule | Test |
|---|---|
| Every function, private ones included, has a complete Google docstring | `tests/test_conventions.py` |
| Names are imported from the module that defines them, never through another | `tests/test_conventions.py` |
| No module, Python or JavaScript, is longer than 500 lines | `tests/test_conventions.py` |
| No function only forwards its parameters to another; no two functions share a body; no run of logic (3 statements) or of JavaScript (4 lines) is repeated, variable names aside | `tests/test_duplication.py` |
| No dead code: every function, class, method and constant of `src/` and `tools/` is used by them (not by the tests alone), every test helper by a test, every JavaScript export by another module | `tests/test_unused.py` |
| Every JavaScript function of both pages has a JSDoc comment; no `var`, loose equality, `innerHTML` or `eval`; every module parses | `tests/server/test_page_js.py` |
| The labels, job states and job kinds the page knows match Python's | `tests/server/test_page_js.py` |
| The vendored three.js matches its manifest; every import of the page resolves | `tests/server/test_page.py` |
| The phantom's scores do not move unnoticed | `tests/regression/test_golden.py` |
| Every setting has a description, and docs/CONFIGURATION.md matches the code | `tests/docs_tool/test_configuration.py` |

When one of them fails, fix the code rather than the rule: split the module, share the logic, write
the docstring.

## CHANGING A SETTING

Every tunable value lives in [src/pybones/config/default.json](src/pybones/config/default.json) (the
server's in [server.json](src/pybones/config/server.json)), and is validated by the pydantic schemas
of [src/pybones/config/](src/pybones/config/). The JSON file is the only source of the values: the
schemas hold no defaults. Sizes are physical (`_um`, `_um2`, `_um3` keys), converted to pixels once.
A new setting takes, together:

1. a schema field with its bounds and a `description` (the reference and its test need it);
2. its value in the JSON file;
3. a validation test in `tests/config/` (an out-of-range value is rejected);
4. `uv run python -m tools.docs`, which rewrites [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

Every segmentation records its configuration, and a new key makes the outputs written before it
unreadable (segment them again); it also changes the golden baseline's fingerprint (record it again,
[above](#running-the-tests)).

## VS CODE AND THE WEB PAGE

VS Code settings (basedpyright, Ruff, pytest), launch configurations (segment, serve, preview, mesh,
quantify, the annotation tool, the current test file) and recommended extensions are versioned in
`.vscode/`, and work the same on Linux, macOS and Windows: no interpreter path is written there. After
`uv sync`, VS Code selects the project's `.venv` by itself (check the interpreter shown in the status
bar; *Python: Select Interpreter* picks it otherwise), and basedpyright finds it through
`[tool.basedpyright]` in `pyproject.toml`.

The web page lives in `src/pybones/server/static/`: plain HTML, CSS and JavaScript modules, no build
step, no package manager. three.js (MIT) is vendored under `static/vendor/three/`: only the files the
page imports, at the same paths as in the npm package, with a `manifest.json` recording the version,
the npm integrity hash and each file's SHA-256. `tests/server/test_page.py` checks those hashes, and
that every import of every module resolves to a served file. To update three.js: download the new
package from the npm registry, check its integrity against the registry's, copy the same files, and
rewrite the manifest; the test then confirms the set is complete.

The `node:test` suites in `tests/js/` cover the page's pure code: formatting helpers, the HTTP client
against a stubbed `fetch` and `EventSource`, the "segment again" settings difference, and the 3D
view's memory helpers.

## TOOLS

Run from the repository root (they are not part of the installed package):

```sh
# Every check of the project: formatting, lint, types, tests (see CHECKING A CHANGE)
python -m tools.check

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

# The configuration reference, docs/CONFIGURATION.md, from the schemas and default values
python -m tools.docs
```

Each tool has its own `config.json`, validated like the main one.

## GENERATED DOCUMENTATION

Some documentation is written from the code or the measurements, never by hand:

- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**: every setting, its default, the values it
  accepts and its description, from the schemas and the JSON files (`python -m tools.docs`; a test
  fails when it is out of date).
- **The README's tables** (phantom scores, mesh files) and **screenshots 01 to 04**: written by
  `python -m tools.report figures`.
- **Measured numbers in the Markdown files** (README, INSTALL, this guide): a number the report
  measures sits between two comments that name it, invisible once rendered; in the source, the
  segmentation's peak memory reads `<!-- pb:PeakGB -->1.7<!-- /pb -->` (GB). The name is one of the
  report's values (`PeakGB` is `\pbValuePeakGB` in `docs/report/data/values.tex`).
  `python -m tools.report figures` rewrites every such number; a name it does not know stops it. Mark
  a measured number this way rather than typing it.

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
history; nothing is typed by hand. It needs **uv** (each past version runs in its own environment) and
**git**. From the repository root, with `data/data_set.zip` in place:

```sh
# 1. Render the synthetic phantom (the stack itself is segmented by step 2, into its own folders)
uv run python -m tools.phantom generate -o output/phantom

# 2. Run every milestone version listed in tools/report/config.json (each in its own git worktree
#    and environment) on the phantom, and on the real data where figures need it; score them and
#    keep a small extract of each (docs/report/data/runs/). Versions whose commit is not in this
#    repository (the development history is not published) keep their archived scores and extracts;
#    so do the 2018 masks, which are not published either.
uv run python -m tools.report history

# 3. Render the figures, plot data, history table, measured values (values.tex), the README's tables,
#    screenshots 01-04 and the marked values of the Markdown files; mesh today's run in every format;
#    generate, segment and measure the branching phantom and measure the sample's vessel network
#    (chapter 6); time each progress stage of the segmentation, a mesh and the quantification, and
#    warn about a stage weight that no longer matches its share (stage_weight_tolerance). About three
#    minutes.
uv run python -m tools.report figures

# 4. Build the PDF (settings in docs/report/latexmkrc; the PDF is copied to docs/report/report.pdf)
cd docs/report && latexmk report.tex
```

The figures read the extracts, never the full runs of past versions, so the report can be rebuilt
and updated from a fresh clone: only the current code is re-run. Step 2 takes a few minutes the first
time (one environment per version, then cached); later runs reuse the stored outputs unless `--force`
is given, or unless a run's stored stats are outdated. Old worktrees can be removed with
`rm -rf output/report/worktrees && git worktree prune`.

- **Timings** are only comparable when every version runs on the same machine, in the same power
  profile, with nothing else running: each run records its machine and power profile, and the report
  prints them. To time every version again, delete their stats
  (`rm output/report/runs/*/*/stats.json`) and run step 2 alone.
- **After a change of the current code** (new version, new configuration key), its stored run is
  refused: `rm -rf output/report/runs/current` before step 2 re-runs it, past versions untouched.
- **Screenshots 05 to 07** of the README (annotation page, web page, Network tab) are taken by hand
  in a browser; the report includes them as they are.
