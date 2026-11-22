"""Checks of the front-end JavaScript (web page, annotation page), partly run with Node.

* every module of the page parses (``node --check``);
* the JavaScript unit tests of ``tests/js`` pass (``node --test``, no dependency);
* the values the page repeats from Python (labels, finished job states) match it;
* every function of the page has a JSDoc comment, as every Python function has a docstring;
* no ``var``, loose equality, or HTML / code built from strings.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from pybones.api.job_status import (
    FINISHED_STATES,
    MESH_JOB,
    QUANTIFICATION_JOB,
    SEGMENTATION_JOB,
)
from pybones.config.fields import Effect, Level
from pybones.config.schema import PyBonesConfig
from pybones.segmentation.labels import FINAL_LABELS, Label
from pybones.server.app import STATIC_DIR
from pybones.server.schemas import SETTINGS_REQUESTS, is_kept
from tests.config.test_effects import settings

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
JS_DIR = STATIC_DIR / "js"
JS_TESTS = Path(__file__).parents[1] / "js"
#: Front-end modules of the annotation tool (served by its own server).
ANNOTATION_JS_DIR = Path(__file__).parents[2] / "tools" / "annotation" / "static"
NODE_TIMEOUT_S = 60
#: A function declaration (``function f(``, ``async function f(``, ``export ...``) or a
#: top-level arrow function (``const f = (...) =>``), with what precedes it.
FUNCTION = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*\("
    + r"|^(?P<arrow_indent>)(?:export\s+)?const\s+(?P<arrow>\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
    re.MULTILINE,
)
#: A JSDoc block ending right before a declaration (blank lines not allowed in between).
JSDOC_END = re.compile(r"\*/\s*$")


def own_modules() -> list[Path]:
    """List the project's front-end modules: the web page's and the annotation page's.

    The vendored three.js is not ours to document.

    Returns:
        The ``.js`` files of both pages.
    """
    return sorted([*JS_DIR.glob("*.js"), *ANNOTATION_JS_DIR.glob("*.js")])


def node(*args: str) -> subprocess.CompletedProcess[str]:
    """Run Node.

    Args:
        args: Its arguments.

    Returns:
        The finished process (output captured).
    """
    assert NODE is not None
    return subprocess.run(
        [NODE, *args], capture_output=True, text=True, timeout=NODE_TIMEOUT_S, check=False
    )


def js_value(module: Path, name: str) -> object:
    """Import a module in Node and read one of its exports, as JSON.

    Args:
        module: Module file.
        name: Exported name (a Set is read as a list).

    Returns:
        The value.
    """
    script = (
        f"const m = await import({json.dumps(module.as_uri())});"
        + f"const v = m[{json.dumps(name)}];"
        + "console.log(JSON.stringify(v instanceof Set ? [...v] : v));"
    )
    result = node("--input-type=module", "-e", script)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
@pytest.mark.parametrize("module", own_modules(), ids=lambda p: p.name)
def test_every_module_parses(module: Path) -> None:
    """Each page module is valid JavaScript."""
    result = node("--check", str(module))
    assert result.returncode == 0, result.stderr


@needs_node
def test_the_javascript_unit_tests_pass() -> None:
    """``node --test tests/js`` reports no failure."""
    result = node("--test", str(JS_TESTS))
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_the_page_draws_the_final_labels() -> None:
    """The page's label list is the final labels of a segmentation, in Python's order."""
    expected = [label.key for label in FINAL_LABELS if label != Label.BACKGROUND]
    assert js_value(JS_DIR / "ui.js", "LABELS") == expected


@needs_node
def test_the_page_knows_when_a_job_is_over() -> None:
    """The page's finished job states are those of the job manager."""
    states = js_value(JS_DIR / "api.js", "FINISHED_STATES")
    assert isinstance(states, list)
    assert set(states) == {str(state) for state in FINISHED_STATES}


@needs_node
def test_the_page_names_effects_and_levels_like_python() -> None:
    """The settings form knows the same effects (widest first) and levels as the schemas."""
    assert js_value(JS_DIR / "settings_model.js", "EFFECT_ORDER") == [e.value for e in Effect]
    levels = js_value(JS_DIR / "settings_model.js", "LEVELS")
    assert isinstance(levels, dict)
    assert set(levels.values()) == {level.value for level in Level}


@needs_node
@pytest.mark.parametrize("kind", sorted(SETTINGS_REQUESTS))
def test_each_form_reads_every_setting_its_job_may_change(tmp_path: Path, kind: str) -> None:
    """On the real schema, a job's form finds exactly the settings it may change, rightly."""
    request = SETTINGS_REQUESTS[kind]
    schema_file = tmp_path / "schema.json"
    _ = schema_file.write_text(json.dumps(PyBonesConfig.model_json_schema()))
    roots, kept = sorted(request.SECTIONS), sorted(request.KEPT)
    script = (
        f"const m = await import({json.dumps((JS_DIR / 'settings_model.js').as_uri())});"
        + "const fs = await import('node:fs');"
        + f"const schema = JSON.parse(fs.readFileSync({json.dumps(str(schema_file))}, 'utf8'));"
        + f"const sections = m.readSections(schema, {json.dumps(roots)}, {json.dumps(kept)});"
        + "const fields = sections.flatMap((s) => s.fields);"
        + "console.log(JSON.stringify(fields.map((f) => [f.path, f.kind, f.effect])));"
    )
    result = node("--input-type=module", "-e", script)
    assert result.returncode == 0, result.stderr
    read = {path: (kind, effect) for path, kind, effect in json.loads(result.stdout)}
    expected = {
        path: effect
        for path, _, effect in settings(PyBonesConfig)
        if path.split(".")[0] in roots and not is_kept(path, kept)
    }
    assert {path: effect for path, (_, effect) in read.items()} == expected
    assert {kind for kind, _ in read.values()} <= {"boolean", "integer", "number", "choice", "json"}


@needs_node
def test_the_page_names_the_job_kinds_like_python() -> None:
    """The page's job kinds are the job manager's, one for one."""
    kinds = js_value(JS_DIR / "api.js", "JOB_KINDS")
    assert isinstance(kinds, dict)
    assert kinds == {
        "SEGMENTATION": SEGMENTATION_JOB,
        "MESH": MESH_JOB,
        "QUANTIFICATION": QUANTIFICATION_JOB,
    }


def undocumented(module: Path) -> list[str]:
    """Find the functions of a module without a JSDoc comment right above them.

    Args:
        module: Module file.

    Returns:
        ``line: name`` of each undocumented function.
    """
    text = module.read_text()
    missing: list[str] = []
    for match in FUNCTION.finditer(text):
        before = text[: match.start()].rstrip(" \t")
        if not JSDOC_END.search(before.removesuffix("\n")):
            line = text.count("\n", 0, match.start()) + 1
            missing.append(f"{line}: {match.group('name') or match.group('arrow')}")
    return missing


@pytest.mark.parametrize("module", own_modules(), ids=lambda p: p.name)
def test_every_function_has_a_jsdoc_comment(module: Path) -> None:
    """Like the Python docstrings: each function says what it does, its parameters, its result."""
    assert undocumented(module) == []


@pytest.mark.parametrize("module", own_modules(), ids=lambda p: p.name)
def test_every_module_starts_with_its_purpose(module: Path) -> None:
    """Each module opens with a comment saying what it is for."""
    assert module.read_text().lstrip().startswith(("//", "/*"))


#: Constructs the page never uses: ``var`` (function-scoped, hoisted: ``const`` / ``let``
#: instead), loose equality (``===`` / ``!==`` instead), and HTML parsing of strings
#: (``textContent`` and built elements instead, so no text can inject markup).
FORBIDDEN = {
    "var": re.compile(r"(?<![\w$.])var\s+[\w$\[{]"),
    "loose equality": re.compile(r"[^=!<>]==[^=]|!=[^=]"),
    "HTML from strings": re.compile(
        r"\.(?:innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write"
    ),
    "code from strings": re.compile(r"\beval\(|new Function\("),
}


def code_lines(module: Path) -> list[tuple[int, str]]:
    """List the lines of a module with their comments and strings blanked out.

    Args:
        module: Module file.

    Returns:
        ``(line number, code)`` pairs.
    """
    strings = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`")
    lines: list[tuple[int, str]] = []
    for number, line in enumerate(module.read_text().splitlines(), start=1):
        code = strings.sub("''", line)
        stripped = code.lstrip()
        if stripped.startswith(("//", "*", "/*")):
            continue
        lines.append((number, code.split("//", 1)[0]))
    return lines


@pytest.mark.parametrize("module", own_modules(), ids=lambda p: p.name)
def test_no_risky_javascript_constructs(module: Path) -> None:
    """No ``var``, no loose equality, no HTML or code built from strings."""
    found = [
        f"{number}: {name}"
        for number, code in code_lines(module)
        for name, pattern in FORBIDDEN.items()
        if pattern.search(code)
    ]
    assert found == []
