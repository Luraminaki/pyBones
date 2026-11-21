"""Checks of the page's JavaScript run with Node (skipped where Node is not installed).

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

from pybones.api.job_status import FINISHED_STATES
from pybones.segmentation.labels import FINAL_LABELS, Label
from pybones.server.app import STATIC_DIR

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
JS_DIR = STATIC_DIR / "js"
JS_TESTS = Path(__file__).parents[1] / "js"
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
    """List the page's own modules (the vendored three.js is not ours to document).

    Returns:
        The ``js/*.js`` files.
    """
    return sorted(JS_DIR.glob("*.js"))


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
