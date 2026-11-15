"""Project conventions that Ruff cannot check, enforced on every Python file.

* Every function and method, private ones included, has a Google docstring whose
  ``Args:`` section names every parameter and which has ``Returns:`` /
  ``Yields:`` / ``Raises:`` sections when the function returns a value, yields or
  raises. Test functions only need a summary line (their name and arguments are
  fixtures, documented where they are defined).
* Names are imported from the module that defines them, never through another
  module that happens to import them.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "tools", "tests")
#: First-party top-level packages and where they live.
PACKAGES = {"pybones": ROOT / "src" / "pybones", "tools": ROOT / "tools", "tests": ROOT / "tests"}
#: Parameters never documented.
IMPLICIT_PARAMETERS = {"self", "cls"}
#: Decorators of members documented as attributes (no Returns section).
PROPERTY_DECORATORS = {"property", "cached_property"}
#: Decorators of pydantic validators (raising is how they report errors).
VALIDATOR_DECORATORS = {"field_validator", "model_validator"}
TEST_PREFIX = "test_"
_SECTION_ENTRY = re.compile(r"^\s{4}(\*{0,2}\w+)(?: \(.*?\))?:", re.MULTILINE)

Function = ast.FunctionDef | ast.AsyncFunctionDef


def python_files() -> list[Path]:
    """List the project's Python files.

    Returns:
        Every ``.py`` file of :data:`SOURCE_DIRS`, sorted.
    """
    return sorted(p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.py"))


def own_nodes(function: Function) -> list[ast.AST]:
    """Collect the nodes of a function body, excluding nested functions, lambdas and classes.

    Args:
        function: Function to walk.

    Returns:
        The nodes that belong to ``function`` itself.
    """
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    stack: list[ast.AST] = list(ast.iter_child_nodes(function))
    nodes: list[ast.AST] = []
    while stack:
        node = stack.pop()
        if isinstance(node, nested):
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def decorator_names(function: Function) -> set[str]:
    """Name the decorators of a function (``@x``, ``@a.x`` and ``@x(...)`` all give ``x``).

    Args:
        function: Decorated function.

    Returns:
        The decorator names.
    """
    names: set[str] = set()
    for decorator in function.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def parameters(function: Function) -> list[str]:
    """List the parameters a docstring must document.

    Args:
        function: Function to inspect.

    Returns:
        Parameter names, ``*args`` and ``**kwargs`` with their stars.
    """
    args = function.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return [n for n in names if n not in IMPLICIT_PARAMETERS]


def section(docstring: str, title: str) -> str | None:
    """Extract a Google docstring section.

    Args:
        docstring: Cleaned docstring.
        title: Section title, e.g. ``Args``.

    Returns:
        The indented body of the section, or ``None`` if absent.
    """
    match = re.search(rf"^{title}:\n((?:[ \t]+.*\n?|\n)*)", docstring, re.MULTILINE)
    return match.group(1) if match else None


def docstring_problems(function: Function, is_test_file: bool) -> list[str]:
    """Check one function's docstring against the conventions.

    Args:
        function: Function to check.
        is_test_file: Whether the function lives in a test module.

    Returns:
        One message per problem (empty when the docstring is complete).
    """
    doc = ast.get_docstring(function)
    if not doc:
        return ["no docstring"]
    if is_test_file and function.name.startswith(TEST_PREFIX):
        return []
    problems: list[str] = []
    expected = parameters(function)
    if expected:
        args = section(doc, "Args")
        listed = set(_SECTION_ENTRY.findall(args)) if args is not None else set()
        missing = [p for p in expected if p not in listed and p.lstrip("*") not in listed]
        if missing:
            problems.append(f"Args section misses {missing}")
    nodes = own_nodes(function)
    decorators = decorator_names(function)
    yields = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in nodes)
    returns = any(
        isinstance(n, ast.Return)
        and n.value is not None
        and not (isinstance(n.value, ast.Constant) and n.value.value is None)
        for n in nodes
    )
    raises = any(isinstance(n, ast.Raise) and n.exc is not None for n in nodes)
    if yields and section(doc, "Yields") is None:
        problems.append("no Yields section")
    is_attribute = bool(decorators & PROPERTY_DECORATORS) or function.name == "__init__"
    if returns and not yields and not is_attribute and section(doc, "Returns") is None:
        problems.append("no Returns section")
    if raises and not decorators & VALIDATOR_DECORATORS and section(doc, "Raises") is None:
        problems.append("no Raises section")
    return problems


def module_path(module: str) -> Path | None:
    """Locate a first-party module.

    Args:
        module: Dotted module name.

    Returns:
        Its file (``__init__.py`` for a package), or ``None`` if not first-party.
    """
    head, *rest = module.split(".")
    if head not in PACKAGES:
        return None
    base = PACKAGES[head].joinpath(*rest)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def defined_names(path: Path) -> set[str]:
    """List the names a module defines at its top level (not the ones it imports).

    Args:
        path: Module file.

    Returns:
        Functions, classes, type aliases and assigned names.
    """
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.TypeAlias):
            names.add(node.name.id)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


def relative(path: Path) -> str:
    """Express a path relative to the repository, for readable test ids.

    Args:
        path: Project file.

    Returns:
        The repository-relative path.
    """
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("path", python_files(), ids=relative)
def test_every_function_has_a_complete_docstring(path: Path) -> None:
    """Every function documents its arguments, return value, yields and exceptions."""
    is_test_file = path.name.startswith(TEST_PREFIX)
    problems = [
        f"line {node.lineno} {node.name}: {problem}"
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for problem in docstring_problems(node, is_test_file)
    ]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", python_files(), ids=relative)
def test_names_are_imported_from_their_defining_module(path: Path) -> None:
    """``from X import Y`` only when ``Y`` is defined in ``X`` (or is a submodule of ``X``)."""
    problems: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom) or node.module is None or node.level:
            continue
        source = module_path(node.module)
        if source is None:
            continue
        names = defined_names(source)
        problems += [
            f"line {node.lineno}: {alias.name} is not defined in {node.module}"
            for alias in node.names
            if alias.name not in names and module_path(f"{node.module}.{alias.name}") is None
        ]
    assert not problems, "\n".join(problems)
