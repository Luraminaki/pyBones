"""No duplicated code and no needless wrappers, in Python and JavaScript.

What a review used to catch by eye, checked on every run:

* **Pure forwarders**: a function whose whole body is ``return other(its own parameters, in
  order)`` adds a name and nothing else; callers should call ``other``. Decorated functions
  (caching, properties, validators...), special methods and tests are exempt: they add behaviour.
* **Identical functions**: two functions with the same body, variable names aside.
* **Repeated logic**: the same :data:`RUN_STATEMENTS` statements in a row in two places, variable
  names aside, is logic to share. Runs of plain calls (``x = f(a)``, ``g(b)``, arguments that
  are names or constants) are wiring, not logic, and are allowed: entry points start alike.
* **Repeated script lines**: the same :data:`RUN_LINES` meaningful lines in a row in two places
  of the project's JavaScript.

Size thresholds keep one-liners and signatures out: only blocks with some substance count.
"""

import ast
import re
from collections import defaultdict
from typing import TYPE_CHECKING, override

from tests.test_conventions import ROOT, SOURCE_DIRS, project_files, python_files

if TYPE_CHECKING:
    from pathlib import Path

#: Statements in a row that make a repeated run of logic.
RUN_STATEMENTS = 3
#: Size (AST dump characters) from which a run, or a function body, has substance.
MIN_RUN_SIZE = 700
MIN_BODY_SIZE = 600
#: Meaningful script lines in a row that make a repeated run, and their smallest total length.
RUN_LINES = 4
MIN_LINES_SIZE = 80
#: Script lines that carry no logic: comments, imports, lone brackets, bare returns.
_NOISE_LINE = re.compile(r"^(//|\*|/\*|import |export \{|[}\])]+;?,?$|\{$|\} else \{$|return;?$)")
_IMPLICIT = {"self", "cls"}
_TEST_PREFIX = "test_"
_DUNDER = "__"
SCRIPT_SUFFIXES = (".js", ".mjs")

Function = ast.FunctionDef | ast.AsyncFunctionDef


class _Renamer(ast.NodeTransformer):
    """Rename every variable and parameter by order of appearance (``v0``, ``v1``...)."""

    def __init__(self) -> None:
        """Start with no name seen."""
        self.names: dict[str, str] = {}

    def _rename(self, name: str) -> str:
        """Give a name its placeholder.

        Args:
            name: Original name.

        Returns:
            Its placeholder, the same for every use.
        """
        return self.names.setdefault(name, f"v{len(self.names)}")

    @override
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename a variable.

        Args:
            node: The name.

        Returns:
            The renamed name.
        """
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)


def _body(function: Function) -> list[ast.stmt]:
    """Take a function's statements, its docstring left out.

    Args:
        function: The function.

    Returns:
        Its statements.
    """
    body = function.body
    first = body[0] if body else None
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    return body[1:] if is_docstring else body


def _shape(statements: list[ast.stmt]) -> str:
    """Describe statements independently of their variable names.

    Args:
        statements: Statements.

    Returns:
        Their AST dump with variables renamed by order of appearance.
    """
    renamer = _Renamer()
    copies = [renamer.visit(ast.parse(ast.unparse(s)).body[0]) for s in statements]
    return ast.dump(ast.Module(body=copies, type_ignores=[]))


def _is_plain_call(node: ast.expr | None) -> bool:
    """Tell whether an expression is a call whose arguments are all names or constants.

    Args:
        node: Expression (or ``None``).

    Returns:
        Whether it is such a call, or a tuple of such calls.
    """
    if isinstance(node, ast.Tuple):
        return all(_is_plain_call(e) for e in node.elts)
    if not isinstance(node, ast.Call):
        return False
    arguments = [*node.args, *(k.value for k in node.keywords)]
    return all(isinstance(a, (ast.Name, ast.Attribute, ast.Constant)) for a in arguments)


def _is_wiring(statements: list[ast.stmt]) -> bool:
    """Tell whether statements are only plain calls, assigned or not (no logic).

    Args:
        statements: Statements.

    Returns:
        Whether every one is ``x = f(...)``, ``x, y = f(...), g(...)`` or ``f(...)``.
    """
    return all(
        isinstance(s, (ast.Assign, ast.Expr)) and _is_plain_call(s.value) for s in statements
    )


def _functions() -> "list[tuple[Path, Function]]":
    """List every function of the project's Python files.

    Returns:
        ``(file, function)`` pairs.
    """
    return [
        (path, node)
        for path in python_files()
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _where(path: "Path", node: ast.stmt) -> str:
    """Name a place in the project.

    Args:
        path: File.
        node: Statement or function there.

    Returns:
        ``file:line``, relative to the repository.
    """
    return f"{path.relative_to(ROOT)}:{node.lineno}"


def _repeated(groups: "dict[str, set[str]]") -> list[str]:
    """Keep the groups found in more than one place.

    Args:
        groups: Places by shape.

    Returns:
        One line per repeated shape, listing its places.
    """
    return [" | ".join(sorted(places)) for places in groups.values() if len(places) > 1]


def test_no_function_only_forwards_its_arguments() -> None:
    """No function's whole body is a call passing on its own parameters, unchanged."""
    found: list[str] = []
    for path, function in _functions():
        body = _body(function)
        call = body[0].value if len(body) == 1 and isinstance(body[0], ast.Return) else None
        exempt = function.decorator_list or function.name.startswith((_DUNDER, _TEST_PREFIX))
        if exempt or not isinstance(call, ast.Call):
            continue
        arguments = function.args
        parameters = [
            a.arg
            for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            if a.arg not in _IMPLICIT
        ]
        passed = [a.id for a in call.args if isinstance(a, ast.Name)] + [
            k.arg for k in call.keywords if isinstance(k.value, ast.Name) and k.value.id == k.arg
        ]
        if (
            parameters
            and passed == parameters
            and len(call.args) + len(call.keywords) == len(parameters)
        ):
            found.append(f"{_where(path, function)} {function.name} -> {ast.unparse(call)}")
    assert found == []


def test_no_two_functions_share_a_body() -> None:
    """No two functions do the same thing under different names."""
    groups: dict[str, set[str]] = defaultdict(set)
    for path, function in _functions():
        shape = _shape(_body(function))
        if len(shape) >= MIN_BODY_SIZE:
            groups[shape].add(f"{_where(path, function)} {function.name}")
    assert _repeated(groups) == []


def test_no_logic_is_repeated() -> None:
    """No run of statements with logic appears in two places (wiring aside)."""
    groups: dict[str, set[str]] = defaultdict(set)
    for path, function in _functions():
        body = _body(function)
        for start in range(len(body) - RUN_STATEMENTS + 1):
            run = body[start : start + RUN_STATEMENTS]
            shape = _shape(run)
            if len(shape) >= MIN_RUN_SIZE and not _is_wiring(run):
                groups[shape].add(f"{_where(path, run[0])} {function.name}")
    assert _repeated(groups) == []


def test_no_script_lines_are_repeated() -> None:
    """No run of meaningful JavaScript lines appears in two places."""
    groups: dict[str, set[str]] = defaultdict(set)
    scripts = [p for p in project_files() if p.suffix in SCRIPT_SUFFIXES]
    for path in scripts:
        lines = [
            (number, text)
            for number, raw in enumerate(path.read_text().splitlines(), start=1)
            if (text := raw.strip()) and not _NOISE_LINE.match(text)
        ]
        for start in range(len(lines) - RUN_LINES + 1):
            run = [text for _, text in lines[start : start + RUN_LINES]]
            if sum(map(len, run)) >= MIN_LINES_SIZE:
                groups["\n".join(run)].add(f"{path.relative_to(ROOT)}:{lines[start][0]}")
    assert _repeated(groups) == []


def test_the_scans_cover_the_project() -> None:
    """Every source directory is scanned (a renamed one would silently escape the checks)."""
    assert {p.relative_to(ROOT).parts[0] for p in python_files()} == set(SOURCE_DIRS)
