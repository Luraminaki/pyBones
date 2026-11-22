r"""Measured values in the Markdown documents, rewritten from the report's own values.

A value is marked in the text as ``<!-- pb:Name -->value<!-- /pb -->``: the markers are
invisible once rendered, and ``Name`` is one of the report's macros (``\pbValueName`` in
``values.tex``). :func:`markdown_values` rewrites every marked value of the documents listed
in ``value_documents``, so the README and the guides say what the report measured, never an
older number typed by hand.
"""

import re
from typing import TYPE_CHECKING

from tools.report.latex import LATEX_SPECIAL, VALUES_TEX

if TYPE_CHECKING:
    from pathlib import Path

    from tools.report.context import Context

#: A marked value of a Markdown document: its macro name, then the value, on one line.
_MARKED = re.compile(r"<!-- pb:(\w+) -->.*?<!-- /pb -->")
#: A macro of ``values.tex``: its name (after ``pbValue``) and its LaTeX text.
_MACRO = re.compile(r"^\\newcommand\{\\pbValue(\w+)\}\{(.*)\}$", re.MULTILINE)
#: LaTeX that is only typesetting: the thousands separator's braces, math mode.
_TYPESETTING = (("{,}", ","), ("$", ""))


def plain_text(latex: str) -> str:
    """Turn a macro's LaTeX text into plain text.

    Args:
        latex: The macro's text, as written in ``values.tex``.

    Returns:
        The same text without typesetting or escapes.
    """
    for typeset, plain in _TYPESETTING:
        latex = latex.replace(typeset, plain)
    for char, escaped in LATEX_SPECIAL.items():
        latex = latex.replace(escaped, char)
    return latex


def read_values(path: "Path") -> dict[str, str]:
    """Read the report's values back from ``values.tex``.

    Args:
        path: The ``values.tex`` file.

    Returns:
        Plain text of every ``pbValue`` macro, by name.
    """
    return {name: plain_text(text) for name, text in _MACRO.findall(path.read_text())}


def fill_values(text: str, values: dict[str, str]) -> str:
    """Rewrite every marked value of a document.

    Args:
        text: The document.
        values: The report's values (see :func:`read_values`).

    Returns:
        The document, each marked value replaced by today's.

    Raises:
        ValueError: If a marker names no value of the report.
    """
    unknown = sorted({name for name in _MARKED.findall(text) if name not in values})
    if unknown:
        raise ValueError(f"no report value named {', '.join(unknown)}")
    return _MARKED.sub(lambda m: f"<!-- pb:{m[1]} -->{values[m[1]]}<!-- /pb -->", text)


def markdown_values(ctx: "Context") -> None:
    """Rewrite the marked values of every listed Markdown document (after :func:`values`).

    Args:
        ctx: Figure context.
    """
    values = read_values(ctx.data / VALUES_TEX)
    for name in ctx.settings.value_documents:
        path = ctx.repo / name
        _ = path.write_text(fill_values(path.read_text(), values))
