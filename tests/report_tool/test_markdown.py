"""Tests of the marked values of the Markdown documents (:mod:`tools.report.markdown`)."""

from typing import TYPE_CHECKING

import pytest

from tools.report.markdown import fill_values, plain_text, read_values

if TYPE_CHECKING:
    from pathlib import Path

VALUES = {"Seconds": "24", "PeakGB": "1.7"}


def test_every_marked_value_is_rewritten() -> None:
    """Each marker gets today's value, several on a line, the text around untouched."""
    text = "in ~<!-- pb:Seconds -->25<!-- /pb --> s, <!-- pb:PeakGB -->1.6<!-- /pb --> GB peak\n"
    assert fill_values(text, VALUES) == (
        "in ~<!-- pb:Seconds -->24<!-- /pb --> s, <!-- pb:PeakGB -->1.7<!-- /pb --> GB peak\n"
    )


def test_an_unknown_marker_is_an_error() -> None:
    """A marker naming no report value (a typo) stops the report tool."""
    with pytest.raises(ValueError, match="Secnds"):
        _ = fill_values("<!-- pb:Secnds -->25<!-- /pb -->", VALUES)


def test_macros_are_read_back_as_plain_text(tmp_path: "Path") -> None:
    """Thousands separators, math mode and escapes of values.tex become plain text."""
    tex = tmp_path / "values.tex"
    _ = tex.write_text(
        "\\newcommand{\\pbValueNetBranches}{9{,}514}\n"
        + "\\newcommand{\\pbValueNetSegLengthErrorPct}{$-3.0$}\n"
        + "\\newcommand{\\pbNumThrBone}{9382}\n"
    )
    assert read_values(tex) == {"NetBranches": "9,514", "NetSegLengthErrorPct": "-3.0"}
    assert plain_text("R\\&D 50\\%") == "R&D 50%"
