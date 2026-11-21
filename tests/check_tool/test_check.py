"""Tests of the check runner (:mod:`tools.check`): the plan, never the checks themselves."""

import importlib.util

import pytest

from tools.check.__main__ import CHECKS, FIXES, plan


def test_every_check_runs_by_default() -> None:
    """Formatting, lint, types and tests, in that order, nothing modified."""
    assert [s.name for s in plan(fix=False)] == ["format", "lint", "types", "tests"]
    assert all("--fix" not in s.args for s in plan(fix=False))
    assert "--check" in plan(fix=False)[0].args


def test_fix_applies_ruff_then_checks_the_rest() -> None:
    """``--fix`` reformats and fixes first, then checks types and runs the tests."""
    steps = plan(fix=True)
    assert steps[: len(FIXES)] == FIXES
    assert steps[len(FIXES) :] == CHECKS[len(FIXES) :]
    assert [s.name for s in steps] == [s.name for s in CHECKS]


@pytest.mark.parametrize("module", sorted({s.module for s in (*CHECKS, *FIXES)}))
def test_every_checker_is_installed(module: str) -> None:
    """Each checker is a module of the development environment (``uv sync`` installs it)."""
    assert importlib.util.find_spec(module) is not None
