"""``python -m tools.check``: run every check of the project, then say which failed.

The checks run in order and all of them run, so one command shows every problem:

1. **format**: Ruff's formatting (``--fix``: the files are reformatted);
2. **lint**: Ruff's rules, the project conventions among them (``--fix``: safe fixes applied);
3. **types**: basedpyright, with the settings of ``pyproject.toml``;
4. **tests**: the whole pytest suite, the JavaScript checks included where Node.js is installed.

Each runs with the interpreter running this tool (``uv run python -m tools.check``), so the
versions are the locked ones. The exit status is 1 if any check failed.
"""

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pybones.config.loader import load_config
from pybones.utils.logs import configure_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

PROGRAM_NAME = "check"
#: Directories holding the project's Python code.
CODE_DIRS = ("src", "tests", "tools")
FAILED_EXIT_STATUS = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Step:
    """One check.

    Attributes:
        name: What it checks.
        module: Python module that runs it (``python -m <module>``).
        args: Its arguments.
    """

    name: str
    module: str
    args: tuple[str, ...]


#: The checks, in order.
CHECKS = (
    Step("format", "ruff", ("format", "--check", *CODE_DIRS)),
    Step("lint", "ruff", ("check", *CODE_DIRS)),
    Step("types", "basedpyright", ()),
    Step("tests", "pytest", ("-q",)),
)
#: What ``--fix`` runs instead of the first two checks.
FIXES = (
    Step("format", "ruff", ("format", *CODE_DIRS)),
    Step("lint", "ruff", ("check", "--fix", *CODE_DIRS)),
)


def plan(fix: bool) -> tuple[Step, ...]:
    """Choose the steps to run.

    Args:
        fix: Apply Ruff's formatting and safe fixes instead of only reporting them.

    Returns:
        The steps, in order.
    """
    return (*FIXES, *CHECKS[len(FIXES) :]) if fix else CHECKS


def run(steps: "Sequence[Step]") -> list[str]:
    """Run every step, whatever the earlier ones gave.

    Args:
        steps: Steps to run.

    Returns:
        The names of the steps that failed.
    """
    failed: list[str] = []
    for step in steps:
        logger.info("%s: python -m %s %s", step.name, step.module, " ".join(step.args))
        t0 = time.perf_counter()
        status = subprocess.run([sys.executable, "-m", step.module, *step.args], check=False)
        seconds = time.perf_counter() - t0
        outcome = "ok" if status.returncode == 0 else f"FAILED (status {status.returncode})"
        logger.info("%s: %s in %.1f s", step.name, outcome, seconds)
        if status.returncode != 0:
            failed.append(step.name)
    return failed


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The ``python -m tools.check`` parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tools.check",
        description="Run every check of the project: formatting, lint, types, tests.",
    )
    _ = parser.add_argument(
        "--fix", action="store_true", help="reformat and apply Ruff's safe fixes first"
    )
    return parser


def main(argv: "Sequence[str] | None" = None) -> None:
    """Run the checks and exit with status 1 if any failed.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Raises:
        SystemExit: If a check failed.
    """
    fix = bool(build_parser().parse_args(argv).fix)
    _ = configure_logging(load_config().logging, PROGRAM_NAME)
    failed = run(plan(fix))
    if failed:
        logger.error("Failed: %s", ", ".join(failed))
        raise SystemExit(FAILED_EXIT_STATUS)
    logger.info("Every check passed")


if __name__ == "__main__":
    main()
