"""``python -m tools.report {history,figures}`` (run from the repository root)."""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from pybones.cli.arguments import parse_invocation
from pybones.config.loader import load_config
from pybones.utils.logs import configure_logging
from tools.phantom.settings import load_settings as load_phantom_settings
from tools.phantom.storage import load_phantom
from tools.report.arguments import Command, FiguresArgs, HistoryArgs, ReportInvocation
from tools.report.context import Context
from tools.report.figures import (
    comparisons,
    overview,
    phantom_figure,
    tubularity_map,
    vessel_network,
)
from tools.report.history import history_rows, write_csv
from tools.report.latex import history_table, plot_data, readme_scores, values
from tools.report.meshes import mesh_files, readme_meshes
from tools.report.network import network_files
from tools.report.settings import DEFAULT_SETTINGS_PATH, load_settings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tools.report.settings import ReportSettings


#: Program name of the log files.
PROGRAM_NAME = "tools-report"

#: Figure steps, in order (``values`` reads the profile written by ``plot_data``).
FIGURE_STEPS: "tuple[Callable[[Context], None], ...]" = (
    overview,
    comparisons,
    vessel_network,
    tubularity_map,
    phantom_figure,
    plot_data,
    history_table,
    mesh_files,
    network_files,
    values,
    readme_scores,
    readme_meshes,
)


logger = logging.getLogger(__name__)


def cmd_history(settings: "ReportSettings", repo: Path, force: bool) -> None:
    """Run every configured version on the phantom (and real data) and tabulate the scores.

    Args:
        settings: Report settings.
        repo: Repository root.
        force: Re-run versions whose outputs exist.
    """
    phantom_settings = load_phantom_settings()
    phantom = load_phantom(repo / settings.phantom_dir, phantom_settings)
    rows = history_rows(settings, phantom, phantom_settings, repo, force)
    path = repo / settings.data_dir / settings.history_csv
    write_csv(rows, path)
    logger.info("Wrote %s", path)


def cmd_figures(settings: "ReportSettings", repo: Path) -> None:
    """Render the report figures, plot data, values and README screenshots.

    Args:
        settings: Report settings.
        repo: Repository root.
    """
    ctx = Context(
        settings=settings, config=load_config(), phantom_settings=load_phantom_settings(), repo=repo
    )
    for step in FIGURE_STEPS:
        step(ctx)
        logger.info("Rendered %s", step.__name__)
    logger.info("Wrote %s, %s and %s", ctx.figures, ctx.data, ctx.screenshots)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The ``python -m tools.report`` parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tools.report",
        description="Benchmark past versions and render the report figures.",
    )
    _ = parser.add_argument(
        "--settings", type=Path, default=DEFAULT_SETTINGS_PATH, help="report tool configuration"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(Command.HISTORY, help="run past versions (git worktrees) and score them")
    _ = p.add_argument("--force", action="store_true", help="re-run versions with existing outputs")
    _ = sub.add_parser(Command.FIGURES, help="render figures, plot data and values from the runs")
    return parser


def main(argv: "Sequence[str] | None" = None) -> None:
    """Parse and validate the arguments, then run the command from the repository root.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).
    """
    parser = build_parser()
    invocation = parse_invocation(parser, ReportInvocation.from_namespace, argv)
    settings = load_settings(invocation.settings)
    _ = configure_logging(load_config().logging, PROGRAM_NAME)
    repo = Path.cwd()
    match invocation.args:
        case HistoryArgs():
            cmd_history(settings, repo, invocation.args.force)
        case FiguresArgs():
            cmd_figures(settings, repo)
        case _:
            assert_never(invocation.args)


if __name__ == "__main__":
    main()
