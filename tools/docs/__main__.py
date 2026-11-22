"""``python -m tools.docs``: write the documentation that comes from the code.

Today, the configuration reference ``docs/CONFIGURATION.md`` (see
:mod:`tools.docs.configuration`). Run it after changing a setting's schema or default value;
``tests/docs_tool`` fails until the file matches the code. Run from the repository root.
"""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pybones.config.loader import load_config
from pybones.utils.logs import configure_logging
from tools.docs.configuration import render_configuration

if TYPE_CHECKING:
    from collections.abc import Sequence

PROGRAM_NAME = "docs"
#: The configuration reference, relative to the repository root.
CONFIGURATION_FILE = Path("docs/CONFIGURATION.md")

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The ``python -m tools.docs`` parser.
    """
    return argparse.ArgumentParser(
        prog="python -m tools.docs",
        description="Write docs/CONFIGURATION.md from the configuration schemas and defaults.",
    )


def main(argv: "Sequence[str] | None" = None) -> None:
    """Write the documentation generated from the code.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).
    """
    _ = build_parser().parse_args(argv)
    _ = configure_logging(load_config().logging, PROGRAM_NAME)
    _ = CONFIGURATION_FILE.write_text(render_configuration())
    logger.info("Wrote %s", CONFIGURATION_FILE)


if __name__ == "__main__":
    main()
