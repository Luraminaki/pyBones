"""Typed, validated arguments of ``python -m tools.report``."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from pybones.cli.arguments import ArgsModel, nest_command

_GLOBAL_OPTIONS = ("settings",)


class Command(StrEnum):
    """Sub-commands of the tool."""

    HISTORY = "history"
    FIGURES = "figures"


class HistoryArgs(ArgsModel):
    """``history [--force]``: run and score every version, refresh the extracts."""

    command: Literal[Command.HISTORY]
    force: bool


class FiguresArgs(ArgsModel):
    """``figures``: render the figures, plot data, values and README screenshots."""

    command: Literal[Command.FIGURES]


ReportCommand = Annotated[HistoryArgs | FiguresArgs, Field(discriminator="command")]


class ReportInvocation(ArgsModel):
    """A complete, validated invocation of the report tool.

    Attributes:
        settings: Report settings file.
        args: The sub-command and its arguments.
    """

    settings: Path
    args: ReportCommand

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any]) -> "ReportInvocation":
        """Validate the dict of an ``argparse.Namespace``.

        Args:
            namespace: ``vars()`` of the parsed namespace.

        Returns:
            The validated invocation.

        Raises:
            pydantic.ValidationError: If an argument is invalid.
        """
        return cls.model_validate(nest_command(namespace, _GLOBAL_OPTIONS))
