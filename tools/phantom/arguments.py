"""Typed, validated arguments of ``python -m tools.phantom``."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from pybones.cli.arguments import ArgsModel, nest_command

_GLOBAL_OPTIONS = ("settings", "config")


class Command(StrEnum):
    """Sub-commands of the tool."""

    GENERATE = "generate"
    SCORE = "score"


class GenerateArgs(ArgsModel):
    """``generate -o OUTPUT``: render the phantom slices and ground truth."""

    command: Literal[Command.GENERATE]
    output: Path


class ScoreArgs(ArgsModel):
    """``score SEGMENTATION --phantom DIR``: score a segmentation of the phantom."""

    command: Literal[Command.SCORE]
    segmentation: Path
    phantom: Path


PhantomCommand = Annotated[GenerateArgs | ScoreArgs, Field(discriminator="command")]


class PhantomInvocation(ArgsModel):
    """A complete, validated invocation of the phantom tool.

    Attributes:
        settings: Phantom settings file.
        config: Optional pyBones configuration (voxel size).
        args: The sub-command and its arguments.
    """

    settings: Path
    config: Path | None
    args: PhantomCommand

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any]) -> "PhantomInvocation":
        """Validate the dict of an ``argparse.Namespace``.

        Args:
            namespace: ``vars()`` of the parsed namespace.

        Returns:
            The validated invocation.

        Raises:
            pydantic.ValidationError: If an argument is invalid.
        """
        return cls.model_validate(nest_command(namespace, _GLOBAL_OPTIONS))
