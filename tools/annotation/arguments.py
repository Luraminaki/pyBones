"""Typed, validated arguments of ``python -m tools.annotation``."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field

from pybones.cli.arguments import ArgsModel, nest_command

_GLOBAL_OPTIONS = ("settings", "config")

_DOT_NAMES = (".", "..")


def _not_a_dot_name(name: str) -> str:
    """Reject ``.`` and ``..``, which would point outside the workspace root.

    Args:
        name: Workspace name.

    Returns:
        ``name``, unchanged.

    Raises:
        ValueError: If ``name`` is ``.`` or ``..``.
    """
    if name in _DOT_NAMES:
        raise ValueError(f"{name!r} is not a workspace name")
    return name


#: A workspace name is one directory under the workspace root: no separator, no ``.``/``..``.
WorkspaceName = Annotated[
    str,
    Field(min_length=1, max_length=200, pattern=r"^[^/\\]+$"),
    AfterValidator(_not_a_dot_name),
]


class Command(StrEnum):
    """Sub-commands of the tool."""

    PREPARE = "prepare"
    REGENERATE = "regenerate"
    SERVE = "serve"
    SCORE = "score"


class _WorkspaceArgs(ArgsModel):
    """Base of the commands acting on one workspace."""

    name: WorkspaceName


class PrepareArgs(_WorkspaceArgs):
    """``prepare SEGMENTATION --name NAME``: choose crops and extract their images."""

    command: Literal[Command.PREPARE]
    segmentation: Path
    source: Path | None
    force: bool


class RegenerateArgs(_WorkspaceArgs):
    """``regenerate --name NAME``: rewrite the crop images of an existing workspace."""

    command: Literal[Command.REGENERATE]


class ServeArgs(_WorkspaceArgs):
    """``serve --name NAME [--open]``: serve the annotation page locally."""

    command: Literal[Command.SERVE]
    open: bool


class ScoreArgs(_WorkspaceArgs):
    """``score SEGMENTATION --name NAME``: score a segmentation against the annotations."""

    command: Literal[Command.SCORE]
    segmentation: Path


AnnotationCommand = Annotated[
    PrepareArgs | RegenerateArgs | ServeArgs | ScoreArgs, Field(discriminator="command")
]


class AnnotationInvocation(ArgsModel):
    """A complete, validated invocation of the annotation tool.

    Attributes:
        settings: Tool settings file.
        config: Optional pyBones configuration.
        args: The sub-command and its arguments.
    """

    settings: Path
    config: Path | None
    args: AnnotationCommand

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any]) -> "AnnotationInvocation":
        """Validate the dict of an ``argparse.Namespace``.

        Args:
            namespace: ``vars()`` of the parsed namespace.

        Returns:
            The validated invocation.

        Raises:
            pydantic.ValidationError: If an argument is invalid.
        """
        return cls.model_validate(nest_command(namespace, _GLOBAL_OPTIONS))
