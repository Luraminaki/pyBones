"""Typed, validated command-line arguments: one pydantic model per sub-command.

argparse handles the command line itself (syntax, help, usage messages); its
result is then validated into an :class:`Invocation`, so every command receives
a typed, immutable object instead of a loose ``argparse.Namespace``.
"""

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pybones.segmentation.labels import FOREGROUND_LABELS, Label, label_keys

#: Bounds of ``preview --scale`` (output scale factor).
MAX_PREVIEW_SCALE = 16.0


class ArgsModel(BaseModel):
    """Base of the argument models: unknown fields are rejected and instances are immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


#: Field of an invocation model holding the sub-command's arguments.
COMMAND_FIELD = "args"


def nest_command(namespace: dict[str, Any], global_options: tuple[str, ...]) -> dict[str, Any]:
    """Split an ``argparse`` namespace into global options and the sub-command's arguments.

    Shared by ``pybones`` and the tools, whose invocation models hold their global
    options as fields and the sub-command (a discriminated union on ``command``) in
    :data:`COMMAND_FIELD`.

    Args:
        namespace: ``vars()`` of the parsed namespace.
        global_options: Names of the options given before the sub-command.

    Returns:
        The global options, plus the remaining values under :data:`COMMAND_FIELD`.
    """
    values = dict(namespace)
    nested: dict[str, Any] = {name: values.pop(name) for name in global_options}
    nested[COMMAND_FIELD] = values
    return nested


def _foreground_label(value: Any) -> Label:
    """Parse a label key (``bone``, ``vessel``, ``burst``) into a foreground :class:`Label`.

    Args:
        value: Label key (command line) or label value.

    Returns:
        The foreground label.

    Raises:
        ValueError: If ``value`` names no label or a non-foreground one.
    """
    label = Label.from_key(value) if isinstance(value, str) else Label(value)
    if label not in FOREGROUND_LABELS:
        raise ValueError(f"choose one of {label_keys(FOREGROUND_LABELS)}")
    return label


class ConfigArgs(ArgsModel):
    """``pybones config``: print the effective configuration."""

    command: Literal["config"]


class SegmentArgs(ArgsModel):
    """``pybones segment SOURCE -o OUTPUT``."""

    command: Literal["segment"]
    source: Path = Field(description="Directory or zip archive of slices.")
    output: Path = Field(description="Output directory.")


class PreviewArgs(ArgsModel):
    """``pybones preview SEGMENTATION -z Z -o OUTPUT``."""

    command: Literal["preview"]
    segmentation: Path
    z: int = Field(ge=0, description="Slice index.")
    output: Path
    source: Path | None = Field(description="Raw stack (default: the one in the metadata).")
    scale: float | None = Field(gt=0.0, le=MAX_PREVIEW_SCALE, description="Output scale factor.")


class _LabelArgs(ArgsModel):
    """Base of the commands that act on one foreground label of a segmentation."""

    segmentation: Path
    label: Label

    @field_validator("label", mode="before")
    @classmethod
    def _parse_label(cls, value: Any) -> Label:
        """Accept a label key on the command line (pydantic ``before`` validator).

        Args:
            value: Raw ``label`` value.

        Returns:
            The foreground label.
        """
        return _foreground_label(value)


class MeshArgs(_LabelArgs):
    """``pybones mesh SEGMENTATION --label L -o OUTPUT.stl``."""

    command: Literal["mesh"]
    output: Path


class ExportArgs(_LabelArgs):
    """``pybones export SEGMENTATION --label L -o OUTPUT_DIR``."""

    command: Literal["export"]
    output: Path


class CompareArgs(_LabelArgs):
    """``pybones compare SEGMENTATION --label L --reference MASKS``."""

    command: Literal["compare"]
    reference: Path


CommandArgs = Annotated[
    ConfigArgs | SegmentArgs | PreviewArgs | MeshArgs | ExportArgs | CompareArgs,
    Field(discriminator="command"),
]


#: Options of ``pybones`` given before the sub-command.
_GLOBAL_OPTIONS = ("config", "overrides")


class Invocation(ArgsModel):
    """A complete, validated ``pybones`` invocation.

    Attributes:
        config: Optional user configuration file (merged over the defaults).
        overrides: ``section.key=value`` configuration overrides.
        args: The sub-command and its arguments.
    """

    config: Path | None
    overrides: tuple[str, ...]
    args: CommandArgs

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any]) -> "Invocation":
        """Validate the dict of an ``argparse.Namespace`` (global options + sub-command).

        Args:
            namespace: ``vars()`` of the parsed namespace; ``command`` selects the model.

        Returns:
            The validated invocation.

        Raises:
            pydantic.ValidationError: If an argument has the wrong type or is out of range.
        """
        return cls.model_validate(nest_command(namespace, _GLOBAL_OPTIONS))
