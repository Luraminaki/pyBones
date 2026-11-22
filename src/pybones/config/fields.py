"""Field types shared by the configuration schemas: physical ranges, file names, sections."""

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from pybones.utils.constants import UINT8_MAX

if TYPE_CHECKING:
    from pydantic.config import JsonDict

#: Physical quantities: generous but finite ranges (1 m, 1 m², 1 m³ as upper bounds).
Micrometres = Annotated[float, Field(gt=0.0, le=1e6)]
Length = Annotated[float, Field(ge=0.0, le=1e6)]
Area = Annotated[float, Field(ge=0.0, le=1e12)]
Volume = Annotated[float, Field(ge=0.0, le=1e18)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Channel = Annotated[int, Field(ge=0, le=UINT8_MAX)]
ColorBGR = tuple[Channel, Channel, Channel]
#: Characters that would turn a file name into a path.
PATH_SEPARATORS = ("/", "\\")


def _plain_name(name: str) -> str:
    """Check that a configured file name stays inside its folder.

    A validator rather than a ``pattern``: a field adding its own ``pattern`` (e.g. an
    extension) would replace this one instead of adding to it.

    Args:
        name: Configured file name.

    Returns:
        ``name``, unchanged.

    Raises:
        ValueError: If it is empty or holds a path separator.
    """
    if not name or any(sep in name for sep in PATH_SEPARATORS):
        raise ValueError(f"must be a plain file name (no {' or '.join(PATH_SEPARATORS)}): {name!r}")
    return name


FileName = Annotated[str, AfterValidator(_plain_name)]


class Effect(StrEnum):
    """What a change of a setting affects: what must be computed again after it."""

    LABELS = "labels"
    MESHES = "meshes"
    NETWORK = "network"
    PREVIEWS = "previews"
    NONE = "none"


#: What each effect means, for the forms and the reference (written once, here).
EFFECT_TEXT = {
    Effect.LABELS: "the segmentation's labels, and so its meshes and its network",
    Effect.MESHES: "the meshes only",
    Effect.NETWORK: "the measured network only",
    Effect.PREVIEWS: "the preview images only",
    Effect.NONE: "no result (speed, memory, progress or file names)",
}


class Level(StrEnum):
    """How prominently a form shows a section."""

    COMMON = "common"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


#: Keys the schemas add to the JSON schema (``json_schema_extra``) for the forms.
EFFECT_KEY, ADVICE_KEY, LEVEL_KEY, EFFECTS_KEY = "effect", "advice", "level", "effects"


def section_config(effect: Effect, level: Level = Level.COMMON) -> ConfigDict:
    """Build a section's pydantic configuration: strict, immutable, and what it affects.

    Args:
        effect: What a change of its settings affects (a setting may say otherwise).
        level: How prominently a form shows it.

    Returns:
        The ``model_config`` of the section.
    """
    return ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={EFFECT_KEY: effect.value, LEVEL_KEY: level.value},
    )


def advice(text: str, effect: Effect | None = None) -> "JsonDict":
    """Describe when to change a setting, and what it affects if not its section's effect.

    Args:
        text: When to change it (the description says what it is).
        effect: What a change affects, when it differs from its section's.

    Returns:
        The field's ``json_schema_extra``.
    """
    extra: JsonDict = {ADVICE_KEY: text}
    if effect is not None:
        extra[EFFECT_KEY] = effect.value
    return extra


def affects(effect: Effect) -> "JsonDict":
    """Say what a change of a setting affects, when it differs from its section's.

    Args:
        effect: What a change affects.

    Returns:
        The field's ``json_schema_extra``.
    """
    return {EFFECT_KEY: effect.value}


class Section(BaseModel):
    """Base of every configuration section: strict and immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
