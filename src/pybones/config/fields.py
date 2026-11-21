"""Field types shared by the configuration schemas: physical ranges, file names, sections."""

from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from pybones.utils.constants import UINT8_MAX

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


class Section(BaseModel):
    """Base of every configuration section: strict and immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
