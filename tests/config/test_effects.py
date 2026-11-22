"""Every setting says what a change affects, and the form's settings say when to change them."""

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from pybones.config.fields import ADVICE_KEY, EFFECT_KEY, EFFECTS_KEY, LEVEL_KEY, Effect, Level
from pybones.config.schema import PyBonesConfig
from pybones.config.server import ServerConfig
from pybones.server.schemas import OVERRIDABLE_SECTIONS

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from pydantic.fields import FieldInfo


def own_extra(model: type[BaseModel]) -> "Mapping[str, object]":
    """Read what a section declares about itself.

    Args:
        model: The section.

    Returns:
        Its ``json_schema_extra`` (empty if none).
    """
    extra = model.model_config.get("json_schema_extra")
    return extra if isinstance(extra, dict) else {}


def settings(model: type[BaseModel], path: str = "") -> "Iterator[tuple[str, FieldInfo, str]]":
    """Walk a configuration's settings with the effect each one resolves to.

    Args:
        model: Section to walk.
        path: Its dotted path (empty for the root).

    Yields:
        ``(path, field, effect)``: the setting's own effect, else its section's.
    """
    section_effect = str(own_extra(model).get(EFFECT_KEY, ""))
    for name, field in model.model_fields.items():
        kind = field.annotation
        if isinstance(kind, type) and issubclass(kind, BaseModel):
            yield from settings(kind, f"{path}{name}.")
            continue
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        yield f"{path}{name}", field, str(extra.get(EFFECT_KEY, section_effect))


def sections(model: type[BaseModel]) -> "Iterator[type[BaseModel]]":
    """Walk a configuration's sections below its root.

    Args:
        model: Section to walk.

    Yields:
        Every nested section.
    """
    for field in model.model_fields.values():
        kind = field.annotation
        if isinstance(kind, type) and issubclass(kind, BaseModel):
            yield kind
            yield from sections(kind)


@pytest.mark.parametrize("root", [PyBonesConfig, ServerConfig])
def test_every_section_says_what_it_affects_and_how_prominent_it_is(
    root: type[BaseModel],
) -> None:
    """A new section cannot be forgotten: it must declare an effect and a level."""
    for section in sections(root):
        extra = own_extra(section)
        assert extra.get(EFFECT_KEY) in set(Effect), section.__name__
        assert extra.get(LEVEL_KEY) in set(Level), section.__name__


def test_every_setting_resolves_to_a_known_effect() -> None:
    """A setting's own effect, if it has one, is a known one too."""
    for path, _, effect in settings(PyBonesConfig):
        assert effect in set(Effect), path


def test_every_setting_of_the_form_says_when_to_change_it() -> None:
    """The settings a run may change (the web page's form) all carry advice."""
    missing = [
        path
        for path, field, _ in settings(PyBonesConfig)
        if path.split(".")[0] in OVERRIDABLE_SECTIONS
        and not (
            isinstance(field.json_schema_extra, dict) and field.json_schema_extra.get(ADVICE_KEY)
        )
    ]
    assert missing == []


def test_the_schema_explains_every_effect() -> None:
    """The root of the JSON schema carries one text per effect, for the form and the reference."""
    effects = PyBonesConfig.model_json_schema()[EFFECTS_KEY]
    assert set(effects) == {e.value for e in Effect}
    assert all(effects.values())


def test_speed_settings_change_no_result() -> None:
    """Threads, deconvolution blocks and progress weights only change speed, memory or progress."""
    effects = {path: effect for path, _, effect in settings(PyBonesConfig)}
    for path in (
        "pipeline.workers",
        "pipeline.stage_weights",
        "thin_vessels.block_slices",
        "mesh.stage_weights",
        "quantification.stage_weights",
        "quantification.backends.skeleton",
    ):
        assert effects[path] == Effect.NONE, path
    assert effects["thin_vessels.enabled"] == Effect.LABELS
    assert effects["mesh.taubin.iterations"] == Effect.MESHES
    assert effects["quantification.pruning.loop_factor"] == Effect.NETWORK
