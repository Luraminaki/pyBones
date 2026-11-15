"""Loading, merging and overriding of JSON configuration files."""

import json
from importlib import resources
from typing import TYPE_CHECKING, Any

from pybones.config.schema import PyBonesConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DEFAULT_CONFIG_RESOURCE = "default.json"
_KEY_SEPARATOR = "."
_ASSIGNMENT = "="


def default_config_text() -> str:
    """Read the configuration shipped with the package.

    Returns:
        Raw JSON text of ``default.json``.
    """
    return resources.files(__package__).joinpath(DEFAULT_CONFIG_RESOURCE).read_text()


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Recursively merge ``update`` into ``base`` in place (dicts merged, others replaced).

    Args:
        base: Mapping modified in place.
        update: Values to merge over ``base``.
    """
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_override(data: dict[str, Any], item: str) -> None:
    """Apply one ``section.key=value`` override (value parsed as JSON, else kept as text).

    Args:
        data: Raw configuration mapping, modified in place.
        item: Override such as ``"pipeline.workers=1"``.

    Raises:
        ValueError: If ``item`` is malformed or names an unknown section or key.
    """
    dotted, sep, raw = item.partition(_ASSIGNMENT)
    if not sep or not dotted:
        raise ValueError(f"Override {item!r} must look like section.key=value")
    *parents, leaf = dotted.split(_KEY_SEPARATOR)
    node = data
    for part in parents:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Override {item!r}: {part!r} is not a configuration section")
        node = child
    if leaf not in node:
        raise ValueError(f"Override {item!r}: unknown key {leaf!r}")
    try:
        node[leaf] = json.loads(raw)
    except json.JSONDecodeError:
        node[leaf] = raw


def load_config(path: "Path | None" = None, overrides: "Sequence[str]" = ()) -> PyBonesConfig:
    """Build the validated configuration.

    The packaged ``default.json`` is loaded first; the optional user file is then
    deep-merged over it (so it only needs the values it changes), and finally the
    ``section.key=value`` overrides are applied.

    Args:
        path: Optional user JSON file.
        overrides: Dotted ``section.key=value`` strings, e.g.
            ``"segmentation.burst.density=0.3"``.

    Returns:
        The validated, immutable configuration.

    Raises:
        pydantic.ValidationError: If a value has the wrong type or is out of range.
        ValueError: If an override is malformed or names an unknown key.
    """
    data: dict[str, Any] = json.loads(default_config_text())
    if path is not None:
        _deep_merge(data, json.loads(path.read_text()))
    for item in overrides:
        _apply_override(data, item)
    return PyBonesConfig.model_validate(data)
