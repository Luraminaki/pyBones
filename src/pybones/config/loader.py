"""Loading, merging and overriding of JSON configuration files."""

import json
from importlib import resources
from typing import TYPE_CHECKING, Any

from pybones.config.schema import PyBonesConfig, ServerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pydantic import BaseModel

DEFAULT_CONFIG_RESOURCE = "default.json"
SERVER_CONFIG_RESOURCE = "server.json"
_KEY_SEPARATOR = "."
_ASSIGNMENT = "="


def packaged_text(resource: str) -> str:
    """Read a configuration file shipped with the package.

    Args:
        resource: File name inside :mod:`pybones.config`.

    Returns:
        Its raw JSON text.
    """
    return resources.files(__package__).joinpath(resource).read_text()


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


def override_section(item: str) -> str:
    """Name the top-level section an override changes.

    Args:
        item: Override such as ``"pipeline.workers=1"``.

    Returns:
        Its first key, e.g. ``"pipeline"``.
    """
    dotted, _, _ = item.partition(_ASSIGNMENT)
    return dotted.split(_KEY_SEPARATOR, 1)[0]


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


def _load[M: BaseModel](
    model: type[M], resource: str, path: "Path | None", overrides: "Sequence[str]"
) -> M:
    """Load a packaged JSON file, merge a user file and overrides over it, validate.

    Args:
        model: Schema of the file.
        resource: Packaged default file.
        path: Optional user JSON file (only the values it changes).
        overrides: Dotted ``section.key=value`` strings.

    Returns:
        The validated, immutable configuration.
    """
    data: dict[str, Any] = json.loads(packaged_text(resource))
    if path is not None:
        _deep_merge(data, json.loads(path.read_text()))
    for item in overrides:
        _apply_override(data, item)
    return model.model_validate(data)


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
    return _load(PyBonesConfig, DEFAULT_CONFIG_RESOURCE, path, overrides)


def load_server_config(path: "Path | None" = None, overrides: "Sequence[str]" = ()) -> ServerConfig:
    """Build the validated web server configuration (packaged ``server.json`` + user changes).

    Args:
        path: Optional user JSON file, merged over the packaged one.
        overrides: Dotted ``section.key=value`` strings, e.g. ``"http.port=8080"``.

    Returns:
        The validated, immutable server configuration.

    Raises:
        pydantic.ValidationError: If a value has the wrong type or is out of range.
        ValueError: If an override is malformed or names an unknown key.
    """
    return _load(ServerConfig, SERVER_CONFIG_RESOURCE, path, overrides)


def apply_overrides(config: PyBonesConfig, overrides: "Sequence[str]") -> PyBonesConfig:
    """Derive a configuration from another one with ``section.key=value`` overrides.

    Args:
        config: Base configuration.
        overrides: Dotted ``section.key=value`` strings.

    Returns:
        The validated, immutable new configuration (``config`` itself is unchanged).

    Raises:
        pydantic.ValidationError: If a value has the wrong type or is out of range.
        ValueError: If an override is malformed or names an unknown key.
    """
    data: dict[str, Any] = config.model_dump(mode="json")
    for item in overrides:
        _apply_override(data, item)
    return PyBonesConfig.model_validate(data)
