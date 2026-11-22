"""Tests of the configuration reference (:mod:`tools.docs.configuration`)."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pybones.config.schema import PyBonesConfig
from pybones.config.server import ServerConfig
from tools.docs.__main__ import CONFIGURATION_FILE
from tools.docs.configuration import accepts, iter_sections, render_configuration, section_of

if TYPE_CHECKING:
    from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
#: A shared definition, as pydantic writes it under ``$defs``.
DEFS = {"Level": {"enum": ["INFO", "DEBUG"], "type": "string"}}
#: A JSON-schema number in ``(0, 1]``.
FRACTION = {"exclusiveMinimum": 0.0, "maximum": 1.0, "type": "number"}


def test_the_reference_matches_the_code() -> None:
    """docs/CONFIGURATION.md is exactly what the schemas and default values give."""
    written = (ROOT / CONFIGURATION_FILE).read_text()
    assert written == render_configuration(), (
        f"{CONFIGURATION_FILE} is out of date: run `uv run python -m tools.docs`"
    )


@pytest.mark.parametrize("root", [PyBonesConfig, ServerConfig])
def test_every_setting_is_described_under_its_section(root: "type[BaseModel]") -> None:
    """Each setting has a description and a row in its own section's table."""
    reference = render_configuration()
    for path, section in iter_sections(root):
        own = reference.split(f"### `{path}`\n", 1)[1].split("\n### ", 1)[0]
        for name, field in section.model_fields.items():
            if section_of(field) is None:
                assert field.description, f"{path}.{name} has no description"
                assert f"| `{name}` |" in own, f"{path}.{name} is missing from its table"


@pytest.mark.parametrize(
    ("schema", "default", "expected"),
    [
        (FRACTION, None, "0 < x ≤ 1"),
        ({"minimum": 1, "maximum": 256, "type": "integer"}, None, "integer, 1 ≤ x ≤ 256"),
        ({"minimum": 0.0, "maximum": 1e18, "type": "number"}, None, "0 ≤ x ≤ 1e18"),
        ({"type": "number"}, None, "any"),
        ({"enum": [4, 8], "type": "integer"}, None, "`4` / `8`"),
        ({"type": "boolean"}, None, "`true` / `false`"),
        ({"$ref": "#/$defs/Level"}, None, '`"INFO"` / `"DEBUG"`'),
        ({"prefixItems": [FRACTION, FRACTION], "type": "array"}, None, "2 values, each 0 < x ≤ 1"),
        (
            {"items": {"minLength": 1, "type": "string"}, "minItems": 1, "type": "array"},
            None,
            "list of non-empty text (at least 1)",
        ),
        (
            {"additionalProperties": FRACTION, "type": "object"},
            {"bone": 0.5, "vessel": 0.5},
            "0 < x ≤ 1, for each of `bone`, `vessel`",
        ),
        ({"pattern": r"\.npy$", "type": "string"}, None, r"text matching `\.npy$`"),
    ],
)
def test_accepted_values_read_from_the_schema(
    schema: dict[str, object], default: object, expected: str
) -> None:
    """Bounds, choices, references, tuples, lists, maps and patterns are written out."""
    assert accepts(schema, DEFS, default) == expected
