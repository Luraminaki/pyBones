"""Tests of the web page served by the server (:mod:`pybones.server.static`).

The page itself is checked in a browser; these tests make sure it is served, that every
JavaScript import points to a file the server serves, and that the vendored three.js is
the one recorded in its manifest.
"""

import hashlib
import json
import re
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

from pybones.server.app import DOCS_PATH, PAGE_CACHE_CONTROL, STATIC_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

INDEX = STATIC_DIR / "index.html"
VENDOR = STATIC_DIR / "vendor" / "three"
MANIFEST = VENDOR / "manifest.json"
#: Import statements starting a line (`import ... from '...'`, `export ... from '...'`,
#: `import '...'`) and dynamic imports (`import('...')`); comments and strings are not imports.
IMPORT = re.compile(
    r"""^(?:import|export)\b[^;'"]*?\bfrom\s*['"]([^'"]+)['"]"""
    + r"""|^import\s*['"]([^'"]+)['"]|\bimport\(\s*['"]([^'"]+)['"]""",
    re.MULTILINE,
)
IMPORT_MAP = re.compile(r'<script type="importmap">(.*?)</script>', re.DOTALL)
#: Assets index.html links to.
ASSET = re.compile(r"""(?:src|href)="([^"]+\.(?:js|css|svg))\"""")
JS_MEDIA_TYPE = "text/javascript"


def import_map() -> dict[str, str]:
    """Read the import map of the page.

    Returns:
        Bare specifier (or prefix) to path, relative to the page.
    """
    match = IMPORT_MAP.search(INDEX.read_text())
    assert match is not None
    return json.loads(match.group(1))["imports"]


def resolve(specifier: str, importer: "Path", mapping: dict[str, str]) -> "Path":
    """Resolve an import the way the browser does (relative path or import map).

    Args:
        specifier: Imported module.
        importer: File doing the import.
        mapping: The page's import map.

    Returns:
        The imported file.
    """
    if specifier.startswith("."):
        return (importer.parent / specifier).resolve()
    if specifier in mapping:
        return (STATIC_DIR / mapping[specifier]).resolve()
    prefix = next(p for p in mapping if p.endswith("/") and specifier.startswith(p))
    return (STATIC_DIR / mapping[prefix] / specifier.removeprefix(prefix)).resolve()


def page_modules() -> list["Path"]:
    """List the page's own scripts and the vendored modules they reach.

    Returns:
        Every JavaScript file of the static folder.
    """
    return sorted(STATIC_DIR.rglob("*.js"))


def test_the_page_is_served_at_the_root(client: "TestClient") -> None:
    """The root serves the page; the API documentation stays at its own address."""
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert "<title>pyBones</title>" in response.text
    assert client.get(DOCS_PATH).status_code == HTTPStatus.OK
    assert client.get("/api/nope").status_code == HTTPStatus.NOT_FOUND


def test_every_linked_asset_is_served(client: "TestClient") -> None:
    """Scripts, stylesheet and icon linked from the page are served, scripts as JavaScript."""
    assets = ASSET.findall(INDEX.read_text())
    assert assets
    for asset in [*assets, *import_map().values()]:
        if asset.endswith("/"):
            continue
        response = client.get(f"/{asset.removeprefix('./')}")
        assert response.status_code == HTTPStatus.OK, asset
        if asset.endswith(".js"):
            assert response.headers["content-type"].startswith(JS_MEDIA_TYPE)


@pytest.mark.parametrize(
    "module", page_modules(), ids=lambda p: p.relative_to(STATIC_DIR).as_posix()
)
def test_every_import_resolves_to_a_served_file(module: "Path") -> None:
    """Each import of each module (relative or through the import map) names an existing file."""
    mapping = import_map()
    for groups in IMPORT.findall(module.read_text()):
        specifier = next(g for g in groups if g)
        target = resolve(specifier, module, mapping)
        assert target.is_file(), f"{module.name} imports {specifier!r}: {target} is missing"
        assert STATIC_DIR.resolve() in target.parents


def test_vendored_three_matches_its_manifest() -> None:
    """The vendored three.js files are exactly the recorded ones (an update is deliberate)."""
    manifest = json.loads(MANIFEST.read_text())
    present = {
        p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*") if p.is_file() and p != MANIFEST
    }
    assert present == set(manifest["files"])
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
    assert manifest["license"] == "MIT"


#: `import { a, b as c } from './x.js'` (named imports of the page's own modules).
NAMED_IMPORT = re.compile(r"^import\s*\{([^}]*)\}\s*from\s*['\"](\.[^'\"]+)['\"]", re.MULTILINE)
EXPORT = re.compile(r"^export\s+(?:async\s+)?(?:function|const|let|class)\s+(\w+)", re.MULTILINE)
OWN_SCRIPTS = sorted((STATIC_DIR / "js").glob("*.js"))


@pytest.mark.parametrize("module", OWN_SCRIPTS, ids=lambda p: p.name)
def test_named_imports_exist_in_their_module(module: "Path") -> None:
    """Every name a page module imports from another one is exported there (no typo)."""
    for names, specifier in NAMED_IMPORT.findall(module.read_text()):
        target = (module.parent / specifier).resolve()
        exported = set(EXPORT.findall(target.read_text()))
        for name in (n.split(" as ")[0].strip() for n in names.split(",") if n.strip()):
            assert name in exported, (
                f"{module.name} imports {name!r}, not exported by {target.name}"
            )


def test_the_page_files_are_checked_on_every_load(client: "TestClient") -> None:
    """A browser may keep a module but asks again each time: an update is never missed."""
    first = client.get("/js/api.js")
    assert first.status_code == HTTPStatus.OK
    assert first.headers["cache-control"] == PAGE_CACHE_CONTROL
    again = client.get("/js/api.js", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == HTTPStatus.NOT_MODIFIED
    assert client.get("/").headers["cache-control"] == PAGE_CACHE_CONTROL
