"""Mesh and quantification jobs with settings of their own (overrides), and their records."""

from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

from pybones.api.job_status import JobState
from pybones.server.routes.segmentations import CONFIG_LOCATION
from tests.helpers import SEGMENTATION
from tests.server.test_app import follow

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tests.helpers import Folders

BASE = f"/api/segmentations/{SEGMENTATION}"
MESH = f"{BASE}/meshes/vessel?format=glb"
MESH_SETTINGS = f"{BASE}/meshes/vessel/settings?format=glb"
QUANTIFICATION = f"{BASE}/quantification"


def run(client: "TestClient", url: str, overrides: list[str] | None) -> None:
    """Start a job (with a body if overrides are given) and wait for its success.

    Args:
        client: The API.
        url: The job's address.
        overrides: Its settings, or ``None`` for no body at all.
    """
    body = None if overrides is None else {"overrides": overrides}
    response = client.post(url, json=body)
    assert response.status_code == HTTPStatus.ACCEPTED, response.text
    assert follow(client, response.json()["job_id"])[-1]["state"] == JobState.SUCCEEDED


def test_a_mesh_records_the_settings_it_was_built_with(client: "TestClient") -> None:
    """Without a body: the server's settings; with overrides: those, recorded next to it."""
    assert client.get(MESH_SETTINGS).status_code == HTTPStatus.NOT_FOUND  # not built yet
    run(client, MESH, None)
    default = client.get(MESH_SETTINGS).json()["settings"]
    run(client, MESH, ["mesh.taubin.iterations=0", "mesh.downsample.vessel=1"])
    changed = client.get(MESH_SETTINGS).json()["settings"]
    assert (changed["taubin"]["iterations"], changed["downsample"]["vessel"]) == (0, 1)
    assert changed["glb_position_bits"] == default["glb_position_bits"]  # the rest: the server's
    assert default["taubin"]["iterations"] != 0


def test_a_quantification_records_the_settings_it_was_measured_with(
    client: "TestClient",
) -> None:
    """The overrides are applied and come back with the summary."""
    run(client, QUANTIFICATION, ["quantification.pruning.loop_factor=2.5"])
    view = client.get(QUANTIFICATION).json()
    assert view["settings"]["pruning"]["loop_factor"] == pytest.approx(2.5)
    assert view["summary"]["branches"] >= 0


@pytest.mark.parametrize(
    ("url", "override", "location"),
    [
        (MESH, "mesh.step_size=0", ["mesh", "step_size"]),
        (QUANTIFICATION, "quantification.max_radius_um=-1", ["quantification", "max_radius_um"]),
    ],
)
def test_an_invalid_job_setting_is_reported_at_its_path(
    client: "TestClient", url: str, override: str, location: list[str]
) -> None:
    """As for a segmentation: one entry per invalid setting, located by its path."""
    response = client.post(url, json={"overrides": [override]})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert [d["loc"] for d in response.json()["detail"]] == [[CONFIG_LOCATION, *location]]
    assert client.get("/api/jobs").json() == []


@pytest.mark.parametrize(
    ("url", "override"),
    [
        (MESH, "refinement.enabled=false"),  # a segmentation setting: not the mesh's to change
        (QUANTIFICATION, "mesh.step_size=2"),
        (MESH, "mesh.no_such_key=1"),
        (QUANTIFICATION, 'quantification.files.network_file="other.json"'),  # kept: the server's
    ],
)
def test_a_job_changes_only_its_own_section(client: "TestClient", url: str, override: str) -> None:
    """Another section, or an unknown key, is refused before any job starts."""
    response = client.post(url, json={"overrides": [override]})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert client.get("/api/jobs").json() == []


def test_settings_never_hide_an_older_segmentation(client: "TestClient") -> None:
    """A segmentation of another version still answers 409, with or without settings."""
    url = "/api/segmentations/broken/meshes/vessel?format=glb"
    for body in (None, {"overrides": ["mesh.taubin.iterations=0"]}):
        response = client.post(url, json=body)
        assert response.status_code == HTTPStatus.CONFLICT, response.text


def test_a_damaged_mesh_record_asks_to_build_again(
    client: "TestClient", folders: "Folders"
) -> None:
    """A settings record that is not this version's JSON answers 409, never a server error."""
    run(client, MESH, None)
    record = next((folders.output / SEGMENTATION).glob("*.glb.json"))
    for content in ("{not json", "[]", '{"pybones_version": "x", "settings": {"step_size": 0}}'):
        _ = record.write_text(content)
        response = client.get(MESH_SETTINGS)
        assert response.status_code == HTTPStatus.CONFLICT
        assert "build it again" in response.json()["detail"]
