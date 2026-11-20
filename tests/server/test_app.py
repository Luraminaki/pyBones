"""End-to-end tests of the HTTP API (:mod:`pybones.server`) through a test client."""

import json
import struct
import threading
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import pytest

from pybones import __version__
from pybones.api.jobs import JobState
from pybones.meshing.formats import MeshFormat
from pybones.meshing.gltf import read_glb
from pybones.segmentation.stages import STAGES_2D, Stage
from pybones.server.context import ServerContext
from pybones.server.routes.jobs import SSE_MEDIA_TYPE
from pybones.server.routes.segmentations import MESH_MEDIA_TYPES, PNG_MEDIA_TYPE
from tests.helpers import N_SYNTHETIC_SLICES, SEGMENTATION, SLICE_SHAPE, STACK_DIR, STACK_ZIP

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from numpy.typing import NDArray

    from pybones.segmentation.stages import ProgressTracker
    from tests.helpers import Folders

SSE_DATA_PREFIX = "data: "
WAIT_S = 30.0
#: Binary STL: 80-byte header, uint32 triangle count, then 50 bytes per triangle.
STL_HEADER_BYTES, STL_COUNT_FORMAT, STL_TRIANGLE_BYTES = 80, "<I", 50
WEIGHTS = dict.fromkeys(Stage, 1.0)


def follow(client: "TestClient", job_id: str) -> list[dict[str, Any]]:
    """Read a job's event stream to its end.

    Args:
        client: Test client.
        job_id: Job id.

    Returns:
        The decoded events, in order.
    """
    events: list[dict[str, Any]] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == HTTPStatus.OK
        assert response.headers["content-type"].startswith(SSE_MEDIA_TYPE)
        events.extend(
            json.loads(line.removeprefix(SSE_DATA_PREFIX))
            for line in response.iter_lines()
            if line.startswith(SSE_DATA_PREFIX)
        )
    return events


def context_of(client: "TestClient") -> ServerContext:
    """Reach the server state behind a test client.

    Args:
        client: Test client.

    Returns:
        The application's context.
    """
    context = cast("FastAPI", client.app).state.context
    assert isinstance(context, ServerContext)
    return context


def occupy_worker(client: "TestClient") -> threading.Event:
    """Keep the server's only job worker busy, so the next jobs stay queued.

    Args:
        client: Test client.

    Returns:
        The event that releases the worker.
    """
    release, started = threading.Event(), threading.Event()

    def work(tracker: "ProgressTracker") -> None:
        """Run until released.

        Args:
            tracker: Progress tracker of the job.
        """
        tracker(Stage.THRESHOLDS, 0, 1)
        started.set()
        _ = release.wait(WAIT_S)

    _ = context_of(client).jobs.submit("blocker", work, STAGES_2D, WEIGHTS)
    assert started.wait(WAIT_S)
    return release


def test_version_and_configuration(client: "TestClient") -> None:
    """The API reports its version, its base configuration and the configuration schema."""
    assert client.get("/api/version").json() == {"version": __version__}
    config = client.get("/api/config").json()
    assert config["pipeline"]["workers"] == context_of(client).config.pipeline.workers
    schema = client.get("/api/config/schema").json()
    assert "segmentation" in schema["properties"]
    server = client.get("/api/server").json()
    assert server["version"] == __version__
    assert server["output_dir"] == str(context_of(client).library.output_dir)
    assert server["default_mesh_format"] == "glb"
    assert "refinement" in server["overridable_sections"]
    assert "io" not in server["overridable_sections"]


def test_datasets_and_segmentations_are_listed(client: "TestClient") -> None:
    """Only real stacks and finished, readable segmentations are listed."""
    assert [d["id"] for d in client.get("/api/datasets").json()] == [STACK_DIR, STACK_ZIP]
    assert [s["id"] for s in client.get("/api/segmentations").json()] == [SEGMENTATION]
    meta = client.get(f"/api/segmentations/{SEGMENTATION}").json()
    assert meta["shape_zyx"] == [N_SYNTHETIC_SLICES, *SLICE_SHAPE]
    assert client.get("/api/segmentations/unfinished").status_code == HTTPStatus.NOT_FOUND
    broken = client.get("/api/segmentations/broken")
    assert broken.status_code == HTTPStatus.CONFLICT
    assert "segment the stack again" in broken.json()["detail"]


def test_segmentation_job_runs_end_to_end(client: "TestClient", folders: "Folders") -> None:
    """A segmentation request runs as a job whose events end in success, then it is listed."""
    request = {"dataset": STACK_ZIP, "name": "new", "overrides": ["refinement.enabled=false"]}
    response = client.post("/api/segmentations", json=request)
    assert response.status_code == HTTPStatus.ACCEPTED
    job = response.json()
    assert (job["segmentation"], job["label"], job["kind"]) == ("new", None, "segmentation")
    events = follow(client, job["job_id"])
    assert events[-1]["state"] == JobState.SUCCEEDED
    fractions = [e["progress"]["fraction"] for e in events if e["progress"] is not None]
    assert fractions == sorted(fractions)
    listed = [s["id"] for s in client.get("/api/segmentations").json()]
    assert listed == ["new", SEGMENTATION]
    meta = client.get("/api/segmentations/new").json()
    assert meta["config"]["refinement"]["enabled"] is False
    assert job["job_id"] in [j["job_id"] for j in client.get("/api/jobs").json()]
    assert (folders.output / "new").is_dir()


@pytest.mark.parametrize(
    ("request_body", "status"),
    [
        ({"dataset": STACK_DIR, "name": SEGMENTATION}, HTTPStatus.CONFLICT),
        ({"dataset": STACK_DIR, "name": "unfinished"}, HTTPStatus.CONFLICT),
        ({"dataset": "missing", "name": "x"}, HTTPStatus.NOT_FOUND),
        ({"dataset": STACK_DIR, "name": "../x"}, HTTPStatus.UNPROCESSABLE_ENTITY),
        ({"dataset": STACK_DIR, "name": "x", "extra": 1}, HTTPStatus.UNPROCESSABLE_ENTITY),
        (
            {"dataset": STACK_DIR, "name": "x", "overrides": ['io.labels_file="l.npy"']},
            HTTPStatus.UNPROCESSABLE_ENTITY,
        ),
        (
            {"dataset": STACK_DIR, "name": "x", "overrides": ["refinement.threads=2"]},
            HTTPStatus.UNPROCESSABLE_ENTITY,
        ),
        (
            {"dataset": STACK_DIR, "name": "x", "overrides": ["pipeline.workers=0"]},
            HTTPStatus.UNPROCESSABLE_ENTITY,
        ),
    ],
)
def test_segmentation_requests_are_checked(
    client: "TestClient", folders: "Folders", request_body: dict[str, Any], status: HTTPStatus
) -> None:
    """Taken names, unknown datasets, invalid names and refused overrides start no job."""
    response = client.post("/api/segmentations", json=request_body)
    assert response.status_code == status
    assert "detail" in response.json()
    assert client.get("/api/jobs").json() == []
    assert not (folders.output / "x").exists()


def test_queued_segmentation_holds_its_name_until_cancelled(
    client: "TestClient", folders: "Folders"
) -> None:
    """A queued job takes its name; cancelled before it starts, it frees it and leaves nothing."""
    release = occupy_worker(client)
    request = {"dataset": STACK_DIR, "name": "queued"}
    job = client.post("/api/segmentations", json=request).json()
    assert job["state"] == JobState.QUEUED
    assert client.post("/api/segmentations", json=request).status_code == HTTPStatus.CONFLICT
    assert client.delete(f"/api/jobs/{job['job_id']}").status_code == HTTPStatus.ACCEPTED
    release.set()
    assert follow(client, job["job_id"])[-1]["state"] == JobState.CANCELLED
    assert not (folders.output / "queued").exists()
    retry = client.post("/api/segmentations", json=request).json()
    assert follow(client, retry["job_id"])[-1]["state"] == JobState.SUCCEEDED


def test_slices_are_rendered_as_png(client: "TestClient") -> None:
    """A slice comes back as a scaled PNG; bad slices, scales and ids are refused."""
    response = client.get(f"/api/segmentations/{SEGMENTATION}/slices/2", params={"scale": 0.5})
    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"] == PNG_MEDIA_TYPE
    pixels = np.frombuffer(response.content, dtype=np.uint8)
    image = cast("NDArray[np.uint8]", cv2.imdecode(pixels, cv2.IMREAD_UNCHANGED))
    assert image.shape[:2] == tuple(s // 2 for s in SLICE_SHAPE)
    base = f"/api/segmentations/{SEGMENTATION}/slices"
    assert client.get(f"{base}/{N_SYNTHETIC_SLICES}").status_code == HTTPStatus.NOT_FOUND
    assert client.get(f"{base}/-1").status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    for scale in (0, 100):
        response = client.get(f"{base}/0", params={"scale": scale})
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert client.get("/api/segmentations/nope/slices/0").status_code == HTTPStatus.NOT_FOUND


def build_mesh(client: "TestClient", url: str, params: dict[str, str]) -> dict[str, Any]:
    """Build a mesh through the API and wait for its job.

    Args:
        client: Test client.
        url: Mesh address.
        params: Query parameters (format).

    Returns:
        The job as first returned.
    """
    job = client.post(url, params=params).json()
    assert follow(client, job["job_id"])[-1]["state"] == JobState.SUCCEEDED
    return job


def test_meshes_are_built_by_jobs_then_downloaded(client: "TestClient", tmp_path: "Path") -> None:
    """A mesh is missing until its job succeeds; GLB by default, STL on request."""
    url = f"/api/segmentations/{SEGMENTATION}/meshes/bone"
    assert client.get(url).status_code == HTTPStatus.NOT_FOUND
    job = build_mesh(client, url, {})
    assert (job["kind"], job["label"], job["mesh_format"]) == ("mesh", "bone", "glb")
    response = client.get(url)
    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"] == MESH_MEDIA_TYPES[MeshFormat.GLB]
    glb = tmp_path / "bone.glb"
    _ = glb.write_bytes(response.content)
    assert len(read_glb(glb).faces) > 0
    stl = {"format": "stl"}
    assert client.get(url, params=stl).status_code == HTTPStatus.NOT_FOUND
    _ = build_mesh(client, url, stl)
    response = client.get(url, params=stl)
    assert response.headers["content-type"] == MESH_MEDIA_TYPES[MeshFormat.STL]
    body = response.content
    (count,) = struct.unpack_from(STL_COUNT_FORMAT, body, STL_HEADER_BYTES)
    assert len(body) == STL_HEADER_BYTES + 4 + count * STL_TRIANGLE_BYTES
    assert count == len(read_glb(glb).faces)
    summary = client.get("/api/segmentations").json()[0]
    assert summary["meshes"] == {"bone": ["stl", "glb"]}
    rebuilt = client.post(url).json()
    assert rebuilt["job_id"] != job["job_id"]
    assert follow(client, rebuilt["job_id"])[-1]["state"] == JobState.SUCCEEDED
    for label in ("ambiguous", "background", "nope"):
        response = client.post(f"/api/segmentations/{SEGMENTATION}/meshes/{label}")
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert client.post(url, params={"format": "obj"}).status_code == (
        HTTPStatus.UNPROCESSABLE_ENTITY
    )


def test_a_mesh_is_built_by_one_job_at_a_time(client: "TestClient") -> None:
    """Asking again for a mesh being built returns the job already building it."""
    release = occupy_worker(client)
    url = f"/api/segmentations/{SEGMENTATION}/meshes/vessel"
    first, second = client.post(url).json(), client.post(url).json()
    assert first["job_id"] == second["job_id"]
    release.set()
    assert follow(client, first["job_id"])[-1]["state"] == JobState.SUCCEEDED


def test_unknown_jobs_are_not_found(client: "TestClient") -> None:
    """Every job route answers 404 to an unknown id."""
    assert client.get("/api/jobs/nope").status_code == HTTPStatus.NOT_FOUND
    assert client.get("/api/jobs/nope/events").status_code == HTTPStatus.NOT_FOUND
    assert client.delete("/api/jobs/nope").status_code == HTTPStatus.NOT_FOUND


def test_slices_are_never_kept_by_the_browser(client: "TestClient") -> None:
    """Slices are rendered on demand: the browser is told to keep none of them."""
    response = client.get(f"/api/segmentations/{SEGMENTATION}/slices/0", params={"scale": 0.5})
    assert response.headers["cache-control"] == "no-store"


def test_meshes_are_revalidated_and_a_rebuild_is_seen(client: "TestClient") -> None:
    """A current mesh is answered 304 without a body; a rebuilt one has a new ETag."""
    url = f"/api/segmentations/{SEGMENTATION}/meshes/burst"
    _ = build_mesh(client, url, {})
    first = client.get(url)
    assert first.headers["cache-control"] == "no-cache"
    etag = first.headers["etag"]
    unchanged = client.get(url, headers={"If-None-Match": etag})
    assert unchanged.status_code == HTTPStatus.NOT_MODIFIED
    assert unchanged.content == b""
    _ = build_mesh(client, url, {})  # rebuilt: same address, new file
    rebuilt = client.get(url, headers={"If-None-Match": etag})
    assert rebuilt.status_code == HTTPStatus.OK
    assert rebuilt.headers["etag"] != etag
    assert rebuilt.content
