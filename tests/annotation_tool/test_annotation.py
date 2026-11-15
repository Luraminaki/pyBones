"""Tests of the annotation tool (:mod:`tools.annotation`)."""

import json
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from pybones.cli.main import main as pybones_main
from pybones.segmentation.labels import Label
from tests.helpers import cli_overrides, small_config
from tools.annotation.__main__ import main as annotation_main
from tools.annotation.scoring import score_segmentation
from tools.annotation.selection import select_crops
from tools.annotation.server import AnnotationApp, make_server
from tools.annotation.settings import AnnotationSettings, load_settings
from tools.annotation.workspace import CropCategory, CropSpec, Manifest, Workspace, now_iso

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SIZE = 32
CROP_ID = "c1"
LOCALHOST = "127.0.0.1"
WORKSPACE_NAME = "t"


def tool_settings(tmp_path: "Path", **crops: int) -> AnnotationSettings:
    """Adapt the default tool settings to small synthetic data.

    Args:
        tmp_path: Directory holding the workspaces.
        **crops: Crop settings to replace (e.g. ``n_burst=1``).

    Returns:
        The validated settings (any free port, 32 px crops).
    """
    data = json.loads(load_settings().model_dump_json())
    data["workspace_root"] = str(tmp_path / "annotations")
    data["server"]["port"] = 0
    data["crops"].update(
        {"size": SIZE, "context_slices": 1, "z_samples": 3, "min_tissue_fraction": 0.1}, **crops
    )
    return AnnotationSettings.model_validate(data)


def one_crop_manifest(settings: AnnotationSettings, z: int, context_slices: int) -> Manifest:
    """Build a manifest holding a single crop at the slice origin.

    Args:
        settings: Tool settings (unsure value).
        z: Slice of the crop.
        context_slices: Context slices on each side.

    Returns:
        The manifest.
    """
    return Manifest(
        source="s",
        segmentation="g",
        segmentation_version="0",
        crop_size=SIZE,
        context_slices=context_slices,
        prefilled=False,
        unsure_value=settings.unsure_value,
        created=now_iso(),
        crops=[CropSpec(crop_id=CROP_ID, z=z, y=0, x=0, category=CropCategory.RANDOM)],
    )


def test_default_settings_are_valid() -> None:
    """The shipped settings validate; "unsure" may not collide with a label."""
    settings = load_settings()
    assert settings.total_crops > 0
    data = json.loads(settings.model_dump_json())
    data["unsure_value"] = int(Label.VESSEL)
    with pytest.raises(ValidationError, match="collides"):
        _ = AnnotationSettings.model_validate(data)


def test_selection_prefers_rich_tiles_and_never_reuses_a_place(tmp_path: "Path") -> None:
    """Burst then vessel tiles come first; no place is used twice; selection is seeded."""
    vol = np.full((7, 4 * SIZE, 4 * SIZE), Label.BONE, np.uint8)
    vol[:, 0:SIZE, 0:SIZE] = Label.BURST  # top-left tile: burst
    vol[:, SIZE : 2 * SIZE, SIZE : 2 * SIZE] = Label.VESSEL  # next diagonal tile: vessel
    settings = tool_settings(tmp_path, n_burst=1, n_vessel=1, n_random=3)
    crops = select_crops(vol, settings.crops)
    assert [c.category for c in crops[:2]] == [CropCategory.BURST, CropCategory.VESSEL]
    assert (crops[0].y, crops[0].x) == (0, 0)
    assert (crops[1].y, crops[1].x) == (SIZE, SIZE)
    assert len({(c.y, c.x) for c in crops}) == len(crops) == settings.total_crops
    assert all(1 <= c.z <= 5 for c in crops)  # room for the context slice on each side
    assert select_crops(vol, settings.crops) == crops


def test_workspace_labels_roundtrip_and_validation(tmp_path: "Path") -> None:
    """Unpainted crops read as "unsure"; saved labels read back; invalid values are refused."""
    settings = tool_settings(tmp_path)
    ws = Workspace(tmp_path / "ws", settings)
    unpainted = ws.read_labels(CROP_ID, SIZE)
    assert (unpainted == settings.unsure_value).all()
    labels = np.full((SIZE, SIZE), Label.BONE, np.uint8)
    labels[0, 0] = settings.unsure_value
    ws.write_labels(CROP_ID, labels)
    np.testing.assert_array_equal(ws.read_labels(CROP_ID, SIZE), labels)
    with pytest.raises(ValueError, match="Unexpected"):
        ws.write_labels(CROP_ID, np.full((SIZE, SIZE), Label.AMBIGUOUS, np.uint8))


@pytest.fixture
def served(tmp_path: "Path") -> "Iterator[tuple[str, Workspace]]":
    """Serve a one-crop workspace on a free local port for the duration of a test.

    Args:
        tmp_path: Temporary directory of the test.

    Yields:
        The base URL of the server and its workspace.
    """
    settings = tool_settings(tmp_path)
    ws = Workspace(tmp_path / "ws", settings)
    ws.save_manifest(one_crop_manifest(settings, z=1, context_slices=1))
    server = make_server(AnnotationApp(ws, settings, small_config()), LOCALHOST, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://{LOCALHOST}:{server.server_address[1]}", ws
    server.shutdown()
    server.server_close()


def test_server_session_and_label_roundtrip(served: tuple[str, Workspace]) -> None:
    """The session lists the crop and labels; labels saved over HTTP read back."""
    base, ws = served
    with urllib.request.urlopen(f"{base}/api/session") as response:
        session = json.load(response)
    assert [c["crop_id"] for c in session["crops"]] == [CROP_ID]
    assert [lab["key"] for lab in session["labels"]][:2] == [Label.BACKGROUND.key, Label.BONE.key]
    labels = np.full((SIZE, SIZE), Label.VESSEL, np.uint8).tobytes()
    url = f"{base}/api/crops/{CROP_ID}/labels"
    request = urllib.request.Request(url, data=labels, method="PUT")
    with urllib.request.urlopen(request) as response:
        assert response.status == HTTPStatus.NO_CONTENT
    with urllib.request.urlopen(url) as response:
        assert response.read() == labels
    assert ws.is_annotated(CROP_ID)


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/crops/other/labels", None),  # not in the manifest
        ("GET", "/static/..%2Fserver.py", None),  # outside the static directory
        ("PUT", f"/api/crops/{CROP_ID}/labels", b"\x00" * 10),  # wrong size
        ("PUT", f"/api/crops/{CROP_ID}/labels", bytes([Label.AMBIGUOUS]) * SIZE * SIZE),
    ],
)
def test_server_rejects_bad_requests(
    served: tuple[str, Workspace], method: str, path: str, body: bytes | None
) -> None:
    """Unknown crops, paths outside the static directory and invalid labels are refused."""
    base, ws = served
    request = urllib.request.Request(base + path, data=body, method=method)
    with pytest.raises(urllib.error.HTTPError):
        _ = urllib.request.urlopen(request)
    assert not ws.is_annotated(CROP_ID)


def test_scoring_ignores_unsure_pixels(tmp_path: "Path") -> None:
    """Pixels painted "unsure" are left out of every score."""
    settings = tool_settings(tmp_path)
    ws = Workspace(tmp_path / "ws", settings)
    manifest = one_crop_manifest(settings, z=0, context_slices=0)
    pred = np.full((1, SIZE, SIZE), Label.BONE, np.uint8)
    pred[0, :, : SIZE // 2] = Label.VESSEL
    ref = np.full((SIZE, SIZE), Label.BONE, np.uint8)
    ref[:, : SIZE // 4] = Label.VESSEL  # half of the predicted vessel is right...
    ref[:, SIZE // 4 : SIZE // 2] = settings.unsure_value  # ...the other half is unsure
    ws.write_labels(CROP_ID, ref)
    score = score_segmentation(pred, manifest, ws)
    assert score.labels.voxels == SIZE * SIZE * 3 // 4
    assert score.labels.accuracy == 1.0
    assert score.labels.per_label[int(Label.VESSEL)].dice == 1.0


def test_prepare_and_score_cli(
    slice_dir: "Path", tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """``prepare`` writes a portable manifest, crop images and pre-fills; ``score`` runs."""
    seg = tmp_path / "seg"
    pybones_main([*cli_overrides(), "segment", str(slice_dir), "-o", str(seg)])
    settings = tool_settings(tmp_path, n_burst=1, n_vessel=1, n_random=1)
    settings_path = tmp_path / "settings.json"
    _ = settings_path.write_text(settings.model_dump_json())
    monkeypatch.chdir(tmp_path)
    common = ["--settings", str(settings_path)]
    annotation_main([*common, "prepare", str(seg), "--name", WORKSPACE_NAME])
    ws = Workspace(tmp_path / "annotations" / WORKSPACE_NAME, settings)
    manifest = ws.load_manifest()
    assert manifest.source == slice_dir.name  # stored relative to the working directory
    crop = manifest.crops[0]
    assert ws.crop_image_path(crop.crop_id, -1).exists()
    assert ws.prefill_path(crop.crop_id).exists()
    ws.write_labels(crop.crop_id, ws.read_labels(crop.crop_id, SIZE))  # accept the pre-fill
    annotation_main([*common, "score", str(seg), "--name", WORKSPACE_NAME])


def test_notes_roundtrip_and_validation(tmp_path: "Path") -> None:
    """Notes are saved as text, an empty note deletes the file, an overlong note is refused."""
    settings = tool_settings(tmp_path)
    ws = Workspace(tmp_path / "ws", settings)
    assert ws.read_note(CROP_ID) == ""
    ws.write_note(CROP_ID, "vessel or trabecula? é")
    assert ws.read_note(CROP_ID) == "vessel or trabecula? é"
    ws.write_note(CROP_ID, "   ")
    assert not ws.note_path(CROP_ID).exists()
    with pytest.raises(ValueError, match="longer"):
        ws.write_note(CROP_ID, "x" * (settings.max_note_chars + 1))


def test_server_note_roundtrip(served: tuple[str, Workspace]) -> None:
    """A note saved over HTTP reads back and is flagged in the session; it is not an annotation."""
    base, ws = served
    url = f"{base}/api/crops/{CROP_ID}/note"
    request = urllib.request.Request(url, data="à revoir".encode(), method="PUT")
    with urllib.request.urlopen(request) as response:
        assert response.status == HTTPStatus.NO_CONTENT
    with urllib.request.urlopen(url) as response:
        assert response.read().decode() == "à revoir"
    with urllib.request.urlopen(f"{base}/api/session") as response:
        assert json.load(response)["crops"][0]["note"] is True
    assert not ws.is_annotated(CROP_ID)
    invalid = urllib.request.Request(url, data=b"\xff\xfe", method="PUT")
    with pytest.raises(urllib.error.HTTPError):
        _ = urllib.request.urlopen(invalid)
