"""Fixtures of the web server tests: a data folder, an output folder and a test client."""

import json
import shutil
import zipfile
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from pybones.config.loader import load_server_config
from pybones.segmentation.pipeline import run_segmentation
from pybones.server.app import create_app
from tests.helpers import SEGMENTATION, STACK_DIR, STACK_ZIP, Folders, small_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pybones.config.server import ServerConfig


@pytest.fixture
def folders(slice_dir: "Path", tmp_path: "Path") -> Folders:
    """Fill a data folder and an output folder.

    Data: ``stack`` (directory), ``stack.zip``, a text file and a hidden folder (not
    stacks). Output: ``seg`` (finished), ``unfinished`` (no metadata) and ``broken``
    (invalid metadata).

    Args:
        slice_dir: Synthetic slices.
        tmp_path: Temporary directory of the test.

    Returns:
        The two folders.
    """
    data, output = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    _ = shutil.copytree(slice_dir, data / STACK_DIR)
    with zipfile.ZipFile(data / STACK_ZIP, "w") as archive:
        for path in sorted(slice_dir.iterdir()):
            archive.write(path, path.name)
    _ = (data / "notes.txt").write_text("not a stack")
    _ = shutil.copytree(slice_dir, data / ".hidden")
    config = small_config()
    _ = run_segmentation(data / STACK_DIR, output / SEGMENTATION, config)
    (output / "unfinished").mkdir()
    (output / "broken").mkdir()
    _ = (output / "broken" / config.io.meta_file).write_text(json.dumps({"source": "x"}))
    return Folders(data=data, output=output)


@pytest.fixture
def server_config(folders: Folders) -> "ServerConfig":
    """Point the server configuration at the fixture folders.

    Args:
        folders: Data and output folders.

    Returns:
        The server configuration.
    """
    return load_server_config(
        overrides=[
            f"library.data_dir={json.dumps(str(folders.data))}",
            f"library.output_dir={json.dumps(str(folders.output))}",
            "api.event_heartbeat_s=0.05",
        ]
    )


@pytest.fixture
def client(server_config: "ServerConfig") -> "Iterator[TestClient]":
    """Run the application (with its life cycle) around a test.

    Args:
        server_config: Server configuration.

    Yields:
        A client of the application.
    """
    with TestClient(create_app(small_config(), server_config)) as test_client:
        yield test_client
