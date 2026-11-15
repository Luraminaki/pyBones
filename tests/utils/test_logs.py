"""Tests of the logging setup (:mod:`pybones.utils.logs`)."""

import logging
import os
import threading
from typing import TYPE_CHECKING

from pybones.config.loader import load_config
from pybones.utils.logs import configure_logging, log_file_path, prune_log_files

if TYPE_CHECKING:
    from pathlib import Path

STEM = "test"
THREAD_NAME = "worker-thread"


def test_one_file_per_process_with_the_full_record_layout(tmp_path: "Path") -> None:
    """Records name their time, process, thread, logger, level and function."""
    config = load_config(overrides=(f'logging.directory="{tmp_path}"',)).logging
    path = configure_logging(config, STEM)
    assert path == log_file_path(config, STEM, os.getpid())
    assert path.name == f"{STEM}-{os.getpid()}.log"

    def emit() -> None:
        """Log from a named thread."""
        logging.getLogger("pybones.test").info("hello %d", 42)

    worker = threading.Thread(target=emit, name=THREAD_NAME)
    worker.start()
    worker.join()
    line = path.read_text().strip().splitlines()[-1]
    for part in (
        f"[{os.getpid()}]",
        f"[{THREAD_NAME}]",
        "[pybones.test]",
        "[INFO]",
        "emit -- hello 42",
    ):
        assert part in line


def test_configuring_twice_replaces_the_handlers(tmp_path: "Path") -> None:
    """A second call does not duplicate records."""
    config = load_config(overrides=(f'logging.directory="{tmp_path}"',)).logging
    _ = configure_logging(config, STEM)
    path = configure_logging(config, STEM)
    logging.getLogger("pybones.test").warning("once")
    assert path.read_text().count("once") == 1


def test_files_rotate_by_size(tmp_path: "Path") -> None:
    """A file past ``max_bytes`` is rotated, keeping ``backup_count`` older files."""
    config = load_config(
        overrides=(
            f'logging.directory="{tmp_path}"',
            "logging.max_bytes=2048",
            "logging.backup_count=2",
        )
    ).logging
    path = configure_logging(config, STEM)
    for i in range(200):
        logging.getLogger("pybones.test").info("record %03d %s", i, "x" * 40)
    rotated = sorted(p.name for p in tmp_path.iterdir())
    assert rotated == [path.name, f"{path.name}.1", f"{path.name}.2"]


def test_old_runs_are_pruned_but_not_running_processes(tmp_path: "Path") -> None:
    """Only the most recent runs' files are kept; a live process's file always survives."""
    dead_pids = range(10**7, 10**7 + 5)  # far above any live PID
    for age, pid in enumerate(dead_pids):
        for name in (f"{STEM}-{pid}.log", f"{STEM}-{pid}.log.1"):
            path = tmp_path / name
            _ = path.write_text("x")
            os.utime(path, (age, age))
    live = tmp_path / f"{STEM}-{os.getpid()}.log"
    _ = live.write_text("x")
    os.utime(live, (0, 0))  # the oldest, but its process (this one) is running
    other = tmp_path / "other-1.log"
    _ = other.write_text("x")
    deleted = prune_log_files(tmp_path, STEM, keep=2)
    kept = {p.name for p in tmp_path.iterdir()}
    newest = [f"{STEM}-{pid}.log" for pid in dead_pids[-2:]]
    assert set(newest) <= kept
    assert live.name in kept
    assert other.name in kept
    assert len(deleted) == 2 * 3  # the three oldest dead runs, with their backups


def test_configuring_logging_applies_the_retention(tmp_path: "Path") -> None:
    """At most ``max_files`` runs' files remain after start-up."""
    for pid in range(10**7, 10**7 + 30):
        _ = (tmp_path / f"{STEM}-{pid}.log").write_text("x")
    config = load_config(
        overrides=(f'logging.directory="{tmp_path}"', "logging.max_files=5")
    ).logging
    _ = configure_logging(config, STEM)
    assert len(list(tmp_path.glob(f"{STEM}-*.log"))) == 5
