"""Logging setup of the pyBones entry points (the command and the tools).

Library modules only create their logger (``logging.getLogger(__name__)``); only
an entry point configures where records go, by calling :func:`configure_logging`.

Every record says when, in which process and thread, in which module and function
something happened::

    [2026-11-15 20:04:12,345] [12345] [MainThread] [pybones.segmentation.pipeline] [INFO]:
        run_segmentation -- Stack: 100 slices of 2300 x 1487

(one line in the file; wrapped here).

**One file per process.** Each process writes ``<directory>/<stem>-<pid>.log``.
:class:`~logging.handlers.RotatingFileHandler` renames the file when it grows past
``max_bytes``; two processes sharing one file would orphan each other's handle at
the first rotation and silently lose records. Per-process files avoid that.
**Threads** of a process share its handlers safely: every handler serialises
``emit`` with its own lock, so records from the worker threads never interleave.

**Retention.** One file per run would pile up, so each start-up keeps only the
``max_files`` most recent log files of its program (with their rotated backups) and
deletes the older ones, except those of processes still running.
"""

import contextlib
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybones.config.schema import LoggingConfig

#: Record layout: time, process, thread, logger, level, function, message.
LOG_FORMAT = (
    "[%(asctime)s] [%(process)d] [%(threadName)s] [%(name)s] [%(levelname)s]: "
    + "%(funcName)s -- %(message)s"
)
#: Console layout: short, for a person watching the terminal (the file keeps the details).
CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"
LOG_FILE_TEMPLATE = "{stem}-{pid}.log"
#: Matches a program's log files and their rotated backups: ``<stem>-<pid>.log[.<n>]``.
_LOG_FILE_PATTERN = r"^{stem}-(?P<pid>\d+)\.log(?:\.\d+)?$"
_POSIX = "posix"
_ENCODING = "utf-8"
#: Marks the handlers installed by :func:`configure_logging` (so a second call replaces them).
_HANDLER_ATTRIBUTE = "_pybones_handler"


def log_file_path(config: "LoggingConfig", stem: str, pid: int | None = None) -> Path:
    """Name the log file of a process.

    Args:
        config: Logging settings (directory).
        stem: Program name, e.g. ``pybones`` or ``tools-report``.
        pid: Process id (default: the current process).

    Returns:
        ``<directory>/<stem>-<pid>.log``.
    """
    pid = os.getpid() if pid is None else pid
    return Path(config.directory) / LOG_FILE_TEMPLATE.format(stem=stem, pid=pid)


def _is_running(pid: int) -> bool:
    """Tell whether a process is still running (best effort).

    Args:
        pid: Process id.

    Returns:
        ``True`` if the process exists. Outside POSIX systems this cannot be checked
        safely and ``False`` is returned: deleting a file another process holds open
        then fails, and is skipped.
    """
    if os.name != _POSIX:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prune_log_files(directory: Path, stem: str, keep: int) -> list[Path]:
    """Delete a program's oldest log files, keeping the ``keep`` most recent runs.

    Each run's file is kept or deleted together with its rotated backups; files of
    processes still running are never deleted.

    Args:
        directory: Log directory.
        stem: Program name.
        keep: Number of runs whose files are kept.

    Returns:
        The deleted files.
    """
    pattern = re.compile(_LOG_FILE_PATTERN.format(stem=re.escape(stem)))
    runs: dict[int, list[Path]] = {}
    for path in directory.glob(f"{stem}-*.log*"):
        match = pattern.match(path.name)
        if match:
            runs.setdefault(int(match["pid"]), []).append(path)

    def newest(pid: int) -> float:
        """Time of the latest write of a run.

        Args:
            pid: Process id of the run.

        Returns:
            Latest modification time of its files.
        """
        return max(p.stat().st_mtime for p in runs[pid])

    old = sorted(runs, key=newest, reverse=True)[keep:]
    deleted: list[Path] = []
    for pid in old:
        if _is_running(pid):
            continue
        for path in runs[pid]:
            with contextlib.suppress(OSError):
                path.unlink()
                deleted.append(path)
    return deleted


def configure_logging(config: "LoggingConfig", stem: str) -> Path:
    """Send the records of this process to its own rotating log file and to stderr.

    Replaces the handlers a previous call installed (other handlers are left alone),
    so calling it twice is harmless.

    Args:
        config: Logging settings (directory, levels, rotation).
        stem: Program name, used in the file name.

    Returns:
        The log file of this process.
    """
    path = log_file_path(config, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Room for this run's file among the kept ones.
    deleted = prune_log_files(path.parent, stem, config.max_files - 1)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        path, maxBytes=config.max_bytes, backupCount=config.backup_count, encoding=_ENCODING
    )
    file_handler.setLevel(config.file_level)
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(config.console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT))

    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_ATTRIBUTE, False)]:
        root.removeHandler(handler)
        handler.close()
    for handler in (file_handler, console):
        setattr(handler, _HANDLER_ATTRIBUTE, True)
        root.addHandler(handler)
    root.setLevel(min(file_handler.level, console.level))
    if deleted:
        logging.getLogger(__name__).info("Deleted %d old log files of %s", len(deleted), stem)
    return path
