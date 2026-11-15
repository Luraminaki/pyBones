"""Description of the machine a benchmark runs on (standard library only).

Timings and memory peaks only mean something next to the hardware that produced
them, so every timed run records the machine it ran on.
"""

import os
import platform
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

#: Linux CPU description (absent on other systems: :func:`platform.processor` is used).
CPUINFO = Path("/proc/cpuinfo")
_CPU_MODEL_KEY = "model name"
_KEY_SEPARATOR = ":"
_BYTES_PER_GIB = 1 << 30
_UNKNOWN = "unknown"


class Machine(BaseModel):
    """Hardware and software of a benchmark machine.

    Attributes:
        cpu: CPU model.
        logical_cores: Logical CPUs (hardware threads).
        memory_gib: Physical memory, in GiB (0 if unknown).
        system: Operating system.
        python: Python version that ran the benchmark tool.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    cpu: str
    logical_cores: int
    memory_gib: float
    system: str
    python: str


def cpu_model() -> str:
    """Name the CPU.

    Returns:
        The model name from ``/proc/cpuinfo`` on Linux, else :func:`platform.processor`.
    """
    if CPUINFO.is_file():
        for line in CPUINFO.read_text().splitlines():
            key, sep, value = line.partition(_KEY_SEPARATOR)
            if sep and key.strip() == _CPU_MODEL_KEY:
                return value.strip()
    return platform.processor() or _UNKNOWN


def memory_gib() -> float:
    """Measure the physical memory.

    Returns:
        Physical memory in GiB, or 0 where ``os.sysconf`` cannot tell (Windows).
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _BYTES_PER_GIB
    except (AttributeError, OSError, ValueError):
        return 0.0


def system_name() -> str:
    """Name the operating system.

    Returns:
        The distribution name on Linux (with the kernel release), else the platform name.
    """
    try:
        distribution = platform.freedesktop_os_release()["PRETTY_NAME"]
    except (OSError, KeyError):
        return platform.platform(terse=True)
    return f"{distribution} (Linux {platform.release()})"


def current_machine() -> Machine:
    """Describe the machine running this process.

    Returns:
        Its description.
    """
    return Machine(
        cpu=cpu_model(),
        logical_cores=os.cpu_count() or 0,
        memory_gib=memory_gib(),
        system=system_name(),
        python=platform.python_version(),
    )
