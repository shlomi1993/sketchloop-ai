"""
Readers for the CPU name, logical cores, and memory size, never the machine's name.
"""

import os
import psutil
import re
import subprocess
import sys

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Final

from sketchloop.domain import Unavailable
from sketchloop.experiments.records import Hardware


def read_sysctl(name: str) -> str | Unavailable:
    """
    Read one named macOS sysctl value, or a fixed reason when the command fails or times out.
    """
    # Bound the command by a timeout, and give a fixed reason, never exception text, which could contain a path.
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return Unavailable(f"sysctl could not run to read {name}")

    value = result.stdout.strip()
    return value if result.returncode == 0 and value else Unavailable(f"sysctl reported no {name}")


def read_linux_cpu_name(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | Unavailable:
    """
    Read the first model name line of the Linux cpuinfo file.
    """
    try:
        lines = cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return Unavailable("/proc/cpuinfo is not readable")

    # Take the value after the colon on the first line that names the model.
    names = [line.partition(":")[2].strip() for line in lines if line.startswith("model name")]
    return names[0] if names and names[0] else Unavailable("/proc/cpuinfo has no model name")


def read_windows_cpu_name() -> str | Unavailable:
    """
    Read the processor name that Windows keeps in the registry for the first CPU.
    """
    # Import winreg only here, since it exists only on Windows, and treat a missing key as unavailable.
    try:
        import winreg

        windows_cpu_key: Final = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, windows_cpu_key) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except (ImportError, OSError):
        return Unavailable("the Windows registry has no processor name")

    return value if isinstance(value, str) and value.strip() else Unavailable("the Windows registry processor name is empty")


def clean_cpu_name(name: str | Unavailable) -> str | Unavailable:
    """
    Collapse whitespace in a CPU name and cut it to 120 characters, keeping an unavailable name as it is.
    """
    if isinstance(name, Unavailable):
        return name

    cleaned = re.sub(r"\s+", " ", name).strip()[:120]
    return cleaned or Unavailable("the CPU name was only whitespace")


# CPU name readers by sys.platform value, since each operating system reports the marketing name differently.
CPU_NAME_READERS: Final[dict[str, Callable[[], str | Unavailable]]] = {
    "darwin": partial(read_sysctl, "machdep.cpu.brand_string"),
    "linux": read_linux_cpu_name,
    "win32": read_windows_cpu_name,
}


def read_cpu_name() -> str | Unavailable:
    """
    Read the processor model name the way this operating system reports it, cleaned for the record.
    """
    reader = CPU_NAME_READERS.get(sys.platform)
    return clean_cpu_name(reader()) if reader else Unavailable("the CPU name is not read on this operating system")


def read_memory_bytes() -> int | Unavailable:
    """
    Read the total physical memory with psutil, which works the same way on every operating system.
    """
    # Give a fixed reason, never exception text, when the system does not report its memory.
    try:
        memory_bytes = int(psutil.virtual_memory().total)
    except (OSError, RuntimeError, NotImplementedError):
        return Unavailable("psutil could not read the memory size")

    return memory_bytes if memory_bytes > 0 else Unavailable("psutil reported no memory")


def collect_hardware() -> Hardware:
    """
    Read only the CPU name, logical core count, and total memory, never a machine name, user name, or serial number.
    """
    cores = os.cpu_count()
    logical_cores = cores if cores else Unavailable("os.cpu_count reported no cores")
    return Hardware(cpu=read_cpu_name(), logical_cores=logical_cores, memory_bytes=read_memory_bytes())
