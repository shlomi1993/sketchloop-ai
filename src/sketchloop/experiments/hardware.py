"""
Readers for the CPU name, logical cores, and memory size, never the machine's name.
"""

import ctypes
import os
import re
import subprocess
import sys

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Final

from sketchloop.domain import Unavailable
from sketchloop.experiments.records import Hardware


class MemoryStatus(ctypes.Structure):
    """
    The MEMORYSTATUSEX structure that GlobalMemoryStatusEx fills in on Windows.
    """
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


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


def read_windows_memory_bytes() -> int | Unavailable:
    """
    Read the total physical memory through GlobalMemoryStatusEx.
    """
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)

    # ctypes.windll exists only on Windows, so its absence also means the size is unavailable.
    try:
        succeeded = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return Unavailable("GlobalMemoryStatusEx could not be called")

    memory_bytes = int(status.ullTotalPhys)
    return memory_bytes if succeeded and memory_bytes > 0 else Unavailable("GlobalMemoryStatusEx reported no memory")


def read_linux_memory_bytes() -> int | Unavailable:
    """
    Multiply the page size by the number of physical pages.
    """
    # Some systems lack these sysconf names, which also means the size is unavailable.
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return Unavailable("sysconf has no page size or page count")

    return memory_bytes if memory_bytes > 0 else Unavailable("sysconf reported no memory")


def clean_cpu_name(name: str | Unavailable) -> str | Unavailable:
    """
    Collapse whitespace in a CPU name and cut it to 120 characters, keeping an unavailable name as it is.
    """
    if isinstance(name, Unavailable):
        return name

    cleaned = re.sub(r"\s+", " ", name).strip()[:120]
    return cleaned or Unavailable("the CPU name was only whitespace")


def read_mac_memory_bytes() -> int | Unavailable:
    """
    Read the total physical memory that macOS reports as text, which must be a whole positive number.
    """
    text = read_sysctl("hw.memsize")
    if isinstance(text, Unavailable):
        return text

    return int(text) if text.isdecimal() and int(text) > 0 else Unavailable("sysctl hw.memsize is not a positive whole number")


# Readers by sys.platform value, so each fact is read only the way its operating system reports it.
CPU_NAME_READERS: Final[dict[str, Callable[[], str | Unavailable]]] = {
    "darwin": partial(read_sysctl, "machdep.cpu.brand_string"),
    "linux": read_linux_cpu_name,
    "win32": read_windows_cpu_name,
}
MEMORY_READERS: Final[dict[str, Callable[[], int | Unavailable]]] = {
    "darwin": read_mac_memory_bytes,
    "linux": read_linux_memory_bytes,
    "win32": read_windows_memory_bytes,
}


def read_cpu_name() -> str | Unavailable:
    """
    Read the processor model name the way this operating system reports it, cleaned for the record.
    """
    reader = CPU_NAME_READERS.get(sys.platform)
    return clean_cpu_name(reader()) if reader else Unavailable("the CPU name is not read on this operating system")


def read_memory_bytes() -> int | Unavailable:
    """
    Read the total physical memory the way this operating system reports it.
    """
    reader = MEMORY_READERS.get(sys.platform)
    return reader() if reader else Unavailable("the memory size is not read on this operating system")


def collect_hardware() -> Hardware:
    """
    Read only the CPU name, logical core count, and total memory, never a machine name, user name, or serial number.
    """
    cores = os.cpu_count()
    logical_cores = cores if cores else Unavailable("os.cpu_count reported no cores")
    return Hardware(cpu=read_cpu_name(), logical_cores=logical_cores, memory_bytes=read_memory_bytes())
