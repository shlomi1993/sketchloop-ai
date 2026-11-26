"""
Tests of the hardware readers on each operating system and on this machine.
"""

import json
import os
import psutil
import pytest
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from sketchloop.domain import SCHEMA_VERSION, Unavailable
from sketchloop.experiments import (
    SessionRecord, collect_environment, collect_hardware, read_cpu_name, read_memory_bytes, session_to_dict)
from sketchloop.experiments.hardware import read_linux_cpu_name


def answer_sysctl(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
    """
    Stand in for sysctl on a synthetic Mac, with extra whitespace in the CPU name.
    """
    values = {"machdep.cpu.brand_string": "  Apple   M2 Pro \n"}
    return subprocess.CompletedProcess(command, 0, stdout=values[command[-1]], stderr="")


def test_mac_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    On macOS the CPU name comes from sysctl, with whitespace collapsed.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", answer_sysctl)
    assert read_cpu_name() == "Apple M2 Pro", f"Unexpected CPU name {read_cpu_name()!r}"


def time_out(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
    """
    Stand in for a sysctl call that times out.
    """
    raise subprocess.TimeoutExpired(command, 5)


def fail_command(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
    """
    Stand in for a sysctl call that exits with an error.
    """
    return subprocess.CompletedProcess(command, 1, stdout="", stderr="unknown oid")


@pytest.mark.parametrize("run_command", [time_out, fail_command], ids=["timeout", "error-exit"])
def test_mac_hardware_unavailable(monkeypatch: pytest.MonkeyPatch, run_command: object) -> None:
    """
    A sysctl call that times out or fails gives an unavailable CPU name instead of raising.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", run_command)
    hardware = collect_hardware()

    # The CPU name is unavailable, while the core count and memory still come from the operating system.
    assert isinstance(hardware.cpu, Unavailable), f"Expected an unavailable CPU name, got {hardware.cpu!r}"
    assert isinstance(hardware.memory_bytes, int), f"Memory must still be read, got {hardware.memory_bytes!r}"


def test_linux_hardware(tmp_path: Path) -> None:
    """
    On Linux the CPU name comes from the first model name line of cpuinfo.
    """
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Synthetic CPU 3000\nmodel name\t: Other\n", encoding="utf-8")
    no_name = tmp_path / "no-name"
    no_name.write_text("processor\t: 0\n", encoding="utf-8")

    # The first model name is read, and a file without one or a missing file gives a reason.
    assert read_linux_cpu_name(cpuinfo) == "Synthetic CPU 3000", f"Unexpected CPU name {read_linux_cpu_name(cpuinfo)!r}"
    assert isinstance(read_linux_cpu_name(no_name), Unavailable), "A cpuinfo without a model name must be unavailable"
    assert isinstance(read_linux_cpu_name(tmp_path / "missing"), Unavailable), "A missing cpuinfo must be unavailable"


def refuse_memory() -> None:
    """
    Stand in for a system where psutil cannot read the memory.
    """
    raise OSError("no memory information")


@pytest.mark.parametrize(("read_memory", "expected"), [
    (lambda: SimpleNamespace(total=34359738368), 34359738368),
    (lambda: SimpleNamespace(total=0), Unavailable("psutil reported no memory")),
    (refuse_memory, Unavailable("psutil could not read the memory size")),
], ids=["read", "zero", "error"])
def test_memory_from_psutil(monkeypatch: pytest.MonkeyPatch, read_memory: object, expected: int | Unavailable) -> None:
    """
    Memory is read through psutil, and a zero or failed reading gives a fixed reason instead of raising.
    """
    monkeypatch.setattr(psutil, "virtual_memory", read_memory)
    assert read_memory_bytes() == expected, f"Expected {expected!r}, got {read_memory_bytes()!r}"


def test_unknown_platform_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An operating system without a CPU name reader gives an unavailable CPU name with its reason.
    """
    monkeypatch.setattr(sys, "platform", "sunos5")
    expected = Unavailable("the CPU name is not read on this operating system")
    assert collect_hardware().cpu == expected, f"Expected an unavailable CPU name, got {collect_hardware().cpu!r}"


def test_session_record_has_no_machine_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The encoded session record never contains the machine's network name, and its hardware holds only the three facts.
    """
    sentinel = "sentinel-host-name"
    monkeypatch.setattr("platform.node", lambda: sentinel)
    monkeypatch.setattr("socket.gethostname", lambda: sentinel)

    # Replace the node name in os.uname too, on the systems that have it.
    if hasattr(os, "uname"):
        real = os.uname()
        renamed = os.uname_result((real.sysname, sentinel, real.release, real.version, real.machine))
        monkeypatch.setattr(os, "uname", lambda: renamed)
    record = SessionRecord(schema_version=SCHEMA_VERSION, session_id="synthetic", created_at=datetime.now(timezone.utc),
                           environment=collect_environment(), generator_setup_seconds=0.1, rerun_of=None, benchmark=None)

    # The machine name appears nowhere in session.json, and hardware has exactly the CPU, cores, and memory.
    encoded = session_to_dict(record)
    assert sentinel not in json.dumps(encoded), "session.json must not contain the machine name"
    hardware_keys = set(encoded["environment"]["hardware"])
    assert hardware_keys == {"cpu", "logical_cores", "memory_bytes"}, f"Unexpected hardware keys {hardware_keys}"


def test_real_hardware() -> None:
    """
    On this machine every fact is a real value or unavailable with a reason, and a supported system reports its memory.
    """
    hardware = collect_hardware()
    cpu, cores, memory = hardware.cpu, hardware.logical_cores, hardware.memory_bytes

    # Reading never raises, and each fact is either a value or names why it is missing.
    assert isinstance(cpu, str) or cpu.reason, f"Expected a CPU name or a reason, got {cpu!r}"
    assert isinstance(cores, int) or cores.reason, f"Expected a core count or a reason, got {cores!r}"

    # psutil reads the memory on every supported system.
    assert isinstance(memory, int) and memory > 0, f"Expected a memory size, got {memory!r}"
