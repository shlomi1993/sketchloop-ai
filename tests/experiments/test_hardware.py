import json
import os
import pytest
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path

from sketchloop.domain import SCHEMA_VERSION, Unavailable
from sketchloop.experiments import (SessionRecord, collect_environment, collect_hardware, read_cpu_name, read_memory_bytes,
                                    session_to_dict)
from sketchloop.experiments.hardware import MEMORY_READERS, read_linux_cpu_name


def answer_sysctl(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
    """
    Stand in for sysctl on a synthetic Mac, with extra whitespace in the CPU name.
    """
    values = {"machdep.cpu.brand_string": "  Apple   M2 Pro \n", "hw.memsize": "34359738368\n"}
    return subprocess.CompletedProcess(command, 0, stdout=values[command[-1]], stderr="")


def test_mac_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    On macOS the CPU name and memory come from sysctl, with whitespace collapsed.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", answer_sysctl)

    # The CPU name is cleaned, and the memory is read as a whole number of bytes.
    assert read_cpu_name() == "Apple M2 Pro", f"Unexpected CPU name {read_cpu_name()!r}"
    assert read_memory_bytes() == 34359738368, f"Unexpected memory {read_memory_bytes()!r}"


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


def miss_command(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
    """
    Stand in for a system without the sysctl command.
    """
    raise FileNotFoundError(command[0])


@pytest.mark.parametrize("run_command", [time_out, fail_command, miss_command], ids=["timeout", "error-exit", "missing"])
def test_mac_hardware_unavailable(monkeypatch: pytest.MonkeyPatch, run_command: object) -> None:
    """
    A sysctl call that times out, fails, or is missing gives unavailable facts instead of raising.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", run_command)
    hardware = collect_hardware()

    # Both sysctl facts are unavailable, while the core count still comes from the operating system.
    unavailable = (isinstance(hardware.cpu, Unavailable), isinstance(hardware.memory_bytes, Unavailable))
    assert unavailable == (True, True), f"Expected unavailable CPU and memory, got {hardware}"


def test_linux_hardware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    On Linux the CPU name comes from the first model name line, and memory from page size times page count.
    """
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Synthetic CPU 3000\nmodel name\t: Other\n", encoding="utf-8")
    no_name = tmp_path / "no-name"
    no_name.write_text("processor\t: 0\n", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("os.sysconf", lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1000}[name])

    # The first model name is read, and a file without one or a missing file gives a reason.
    assert read_linux_cpu_name(cpuinfo) == "Synthetic CPU 3000", f"Unexpected CPU name {read_linux_cpu_name(cpuinfo)!r}"
    assert isinstance(read_linux_cpu_name(no_name), Unavailable), "A cpuinfo without a model name must be unavailable"
    assert isinstance(read_linux_cpu_name(tmp_path / "missing"), Unavailable), "A missing cpuinfo must be unavailable"

    # The memory is the page size times the number of pages.
    assert read_memory_bytes() == 4096 * 1000, f"Unexpected memory {read_memory_bytes()!r}"


def test_unknown_platform_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An operating system without a reader gives unavailable CPU and memory, each with its own reason.
    """
    monkeypatch.setattr(sys, "platform", "sunos5")
    hardware = collect_hardware()

    # Each fact says which value this operating system does not read.
    expected = (Unavailable("the CPU name is not read on this operating system"),
                Unavailable("the memory size is not read on this operating system"))
    assert (hardware.cpu, hardware.memory_bytes) == expected, f"Expected unavailable facts, got {hardware}"


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
                           environment=collect_environment(), generator_setup_seconds=0.1, rerun_of=None)

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

    # macOS, Linux, and Windows each have a memory reader, so the size is read for real there.
    if sys.platform in MEMORY_READERS:
        assert isinstance(memory, int) and memory > 0, f"Expected a memory size, got {memory!r}"
