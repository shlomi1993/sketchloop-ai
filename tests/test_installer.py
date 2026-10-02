import os
import pytest
import runpy
import shlex
import subprocess
import sys
import venv

from collections.abc import Callable
from pathlib import Path


@pytest.fixture
def run_command() -> Callable[..., None]:
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "install.py"))["run"]


def test_failed_command_propagates(run_command: Callable[..., None]) -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_command("Checking failure handling", sys.executable, "-c", "raise SystemExit(7)")

    assert error.value.returncode == 7, f"Expected exit code 7, got {error.value.returncode}"


def test_refresh_existing_environment_prompt(tmp_path: Path) -> None:
    environment = tmp_path / "project environment"
    venv.EnvBuilder(with_pip=False, prompt="old-name").create(environment)
    marker = environment / "keep-existing-packages.txt"
    marker.write_text("preserved")
    installer = runpy.run_path(str(Path(__file__).resolve().parents[1] / "install.py"))
    installer["update_environment_prompt"](environment)
    assert marker.read_text() == "preserved", "Refreshing the prompt must keep existing files"
    assert "prompt = 'sketchloop-ai'" in (environment / "pyvenv.cfg").read_text(), "pyvenv.cfg must get the new prompt"
    if os.name == "nt":
        assert "sketchloop-ai" in (environment / "Scripts/activate.bat").read_text(), "activate.bat must use the new prompt"
        return

    activate = shlex.quote(str(environment / "bin/activate"))
    code = shlex.quote("import sys; print(sys.prefix)")
    command = f'. {activate} && printf "%s\\n" "$VIRTUAL_ENV_PROMPT" && python -c {code}'
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert "sketchloop-ai" in lines[0], f"Activated prompt must be sketchloop-ai, got {lines[0]!r}"
    assert Path(lines[1]).resolve() == environment.resolve(), f"Activation must select the venv, got {lines[1]!r}"
