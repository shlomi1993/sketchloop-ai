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
    """
    Load the installer's run helper, since install.py is a script outside the package.
    """
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "install.py"))["run"]


def test_failed_command_propagates(run_command: Callable[..., None]) -> None:
    """
    A failing step must stop the installer with the child's exit code, not continue silently.
    """
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_command("Checking failure handling", sys.executable, "-c", "raise SystemExit(7)")

    # The raised error carries the child's exit code.
    assert error.value.returncode == 7, f"Expected exit code 7, got {error.value.returncode}"


def test_refresh_existing_environment_prompt(tmp_path: Path) -> None:
    """
    Refreshing the venv prompt must keep existing files and update every activation script.
    """
    # Build a venv with an old prompt and a marker file, in a path with a space to test quoting.
    environment = tmp_path / "project environment"
    venv.EnvBuilder(with_pip=False, prompt="old-name").create(environment)
    marker = environment / "keep-existing-packages.txt"
    marker.write_text("preserved")

    # Refresh the prompt in place and check that nothing else was rebuilt.
    installer = runpy.run_path(str(Path(__file__).resolve().parents[1] / "install.py"))
    installer["update_environment_prompt"](environment)
    assert marker.read_text() == "preserved", "Refreshing the prompt must keep existing files"
    assert "prompt = 'sketchloop-ai'" in (environment / "pyvenv.cfg").read_text(), "pyvenv.cfg must get the new prompt"

    # Windows cannot run the POSIX activate script, so inspect activate.bat instead.
    if os.name == "nt":
        assert "sketchloop-ai" in (environment / "Scripts/activate.bat").read_text(), "activate.bat must use the new prompt"
        return

    # Source the activate script in a real shell and report the prompt and the active interpreter prefix.
    activate = shlex.quote(str(environment / "bin/activate"))
    code = shlex.quote("import sys; print(sys.prefix)")
    command = f'. {activate} && printf "%s\\n" "$VIRTUAL_ENV_PROMPT" && python -c {code}'
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()

    # The activated shell shows the new prompt and uses the venv's interpreter.
    assert "sketchloop-ai" in lines[0], f"Activated prompt must be sketchloop-ai, got {lines[0]!r}"
    assert Path(lines[1]).resolve() == environment.resolve(), f"Activation must select the venv, got {lines[1]!r}"
