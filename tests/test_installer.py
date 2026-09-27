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
def run_command() -> Callable[[str, str], None]:
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "install.py"))["run"]


def test_command_preserves_quoted_arguments(tmp_path: Path, run_command: Callable[[str, str], None]) -> None:
    output = tmp_path / "directory with spaces" / "result.txt"
    output.parent.mkdir()
    value = "literal ; $variable 'quoted' \\path"
    code = "import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    run_command("Checking quoted arguments", f"{command} {shlex.quote(str(output))} {shlex.quote(value)}")
    assert output.read_text() == value


def test_failed_command_propagates(run_command: Callable[[str, str], None]) -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_command("Checking failure handling", f'{shlex.quote(sys.executable)} -c "raise SystemExit(7)"')

    assert error.value.returncode == 7


def test_refresh_existing_environment_prompt(tmp_path: Path) -> None:
    environment = tmp_path / "project environment"
    venv.EnvBuilder(with_pip=False, prompt="old-name").create(environment)
    marker = environment / "keep-existing-packages.txt"
    marker.write_text("preserved")
    installer = runpy.run_path(str(Path(__file__).resolve().parents[1] / "install.py"))
    installer["update_environment_prompt"](environment)
    assert marker.read_text() == "preserved"
    assert "prompt = 'sketchloop-ai'" in (environment / "pyvenv.cfg").read_text()
    if os.name == "nt":
        assert "sketchloop-ai" in (environment / "Scripts/activate.bat").read_text()
        return

    activate = shlex.quote(str(environment / "bin/activate"))
    code = shlex.quote("import sys; print(sys.prefix)")
    command = f'. {activate} && printf "%s\\n" "$VIRTUAL_ENV_PROMPT" && python -c {code}'
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert "sketchloop-ai" in lines[0]
    assert Path(lines[1]).resolve() == environment.resolve()
