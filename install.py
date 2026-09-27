import os
import shlex
import subprocess
import sys
import venv

from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
ENV_NAME = "sketchloop-ai"


def run(label: str, command: str) -> None:
    print(label, flush=True)
    subprocess.run(shlex.split(command), cwd=ROOT, check=True)


def update_environment_prompt(environment: Path) -> None:
    builder = venv.EnvBuilder(prompt=ENV_NAME)
    context = builder.ensure_directories(str(environment))
    builder.setup_scripts(context)
    config = environment / "pyvenv.cfg"
    lines = [line for line in config.read_text().splitlines() if not line.startswith("prompt =")]
    config.write_text("\n".join([*lines, f"prompt = {ENV_NAME!r}"]) + "\n")


def main() -> int:
    if sys.version_info < (3, 11):
        print("Python 3.11 or newer is required.", file=sys.stderr)
        return 1

    python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if VENV.is_symlink():
            raise RuntimeError(".venv is a symlink; use a local project environment.")

        if VENV.exists():
            if not (VENV / "pyvenv.cfg").is_file() or not python.is_file():
                raise RuntimeError("Existing .venv is invalid or incomplete. Move it aside and rerun the installer.")

            print("Reusing .venv.", flush=True)
        else:
            run("Creating .venv...", f"{shlex.quote(sys.executable)} -m venv --prompt {ENV_NAME} {shlex.quote(str(VENV))}")

        executable = shlex.quote(str(python))
        interpreter_check = (
            "import sys; from pathlib import Path; "
            "valid = sys.version_info >= (3, 11) and sys.prefix != sys.base_prefix "
            "and Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(); "
            "sys.exit(0 if valid else 'Invalid .venv: move it aside and rerun the installer.')"
        )
        run(
            "Checking the environment interpreter...",
            f"{executable} -c {shlex.quote(interpreter_check)} {shlex.quote(str(VENV))}",
        )
        update_environment_prompt(VENV)
        run(
            "Installing the project and development dependencies...",
            f'{executable} -m pip --disable-pip-version-check install -e ".[dev]"',
        )
        run("Checking dependency compatibility...", f"{executable} -m pip check")
        run("Running readiness tests...", f"{executable} -m pytest -q tests/test_environment.py")

    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("Installation did not complete: " + str(exc), file=sys.stderr)
        print("Resolve the error above and rerun. Python must include venv/pip; "
              "dependency installation needs package-index access or cached packages.", file=sys.stderr)
        return 1

    activation = r".venv\Scripts\Activate.ps1" if os.name == "nt" else ". .venv/bin/activate"
    print(f"\nEnvironment ready: {ENV_NAME} (stored in .venv).")
    print("Activate it in your current terminal with the command below.")
    print("The installer cannot change the environment of the shell that launched it.")
    if os.environ.get("CONDA_DEFAULT_ENV"):
        print("  conda deactivate  # Leave the active Conda environment first")

    print("From the repository root:")
    print("  " + activation)
    print("  python scripts/check.py  # Full repository checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
