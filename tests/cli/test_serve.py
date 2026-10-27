"""
Tests that the ui subcommand launches the page with telemetry off and the settings applied.
"""

import os
import pytest
import sys
import types

from pathlib import Path

import sketchloop
from sketchloop.app import AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.cli import main


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AppController, dict[str, object]]]:
    """
    Replace sketchloop.ui with a stub whose launch_app records its arguments and the telemetry settings, without gradio.
    """
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    calls: list[tuple[AppController, dict[str, object]]] = []

    def record_launch(controller: AppController, **options: object) -> None:
        """
        Record a launch instead of starting a server.
        """
        telemetry = {name: os.environ.get(name) for name in ("GRADIO_ANALYTICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY")}
        calls.append((controller, options | telemetry))

    stub = types.ModuleType("sketchloop.ui")
    stub.launch_app = record_launch
    monkeypatch.setitem(sys.modules, "sketchloop.ui", stub)
    monkeypatch.setattr(sketchloop, "ui", stub, raising=False)
    return calls


def test_cli_ui_launches_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                    launches: list[tuple[AppController, dict[str, object]]]) -> None:
    """
    sketchloop ui launches with the defaults and telemetry off, refuses too many candidates, and takes its model from --config.
    """
    runs_dir = tmp_path / "runs"
    exit_code = main(["ui", "--runs-dir", str(runs_dir), "--seed", "5"])
    over_limit_exit = main(["ui", "--runs-dir", str(runs_dir), "--candidates", "9"])
    errors = capsys.readouterr().err
    diffusers_calls: list[tuple[object, Path]] = []

    def record_diffusers(model_dir: Path = Path("models"), *, mode: object = None, **options: object) -> FakeGenerator:
        """
        Record the diffusers mode and models folder, and return a fake so no model loads.
        """
        diffusers_calls.append((mode, Path(model_dir)))
        return FakeGenerator()

    # A settings file in another folder chooses diffusers in quality mode, its models and runs folders, and 3 candidates.
    monkeypatch.setattr("sketchloop.backends.DiffusersSketchGenerator", record_diffusers)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "ui.toml"
    config.write_text("\n".join(['[generator]', 'backend = "diffusers"', '[generator.diffusers]', 'mode = "quality"',
                                 'models_dir = "weights"', '[storage]', 'runs_dir = "file-runs"', '[ui]', 'candidates = 3', '']),
                      encoding="utf-8")
    config_exit = main(["ui", "--config", str(config)])
    fake_flag_exit = main(["ui", "--config", str(config), "--backend", "fake"])

    # The three valid commands each launch once, and the first has a controller for the runs folder and 2 candidates.
    exit_codes = (exit_code, config_exit, fake_flag_exit)
    assert exit_codes == (0, 0, 0) and len(launches) == 3, f"Expected three launches, got {exit_codes} and {launches}"
    controller, options = launches[0]
    assert isinstance(controller, AppController) and controller.store.root == runs_dir, f"Bad controller {controller}"
    assert options["default_candidates"] == 1 and options["default_seed"] == 5, f"Unexpected launch options {options}"

    # Telemetry is off before the UI module runs.
    telemetry = (options["GRADIO_ANALYTICS_ENABLED"], options["HF_HUB_DISABLE_TELEMETRY"])
    assert telemetry == ("False", "1"), f"Telemetry must be disabled at launch, got {telemetry}"

    # Asking for more candidates than the fake allows fails with the limit and does not launch.
    limit_message = "9 candidates exceed the limit of 8. Lower --candidates or ui.candidates."
    assert over_limit_exit == 1 and limit_message in errors, f"Expected the limit error, got {over_limit_exit} and {errors!r}"
    assert not runs_dir.exists(), "Launching the UI must not create a session folder before the first round"

    # The file's diffusers backend gets quality mode and the models folder next to the file, and its other defaults apply.
    config_controller, config_options = launches[1]
    expected_call = ("quality", (config_dir / "weights").resolve())
    resolved_calls = [(mode, model_dir.resolve()) for mode, model_dir in diffusers_calls]
    assert resolved_calls == [expected_call], f"Expected one diffusers backend {expected_call}, got {diffusers_calls}"
    assert config_options["default_candidates"] == 3, f"The file's ui.candidates must apply, got {config_options}"
    controller_runs = config_controller.store.root.resolve()
    assert controller_runs == (config_dir / "file-runs").resolve(), f"Runs must sit next to the file, got {controller_runs}"

    # --backend fake beats the file without a mode conflict, so no diffusers backend is created.
    assert len(diffusers_calls) == 1, f"--backend fake must win over the file's diffusers, got {len(diffusers_calls)} calls"
