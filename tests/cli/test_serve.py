"""
Tests that the ui subcommand launches the page with telemetry off and the settings applied.
"""

import io
import os
import pytest
import sys
import types

from pathlib import Path
from rich.console import Console

import sketchloop
from sketchloop.app import AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.cli import main
from sketchloop.cli.serve import warm_up_suggester
from sketchloop.experiments import load_session
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.suggestion import FLORENCE_MODEL, FlorenceSuggester, SuggestionError, TemplateSuggester


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


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
    sketchloop start launches with telemetry off, refuses too many candidates, and takes its components from --config.
    """
    runs_dir = tmp_path / "runs"
    fakes = ["--backend", "fake", "--stroke-suggester", "fake"]
    exit_code = main(["start", *fakes, "--runs-dir", str(runs_dir), "--seed", "5"])
    over_limit_exit = main(["start", *fakes, "--runs-dir", str(runs_dir), "--candidates", "9"])
    errors = capsys.readouterr().err
    diffusers_calls: list[tuple[object, Path]] = []

    def record_diffusers(model_dir: Path = Path("models"), *, mode: object = None, **options: object) -> FakeGenerator:
        """
        Record the diffusers mode and models folder, and return a fake so no model loads.
        """
        diffusers_calls.append((mode, Path(model_dir)))
        return FakeGenerator()

    # A settings file in another folder chooses diffusers in quality mode, its models and runs folders, and 3 candidates.
    monkeypatch.setattr("sketchloop.components.DiffusersSketchGenerator", record_diffusers)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "ui.toml"
    config.write_text("\n".join(['[generator]', 'backend = "diffusers"', '[generator.diffusers]', 'mode = "quality"',
                                 '[models]', 'dir = "weights"', '[storage]', 'runs_dir = "file-runs"', '[ui]', 'candidates = 3',
                                 '[prompt_suggester]', 'name = "florence"', '[stroke_suggester]', 'name = "fake"', '[drawing]',
                                 'canvas_width = 640', 'canvas_height = 480', 'save_interval_seconds = 3.0', '']),
                      encoding="utf-8")
    config_exit = main(["start", "--config", str(config)])
    fake_flag_exit = main(["start", "--config", str(config), "--backend", "fake"])

    # The three valid commands each launch once, and the first has a controller for the runs folder and 2 candidates.
    exit_codes = (exit_code, config_exit, fake_flag_exit)
    assert exit_codes == (0, 0, 0) and len(launches) == 3, f"Expected three launches, got {exit_codes} and {launches}"
    controller, options = launches[0]
    assert isinstance(controller, AppController) and controller.store.root == runs_dir, f"Bad controller {controller}"

    # Telemetry is off before the UI module runs.
    telemetry = (options["GRADIO_ANALYTICS_ENABLED"], options["HF_HUB_DISABLE_TELEMETRY"])
    assert telemetry == ("False", "1"), f"Telemetry must be disabled at launch, got {telemetry}"

    # Asking for more candidates than the fake allows fails with the limit and does not launch.
    limit_message = "9 candidates exceed the limit of 8. Lower --candidates or ui.candidates."
    assert over_limit_exit == 1 and limit_message in errors, f"Expected the limit error, got {over_limit_exit} and {errors!r}"
    assert not runs_dir.exists(), "Launching the UI must not create a session folder before the first round"

    # The file's diffusers backend gets quality mode and the models folder next to the file, and its other defaults apply.
    config_controller = launches[1][0]
    expected_call = ("quality", (config_dir / "weights").resolve())
    resolved_calls = [(mode, model_dir.resolve()) for mode, model_dir in diffusers_calls]
    assert resolved_calls == [expected_call], f"Expected one diffusers backend {expected_call}, got {diffusers_calls}"
    controller_runs = config_controller.store.root.resolve()
    assert controller_runs == (config_dir / "file-runs").resolve(), f"Runs must sit next to the file, got {controller_runs}"

    # The file's Florence-2 prompt suggester reads its model from the same [models] folder as the backend.
    suggester_folder = config_controller.suggester._model_folder.resolve()
    expected_folder = (config_dir / "weights" / FLORENCE_MODEL.folder).resolve()
    assert suggester_folder == expected_folder, f"The suggester must read {expected_folder}, got {suggester_folder}"

    # The controller holds the file's stroke suggester and drawing settings.
    assert isinstance(config_controller.stroke_suggester, FakeStrokeSuggester), f"Bad suggester {config_controller}"
    drawing = (config_controller.canvas_size, config_controller.save_interval_seconds)
    assert drawing == ((640, 480), 3.0), f"Expected the file's canvas and interval, got {drawing}"

    # --backend fake beats the file without a mode conflict, so no diffusers backend is created.
    assert len(diffusers_calls) == 1, f"--backend fake must win over the file's diffusers, got {len(diffusers_calls)} calls"


def test_cli_ui_builds_prompt_suggester_from_settings(tmp_path: Path,
                                                      launches: list[tuple[AppController, dict[str, object]]]) -> None:
    """
    The ui command builds the templates by default and Florence-2 from the file, which warns and records a failed press.
    """
    config = tmp_path / "prompts.toml"
    config.write_text('[prompt_suggester]\nname = "florence"\n\n[models]\ndir = "no-weights"\n', encoding="utf-8")
    default_exit = main(["start", "--backend", "fake", "--stroke-suggester", "fake"])
    florence_exit = main(["start", "--backend", "fake", "--stroke-suggester", "fake", "--config", str(config)])
    default_controller, florence_controller = launches[0][0], launches[1][0]

    # The templates are the default with the fake backend, and the file chooses Florence-2 with the fake backend too.
    assert (default_exit, florence_exit) == (0, 0), f"Expected two launches, got {default_exit}, {florence_exit}"
    assert isinstance(default_controller.suggester, TemplateSuggester), f"Got {type(default_controller.suggester).__name__}"
    assert isinstance(florence_controller.suggester, FlorenceSuggester), f"Got {type(florence_controller.suggester).__name__}"

    # The warm-up prints why Florence-2 is unavailable without weights.
    output = io.StringIO()
    warm_up_suggester(florence_controller.suggester, Console(file=output, width=200))
    assert "Prompt suggestions are unavailable: Model folder" in output.getvalue(), f"Got {output.getvalue()!r}"

    # A press is then refused with the download hint and recorded as failed in the next round.
    with pytest.raises(SuggestionError, match="download_models.py"):
        florence_controller.request_prompt_suggestion(EXAMPLE_SKETCH, 0, False)
    florence_controller.run_round(EXAMPLE_SKETCH, "upload", "a chair", n_candidates=1)
    recorded = load_session(florence_controller.session.folder).rounds[0].prompt_suggestions
    assert [item.status for item in recorded] == ["failed"], f"Expected one failed press, got {recorded}"


def test_cli_ui_warns_about_stand_ins(capsys: pytest.CaptureFixture[str],
                                      launches: list[tuple[AppController, dict[str, object]]]) -> None:
    """
    The page prints a warning for each stand-in suggester before it opens.
    """
    exit_code = main(["start", "--backend", "fake", "--stroke-suggester", "fake"])
    output = " ".join(capsys.readouterr().out.split())

    # Both stand-ins are named before the page opens.
    assert exit_code == 0 and len(launches) == 1, f"Expected one launch, got {exit_code}"
    assert "fake stroke suggester" in output and "fixed style templates" in output, f"Missing warnings in {output}"
