import os
import pytest
import shutil
import sys
import types

from pathlib import Path

import sketchloop
from sketchloop.app import AppController
from sketchloop.capture import load_sketch_file
from sketchloop.cli import main
from sketchloop.experiments import load_session
from sketchloop.fakes import FakeGenerator


EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch-photo.jpg"


def test_cli_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A run on the example photo stores raw and processed sketches and candidates, labels the fake backend, and records the pick.
    """
    sketch = tmp_path / "sketch-photo.jpg"
    shutil.copyfile(EXAMPLE_SKETCH, sketch)
    answers = iter(["²", "1 01", "1", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))

    exit_code = main(["run", str(sketch), "--prompt", "modern chair", "--runs-dir", str(tmp_path / "runs")])
    output = capsys.readouterr().out
    session_dirs = list((tmp_path / "runs").iterdir())
    round_dir = session_dirs[0] / "round-1"

    # The run succeeds, labels the fake backend, and keeps the raw sketch byte for byte.
    assert exit_code == 0, "A valid run must exit with code 0"
    assert "FAKE BACKEND" in output, "The output must say the backend is fake"
    assert len(session_dirs) == 1 and (round_dir / "sketch-raw.jpg").read_bytes() == sketch.read_bytes(), "Raw sketch not kept"

    # The processed sketch, its steps, four candidates, and the selection are saved and shown.
    steps_shown = "Preprocessing: grayscale, rotate, isolate_paper, extract_strokes, crop_to_drawing" in output
    processed_saved_and_shown = (round_dir / "sketch.png").is_file() and steps_shown
    assert processed_saved_and_shown, "The processed sketch must be saved and its steps shown"
    assert len(list((round_dir / "candidates").glob("*.png"))) == 4, "Four candidate PNGs must be written by default"
    assert "Selected: 1" in output, "The summary must show the selected candidate"

    # A superscript digit and a repeat written as "01" each ask again instead of ending the session.
    hints = output.count("Use distinct numbers from 1 to 4")
    assert hints == 2, f"Each invalid answer must ask again, got {hints} hints"


def test_cli_session_survives_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A failed round and a missing new file are reported, and the session continues to the menu and summary.
    """
    answers = iter(["f missing.png", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))

    # The fake allows at most 8 candidates, so this round fails in generation.
    arguments = ["run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", "9", "--runs-dir", str(tmp_path / "runs")]
    exit_code = main(arguments)
    output = capsys.readouterr().out

    # Both errors are shown, and the session still ends normally with its summary.
    assert exit_code == 0, f"Quitting after errors must exit with code 0, got {exit_code}"
    assert "Round failed: Unsupported request" in output, "The failed round must be reported"
    assert "Cannot read sketch missing.png" in output and "Choose again." in output, "The missing file must be reported"
    assert "Rounds: 0" in output, "The summary must still be printed"


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


@pytest.mark.usefixtures("launches")
@pytest.mark.parametrize(("command", "option"), [
    ("run", ["--candidates", "0"]),
    ("run", ["--seed", "-1"]),
    ("ui", ["--candidates", "0"]),
    ("ui", ["--mode", "fast"])
])
def test_cli_rejects_invalid_options(tmp_path: Path, command: str, option: list[str]) -> None:
    """
    A candidate count below 1, a negative seed, or a mode without the diffusers backend is rejected before anything is saved.
    """
    runs_option = ["--runs-dir", str(tmp_path / "runs")]
    base = {"run": ["run", str(EXAMPLE_SKETCH), "--prompt", "chair", *runs_option], "ui": ["ui", *runs_option]}[command]

    # Argparse exits with its usage error code.
    with pytest.raises(SystemExit) as raised:
        main([*base, *option])

    # Nothing is saved for the rejected command.
    assert raised.value.code == 2, f"Expected the usage error code 2, got {raised.value.code}"
    assert not (tmp_path / "runs").exists(), "No session folder may be created for invalid options"


def test_cli_rounds_track_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A camera round, a repeat, and a new file are saved with one backend instance and sources camera, camera, and file.
    """
    answers = iter(["1", "", "", f"f {EXAMPLE_SKETCH}", "", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    monkeypatch.setattr("sketchloop.cli.capture_from_camera", lambda camera_index: load_sketch_file(EXAMPLE_SKETCH))
    created: list[FakeGenerator] = []

    def create_counted_fake() -> FakeGenerator:
        """
        Create a fake generator and remember it, so the test can count backend creations.
        """
        created.append(FakeGenerator())
        return created[-1]

    monkeypatch.setattr("sketchloop.backends.FakeGenerator", create_counted_fake)

    exit_code = main(["run", "--camera", "--prompt", "chair", "--candidates", "2", "--runs-dir", str(tmp_path / "runs")])
    output = capsys.readouterr().out
    session_dirs = list((tmp_path / "runs").iterdir())
    sources = [record.source for record in load_session(session_dirs[0]).rounds]

    # The session ends cleanly with three rounds and a single backend instance.
    assert exit_code == 0, "A session ended with q must exit with code 0"
    assert len(created) == 1, f"The backend must be created once per session, got {len(created)}"

    # A repeat keeps the camera source, and a new file switches to file, even with the same bytes.
    assert sources == ["camera", "camera", "file"], f"Expected sources camera, camera, file, got {sources}"

    # The summary lists every round with its selection.
    selections_shown = all(line in output for line in ("Round 1 selected: 1", "Round 2 selected: none", "Round 3 selected: none"))
    assert "Rounds: 3" in output and selections_shown, "The summary must list all three rounds and their selections"


def test_cli_ui_launches_controller(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                    launches: list[tuple[AppController, dict[str, object]]]) -> None:
    """
    sketchloop ui launches one fake-backed controller with the defaults and telemetry off, and refuses too many candidates.
    """
    runs_dir = tmp_path / "runs"
    exit_code = main(["ui", "--runs-dir", str(runs_dir), "--seed", "5"])
    over_limit_exit = main(["ui", "--runs-dir", str(runs_dir), "--candidates", "9"])
    errors = capsys.readouterr().err

    # The valid command launches once with a controller for the runs folder and the default of 2 candidates.
    assert exit_code == 0 and len(launches) == 1, f"Expected one launch and exit code 0, got {exit_code} and {launches}"
    controller, options = launches[0]
    assert isinstance(controller, AppController) and controller.runs_dir == runs_dir, f"Bad controller {controller}"
    assert options["default_candidates"] == 2 and options["default_seed"] == 5, f"Unexpected launch options {options}"

    # Telemetry is off before the UI module runs.
    telemetry = (options["GRADIO_ANALYTICS_ENABLED"], options["HF_HUB_DISABLE_TELEMETRY"])
    assert telemetry == ("False", "1"), f"Telemetry must be disabled at launch, got {telemetry}"

    # Asking for more candidates than the fake allows fails with the limit and does not launch.
    limit_message = "--candidates 9 is above the backend limit of 8. Use 8 or fewer."
    assert over_limit_exit == 1 and limit_message in errors, f"Expected the limit error, got {over_limit_exit} and {errors!r}"
    assert not runs_dir.exists(), "Launching the UI must not create a session folder before the first round"


def test_cli_rejects_bad_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A file with a JPEG signature but corrupt data exits with code 1 and a single-line error, without a traceback.
    """
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff" + b"not an image")

    exit_code = main(["run", str(corrupt), "--prompt", "modern chair", "--runs-dir", str(tmp_path / "runs")])
    errors = capsys.readouterr().err

    # The corrupt image fails with one concise error line and leaves no run folder.
    assert exit_code == 1, "An invalid sketch must exit with code 1"
    assert errors.count("\n") == 1 and "cannot be decoded" in errors, f"Expected one concise error line, got {errors!r}"
    assert not (tmp_path / "runs").exists(), "No run folder may be created for an invalid sketch"
