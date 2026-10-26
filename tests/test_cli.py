import errno
import os
import pytest
import re
import shutil
import sys
import types

from collections.abc import Mapping
from pathlib import Path

import sketchloop
from sketchloop.app import AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CaptureSource, load_sketch_file
from sketchloop.cli import main
from sketchloop.components import CAPTURE_SOURCES, EVALUATORS, STORES
from sketchloop.domain import ImageRef, SketchSource
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import FolderStore, RoundRecord, SessionRecord, load_session
from sketchloop.settings import StorageSettings


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")

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

    # The run succeeds without a settings file, labels the fake backend, and keeps the raw sketch byte for byte.
    assert exit_code == 0, "A valid run must exit with code 0"
    assert "Settings:" not in output, "Without a settings file, no settings file may be announced"
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


def raise_disk_full(source: object, target: object) -> None:
    """
    Stand in for os.replace on a full disk.
    """
    raise OSError(errno.ENOSPC, "No space left on device")


# The fake allows at most 8 candidates, so 9 fails in generation, and a full disk fails saving session.json.
@pytest.mark.parametrize(("candidates", "disk_full", "failure_pattern"), [
    ("9", False, r"Round failed: Unsupported request.*"),
    ("2", True, r"Round failed: Cannot save [\w-]+/session\.json: No space left on device\. Free disk space or check write access to the runs folder, then retry\.")  # noqa: E501
], ids=["candidate-limit", "full-disk"])
def test_cli_session_survives_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                     candidates: str, disk_full: bool, failure_pattern: str) -> None:
    """
    A failed round or a failed save and a missing new file are reported, and the session continues to the menu and summary.
    """
    answers = iter(["f missing.png", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    if disk_full:
        monkeypatch.setattr("sketchloop.experiments.os.replace", raise_disk_full)

    runs_dir = str(tmp_path / "runs")
    exit_code = main(["run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", candidates, "--runs-dir", runs_dir])
    output = capsys.readouterr().out
    failure_lines = [line for line in output.splitlines() if line.startswith("Round failed:")]

    # The failure is one line naming the cause and recovery, with no absolute path that could name the user or leftover file.
    assert len(failure_lines) == 1, f"Expected one failure line, got {failure_lines}"
    assert re.fullmatch(failure_pattern, failure_lines[0]), f"Unexpected failure line {failure_lines[0]!r}"
    assert str(tmp_path) not in failure_lines[0] and tmp_path.as_posix() not in failure_lines[0], "The message shows a path"
    leftovers = list((tmp_path / "runs").rglob("*.tmp"))
    assert leftovers == [], f"A failed save must remove its temporary file, found {leftovers}"

    # The missing file is shown too, and the session still ends normally with its summary.
    assert exit_code == 0, f"Quitting after errors must exit with code 0, got {exit_code}"
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
@pytest.mark.parametrize(("command", "option", "settings_text", "error_pattern"), [
    ("run", ["--candidates", "0"], None, r"must be at least 1, got 0"),
    ("run", ["--seed", "-1"], None, r"must be at least 0, got -1"),
    ("ui", ["--candidates", "0"], None, r"must be at least 1, got 0"),
    ("ui", ["--mode", "fast"], None, r"--mode needs --backend diffusers"),
    ("run", [], '[generator]\nbackend = "difusers"\n', r'sketchloop\.toml: generator\.backend must be one of .+"difusers"\.'),
    ("ui", [], "[ui]\ncandidates = true\n", r"sketchloop\.toml: ui\.candidates must be an integer, got (?i:true)\."),
    ("run", [], '[generator]\nmdoe = "fast"\n', r"sketchloop\.toml: unknown key generator\.mdoe\. Keys in \[generator\]: "),
    ("run", [], '[generator\nbackend = "fake"\n', r"sketchloop\.toml is not valid TOML: .+\. Fix the file and retry\."),
    ("run", [], "[preprocessing]\nrotation = 90\nraw = true\n",
     re.escape("preprocessing.rotation 90 cannot be combined with preprocessing.raw, which skips preprocessing. Change one in sketchloop.toml or pass --rotate 0 or --no-raw.")),  # noqa: E501
    ("run", [], '[storage]\nstore = "memory"\n', re.escape('storage.store must be one of folder, got "memory".')),
    ("rerun", ["--config", "missing.toml"], None, re.escape("Settings file missing.toml not found. Check the --config path.")),
    ("ui", ["--capture-interval", "inf"], None, r"must be greater than 0 seconds, got inf"),
    ("ui", [], "[ui]\ncapture_interval_seconds = inf\n", r"capture_interval_seconds must be greater than 0, got inf\."),
    ("run", [], "seed = 7\n", re.escape("sketchloop.toml: seed must be inside a section such as [run]."))
], ids=["candidates-0", "negative-seed", "ui-candidates-0", "mode-without-diffusers", "misspelled-backend", "boolean-candidates",
        "unknown-key", "broken-toml", "rotation-with-raw", "memory-store", "missing-config", "infinite-interval-flag",
        "infinite-interval-file", "top-level-key"])
def test_cli_rejects_invalid_options(tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, option: list[str],
                                     settings_text: str | None, error_pattern: str) -> None:
    """
    Bad flags, a bad settings file, or a missing --config file are rejected with a one-line reason before anything is saved.
    """
    runs_option = ["--runs-dir", str(tmp_path / "runs")]
    base = {
        "run": ["run", str(EXAMPLE_SKETCH), "--prompt", "chair", *runs_option],
        "ui": ["ui", *runs_option],
        "rerun": ["rerun", str(tmp_path / "runs" / "earlier"), "--round", "1", *runs_option],
    }[command]

    # A settings file in the working folder is found without --config.
    if settings_text is not None:
        (tmp_path / "sketchloop.toml").write_text(settings_text, encoding="utf-8")

    # Argparse exits with its usage error code.
    with pytest.raises(SystemExit) as raised:
        main([*base, *option])
    errors = capsys.readouterr().err

    # The error names the reason, and nothing is saved for the rejected command.
    assert raised.value.code == 2, f"Expected the usage error code 2, got {raised.value.code}"
    assert re.search(error_pattern, errors), f"Expected an error matching {error_pattern!r}, got {errors!r}"
    assert not (tmp_path / "runs").exists(), "No session folder may be created for invalid options"


class ExampleCamera(CaptureSource):
    """
    Capture source standing in for a webcam, returning the example photo and counting captures.
    """

    def __init__(self) -> None:
        self.captures = 0

    @property
    def kind(self) -> SketchSource:
        """
        Record sketches from this source as camera input.
        """
        return SketchSource.CAMERA

    def capture(self) -> tuple[ImageRef, bytes]:
        """
        Count the capture and return the example photo.
        """
        self.captures += 1
        return load_sketch_file(EXAMPLE_SKETCH)


class CountingStore(FolderStore):
    """
    Folder store that lists every save it is asked to make.
    """

    def __init__(self, runs_dir: Path) -> None:
        super().__init__(runs_dir)
        self.saves: list[str] = []

    def save_session(self, record: SessionRecord) -> None:
        """
        List the session save, then save it in the folder.
        """
        self.saves.append("session")
        super().save_session(record)

    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        List the round save, then save it in the folder.
        """
        self.saves.append(f"round-{record.number}")
        super().save_round(session_id, record, files)


def test_cli_rounds_track_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                 constant_evaluator: Evaluator) -> None:
    """
    A settings file picks the camera, evaluator, store, and runs folder, and flags still win, across camera and file rounds.
    """
    settings_text = "\n".join(['[capture]', 'source = "camera"', '[evaluator]', 'name = "constant"', '[storage]',
                               'store = "counting"', 'runs_dir = "file-runs"', '[run]', 'candidates = 3', 'seed = 4', ''])
    (tmp_path / "sketchloop.toml").write_text(settings_text, encoding="utf-8")
    camera = ExampleCamera()
    stores: list[CountingStore] = []

    def make_counting_store(settings: StorageSettings) -> CountingStore:
        """
        Create a counting folder store for the configured runs folder and remember it.
        """
        stores.append(CountingStore(settings.runs_dir))
        return stores[-1]

    # Register the substitutes by the names the settings file uses.
    monkeypatch.setitem(CAPTURE_SOURCES, "camera", lambda settings, sketch_file: camera)
    monkeypatch.setitem(EVALUATORS, "constant", lambda settings: constant_evaluator)
    monkeypatch.setitem(STORES, "counting", make_counting_store)

    # Pick 1, recapture, pick none, load a file, pick none, and quit.
    answers = iter(["1", "c", "", f"f {EXAMPLE_SKETCH}", "", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    created: list[FakeGenerator] = []

    def create_counted_fake() -> FakeGenerator:
        """
        Create a fake generator and remember it, so the test can count backend creations.
        """
        created.append(FakeGenerator())
        return created[-1]

    monkeypatch.setattr("sketchloop.backends.FakeGenerator", create_counted_fake)

    exit_code = main(["run", "--prompt", "chair", "--candidates", "2"])
    output = capsys.readouterr().out
    session_dirs = list((tmp_path / "file-runs").iterdir())
    rounds = load_session(session_dirs[0]).rounds
    sources = [record.source for record in rounds]

    # The session ends cleanly with three rounds and a single backend instance, after naming the settings file it found.
    assert exit_code == 0, "A session ended with q must exit with code 0"
    assert output.startswith("Settings: sketchloop.toml"), f"The settings file must be named first, got {output[:80]!r}"
    assert len(session_dirs) == 1, f"The session must be saved in the file's runs folder, got {session_dirs}"
    assert len(created) == 1, f"The backend must be created once per session, got {len(created)}"

    # A recapture goes through the configured camera again, and a new file switches to file, even with the same bytes.
    assert sources == ["camera", "camera", "file"], f"Expected sources camera, camera, file, got {sources}"
    assert camera.captures == 2, f"The first capture and the recapture must use the configured camera, got {camera.captures}"

    # The --candidates flag beats the file's 3, and the file's seed 4 beats the random default.
    counts = [len(record.iteration.result.candidates) for record in rounds]
    first_seed = rounds[0].iteration.result.candidates[0].seed
    assert counts == [2, 2, 2] and first_seed == 4, f"Expected 2 candidates per round and seed 4, got {counts} and {first_seed}"

    # The configured evaluator scores every round, and the configured store makes every save.
    scored = all(record.evaluator == "ConstantEvaluator" and list(record.scores.values()) == [0.5, 0.5] for record in rounds)
    expected_saves = ["session", "round-1", "round-1", "round-2", "round-2", "round-3", "round-3"]
    assert scored, f"Every round must record the constant evaluator's scores, got {[record.scores for record in rounds]}"
    assert len(stores) == 1 and stores[0].saves == expected_saves, f"Unexpected store saves {[store.saves for store in stores]}"

    # The summary lists every round with its selection.
    selections_shown = all(line in output for line in ("Round 1 selected: 1", "Round 2 selected: none", "Round 3 selected: none"))
    assert "Rounds: 3" in output and selections_shown, "The summary must list all three rounds and their selections"


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
    assert options["default_candidates"] == 2 and options["default_seed"] == 5, f"Unexpected launch options {options}"

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
