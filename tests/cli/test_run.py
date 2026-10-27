import errno
import json
import pytest
import re
import shutil

from collections.abc import Mapping
from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CapturedSketch, CaptureSource, FileCapture
from sketchloop.cli import main
from sketchloop.components import CAPTURE_SOURCES, EVALUATORS, STORES
from sketchloop.domain import SketchSource
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import FolderStore, RoundRecord, SessionRecord, load_session
from sketchloop.settings import StorageSettings


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


def test_cli_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A run stores sketches and candidates, labels the fake backend, records the pick, and saves its timings.
    """
    sketch = tmp_path / "sketch-photo.jpg"
    shutil.copyfile(EXAMPLE_SKETCH, sketch)
    answers = iter(["²", "1 01", "1", "", "", "q"])
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

    # Round 1 saves its total, and the repeated round 2 reuses the sketch without a capture.
    round1 = json.loads((round_dir / "round.json").read_text(encoding="utf-8"))
    round2 = json.loads((session_dirs[0] / "round-2" / "round.json").read_text(encoding="utf-8"))
    assert isinstance(round1["total_seconds"], float), f"round.json must hold a total, got {round1.get('total_seconds')}"
    assert "capture" in round1["timings_seconds"] and "capture" not in round2["timings_seconds"], "Only round 1 captured"


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
        monkeypatch.setattr("sketchloop.experiments.stores.os.replace", raise_disk_full)

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

    def capture(self) -> CapturedSketch:
        """
        Count the capture and return the example photo.
        """
        self.captures += 1
        return FileCapture(EXAMPLE_SKETCH).capture()


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
