import pytest
import shutil

from pathlib import Path

from sketchloop.cli import main
from sketchloop.fakes import FakeGenerator


EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch-photo.jpg"


def test_cli_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A run on the example photo stores raw and processed sketches and candidates, labels the fake backend, and records the pick.
    """
    sketch = tmp_path / "sketch-photo.jpg"
    shutil.copyfile(EXAMPLE_SKETCH, sketch)
    answers = iter(["1", "q"])
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


def test_cli_two_rounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Pick one, press Enter for a second round, pick none, and quit: both rounds are saved with one backend instance.
    """
    answers = iter(["1", "", "", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    created: list[FakeGenerator] = []

    def create_counted_fake() -> FakeGenerator:
        """
        Create a fake generator and remember it, so the test can count backend creations.
        """
        created.append(FakeGenerator())
        return created[-1]

    monkeypatch.setattr("sketchloop.cli.FakeGenerator", create_counted_fake)

    exit_code = main(["run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", "2", "--runs-dir", str(tmp_path / "runs")])
    output = capsys.readouterr().out
    session_dirs = list((tmp_path / "runs").iterdir())
    round_names = sorted(path.name for path in session_dirs[0].glob("round-*"))

    # The session ends cleanly with two round folders and a single backend instance.
    assert exit_code == 0, "A session ended with q must exit with code 0"
    assert round_names == ["round-1", "round-2"], f"Expected two round folders, got {round_names}"
    assert len(created) == 1, f"The backend must be created once per session, got {len(created)}"

    # The summary lists both rounds with their selections.
    summary_shown = "Rounds: 2" in output and "Round 1 selected: 1" in output and "Round 2 selected: none" in output
    assert summary_shown, "The summary must list both rounds and their selections"


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
