import pytest
import shutil

from pathlib import Path

from sketchloop.cli import main


EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch-photo.jpg"


def test_cli_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A run on the example photo stores raw and processed sketches and candidates, labels the fake backend, and records the pick.
    """
    sketch = tmp_path / "sketch-photo.jpg"
    shutil.copyfile(EXAMPLE_SKETCH, sketch)
    monkeypatch.setattr("builtins.input", lambda *prompt: "1")

    exit_code = main([str(sketch), "--prompt", "modern chair", "--runs-dir", str(tmp_path / "runs")])
    output = capsys.readouterr().out
    run_dirs = list((tmp_path / "runs").iterdir())

    assert exit_code == 0, "A valid run must exit with code 0"
    assert "FAKE BACKEND" in output, "The output must say the backend is fake"
    assert len(run_dirs) == 1 and (run_dirs[0] / "sketch-raw.jpg").read_bytes() == sketch.read_bytes(), "Raw sketch not kept"

    steps_shown = "Preprocessing: grayscale, rotate, isolate_paper, extract_strokes, crop_to_drawing" in output
    processed_saved_and_shown = (run_dirs[0] / "sketch.png").is_file() and steps_shown
    assert processed_saved_and_shown, "The processed sketch must be saved and its steps shown"
    assert len(list((run_dirs[0] / "candidates").glob("*.png"))) == 4, "Four candidate PNGs must be written by default"
    assert "Selected: 1" in output, "The summary must show the selected candidate"


def test_cli_rejects_bad_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A file with a JPEG signature but corrupt data exits with code 1 and a single-line error, without a traceback.
    """
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff" + b"not an image")

    exit_code = main([str(corrupt), "--prompt", "modern chair", "--runs-dir", str(tmp_path / "runs")])
    errors = capsys.readouterr().err

    assert exit_code == 1, "An invalid sketch must exit with code 1"
    assert errors.count("\n") == 1 and "cannot be decoded" in errors, f"Expected one concise error line, got {errors!r}"
    assert not (tmp_path / "runs").exists(), "No run folder may be created for an invalid sketch"
