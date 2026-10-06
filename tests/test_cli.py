import pytest
import shutil

from pathlib import Path

from sketchloop.cli import main

EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch.png"


def test_cli_run_writes_files_and_records_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A run on the example sketch stores the sketch and candidates, labels the backend fake, and records the pick.
    """
    sketch = tmp_path / "sketch.png"
    shutil.copyfile(EXAMPLE_SKETCH, sketch)
    monkeypatch.setattr("builtins.input", lambda *prompt: "1")
    exit_code = main([str(sketch), "--prompt", "modern chair", "--runs-dir", str(tmp_path / "runs")])
    output = capsys.readouterr().out
    run_dirs = list((tmp_path / "runs").iterdir())
    assert exit_code == 0, "A valid run must exit with code 0"
    assert "FAKE BACKEND" in output, "The output must say the backend is fake"
    assert len(run_dirs) == 1 and (run_dirs[0] / "sketch.png").read_bytes() == sketch.read_bytes(), "Sketch must be copied"
    assert len(list((run_dirs[0] / "candidates").glob("*.png"))) == 4, "Four candidate PNGs must be written by default"
    assert "Selected: 1" in output, "The summary must show the selected candidate"


def test_cli_rejects_non_png_with_one_line_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A file that is not a PNG exits with code 1 and a single-line error, without a traceback.
    """
    not_png = tmp_path / "notes.png"
    not_png.write_text("not an image")
    exit_code = main([str(not_png), "--prompt", "modern chair", "--runs-dir", str(tmp_path / "runs")])
    errors = capsys.readouterr().err
    assert exit_code == 1, "An invalid sketch must exit with code 1"
    assert errors.count("\n") == 1 and "not a PNG" in errors, f"Expected one concise error line, got {errors!r}"
    assert not (tmp_path / "runs").exists(), "No run folder may be created for an invalid sketch"
