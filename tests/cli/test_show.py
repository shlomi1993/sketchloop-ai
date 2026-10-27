import json
import pytest

from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import load_sketch_file
from sketchloop.cli import main
from sketchloop.domain import Guidance, SketchSource
from sketchloop.experiments import FolderStore
from sketchloop.session import SketchSession


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


def run_webcam_round(runs_dir: Path) -> Path:
    """
    Save one fake webcam round with a capture time, a 0.42 s frame age, and a 1.25 s generator setup, and return its folder.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(runs_dir), generator_setup_seconds=1.25)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.WEBCAM, capture_seconds=0.1,
                      frame_age_seconds=0.42)
    return session.folder


def show_text(folder: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """
    Run sketchloop show on a session folder and return its output with line wrapping removed.
    """
    exit_code = main(["show", str(folder)])
    assert exit_code == 0, f"show must succeed, got exit code {exit_code}"
    return " ".join(capsys.readouterr().out.split())


def test_show_prints_timings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    show prints the generator setup, the hardware, each round's total, and the frame age of a webcam round.
    """
    output = show_text(run_webcam_round(tmp_path / "runs"), capsys)

    # The session header and the round name every new measurement.
    assert "Generator setup: 1.25 s" in output, "show must print the generator setup time"
    assert "Hardware: " in output, "show must print the hardware"
    assert "Timings: capture 0.10 s" in output and "total " in output, "show must print the capture and the total"
    assert "(source: webcam), frame age 0.42 s" in output, "show must print the webcam frame age"
    assert "Recorded by an older sketchloop" not in output, "A current round must not get the older-timings note"


def test_show_older_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A session saved by an older sketchloop shows its hardware and setup as unavailable and notes that its timings are incomplete.
    """
    folder = run_webcam_round(tmp_path / "runs")
    session_file, round_file = folder / "session.json", folder / "round-1" / "round.json"
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    round_data = json.loads(round_file.read_text(encoding="utf-8"))

    # Delete the timing and hardware keys, standing in for older files.
    del session_data["environment"]["hardware"], session_data["generator_setup_seconds"], round_data["total_seconds"]
    session_file.write_text(json.dumps(session_data), encoding="utf-8")
    round_file.write_text(json.dumps(round_data), encoding="utf-8")
    output = show_text(folder, capsys)

    # Each missing value is named with its reason, and the round gets the older-timings note.
    assert "Hardware: unavailable (recorded before hardware was tracked)" in output, "Missing hardware must be named"
    assert "Generator setup: unavailable (recorded before generator setup was timed)" in output, "Missing setup must be named"
    assert "Recorded by an older sketchloop: generation may include the model load" in output, "An older round must get the note"

