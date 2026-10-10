from pathlib import Path

from sketchloop.capture import load_sketch_file
from sketchloop.domain import Guidance, record_no_selection
from sketchloop.fakes import FakeGenerator
from sketchloop.session import SketchSession


EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch-photo.jpg"


def test_session_links_rounds(tmp_path: Path) -> None:
    """
    Two rounds save into their own folders under one session folder, and round 2 links to round 1.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), tmp_path)
    round1 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), n_candidates=2)
    session.record_selection(record_no_selection(round1.iteration))
    round2 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="lamp"), use_raw=True)

    # Round 1 holds its sketches, candidates, and record, and the raw round 2 skips preprocessing.
    round1_files = sorted(path.relative_to(round1.folder).as_posix() for path in round1.folder.rglob("*.*"))
    expected_files = ["candidates/candidate-1.png", "candidates/candidate-2.png", "round.json", "sketch-raw.jpg", "sketch.png"]
    assert round1_files == expected_files, f"Round 1 must hold the sketches, candidates, and record, got {round1_files}"
    assert round2.folder == session.folder / "round-2", f"Round 2 must save under the session folder, got {round2.folder}"
    assert not (round2.folder / "sketch.png").exists() and round2.steps == (), "A raw round must skip preprocessing"

    # Round 2 links to round 1, and the no-op evaluator gives no scores.
    parents = [round1.iteration.parent_id, round2.iteration.parent_id]
    assert parents == [None, round1.iteration.id], f"Round 2 must link to round 1, got parents {parents}"
    assert round1.scores == {} and round2.scores == {}, "The default no-op evaluator must give no scores"
