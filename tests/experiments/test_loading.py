import json
import pytest

from collections.abc import Callable
from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import load_sketch_file
from sketchloop.domain import (Guidance, SketchSource, Unavailable, UnsupportedConfigurationError, record_no_selection,
                               select_candidates)
from sketchloop.experiments import (MISSING_HARDWARE_REASON, MISSING_TOTAL_REASON, ExperimentRecordError, FolderStore,
                                    Hardware, load_session)
from sketchloop.session import SketchSession


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


def run_two_rounds(runs_dir: Path) -> SketchSession:
    """
    Run two fake rounds, a file round picking candidate 1 and a raw camera round picking none.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(runs_dir))
    chair = Guidance(prompt="chair")
    round1 = session.run_round(raw_sketch, raw_payload, chair, source=SketchSource.FILE, n_candidates=2, seed=7)
    session.record_selection(select_candidates(round1.iteration, [round1.iteration.result.candidates[0].id]))
    lamp = Guidance(prompt="lamp", controls={"steps": 8})
    round2 = session.run_round(raw_sketch, raw_payload, lamp, source=SketchSource.CAMERA, use_raw=True)
    session.record_selection(record_no_selection(round2.iteration))
    return session


def test_records_round_trip(tmp_path: Path) -> None:
    """
    Two rounds and a failed third save their files and load back with equal iterations, steps, selections, sources, and checksums.
    """
    session = run_two_rounds(tmp_path)
    round1_folder = session.folder / "round-1"
    round1_files = sorted(path.relative_to(round1_folder).as_posix() for path in round1_folder.rglob("*.*"))
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="vase"), source=SketchSource.WEBCAM, n_candidates=99)
    (session.folder / "round-1" / "round.json.tmp").write_text("{", encoding="utf-8")

    saved = load_session(session.folder)
    round1, round2, round3 = saved.rounds
    outcomes = session.rounds

    # Round 1 holds its sketches, candidates, and record, and the raw round 2 writes no processed sketch.
    expected_files = ["candidates/candidate-1.png", "candidates/candidate-2.png", "round.json", "sketch-raw.jpg", "sketch.png"]
    assert round1_files == expected_files, f"Round 1 must hold the sketches, candidates, and record, got {round1_files}"
    assert not (session.folder / "round-2" / "sketch.png").exists(), "A raw round must skip preprocessing"

    # The saved records reload into the same iterations, preprocessing steps, and round-to-round lineage.
    iterations_equal = [round1.iteration, round2.iteration] == [outcome.iteration for outcome in outcomes]
    assert iterations_equal, "Loaded requests, results, and effective settings must equal the session's"
    assert round1.steps == outcomes[0].steps and round2.steps == (), "Preprocessing steps must load unchanged"
    assert round2.iteration.parent_id == round1.iteration.id, "Round 2 must link to round 1"

    # Selections, stage timings, the failed round, and every file checksum survive the save and load.
    selections = [round1.selection, round2.selection]
    assert selections == [session.selections[outcome.iteration.id] for outcome in outcomes], f"Selections changed: {selections}"
    assert set(round1.timings_seconds) == {"preprocessing", "generation", "evaluation", "saving"}, "Round 1 must time every stage"
    assert round3.status == "failed" and round3.failure.stage == "generation", f"Round 3 must fail in generation: {round3}"
    assert saved.problems == () and saved.checked_files == 8, f"All files must match: {saved.problems}"

    # Each round's sketch source, including the failed round's, loads back unchanged.
    sources = [record.source for record in saved.rounds]
    assert sources == ["file", "camera", "webcam"], f"Sources changed on reload: {sources}"

    # Package versions are recorded as real versions or as unavailable with a reason, never invented.
    versions = list(saved.session.environment.packages.values())
    assert all(isinstance(value, str | Unavailable) for value in versions), f"Unexpected versions {versions}"


@pytest.mark.parametrize(("damage", "problem"), [("flip", "does not match its recorded checksum"), ("delete", "is missing")])
def test_damaged_candidate_listed(tmp_path: Path, damage: str, problem: str) -> None:
    """
    A changed or deleted candidate file is listed as one problem instead of failing the load.
    """
    session = run_two_rounds(tmp_path)
    candidate = session.folder / "round-1" / "candidates" / "candidate-1.png"
    if damage == "flip":
        payload = bytearray(candidate.read_bytes())
        payload[-1] ^= 1
        candidate.write_bytes(payload)
    else:
        candidate.unlink()

    # The damaged file is reported as exactly one problem naming the file, while the session still loads.
    problems = load_session(session.folder).problems
    assert problems == (f"round-1/candidates/candidate-1.png {problem}.",), f"Expected one problem, got {problems}"


@pytest.mark.parametrize(("file_name", "edit", "message"), [
    ("round-1/round.json", lambda data: data["input"]["raw_sketch"].update(path="../x.png"), "outside its round folder"),
    ("session.json", lambda data: data.update(schema_version=2), "schema version 2"),
    ("round-2/round.json", lambda data: data.update(parent_id="0" * 32), "does not link to round-1"),
    ("round-1/round.json", lambda data: data.update(started_at="2026-10-09T11:25:02"), "timezone-aware"),
    ("round-1/round.json", lambda data: data["evaluation"].update(scores={"x": float("nan")}), "finite"),
    ("round-1/round.json", lambda data: data["input"].update(source="scanner"), "source must be file, camera, webcam, or upload"),
    ("round-1/round.json", lambda data: data["timings_seconds"].update(generation=-1.0), "finite seconds"),
    ("round-1/round.json", lambda data: data.update(total_seconds=float("inf")), "finite seconds"),
    ("session.json", lambda data: data["environment"].update(hardware={"cpu": "", "logical_cores": 0, "memory_bytes": 1}),
     "Hardware cpu must be non-empty text")
])
def test_invalid_records_rejected(tmp_path: Path, file_name: str, edit: Callable[[dict], None], message: str) -> None:
    """
    Traversal, a bad schema, lineage, time, score, source, duration, or hardware refuse to load.
    """
    session = run_two_rounds(tmp_path)
    record_file = session.folder / file_name
    data = json.loads(record_file.read_text(encoding="utf-8"))
    edit(data)
    record_file.write_text(json.dumps(data), encoding="utf-8")

    # Loading refuses the damaged record with a one-line error that names the reason, which show prints instead of a traceback.
    with pytest.raises(ExperimentRecordError, match=message) as raised:
        load_session(session.folder)
    assert "\n" not in str(raised.value), f"The error must be one line, got {raised.value!r}"


def test_timing_fields_round_trip(tmp_path: Path) -> None:
    """
    Totals and hardware load back unchanged, and older files load as unavailable.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path))
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.WEBCAM, capture_seconds=0.1)
    saved = load_session(session.folder)
    record, expected = saved.rounds[0], session.records[0]

    # The new round and session fields load back as recorded.
    loaded_round = (record.timings_seconds, record.total_seconds)
    assert loaded_round == (expected.timings_seconds, expected.total_seconds), f"Round fields changed: {loaded_round}"
    recorded_session = session.session_record
    loaded_hardware = saved.session.environment.hardware
    assert loaded_hardware == recorded_session.environment.hardware, f"Hardware changed: {loaded_hardware}"

    # Delete the new keys to stand in for files written before they were tracked.
    session_file, round_file = session.folder / "session.json", session.folder / "round-1" / "round.json"
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    round_data = json.loads(round_file.read_text(encoding="utf-8"))
    removals = [(session_data["environment"], "hardware"), (round_data, "total_seconds")]
    for data, key in removals:
        del data[key]
    session_file.write_text(json.dumps(session_data), encoding="utf-8")
    round_file.write_text(json.dumps(round_data), encoding="utf-8")
    older = load_session(session.folder)

    # Each missing value loads as unavailable with its reason.
    older_session, older_round = older.session, older.rounds[0]
    assert older_session.environment.hardware == Unavailable(MISSING_HARDWARE_REASON), "Missing hardware must be unavailable"
    assert older_round.total_seconds == Unavailable(MISSING_TOTAL_REASON), "A missing total must be unavailable"



def test_hardware_with_unavailable_fact_round_trips(tmp_path: Path) -> None:
    """
    Hardware with one unavailable fact keeps that fact's reason and the other values through session.json.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path))
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE)
    session_file = session.folder / "session.json"
    data = json.loads(session_file.read_text(encoding="utf-8"))

    # Write synthetic hardware whose memory could not be read.
    unread_memory = {"unavailable": "synthetic reason"}
    data["environment"]["hardware"] = {"cpu": "Synthetic CPU", "logical_cores": 4, "memory_bytes": unread_memory}
    session_file.write_text(json.dumps(data), encoding="utf-8")
    hardware = load_session(session.folder).session.environment.hardware

    # The readable facts load as values and the unreadable one as unavailable with its reason.
    expected = Hardware(cpu="Synthetic CPU", logical_cores=4, memory_bytes=Unavailable("synthetic reason"))
    assert hardware == expected, f"Expected {expected}, got {hardware}"
