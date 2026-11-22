"""
Tests that saved sessions load back unchanged and that damaged records are refused.
"""

import json
import pytest

from collections.abc import Callable
from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CapturedSketch, load_sketch_file
from sketchloop.domain import (Guidance, SketchSource, Unavailable, UnsupportedConfigurationError, record_no_selection,
                               select_candidates)
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import (MISSING_FRAME_AGE_REASON, MISSING_HARDWARE_REASON, MISSING_SETUP_REASON,
                                    MISSING_TOTAL_REASON, ORIGIN_NOT_RECORDED, BenchmarkPlan, ExperimentRecordError, FolderStore,
                                    Hardware, PromptSuggestionRecord, load_session)
from sketchloop.session import SketchSession
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"
PromptMaker = Callable[..., PromptSuggestionRecord]  # Builds a shown prompt suggestion from its text and ID


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
    ("round-1/round.json", lambda data: data["input"]["raw_sketch"].update(path="../x.png"), "outside its folder"),
    ("session.json", lambda data: data.update(schema_version=2), "schema version 2"),
    ("round-2/round.json", lambda data: data.update(parent_id="0" * 32), "does not link to round-1"),
    ("round-1/round.json", lambda data: data.update(started_at="2026-10-09T11:25:02"), "timezone-aware"),
    ("round-1/round.json", lambda data: data["evaluation"].update(scores={"x": float("nan")}), "finite"),
    ("round-1/round.json", lambda data: data["input"].update(source="scanner"), "source must be file, camera, webcam, or upload"),
    ("round-1/round.json", lambda data: data["timings_seconds"].update(generation=-1.0), "finite seconds"),
    ("round-1/round.json", lambda data: data.update(total_seconds=float("inf")), "finite seconds"),
    ("session.json", lambda data: data.update(generator_setup_seconds=-0.5), "finite seconds"),
    ("round-1/round.json", lambda data: data["input"].update(frame_age_seconds=0.3), "Only a webcam round has a frame age"),
    ("session.json", lambda data: data.update(benchmark={"warmup_rounds": -1, "measured_rounds": 5}), "Benchmark needs"),
    ("session.json", lambda data: data["environment"].update(hardware={"cpu": "", "logical_cores": 0, "memory_bytes": 1}),
     "Hardware cpu must be non-empty text")
])
def test_invalid_records_rejected(tmp_path: Path, file_name: str, edit: Callable[[dict], None], message: str) -> None:
    """
    Traversal, a bad schema, lineage, time, score, source, duration, setup, frame age, benchmark plan, or hardware refuse to load.
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
    Totals, frame age, hardware, generator setup, and the benchmark plan load back unchanged, and older files load as unavailable.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path), benchmark=BenchmarkPlan(warmup_rounds=1, measured_rounds=2),
                            generator_setup_seconds=2.5)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.WEBCAM, capture_seconds=0.1,
                      frame_age_seconds=0.3)
    saved = load_session(session.folder)
    record, expected = saved.rounds[0], session.records[0]

    # The new round and session fields load back as recorded.
    loaded_round = (record.timings_seconds, record.total_seconds, record.frame_age_seconds)
    assert loaded_round == (expected.timings_seconds, expected.total_seconds, 0.3), f"Round fields changed: {loaded_round}"
    recorded_session = session.session_record
    loaded_session = (saved.session.environment.hardware, saved.session.generator_setup_seconds, saved.session.benchmark)
    expected_session = (recorded_session.environment.hardware, 2.5, BenchmarkPlan(warmup_rounds=1, measured_rounds=2))
    assert loaded_session == expected_session, f"Session fields changed: {loaded_session}"

    # Delete the new keys to stand in for files written before they were tracked.
    session_file, round_file = session.folder / "session.json", session.folder / "round-1" / "round.json"
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    round_data = json.loads(round_file.read_text(encoding="utf-8"))
    removals = [(session_data["environment"], "hardware"), (session_data, "generator_setup_seconds"), (session_data, "benchmark"),
                (round_data, "total_seconds"), (round_data["input"], "frame_age_seconds")]
    for data, key in removals:
        del data[key]
    session_file.write_text(json.dumps(session_data), encoding="utf-8")
    round_file.write_text(json.dumps(round_data), encoding="utf-8")
    older = load_session(session.folder)

    # Each missing value loads as unavailable with its reason, and a missing benchmark plan as none.
    older_session, older_round = older.session, older.rounds[0]
    assert older_session.environment.hardware == Unavailable(MISSING_HARDWARE_REASON), "Missing hardware must be unavailable"
    assert older_session.generator_setup_seconds == Unavailable(MISSING_SETUP_REASON), "Missing setup must be unavailable"
    assert older_session.benchmark is None, f"A missing benchmark plan must load as none, got {older_session.benchmark}"
    assert older_round.total_seconds == Unavailable(MISSING_TOTAL_REASON), "A missing total must be unavailable"
    assert older_round.frame_age_seconds == Unavailable(MISSING_FRAME_AGE_REASON), "A missing frame age must be unavailable"


def test_score_details_round_trip(tmp_path: Path, constant_evaluator: Evaluator) -> None:
    """
    The evaluator's description of its scores is saved with each round, and a round saved before it loads with none.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path), constant_evaluator)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE, n_candidates=2, seed=7)
    saved = load_session(session.folder).rounds[0]
    expected = dict(constant_evaluator.describe_scores())
    assert dict(saved.score_details) == expected, f"Expected score details {expected}, got {saved.score_details}"

    # Delete the key to stand in for a round written before evaluators described their scores.
    round_file = session.folder / "round-1" / "round.json"
    round_data = json.loads(round_file.read_text(encoding="utf-8"))
    del round_data["evaluation"]["score_details"]
    round_file.write_text(json.dumps(round_data), encoding="utf-8")
    older = load_session(session.folder).rounds[0]
    assert dict(older.score_details) == {} and older.scores, f"An older round must load with no details, got {older}"


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


def test_session_without_drawings_or_name_loads(tmp_path: Path) -> None:
    """
    A session saved before T09, without drawing folders or a name in session.json, loads with no drawings and no name.
    """
    session = run_two_rounds(tmp_path)
    session_file = session.folder / "session.json"
    data = json.loads(session_file.read_text(encoding="utf-8"))
    del data["name"]
    session_file.write_text(json.dumps(data), encoding="utf-8")
    saved = load_session(session.folder)

    # The older session has no drawings and no experiment name.
    assert saved.drawings == () and saved.session.name is None, f"Expected no drawings or name, got {saved}"


def run_named_drawing(runs_dir: Path, captured: CapturedSketch) -> SketchSession:
    """
    Start a named session, suggest once with the fake, take an interval capture, and finish the drawing.
    """
    session = SketchSession(FakeGenerator(), FolderStore(runs_dir), name="p1_car")
    drawing = session.start_drawing(FakeStrokeSuggester(), canvas_width=512, canvas_height=360, save_interval_seconds=2.0)
    drawing.suggest(captured, source=SketchSource.UPLOAD, rotation=0, prompt="a car")
    drawing.record_capture(captured, source=SketchSource.UPLOAD, rotation=0)
    drawing.finish(captured, source=SketchSource.UPLOAD)
    return session


def test_drawing_loads_and_reports_changed_mask(tmp_path: Path, sheet_capture: CapturedSketch) -> None:
    """
    A saved drawing loads equal to the record, and a changed mask or a drawing folder without drawing.json is a problem.
    """
    session = run_named_drawing(tmp_path, sheet_capture)
    record = session.drawings[0].record
    saved = load_session(session.folder)
    (session.folder / "drawing-1" / "suggestions" / "suggestion-1" / "part-1.png").write_bytes(b"changed")
    (session.folder / "drawing-1" / "captures" / "capture-0002-raw.jpg").unlink()
    (session.folder / "drawing-2").mkdir()
    damaged = load_session(session.folder)

    # The drawing loads unchanged with every file matching.
    assert saved.drawings == (record,) and saved.problems == (), f"Expected the drawing unchanged, got {saved.problems}"
    assert saved.checked_files == len(record.captures) * 3 + 5, f"Unexpected checked file count {saved.checked_files}"

    # A missing interval frame, a changed mask, and an interrupted drawing are listed as problems, not errors.
    expected = ("drawing-1/captures/capture-0002-raw.jpg is missing.",
                "drawing-1/suggestions/suggestion-1/part-1.png does not match its recorded checksum.",
                "drawing-2 has no drawing.json, so the drawing was interrupted.")
    assert damaged.problems == expected, f"Unexpected problems {damaged.problems}"


@pytest.mark.parametrize(("edit", "message"), [
    (lambda data: data["captures"][0]["raw"].update(path="../x.png"), "outside its folder"),
    (lambda data: data["captures"][1].update(number=3), "Capture numbers must run from 1 without gaps"),
    (lambda data: data["suggestions"][0].update(baseline_capture=2), "baseline must be a suggestion capture"),
    (lambda data: data["suggestions"][0]["request"].update(max_parts=1), "more than the 1 requested"),
    (lambda data: data.update(final_capture=1), "must have trigger finish"),
    (lambda data: data.update(end=None), "end time exactly when it has an end"),
    (lambda data: data["captures"][1]["follow"].update(suggestion=2), "not the one open when it was taken"),
    (lambda data: data["captures"][0].update(follow=data["captures"][1]["follow"]), "not the one open when it was taken"),
    (lambda data: data.update(follow_detector=None), "follow scores but no follow method"),
    (lambda data: data["captures"][1]["follow"]["alignment"].update(warp=None), "3x3 warp"),
    (lambda data: data["captures"][1]["follow"]["parts"].pop(), "parts, not suggestion 1's"),
    (lambda data: data["captures"][1]["follow"]["parts"][0].update(new_pixels_near=5), "precision exactly when new strokes"),
    (lambda data: data["captures"][1]["filled"].update(width=64), "filled mask must be L, canvas-sized"),
    (lambda data: data.update(interval_jpeg_quality=0), "interval_jpeg_quality must be an int from 1 to 100"),
])
def test_invalid_drawing_rejected(tmp_path: Path, sheet_capture: CapturedSketch, edit: Callable[[dict], None],
                                  message: str) -> None:
    """
    A drawing.json with an outside path, a capture gap, a wrong baseline, too many parts, or a bad end or alignment fails.
    """
    session = run_named_drawing(tmp_path, sheet_capture)
    drawing_file = session.folder / "drawing-1" / "drawing.json"
    data = json.loads(drawing_file.read_text(encoding="utf-8"))
    edit(data)
    drawing_file.write_text(json.dumps(data), encoding="utf-8")

    # Loading names the file and the broken rule.
    with pytest.raises(ExperimentRecordError, match=message) as caught:
        load_session(session.folder)
    assert str(caught.value).startswith("drawing-1/drawing.json"), f"The error must name the file, got {caught.value}"


def run_prompted_rounds(runs_dir: Path, make_prompt_suggestion: PromptMaker) -> SketchSession:
    # Show a prompt suggestion outside a drawing, use it unchanged in round 1, and use it edited in round 2.
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(runs_dir))
    session.record_prompt_suggestion(make_prompt_suggestion("a chair, concept art", "prompt-a"))
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="a chair, concept art"), source=SketchSource.FILE,
                      prompt_suggestion_id="prompt-a")
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="a chair, ink"), source=SketchSource.FILE,
                      prompt_suggestion_id="prompt-a")
    return session


def test_round_prompt_fields_round_trip_and_older_rounds_load(tmp_path: Path, make_prompt_suggestion: PromptMaker) -> None:
    session = run_prompted_rounds(tmp_path, make_prompt_suggestion)
    saved = load_session(session.folder)
    round_file = session.folder / "round-1" / "round.json"
    data = json.loads(round_file.read_text(encoding="utf-8"))

    # Both rounds load with their prompt suggestions and origins, the second naming the first round's suggestion.
    loaded = [(record.prompt_suggestions, record.prompt_origin) for record in saved.rounds]
    expected = [(record.prompt_suggestions, record.prompt_origin) for record in session.records]
    assert loaded == expected and saved.problems == (), f"Prompt fields changed on reload: {loaded}, {saved.problems}"
    assert saved.rounds[1].prompt_origin.kind == "edited", f"Round 2 must be edited, got {saved.rounds[1].prompt_origin}"

    # The new keys follow the evaluation in round.json.
    keys = list(data)
    position = keys.index("evaluation")
    assert keys[position + 1:position + 3] == ["prompt_suggestions", "prompt_origin"], f"Unexpected key order {keys}"

    # A round saved before prompt records loads with no list and an origin not recorded.
    del data["prompt_suggestions"], data["prompt_origin"]
    round_file.write_text(json.dumps(data), encoding="utf-8")
    older = load_session(session.folder).rounds[0]
    assert older.prompt_suggestions is None, f"An older round has no prompt suggestion list, got {older.prompt_suggestions}"
    assert older.prompt_origin == Unavailable(ORIGIN_NOT_RECORDED), f"Expected not recorded, got {older.prompt_origin}"


def test_round_origin_naming_an_unknown_suggestion_is_a_problem(tmp_path: Path, make_prompt_suggestion: PromptMaker) -> None:
    session = run_prompted_rounds(tmp_path, make_prompt_suggestion)
    round_file = session.folder / "round-2" / "round.json"
    data = json.loads(round_file.read_text(encoding="utf-8"))
    data["prompt_origin"]["prompt_suggestion"] = "unknown"
    round_file.write_text(json.dumps(data), encoding="utf-8")
    saved = load_session(session.folder)

    # The session still loads, and the unknown ID is one problem line.
    expected = ("round-2 prompt origin names prompt suggestion unknown, which is not in this session.",)
    assert saved.problems == expected, f"Expected one problem, got {saved.problems}"


def test_round_origin_disagreeing_with_its_own_suggestion_is_refused(tmp_path: Path, make_prompt_suggestion: PromptMaker) -> None:
    session = run_prompted_rounds(tmp_path, make_prompt_suggestion)
    round_file = session.folder / "round-1" / "round.json"
    data = json.loads(round_file.read_text(encoding="utf-8"))
    data["prompt_origin"]["kind"] = "edited"
    round_file.write_text(json.dumps(data), encoding="utf-8")

    # The round's own suggestion gives suggested, so the recorded kind is refused.
    with pytest.raises(ExperimentRecordError, match="recorded as edited, but its prompt and suggestion give suggested"):
        load_session(session.folder)
