"""
Tests of what the show subcommand prints for current, older, and benchmark sessions.
"""

import json
import numpy as np
import pytest
import re

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CapturedSketch, encode_frame, load_sketch_file
from sketchloop.cli import main
from sketchloop.domain import Guidance, SketchSource
from sketchloop.drawing import GuidedDrawing
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import DrawingStudy, FolderStore, PromptSuggestionRecord, StudyCondition
from sketchloop.session import SketchSession
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggestionError, StrokeSuggestionOutput, StrokeSuggestionRequest


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

    # A time-named session without drawings prints no experiment or drawing.
    assert "Experiment:" not in output and "Drawing 1" not in output, f"Unexpected experiment or drawing in {output}"


def test_show_names_what_scored(tmp_path: Path, capsys: pytest.CaptureFixture[str], constant_evaluator: Evaluator) -> None:
    """
    show names the evaluator and what its scores mean when a round has scores, and stays silent without them.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path / "runs"), constant_evaluator)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE, n_candidates=2, seed=7)
    output = show_text(session.folder, capsys)
    assert "Scores: ConstantEvaluator, score=0.5 for every candidate, a test value" in output, f"Missing score line in {output}"

    # A round without scores prints no score line.
    unscored = show_text(run_webcam_round(tmp_path / "unscored"), capsys)
    assert "Scores:" not in unscored, "A round without scores must not print a score line"


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


def test_show_benchmark_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    show recomputes and prints the statistics of a benchmark session.
    """
    runs_dir = tmp_path / "runs"
    exit_code = main(["benchmark", str(EXAMPLE_SKETCH), "--prompt", "chair", "--rounds", "2", "--candidates", "1", "--runs-dir",
                      str(runs_dir)])
    capsys.readouterr()
    output = show_text(next(runs_dir.iterdir()), capsys)

    # The benchmark ran, and show prints the summary with its round roles and total row.
    assert exit_code == 0, f"The benchmark must succeed, got exit code {exit_code}"
    has_summary = "Benchmark summary" in output and "Warm-up rounds: 1 Measured rounds: 2, 3" in output
    assert has_summary, "show must summarize the benchmark"
    assert "│ total " in output, "The summary table must have a total row"



class OutOfMemorySuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester whose suggestion fails as if the device ran out of memory.
    """

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Raise the error a real suggester raises when generation fails.
        """
        raise StrokeSuggestionError("Next-stroke suggestion failed: out of memory. Free memory and retry.")


def capture_webcam_frame(frame: np.ndarray) -> CapturedSketch:
    """
    Encode a frame as a captured PNG raw sketch with a fixed capture time.
    """
    image, payload = encode_frame(frame)
    return CapturedSketch(image=image, payload=payload, capture_seconds=0.01)


def draw_named_experiment(runs_dir: Path, sheet_frame: np.ndarray, draw_part_on_sheet: Callable[[np.ndarray, bytes], np.ndarray]
                          ) -> tuple[SketchSession, GuidedDrawing, CapturedSketch]:
    """
    Save a named experiment with a round and an open drawing whose first part was drawn, then a failed press, and the frame.
    """
    session = SketchSession(FakeGenerator(), FolderStore(runs_dir), name="participant1_car")
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE)
    drawing = session.start_drawing(FakeStrokeSuggester(), canvas_width=512, canvas_height=360, save_interval_seconds=2.0)
    first = drawing.suggest(capture_webcam_frame(sheet_frame), source=SketchSource.WEBCAM, rotation=0, prompt="a [bold]car",
                            frame_age_seconds=0.2)

    # Draw the first part, then fail a press with an empty prompt, which keeps the first window open.
    ring = (session.folder / "drawing-1" / first.result.parts[0].mask.path).read_bytes()
    with_ring = capture_webcam_frame(draw_part_on_sheet(sheet_frame, ring))
    drawing.record_capture(with_ring, source=SketchSource.UPLOAD, rotation=0)
    drawing.suggester = OutOfMemorySuggester()
    with pytest.raises(StrokeSuggestionError):
        drawing.suggest(with_ring, source=SketchSource.UPLOAD, rotation=0, prompt=" ")
    return session, drawing, with_ring


def test_show_prints_a_finished_drawing(tmp_path: Path, capsys: pytest.CaptureFixture[str], sheet_frame: np.ndarray,
                                        draw_part_on_sheet: Callable[[np.ndarray, bytes], np.ndarray]) -> None:
    """
    show prints the experiment, the round as before, and each suggestion of the drawing with its recorded decision.
    """
    session, drawing, with_ring = draw_named_experiment(tmp_path / "runs", sheet_frame, draw_part_on_sheet)
    drawing.finish(with_ring, source=SketchSource.UPLOAD)
    output = show_text(session.folder, capsys)

    # The experiment heads the session, and the round is printed as before.
    assert "Experiment: participant1_car" in output, f"Missing experiment name in {output}"
    assert "Round 1 (complete)" in output and "Prompt: chair" in output, "The round must still be printed"

    # The drawing names its state, captures, final drawing, suggester, fake warning, and rule, with any elapsed time.
    drawing_line = r"Drawing 1: finished after \d+ s, 4 captures, canvas 512x360, final drawing captures/capture-0004\.png"
    assert re.search(drawing_line, output), f"Missing drawing line in {output}"
    assert "Suggester: FakeStrokeSuggester, sketchloop.fake-strokes" in output, f"Missing suggester in {output}"
    assert "follow method aligned-strokes 3" in output and "FAKE SUGGESTER" in output, f"Missing method or warning {output}"
    assert "Rule: coverage_16 >= 0.35 and precision_16 >= 0.4" in output, f"Missing rule in {output}"

    # Suggestion 1 keeps its typed prompt and its recorded decision, and the failed press names its stage.
    first_line = r'Suggestion 1 at \+\d\d:\d\d: "a \[bold\]car", origin typed, 2 parts, seed'
    assert re.search(first_line, output), f"Missing suggestion 1 in {output}"
    window_line = r"followed: part 1 followed, part 2 not followed\. Window \d+ s, 3 captures, ended by finished\."
    assert re.search(window_line, output), f"Missing decision line in {output}"
    failed_line = r"Suggestion 2 at \+\d\d:\d\d: \(empty\), origin typed, failed during generation: StrokeSuggestionError"
    assert re.search(failed_line, output), f"Missing failed press in {output}"
    assert "computed by show" not in output, "A finished drawing's decisions are recorded, not computed"

    # Every drawing file matches its checksum.
    assert "files match their recorded checksums" in output, f"Expected matching files in {output}"


def test_show_computes_an_open_window_of_an_unfinished_drawing(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                               sheet_frame: np.ndarray,
                                                               draw_part_on_sheet: Callable[[np.ndarray, bytes],
                                                                                            np.ndarray]) -> None:
    """
    An unfinished drawing prints as not finished, and its open window is decided by show and labelled as such.
    """
    session, _, _ = draw_named_experiment(tmp_path / "runs", sheet_frame, draw_part_on_sheet)
    output = show_text(session.folder, capsys)

    # The drawing is not finished, and the open window's result is marked as computed.
    assert "Drawing 1: not finished, 3 captures" in output, f"Expected an unfinished drawing in {output}"
    assert "followed, computed by show, not recorded: part 1 followed" in output, f"Missing computed result in {output}"
    assert "Window open, 2 captures so far." in output, f"Expected an open window in {output}"


def test_show_older_drawing_and_changed_files(tmp_path: Path, capsys: pytest.CaptureFixture[str], sheet_frame: np.ndarray,
                                              draw_part_on_sheet: Callable[[np.ndarray, bytes], np.ndarray]) -> None:
    """
    A drawing saved before decisions gets the provisional rule, marked as computed, and a changed mask is a listed problem.
    """
    session, drawing, _ = draw_named_experiment(tmp_path / "runs", sheet_frame, draw_part_on_sheet)
    drawing.finish(None)
    drawing_file = session.folder / "drawing-1" / "drawing.json"
    data = json.loads(drawing_file.read_text(encoding="utf-8"))
    del data["rule"]
    for suggestion in data["suggestions"]:
        del suggestion["window"], suggestion["decision"]
    drawing_file.write_text(json.dumps(data), encoding="utf-8")
    (session.folder / "drawing-1" / "suggestions" / "suggestion-1" / "part-2.png").write_bytes(b"changed")
    output = show_text(session.folder, capsys)

    # The rule is named as applied by show, the result is marked as computed, and the changed mask is a problem.
    assert "Rule, not recorded, so show applies the provisional one: coverage_16 >= 0.35" in output, output
    assert "followed, computed by show, not recorded" in output, f"Missing computed result in {output}"
    assert "drawing-1/suggestions/suggestion-1/part-2.png does not match its recorded checksum." in output, output


def test_show_prints_study_conditions_prompt_suggestions_and_final_prompts(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], sheet_capture: CapturedSketch,
        make_prompt_suggestion: Callable[..., PromptSuggestionRecord]) -> None:
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path / "runs"), name="p1_car")
    with_ai = session.start_drawing(FakeStrokeSuggester(), canvas_width=512, canvas_height=360, save_interval_seconds=2.0,
                                    study=DrawingStudy(condition=StudyCondition.AI))
    now = datetime.now(timezone.utc)
    shown = with_ai.record_prompt_suggestion(replace(make_prompt_suggestion("a car, studio render", "p-1"), requested_at=now,
                                                     shown_at=now))
    with_ai.suggest(sheet_capture, source=SketchSource.UPLOAD, rotation=0, prompt="a car, studio render",
                    prompt_suggestion_id=shown.id)
    with_ai.finish(sheet_capture, source=SketchSource.UPLOAD, prompt="a red car", prompt_suggestion_id=shown.id)
    without_ai = session.start_drawing(None, canvas_width=512, canvas_height=360, save_interval_seconds=2.0,
                                       study=DrawingStudy(condition=StudyCondition.NO_AI))
    without_ai.finish(sheet_capture, source=SketchSource.UPLOAD, prompt="")
    output = show_text(session.folder, capsys)

    # Each drawing names its condition and final prompt with how it came about.
    assert "Condition: with AI" in output and "Condition: without AI" in output, f"Missing conditions in {output}"
    assert 'Final prompt: "a red car", origin edited 1' in output, f"Missing edited final prompt in {output}"
    assert "Final prompt: (empty), origin typed Suggestions: off" in output, f"Missing typed final prompt in {output}"

    # The prompt suggestion is listed with its suggester, text, and position, and the stroke suggestion names it.
    assert "Prompt suggester: sketchloop.prompt-templates 1, model unavailable (fixed style templates" in output, output
    suggestion_line = r'Prompt suggestion 1 at \+00:00: "a car, studio render", list position 0, suggester ran 0\.01 s'
    assert re.search(suggestion_line, output), f"Missing prompt suggestion line in {output}"
    assert '"a car, studio render", origin suggested 1, 2 parts' in output, f"Missing suggested origin in {output}"


def test_show_older_records_say_prompt_details_were_not_recorded(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                                 sheet_capture: CapturedSketch) -> None:
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path / "runs"), name="p1_car")
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE)
    drawing = session.start_drawing(FakeStrokeSuggester(), canvas_width=512, canvas_height=360, save_interval_seconds=2.0)
    drawing.suggest(sheet_capture, source=SketchSource.UPLOAD, rotation=0, prompt="a car")
    drawing.finish(sheet_capture, source=SketchSource.UPLOAD, prompt="a car")

    # Delete the keys T10 added, standing in for records saved before it.
    round_file, drawing_file = session.folder / "round-1" / "round.json", session.folder / "drawing-1" / "drawing.json"
    round_data = json.loads(round_file.read_text(encoding="utf-8"))
    drawing_data = json.loads(drawing_file.read_text(encoding="utf-8"))
    del round_data["prompt_origin"], round_data["prompt_suggestions"]
    for key in ("study", "prompt_suggestions", "final_prompt", "final_prompt_origin"):
        del drawing_data[key]
    del drawing_data["suggestions"][0]["prompt_origin"]
    round_file.write_text(json.dumps(round_data), encoding="utf-8")
    drawing_file.write_text(json.dumps(drawing_data), encoding="utf-8")
    output = show_text(session.folder, capsys)

    # Every missing value says it was not recorded, and no condition is invented.
    assert "Prompt origin: unavailable (not recorded before prompt origins were)" in output, f"Missing round origin in {output}"
    assert output.count("Prompt suggestions: not recorded") == 2, f"Expected two unrecorded lists in {output}"
    assert "Final prompt: unavailable (not recorded in drawings saved before final prompts were)" in output, output
    assert '"a car", origin unavailable (not recorded before prompt origins were)' in output, f"Missing origin in {output}"
    assert "Condition:" not in output, f"An older drawing has no condition, got {output}"
