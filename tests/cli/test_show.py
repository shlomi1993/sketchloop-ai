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


def show_text(folder: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """
    Run sketchloop show on a session folder and return its output with line wrapping removed.
    """
    exit_code = main(["show", str(folder)])
    assert exit_code == 0, f"show must succeed, got exit code {exit_code}"
    return " ".join(capsys.readouterr().out.split())


def test_show_prints_round_timings_scores_and_older_records(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                            constant_evaluator: Evaluator) -> None:
    """
    show prints setup, hardware, timings, frame age, and scores, and names what a session saved by an older version lacks.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path / "runs"), constant_evaluator, generator_setup_seconds=1.25)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.WEBCAM, capture_seconds=0.1,
                      frame_age_seconds=0.42)
    folder = session.folder
    output = show_text(folder, capsys)

    # The session header and the round name every measurement and what the scores mean.
    assert "Generator setup: 1.25 s" in output and "Hardware: " in output, "show must print the setup time and hardware"
    assert "Timings: capture 0.10 s" in output and "total " in output, "show must print the capture and the total"
    assert "(source: webcam), frame age 0.42 s" in output, "show must print the webcam frame age"
    assert "Scores: ConstantEvaluator, score=0.5 for every candidate, a test value" in output, f"Missing score line in {output}"
    assert "Recorded by an older sketchloop" not in output, "A current round must not get the older-timings note"

    # A time-named session without drawings prints no experiment or drawing.
    assert "Experiment:" not in output and "Drawing 1" not in output, f"Unexpected experiment or drawing in {output}"

    # Delete the timing and hardware keys, standing in for older files.
    session_file, round_file = folder / "session.json", folder / "round-1" / "round.json"
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    round_data = json.loads(round_file.read_text(encoding="utf-8"))
    del session_data["environment"]["hardware"], session_data["generator_setup_seconds"], round_data["total_seconds"]
    session_file.write_text(json.dumps(session_data), encoding="utf-8")
    round_file.write_text(json.dumps(round_data), encoding="utf-8")
    older = show_text(folder, capsys)

    # Each missing value is named with its reason, and the round gets the older-timings note.
    assert "Hardware: unavailable (recorded before hardware was tracked)" in older, "Missing hardware must be named"
    assert "Generator setup: unavailable (recorded before generator setup was timed)" in older, "Missing setup must be named"
    assert "Recorded by an older sketchloop: generation may include the model load" in older, "An older round must get the note"


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


def test_show_prints_a_drawing_open_finished_and_older(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                      sheet_frame: np.ndarray,
                                                      draw_part_on_sheet: Callable[[np.ndarray, bytes], np.ndarray]) -> None:
    """
    show decides an open window itself and says so, prints recorded decisions once finished, and applies the rule to older ones.
    """
    session, drawing, with_ring = draw_named_experiment(tmp_path / "runs", sheet_frame, draw_part_on_sheet)
    unfinished = show_text(session.folder, capsys)

    # The unfinished drawing's open window is decided by show and labelled as such.
    assert "Drawing 1: not finished, 3 captures" in unfinished, f"Expected an unfinished drawing in {unfinished}"
    assert "followed, computed by show, not recorded: part 1 followed" in unfinished, f"Missing computed result in {unfinished}"
    assert "Window open, 2 captures so far." in unfinished, f"Expected an open window in {unfinished}"

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
    assert "files match their recorded checksums" in output, f"Expected matching files in {output}"

    # Delete the rule and decisions, standing in for a drawing saved before them, and change a mask.
    drawing_file = session.folder / "drawing-1" / "drawing.json"
    data = json.loads(drawing_file.read_text(encoding="utf-8"))
    del data["rule"]
    for suggestion in data["suggestions"]:
        del suggestion["window"], suggestion["decision"]
    drawing_file.write_text(json.dumps(data), encoding="utf-8")
    (session.folder / "drawing-1" / "suggestions" / "suggestion-1" / "part-2.png").write_bytes(b"changed")
    older = show_text(session.folder, capsys)

    # The rule is named as applied by show, the result is marked as computed, and the changed mask is a problem.
    assert "Rule, not recorded, so show applies the provisional one: coverage_16 >= 0.35" in older, older
    assert "followed, computed by show, not recorded" in older, f"Missing computed result in {older}"
    assert "drawing-1/suggestions/suggestion-1/part-2.png does not match its recorded checksum." in older, older


def test_show_prints_study_conditions_prompt_suggestions_and_final_prompts(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], sheet_capture: CapturedSketch,
        make_prompt_suggestion: Callable[..., PromptSuggestionRecord]) -> None:
    """
    show prints each drawing's condition, prompt suggestions, and final prompt, and names them unrecorded in older records.
    """
    session = SketchSession(FakeGenerator(), FolderStore(tmp_path / "runs"), name="p1_car")
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=SketchSource.FILE)
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

    # Delete the keys T10 added, standing in for records saved before it.
    round_file = session.folder / "round-1" / "round.json"
    round_data = json.loads(round_file.read_text(encoding="utf-8"))
    del round_data["prompt_origin"], round_data["prompt_suggestions"]
    round_file.write_text(json.dumps(round_data), encoding="utf-8")
    for number in (1, 2):
        drawing_file = session.folder / f"drawing-{number}" / "drawing.json"
        drawing_data = json.loads(drawing_file.read_text(encoding="utf-8"))
        for key in ("study", "prompt_suggestions", "final_prompt", "final_prompt_origin"):
            del drawing_data[key]
        for suggestion in drawing_data["suggestions"]:
            del suggestion["prompt_origin"]
        drawing_file.write_text(json.dumps(drawing_data), encoding="utf-8")
    older = show_text(session.folder, capsys)

    # Every missing value says it was not recorded, and no condition is invented.
    assert "Prompt origin: unavailable (not recorded before prompt origins were)" in older, f"Missing round origin in {older}"
    assert older.count("Prompt suggestions: not recorded") == 2, f"Expected the round and AI drawing lists unrecorded in {older}"
    assert "Final prompt: unavailable (not recorded in drawings saved before final prompts were)" in older, older
    assert '"a car, studio render", origin unavailable (not recorded before prompt origins were)' in older, older
    assert "Condition:" not in older, f"An older drawing has no condition, got {older}"
