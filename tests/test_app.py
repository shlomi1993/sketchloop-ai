"""
Tests of the browser controller's rounds, selections, history, and new sessions.
"""

import cv2
import hashlib
import json
import numpy as np
import pytest
import re
import time

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from sketchloop.app import NO_DRAWING_MESSAGE, NO_NEW_STROKES_MESSAGE, AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import InvalidSketchError
from sketchloop.domain import (BackendIdentity, ExecutionKind, InvalidSelectionError, SketchLoopError, Unavailable,
                               UnsupportedConfigurationError)
from sketchloop.experiments import (OUTSIDE_SESSION_REASON, SETUP_REUSED_REASON, ExperimentNameError, ExperimentNameTakenError,
                                    FolderStore, PromptOrigin, PromptOriginKind, PromptSuggestionStatus, StudyCondition,
                                    load_session)
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggestionError, StrokeSuggestionOutput, StrokeSuggestionRequest
from sketchloop.suggestion import STYLE_TEMPLATES, FlorenceSuggester, PromptSuggester, TemplateSuggester


class EmptySuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester that finds nothing new to suggest.
    """

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Return the fake's suggestion without parts, keeping only the line drawing.
        """
        output = super().suggest(request, canvas_payload)
        line_drawing = output.suggestion.line_drawing
        return StrokeSuggestionOutput(suggestion=replace(output.suggestion, parts=()),
                                      payloads={line_drawing.path: output.payloads[line_drawing.path]})


class FailingStrokeSuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester whose suggestion fails as if the device ran out of memory.
    """

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Raise the error a real suggester raises when generation fails.
        """
        raise StrokeSuggestionError("The device ran out of memory. Close other apps and retry.")


class StubPromptSuggester(PromptSuggester):
    # Test-only prompt suggester that returns fixed prompts, counts its runs, and can fail as a missing model would.

    def __init__(self, prompts: tuple[str, ...] = ("a chair, studio render", "a chair, watercolor")) -> None:
        self.prompts = prompts
        self.error: Exception | None = None
        self.calls = 0

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(adapter="tests.stub-prompts", adapter_version="1", execution=ExecutionKind.FAKE, model_id="stub")

    def suggest(self, sketch_payload: bytes) -> tuple[str, ...]:
        # Count every run, and raise the configured error instead of suggesting when there is one.
        self.calls += 1
        if self.error is not None:
            raise self.error

        return self.prompts


def write_sketch_png(path: Path, channels: int) -> Path:
    """
    Write a synthetic light page with one dark rectangle as an RGB or RGBA PNG, like a browser webcam or upload file.
    """
    page = np.full((600, 900, channels), 210, dtype=np.uint8)
    cv2.rectangle(page, (500, 100), (700, 200), (30,) * channels, 4)

    # Keep an RGBA page fully opaque, as a browser upload of a scanned page would be.
    if channels == 4:
        page[..., 3] = 255

    path.write_bytes(cv2.imencode(".png", page)[1].tobytes())
    return path


@pytest.fixture
def sketch_file(tmp_path: Path) -> Path:
    """
    Provide a synthetic RGB PNG sketch standing in for the browser's file.
    """
    return write_sketch_png(tmp_path / "browser-sketch.png", channels=3)


@pytest.fixture
def controller(tmp_path: Path) -> AppController:
    """
    Provide a controller with the deterministic fake backend and a runs folder under tmp_path.
    """
    return AppController(FakeGenerator(), FolderStore(tmp_path / "runs"))


@pytest.fixture(name="prompt_suggester")
def provide_prompt_suggester() -> StubPromptSuggester:
    return StubPromptSuggester()


def read_round_files(controller: AppController) -> dict[str, bytes]:
    """
    Read every round.json of the controller's current session, keyed by round folder name.
    """
    return {path.parent.name: path.read_bytes() for path in sorted(controller.session.folder.glob("round-*/round.json"))}


@pytest.mark.parametrize(("source", "channels"), [("webcam", 3), ("upload", 4)])
def test_app_round_records_browser_source(tmp_path: Path, controller: AppController, source: str, channels: int) -> None:
    """
    A streamed webcam frame or an uploaded file gives fake-labeled candidates on disk, and records the source and raw bytes.
    """
    # The webcam streams pixels, which are saved as a lossless PNG, while an upload keeps the file's own bytes.
    sketch = write_sketch_png(tmp_path / f"{source}.png", channels)
    frame = cv2.imread(str(sketch))
    sketch_input = frame if source == "webcam" else sketch
    raw_bytes = cv2.imencode(".png", frame)[1].tobytes() if source == "webcam" else sketch.read_bytes()
    view = controller.run_round(sketch_input, source, "modern chair", n_candidates=3, seed=4)
    round_folder = controller.session.folder / "round-1"
    data = json.loads((round_folder / "round.json").read_text(encoding="utf-8"))
    saved = load_session(controller.session.folder)

    # The view is round 1 with the requested candidates, each stored under the round's candidates folder.
    candidate_folders = {candidate.image_path.resolve().parent for candidate in view.candidates}
    assert view.number == 1 and len(view.candidates) == 3, f"Expected round 1 with 3 candidates, got {view}"
    assert candidate_folders == {(round_folder / "candidates").resolve()}, f"Candidates stored elsewhere: {candidate_folders}"
    assert all(candidate.image_path.is_file() for candidate in view.candidates), "Every candidate image must exist"
    assert view.sketch_path.is_file() and view.selection is None, f"Expected a stored sketch and no selection, got {view}"

    # The fake backend is labeled on the view and on every caption.
    labels = [candidate.label for candidate in view.candidates]
    assert view.is_fake and labels == ["1 (fake backend)", "2 (fake backend)", "3 (fake backend)"], f"Bad labels {labels}"

    # The record names the browser source, and the raw sketch holds the frame's PNG or the uploaded file's bytes.
    expected_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    assert data["input"]["source"] == source, f"Expected input.source {source}, got {data['input'].get('source')}"
    assert data["input"]["raw_sketch"]["sha256"] == expected_sha256, "The raw sketch must hold the frame or file bytes"

    # The source loads back unchanged and every file matches its checksum.
    assert saved.rounds[0].source == source and saved.problems == (), f"Bad reload: {saved.rounds[0].source}, {saved.problems}"

    # The browser frame encode or upload load is the capture stage, and the view carries the saved total.
    assert "capture" in data["timings_seconds"], f"The round must time its capture, got {data['timings_seconds']}"
    assert view.total_seconds == data["total_seconds"], f"The view total {view.total_seconds} must equal the saved total"


@pytest.mark.parametrize(("sketch_kind", "source", "prompt", "options", "error", "message"), [
    ("text", "upload", "chair", {}, InvalidSketchError, "Sketch notes.png is not a PNG or JPEG file."),
    ("png", "upload", "   ", {}, SketchLoopError, "Enter a prompt, then generate."),
    ("png", "upload", "chair", {"rotation": 45}, SketchLoopError, "Rotation must be one of 0, 90, 180, or 270, got 45."),
    ("png", "upload", "chair", {"rotation": 90, "use_raw": True}, SketchLoopError,
     "Rotation cannot be combined with the raw sketch option, which skips preprocessing.")
], ids=["not-an-image", "blank-prompt", "bad-rotation", "rotation-with-raw"])
def test_app_rejects_input_before_writing(tmp_path: Path, controller: AppController, sketch_file: Path, sketch_kind: str,
                                          source: str, prompt: str, options: dict[str, object], error: type[Exception],
                                          message: str) -> None:
    """
    A non-image sketch, a blank prompt, a bad rotation, or rotation with raw saves nothing.
    """
    text_file = tmp_path / "notes.png"
    text_file.write_text("synthetic text, not an image", encoding="utf-8")
    sketch = {"text": text_file, "png": sketch_file}[sketch_kind]

    # The controller raises the documented one-line message.
    with pytest.raises(error, match=re.escape(message)):
        controller.run_round(sketch, source, prompt, n_candidates=2, **options)

    # No session folder is created for refused input.
    assert not controller.store.root.exists(), "Refused input must not create a session folder"


def test_app_selection_rules(controller: AppController, sketch_file: Path) -> None:
    """
    Candidates 1 and 3 are saved, none is explicit, and nothing to select, a repeat, a stale page, or a foreign ID is refused.
    """
    # Nothing can be selected before the first round.
    with pytest.raises(InvalidSelectionError, match=re.escape("No round to select from yet. Generate first.")):
        controller.record_selection(None, [])

    first = controller.run_round(sketch_file, "upload", "chair", n_candidates=3)
    picked = controller.record_selection(first.iteration_id, [first.candidates[0].candidate_id, first.candidates[2].candidate_id])
    round1_data = json.loads((controller.session.folder / "round-1" / "round.json").read_text(encoding="utf-8"))

    # The pick is shown as "1, 3" and saved with the two candidate IDs.
    expected_ids = [first.candidates[0].candidate_id, first.candidates[2].candidate_id]
    assert picked.selection == "1, 3", f"Expected selection 1, 3, got {picked.selection}"
    assert round1_data["selection"]["candidate_ids"] == expected_ids, f"Saved selection differs: {round1_data['selection']}"

    # A second selection for the same round is refused without touching its record.
    before = read_round_files(controller)
    repeat_message = "Round 1 already has a recorded selection. Generate the next round."
    with pytest.raises(InvalidSelectionError, match=re.escape(repeat_message)):
        controller.record_selection(first.iteration_id, [])
    assert read_round_files(controller) == before, "A refused repeat selection must not change any round.json"

    # A stale tab still showing round 1 cannot record a choice once round 2 exists.
    second = controller.run_round(sketch_file, "upload", "lamp", n_candidates=2)
    before = read_round_files(controller)
    stale_message = "This page shows an older round. Reload the page to see round 2."
    with pytest.raises(InvalidSelectionError, match=re.escape(stale_message)):
        controller.record_selection(first.iteration_id, [])
    assert read_round_files(controller) == before, "A stale selection must leave every round.json unchanged"

    # A candidate from round 1 cannot be selected in round 2.
    with pytest.raises(InvalidSelectionError):
        controller.record_selection(second.iteration_id, [first.candidates[0].candidate_id])
    assert read_round_files(controller) == before, "A foreign candidate ID must leave every round.json unchanged"

    # An empty pick is recorded as an explicit none.
    declined = controller.record_selection(second.iteration_id, [])
    round2_data = json.loads((controller.session.folder / "round-2" / "round.json").read_text(encoding="utf-8"))
    assert declined.selection == "none", f"An empty pick must show none, got {declined.selection}"
    assert round2_data["selection"]["candidate_ids"] == [], f"An explicit none must be saved, got {round2_data['selection']}"

    # A tab still showing round 1 of the old session cannot record none for round 1 of a new session.
    controller.start_new_session()
    controller.run_round(sketch_file, "upload", "stool", n_candidates=1)
    with pytest.raises(InvalidSelectionError, match=re.escape("Reload the page to see round 1.")):
        controller.record_selection(first.iteration_id, [])


def test_app_history_refresh_and_new_session(controller: AppController, sketch_file: Path) -> None:
    """
    Rounds link, a failed round joins the history but not the view, a reload view matches, and a new session keeps the backend.
    """
    generator = controller.session.generator
    empty_view, empty_history = controller.latest_round(), controller.list_history()
    first = controller.run_round(sketch_file, "webcam", "chair", n_candidates=2)
    picked = controller.record_selection(first.iteration_id, [first.candidates[1].candidate_id])
    reloaded_first = controller.latest_round()
    second = controller.run_round(sketch_file, "upload", "lamp", n_candidates=1, use_raw=True)
    with pytest.raises(UnsupportedConfigurationError):
        controller.run_round(sketch_file, "upload", "vase", n_candidates=99)

    history = controller.list_history()
    saved = load_session(controller.session.folder)
    old_folder = controller.session.folder

    # Before any round there is nothing to restore and no history.
    assert empty_view is None and empty_history == (), f"Expected no view or history, got {empty_view}, {empty_history}"

    # A reload shows the latest completed round with its recorded selection, even after a failed round.
    assert reloaded_first == picked and picked.selection == "2", f"A reload must show round 1 picked as 2, got {reloaded_first}"
    assert controller.latest_round() == second, f"A failed round must not replace round 2, got {controller.latest_round()}"

    # Round 2 links to round 1, and the history lists all three rounds, including the failure.
    rows = [(row.number, row.status, row.prompt) for row in history]
    assert saved.rounds[1].iteration.parent_id == saved.rounds[0].iteration.id, "Round 2 must link to round 1"
    assert rows == [(1, "complete", "chair"), (2, "complete", "lamp"), (3, "failed", "vase")], f"Unexpected history {rows}"
    outcomes = [row.outcome for row in history]
    assert outcomes[:2] == ["selected 2", "not selected"], f"Unexpected outcomes {outcomes}"
    assert outcomes[2].startswith("failed during generation: "), f"Round 3 must fail in generation, got {outcomes[2]}"

    # A new session starts empty, keeps the loaded generator, and saves its first round in a new unlinked folder.
    controller.start_new_session()
    new_history, new_view = controller.list_history(), controller.latest_round()
    restarted = controller.run_round(sketch_file, "webcam", "stool", n_candidates=1)
    new_saved = load_session(controller.session.folder)
    assert new_history == () and new_view is None, f"A new session must start empty, got {new_history}, {new_view}"
    assert controller.session.generator is generator, "A new session must reuse the loaded generator"
    assert controller.session.folder != old_folder and restarted.number == 1, "A new session must number rounds from 1 anew"
    assert new_saved.rounds[0].iteration.parent_id is None, "The first round of a new session must not link to the old one"


def test_app_records_frame_age_and_setup(tmp_path: Path, sketch_file: Path) -> None:
    """
    A webcam round records its frame age, an age with an upload is refused, and only the first session has the setup time.
    """
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), generator_setup_seconds=2.0)
    frame = cv2.imread(str(sketch_file))

    # An age with the upload source is refused before anything is read or saved.
    with pytest.raises(SketchLoopError, match="Only a streamed webcam frame has a frame age"):
        controller.run_round(sketch_file, "upload", "chair", n_candidates=1, frame_age_seconds=0.4)
    assert not controller.store.root.exists(), "A refused frame age must save nothing"

    # A webcam round records the age, and the first session the setup time.
    controller.run_round(frame, "webcam", "chair", n_candidates=1, frame_age_seconds=0.4)
    first = load_session(controller.session.folder)
    assert first.rounds[0].frame_age_seconds == 0.4, f"Expected frame age 0.4, got {first.rounds[0].frame_age_seconds}"
    assert first.session.generator_setup_seconds == 2.0, f"Expected setup 2.0, got {first.session.generator_setup_seconds}"

    # A new session reuses the generator, so it records the reused reason instead of the setup time again.
    controller.start_new_session()
    controller.run_round(sketch_file, "upload", "lamp", n_candidates=1)
    reused = load_session(controller.session.folder).session.generator_setup_seconds
    assert reused == Unavailable(SETUP_REUSED_REASON), f"A new session must record the reused reason, got {reused}"


def test_app_suggests_the_next_style_for_the_same_drawing(tmp_path: Path, sketch_file: Path) -> None:
    """
    The first press names the drawing in the first style, and each later press for the same drawing gives the next style.
    """
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"),
                               suggester=FlorenceSuggester(captioner=lambda payload: "a drawing of a chair"))
    first = controller.request_prompt_suggestion(sketch_file, 0, False).text
    second = controller.request_prompt_suggestion(sketch_file, 0, False).text

    # The styles follow the template order for the named object.
    assert first == f"a chair, {STYLE_TEMPLATES[0].split(', ', 1)[1]}", f"Unexpected first suggestion {first!r}"
    assert second == f"a chair, {STYLE_TEMPLATES[1].split(', ', 1)[1]}", f"Expected the second style, got {second!r}"


def test_app_guided_drawing_in_named_folder(tmp_path: Path, sheet_frame: np.ndarray) -> None:
    """
    Starting a drawing names the session folder, an empty prompt is suggested for, and a round lands in the same folder.
    """
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), generator_setup_seconds=3.0)
    started = controller.start_drawing("participant1_car")
    view = controller.suggest_strokes(sheet_frame, "webcam", "", rotation=0, frame_age_seconds=0.1)
    preview = cv2.imdecode(np.frombuffer(controller.preview_guided_sketch(sheet_frame, 0), np.uint8), cv2.IMREAD_COLOR)
    controller.run_round(sheet_frame, "webcam", "a car", n_candidates=1)
    finished = controller.finish_drawing(sheet_frame, "webcam", rotation=0)
    folder = tmp_path / "runs" / "participant1_car"
    saved = load_session(folder)

    # The experiment's folder is its name and holds the drawing and the round, with the unrecorded setup time carried over.
    assert controller.session.folder == folder and saved.session.name == "participant1_car", f"Unexpected folder {folder}"
    assert len(saved.drawings) == 1 and len(saved.rounds) == 1 and saved.problems == (), f"Unexpected session {saved}"
    setup_seconds = saved.session.generator_setup_seconds
    assert setup_seconds == 3.0, f"The unrecorded setup time must be carried over, got {setup_seconds}"

    # The views report the drawing and a fake suggestion with its colors, recorded with the prompt exactly as typed.
    assert started.is_open and started.experiment == "participant1_car", f"Unexpected start view {started}"
    assert view.part_count == 2 and view.is_fake and view.status == "Suggestion 1: 2 colored parts", f"Bad view {view}"
    assert saved.drawings[0].suggestions[0].request.prompt == "", "The empty prompt must be recorded unchanged"
    assert not finished.is_open and finished.suggestion_lines == ("Suggestion 1: 2 colored parts",), f"Bad finish view {finished}"
    assert controller.latest_drawing() == finished, "latest_drawing must restore the finished view"

    # The guided preview has the canvas size.
    assert preview.shape[:2] == (360, 512), f"Expected a 512x360 preview, got {preview.shape}"


def test_app_drawing_operations_need_an_open_drawing(controller: AppController, sheet_frame: np.ndarray) -> None:
    """
    Without an open drawing, suggest and finish are refused, and the guided preview and latest drawing are None.
    """
    with pytest.raises(SketchLoopError, match=NO_DRAWING_MESSAGE):
        controller.suggest_strokes(sheet_frame, "webcam", "a car", rotation=0)
    with pytest.raises(SketchLoopError, match=NO_DRAWING_MESSAGE):
        controller.finish_drawing(sheet_frame, "webcam", rotation=0)

    # The page keeps its ordinary preview, there is no drawing to restore, and the header names the fake suggester.
    assert controller.preview_guided_sketch(sheet_frame, 0) is None, "No drawing means no guided preview"
    assert "fake stroke suggester" in controller.describe_stroke_suggester(), "The default suggester is the labeled fake"
    assert controller.latest_drawing() is None, "No drawing to restore yet"

    # With a drawing open, a missing frame is refused before anything is read.
    controller.start_drawing("p1")
    with pytest.raises(InvalidSketchError):
        controller.suggest_strokes(None, "webcam", "a car", rotation=0)
    assert controller.preview_guided_sketch(None, 0) is None, "No frame means no guided preview"


@pytest.mark.parametrize(("name", "error"), [("P1", ExperimentNameTakenError), ("../p1", ExperimentNameError)])
def test_app_bad_name_keeps_current_drawing(tmp_path: Path, controller: AppController, name: str, error: type[Exception]) -> None:
    """
    A taken or invalid name is refused, and the current session and its open drawing stay as they were.
    """
    controller.start_drawing("p1")
    session = controller.session
    with pytest.raises(error):
        controller.start_drawing(name)

    # The session is unchanged, its drawing is still open, and no other folder was created.
    assert controller.session is session and controller.latest_drawing().is_open, "The current drawing must stay open"
    assert [path.name for path in (tmp_path / "runs").iterdir()] == ["p1"], "A refused name must create nothing"


def test_app_new_session_ends_open_drawing(controller: AppController) -> None:
    """
    Starting a new session ends the open drawing as new_session, without a final capture.
    """
    controller.start_drawing("p1")
    drawing_file = controller.session.folder / "drawing-1" / "drawing.json"
    controller.start_new_session()
    data = json.loads(drawing_file.read_text(encoding="utf-8"))

    # The drawing ended for the new session, and the new session has no drawing.
    assert (data["end"], data["final_capture"]) == ("new_session", None), f"Unexpected end {data['end']}"
    assert controller.latest_drawing() is None, "The new session has no drawing yet"


def test_app_reports_no_new_strokes(tmp_path: Path, sheet_frame: np.ndarray) -> None:
    """
    A suggestion without parts gives a plain one-line message and is still recorded.
    """
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), stroke_suggester=EmptySuggester())
    controller.start_drawing("p1")
    view = controller.suggest_strokes(sheet_frame, "upload", "a car", rotation=0)

    # The page gets the message, and the drawing holds the attempt, which did not fail.
    assert view.status == NO_NEW_STROKES_MESSAGE and view.part_count == 0, f"Unexpected view {view}"
    assert controller.latest_drawing().suggestion_lines == ("Suggestion 1: no new strokes",), "The attempt must be recorded"
    assert controller.latest_drawing().latest_suggestion_failed is False, "A suggestion without parts did not fail"


def test_app_marks_a_failed_latest_suggestion(tmp_path: Path, sheet_frame: np.ndarray) -> None:
    """
    A failed suggestion is recorded and marked in the drawing view, so the page can word it for the participant.
    """
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), stroke_suggester=FailingStrokeSuggester())
    controller.start_drawing("p1")
    with pytest.raises(StrokeSuggestionError):
        controller.suggest_strokes(sheet_frame, "upload", "a car", rotation=0)

    # The view marks the failure, and its line keeps the cause for the experimenter.
    view = controller.latest_drawing()
    assert view.latest_suggestion_failed is True, f"The failed suggestion must be marked, got {view}"
    assert "ran out of memory" in view.suggestion_lines[-1], f"The line must keep the cause, got {view.suggestion_lines}"


class ManualClock:
    """
    Test clock for time.monotonic that moves only when the test advances it.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        """
        Return the current time.
        """
        return self.now


def test_app_records_interval_captures_when_due(tmp_path: Path, sheet_frame: np.ndarray,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Interval captures are saved as JPEG at most once per 0.8 of the interval, and none without a drawing or a frame.
    """
    clock = ManualClock()
    monkeypatch.setattr(time, "monotonic", clock)
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"))

    # Nothing is saved before a drawing starts or without a frame.
    assert controller.record_drawing_capture(sheet_frame, "webcam", rotation=0) is None, "No drawing means no capture"
    controller.start_drawing("p1")
    assert controller.record_drawing_capture(None, "webcam", rotation=0) is None, "No frame means no capture"

    # The first call saves capture 1, an immediate repeat is turned away, and one after 1.6 s saves capture 2.
    first = controller.record_drawing_capture(sheet_frame, "webcam", rotation=0, frame_age_seconds=0.2)
    clock.now += 1.5
    early = controller.record_drawing_capture(sheet_frame, "webcam", rotation=0)
    clock.now += 0.1
    second = controller.record_drawing_capture(sheet_frame, "webcam", rotation=0)
    assert (first.drawing, first.number, early, second.number) == (1, 1, None, 2), f"Unexpected captures {first}, {early}"
    assert first.follow is None and not first.canvas_failed, f"Expected a plain capture before any suggestion, got {first}"

    # After a suggestion, the next due capture reports its alignment to it.
    controller.suggest_strokes(sheet_frame, "webcam", "", rotation=0)
    clock.now += 2.0
    aligned = controller.record_drawing_capture(sheet_frame, "webcam", rotation=0)
    assert aligned.number == 4 and aligned.follow.suggestion == 1, f"Expected capture 4 aligned to suggestion 1, got {aligned}"
    assert aligned.follow.alignment.converged, f"The same frame must align, got {aligned.follow.alignment}"

    # The interval frames are JPEG on disk, and the saved drawing loads with every file matching.
    saved = load_session(controller.session.folder)
    raw_paths = [capture.raw.path for capture in saved.drawings[0].captures]
    assert raw_paths[0].endswith("-raw.jpg") and raw_paths[2].endswith("-raw.png"), f"Unexpected raw formats {raw_paths}"
    assert saved.problems == (), f"Expected no problems, got {saved.problems}"


def test_app_capture_turns_away_concurrent_call(controller: AppController, sheet_frame: np.ndarray) -> None:
    """
    A capture call that arrives while another is saving returns None at once instead of saving a second frame.
    """
    controller.start_drawing("p1")
    with controller._capture_lock:
        busy = controller.record_drawing_capture(sheet_frame, "webcam", rotation=0)

    # The busy call saved nothing, and the next call saves capture 1.
    assert busy is None and controller.latest_drawing().capture_count == 0, "A concurrent call must save nothing"
    assert controller.record_drawing_capture(sheet_frame, "webcam", rotation=0).number == 1, "The next call must save"


def test_app_capture_refuses_bad_input(controller: AppController, sheet_frame: np.ndarray) -> None:
    """
    A due capture with an unsupported rotation or a frame age for an upload is refused before anything is saved.
    """
    controller.start_drawing("p1")
    with pytest.raises(SketchLoopError, match="Rotation must be one of"):
        controller.record_drawing_capture(sheet_frame, "webcam", rotation=45)
    with pytest.raises(SketchLoopError, match="Only a streamed webcam frame has a frame age"):
        controller.record_drawing_capture(sheet_frame, "upload", rotation=0, frame_age_seconds=0.2)

    # Neither call saved a capture.
    assert controller.latest_drawing().capture_count == 0, "A refused capture must save nothing"


def make_study_controller(tmp_path: Path, prompt_suggester: StubPromptSuggester) -> AppController:
    # A controller with the fake backend, the fake stroke suggester, and the given prompt suggester.
    return AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), suggester=prompt_suggester)


def test_app_no_ai_drawing_refuses_ai_help(tmp_path: Path, sheet_frame: np.ndarray,
                                           prompt_suggester: StubPromptSuggester) -> None:
    controller = make_study_controller(tmp_path, prompt_suggester)
    started = controller.start_drawing("p1_car", StudyCondition.NO_AI)

    # The view carries the condition and says the page offers no AI help, with a local start time.
    assert started.condition == StudyCondition.NO_AI and not started.suggestions_enabled, f"Unexpected view {started}"
    assert started.started_at.utcoffset() == datetime.now().astimezone().utcoffset(), f"Expected local time, got {started}"

    # A prompt suggestion is refused before the suggester runs, and so is a stroke suggestion.
    with pytest.raises(SketchLoopError, match="Suggestions are off in this drawing."):
        controller.request_prompt_suggestion(sheet_frame, 0, False)
    with pytest.raises(SketchLoopError, match="Suggestions are off"):
        controller.suggest_strokes(sheet_frame, "webcam", "a car", rotation=0)
    assert prompt_suggester.calls == 0, f"The suggester must not run, ran {prompt_suggester.calls} times"

    # Finishing records the typed final prompt, and the drawing holds the condition and no prompt suggestions.
    controller.finish_drawing(sheet_frame, "webcam", rotation=0, prompt="a fast car")
    drawing = load_session(controller.session.folder).drawings[0]
    assert drawing.study.condition == StudyCondition.NO_AI and drawing.suggester is None, f"Unexpected drawing {drawing.study}"
    assert drawing.prompt_suggestions == (), f"Expected no prompt suggestions, got {drawing.prompt_suggestions}"
    typed = PromptOrigin(kind=PromptOriginKind.TYPED, prompt_suggestion=None)
    final = (drawing.final_prompt, drawing.final_prompt_origin)
    assert final == ("a fast car", typed), f"Expected the typed final prompt, got {final}"


def test_app_ai_drawing_records_prompt_suggestions_and_origins(tmp_path: Path, sheet_frame: np.ndarray,
                                                               prompt_suggester: StubPromptSuggester) -> None:
    controller = make_study_controller(tmp_path, prompt_suggester)
    started = controller.start_drawing("p1_car", "ai")
    first = controller.request_prompt_suggestion(sheet_frame, 0, False)
    second = controller.request_prompt_suggestion(sheet_frame, 0, False)
    controller.suggest_strokes(sheet_frame, "webcam", f"  {second.text} ", rotation=0, prompt_suggestion_id=second.suggestion_id)
    controller.finish_drawing(sheet_frame, "webcam", rotation=0, prompt="a red chair", prompt_suggestion_id=first.suggestion_id)
    saved = load_session(controller.session.folder)
    drawing = saved.drawings[0]

    # A page value for the condition is accepted, and the view offers AI help.
    assert started.condition == StudyCondition.AI and started.suggestions_enabled, f"Unexpected view {started}"

    # The first press ran the suggester and the second took the next entry, both recorded under the returned IDs.
    records = drawing.prompt_suggestions
    assert [record.id for record in records] == [first.suggestion_id, second.suggestion_id], f"Unexpected IDs {records}"
    assert (first.text, second.text) == prompt_suggester.prompts, f"Expected the prompts in order, got {first}, {second}"
    ran = [(record.ran_suggester, record.list_position, record.status) for record in records]
    assert ran == [(True, 0, "shown"), (False, 1, "shown")], f"Unexpected runs {ran}"
    assert records[0].suggester_seconds >= 0 and records[1].suggester_seconds is None, "Only a run is timed"
    assert prompt_suggester.calls == 1 and records[0].suggester.adapter == "tests.stub-prompts", "The stub ran once"

    # The stroke suggestion's prompt came from the second suggestion, and the final prompt was edited from the first.
    origin = drawing.suggestions[0].prompt_origin
    assert origin == PromptOrigin(kind=PromptOriginKind.SUGGESTED, prompt_suggestion=second.suggestion_id), f"Got {origin}"
    edited = PromptOrigin(kind=PromptOriginKind.EDITED, prompt_suggestion=first.suggestion_id)
    assert drawing.final_prompt_origin == edited, f"Expected an edited final prompt, got {drawing.final_prompt_origin}"
    assert saved.problems == (), f"Expected no problems, got {saved.problems}"

    # A new experiment forgets the cached list, so its first press runs the suggester again.
    controller.start_drawing("p1_boat", StudyCondition.AI)
    controller.request_prompt_suggestion(sheet_frame, 0, False)
    assert prompt_suggester.calls == 2, f"A new experiment must run the suggester again, ran {prompt_suggester.calls} times"


def test_app_records_failed_prompt_suggestion(tmp_path: Path, sheet_frame: np.ndarray,
                                              prompt_suggester: StubPromptSuggester) -> None:
    """
    An unexpected prompt suggester error reaches the page as one line and is recorded as a failed press.
    """
    controller = make_study_controller(tmp_path, prompt_suggester)
    controller.start_drawing("p1_car", StudyCondition.AI)
    error = RuntimeError("MPS backend out of memory\nstack details")
    message = "Prompt suggestion failed: MPS backend out of memory. Retry, or type a prompt."
    prompt_suggester.error = error
    with pytest.raises(SketchLoopError, match=re.escape(message)) as raised:
        controller.request_prompt_suggestion(sheet_frame, 0, False)
    records = load_session(controller.session.folder).drawings[0].prompt_suggestions

    # The page gets one line, and the drawing records a failed press with the error's class and first line.
    assert "\n" not in str(raised.value), f"Expected one line, got {raised.value!r}"
    failure = records[0].failure
    assert len(records) == 1 and records[0].status == PromptSuggestionStatus.FAILED, f"Expected one failed press, got {records}"
    assert failure.error == type(error).__name__ and "stack details" not in failure.message, f"Unexpected failure {failure}"
    assert records[0].ran_suggester and records[0].text is None, f"A failed run is timed and has no text, got {records[0]}"


def test_app_prompt_suggestion_outside_drawing_goes_to_the_next_round(tmp_path: Path, sketch_file: Path,
                                                                      prompt_suggester: StubPromptSuggester) -> None:
    controller = make_study_controller(tmp_path, prompt_suggester)
    shown = controller.request_prompt_suggestion(sketch_file, 0, False)
    controller.run_round(sketch_file, "upload", shown.text, n_candidates=1, prompt_suggestion_id=shown.suggestion_id)
    controller.run_round(sketch_file, "upload", "a lamp", n_candidates=1, prompt_suggestion_id="from-a-stale-tab")
    rounds = load_session(controller.session.folder).rounds

    # The first round holds the suggestion and its prompt came from it, and the second round names the stale ID as unavailable.
    assert [item.id for item in rounds[0].prompt_suggestions] == [shown.suggestion_id], f"Got {rounds[0].prompt_suggestions}"
    suggested = PromptOrigin(kind=PromptOriginKind.SUGGESTED, prompt_suggestion=shown.suggestion_id)
    assert rounds[0].prompt_origin == suggested, f"Expected a suggested prompt, got {rounds[0].prompt_origin}"
    assert rounds[1].prompt_suggestions == () and rounds[1].prompt_origin == Unavailable(OUTSIDE_SESSION_REASON), "Bad round 2"


def test_app_refuses_an_unknown_condition_before_saving(controller: AppController) -> None:
    with pytest.raises(SketchLoopError, match="Study condition must be one of ai, no_ai"):
        controller.start_drawing("p1_car", "maybe")

    # Nothing was created, and no drawing is open.
    assert not controller.store.root.exists() and controller.latest_drawing() is None, "A bad condition must create nothing"


def test_app_describes_prompt_suggester_and_stand_ins(tmp_path: Path) -> None:
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), suggester=TemplateSuggester())
    real = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), stroke_suggester=EmptySuggester(),
                         suggester=FlorenceSuggester(tmp_path, captioner=lambda payload: "a lamp"))

    # The templates say they use no model, and Florence-2 is named by its repository.
    description = controller.describe_prompt_suggester()
    assert description == "Prompt suggestions: fixed style templates, no model.", f"Unexpected description {description!r}"
    florence = real.describe_prompt_suggester()
    assert florence.startswith("Prompt suggestions: florence-community/Florence-2-base@"), f"Unexpected {florence!r}"

    # Both stand-ins are warned about, and the fake stroke suggester is still one with Florence-2 prompts.
    warnings = controller.list_stand_in_warnings()
    assert len(warnings) == 2 and "fake stroke suggester" in warnings[0], f"Unexpected warnings {warnings}"
    assert "templates" in warnings[1], f"Expected a template warning, got {warnings}"
    assert len(real.list_stand_in_warnings()) == 1, f"Expected one warning, got {real.list_stand_in_warnings()}"
