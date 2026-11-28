"""
Tests of the browser page's handlers without starting a server.
"""

import cv2
import numpy as np
import pytest
import re
import socket

from datetime import datetime
from pathlib import Path

from sketchloop.app import (NO_FRAME_MESSAGE, NO_NEW_STROKES_MESSAGE, NO_SKETCH_MESSAGE, SUGGESTIONS_OFF_MESSAGE, AppController,
                            DrawingView, SuggestionView)
from sketchloop.backends.fake import FakeGenerator
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggestionError, StrokeSuggestionOutput, StrokeSuggestionRequest
from sketchloop.domain import Rotation, Unavailable
from sketchloop.experiments import (EXPERIMENT_NAME_RULE, NOT_STREAMED_REASON, FolderStore, PromptOriginKind, StoreError,
                                    StudyCondition, load_session)


@pytest.fixture
def controller(tmp_path: Path) -> AppController:
    """
    Provide a controller with the deterministic fake backend and a runs folder under tmp_path.
    """
    return AppController(FakeGenerator(), FolderStore(tmp_path / "runs"))


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make every non-loopback socket connection fail and disable Gradio analytics, so a build that needs the network fails.
    """
    connect = socket.socket.connect

    def refuse_remote_connection(sock: socket.socket, address: object) -> None:
        """
        Refuse a connection unless it targets loopback, which the Windows asyncio event loop uses internally.
        """
        # Allow loopback, since asyncio builds its self-pipe with a local socket pair.
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1", "localhost"):
            return connect(sock, address)

        raise OSError(f"Network access is blocked in this test, refused {address!r}.")

    monkeypatch.setattr(socket.socket, "connect", refuse_remote_connection)
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")


@pytest.mark.usefixtures("no_network")
def test_ui_builds_offline(controller: AppController, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The page builds offline without launching or naming a goal, analytics are off, and Generate waits for a sketch and prompt.
    """
    import gradio

    from sketchloop import ui

    def refuse_launch(*args: object, **kwargs: object) -> None:
        """
        Fail the test if the app is launched.
        """
        raise AssertionError("Building the page must not launch it")

    monkeypatch.setattr(gradio.Blocks, "launch", refuse_launch)
    app = ui.SketchPage(controller, default_candidates=2, default_seed=None).build()

    # The app is a Gradio Blocks object with analytics disabled.
    assert isinstance(app, gradio.Blocks), f"Expected gr.Blocks, got {type(app)}"
    assert app.analytics_enabled is False, "The UI must be built with analytics disabled"

    # Generate is enabled only when the chosen source has a sketch and the prompt is not blank.
    readiness = [ui.can_generate("webcam", True, None, "chair"), ui.can_generate("upload", True, None, "chair"),
                 ui.can_generate("webcam", True, None, "  "), ui.can_generate("upload", False, "b.png", "chair")]
    assert readiness == [True, False, False, True], f"Unexpected Generate readiness {readiness}"

    # The raw sketch always gets rotation 0, which the controller accepts, while each source keeps its default otherwise.
    rotations = [ui.pick_rotation("webcam", False), ui.pick_rotation("upload", False), ui.pick_rotation("webcam", True)]
    assert rotations == [180, 0, 0], f"Unexpected rotations {rotations}"

    # The stylesheet turns the preview for every rotation but none, and the script starts the webcam on load by itself.
    stylesheet = ui.STYLESHEET_PATH.read_text(encoding="utf-8")
    missing = [rotation for rotation in Rotation if rotation and f"#webcam.rotate-{rotation} " not in stylesheet]
    assert not missing, f"ui.css has no preview rule for rotations {missing}"
    assert "\nwindow.sketchloop.watchWebcam();" in ui.SCRIPT_PATH.read_text(encoding="utf-8"), "ui.js must watch the webcam"

    # The experimenter names the experiment with a code, and the participant has the drawing buttons.
    labels = [str(getattr(block, "label", "") or "") for block in app.blocks.values()]
    buttons = [block.value for block in app.blocks.values() if isinstance(block, gradio.Button)]
    hints = [block.info for block in app.blocks.values() if isinstance(block, gradio.Textbox)]
    assert "Experiment name" in labels and ui.EXPERIMENT_NAME_HINT in hints, f"Missing the name box or its hint in {labels}"
    missing = {"Start drawing", "Suggest next strokes", "Finish drawing"} - set(buttons)
    assert not missing, f"Missing buttons {missing}"

    # No component asks for a goal or a task, since the system is never told one.
    named = [label for label in labels if re.search(r"\b(goal|task)\b", label, re.IGNORECASE)]
    assert not named, f"The page must not name a goal or task, found {named}"


@pytest.mark.usefixtures("no_network")
def test_page_handlers_keep_frame_and_record_its_age(controller: AppController, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Handlers refuse a missing or stale webcam frame, keep a BGR copy, and record a webcam frame's age but not an upload's.
    """
    import gradio

    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=2, default_seed=None)
    clock_seconds = [10.0]
    monkeypatch.setattr(ui.time, "monotonic", lambda: clock_seconds[0])

    # Without a frame, Generate shows the controller's no-sketch message.
    with pytest.raises(gradio.Error, match=re.escape(NO_SKETCH_MESSAGE)):
        page.run_generation("webcam", None, "chair", None, 2, None, 0, False, progress=lambda *args, **kwargs: None)

    # A streamed RGB frame is kept in BGR order and enables Generate.
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[..., 0] = 255
    button = page.keep_frame(frame, "webcam", None, "chair")
    kept = page.latest_frame
    assert kept[0, 0].tolist() == [0, 0, 255], f"Expected the frame in BGR order, got {kept[0, 0].tolist()}"
    assert button.interactive is True, "Generate must be enabled once a frame and prompt exist"

    # A frame from before the stream stopped is not taken, disables Generate, and is refused with its own message.
    page.latest_frame_time -= ui.STALE_FRAME_INTERVALS * page.capture_interval_seconds + 1
    assert page.take_fresh_frame() is None, "A stale frame must not be taken"
    assert page.update_generate_button("webcam", None, "chair").interactive is False, "A stale frame must disable Generate"
    with pytest.raises(gradio.Error, match=re.escape(ui.STALE_FRAME_MESSAGE)):
        page.run_generation("webcam", None, "chair", None, 2, None, 0, False, progress=lambda *args, **kwargs: None)

    # A fresh frame kept at 10 s and generated from at 10.4 s is 0.4 s old, in the record and in the taken frame.
    page_sketch = np.full((600, 900, 3), 210, dtype=np.uint8)
    cv2.rectangle(page_sketch, (500, 100), (700, 200), (30, 30, 30), 4)
    page.keep_frame(page_sketch, "webcam", None, "chair")
    clock_seconds[0] = 10.4
    taken = page.take_fresh_frame()
    assert taken is not None and taken[1] == pytest.approx(0.4), f"Expected a fresh frame aged 0.4 s, got {taken}"
    page.run_generation("webcam", None, "chair", None, 1, 7, 0, False, progress=lambda *args, **kwargs: None)
    webcam_age = load_session(controller.session.folder).rounds[0].frame_age_seconds
    assert webcam_age == pytest.approx(0.4), f"Expected a recorded frame age of 0.4 s, got {webcam_age}"

    # An upload round passes no age, so it records that its sketch was not streamed.
    upload_path = tmp_path / "upload.png"
    upload_path.write_bytes(cv2.imencode(".png", page_sketch)[1].tobytes())
    page.run_generation("upload", str(upload_path), "lamp", None, 1, 7, 0, False, progress=lambda *args, **kwargs: None)
    upload_age = load_session(controller.session.folder).rounds[1].frame_age_seconds
    assert upload_age == Unavailable(NOT_STREAMED_REASON), f"Expected the not-streamed reason, got {upload_age}"


def make_drawing_view(*, is_open: bool, suggestion_lines: tuple[str, ...] = (), condition: StudyCondition | None = None,
                      latest_suggestion_failed: bool = False) -> DrawingView:
    """
    Build a drawing view of experiment p1 with the given state, suggestion lines, study condition, and failure flag.
    """
    return DrawingView(number=1, experiment="p1", is_open=is_open, capture_count=1, suggestion_count=len(suggestion_lines),
                       suggestion_lines=suggestion_lines, condition=condition,
                       suggestions_enabled=condition != StudyCondition.NO_AI, started_at=datetime(2026, 11, 22, 10, 2),
                       latest_suggestion_failed=latest_suggestion_failed)


def make_suggestion_view(*, status: str, is_fake: bool) -> SuggestionView:
    """
    Build a suggestion view with the given status line and fake flag.
    """
    return SuggestionView(drawing=1, number=1, part_count=2, colors=("#e6194b", "#3cb44b"), is_fake=is_fake,
                          generation_seconds=0.1, status=status)


@pytest.mark.usefixtures("no_network")
def test_drawing_status_and_buttons_follow_the_drawing() -> None:
    """
    Start waits for a name, the drawing buttons and status follow the drawing, and an upload's capture key names its file.
    """
    from sketchloop import ui

    open_view = make_drawing_view(is_open=True)

    # Start drawing is enabled only for a name that is not blank, and never while a drawing is open.
    readiness = [ui.can_start_drawing("p1", False), ui.can_start_drawing("  ", False), ui.can_start_drawing(None, False),
                 ui.can_start_drawing("p1", True)]
    assert readiness == [True, False, False, False], f"Unexpected Start drawing readiness {readiness}"

    # Without an open drawing, Suggest and Finish are disabled, while a new session and a new name are allowed.
    for view in (None, make_drawing_view(is_open=False)):
        status, suggest, finish, new_session, name_box, _ = ui.build_drawing_outputs(view)
        assert suggest.interactive is False and finish.interactive is False, f"Buttons must be disabled for {view}"
        assert new_session.interactive is True and name_box.interactive is True, f"New session must be allowed for {view}"
    assert ui.build_drawing_outputs(None)[0] == ui.NO_DRAWING_STATUS, "No drawing must show how to start one"

    # An open drawing enables Suggest and Finish, locks a new session and the name box, and names the experiment.
    status, suggest, finish, new_session, name_box, sketch_view = ui.build_drawing_outputs(open_view)
    assert suggest.interactive is True and finish.interactive is True, "An open drawing must enable Suggest and Finish"
    assert new_session.interactive is False, "Start a new session must be disabled while a drawing is open"
    assert name_box.interactive is False and name_box.info == ui.FINISH_FIRST_MESSAGE, "The name box must say to finish first"
    assert "p1" in status and "drawing 1" in status, f"The status must name the experiment and drawing, got {status!r}"

    # The open drawing's preview is taller, and its label names suggested strokes only once there are some.
    with_suggestion = ui.style_sketch_view(make_drawing_view(is_open=True, suggestion_lines=("Suggestion 1: 2 colored parts",)))
    assert sketch_view.height == ui.GUIDED_PREVIEW_HEIGHT, f"Expected a taller guided preview, got {sketch_view.height}"
    labels = [sketch_view.label, with_suggestion.label]
    assert labels == [ui.OPEN_DRAWING_LABEL, ui.GUIDED_PREVIEW_LABEL], f"Unexpected preview labels {labels}"

    # A suggestion's line follows, a fake one is labeled, and the no-new-strokes message is passed through.
    fake = ui.describe_drawing_status(open_view, make_suggestion_view(status="Suggestion 1: 2 colored parts", is_fake=True))
    empty = ui.describe_drawing_status(open_view, make_suggestion_view(status=NO_NEW_STROKES_MESSAGE, is_fake=False))
    assert fake.startswith(ui.FAKE_SUGGESTER_PREFIX) and fake.endswith("Suggestion 1: 2 colored parts"), f"Bad {fake!r}"
    assert empty.endswith(NO_NEW_STROKES_MESSAGE) and ui.FAKE_SUGGESTER_PREFIX not in empty, f"Bad {empty!r}"

    # After a reload the latest suggestion line is shown, and a finished drawing says how many suggestions it had.
    restored = ui.describe_drawing_status(make_drawing_view(is_open=True, suggestion_lines=("Suggestion 1: 2 colored parts",)))
    finished = ui.describe_drawing_status(make_drawing_view(is_open=False, suggestion_lines=("Suggestion 1: 2 colored parts",)))
    assert restored.endswith("Suggestion 1: 2 colored parts"), f"A reload must show the latest suggestion, got {restored!r}"
    assert "finished" in finished and "1 suggestion" in finished, f"Unexpected finished status {finished!r}"

    drawing = make_drawing_view(is_open=True)
    other = DrawingView(number=1, experiment="p2", is_open=True, capture_count=0, suggestion_count=0, suggestion_lines=())

    # The same file in the same drawing gives the same key, and another file or experiment gives another.
    keys = [ui.name_upload_capture(drawing, "upload", "a.png"), ui.name_upload_capture(drawing, "upload", "b.png"),
            ui.name_upload_capture(other, "upload", "a.png")]
    assert keys[0] == ui.name_upload_capture(drawing, "upload", "a.png"), "The same upload must keep its key"
    assert len(set(keys)) == 3, f"Another file or experiment must change the key, got {keys}"

    # A webcam frame and a missing upload have no key.
    assert ui.name_upload_capture(drawing, "webcam", "a.png") is None, "A webcam frame must have no upload key"
    assert ui.name_upload_capture(drawing, "upload", None) is None, "A missing upload must have no key"


@pytest.mark.usefixtures("no_network")
def test_page_starts_suggests_and_finishes_a_drawing(controller: AppController, sheet_frame: np.ndarray,
                                                     tmp_path: Path) -> None:
    """
    The page handlers start a named drawing, suggest from the webcam frame with an empty prompt, show the overlay, and finish.
    """
    import gradio

    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=1, default_seed=None)

    # An invalid name is refused with the naming rule, and nothing is started.
    with pytest.raises(gradio.Error, match=re.escape(EXPERIMENT_NAME_RULE)):
        page.start_drawing("has space")
    assert controller.latest_drawing() is None, "An invalid name must not start a drawing"

    # A valid name starts the drawing, enables Suggest, and clears the name, prompt, its ID, and upload so nothing carries over.
    started = page.start_drawing(" participant1_car ")
    status, suggest, finish, new_session, name_box, sketch_view, start_button, prompt_box, prompt_id, upload_box, *_ = started
    assert name_box.value == "" and start_button.interactive is False, "Starting must clear the name and disable Start"
    assert prompt_box.value == "" and upload_box.value is None, "Starting must clear the previous prompt and upload"
    assert prompt_id is None, f"Starting must forget the kept prompt suggestion ID, got {prompt_id!r}"
    assert suggest.interactive is True and "participant1_car" in status, f"Unexpected start status {status!r}"
    assert (tmp_path / "runs" / "participant1_car").is_dir(), "The experiment's folder must be named after it"

    # While the drawing is open, Start drawing stays disabled and a second start is refused.
    assert page.update_start_button("p2").interactive is False, "Start drawing must stay disabled while a drawing is open"
    with pytest.raises(gradio.Error, match=re.escape(ui.FINISH_FIRST_MESSAGE)):
        page.start_drawing("p2")

    # Finishing without a webcam frame is refused, so the drawing never ends without its final capture.
    with pytest.raises(gradio.Error, match=re.escape(NO_FRAME_MESSAGE)):
        page.finish_drawing("webcam", None, 0)
    assert controller.latest_drawing().is_open, "A refused finish must leave the drawing open"

    # Suggest uses the empty prompt as typed and shows the fake suggestion's status and the canvas-sized overlay preview.
    page.keep_frame(np.ascontiguousarray(sheet_frame[:, :, ::-1]), "webcam", None, "")
    waiting, disabled = page.show_suggesting()
    assert waiting == ui.SUGGESTING_STATUS and disabled.interactive is False, "A running suggestion must say so and lock"
    status, sketch_view = page.suggest_next_strokes("webcam", None, "", None, 0)
    assert status.startswith(ui.FAKE_SUGGESTER_PREFIX) and "Suggestion 1: 2 colored parts" in status, f"Bad status {status!r}"
    assert sketch_view.label == ui.GUIDED_PREVIEW_LABEL and sketch_view.value, f"Bad sketch view {sketch_view.label!r}"
    recorded_prompt = controller.session.drawings[0].record.suggestions[0].request.prompt
    assert recorded_prompt == "", f"The empty prompt must be sent as typed, got {recorded_prompt!r}"

    # The timer shows the guided preview while the drawing is open.
    refreshed = page.refresh_webcam_preview("webcam", None, 0, False)
    assert refreshed.shape[:2] == (360, 512), f"The timer must show the guided preview, got {refreshed.shape}"

    # After the suggestion its button is enabled again, and a failure would show the drawing's own status line.
    assert page.update_suggest_button().interactive is True, "Suggest must be enabled again while the drawing is open"
    assert page.show_drawing_status().endswith("Suggestion 1: 2 colored parts"), "Status must show the latest suggestion"

    # Finish keeps the frame as the final capture, disables Suggest and Finish, and allows a new session again.
    status, suggest, finish, new_session, *_ = page.finish_drawing("webcam", None, 0)
    assert suggest.interactive is False and finish.interactive is False, "A finished drawing must disable its buttons"
    assert new_session.interactive is True, "A finished drawing must allow a new session"
    assert "finished" in status and controller.session.drawings[0].record.final_capture is not None, f"Bad finish {status!r}"


@pytest.mark.usefixtures("no_network")
def test_page_saves_drawing_captures_quietly(sheet_frame: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The capture timer saves the open drawing's frames, skips stale frames and repeated uploads, and shows a failure once.
    """
    import gradio

    from sketchloop import ui

    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), save_interval_seconds=0.001)
    page = ui.SketchPage(controller, default_candidates=1, default_seed=None)

    def captures() -> int:
        """
        Count the first drawing's saved captures.
        """
        return len(controller.session.drawings[0].record.captures)

    # Without an open drawing, a tick saves nothing and changes nothing on the page.
    assert page.save_drawing_capture("webcam", None, 0, "") == gradio.skip(), "A tick without a drawing must change nothing"
    assert controller.latest_drawing() is None, "A tick must not start a drawing"

    # While a drawing is open, each tick saves the fresh webcam frame as an interval JPEG.
    page.start_drawing("p1")
    page.keep_frame(np.ascontiguousarray(sheet_frame[:, :, ::-1]), "webcam", None, "")
    for _ in range(2):
        assert page.save_drawing_capture("webcam", None, 0, "") == gradio.skip(), "A saved capture must change nothing"
    jpegs = list((tmp_path / "runs" / "p1").rglob("*.jpg"))
    assert captures() == 2 and jpegs, f"Expected 2 captures with JPEG files, got {captures()} and {jpegs}"

    # A frame from before the stream stopped is skipped without an error.
    page.latest_frame_time -= ui.STALE_FRAME_INTERVALS * page.capture_interval_seconds + 1
    assert page.save_drawing_capture("webcam", None, 0, "") == gradio.skip(), "A stale frame must be skipped quietly"
    assert captures() == 2, f"A stale frame must not be saved, got {captures()} captures"

    # An upload is saved once per drawing, and again only when another file is uploaded.
    first_upload, second_upload = tmp_path / "first.png", tmp_path / "second.png"
    for path in (first_upload, second_upload):
        path.write_bytes(cv2.imencode(".png", sheet_frame)[1].tobytes())
    for path in (first_upload, first_upload, second_upload):
        page.save_drawing_capture("upload", str(path), 0, "")
    assert captures() == 4, f"Expected one capture per uploaded file, got {captures() - 2}"

    # A failed save shows one calm notice, keeps it without updating on later ticks, and clears it after a save works.
    def fail_to_save(*args: object, **kwargs: object) -> None:
        """
        Fail like a full disk.
        """
        raise StoreError("Could not save drawing.json. Free disk space and retry.")

    monkeypatch.setattr(controller, "record_drawing_capture", fail_to_save)
    page.keep_frame(np.ascontiguousarray(sheet_frame[:, :, ::-1]), "webcam", None, "")
    notice = page.save_drawing_capture("webcam", None, 0, "")
    assert notice.visible is True and "Free disk space" in notice.value, f"A failed save must show its line, got {notice}"
    assert page.save_drawing_capture("webcam", None, 0, notice.value) == gradio.skip(), "The notice must not repeat"
    monkeypatch.undo()
    cleared = page.save_drawing_capture("webcam", None, 0, notice.value)
    assert cleared.visible is False and cleared.value == "", f"A working save must clear the notice, got {cleared}"

    # After Finish, ticks save nothing more.
    page.finish_drawing("webcam", None, 0)
    finished = captures()
    page.save_drawing_capture("webcam", None, 0, "")
    assert captures() == finished, "A finished drawing must not get interval captures"


@pytest.mark.usefixtures("no_network")
def test_page_sends_the_kept_prompt_suggestion_id(controller: AppController, sheet_frame: np.ndarray) -> None:
    """
    The generation page keeps the ID of the prompt suggestion it wrote, so stroke suggestions, finish, and rounds record origins.
    """
    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=1, default_seed=None)
    page.start_drawing("p1_car")
    page.keep_frame(np.ascontiguousarray(sheet_frame[:, :, ::-1]), "webcam", None, "")

    # Suggest a prompt returns the text for the box and the ID the tab keeps.
    text, suggestion_id = page.suggest_another("webcam", None, 0, False)
    assert text and suggestion_id, f"Expected a prompt and its ID, got {text!r} and {suggestion_id!r}"

    # A stroke suggestion with the unchanged text records suggested, and one with an edited text records edited.
    page.suggest_next_strokes("webcam", None, text, suggestion_id, 0)
    page.suggest_next_strokes("webcam", None, f"{text}, low and wide", suggestion_id, 0)
    origins = [suggestion.prompt_origin for suggestion in controller.session.drawings[0].record.suggestions]
    kinds = [origin.kind for origin in origins]
    assert kinds == [PromptOriginKind.SUGGESTED, PromptOriginKind.EDITED], f"Unexpected stroke suggestion origins {kinds}"
    assert {origin.prompt_suggestion for origin in origins} == {suggestion_id}, f"Origins must name the kept ID, got {origins}"

    # Finish sends the prompt box text and the kept ID as the final prompt.
    page.finish_drawing("webcam", None, 0, f"{text}, low and wide", suggestion_id)
    record = controller.session.drawings[0].record
    assert record.final_prompt == f"{text}, low and wide", f"Finish must keep the prompt box text, got {record.final_prompt!r}"
    assert record.final_prompt_origin.kind == PromptOriginKind.EDITED, f"Unexpected final origin {record.final_prompt_origin}"

    # Generate sends the kept ID too, so a round from an unchanged suggestion records suggested.
    text, suggestion_id = page.suggest_another("webcam", None, 0, False)
    page.run_generation("webcam", None, text, suggestion_id, 1, 7, 0, False, progress=lambda *args, **kwargs: None)
    origin = load_session(controller.session.folder).rounds[0].prompt_origin
    assert origin.kind == PromptOriginKind.SUGGESTED and origin.prompt_suggestion == suggestion_id, f"Bad round origin {origin}"

    # A new session empties the prompt box together with its kept ID, so a later typed prompt is never taken as suggested.
    *_, prompt_box, prompt_id = page.start_session()
    assert prompt_box.value == "" and prompt_id is None, f"A new session must clear the prompt and its ID, got {prompt_box}"


class FailingStrokeSuggester(FakeStrokeSuggester):
    """
    Test-only fake suggester whose suggestion fails as if the device ran out of memory.
    """

    def suggest(self, request: StrokeSuggestionRequest, canvas_payload: bytes) -> StrokeSuggestionOutput:
        """
        Raise the error a real suggester raises when generation fails.
        """
        raise StrokeSuggestionError("The device ran out of memory. Close other apps and retry.")


def test_study_status_and_controls_follow_the_condition_and_drawing() -> None:
    """
    The participant's status hides the name and condition, the experimenter's names both, and the controls follow the drawing.
    """
    from sketchloop import ui

    with_ai = make_drawing_view(is_open=True, condition=StudyCondition.AI)
    without_ai = make_drawing_view(is_open=True, condition=StudyCondition.NO_AI)

    # The participant sees the same words in both conditions, before, during, and after a drawing, without name or condition.
    open_lines = {ui.describe_participant_status(with_ai), ui.describe_participant_status(without_ai)}
    assert open_lines == {ui.DRAWING_STATUS}, f"An open drawing must read the same in both conditions, got {open_lines}"
    finished = ui.describe_participant_status(make_drawing_view(is_open=False, condition=StudyCondition.NO_AI))
    assert [ui.describe_participant_status(None), finished] == [ui.WAITING_STATUS, ui.FINISHED_STATUS], f"Bad {finished!r}"
    leaks = [line for line in (*open_lines, finished) if re.search(r"p1|\bAI\b", line)]
    assert not leaks, f"The participant status must not name the experiment or condition, got {leaks}"

    # A suggestion's line comes first, with the fake warning, and the reminder to say when done stays below it.
    fake_suggestion = make_suggestion_view(status="Suggestion 1: 2 colored parts", is_fake=True)
    suggested = ui.describe_participant_status(with_ai, fake_suggestion)
    expected = f"{ui.FAKE_SUGGESTER_PREFIX} Suggestion 1: 2 colored parts\n\n{ui.DONE_REMINDER}"
    assert suggested == expected, f"Unexpected suggestion status {suggested!r}"

    # A failed suggestion is worded calmly for the participant, and its full line goes to the experimenter.
    failed = make_drawing_view(is_open=True, suggestion_lines=("Suggestion 1: failed during generation: out of memory",),
                               condition=StudyCondition.AI, latest_suggestion_failed=True)
    participant = ui.describe_participant_status(failed)
    assert participant.startswith(ui.SUGGESTION_FAILED_STATUS) and "memory" not in participant, f"Bad {participant!r}"
    assert ui.describe_experimenter_status(failed).endswith("out of memory"), "The experimenter must see the failure"

    # The experimenter sees the name, condition, start time, and state.
    experimenter = ui.describe_experimenter_status(without_ai)
    assert experimenter == "p1, without AI, started 10:02, drawing open", f"Unexpected experimenter status {experimenter!r}"
    assert ui.describe_experimenter_status(None) == ui.NO_EXPERIMENT_STATUS, "No drawing must say how to start one"

    # Start is enabled only with a name and a condition and no open drawing, and it repeats the chosen condition.
    readiness = [ui.can_start_study_drawing("p1", "ai", False), ui.can_start_study_drawing("p1", None, False),
                 ui.can_start_study_drawing(" ", "no_ai", False), ui.can_start_study_drawing("p1", "ai", True)]
    assert readiness == [True, False, False, False], f"Unexpected study Start readiness {readiness}"
    labels = [ui.label_start_button("ai"), ui.label_start_button("no_ai"), ui.label_start_button(None)]
    assert labels == ["Start with AI", "Start without AI", ui.START_LABEL], f"Unexpected Start labels {labels}"

    # The two AI buttons are visible only in an open drawing with AI, and removed, not disabled, everywhere else.
    views = [make_drawing_view(is_open=True, condition=StudyCondition.AI),
             make_drawing_view(is_open=True, condition=StudyCondition.NO_AI),
             make_drawing_view(is_open=False, condition=StudyCondition.AI), None]
    visible = [[button.visible for button in ui.build_study_drawing_outputs(view)[1:3]] for view in views]
    assert visible == [[True, True], [False, False], [False, False], [False, False]], f"Unexpected AI buttons {visible}"

    # After a start the name and condition are emptied and locked, and Finish is enabled under its own label.
    _, _, _, name_box, condition, finish, _ = ui.build_study_drawing_outputs(views[1], clear_inputs=True)
    assert name_box.value == "" and condition.value is None, "A start must clear the name and condition"
    assert name_box.interactive is False and condition.interactive is False, "An open drawing must lock the start inputs"
    assert finish.interactive is True and finish.value == ui.FINISH_LABEL, f"An open drawing must enable Finish, got {finish}"

    # A second press finishes only within the confirmation window.
    armed = [ui.is_finish_armed(None, 10.0), ui.is_finish_armed(10.0, 14.0), ui.is_finish_armed(10.0, 16.0)]
    assert armed == [False, True, False], f"Unexpected Finish confirmation {armed}"

    # The panel asks for a check in neutral words and is marked so the stylesheet can flag it.
    calm, flagged = ui.style_experimenter_panel(False), ui.style_experimenter_panel(True)
    assert (calm.label, calm.elem_classes) == (ui.EXPERIMENTER_LABEL, []), f"Unexpected calm panel {calm}"
    assert (flagged.label, flagged.elem_classes) == (ui.CHECK_NEEDED_LABEL, ["needs-check"]), f"Unexpected panel {flagged}"
    assert "failed" not in ui.CHECK_NEEDED_LABEL.lower(), "The label the participant may see must not alarm"


@pytest.mark.usefixtures("no_network")
def test_study_page_builds_only_the_participant_and_experimenter_controls(controller: AppController) -> None:
    """
    The study page builds no generation components, its header names no model, and its prompt placeholder names no object.
    """
    import gradio

    from sketchloop import ui

    app = ui.SketchPage(controller, default_candidates=1, default_seed=None, study=True).build()
    blocks = list(app.blocks.values())
    buttons = {block.value: block for block in blocks if isinstance(block, gradio.Button)}

    # Only the two AI buttons and Start and Finish exist, the AI buttons start removed, and Finish is marked as stopping.
    expected = {"Suggest a prompt", "Suggest next strokes", ui.START_LABEL, ui.FINISH_LABEL}
    assert set(buttons) == expected, f"Unexpected study buttons {set(buttons)}"
    ai_buttons = [buttons[name].visible for name in ("Suggest a prompt", "Suggest next strokes")]
    assert ai_buttons == [False, False], f"The AI buttons must start removed, got {ai_buttons}"
    assert buttons[ui.FINISH_LABEL].variant == "stop", f"Finish must look like a stop, got {buttons[ui.FINISH_LABEL].variant}"

    # No gallery, choice, sliders, seed, history, or raw option is built.
    generation_types = (gradio.Gallery, gradio.CheckboxGroup, gradio.Slider, gradio.Dataframe, gradio.Checkbox)
    built = [type(block).__name__ for block in blocks if isinstance(block, generation_types)]
    labels = [str(getattr(block, "label", "") or "") for block in blocks]
    assert not built and "Seed" not in labels, f"The study page must not build generation components, found {built}"

    # The header is the title only, and the panel names the prompt suggester and says to use the participant's tab.
    texts = [block.value or "" for block in blocks if isinstance(block, gradio.Markdown)]
    assert "# SketchLoop" in texts and controller.describe_backend() not in texts[0], f"Unexpected header {texts[0]!r}"
    assert any(controller.describe_prompt_suggester() in text for text in texts), "The panel must name the prompt suggester"
    assert ui.SAME_TAB_NOTE in texts, "The panel must say to start and finish in the participant's tab"

    # The prompt names no object, the condition has no default, and the panel starts open so the webcam can start.
    prompts = [block for block in blocks if isinstance(block, gradio.Textbox) and block.label == "Prompt"]
    assert [box.placeholder for box in prompts] == [ui.STUDY_PROMPT_PLACEHOLDER], f"Unexpected prompt boxes {prompts}"
    conditions = [block for block in blocks if isinstance(block, gradio.Radio) and block.label == "Condition"]
    assert len(conditions) == 1 and conditions[0].value is None, "The condition must have no default"
    panels = [block for block in blocks if isinstance(block, gradio.Accordion)]
    assert [(panel.label, panel.open) for panel in panels] == [(ui.EXPERIMENTER_LABEL, True)], f"Unexpected panels {panels}"


@pytest.mark.usefixtures("no_network")
def test_study_page_runs_a_no_ai_and_an_ai_drawing(controller: AppController, sheet_frame: np.ndarray,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The study handlers start each condition, clear the inputs, close the panel, refuse AI help without AI, and keep final prompts.
    """
    import gradio

    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=1, default_seed=None, study=True)
    frame = np.ascontiguousarray(sheet_frame[:, :, ::-1])
    clock_seconds = [100.0]
    monkeypatch.setattr(ui.time, "monotonic", lambda: clock_seconds[0])
    page.store_frame(frame)

    # A start without a condition is refused, and nothing is started.
    with pytest.raises(gradio.Error, match=re.escape(ui.CHOOSE_CONDITION_MESSAGE)):
        page.start_study_drawing("p1_car", None)
    assert controller.latest_drawing() is None, "A start without a condition must not start a drawing"

    # Without AI, the buttons are removed, the inputs and prompt are cleared, and the panel closes.
    started = page.start_study_drawing("p1_car", "no_ai")
    status, suggest_prompt, suggest_strokes, name_box, condition, _, _, start, sketch_view, prompt, prompt_id, *_ = started
    panel, shown, armed = started[-3:]
    assert [suggest_prompt.visible, suggest_strokes.visible] == [False, False], "Without AI the AI buttons must be removed"
    assert name_box.value == "" and condition.value is None and start.interactive is False, "A start must clear its inputs"
    assert start.value == ui.START_LABEL, f"Start must go back to its plain label, got {start.value!r}"
    assert prompt.value == "" and prompt_id is None and sketch_view.value is None, "A start must clear the prompt and view"
    assert panel.open is False and status == ui.DRAWING_STATUS, f"A start must close the panel, got {panel.open} and {status!r}"
    assert shown == ("p1_car", 1, True, 0) and armed is None, f"A start must record what the tab shows, got {shown}"

    # Prompt suggestions are refused without AI.
    with pytest.raises(gradio.Error, match=re.escape(SUGGESTIONS_OFF_MESSAGE)):
        page.suggest_another("webcam", None, 0)

    # A first Finish press only asks for a second, and the timer turns it back after the window.
    first = page.press_finish("webcam", None, 0, "a lamp", None, None)
    assert first[5].value == ui.FINISH_CONFIRM_LABEL and first[-1] == 100.0, f"A first press must arm Finish, got {first[5]}"
    assert controller.latest_drawing().is_open, "A first press must not finish the drawing"
    clock_seconds[0] = 106.0
    reset_button, reset_armed = page.disarm_finish(100.0)
    assert reset_button.value == ui.FINISH_LABEL and reset_armed is None, "An unconfirmed press must turn back"

    # Two presses within the window finish with the typed prompt as typed.
    page.store_frame(frame)
    armed_at = page.press_finish("webcam", None, 0, "a lamp", None, None)[-1]
    clock_seconds[0] = 108.0
    finished = page.press_finish("webcam", None, 0, "a lamp", None, armed_at)
    record = controller.session.drawings[0].record
    assert finished[0] == ui.FINISHED_STATUS and finished[7].value is None, "Finish must thank and empty the view"
    assert record.final_prompt == "a lamp" and record.final_prompt_origin.kind == PromptOriginKind.TYPED, "Bad final prompt"

    # With AI, the buttons show, and a suggested prompt kept unchanged until Finish records suggested.
    started = page.start_study_drawing("p2_car", "ai")
    assert [started[1].visible, started[2].visible] == [True, True], "With AI the AI buttons must show"
    text, suggestion_id = page.suggest_another("webcam", None, 0)
    armed_at = page.press_finish("webcam", None, 0, text, suggestion_id, None)[-1]
    page.press_finish("webcam", None, 0, text, suggestion_id, armed_at)
    origin = controller.session.drawings[0].record.final_prompt_origin
    assert origin.kind == PromptOriginKind.SUGGESTED and origin.prompt_suggestion == suggestion_id, f"Bad final origin {origin}"


@pytest.mark.usefixtures("no_network")
def test_study_save_timer_follows_tabs_and_flags_problems(controller: AppController, sheet_frame: np.ndarray,
                                                                monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    """
    The save timer follows other tabs, flags a stopped webcam or failed save once, and clears the flag on recovery.
    """
    import gradio

    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=1, default_seed=None, study=True)
    frame = np.ascontiguousarray(sheet_frame[:, :, ::-1])
    page.store_frame(frame)
    page.start_study_drawing("p1_car", "ai")

    # A tab that has not shown the drawing started elsewhere shows it at the next tick, with its buttons, and only once.
    _, _, status, suggest_prompt, suggest_strokes, shown = page.save_study_capture("webcam", None, 0, "", None)
    assert status == ui.DRAWING_STATUS and suggest_prompt.visible and suggest_strokes.visible, f"Bad follow {status!r}"
    assert page.save_study_capture("webcam", None, 0, "", shown) == (gradio.skip(),) * 6, "A working tick must change nothing"

    # A reload during the drawing marks the status so the script closes the panel once the webcam streams.
    restored_status, *_, start, restored_shown = page.restore_study_view()
    assert restored_status.elem_classes == [ui.CLOSE_PANEL_CLASS] and restored_shown == shown, "A reload must mark the panel"
    assert start.interactive is False, "Start must wait for a name and condition after a reload"

    # A webcam that stopped sending frames flags the panel and prints one line.
    page.latest_frame_time -= ui.STALE_FRAME_INTERVALS * page.capture_interval_seconds + 1
    notice, panel, *rest = page.save_study_capture("webcam", None, 0, "", shown)
    assert notice.value == ui.CAMERA_STOPPED_NOTICE and panel.label == ui.CHECK_NEEDED_LABEL, f"Bad flag {notice}, {panel}"
    assert rest == [gradio.skip()] * 4, "A camera problem must not touch the participant's status"
    assert page.save_study_capture("webcam", None, 0, notice.value, shown)[:2] == (gradio.skip(),) * 2, "No repeat"
    assert capsys.readouterr().out.count("check needed") == 1, "The terminal must name the problem once"

    # Frames that return clear the flag and print the recovery.
    page.store_frame(frame)
    notice, panel, *_ = page.save_study_capture("webcam", None, 0, notice.value, shown)
    assert notice.visible is False and panel.label == ui.EXPERIMENTER_LABEL, f"Frames must clear the flag, got {panel}"
    assert "work again" in capsys.readouterr().out, "The terminal must say when it works again"

    # A failed save flags the panel with its own notice.
    def fail_to_save(*args: object, **kwargs: object) -> None:
        """
        Fail like a full disk.
        """
        raise StoreError("Could not save drawing.json. Free disk space and retry.")

    monkeypatch.setattr(controller, "record_drawing_capture", fail_to_save)
    notice, panel, *_ = page.save_study_capture("webcam", None, 0, "", shown)
    assert "Free disk space" in notice.value and panel.elem_classes == ["needs-check"], f"A failed save must flag, got {notice}"


@pytest.mark.usefixtures("no_network")
def test_study_failed_suggestions_are_calm_for_the_participant(tmp_path: Path, sheet_frame: np.ndarray) -> None:
    """
    Failed prompt and stroke suggestions give the participant a calm line, while the experimenter status keeps the error.
    """
    import gradio

    from sketchloop import ui

    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), stroke_suggester=FailingStrokeSuggester())
    page = ui.SketchPage(controller, default_candidates=1, default_seed=None, study=True)
    page.start_study_drawing("p1_car", "ai")

    # Without a frame a prompt suggestion fails, the prompt and its ID stay, and the participant reads the calm line.
    prompt, prompt_id, status, experimenter = page.suggest_study_prompt("webcam", None, 0)
    assert (prompt, prompt_id) == (gradio.skip(), gradio.skip()), "A failure must leave the prompt box and its ID as they were"
    assert status == f"{ui.PROMPT_FAILED_STATUS}\n\n{ui.DONE_REMINDER}", f"Unexpected participant status {status!r}"
    assert "Prompt suggestion failed: No sketch yet" in experimenter, f"Unexpected experimenter status {experimenter!r}"
    page.store_frame(np.ascontiguousarray(sheet_frame[:, :, ::-1]))

    # The participant reads the calm line and the reminder, and the experimenter reads the cause.
    status, sketch_view, experimenter, shown = page.suggest_study_strokes("webcam", None, "", None, 0)
    assert status == f"{ui.SUGGESTION_FAILED_STATUS}\n\n{ui.DONE_REMINDER}", f"Unexpected participant status {status!r}"
    assert "ran out of memory" in experimenter and shown == ("p1_car", 1, True, 1), f"Bad experimenter status {experimenter!r}"

    # After a reload the participant still sees the calm line.
    assert page.restore_study_view()[0].value.startswith(ui.SUGGESTION_FAILED_STATUS), "A reload must keep the calm line"
