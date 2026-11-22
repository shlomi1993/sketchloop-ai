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
    The page builds offline without launching, analytics are off, and Generate waits for a sketch and prompt.
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


@pytest.mark.usefixtures("no_network")
def test_page_handlers_keep_frame_and_refuse_missing_sketch(controller: AppController) -> None:
    """
    SketchPage handlers work without building the page, refusing a webcam round with no frame and keeping a BGR copy.
    """
    import gradio

    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=2, default_seed=None)

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


@pytest.mark.usefixtures("no_network")
def test_generation_records_webcam_frame_age(controller: AppController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A webcam round records how old its frame was when Generate ran, and an upload round records no frame age.
    """
    from sketchloop import ui

    page = ui.SketchPage(controller, default_candidates=1, default_seed=None)
    page_sketch = np.full((600, 900, 3), 210, dtype=np.uint8)
    cv2.rectangle(page_sketch, (500, 100), (700, 200), (30, 30, 30), 4)
    clock_seconds = [10.0]
    monkeypatch.setattr(ui.time, "monotonic", lambda: clock_seconds[0])

    # A frame kept at 10 s and generated from at 10.4 s is 0.4 s old, in the record and in the taken frame.
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


def make_drawing_view(*, is_open: bool, suggestion_lines: tuple[str, ...] = (),
                      condition: StudyCondition | None = None) -> DrawingView:
    """
    Build a drawing view of experiment p1 with the given state, suggestion lines, and study condition.
    """
    return DrawingView(number=1, experiment="p1", is_open=is_open, capture_count=1, suggestion_count=len(suggestion_lines),
                       suggestion_lines=suggestion_lines, condition=condition,
                       suggestions_enabled=condition != StudyCondition.NO_AI, started_at=datetime(2026, 11, 22, 10, 2))


def make_suggestion_view(*, status: str, is_fake: bool) -> SuggestionView:
    """
    Build a suggestion view with the given status line and fake flag.
    """
    return SuggestionView(drawing=1, number=1, part_count=2, colors=("#e6194b", "#3cb44b"), is_fake=is_fake,
                          generation_seconds=0.1, status=status)


@pytest.mark.usefixtures("no_network")
def test_drawing_status_and_buttons_follow_the_drawing() -> None:
    """
    Start drawing waits for a name and no open drawing, the drawing buttons follow the drawing, and the status names each state.
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
def test_page_has_experiment_controls_and_no_goal(controller: AppController) -> None:
    """
    The built page has the experiment name with its privacy hint and the drawing buttons, and nothing names a goal or task.
    """
    import gradio

    from sketchloop import ui

    app = ui.SketchPage(controller, default_candidates=1, default_seed=None).build()
    labels = [str(getattr(block, "label", "") or "") for block in app.blocks.values()]
    buttons = [block.value for block in app.blocks.values() if isinstance(block, gradio.Button)]
    hints = [block.info for block in app.blocks.values() if isinstance(block, gradio.Textbox)]

    # The experimenter names the experiment with a code, and the participant has the drawing buttons.
    assert "Experiment name" in labels and ui.EXPERIMENT_NAME_HINT in hints, f"Missing the name box or its hint in {labels}"
    missing = {"Start drawing", "Suggest next strokes", "Finish drawing"} - set(buttons)
    assert not missing, f"Missing buttons {missing}"

    # No component asks for a goal or a task, since the system is never told one.
    named = [label for label in labels if re.search(r"\b(goal|task)\b", label, re.IGNORECASE)]
    assert not named, f"The page must not name a goal or task, found {named}"


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


def test_upload_capture_key_names_the_drawing_and_file() -> None:
    """
    An upload's capture key changes with the drawing or the file, and a webcam frame has none, so it is always saved.
    """
    from sketchloop import ui

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

    # A new session forgets the kept ID along with the round display.
    assert page.start_session()[-1] is None, "Start a new session must forget the kept prompt suggestion ID"


def test_study_status_lines_hide_the_experiment_and_condition() -> None:
    """
    The participant's status is the same in both conditions and names neither, while the experimenter's status names both.
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

    # A suggestion's line replaces the drawing line, with the fake warning first.
    fake_suggestion = make_suggestion_view(status="Suggestion 1: 2 colored parts", is_fake=True)
    suggested = ui.describe_participant_status(with_ai, fake_suggestion)
    assert suggested == f"{ui.FAKE_SUGGESTER_PREFIX} Suggestion 1: 2 colored parts", f"Unexpected suggestion line {suggested!r}"

    # The experimenter sees the name, condition, start time, and state.
    experimenter = ui.describe_experimenter_status(without_ai)
    assert experimenter == "p1, without AI, started 10:02, drawing open", f"Unexpected experimenter status {experimenter!r}"
    assert ui.describe_experimenter_status(None) == ui.NO_EXPERIMENT_STATUS, "No drawing must say how to start one"


def test_study_start_needs_name_and_condition_and_ai_buttons_follow_the_condition() -> None:
    """
    Study Start needs a name, a condition, and no open drawing, and the AI buttons show only in an open drawing with AI.
    """
    from sketchloop import ui

    # Start is enabled only with a name and a condition and no open drawing.
    readiness = [ui.can_start_study_drawing("p1", "ai", False), ui.can_start_study_drawing("p1", None, False),
                 ui.can_start_study_drawing(" ", "no_ai", False), ui.can_start_study_drawing("p1", "ai", True)]
    assert readiness == [True, False, False, False], f"Unexpected study Start readiness {readiness}"

    # The two AI buttons are visible only in an open drawing with AI, and removed, not disabled, everywhere else.
    views = [make_drawing_view(is_open=True, condition=StudyCondition.AI),
             make_drawing_view(is_open=True, condition=StudyCondition.NO_AI),
             make_drawing_view(is_open=False, condition=StudyCondition.AI), None]
    visible = [[button.visible for button in ui.build_study_drawing_outputs(view)[1:3]] for view in views]
    assert visible == [[True, True], [False, False], [False, False], [False, False]], f"Unexpected AI buttons {visible}"

    # After a start the name and condition are emptied and locked, and Finish is enabled.
    _, _, _, name_box, condition, finish, _ = ui.build_study_drawing_outputs(views[1], clear_inputs=True)
    assert name_box.value == "" and condition.value is None, "A start must clear the name and condition"
    assert name_box.interactive is False and condition.interactive is False, "An open drawing must lock the start inputs"
    assert finish.interactive is True, "An open drawing must enable Finish"

    # The panel names a failed save in its label.
    labels = [ui.label_experimenter_panel(False), ui.label_experimenter_panel(True)]
    assert labels == [ui.EXPERIMENTER_LABEL, ui.SAVING_FAILED_LABEL], f"Unexpected panel labels {labels}"


@pytest.mark.usefixtures("no_network")
def test_study_page_builds_only_the_participant_and_experimenter_controls(controller: AppController) -> None:
    """
    The study page builds no generation components, its header names no model, and its prompt placeholder names no object.
    """
    import gradio

    from sketchloop import ui

    app = ui.SketchPage(controller, default_candidates=1, default_seed=None, study=True).build()
    blocks = list(app.blocks.values())
    buttons = {block.value for block in blocks if isinstance(block, gradio.Button)}

    # Only the two AI buttons and Start and Finish exist, and the AI buttons start removed.
    expected = {"Suggest a prompt", "Suggest next strokes", "Start drawing", "Finish drawing"}
    assert buttons == expected, f"Unexpected study buttons {buttons}"
    ai_buttons = [block.visible for block in blocks if isinstance(block, gradio.Button) and block.value.startswith("Suggest")]
    assert ai_buttons == [False, False], f"The AI buttons must start removed, got {ai_buttons}"

    # No gallery, choice, sliders, seed, history, or raw option is built.
    generation_types = (gradio.Gallery, gradio.CheckboxGroup, gradio.Slider, gradio.Dataframe, gradio.Checkbox)
    built = [type(block).__name__ for block in blocks if isinstance(block, generation_types)]
    labels = [str(getattr(block, "label", "") or "") for block in blocks]
    assert not built and "Seed" not in labels, f"The study page must not build generation components, found {built}"

    # The header is the title only, and the backend and suggesters are described in the experimenter panel.
    texts = [block.value or "" for block in blocks if isinstance(block, gradio.Markdown)]
    assert "# SketchLoop" in texts and controller.describe_backend() not in texts[0], f"Unexpected header {texts[0]!r}"
    assert any(controller.describe_prompt_suggester() in text for text in texts), "The panel must name the prompt suggester"

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
    page.store_frame(np.ascontiguousarray(sheet_frame[:, :, ::-1]))

    # A start without a condition is refused, and nothing is started.
    with pytest.raises(gradio.Error, match=re.escape(ui.CHOOSE_CONDITION_MESSAGE)):
        page.start_study_drawing("p1_car", None)
    assert controller.latest_drawing() is None, "A start without a condition must not start a drawing"

    # Without AI, the buttons are removed, the inputs and prompt are cleared, and the panel closes.
    started = page.start_study_drawing("p1_car", "no_ai")
    status, suggest_prompt, suggest_strokes, name_box, condition, _, _, start, sketch_view, prompt, prompt_id, panel = started
    assert [suggest_prompt.visible, suggest_strokes.visible] == [False, False], "Without AI the AI buttons must be removed"
    assert name_box.value == "" and condition.value is None and start.interactive is False, "A start must clear its inputs"
    assert prompt.value == "" and prompt_id is None and sketch_view.value is None, "A start must clear the prompt and view"
    assert panel.open is False and status == ui.DRAWING_STATUS, f"A start must close the panel, got {panel.open} and {status!r}"

    # Prompt suggestions are refused without AI, and Finish keeps the typed prompt as typed.
    with pytest.raises(gradio.Error, match=re.escape(SUGGESTIONS_OFF_MESSAGE)):
        page.suggest_another("webcam", None, 0)
    finished = page.finish_study_drawing("webcam", None, 0, "a lamp", None)
    record = controller.session.drawings[0].record
    assert finished[0] == ui.FINISHED_STATUS and finished[-1].value is None, "Finish must thank and empty the view"
    assert record.final_prompt == "a lamp" and record.final_prompt_origin.kind == PromptOriginKind.TYPED, "Bad final prompt"

    # With AI, the buttons show, and a suggested prompt kept unchanged until Finish records suggested.
    started = page.start_study_drawing("p2_car", "ai")
    assert [started[1].visible, started[2].visible] == [True, True], "With AI the AI buttons must show"
    text, suggestion_id = page.suggest_another("webcam", None, 0)
    page.finish_study_drawing("webcam", None, 0, text, suggestion_id)
    origin = controller.session.drawings[0].record.final_prompt_origin
    assert origin.kind == PromptOriginKind.SUGGESTED and origin.prompt_suggestion == suggestion_id, f"Bad final origin {origin}"

    # A failed save relabels the panel, and a working save restores the label.
    page.start_study_drawing("p3_car", "ai")

    def fail_to_save(*args: object, **kwargs: object) -> None:
        """
        Fail like a full disk.
        """
        raise StoreError("Could not save drawing.json. Free disk space and retry.")

    monkeypatch.setattr(controller, "record_drawing_capture", fail_to_save)
    notice, panel = page.save_study_capture("webcam", None, 0, "")
    assert notice.visible is True and panel.label == ui.SAVING_FAILED_LABEL, f"A failed save must relabel the panel, got {panel}"
    monkeypatch.undo()
    notice, panel = page.save_study_capture("webcam", None, 0, notice.value)
    assert notice.visible is False and panel.label == ui.EXPERIMENTER_LABEL, f"A working save must restore the label, got {panel}"
