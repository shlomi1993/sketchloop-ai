"""
Tests of the browser page's handlers without starting a server.
"""

import cv2
import numpy as np
import pytest
import re
import socket

from pathlib import Path

from sketchloop.app import NO_FRAME_MESSAGE, NO_NEW_STROKES_MESSAGE, NO_SKETCH_MESSAGE, AppController, DrawingView, SuggestionView
from sketchloop.backends.fake import FakeGenerator
from sketchloop.domain import Rotation, Unavailable
from sketchloop.experiments import EXPERIMENT_NAME_RULE, NOT_STREAMED_REASON, FolderStore, load_session


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
        page.run_generation("webcam", None, "chair", 2, None, 0, False, progress=lambda *args, **kwargs: None)

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
        page.run_generation("webcam", None, "chair", 2, None, 0, False, progress=lambda *args, **kwargs: None)


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
    page.run_generation("webcam", None, "chair", 1, 7, 0, False, progress=lambda *args, **kwargs: None)
    webcam_age = load_session(controller.session.folder).rounds[0].frame_age_seconds
    assert webcam_age == pytest.approx(0.4), f"Expected a recorded frame age of 0.4 s, got {webcam_age}"

    # An upload round passes no age, so it records that its sketch was not streamed.
    upload_path = tmp_path / "upload.png"
    upload_path.write_bytes(cv2.imencode(".png", page_sketch)[1].tobytes())
    page.run_generation("upload", str(upload_path), "lamp", 1, 7, 0, False, progress=lambda *args, **kwargs: None)
    upload_age = load_session(controller.session.folder).rounds[1].frame_age_seconds
    assert upload_age == Unavailable(NOT_STREAMED_REASON), f"Expected the not-streamed reason, got {upload_age}"


def make_drawing_view(*, is_open: bool, suggestion_lines: tuple[str, ...] = ()) -> DrawingView:
    """
    Build a drawing view of experiment p1 with the given state and suggestion lines.
    """
    return DrawingView(number=1, experiment="p1", is_open=is_open, capture_count=1, suggestion_count=len(suggestion_lines),
                       suggestion_lines=suggestion_lines)


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
    with_suggestion = ui.style_sketch_view(make_drawing_view(is_open=True, suggestion_lines=("Suggestion 1: 2 parts",)))
    assert sketch_view.height == ui.GUIDED_PREVIEW_HEIGHT, f"Expected a taller guided preview, got {sketch_view.height}"
    labels = [sketch_view.label, with_suggestion.label]
    assert labels == [ui.OPEN_DRAWING_LABEL, ui.GUIDED_PREVIEW_LABEL], f"Unexpected preview labels {labels}"

    # A suggestion's line follows, a fake one is labeled, and the no-new-strokes message is passed through.
    fake = ui.describe_drawing_status(open_view, make_suggestion_view(status="Suggestion 1: 2 parts in 0.1 s", is_fake=True))
    empty = ui.describe_drawing_status(open_view, make_suggestion_view(status=NO_NEW_STROKES_MESSAGE, is_fake=False))
    assert fake.startswith(ui.FAKE_SUGGESTER_PREFIX) and fake.endswith("Suggestion 1: 2 parts in 0.1 s"), f"Bad {fake!r}"
    assert empty.endswith(NO_NEW_STROKES_MESSAGE) and ui.FAKE_SUGGESTER_PREFIX not in empty, f"Bad {empty!r}"

    # After a reload the latest suggestion line is shown, and a finished drawing says how many suggestions it had.
    restored = ui.describe_drawing_status(make_drawing_view(is_open=True, suggestion_lines=("Suggestion 1: 2 parts",)))
    finished = ui.describe_drawing_status(make_drawing_view(is_open=False, suggestion_lines=("Suggestion 1: 2 parts",)))
    assert restored.endswith("Suggestion 1: 2 parts"), f"A reload must show the latest suggestion, got {restored!r}"
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

    # A valid name starts the drawing, enables Suggest, and clears the name, prompt, and upload so nothing carries over.
    started = page.start_drawing(" participant1_car ")
    status, suggest, finish, new_session, name_box, sketch_view, start_button, prompt_box, upload_box, *_ = started
    assert name_box.value == "" and start_button.interactive is False, "Starting must clear the name and disable Start"
    assert prompt_box.value == "" and upload_box.value is None, "Starting must clear the previous prompt and upload"
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
    status, sketch_view = page.suggest_next_strokes("webcam", None, "", 0)
    assert status.startswith(ui.FAKE_SUGGESTER_PREFIX) and "Suggestion 1: 2 parts" in status, f"Bad status {status!r}"
    assert sketch_view.label == ui.GUIDED_PREVIEW_LABEL and sketch_view.value, f"Bad sketch view {sketch_view.label!r}"
    recorded_prompt = controller.session.drawings[0].record.suggestions[0].request.prompt
    assert recorded_prompt == "", f"The empty prompt must be sent as typed, got {recorded_prompt!r}"

    # The timer shows the guided preview while the drawing is open.
    refreshed = page.refresh_webcam_preview("webcam", None, 0, False)
    assert refreshed.shape[:2] == (360, 512), f"The timer must show the guided preview, got {refreshed.shape}"

    # After the suggestion its button is enabled again, and a failure would show the drawing's own status line.
    assert page.update_suggest_button().interactive is True, "Suggest must be enabled again while the drawing is open"
    assert page.show_drawing_status().endswith("Suggestion 1: 2 parts"), "The status must fall back to the latest suggestion"

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
