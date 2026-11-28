"""
Tests of the browser page's handlers without starting a server.
"""

import numpy as np
import pytest
import re
import socket

from datetime import datetime
from pathlib import Path

from sketchloop.app import (SUGGESTIONS_OFF_MESSAGE, AppController,
                            DrawingView, SuggestionView)
from sketchloop.backends.fake import FakeGenerator
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggestionError, StrokeSuggestionOutput, StrokeSuggestionRequest
from sketchloop.experiments import (FolderStore, PromptOriginKind, StoreError,
                                    StudyCondition)


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

    # The sidebar is marked when something needs a check, so the stylesheet can flag it.
    calm, flagged = ui.style_experimenter_panel(False), ui.style_experimenter_panel(True)
    assert (calm.elem_classes, flagged.elem_classes) == (["card"], ["card", "needs-check"]), f"Unexpected sidebar {flagged}"


@pytest.mark.usefixtures("no_network")
def test_study_page_builds_only_the_participant_and_experimenter_controls(controller: AppController) -> None:
    """
    The study page builds no generation components, its header names no model, and its prompt placeholder names no object.
    """
    import gradio

    from sketchloop import ui

    app = ui.SketchPage(controller).build()
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

    # The header is the title only, and the panel names the prompt suggester in its settings card.
    texts = [block.value or "" for block in blocks if isinstance(block, gradio.Markdown)]
    assert "# SketchLoop" in texts and controller.describe_backend() not in texts[0], f"Unexpected header {texts[0]!r}"
    system = next(block for block in blocks if isinstance(block, gradio.HTML) and block.elem_id == "system")
    assert "System settings" in system.value and "templates" in system.value, "The panel must name the prompt suggester"

    # The prompt names no object, the condition has no default, and the panel starts open so the webcam can start.
    prompts = [block for block in blocks if isinstance(block, gradio.Textbox) and block.label == "Prompt"]
    assert [box.placeholder for box in prompts] == [ui.STUDY_PROMPT_PLACEHOLDER], f"Unexpected prompt boxes {prompts}"
    conditions = [block for block in blocks if isinstance(block, gradio.Radio) and block.label == "Condition"]
    assert len(conditions) == 1 and conditions[0].value is None, "The condition must have no default"


@pytest.mark.usefixtures("no_network")
def test_study_page_runs_a_no_ai_and_an_ai_drawing(controller: AppController, sheet_frame: np.ndarray,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The study handlers start each condition, clear the inputs, close the panel, refuse AI help without AI, and keep final prompts.
    """
    import gradio

    from sketchloop import ui

    page = ui.SketchPage(controller)
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
    assert status == ui.DRAWING_STATUS, f"Unexpected status after a start {status!r}"
    assert shown == ("p1_car", 1, True, 0) and armed is None, f"A start must record what the tab shows, got {shown}"

    # Prompt suggestions are refused without AI.
    with pytest.raises(gradio.Error, match=re.escape(SUGGESTIONS_OFF_MESSAGE)):
        page.suggest_another("webcam", None, 0)

    # A first Finish press only asks for a second, and the timer turns it back after the window.
    first = page.press_finish("webcam", None, 0, "a lamp", None, None)
    assert first[5].value == ui.FINISH_CONFIRM_LABEL and first[-2] == 100.0, f"A first press must arm Finish, got {first[5]}"
    assert controller.latest_drawing().is_open, "A first press must not finish the drawing"
    clock_seconds[0] = 106.0
    reset_button, reset_armed = page.disarm_finish(100.0)
    assert reset_button.value == ui.FINISH_LABEL and reset_armed is None, "An unconfirmed press must turn back"

    # Two presses within the window finish with the typed prompt as typed.
    page.store_frame(frame)
    armed_at = page.press_finish("webcam", None, 0, "a lamp", None, None)[-2]
    clock_seconds[0] = 108.0
    finished = page.press_finish("webcam", None, 0, "a lamp", None, armed_at)
    record = controller.session.drawings[0].record
    assert finished[0] == ui.FINISHED_STATUS and finished[7].value is None, "Finish must thank and empty the view"
    assert record.final_prompt == "a lamp" and record.final_prompt_origin.kind == PromptOriginKind.TYPED, "Bad final prompt"

    # With AI, the buttons show, and a suggested prompt kept unchanged until Finish records suggested.
    started = page.start_study_drawing("p2_car", "ai")
    assert [started[1].visible, started[2].visible] == [True, True], "With AI the AI buttons must show"
    text, suggestion_id = page.suggest_another("webcam", None, 0)
    armed_at = page.press_finish("webcam", None, 0, text, suggestion_id, None)[-2]
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

    page = ui.SketchPage(controller)
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
    assert notice.value == ui.CAMERA_STOPPED_NOTICE and "needs-check" in panel.elem_classes, f"Bad flag {notice}, {panel}"
    assert rest == [gradio.skip()] * 4, "A camera problem must not touch the participant's status"
    assert page.save_study_capture("webcam", None, 0, notice.value, shown)[:2] == (gradio.skip(),) * 2, "No repeat"
    assert capsys.readouterr().out.count("check needed") == 1, "The terminal must name the problem once"

    # Frames that return clear the flag and print the recovery.
    page.store_frame(frame)
    notice, panel, *_ = page.save_study_capture("webcam", None, 0, notice.value, shown)
    assert notice.visible is False and "needs-check" not in panel.elem_classes, f"Frames must clear the flag, got {panel}"
    assert "work again" in capsys.readouterr().out, "The terminal must say when it works again"

    # A failed save flags the panel with its own notice.
    def fail_to_save(*args: object, **kwargs: object) -> None:
        """
        Fail like a full disk.
        """
        raise StoreError("Could not save drawing.json. Free disk space and retry.")

    monkeypatch.setattr(controller, "record_drawing_capture", fail_to_save)
    notice, panel, *_ = page.save_study_capture("webcam", None, 0, "", shown)
    assert "Free disk space" in notice.value and "needs-check" in panel.elem_classes, f"A failed save must flag, got {notice}"


@pytest.mark.usefixtures("no_network")
def test_study_failed_suggestions_are_calm_for_the_participant(tmp_path: Path, sheet_frame: np.ndarray) -> None:
    """
    Failed prompt and stroke suggestions give the participant a calm line, while the experimenter status keeps the error.
    """
    import gradio

    from sketchloop import ui

    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), stroke_suggester=FailingStrokeSuggester())
    page = ui.SketchPage(controller)
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
