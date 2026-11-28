"""
The Gradio browser page, a thin layer over AppController for webcam or upload rounds and guided drawings.
"""

import cv2
import gradio as gr
import numpy as np
import threading
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from sketchloop.app import NO_FRAME_MESSAGE, AppController, DrawingView, SuggestionView
from sketchloop.domain import Rotation, SketchLoopError, SketchSource
from sketchloop.experiments import StudyCondition


SOURCE_CHOICES: Final = [("Live webcam", SketchSource.WEBCAM), ("Upload a file", SketchSource.UPLOAD)]

# A webcam facing the person sees the paper upside down.
DEFAULT_ROTATION_BY_SOURCE: Final = {SketchSource.WEBCAM: Rotation.DEG_180, SketchSource.UPLOAD: Rotation.DEG_000}


# The page's CSS and script, shipped as package data.
STATIC_DIR: Final = Path(__file__).parent / "static"
STYLESHEET_PATH: Final = STATIC_DIR / "ui.css"
SCRIPT_PATH: Final = STATIC_DIR / "ui.js"

FAKE_BACKEND_PREFIX: Final = "**Fake backend:** these images are deterministic noise, not output from a real model."
SHARED_SESSION_NOTE: Final = "All browser tabs share one session, and reloading keeps it."
STALE_FRAME_MESSAGE: Final = "The webcam stopped sending frames. Restart it and try again."

# Guided drawing texts. Nothing on the page names a task or goal, since the system is never told one.
EXPERIMENT_NAME_HINT: Final = "Use a code such as participant1_car. Never use a real name."
NO_DRAWING_STATUS: Final = "For next-stroke suggestions, enter a name under **Experiment** and press **Start drawing**."
FAKE_SUGGESTER_PREFIX: Final = "**Fake suggester:** these are test shapes, not AI suggestions."
FINISH_FIRST_MESSAGE: Final = "Finish the current drawing first."
SUGGESTING_STATUS: Final = "Suggesting next strokes, usually a few seconds."
PLAIN_PREVIEW_LABEL: Final = "Sketch the model will receive"
OPEN_DRAWING_LABEL: Final = "Your sketch"
GUIDED_PREVIEW_LABEL: Final = "Your sketch with suggested strokes"

# Sketch view heights in pixels, taller while a drawing is open so the suggested strokes read at a glance.
PLAIN_PREVIEW_HEIGHT: Final = 320
GUIDED_PREVIEW_HEIGHT: Final = 440

# Study page texts. The participant's words are the same in both conditions and never name the experiment or condition.
STUDY_PROMPT_PLACEHOLDER: Final = "Optional. Describe your drawing in a few words"
WAITING_STATUS: Final = "Waiting for the experimenter to start."
DRAWING_STATUS: Final = "Drawing. Tell the experimenter when you are done."
FINISHED_STATUS: Final = "Finished. Thank you."
DONE_REMINDER: Final = "Tell the experimenter when you are done."
SUGGESTION_FAILED_STATUS: Final = "That suggestion did not work. Keep drawing, or press again."
PROMPT_FAILED_STATUS: Final = "That did not work. Keep drawing, or press again."
CONDITION_CHOICES: Final = [("With AI", StudyCondition.AI), ("Without AI", StudyCondition.NO_AI)]
CONDITION_WORDS: Final = {StudyCondition.AI: "with AI", StudyCondition.NO_AI: "without AI"}
CHOOSE_CONDITION_MESSAGE: Final = "Choose a condition, then start the drawing."
NO_EXPERIMENT_STATUS: Final = "No experiment started yet. Enter a name, choose a condition, and press **Start drawing**."
EXPERIMENTER_LABEL: Final = "Experimenter"
CAMERA_STOPPED_NOTICE: Final = "**The webcam stopped sending frames, so nothing is being saved.** Open this panel and restart it."
START_LABEL: Final = "Start drawing"
FINISH_LABEL: Final = "Finish drawing"
FINISH_CONFIRM_LABEL: Final = "Press again to finish"

# The class that tells ui.js to close the experimenter panel once the webcam streams, after a reload during a drawing.
CLOSE_PANEL_CLASS: Final = "close-panel-when-streaming"

# Seconds a first press of Finish waits for the second, so one stray click never ends a drawing.
FINISH_CONFIRM_SECONDS: Final = 5.0

# The study page's sketch view height and its right column's width in pixels, fixed so nothing moves between conditions.
STUDY_SKETCH_HEIGHT: Final = 560
STUDY_CONTROLS_WIDTH: Final = 360
CAMERA_HEIGHT: Final = 240


# A webcam frame older than this many capture intervals means the stream stopped.
STALE_FRAME_INTERVALS: Final = 10

# Serve only on this machine.
SERVER_HOST: Final = "127.0.0.1"

# The study page's only footer link, leaving out the API, runs, and settings pages.
STUDY_FOOTER_LINKS: Final = ["gradio"]

# A neutral look, so the images stand out.
THEME: Final = gr.themes.Default(primary_hue="indigo", neutral_hue="slate", radius_size="lg")


# A drawing's experiment, its number, and an uploaded file's path, so an unchanged upload is saved once per drawing.
UploadCaptureKey: TypeAlias = tuple[str | None, int, str]


# Participant status, Suggest a prompt, Suggest next strokes, the name box, the condition, Finish, and the experimenter status.
StudyDrawingOutputs: TypeAlias = tuple[str, gr.Button, gr.Button, gr.Textbox, gr.Radio, gr.Button, str]

# A drawing's experiment, number, open state, and suggestion count, or None before the first drawing.
DrawingSignature: TypeAlias = tuple[str | None, int, bool, int] | None


def pick_rotation(choice: str, use_raw: bool) -> int:
    """
    Pick the rotation the page offers, 0 for the raw sketch, which skips preprocessing and so cannot be turned.
    """
    return Rotation.DEG_000 if use_raw else DEFAULT_ROTATION_BY_SOURCE[choice]


def can_start_drawing(name: str | None, is_open: bool) -> bool:
    """
    Tell whether an experiment name was typed and no drawing is open, so a drawing can start.
    """
    return bool((name or "").strip()) and not is_open


def is_drawing_open(view: DrawingView | None) -> bool:
    """
    Tell whether a drawing is open and so takes suggestions.
    """
    return view is not None and view.is_open


def can_start_study_drawing(name: str | None, condition: str | None, is_open: bool) -> bool:
    """
    Tell whether a name was typed, a condition chosen, and no drawing is open, so a study drawing can start.
    """
    return can_start_drawing(name, is_open) and bool(condition)


def label_start_button(condition: str | None) -> str:
    """
    Repeat the chosen condition on Start, such as "Start with AI", so a wrong choice shows before the click.
    """
    return f"Start {CONDITION_WORDS[StudyCondition(condition)]}" if condition else START_LABEL


def offers_ai_help(view: DrawingView | None) -> bool:
    """
    Tell whether a drawing is open with AI help, so the participant sees the two suggestion buttons.
    """
    return is_drawing_open(view) and view.suggestions_enabled


def describe_participant_status(view: DrawingView | None, suggestion: SuggestionView | None = None) -> str:
    """
    Say the drawing's state to the participant in the same words in both conditions, never naming the experiment or condition.

    Args:
        view (DrawingView | None): The latest drawing, or None before the first.
        suggestion (SuggestionView | None, optional): The suggestion just shown, if any. Defaults to None.

    Returns:
        str: The participant's status, with the reminder to say when done under any suggestion line.
    """
    if view is None:
        return WAITING_STATUS

    if not view.is_open:
        return FINISHED_STATUS

    # Show the suggestion just made, warning first when it is fake, or the latest one after a reload, worded calmly if it failed.
    if suggestion is not None:
        line = f"{FAKE_SUGGESTER_PREFIX} {suggestion.status}" if suggestion.is_fake else suggestion.status
    elif view.latest_suggestion_failed:
        line = SUGGESTION_FAILED_STATUS
    elif view.suggestion_lines:
        line = view.suggestion_lines[-1]
    else:
        return DRAWING_STATUS

    return f"{line}\n\n{DONE_REMINDER}"


def describe_experimenter_status(view: DrawingView | None, problem: str | None = None) -> str:
    """
    Name the experiment, its condition, start time, and state, with a failed suggestion's full line or another problem.

    Args:
        view (DrawingView | None): The latest drawing, or None before the first.
        problem (str | None, optional): A problem to name, such as a suggestion error, or None for the latest failed
            suggestion's line, if any. Defaults to None.

    Returns:
        str: One status line for the experimenter, such as "p1_car, without AI, started 10:02, drawing open".
    """
    if view is None:
        return NO_EXPERIMENT_STATUS

    # Name each known part, leaving out a condition or start time the drawing does not have.
    parts = [view.experiment or f"Drawing {view.number}"]
    if view.condition is not None:
        parts.append(CONDITION_WORDS[view.condition])
    if view.started_at is not None:
        parts.append(f"started {view.started_at:%H:%M}")
    parts.append("drawing open" if view.is_open else "drawing finished")
    status = ", ".join(parts)

    # Add the problem, or the latest failed suggestion's line, which the participant sees only in calm words.
    detail = problem or (view.suggestion_lines[-1] if view.latest_suggestion_failed and view.suggestion_lines else None)
    return f"{status}. {detail}" if detail else status


def style_experimenter_panel(needs_check: bool) -> gr.Column:
    """
    Label and mark the experimenter panel when something needs a check, in neutral words the participant may see.
    """
    return gr.Column(elem_classes=["card", "needs-check"] if needs_check else ["card"])


def sign_drawing(view: DrawingView | None) -> DrawingSignature:
    """
    Sum up what the participant's status and buttons show for a drawing, so a tab updates them only when it changes.
    """
    if view is None:
        return None

    return view.experiment, view.number, view.is_open, view.suggestion_count


def is_finish_armed(armed_at: float | None, now: float) -> bool:
    """
    Tell whether Finish was pressed once within the confirmation window, so the next press finishes.
    """
    return armed_at is not None and 0 <= now - armed_at <= FINISH_CONFIRM_SECONDS


def build_study_drawing_outputs(view: DrawingView | None, *, clear_inputs: bool = False) -> StudyDrawingOutputs:
    """
    Turn the latest drawing into the study page's statuses, the AI buttons, the start inputs, and Finish.

    Args:
        view (DrawingView | None): The latest drawing, or None before the first.
        clear_inputs (bool, optional): Empty the name and condition, as after a start. Defaults to False.

    Returns:
        StudyDrawingOutputs: Values for the study drawing components.
    """
    # Show the AI buttons only in an open drawing with AI help, so they are removed, not disabled, everywhere else.
    is_open = is_drawing_open(view)
    shows_ai = offers_ai_help(view)

    # Lock the name and condition while a drawing is open, and clear both after a start so neither carries over.
    name_update: dict[str, object] = {"value": ""} if clear_inputs else {}
    condition_update: dict[str, object] = {"value": None} if clear_inputs else {}
    name_box = gr.Textbox(info=FINISH_FIRST_MESSAGE if is_open else EXPERIMENT_NAME_HINT, interactive=not is_open, **name_update)
    condition = gr.Radio(interactive=not is_open, **condition_update)
    finish = gr.Button(value=FINISH_LABEL, interactive=is_open)
    return (describe_participant_status(view), gr.Button(visible=shows_ai), gr.Button(visible=shows_ai, interactive=shows_ai),
            name_box, condition, finish, describe_experimenter_status(view))


def name_upload_capture(view: DrawingView | None, choice: str, upload_path: str | None) -> UploadCaptureKey | None:
    """
    Name an upload's interval capture by its drawing and file, or None for a webcam frame, which changes with every tick.
    """
    if view is None or upload_path is None or SketchSource(choice) != SketchSource.UPLOAD:
        return None

    return view.experiment, view.number, upload_path


def describe_capture_failure(error: Exception, interval_seconds: float) -> str:
    """
    Say in one line that saving the drawing failed and that the page keeps trying.
    """
    return f"**Saving the drawing failed.** {error} The page tries again every {interval_seconds:g} s."


def decode_preview(payload: bytes) -> np.ndarray:
    """
    Decode encoded preview bytes into an RGB image for display.
    """
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParticipantArea:
    """
    The study page's participant components: the sketch view, the prompt, the AI buttons, and the status.
    """
    sketch_view: gr.Image  # Paper canvas with the open suggestion's overlay, empty outside a drawing
    prompt: gr.Textbox  # Optional prompt, sent with each stroke suggestion and kept as the final prompt
    prompt_suggestion: gr.State  # ID of the prompt suggestion written into the prompt box, None until one is
    suggest_prompt: gr.Button  # Writes a suggested prompt, shown only in an open drawing with AI help
    suggest_strokes: gr.Button  # Suggests next strokes, shown only in an open drawing with AI help
    status: gr.Markdown  # The drawing's state in the same words in both conditions
    shown_drawing: gr.State  # Signature of the drawing the status and buttons show, so the timer follows other tabs


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimenterPanel:
    """
    The study page's collapsible experimenter controls, away from the participant's view.
    """
    panel: gr.Column  # The sidebar, marked when saving fails or the webcam stops
    name: gr.Textbox  # Experiment name, which also names the session folder
    condition: gr.Radio  # With AI or without AI, with no default
    start: gr.Button  # Starts a study drawing with the name and condition
    finish: gr.Button  # Finishes the open drawing with the current sketch and the prompt box text
    status: gr.Markdown  # Experiment, condition, start time, and state
    capture_notice: gr.Markdown  # A failed interval save, hidden while saving works
    source: gr.Radio  # Webcam or upload choice
    rotation: gr.Dropdown  # Degrees to turn the sketch clockwise
    webcam: gr.Image  # Live webcam preview that streams frames
    upload: gr.Image  # Uploaded sketch file, for dry runs
    finish_armed: gr.State  # When this tab's Finish was pressed once, None until it is


class SketchPage:
    """
    The browser page, with handlers that share the controller and the newest webcam frame.
    """

    def __init__(self, controller: AppController, *, capture_interval_seconds: float = 0.5) -> None:
        """
        Store the controller and the page defaults.

        Args:
            controller (AppController): Service holding the generator and the session.
            capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.
        """
        self.controller = controller
        self.capture_interval_seconds = capture_interval_seconds

        # The newest webcam frame in BGR order and when it arrived, shared by all tabs and read together under the lock.
        self.frame_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_time = 0.0

        # The failed interval save the experimenter sees, empty while saving works, and the upload saved last.
        self.capture_failure = ""
        self.saved_upload: UploadCaptureKey | None = None

        # The study page's problem last printed in the terminal, empty while everything works.
        self.reported_problem = ""

    def take_fresh_frame(self) -> tuple[np.ndarray, float] | None:
        """
        Return the newest webcam frame and its age in seconds, or None when it is missing or stale.
        """
        max_age_seconds = STALE_FRAME_INTERVALS * self.capture_interval_seconds

        # Read the frame and its arrival time together.
        with self.frame_lock:
            frame_age_seconds = time.monotonic() - self.latest_frame_time
            is_fresh = self.latest_frame is not None and frame_age_seconds <= max_age_seconds
            return (self.latest_frame, frame_age_seconds) if is_fresh else None

    def toggle_source(self, choice: str, use_raw: bool = False) -> tuple[gr.Image, gr.Image, gr.Dropdown]:
        """
        Show the input for the chosen source and reset the rotation to its default.
        """
        rotation_choice = gr.Dropdown(value=pick_rotation(choice, use_raw), interactive=not use_raw)
        is_webcam = choice == SketchSource.WEBCAM
        return gr.Image(visible=is_webcam), gr.Image(visible=not is_webcam), rotation_choice

    def turn_preview(self, degrees: int | None) -> gr.Image:
        """
        Turn the webcam preview by switching its rotation CSS class.
        """
        return gr.Image(elem_classes=[f"rotate-{int(degrees or 0)}"])

    def store_frame(self, frame: np.ndarray | None) -> None:
        """
        Keep the newest streamed webcam frame, converted from RGB to BGR, and when it arrived, ignoring a missing one.
        """
        if frame is not None:
            bgr_frame = np.ascontiguousarray(frame[:, :, ::-1])
            with self.frame_lock:
                self.latest_frame = bgr_frame
                self.latest_frame_time = time.monotonic()

    def pick_current_sketch(self, choice: str, upload_path: str | None) -> Path | np.ndarray | None:
        """
        Return the newest fresh webcam frame or the uploaded file, whichever source is chosen, or None when there is none.
        """
        if SketchSource(choice) == SketchSource.WEBCAM:
            fresh_frame = self.take_fresh_frame()
            return fresh_frame[0] if fresh_frame else None

        return None if upload_path is None else Path(upload_path)

    def show_sketch_preview(self, choice: str, upload_path: str | None, rotation_degrees: int,
                            use_raw: bool) -> np.ndarray | dict | None:
        """
        Show the current sketch with the open drawing's suggestion, or as the generator would receive it otherwise.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            use_raw (bool): Whether to skip preprocessing and rotation.

        Returns:
            np.ndarray | dict | None: The preview in RGB order, a skip that keeps the last guided preview, or None when
                there is no usable sketch yet.
        """
        sketch = self.pick_current_sketch(choice, upload_path)
        rotation = int(rotation_degrees or 0)

        # While a drawing is open, show the paper with the suggestion's overlay, keeping the last one when the paper is hidden.
        guided = self.controller.preview_guided_sketch(sketch, rotation)
        if guided is None and is_drawing_open(self.controller.latest_drawing()):
            return gr.skip()

        payload = guided or self.controller.preview_sketch(sketch, rotation, use_raw)
        return None if payload is None else decode_preview(payload)

    def suggest_another(self, choice: str, upload_path: str | None, rotation_degrees: int,
                        use_raw: bool = False) -> tuple[str, str]:
        """
        Write a suggested style for the current sketch, the next one for the same drawing, or show why there is none.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            use_raw (bool, optional): Whether to skip preprocessing and rotation. Defaults to False.

        Returns:
            tuple[str, str]: The suggested prompt for the box, and its ID, which the tab keeps and sends with each use.
        """
        # A failure leaves the prompt box and the kept ID as they were.
        sketch = self.pick_current_sketch(choice, upload_path)
        try:
            suggestion = self.controller.request_prompt_suggestion(sketch, int(rotation_degrees or 0), use_raw)
        except (SketchLoopError, OSError) as error:
            raise gr.Error(str(error)) from error

        return suggestion.text, suggestion.suggestion_id

    def find_sketch(self, choice: str, upload_path: str | None) -> tuple[Path | np.ndarray | None, float | None]:
        """
        Return the chosen source's sketch and a webcam frame's age, or None and None when there is no fresh sketch.
        """
        if SketchSource(choice) != SketchSource.WEBCAM:
            return None if upload_path is None else Path(upload_path), None

        return self.take_fresh_frame() or (None, None)

    def take_sketch(self, choice: str, upload_path: str | None) -> tuple[Path | np.ndarray | None, float | None]:
        """
        Return the chosen source's sketch and a webcam frame's age, refusing a frame from before the stream stopped.
        """
        sketch, frame_age_seconds = self.find_sketch(choice, upload_path)
        if sketch is None and SketchSource(choice) == SketchSource.WEBCAM and self.latest_frame is not None:
            raise gr.Error(STALE_FRAME_MESSAGE)

        return sketch, frame_age_seconds

    def open_drawing(self, name: str | None, condition: StudyCondition | None = None) -> DrawingView:
        """
        Start a named experiment with its first drawing, refusing while a drawing is open and showing errors as one line.
        """
        # Never end an open drawing by starting another, even from a second tab.
        if is_drawing_open(self.controller.latest_drawing()):
            raise gr.Error(FINISH_FIRST_MESSAGE)

        # An invalid or taken name is refused with one line, and the current session stays.
        try:
            return self.controller.start_drawing(name or "", condition)
        except (SketchLoopError, OSError) as error:
            raise gr.Error(str(error)) from error

    def show_suggesting(self) -> tuple[str, gr.Button]:
        """
        Say that a suggestion is running and disable its button until it ends.
        """
        return SUGGESTING_STATUS, gr.Button(interactive=False)

    def describe_status(self, view: DrawingView | None, suggestion: SuggestionView | None = None) -> str:
        """
        Describe the drawing in the participant's words on the study page, and with the experiment's name otherwise.
        """
        return describe_participant_status(view, suggestion)

    def style_view(self, view: DrawingView | None, **update: object) -> gr.Image:
        """
        Restyle the sketch view for the drawing, except on the study page, whose view keeps one label and height.
        """
        return gr.Image(**update)

    def suggest_next_strokes(self, choice: str, upload_path: str | None, prompt_text: str | None,
                             prompt_suggestion_id: str | None, rotation_degrees: int) -> tuple[str, gr.Image]:
        """
        Suggest next strokes for the open drawing from the current sketch and the prompt as typed, and show the overlay.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            prompt_text (str | None): Prompt box text, sent exactly as typed, possibly empty.
            prompt_suggestion_id (str | None): ID of the prompt suggestion written into the prompt box, or None.
            rotation_degrees (int): Degrees to turn the sketch clockwise.

        Returns:
            tuple[str, gr.Image]: The drawing status and the sketch view with the guided preview.
        """
        # Suggest from the chosen source, refusing a webcam frame from before the stream stopped.
        sketch, frame_age_seconds = self.take_sketch(choice, upload_path)
        try:
            suggestion = self.controller.suggest_strokes(sketch, SketchSource(choice), prompt_text or "",
                                                         rotation=int(rotation_degrees or 0), frame_age_seconds=frame_age_seconds,
                                                         prompt_suggestion_id=prompt_suggestion_id)
        except (SketchLoopError, OSError) as error:
            raise gr.Error(str(error)) from error

        # Show the overlay at once rather than at the next timer tick, naming the suggested strokes in the label.
        view = self.controller.latest_drawing()
        preview = self.show_sketch_preview(choice, upload_path, rotation_degrees, False)
        sketch_view = self.style_view(view, value=preview) if isinstance(preview, np.ndarray) else self.style_view(view)
        return self.describe_status(view, suggestion), sketch_view

    def close_drawing(self, choice: str, upload_path: str | None, rotation_degrees: int, prompt_text: str | None,
                      prompt_suggestion_id: str | None) -> DrawingView:
        """
        Finish the open drawing, keeping the current sketch as its final capture and the prompt box text as its final prompt.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            prompt_text (str | None): Prompt box text at finish, or None when the page sent none.
            prompt_suggestion_id (str | None): ID of the prompt suggestion written into the prompt box, or None.

        Returns:
            DrawingView: The finished drawing.
        """
        # Refuse a stopped or missing webcam frame, so the drawing never ends without its final capture.
        sketch, frame_age_seconds = self.take_sketch(choice, upload_path)
        if sketch is None:
            raise gr.Error(NO_FRAME_MESSAGE)

        try:
            return self.controller.finish_drawing(sketch, SketchSource(choice), rotation=int(rotation_degrees or 0),
                                                  frame_age_seconds=frame_age_seconds, prompt=prompt_text,
                                                  prompt_suggestion_id=prompt_suggestion_id)
        except (SketchLoopError, OSError) as error:
            raise gr.Error(str(error)) from error

    def record_capture_quietly(self, choice: str, upload_path: str | None, rotation_degrees: int,
                               upload_key: UploadCaptureKey | None) -> None:
        """
        Save an interval capture of the open drawing when one is due, keeping any failure for the notice instead of raising.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            upload_key (UploadCaptureKey | None): The upload's capture key, or None for a webcam frame.
        """
        # A stopped stream passes no frame, which the controller skips, so gaps show only in the capture times.
        sketch, frame_age_seconds = self.find_sketch(choice, upload_path)
        try:
            capture = self.controller.record_drawing_capture(sketch, SketchSource(choice), rotation=int(rotation_degrees or 0),
                                                             frame_age_seconds=frame_age_seconds)
        except (SketchLoopError, OSError) as error:
            # Ignore a drawing that finished meanwhile, and otherwise keep the line until a save works.
            is_open = is_drawing_open(self.controller.latest_drawing())
            self.capture_failure = describe_capture_failure(error, self.controller.save_interval_seconds) if is_open else ""
            return

        # A saved capture clears the notice and marks the upload as saved, while a capture that was not due changes nothing.
        if capture is not None:
            self.capture_failure = ""
            self.saved_upload = upload_key

    def save_due_capture(self, choice: str, upload_path: str | None, rotation_degrees: int) -> None:
        """
        Save the current sketch into the open drawing when a capture is due, keeping any failure for the notice.
        """
        # Save while a drawing is open, skipping an upload already saved in it, since saving it again stores a duplicate.
        view = self.controller.latest_drawing()
        upload_key = name_upload_capture(view, choice, upload_path)
        if not is_drawing_open(view):
            self.capture_failure = ""
        elif upload_key is None or upload_key != self.saved_upload:
            self.record_capture_quietly(choice, upload_path, rotation_degrees, upload_key)

    def update_suggest_button(self) -> gr.Button:
        """
        Enable Suggest next strokes again while a drawing is open.
        """
        return gr.Button(interactive=is_drawing_open(self.controller.latest_drawing()))

    def update_study_start_button(self, name: str | None, condition: str | None) -> gr.Button:
        """
        Name the chosen condition on Start, and enable it once a name and condition are given, unless a drawing is open.
        """
        is_open = is_drawing_open(self.controller.latest_drawing())
        return gr.Button(value=label_start_button(condition), interactive=can_start_study_drawing(name, condition, is_open))

    def start_study_drawing(self, name: str | None, condition: str | None) -> tuple[object, ...]:
        """
        Start a study drawing with the chosen condition, clearing the name, condition, and prompt, and closing the panel.

        Args:
            name (str | None): Experiment name as typed.
            condition (str | None): Chosen condition, ai or no_ai, or None when none was chosen.

        Returns:
            tuple[object, ...]: The study drawing outputs, Start drawing, the emptied sketch view, the cleared prompt and its
                kept ID, the closed panel, the drawing this tab shows, and a disarmed Finish.
        """
        # Refuse a start without a condition, so no drawing gets a condition nobody chose.
        if not condition:
            raise gr.Error(CHOOSE_CONDITION_MESSAGE)

        view = self.open_drawing(name, StudyCondition(condition))

        # Empty the view until the paper is found, clear the previous prompt and its ID, and hide the panel from the participant.
        return (*build_study_drawing_outputs(view, clear_inputs=True), gr.Button(value=START_LABEL, interactive=False),
                gr.Image(value=None), gr.Textbox(value=""), None, gr.skip(), sign_drawing(view), None)

    def press_finish(self, choice: str, upload_path: str | None, rotation_degrees: int, prompt_text: str | None,
                     prompt_suggestion_id: str | None, armed_at: float | None) -> tuple[object, ...]:
        """
        Arm Finish on a first press, and on a second press within the window finish the drawing and empty the sketch view.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            prompt_text (str | None): Prompt box text at finish, kept as the final prompt.
            prompt_suggestion_id (str | None): ID of the prompt suggestion written into the prompt box, or None.
            armed_at (float | None): When this tab's Finish was pressed first, or None.

        Returns:
            tuple[object, ...]: The study drawing outputs, the sketch view, the drawing this tab shows, and the armed time.
        """
        # A first press only asks for a second, leaving everything else as it is.
        now = time.monotonic()
        if not is_finish_armed(armed_at, now):
            unchanged = [gr.skip()] * 5
            return (*unchanged, gr.Button(value=FINISH_CONFIRM_LABEL), gr.skip(), gr.skip(), gr.skip(), now)

        view = self.close_drawing(choice, upload_path, rotation_degrees, prompt_text or "", prompt_suggestion_id)
        return *build_study_drawing_outputs(view), gr.Image(value=None), sign_drawing(view), None

    def disarm_finish(self, armed_at: float | None) -> tuple[gr.Button | dict, float | None | dict]:
        """
        Turn an armed Finish back into Finish drawing once the confirmation window has passed, and change nothing otherwise.
        """
        if armed_at is None or is_finish_armed(armed_at, time.monotonic()):
            return gr.skip(), gr.skip()

        return gr.Button(value=FINISH_LABEL), None

    def suggest_study_strokes(self, choice: str, upload_path: str | None, prompt_text: str | None,
                              prompt_suggestion_id: str | None, rotation_degrees: int) -> tuple[object, ...]:
        """
        Suggest next strokes on the study page, telling the participant calmly about a failure and the experimenter its cause.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            prompt_text (str | None): Prompt box text, sent exactly as typed, possibly empty.
            prompt_suggestion_id (str | None): ID of the prompt suggestion written into the prompt box, or None.
            rotation_degrees (int): Degrees to turn the sketch clockwise.

        Returns:
            tuple[object, ...]: The participant status, the sketch view, the experimenter status, and the drawing this tab shows.
        """
        # Show the suggestion, or a calm line for the participant with the full error in the experimenter's status.
        try:
            status, sketch_view = self.suggest_next_strokes(choice, upload_path, prompt_text, prompt_suggestion_id,
                                                            rotation_degrees)
        except gr.Error as error:
            view = self.controller.latest_drawing()
            problem = f"Suggestion failed: {error.message}"
            return (f"{SUGGESTION_FAILED_STATUS}\n\n{DONE_REMINDER}", gr.skip(), describe_experimenter_status(view, problem),
                    sign_drawing(view))

        view = self.controller.latest_drawing()
        return status, sketch_view, describe_experimenter_status(view), sign_drawing(view)

    def suggest_study_prompt(self, choice: str, upload_path: str | None, rotation_degrees: int) -> tuple[object, ...]:
        """
        Write a suggested prompt on the study page, telling the participant calmly about a failure and the experimenter its cause.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.

        Returns:
            tuple[object, ...]: The prompt, its kept ID, the participant status, and the experimenter status, with the prompt
                and ID left as they were after a failure and both statuses left as they were after a success.
        """
        # Show a calm line instead of a popup, and keep the prompt box and its ID as they were.
        try:
            text, suggestion_id = self.suggest_another(choice, upload_path, rotation_degrees)
        except gr.Error as error:
            problem = f"Prompt suggestion failed: {error.message}"
            return (gr.skip(), gr.skip(), f"{PROMPT_FAILED_STATUS}\n\n{DONE_REMINDER}",
                    describe_experimenter_status(self.controller.latest_drawing(), problem))

        return text, suggestion_id, gr.skip(), gr.skip()

    def refresh_study_preview(self, choice: str, upload_path: str | None, rotation_degrees: int) -> np.ndarray | dict:
        """
        Show the paper with the open suggestion's overlay, leaving the view as it is outside a drawing or without paper.
        """
        payload = self.controller.preview_guided_sketch(self.pick_current_sketch(choice, upload_path), int(rotation_degrees or 0))
        return gr.skip() if payload is None else decode_preview(payload)

    def find_study_problem(self, view: DrawingView | None, choice: str) -> str:
        """
        Name what the experimenter must check: a failed save, or a webcam that stopped sending frames in an open drawing.
        """
        if self.capture_failure:
            return self.capture_failure

        is_webcam = SketchSource(choice) == SketchSource.WEBCAM
        return CAMERA_STOPPED_NOTICE if is_drawing_open(view) and is_webcam and self.take_fresh_frame() is None else ""

    def report_problem(self, problem: str) -> None:
        """
        Print one terminal line when a problem starts or changes, and one when everything works again.
        """
        if problem == self.reported_problem:
            return

        # Print without the Markdown emphasis the panel uses.
        plain = problem.replace("**", "")
        print(f"Study page check needed: {plain}" if problem else "Study page: saving and the webcam work again.", flush=True)
        self.reported_problem = problem

    def save_study_capture(self, choice: str, upload_path: str | None, rotation_degrees: int, shown_notice: str | None,
                           shown_drawing: DrawingSignature) -> tuple[object, ...]:
        """
        Save the open drawing when due, flag a failed save or a stopped webcam, and follow drawing changes from other tabs.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            shown_notice (str | None): The notice this tab shows.
            shown_drawing (DrawingSignature): The drawing state this tab's participant status and buttons show.

        Returns:
            tuple[object, ...]: The notice, the panel, the participant status, both AI buttons, and the drawing this tab
                shows, each a skip when unchanged.
        """
        # Save when due, then name any problem once in the terminal.
        self.save_due_capture(choice, upload_path, rotation_degrees)
        view = self.controller.latest_drawing()
        problem = self.find_study_problem(view, choice)
        self.report_problem(problem)

        # Update the notice and the panel only when the problem changes, so ticks never flicker.
        if (shown_notice or "") == problem:
            notice_updates: tuple[object, ...] = (gr.skip(), gr.skip())
        else:
            notice_updates = (gr.Markdown(value=problem, visible=bool(problem)), style_experimenter_panel(bool(problem)))

        # Follow a start, finish, or suggestion from another tab, and leave this tab's status alone otherwise.
        signature = sign_drawing(view)
        if signature == shown_drawing:
            return *notice_updates, gr.skip(), gr.skip(), gr.skip(), gr.skip()

        shows_ai = offers_ai_help(view)
        return (*notice_updates, describe_participant_status(view), gr.Button(visible=shows_ai),
                gr.Button(visible=shows_ai, interactive=shows_ai), signature)

    def restore_study_view(self) -> tuple[object, ...]:
        """
        Show the latest drawing on page load, marking an open one so the script closes the panel once the webcam streams.
        """
        view = self.controller.latest_drawing()
        status, *outputs = build_study_drawing_outputs(view)
        marked_status = gr.Markdown(value=status, elem_classes=[CLOSE_PANEL_CLASS] if is_drawing_open(view) else [])
        return marked_status, *outputs, gr.Button(value=START_LABEL, interactive=False), sign_drawing(view)

    def describe_system(self) -> str:
        """
        Describe the backend, both suggesters, and the shared session for the experimenter, with a warning per stand-in.
        """
        controller = self.controller
        lines = [controller.describe_backend(), controller.describe_stroke_suggester(), controller.describe_prompt_suggester(),
                 SHARED_SESSION_NOTE, *(f"**Stand-in:** {warning}" for warning in controller.list_stand_in_warnings())]
        return "\n".join(f"- {line}" for line in lines)

    def _build_layout(self) -> tuple[ParticipantArea, ExperimenterPanel]:
        """
        Lay out the logo and title, the sketch with the camera and its settings below it, and the experiment sidebar.
        """
        # A logo placeholder beside the title.
        with gr.Row(elem_id="header"):
            gr.HTML('<div id="logo">SL</div>', elem_id="logo-box")
            gr.Markdown("# SketchLoop")

        with gr.Row(equal_height=False):
            # The large sketch, and below it the camera beside its source and rotation.
            with gr.Column(scale=3):
                sketch_view = gr.Image(label=OPEN_DRAWING_LABEL, interactive=False, height=STUDY_SKETCH_HEIGHT, buttons=[],
                                       elem_id="study-sketch-view")
                with gr.Row(equal_height=True):
                    webcam = gr.Image(label="Camera", sources=["webcam"], streaming=True, type="numpy",
                                      webcam_options=gr.WebcamOptions(mirror=False), height=CAMERA_HEIGHT, elem_id="webcam",
                                      elem_classes=[f"rotate-{DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM]}"])
                    upload = gr.Image(label="Sketch file", sources=["upload"], type="filepath", image_mode=None,
                                      visible=False, height=CAMERA_HEIGHT)
                    with gr.Column(elem_classes="card"):
                        source = gr.Radio(choices=SOURCE_CHOICES, value=SketchSource.WEBCAM, label="Source")
                        rotation = gr.Dropdown(choices=list(Rotation), value=DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM],
                                               label="Clockwise rotation")

            # The experiment sidebar, from the name and condition down to the suggestions, with the fixed info at the end.
            with gr.Column(scale=1, min_width=STUDY_CONTROLS_WIDTH, elem_id="experimenter", elem_classes="card") as panel:
                gr.Markdown(f"### {EXPERIMENTER_LABEL}")
                name = gr.Textbox(label="Experiment name", info=EXPERIMENT_NAME_HINT, max_lines=1)
                condition = gr.Radio(choices=CONDITION_CHOICES, value=None, label="Condition")
                prompt = gr.Textbox(label="Prompt", placeholder=STUDY_PROMPT_PLACEHOLDER, lines=3)
                with gr.Row():
                    start = gr.Button(START_LABEL, interactive=False)
                    finish = gr.Button(FINISH_LABEL, variant="stop", interactive=False)
                with gr.Column(elem_id="ai-help"):
                    suggest_prompt = gr.Button("Suggest a prompt", visible=False, elem_id="suggest-prompt")
                    suggest_strokes = gr.Button("Suggest next strokes", variant="primary", visible=False,
                                                elem_id="suggest-strokes")
                status = gr.Markdown(WAITING_STATUS, visible=False)
                experimenter_status = gr.Markdown(NO_EXPERIMENT_STATUS, elem_id="experimenter-status")
                capture_notice = gr.Markdown(visible=False, elem_id="capture-notice")
                gr.Markdown(self.describe_system(), elem_id="system")

        participant = ParticipantArea(sketch_view=sketch_view, prompt=prompt, prompt_suggestion=gr.State(None),
                                      suggest_prompt=suggest_prompt, suggest_strokes=suggest_strokes, status=status,
                                      shown_drawing=gr.State(None))
        experimenter = ExperimenterPanel(panel=panel, name=name, condition=condition, start=start, finish=finish,
                                         status=experimenter_status, capture_notice=capture_notice, source=source,
                                         rotation=rotation, webcam=webcam, upload=upload, finish_armed=gr.State(None))
        return participant, experimenter

    def _wire_study_events(self, app: gr.Blocks, participant: ParticipantArea, experimenter: ExperimenterPanel) -> None:
        """
        Wire the study page's camera, start, finish, suggestion, timer, and page load events.

        Args:
            app (gr.Blocks): The page being built.
            participant (ParticipantArea): The participant's components.
            experimenter (ExperimenterPanel): The experimenter's components.
        """
        # Switch inputs, turn the preview, and keep webcam frames.
        sketch_inputs = [experimenter.source, experimenter.upload, experimenter.rotation]
        experimenter.source.change(self.toggle_source, [experimenter.source],
                                   [experimenter.webcam, experimenter.upload, experimenter.rotation])
        experimenter.rotation.change(self.turn_preview, experimenter.rotation, [experimenter.webcam], show_progress="hidden")
        experimenter.webcam.stream(self.store_frame, [experimenter.webcam], None, stream_every=self.capture_interval_seconds,
                                   show_progress="hidden", time_limit=None)

        # Start needs a name and a condition and names the condition, and it runs in the session queue with errors as one line.
        drawing_outputs = [participant.status, participant.suggest_prompt, participant.suggest_strokes, experimenter.name,
                           experimenter.condition, experimenter.finish, experimenter.status]
        start_inputs = [experimenter.name, experimenter.condition]
        for component in start_inputs:
            component.change(self.update_study_start_button, start_inputs, [experimenter.start], show_progress="hidden")
        start_outputs = [*drawing_outputs, experimenter.start, participant.sketch_view, participant.prompt,
                         participant.prompt_suggestion, experimenter.panel, participant.shown_drawing, experimenter.finish_armed]
        experimenter.start.click(self.start_study_drawing, start_inputs, start_outputs, concurrency_id="session",
                                 concurrency_limit=1, show_progress="hidden")

        # Finish takes two presses within a few seconds, and a timer turns an unconfirmed first press back.
        prompt_inputs = [participant.prompt, participant.prompt_suggestion]
        finish_outputs = [*drawing_outputs, participant.sketch_view, participant.shown_drawing, experimenter.finish_armed]
        experimenter.finish.click(self.press_finish, [*sketch_inputs, *prompt_inputs, experimenter.finish_armed], finish_outputs,
                                  concurrency_id="session", concurrency_limit=1, show_progress="hidden")
        finish_timer = gr.Timer(1)
        finish_timer.tick(self.disarm_finish, [experimenter.finish_armed], [experimenter.finish, experimenter.finish_armed],
                          show_progress="hidden")

        # Write a suggested prompt and keep its ID in the session queue, so it cannot cross a Finish and the next Start.
        participant.suggest_prompt.click(self.suggest_study_prompt, sketch_inputs,
                                         [*prompt_inputs, participant.status, experimenter.status], concurrency_id="session",
                                         concurrency_limit=1)

        # Say at once that a suggestion is running, then show it or a calm failure line, and re-enable the button.
        status = participant.status
        waiting = participant.suggest_strokes.click(self.show_suggesting, None, [status, participant.suggest_strokes],
                                                    queue=False, show_progress="hidden")
        suggest_inputs = [experimenter.source, experimenter.upload, *prompt_inputs, experimenter.rotation]
        suggesting = waiting.then(self.suggest_study_strokes, suggest_inputs,
                                  [status, participant.sketch_view, experimenter.status, participant.shown_drawing],
                                  concurrency_id="session", concurrency_limit=1, show_progress="minimal",
                                  show_progress_on=[status])
        suggesting.then(self.update_suggest_button, None, [participant.suggest_strokes], show_progress="hidden")

        # Follow the paper at the capture interval, and save, check the camera, and follow other tabs at the save interval.
        frame_timer = gr.Timer(self.capture_interval_seconds)
        frame_timer.tick(self.refresh_study_preview, sketch_inputs, [participant.sketch_view], show_progress="hidden")
        capture_timer = gr.Timer(self.controller.save_interval_seconds)
        capture_outputs = [experimenter.capture_notice, experimenter.panel, participant.status, participant.suggest_prompt,
                           participant.suggest_strokes, participant.shown_drawing]
        capture_timer.tick(self.save_study_capture, [*sketch_inputs, experimenter.capture_notice, participant.shown_drawing],
                           capture_outputs, concurrency_id="drawing-capture", concurrency_limit=1, show_progress="hidden")

        # Restore the latest drawing on load.
        app.load(self.restore_study_view, None, [*drawing_outputs, experimenter.start, participant.shown_drawing])

    def _build_study_page(self, app: gr.Blocks) -> None:
        """
        Lay out and wire the study page: the title, the participant area, and the experimenter panel below it.
        """
        participant, experimenter = self._build_layout()

        self._wire_study_events(app, participant, experimenter)

    def build(self) -> gr.Blocks:
        """
        Build the page and wire its events, without launching it.
        """
        with gr.Blocks(title="SketchLoop", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
            self._build_study_page(app)

        return app


def serve_app(app: gr.Blocks, controller: AppController) -> None:
    """
    Serve a built page on this machine only, with no share link, the theme, CSS, and script, and access to the runs folder only.

    Args:
        app (gr.Blocks): Page built by SketchPage.
        controller (AppController): Service whose store holds the runs folder.
    """
    # Show only the "Built with Gradio" footer link and keep no call history, so participants find no API or settings pages.
    head = f"<script>{SCRIPT_PATH.read_text(encoding='utf-8')}</script>"
    app.launch(server_name=SERVER_HOST, share=False, ssr_mode=False, inbrowser=True, theme=THEME, css_paths=[STYLESHEET_PATH],
               head=head, allowed_paths=[str(controller.store.root.resolve())], max_file_size="20mb", show_error=False,
               footer_links=STUDY_FOOTER_LINKS, run_history=False)


def launch_app(controller: AppController, *, capture_interval_seconds: float = 0.5) -> None:
    """
    Build the page for the controller and serve it on this machine.

    Args:
        controller (AppController): Service holding the generator and the session.
        capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.
    """
    page = SketchPage(controller, capture_interval_seconds=capture_interval_seconds)
    serve_app(page.build(), controller)
