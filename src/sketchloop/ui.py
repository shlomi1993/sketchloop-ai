"""
The Gradio browser page, a thin layer over AppController for webcam or upload rounds and guided drawings.
"""

import cv2
import gradio as gr
import numpy as np
import threading
import time

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from sketchloop.app import NO_FRAME_MESSAGE, AppController, DrawingView, HistoryRow, RoundView, SuggestionView
from sketchloop.domain import ControlValue, Rotation, SketchLoopError, SketchSource
from sketchloop.experiments import StudyCondition
from sketchloop.generation import ControlKind, ControlSpec


HISTORY_HEADERS: Final = ["Round", "Status", "Prompt", "Outcome"]
SOURCE_CHOICES: Final = [("Live webcam", SketchSource.WEBCAM), ("Upload a file", SketchSource.UPLOAD)]
PROMPT_PLACEHOLDER: Final = "Describe the style, for example: a sculptural oak lounge chair, soft studio light"

# A webcam facing the person sees the paper upside down.
DEFAULT_ROTATION_BY_SOURCE: Final = {SketchSource.WEBCAM: Rotation.DEG_180, SketchSource.UPLOAD: Rotation.DEG_000}

# Starting slider values that replace a backend control's default, such as 10 steps in either mode.
PAGE_CONTROL_DEFAULTS: Final = {"steps": 10}

# The page's CSS and script, shipped as package data.
STATIC_DIR: Final = Path(__file__).parent / "static"
STYLESHEET_PATH: Final = STATIC_DIR / "ui.css"
SCRIPT_PATH: Final = STATIC_DIR / "ui.js"

# Status lines that guide each step.
NO_ROUNDS_STATUS: Final = "Draw on paper, point the webcam at it, describe the style, and press **Generate**."
NEW_SESSION_STATUS: Final = "New session started. Its folder is created with the first round."
FAKE_BACKEND_PREFIX: Final = "**Fake backend:** these images are deterministic noise, not output from a real model."
PROGRESS_MESSAGE: Final = "Generating. The first round also loads the model."
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
CONDITION_CHOICES: Final = [("With AI", StudyCondition.AI), ("Without AI", StudyCondition.NO_AI)]
CONDITION_WORDS: Final = {StudyCondition.AI: "with AI", StudyCondition.NO_AI: "without AI"}
CHOOSE_CONDITION_MESSAGE: Final = "Choose a condition, then start the drawing."
NO_EXPERIMENT_STATUS: Final = "No experiment started yet. Enter a name, choose a condition, and press **Start drawing**."
EXPERIMENTER_LABEL: Final = "Experimenter"
SAVING_FAILED_LABEL: Final = "Experimenter: saving failed"

# The study page's sketch view height and its right column's width in pixels, fixed so nothing moves between conditions.
STUDY_SKETCH_HEIGHT: Final = 560
STUDY_CONTROLS_WIDTH: Final = 360

# The largest seed the page accepts, so seed plus candidate index stays within every backend's random generator.
MAX_SEED: Final = 2**31 - 1

# A webcam frame older than this many capture intervals means the stream stopped.
STALE_FRAME_INTERVALS: Final = 10

# Serve only on this machine.
SERVER_HOST: Final = "127.0.0.1"

# A neutral look, so the images stand out.
THEME: Final = gr.themes.Default(primary_hue="indigo", neutral_hue="slate", radius_size="lg")

# Gallery, sketch, picks, iteration ID, and status.
RoundOutputs: TypeAlias = tuple[list[tuple[str, str]], str | None, gr.CheckboxGroup, str | None, str]

# A drawing's experiment, its number, and an uploaded file's path, so an unchanged upload is saved once per drawing.
UploadCaptureKey: TypeAlias = tuple[str | None, int, str]

# Drawing status, Suggest next strokes, Finish drawing, Start a new session, the name box, and the sketch view's style.
DrawingOutputs: TypeAlias = tuple[str, gr.Button, gr.Button, gr.Button, gr.Textbox, gr.Image]

# Participant status, Suggest a prompt, Suggest next strokes, the name box, the condition, Finish, and the experimenter status.
StudyDrawingOutputs: TypeAlias = tuple[str, gr.Button, gr.Button, gr.Textbox, gr.Radio, gr.Button, str]


def describe_round_status(view: RoundView) -> str:
    """
    Say what the person can do next with a round, warning first when its images are fake.
    """
    # Say what to do next, warning first about fake images.
    if view.selection is None:
        status = f"Round {view.number} is ready after {view.total_seconds:.1f} s. Tick the candidates you like, or none, and record your choice."  # noqa: E501
    else:
        status = f"Round {view.number} selected: {view.selection}. Change the sketch, prompt, or settings and generate the next round."  # noqa: E501

    return f"{FAKE_BACKEND_PREFIX} {status}" if view.is_fake else status


def build_round_outputs(view: RoundView | None, empty_status: str = NO_ROUNDS_STATUS) -> RoundOutputs:
    """
    Turn a round view into the gallery, sketch, choices, iteration ID, and status the page shows.

    Args:
        view (RoundView | None): Round to show, or None to clear the display.
        empty_status (str, optional): Status shown when there is no round. Defaults to NO_ROUNDS_STATUS.

    Returns:
        RoundOutputs: Values for the round display components.
    """
    # Clear the display before the first round.
    if view is None:
        return [], None, gr.CheckboxGroup(choices=[], value=[]), None, empty_status

    gallery = [(str(candidate.image_path), candidate.label) for candidate in view.candidates]
    picks = gr.CheckboxGroup(choices=[(candidate.label, candidate.candidate_id) for candidate in view.candidates], value=[])
    return gallery, str(view.sketch_path), picks, view.iteration_id, describe_round_status(view)


def can_generate(choice: str, has_webcam_frame: bool, upload_path: str | None, prompt_text: str | None) -> bool:
    """
    Tell whether the chosen source has a sketch and the prompt is not blank, so generation can start.
    """
    has_sketch = has_webcam_frame if choice == SketchSource.WEBCAM else upload_path is not None
    return has_sketch and bool((prompt_text or "").strip())


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


def count_suggestions(count: int) -> str:
    """
    Say a suggestion count in words, such as "1 suggestion" or "3 suggestions".
    """
    return f"{count} suggestion" if count == 1 else f"{count} suggestions"


def describe_drawing_status(view: DrawingView | None, suggestion: SuggestionView | None = None) -> str:
    """
    Name the experiment and the drawing's state, with the latest suggestion's line and a fake-suggester warning.

    Args:
        view (DrawingView | None): The latest drawing, or None before the first.
        suggestion (SuggestionView | None, optional): The suggestion just shown, if any. Defaults to None.

    Returns:
        str: One status line for the drawing controls.
    """
    if view is None:
        return NO_DRAWING_STATUS

    drawing = f"Experiment {view.experiment}, drawing {view.number}" if view.experiment else f"Drawing {view.number}"
    if not view.is_open:
        return f"{drawing} is finished with {count_suggestions(view.suggestion_count)}. Start a new experiment to draw again."

    # Show the suggestion just made, warning first when it is fake, or the latest one after a reload.
    if suggestion is not None:
        status = f"{drawing}. {suggestion.status}"
        return f"{FAKE_SUGGESTER_PREFIX} {status}" if suggestion.is_fake else status

    if view.suggestion_lines:
        return f"{drawing}. {view.suggestion_lines[-1]}"

    return f"{drawing} is open. Draw on paper and press **Suggest next strokes** whenever you want ideas."


def style_sketch_view(view: DrawingView | None, **update: object) -> gr.Image:
    """
    Label, widen, and heighten the sketch view while a drawing is open, naming the suggestions once there are any.
    """
    if is_drawing_open(view):
        label = GUIDED_PREVIEW_LABEL if view.suggestion_count else OPEN_DRAWING_LABEL
        return gr.Image(label=label, height=GUIDED_PREVIEW_HEIGHT, elem_classes=["guided"], **update)

    return gr.Image(label=PLAIN_PREVIEW_LABEL, height=PLAIN_PREVIEW_HEIGHT, elem_classes=[], **update)


def build_drawing_outputs(view: DrawingView | None, *, clear_name: bool = False) -> DrawingOutputs:
    """
    Turn the latest drawing into its status, the drawing buttons, the name box, and the sketch view's style.

    Args:
        view (DrawingView | None): The latest drawing, or None before the first.
        clear_name (bool, optional): Empty the name box, as after a start. Defaults to False.

    Returns:
        DrawingOutputs: Values for the drawing components.
    """
    # While a drawing is open, lock the name box and Start a new session, so nothing ends the drawing by accident.
    is_open = is_drawing_open(view)
    name_update = {"value": ""} if clear_name else {}
    name_box = gr.Textbox(info=FINISH_FIRST_MESSAGE if is_open else EXPERIMENT_NAME_HINT, interactive=not is_open, **name_update)
    return (describe_drawing_status(view), gr.Button(interactive=is_open), gr.Button(interactive=is_open),
            gr.Button(interactive=not is_open), name_box, style_sketch_view(view))


def can_start_study_drawing(name: str | None, condition: str | None, is_open: bool) -> bool:
    """
    Tell whether a name was typed, a condition chosen, and no drawing is open, so a study drawing can start.
    """
    return can_start_drawing(name, is_open) and bool(condition)


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
        str: One status line for the participant.
    """
    if view is None:
        return WAITING_STATUS

    if not view.is_open:
        return FINISHED_STATUS

    # Show the suggestion just made, warning first when it is fake, or the latest one after a reload.
    if suggestion is not None:
        return f"{FAKE_SUGGESTER_PREFIX} {suggestion.status}" if suggestion.is_fake else suggestion.status

    return view.suggestion_lines[-1] if view.suggestion_lines else DRAWING_STATUS


def describe_experimenter_status(view: DrawingView | None) -> str:
    """
    Name the experiment, its condition, its start time, and whether it is open, such as "p1_car, without AI, started 10:02".
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
    return ", ".join(parts)


def label_experimenter_panel(saving_failed: bool) -> str:
    """
    Label the experimenter panel, naming a failed save so it shows while the panel is closed.
    """
    return SAVING_FAILED_LABEL if saving_failed else EXPERIMENTER_LABEL


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
    return (describe_participant_status(view), gr.Button(visible=shows_ai), gr.Button(visible=shows_ai, interactive=shows_ai),
            name_box, condition, gr.Button(interactive=is_open), describe_experimenter_status(view))


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


def build_history_table(rows: Sequence[HistoryRow]) -> list[list[str | int]]:
    """
    Turn history rows into table rows, showing a missing prompt as not recorded.
    """
    return [[row.number, row.status, "not recorded" if row.prompt is None else row.prompt, row.outcome] for row in rows]


def make_control_slider(spec: ControlSpec) -> gr.Slider:
    """
    Build a slider for a backend control within its bounds, starting at its default, such as "Steps" from 1 to 50.
    """
    step = 1 if spec.kind == ControlKind.INT else 0.05
    label = spec.name.replace("_", " ").capitalize()
    value = PAGE_CONTROL_DEFAULTS.get(spec.name, spec.default)
    return gr.Slider(minimum=spec.minimum, maximum=spec.maximum, step=step, value=value, label=label)


def read_seed(seed: str | float | None) -> int | None:
    """
    Read the seed field, where blank means a random seed and anything else must be a whole number from 0 to MAX_SEED.
    """
    text = "" if seed is None else str(seed).strip()
    if not text:
        return None

    # Parse digits exactly, since a float loses precision on long numbers, and refuse anything else with one line.
    if not text.isdecimal() or int(text) > MAX_SEED:
        raise gr.Error(f"The seed must be a whole number from 0 to {MAX_SEED}, or blank for random, got {text!r}.")

    return int(text)


def read_control_values(specs: Sequence[ControlSpec], values: Sequence[float]) -> dict[str, ControlValue]:
    """
    Pair each slider value with its control, as an int for integer controls, which Gradio sends as floats.

    Args:
        specs (Sequence[ControlSpec]): Controls the sliders show, in slider order.
        values (Sequence[float]): Slider values, in the same order.

    Returns:
        dict[str, ControlValue]: Requested value by control name.
    """
    return {spec.name: int(value) if spec.kind == ControlKind.INT else float(value) for spec, value in zip(specs, values)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SketchCard:
    """
    Components of step 1, where the person picks and shows the sketch.
    """
    source: gr.Radio  # Webcam or upload choice
    webcam: gr.Image  # Live webcam preview that streams frames
    upload: gr.Image  # Uploaded sketch file
    rotation: gr.Dropdown  # Degrees to turn the sketch clockwise
    raw: gr.Checkbox  # Skips preprocessing and rotation


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionCard:
    """
    Components of step 2, the prompt, the Generate button, and the settings.
    """
    prompt: gr.Textbox  # Style prompt
    prompt_suggestion: gr.State  # ID of the prompt suggestion written into the prompt box, None until one is
    suggest: gr.Button  # Writes another suggested style into the prompt
    generate: gr.Button  # Starts a round
    candidates: gr.Slider  # Candidates per round
    seed: gr.Textbox  # Whole-number seed, or blank for random
    controls: tuple[gr.Slider, ...]  # One slider per backend control, in AppController.slider_controls order


@dataclass(frozen=True, slots=True, kw_only=True)
class ChooseCard:
    """
    Components of step 3, the round display and the choice, plus the iteration ID the page shows.
    """
    status: gr.Markdown  # What to do next
    sketch_view: gr.Image  # Sketch the model will receive, or received in the shown round
    gallery: gr.Gallery  # Candidate images
    picks: gr.CheckboxGroup  # Candidates the person ticks
    record: gr.Button  # Records the choice
    new_session: gr.Button  # Starts a new session
    round_state: gr.State  # Iteration ID of the shown round
    drawing_status: gr.Markdown  # Experiment, drawing state, and the latest suggestion
    suggest_strokes: gr.Button  # Suggests next strokes for the open drawing


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentCard:
    """
    Experimenter controls, away from the participant's buttons: the experiment name, Start drawing, and Finish drawing.
    """
    name: gr.Textbox  # Experiment name, which also names the session folder
    start: gr.Button  # Starts a named experiment and its drawing
    finish: gr.Button  # Finishes the open drawing with the current sketch as its final capture
    capture_notice: gr.Markdown  # A failed interval save, hidden while saving works


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimenterPanel:
    """
    The study page's collapsible experimenter controls, away from the participant's view.
    """
    panel: gr.Accordion  # The panel, closed after each start and relabeled when saving fails
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


class SketchPage:
    """
    The browser page, with handlers that share the controller and the newest webcam frame.
    """

    def __init__(self, controller: AppController, *, default_candidates: int, default_seed: int | None,
                 capture_interval_seconds: float = 0.5, study: bool = False) -> None:
        """
        Store the controller and the page defaults.

        Args:
            controller (AppController): Service holding the generator and the session.
            default_candidates (int): Candidates per round shown at start.
            default_seed (int | None): Seed shown at start, or None for random.
            capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.
            study (bool, optional): Build the study page, with the participant area and the experimenter panel, instead of
                the generation page. Defaults to False.
        """
        self.controller = controller
        self.default_candidates = default_candidates
        self.default_seed = default_seed
        self.capture_interval_seconds = capture_interval_seconds
        self.study = study

        # The newest webcam frame in BGR order and when it arrived, shared by all tabs and read together under the lock.
        self.frame_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_time = 0.0

        # The failed interval save the experimenter sees, empty while saving works, and the upload saved last.
        self.capture_failure = ""
        self.saved_upload: UploadCaptureKey | None = None

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

    def toggle_raw(self, use_raw: bool, choice: str) -> gr.Dropdown:
        """
        Lock the rotation at 0 for the raw sketch, and restore the source's default otherwise.
        """
        return gr.Dropdown(value=pick_rotation(choice, use_raw), interactive=not use_raw)

    def turn_preview(self, degrees: int | None) -> gr.Image:
        """
        Turn the webcam preview by switching its rotation CSS class.
        """
        return gr.Image(elem_classes=[f"rotate-{int(degrees or 0)}"])

    def update_generate_button(self, choice: str, upload_path: str | None, prompt_text: str | None) -> gr.Button:
        """
        Enable Generate once the chosen source has a sketch and the prompt is not blank.
        """
        has_webcam_frame = self.take_fresh_frame() is not None
        return gr.Button(interactive=can_generate(choice, has_webcam_frame, upload_path, prompt_text))

    def store_frame(self, frame: np.ndarray | None) -> None:
        """
        Keep the newest streamed webcam frame, converted from RGB to BGR, and when it arrived, ignoring a missing one.
        """
        if frame is not None:
            bgr_frame = np.ascontiguousarray(frame[:, :, ::-1])
            with self.frame_lock:
                self.latest_frame = bgr_frame
                self.latest_frame_time = time.monotonic()

    def keep_frame(self, frame: np.ndarray | None, choice: str, upload_path: str | None, prompt_text: str | None) -> gr.Button:
        """
        Keep the newest streamed webcam frame and update Generate.

        Args:
            frame (np.ndarray | None): RGB frame from the browser, or None when none arrived.
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            prompt_text (str | None): Style prompt.

        Returns:
            gr.Button: Generate, enabled when generation can start.
        """
        self.store_frame(frame)
        return self.update_generate_button(choice, upload_path, prompt_text)

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

    def refresh_webcam_preview(self, choice: str, upload_path: str | None, rotation_degrees: int,
                               use_raw: bool) -> np.ndarray | dict | None:
        """
        Show the newest webcam frame as the generator would receive it, leaving an upload's preview as it is.
        """
        if SketchSource(choice) != SketchSource.WEBCAM:
            return gr.skip()

        return self.show_sketch_preview(choice, upload_path, rotation_degrees, use_raw)

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

    def run_generation(self, choice: str, upload_path: str | None, prompt_text: str, prompt_suggestion_id: str | None,
                       candidate_count: float, seed: str | float | None, rotation_degrees: int, use_raw: bool,
                       *control_values: float, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> RoundOutputs:
        """
        Generate a round from the webcam frame or the upload, showing any error as one line.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            prompt_text (str): Style prompt.
            prompt_suggestion_id (str | None): ID of the prompt suggestion written into the prompt box, or None.
            candidate_count (float): Candidates to generate.
            seed (str | float | None): Seed as typed, or blank or None for random.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            use_raw (bool): Whether to skip preprocessing and rotation.
            *control_values (float): Slider values, in AppController.slider_controls order.
            progress (gr.Progress, optional): Progress bar Gradio fills in. Defaults to one that tracks tqdm.

        Returns:
            RoundOutputs: Values for the round display components.
        """
        # Pick the sketch for the chosen source, refusing a webcam frame from before the stream stopped.
        progress(None, desc=PROGRESS_MESSAGE)
        source = SketchSource(choice)
        sketch, frame_age_seconds = self.take_sketch(choice, upload_path)

        # Generate with the chosen controls, showing any error.
        controls = read_control_values(self.controller.slider_controls, control_values)
        try:
            view = self.controller.run_round(sketch, source, prompt_text, n_candidates=int(candidate_count),
                                             seed=read_seed(seed), rotation=int(rotation_degrees or 0),
                                             use_raw=use_raw, frame_age_seconds=frame_age_seconds, controls=controls,
                                             prompt_suggestion_id=prompt_suggestion_id)
        except (SketchLoopError, OSError) as error:
            raise gr.Error(str(error)) from error

        return build_round_outputs(view)

    def save_selection(self, iteration_id: str | None, candidate_ids: list[str]) -> str:
        """
        Record the person's choice for the shown round and return the new status.

        Args:
            iteration_id (str | None): Iteration the page showed.
            candidate_ids (list[str]): Candidates the person ticked.

        Returns:
            str: Status line for the round.
        """
        # A stale page is refused, so the person reloads.
        try:
            view = self.controller.record_selection(iteration_id, candidate_ids)
        except (SketchLoopError, OSError) as error:
            raise gr.Error(str(error)) from error

        return describe_round_status(view)

    def clear_rounds(self, status: str) -> tuple[object, ...]:
        """
        Clear the round display of a replaced session, keeping the sketch view, and show the new session's history.
        """
        gallery, _, picks, iteration_id, status = build_round_outputs(None, status)
        return gallery, picks, iteration_id, status, build_history_table(self.controller.list_history())

    def start_session(self) -> tuple[object, ...]:
        """
        Start a new session, which ends an open drawing, clear the round display and history, and forget the kept prompt ID.
        """
        self.controller.start_new_session()
        return *self.clear_rounds(NEW_SESSION_STATUS), *build_drawing_outputs(self.controller.latest_drawing()), None

    def restore_view(self) -> tuple[object, ...]:
        """
        Show the latest round, the history, and the latest drawing, as on page load.
        """
        gallery, sketch_path, picks, iteration_id, status = build_round_outputs(self.controller.latest_round())
        drawing = self.controller.latest_drawing()
        *drawing_controls, _ = build_drawing_outputs(drawing)
        return (gallery, style_sketch_view(drawing, value=sketch_path), picks, iteration_id, status,
                build_history_table(self.controller.list_history()), *drawing_controls)

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

    def start_drawing(self, name: str | None) -> tuple[object, ...]:
        """
        Start a named experiment with its first drawing, clearing the name, the prompt, the upload, and the previous rounds.

        Args:
            name (str | None): Experiment name as typed.

        Returns:
            tuple[object, ...]: The drawing outputs, Start drawing, the cleared prompt, prompt ID, and upload, and the cleared
                rounds.
        """
        view = self.open_drawing(name)

        # Clear the name, and the previous participant's prompt, its kept ID, and upload, so nothing carries over.
        cleared_inputs = (gr.Button(interactive=False), gr.Textbox(value=""), None, gr.Image(value=None))
        return *build_drawing_outputs(view, clear_name=True), *cleared_inputs, *self.clear_rounds(NO_ROUNDS_STATUS)

    def show_suggesting(self) -> tuple[str, gr.Button]:
        """
        Say that a suggestion is running and disable its button until it ends.
        """
        return SUGGESTING_STATUS, gr.Button(interactive=False)

    def describe_status(self, view: DrawingView | None, suggestion: SuggestionView | None = None) -> str:
        """
        Describe the drawing in the participant's words on the study page, and with the experiment's name otherwise.
        """
        return describe_participant_status(view, suggestion) if self.study else describe_drawing_status(view, suggestion)

    def style_view(self, view: DrawingView | None, **update: object) -> gr.Image:
        """
        Restyle the sketch view for the drawing, except on the study page, whose view keeps one label and height.
        """
        return gr.Image(**update) if self.study else style_sketch_view(view, **update)

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

    def finish_drawing(self, choice: str, upload_path: str | None, rotation_degrees: int, prompt_text: str | None = None,
                       prompt_suggestion_id: str | None = None) -> DrawingOutputs:
        """
        Finish the open drawing with its final capture and prompt, and update the generation page's drawing controls.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            prompt_text (str | None, optional): Prompt box text at finish. Defaults to None.
            prompt_suggestion_id (str | None, optional): ID of the prompt suggestion in the prompt box. Defaults to None.

        Returns:
            DrawingOutputs: Values for the drawing components.
        """
        return build_drawing_outputs(self.close_drawing(choice, upload_path, rotation_degrees, prompt_text, prompt_suggestion_id))

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

    def save_drawing_capture(self, choice: str, upload_path: str | None, rotation_degrees: int,
                             shown_notice: str | None) -> gr.Markdown | dict:
        """
        Save the current sketch into the open drawing at the save interval, updating the notice only when it changes.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            shown_notice (str | None): The failure notice this tab shows.

        Returns:
            gr.Markdown | dict: The notice, shown with a failure and hidden without one, or a skip when it is unchanged.
        """
        # Save while a drawing is open, skipping an upload already saved in it, since saving it again stores a duplicate.
        view = self.controller.latest_drawing()
        upload_key = name_upload_capture(view, choice, upload_path)
        if not is_drawing_open(view):
            self.capture_failure = ""
        elif upload_key is None or upload_key != self.saved_upload:
            self.record_capture_quietly(choice, upload_path, rotation_degrees, upload_key)

        # Leave the page untouched unless the notice changes, so ticks never flicker or repeat a message.
        if (shown_notice or "") == self.capture_failure:
            return gr.skip()

        return gr.Markdown(value=self.capture_failure, visible=bool(self.capture_failure))

    def show_drawing_status(self) -> str:
        """
        Show the latest drawing's status, such as a failed suggestion's line, in place of the waiting line.
        """
        return self.describe_status(self.controller.latest_drawing())

    def update_suggest_button(self) -> gr.Button:
        """
        Enable Suggest next strokes again while a drawing is open.
        """
        return gr.Button(interactive=is_drawing_open(self.controller.latest_drawing()))

    def update_start_button(self, name: str | None) -> gr.Button:
        """
        Enable Start drawing once an experiment name is typed, unless a drawing is open.
        """
        return gr.Button(interactive=can_start_drawing(name, is_drawing_open(self.controller.latest_drawing())))

    def update_study_start_button(self, name: str | None, condition: str | None) -> gr.Button:
        """
        Enable the study page's Start drawing once a name is typed and a condition chosen, unless a drawing is open.
        """
        is_open = is_drawing_open(self.controller.latest_drawing())
        return gr.Button(interactive=can_start_study_drawing(name, condition, is_open))

    def start_study_drawing(self, name: str | None, condition: str | None) -> tuple[object, ...]:
        """
        Start a study drawing with the chosen condition, clearing the name, condition, and prompt, and closing the panel.

        Args:
            name (str | None): Experiment name as typed.
            condition (str | None): Chosen condition, ai or no_ai, or None when none was chosen.

        Returns:
            tuple[object, ...]: The study drawing outputs, Start drawing, the emptied sketch view, the cleared prompt and its
                kept ID, and the closed panel.
        """
        # Refuse a start without a condition, so no drawing gets a condition nobody chose.
        if not condition:
            raise gr.Error(CHOOSE_CONDITION_MESSAGE)

        view = self.open_drawing(name, StudyCondition(condition))

        # Empty the view until the paper is found, clear the previous prompt and its ID, and hide the panel from the participant.
        return (*build_study_drawing_outputs(view, clear_inputs=True), gr.Button(interactive=False), gr.Image(value=None),
                gr.Textbox(value=""), None, gr.Accordion(open=False))

    def finish_study_drawing(self, choice: str, upload_path: str | None, rotation_degrees: int, prompt_text: str | None,
                             prompt_suggestion_id: str | None) -> tuple[object, ...]:
        """
        Finish the open study drawing with its final capture and the prompt box text, and empty the sketch view.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            prompt_text (str | None): Prompt box text at finish, kept as the final prompt.
            prompt_suggestion_id (str | None): ID of the prompt suggestion written into the prompt box, or None.

        Returns:
            tuple[object, ...]: The study drawing outputs and the emptied sketch view.
        """
        view = self.close_drawing(choice, upload_path, rotation_degrees, prompt_text or "", prompt_suggestion_id)
        return *build_study_drawing_outputs(view), gr.Image(value=None)

    def refresh_study_preview(self, choice: str, upload_path: str | None, rotation_degrees: int) -> np.ndarray | dict:
        """
        Show the paper with the open suggestion's overlay, leaving the view as it is outside a drawing or without paper.
        """
        payload = self.controller.preview_guided_sketch(self.pick_current_sketch(choice, upload_path), int(rotation_degrees or 0))
        return gr.skip() if payload is None else decode_preview(payload)

    def save_study_capture(self, choice: str, upload_path: str | None, rotation_degrees: int,
                           shown_notice: str | None) -> tuple[gr.Markdown | dict, gr.Accordion | dict]:
        """
        Save the open drawing at the save interval, naming a failed save in the panel's label as well as in its notice.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            shown_notice (str | None): The failure notice this tab shows.

        Returns:
            tuple[gr.Markdown | dict, gr.Accordion | dict]: The notice and the panel's label, or skips when unchanged.
        """
        notice = self.save_drawing_capture(choice, upload_path, rotation_degrees, shown_notice)
        if not isinstance(notice, gr.Markdown):
            return notice, gr.skip()

        return notice, gr.Accordion(label=label_experimenter_panel(bool(self.capture_failure)))

    def restore_study_view(self) -> tuple[object, ...]:
        """
        Show the latest drawing's statuses and buttons on page load, with Start drawing waiting for a name and condition.
        """
        return *build_study_drawing_outputs(self.controller.latest_drawing()), gr.Button(interactive=False)

    def show_history(self) -> list[list[str | int]]:
        """
        Return the current session's history table.
        """
        return build_history_table(self.controller.list_history())

    def _build_header(self) -> None:
        """
        Show the title, the backend, and the shared-session note.
        """
        gr.Markdown(f"# SketchLoop\nSketch on paper, explore AI alternatives, and choose your direction.\n\n"
                    f"{self.controller.describe_backend()} {self.controller.describe_stroke_suggester()} {SHARED_SESSION_NOTE}",
                    elem_id="header")

    def _build_sketch_card(self) -> SketchCard:
        """
        Lay out step 1, the source choice, the live webcam, the upload, and how the sketch is turned and prepared.
        """
        with gr.Column(scale=3, elem_classes="card"):
            gr.Markdown("### 1 · Sketch", elem_classes="step-title")
            source = gr.Radio(choices=SOURCE_CHOICES, value=SketchSource.WEBCAM, show_label=False, container=False)
            webcam = gr.Image(label="Live webcam", sources=["webcam"], streaming=True, type="numpy",
                              webcam_options=gr.WebcamOptions(mirror=False), height=360, elem_id="webcam",
                              elem_classes=[f"rotate-{DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM]}"])
            upload = gr.Image(label="Sketch file", sources=["upload"], type="filepath", image_mode=None, visible=False,
                              height=360)
            with gr.Row():
                rotation = gr.Dropdown(choices=list(Rotation), value=DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM],
                                       label="Clockwise rotation")
                raw = gr.Checkbox(label="Use the raw sketch, skipping preprocessing and rotation")

        return SketchCard(source=source, webcam=webcam, upload=upload, rotation=rotation, raw=raw)

    def _build_direction_card(self) -> DirectionCard:
        """
        Lay out step 2, the prompt, Generate, and the settings, which are always visible.
        """
        with gr.Column(scale=2, elem_classes="card"):
            gr.Markdown("### 2 · Direction", elem_classes="step-title")
            prompt = gr.Textbox(label="Style prompt", placeholder=PROMPT_PLACEHOLDER, lines=4)
            suggest = gr.Button("Suggest a prompt", size="sm")
            generate = gr.Button("Generate", variant="primary", interactive=False, elem_id="generate")

            # Show the settings under Generate in rows of two, so every choice is in view and the card stays short.
            gr.Markdown("**Settings**")
            with gr.Row():
                candidates = gr.Slider(minimum=1, maximum=self.controller.max_candidates, step=1,
                                       value=self.default_candidates, label="Candidates per round")
                seed = gr.Textbox(label="Seed", value="" if self.default_seed is None else str(self.default_seed),
                                  placeholder="Random")
            specs = self.controller.slider_controls
            sliders: list[gr.Slider] = []
            for start in range(0, len(specs), 2):
                with gr.Row():
                    sliders.extend(make_control_slider(spec) for spec in specs[start:start + 2])
            controls = tuple(sliders)

        return DirectionCard(prompt=prompt, prompt_suggestion=gr.State(None), suggest=suggest, generate=generate,
                             candidates=candidates, seed=seed, controls=controls)

    def _build_choose_card(self) -> ChooseCard:
        """
        Lay out step 3, the status, sketch, candidates, and the choice.
        """
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 3 · Choose", elem_classes="step-title")
            status = gr.Markdown(NO_ROUNDS_STATUS, elem_id="status")

            # Suggest next strokes beside its status, right above the preview that shows the suggestion.
            with gr.Row(equal_height=True):
                drawing_status = gr.Markdown(describe_drawing_status(None), elem_id="drawing-status")
                suggest_strokes = gr.Button("Suggest next strokes", variant="primary", interactive=False, scale=0,
                                            min_width=240, elem_id="suggest-strokes")

            # The received sketch, or the guided preview while a drawing is open, beside the candidates.
            with gr.Row(equal_height=False):
                sketch_view = gr.Image(label=PLAIN_PREVIEW_LABEL, interactive=False, type="filepath", scale=1,
                                       height=PLAIN_PREVIEW_HEIGHT, elem_id="sketch-view")
                gallery = gr.Gallery(label="Candidates", interactive=False, format="png", columns=4, height=320,
                                     object_fit="contain", scale=3)
            picks = gr.CheckboxGroup(label="Candidates you choose")

            # Record the choice or start over.
            with gr.Row():
                record = gr.Button("Record my choice", variant="primary")
                new_session = gr.Button("Start a new session", variant="secondary")

        return ChooseCard(status=status, sketch_view=sketch_view, gallery=gallery, picks=picks, record=record,
                          new_session=new_session, round_state=gr.State(None), drawing_status=drawing_status,
                          suggest_strokes=suggest_strokes)

    def _build_experiment_card(self) -> ExperimentCard:
        """
        Lay out the experimenter's name box with its hint, Start drawing, and Finish drawing, below the participant's steps.
        """
        with gr.Accordion("Experiment", open=True, elem_id="experiment"):
            with gr.Row(equal_height=True):
                name = gr.Textbox(label="Experiment name", info=EXPERIMENT_NAME_HINT, max_lines=1, scale=3)
                start = gr.Button("Start drawing", interactive=False, scale=1)
                finish = gr.Button("Finish drawing", interactive=False, scale=1)
            capture_notice = gr.Markdown(visible=False, elem_id="capture-notice")

        return ExperimentCard(name=name, start=start, finish=finish, capture_notice=capture_notice)

    def _build_history(self) -> gr.Dataframe:
        """
        Lay out the collapsed session history table.
        """
        with gr.Accordion("Session history", open=False):
            return gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, show_label=False)

    def _wire_input_events(self, sketch: SketchCard, direction: DirectionCard) -> None:
        """
        Wire the source, raw, rotation, webcam stream, and Generate readiness events.

        Args:
            sketch (SketchCard): Step 1 components.
            direction (DirectionCard): Step 2 components.
        """
        # Switch inputs, lock rotation for the raw sketch, turn the preview, and keep webcam frames.
        source_outputs = [sketch.webcam, sketch.upload, sketch.rotation]
        sketch.source.change(self.toggle_source, [sketch.source, sketch.raw], source_outputs)
        sketch.raw.change(self.toggle_raw, [sketch.raw, sketch.source], [sketch.rotation])
        sketch.rotation.change(self.turn_preview, sketch.rotation, [sketch.webcam], show_progress="hidden")
        stream_inputs = [sketch.webcam, sketch.source, sketch.upload, direction.prompt]
        sketch.webcam.stream(self.keep_frame, stream_inputs, [direction.generate], stream_every=self.capture_interval_seconds,
                             show_progress="hidden", time_limit=None)

        # Enable Generate once there is a sketch and a prompt.
        readiness_inputs = [sketch.source, sketch.upload, direction.prompt]
        for component in readiness_inputs:
            component.change(self.update_generate_button, readiness_inputs, [direction.generate])

    def _wire_preview_events(self, sketch: SketchCard, choose: ChooseCard) -> None:
        """
        Keep the sketch view on the sketch the next round would use, whenever its input changes or a frame is kept.

        Args:
            sketch (SketchCard): Step 1 components.
            choose (ChooseCard): Step 3 components, whose sketch view shows the preview.
        """
        preview_inputs = [sketch.source, sketch.upload, sketch.rotation, sketch.raw]
        for event in (sketch.source.change, sketch.upload.change, sketch.rotation.change, sketch.raw.change):
            event(self.show_sketch_preview, preview_inputs, [choose.sketch_view], show_progress="hidden")

        # A stream event never finishes, so a timer at the capture interval follows the kept webcam frames instead.
        frame_timer = gr.Timer(self.capture_interval_seconds)
        frame_timer.tick(self.refresh_webcam_preview, preview_inputs, [choose.sketch_view], show_progress="hidden")

    def _wire_session_events(self, app: gr.Blocks, sketch: SketchCard, direction: DirectionCard, choose: ChooseCard,
                             history: gr.Dataframe, experiment: ExperimentCard) -> None:
        """
        Wire Generate, the choice, the new session, the guided drawing buttons, and the page load.

        Args:
            app (gr.Blocks): The page being built.
            sketch (SketchCard): Step 1 components.
            direction (DirectionCard): Step 2 components.
            choose (ChooseCard): Step 3 components.
            history (gr.Dataframe): Session history table.
            experiment (ExperimentCard): Experimenter controls.
        """
        # The round display, and the inputs Generate reads.
        round_outputs = [choose.gallery, choose.sketch_view, choose.picks, choose.round_state, choose.status]
        settings = [direction.candidates, direction.seed, sketch.rotation, sketch.raw, *direction.controls]
        generate_inputs = [sketch.source, sketch.upload, direction.prompt, direction.prompt_suggestion, *settings]

        # Run session changes one at a time.
        direction.generate.click(self.run_generation, generate_inputs, round_outputs, concurrency_id="session",
                                 concurrency_limit=1).then(self.show_history, None, [history])
        choose.record.click(self.save_selection, [choose.round_state, choose.picks], [choose.status],
                            concurrency_id="session", concurrency_limit=1).then(self.show_history, None, [history])
        cleared_rounds = [choose.gallery, choose.picks, choose.round_state, choose.status, history]
        drawing_outputs = [choose.drawing_status, choose.suggest_strokes, experiment.finish, choose.new_session, experiment.name,
                           choose.sketch_view]
        choose.new_session.click(self.start_session, None, [*cleared_rounds, *drawing_outputs, direction.prompt_suggestion],
                                 concurrency_id="session", concurrency_limit=1)

        # Start and finish drawings in the session queue, leaving errors to the one-line message so the name box stays readable.
        start_outputs = [*drawing_outputs, experiment.start, direction.prompt, direction.prompt_suggestion, sketch.upload,
                         *cleared_rounds]
        for event in (experiment.start.click, experiment.name.submit):
            event(self.start_drawing, [experiment.name], start_outputs, concurrency_id="session", concurrency_limit=1,
                  show_progress="hidden")
        experiment.name.change(self.update_start_button, [experiment.name], [experiment.start], show_progress="hidden")
        sketch_inputs = [sketch.source, sketch.upload]
        prompt_inputs = [direction.prompt, direction.prompt_suggestion]
        experiment.finish.click(self.finish_drawing, [*sketch_inputs, sketch.rotation, *prompt_inputs], drawing_outputs,
                                concurrency_id="session", concurrency_limit=1, show_progress="hidden")

        # Say at once that a suggestion is running, with a small timer, then show it or the failure line, and re-enable it.
        status = choose.drawing_status
        waiting = choose.suggest_strokes.click(self.show_suggesting, None, [status, choose.suggest_strokes], queue=False,
                                               show_progress="hidden")
        suggesting = waiting.then(self.suggest_next_strokes, [*sketch_inputs, *prompt_inputs, sketch.rotation],
                                  [status, choose.sketch_view], concurrency_id="session", concurrency_limit=1,
                                  show_progress="minimal", show_progress_on=[status])
        suggesting.then(self.update_suggest_button, None, [choose.suggest_strokes], show_progress="hidden")
        suggesting.failure(self.show_drawing_status, None, [status], show_progress="hidden")

        # Save the open drawing at the save interval in its own queue, so captures never wait behind a round or a suggestion.
        capture_timer = gr.Timer(self.controller.save_interval_seconds)
        capture_inputs = [*sketch_inputs, sketch.rotation, experiment.capture_notice]
        capture_timer.tick(self.save_drawing_capture, capture_inputs, [experiment.capture_notice],
                           concurrency_id="drawing-capture", concurrency_limit=1, show_progress="hidden")

        # Write a suggested style on request, one model call at a time.
        suggest_inputs = [sketch.source, sketch.upload, sketch.rotation, sketch.raw]
        direction.suggest.click(self.suggest_another, suggest_inputs, prompt_inputs, concurrency_id="suggest",
                                concurrency_limit=1)

        # Restore the latest round and drawing on load, even during a slow round.
        app.load(self.restore_view, None, [*round_outputs, history, *drawing_outputs[:-1]])

    def _build_participant_area(self) -> ParticipantArea:
        """
        Lay out the study page's large sketch view beside a fixed-width column with the prompt, the AI buttons, and the status.
        """
        with gr.Row(equal_height=False):
            sketch_view = gr.Image(label=OPEN_DRAWING_LABEL, interactive=False, height=STUDY_SKETCH_HEIGHT, buttons=[],
                                   elem_id="study-sketch-view")
            with gr.Column(scale=0, min_width=STUDY_CONTROLS_WIDTH, elem_id="participant-controls"):
                prompt = gr.Textbox(label="Prompt", placeholder=STUDY_PROMPT_PLACEHOLDER, lines=3)

                # Reserve the AI buttons' space in both conditions, so removing them moves nothing below.
                with gr.Column(elem_id="ai-help"):
                    suggest_prompt = gr.Button("Suggest a prompt", visible=False, elem_id="suggest-prompt")
                    suggest_strokes = gr.Button("Suggest next strokes", variant="primary", visible=False,
                                                elem_id="suggest-strokes")
                status = gr.Markdown(WAITING_STATUS, elem_id="participant-status")

        return ParticipantArea(sketch_view=sketch_view, prompt=prompt, prompt_suggestion=gr.State(None),
                               suggest_prompt=suggest_prompt, suggest_strokes=suggest_strokes, status=status)

    def describe_system(self) -> str:
        """
        Describe the backend, both suggesters, and the shared session for the experimenter, with a warning per stand-in.
        """
        controller = self.controller
        lines = [controller.describe_backend(), controller.describe_stroke_suggester(), controller.describe_prompt_suggester(),
                 SHARED_SESSION_NOTE, *(f"**Stand-in:** {warning}" for warning in controller.list_stand_in_warnings())]
        return "\n".join(f"- {line}" for line in lines)

    def _build_experimenter_panel(self) -> ExperimenterPanel:
        """
        Lay out the experimenter panel with the start row, the statuses, the camera, and the system description.
        """
        # Open on load, since Gradio renders a closed accordion only once opened, and the webcam keeps streaming once it closes.
        with gr.Accordion(EXPERIMENTER_LABEL, open=True, elem_id="experimenter") as panel:
            with gr.Row(equal_height=True):
                name = gr.Textbox(label="Experiment name", info=EXPERIMENT_NAME_HINT, max_lines=1, scale=3)
                condition = gr.Radio(choices=CONDITION_CHOICES, value=None, label="Condition", scale=2)
                start = gr.Button("Start drawing", interactive=False, scale=1)
                finish = gr.Button("Finish drawing", interactive=False, scale=1)
            status = gr.Markdown(NO_EXPERIMENT_STATUS, elem_id="experimenter-status")
            capture_notice = gr.Markdown(visible=False, elem_id="capture-notice")

            # The camera the experimenter aims and turns, beside the source choice and what the system runs.
            with gr.Row(equal_height=False):
                webcam = gr.Image(label="Live webcam", sources=["webcam"], streaming=True, type="numpy",
                                  webcam_options=gr.WebcamOptions(mirror=False), height=300, elem_id="webcam",
                                  elem_classes=[f"rotate-{DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM]}"])
                upload = gr.Image(label="Sketch file", sources=["upload"], type="filepath", image_mode=None, visible=False,
                                  height=300)
                with gr.Column():
                    source = gr.Radio(choices=SOURCE_CHOICES, value=SketchSource.WEBCAM, label="Source")
                    rotation = gr.Dropdown(choices=list(Rotation), value=DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM],
                                           label="Clockwise rotation")
                    gr.Markdown(self.describe_system(), elem_id="system")

        return ExperimenterPanel(panel=panel, name=name, condition=condition, start=start, finish=finish, status=status,
                                 capture_notice=capture_notice, source=source, rotation=rotation, webcam=webcam, upload=upload)

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

        # Start needs a name and a condition, and Start and Finish run in the session queue with errors as one line.
        drawing_outputs = [participant.status, participant.suggest_prompt, participant.suggest_strokes, experimenter.name,
                           experimenter.condition, experimenter.finish, experimenter.status]
        start_inputs = [experimenter.name, experimenter.condition]
        for component in start_inputs:
            component.change(self.update_study_start_button, start_inputs, [experimenter.start], show_progress="hidden")
        start_outputs = [*drawing_outputs, experimenter.start, participant.sketch_view, participant.prompt,
                         participant.prompt_suggestion, experimenter.panel]
        experimenter.start.click(self.start_study_drawing, start_inputs, start_outputs, concurrency_id="session",
                                 concurrency_limit=1, show_progress="hidden")
        prompt_inputs = [participant.prompt, participant.prompt_suggestion]
        experimenter.finish.click(self.finish_study_drawing, [*sketch_inputs, *prompt_inputs],
                                  [*drawing_outputs, participant.sketch_view], concurrency_id="session", concurrency_limit=1,
                                  show_progress="hidden")

        # Write a suggested prompt and keep its ID, one model call at a time.
        participant.suggest_prompt.click(self.suggest_another, sketch_inputs, prompt_inputs, concurrency_id="suggest",
                                         concurrency_limit=1)

        # Say at once that a suggestion is running, then show it or the failure line, and re-enable the button.
        status = participant.status
        waiting = participant.suggest_strokes.click(self.show_suggesting, None, [status, participant.suggest_strokes],
                                                    queue=False, show_progress="hidden")
        suggesting = waiting.then(self.suggest_next_strokes, [experimenter.source, experimenter.upload, *prompt_inputs,
                                                              experimenter.rotation],
                                  [status, participant.sketch_view], concurrency_id="session", concurrency_limit=1,
                                  show_progress="minimal", show_progress_on=[status])
        suggesting.then(self.update_suggest_button, None, [participant.suggest_strokes], show_progress="hidden")
        suggesting.failure(self.show_drawing_status, None, [status], show_progress="hidden")

        # Follow the paper at the capture interval, and save the open drawing at the save interval in its own queue.
        frame_timer = gr.Timer(self.capture_interval_seconds)
        frame_timer.tick(self.refresh_study_preview, sketch_inputs, [participant.sketch_view], show_progress="hidden")
        capture_timer = gr.Timer(self.controller.save_interval_seconds)
        capture_timer.tick(self.save_study_capture, [*sketch_inputs, experimenter.capture_notice],
                           [experimenter.capture_notice, experimenter.panel], concurrency_id="drawing-capture",
                           concurrency_limit=1, show_progress="hidden")

        # Restore the latest drawing on load.
        app.load(self.restore_study_view, None, [*drawing_outputs, experimenter.start])

    def _build_generation_page(self, app: gr.Blocks) -> None:
        """
        Lay out and wire the generation page's three steps, the experiment controls, and the history.
        """
        self._build_header()

        # Steps 1 and 2 side by side at equal height, then step 3, then the experimenter's controls and the history.
        with gr.Row(equal_height=True):
            sketch = self._build_sketch_card()
            direction = self._build_direction_card()
        choose = self._build_choose_card()
        experiment = self._build_experiment_card()
        history = self._build_history()

        # Wire the events inside the Blocks context.
        self._wire_input_events(sketch, direction)
        self._wire_session_events(app, sketch, direction, choose, history, experiment)
        self._wire_preview_events(sketch, choose)

    def _build_study_page(self, app: gr.Blocks) -> None:
        """
        Lay out and wire the study page: the title, the participant area, and the experimenter panel below it.
        """
        # The title only, so nothing above the drawing names a model or differs between conditions.
        with gr.Column(elem_classes="study"):
            gr.Markdown("# SketchLoop", elem_id="header")
            participant = self._build_participant_area()
            experimenter = self._build_experimenter_panel()

        self._wire_study_events(app, participant, experimenter)

    def build(self) -> gr.Blocks:
        """
        Build the study page or the generation page and wire its events, without launching it.
        """
        with gr.Blocks(title="SketchLoop", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
            # Build only the chosen page, so the study page has no generation components that could fire an event.
            if self.study:
                self._build_study_page(app)
            else:
                self._build_generation_page(app)

        return app


def serve_app(app: gr.Blocks, controller: AppController) -> None:
    """
    Serve a built page on this machine only, with no share link, the theme, CSS, and script, and access to the runs folder only.

    Args:
        app (gr.Blocks): Page built by SketchPage.
        controller (AppController): Service whose store holds the runs folder.
    """
    head = f"<script>{SCRIPT_PATH.read_text(encoding='utf-8')}</script>"
    app.launch(server_name=SERVER_HOST, share=False, ssr_mode=False, inbrowser=True, theme=THEME, css_paths=[STYLESHEET_PATH],
               head=head, allowed_paths=[str(controller.store.root.resolve())], max_file_size="20mb", show_error=False)


def launch_app(controller: AppController, *, default_candidates: int, default_seed: int | None,
               capture_interval_seconds: float = 0.5, study: bool = False) -> None:
    """
    Build the page for the controller and serve it on this machine.

    Args:
        controller (AppController): Service holding the generator and the session.
        default_candidates (int): Candidates per round shown at start.
        default_seed (int | None): Seed shown at start, or None for random.
        capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.
        study (bool, optional): Serve the study page instead of the generation page. Defaults to False.
    """
    page = SketchPage(controller, default_candidates=default_candidates, default_seed=default_seed,
                      capture_interval_seconds=capture_interval_seconds, study=study)
    serve_app(page.build(), controller)
