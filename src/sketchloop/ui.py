"""
The Gradio browser page, a thin layer over AppController for webcam or upload rounds.
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

from sketchloop.app import AppController, HistoryRow, RoundView
from sketchloop.domain import ControlValue, Rotation, SketchLoopError, SketchSource
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
STALE_FRAME_MESSAGE: Final = "The webcam stopped sending frames. Restart it, then generate."

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


class SketchPage:
    """
    The browser page, with handlers that share the controller and the newest webcam frame.
    """

    def __init__(self, controller: AppController, *, default_candidates: int, default_seed: int | None,
                 capture_interval_seconds: float = 0.5) -> None:
        """
        Store the controller and the page defaults.

        Args:
            controller (AppController): Service holding the generator and the session.
            default_candidates (int): Candidates per round shown at start.
            default_seed (int | None): Seed shown at start, or None for random.
            capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.
        """
        self.controller = controller
        self.default_candidates = default_candidates
        self.default_seed = default_seed
        self.capture_interval_seconds = capture_interval_seconds

        # The newest webcam frame in BGR order and when it arrived, shared by all tabs and read together under the lock.
        self.frame_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_time = 0.0

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

    def toggle_source(self, choice: str, use_raw: bool) -> tuple[gr.Image, gr.Image, gr.Dropdown]:
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
        # Keep the newest frame, converted from RGB to BGR, and when it arrived.
        if frame is not None:
            bgr_frame = np.ascontiguousarray(frame[:, :, ::-1])
            with self.frame_lock:
                self.latest_frame = bgr_frame
                self.latest_frame_time = time.monotonic()

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
                            use_raw: bool) -> np.ndarray | None:
        """
        Show the current webcam frame or upload as the generator would receive it, preprocessed unless raw.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            use_raw (bool): Whether to skip preprocessing and rotation.

        Returns:
            np.ndarray | None: The preview in RGB order, or None when there is no usable sketch yet.
        """
        # Decode the preview bytes for display, in RGB order.
        payload = self.controller.preview_sketch(self.pick_current_sketch(choice, upload_path), int(rotation_degrees or 0),
                                                 use_raw)
        if payload is None:
            return None

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def refresh_webcam_preview(self, choice: str, upload_path: str | None, rotation_degrees: int,
                               use_raw: bool) -> np.ndarray | dict | None:
        """
        Show the newest webcam frame as the generator would receive it, leaving an upload's preview as it is.
        """
        if SketchSource(choice) != SketchSource.WEBCAM:
            return gr.skip()

        return self.show_sketch_preview(choice, upload_path, rotation_degrees, use_raw)

    def suggest_another(self, choice: str, upload_path: str | None, rotation_degrees: int, use_raw: bool) -> str:
        """
        Write a suggested style for the current sketch, the next one for the same drawing, or show why there is none.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            use_raw (bool): Whether to skip preprocessing and rotation.

        Returns:
            str: The suggested prompt.
        """
        try:
            return self.controller.suggest_prompt(self.pick_current_sketch(choice, upload_path), int(rotation_degrees or 0),
                                                  use_raw)
        except (SketchLoopError, OSError) as error:
            raise gr.Error(str(error)) from error

    def run_generation(self, choice: str, upload_path: str | None, prompt_text: str, candidate_count: float,
                       seed: str | float | None, rotation_degrees: int, use_raw: bool, *control_values: float,
                       progress: gr.Progress = gr.Progress(track_tqdm=True)) -> RoundOutputs:
        """
        Generate a round from the webcam frame or the upload, showing any error as one line.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            prompt_text (str): Style prompt.
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
        sketch: Path | np.ndarray | None = None if upload_path is None else Path(upload_path)
        frame_age_seconds = None
        if source == SketchSource.WEBCAM:
            fresh_frame = self.take_fresh_frame()
            if fresh_frame is None and self.latest_frame is not None:
                raise gr.Error(STALE_FRAME_MESSAGE)

            sketch, frame_age_seconds = fresh_frame or (None, None)

        # Generate with the chosen controls, showing any error.
        controls = read_control_values(self.controller.slider_controls, control_values)
        try:
            view = self.controller.run_round(sketch, source, prompt_text, n_candidates=int(candidate_count),
                                             seed=read_seed(seed), rotation=int(rotation_degrees or 0),
                                             use_raw=use_raw, frame_age_seconds=frame_age_seconds, controls=controls)
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

    def start_session(self) -> tuple[object, ...]:
        """
        Start a new session and clear the round display and history.
        """
        self.controller.start_new_session()
        return *build_round_outputs(None, NEW_SESSION_STATUS), build_history_table(self.controller.list_history())

    def restore_view(self) -> tuple[object, ...]:
        """
        Show the latest round and the history, as on page load.
        """
        return *build_round_outputs(self.controller.latest_round()), build_history_table(self.controller.list_history())

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
                    f"{self.controller.describe_backend()} {SHARED_SESSION_NOTE}", elem_id="header")

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

        return DirectionCard(prompt=prompt, suggest=suggest, generate=generate, candidates=candidates,
                             seed=seed, controls=controls)

    def _build_choose_card(self) -> ChooseCard:
        """
        Lay out step 3, the status, sketch, candidates, and the choice.
        """
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 3 · Choose", elem_classes="step-title")
            status = gr.Markdown(NO_ROUNDS_STATUS, elem_id="status")

            # The received sketch beside the candidates.
            with gr.Row(equal_height=False):
                sketch_view = gr.Image(label="Sketch the model will receive", interactive=False, type="filepath", scale=1,
                                       height=320)
                gallery = gr.Gallery(label="Candidates", interactive=False, format="png", columns=4, height=320,
                                     object_fit="contain", scale=3)
            picks = gr.CheckboxGroup(label="Candidates you choose")

            # Record the choice or start over.
            with gr.Row():
                record = gr.Button("Record my choice", variant="primary")
                new_session = gr.Button("Start a new session", variant="secondary")

        return ChooseCard(status=status, sketch_view=sketch_view, gallery=gallery, picks=picks, record=record,
                          new_session=new_session, round_state=gr.State(None))

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
                             history: gr.Dataframe) -> None:
        """
        Wire Generate, the choice, the new session, and the page load.

        Args:
            app (gr.Blocks): The page being built.
            sketch (SketchCard): Step 1 components.
            direction (DirectionCard): Step 2 components.
            choose (ChooseCard): Step 3 components.
            history (gr.Dataframe): Session history table.
        """
        # The round display, and the inputs Generate reads.
        round_outputs = [choose.gallery, choose.sketch_view, choose.picks, choose.round_state, choose.status]
        settings = [direction.candidates, direction.seed, sketch.rotation, sketch.raw, *direction.controls]
        generate_inputs = [sketch.source, sketch.upload, direction.prompt, *settings]

        # Run session changes one at a time.
        direction.generate.click(self.run_generation, generate_inputs, round_outputs, concurrency_id="session",
                                 concurrency_limit=1).then(self.show_history, None, [history])
        choose.record.click(self.save_selection, [choose.round_state, choose.picks], [choose.status],
                            concurrency_id="session", concurrency_limit=1).then(self.show_history, None, [history])
        choose.new_session.click(self.start_session, None, [*round_outputs, history], concurrency_id="session",
                                 concurrency_limit=1)

        # Write a suggested style on request, one model call at a time.
        suggest_inputs = [sketch.source, sketch.upload, sketch.rotation, sketch.raw]
        direction.suggest.click(self.suggest_another, suggest_inputs, [direction.prompt], concurrency_id="suggest",
                                concurrency_limit=1)

        # Restore the latest round on load, even during a slow round.
        app.load(self.restore_view, None, [*round_outputs, history])

    def build(self) -> gr.Blocks:
        """
        Build the page and wire its events, without launching it.
        """
        with gr.Blocks(title="SketchLoop", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
            self._build_header()

            # Steps 1 and 2 side by side at equal height, then step 3 and the history.
            with gr.Row(equal_height=True):
                sketch = self._build_sketch_card()
                direction = self._build_direction_card()
            choose = self._build_choose_card()
            history = self._build_history()

            # Wire the events inside the Blocks context.
            self._wire_input_events(sketch, direction)
            self._wire_session_events(app, sketch, direction, choose, history)
            self._wire_preview_events(sketch, choose)

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
               capture_interval_seconds: float = 0.5) -> None:
    """
    Build the page for the controller and serve it on this machine.

    Args:
        controller (AppController): Service holding the generator and the session.
        default_candidates (int): Candidates per round shown at start.
        default_seed (int | None): Seed shown at start, or None for random.
        capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.
    """
    page = SketchPage(controller, default_candidates=default_candidates, default_seed=default_seed,
                      capture_interval_seconds=capture_interval_seconds)
    serve_app(page.build(), controller)
