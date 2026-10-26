import gradio as gr
import numpy as np

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from sketchloop.app import AppController, HistoryRow, RoundView
from sketchloop.domain import Rotation, SketchLoopError, SketchSource


HISTORY_HEADERS: Final = ["Round", "Status", "Prompt", "Outcome"]
SOURCE_CHOICES: Final = [("Live webcam", SketchSource.WEBCAM), ("Upload a file", SketchSource.UPLOAD)]
PROMPT_PLACEHOLDER: Final = "Describe the style, for example: a sculptural oak lounge chair, soft studio light"

# A webcam facing the person sees the paper upside down.
DEFAULT_ROTATION_BY_SOURCE: Final = {SketchSource.WEBCAM: Rotation.DEG_180, SketchSource.UPLOAD: Rotation.DEG_000}

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
        status = f"Round {view.number} is ready after {view.generation_seconds:.1f} s. Tick the candidates you like, or none, and record your choice."  # noqa: E501
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


def build_history_table(rows: Sequence[HistoryRow]) -> list[list[object]]:
    """
    Turn history rows into table rows, showing a missing prompt as not recorded.
    """
    return [[row.number, row.status, "not recorded" if row.prompt is None else row.prompt, row.outcome] for row in rows]


@dataclass(frozen=True, slots=True, kw_only=True)
class SketchCard:
    """
    Components of step 1, where the person picks and shows the sketch.
    """
    source: gr.Radio  # Webcam or upload choice
    webcam: gr.Image  # Live webcam preview that streams frames
    upload: gr.Image  # Uploaded sketch file


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionCard:
    """
    Components of step 2, the prompt, the Generate button, and the settings.
    """
    prompt: gr.Textbox  # Style prompt
    generate: gr.Button  # Starts a round
    candidates: gr.Slider  # Candidates per round
    seed: gr.Number  # Seed, or blank for random
    rotation: gr.Dropdown  # Degrees to turn the sketch clockwise
    raw: gr.Checkbox  # Skips preprocessing and rotation


@dataclass(frozen=True, slots=True, kw_only=True)
class ChooseCard:
    """
    Components of step 3, the round display and the choice, plus the iteration ID the page shows.
    """
    status: gr.Markdown  # What to do next
    sketch_view: gr.Image  # Sketch the model received
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

        # The newest webcam frame in BGR order, shared by all tabs.
        self.latest_frames: dict[SketchSource, np.ndarray] = {}

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
        has_webcam_frame = SketchSource.WEBCAM in self.latest_frames
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
        # Keep the newest frame, converted from RGB to BGR.
        if frame is not None:
            self.latest_frames[SketchSource.WEBCAM] = np.ascontiguousarray(frame[:, :, ::-1])

        return self.update_generate_button(choice, upload_path, prompt_text)

    def run_generation(self, choice: str, upload_path: str | None, prompt_text: str, candidate_count: float,
                       seed: float | None, rotation_degrees: int, use_raw: bool,
                       progress: gr.Progress = gr.Progress(track_tqdm=True)) -> RoundOutputs:
        """
        Generate a round from the webcam frame or the upload, showing any error as one line.

        Args:
            choice (str): Chosen sketch source.
            upload_path (str | None): Uploaded sketch file, if any.
            prompt_text (str): Style prompt.
            candidate_count (float): Candidates to generate.
            seed (float | None): Seed, or None for random.
            rotation_degrees (int): Degrees to turn the sketch clockwise.
            use_raw (bool): Whether to skip preprocessing and rotation.
            progress (gr.Progress, optional): Progress bar Gradio fills in. Defaults to one that tracks tqdm.

        Returns:
            RoundOutputs: Values for the round display components.
        """
        # Pick the sketch for the chosen source.
        progress(None, desc=PROGRESS_MESSAGE)
        source = SketchSource(choice)
        upload = None if upload_path is None else Path(upload_path)
        sketch = self.latest_frames.get(source) if source == SketchSource.WEBCAM else upload

        # Generate, showing any error.
        try:
            view = self.controller.run_round(sketch, source, prompt_text, n_candidates=int(candidate_count),
                                             seed=None if seed is None else int(seed), rotation=int(rotation_degrees or 0),
                                             use_raw=use_raw)
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

    def show_history(self) -> list[list[object]]:
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
        Lay out step 1, the source choice, the live webcam, and the upload.
        """
        with gr.Column(scale=3, elem_classes="card"):
            gr.Markdown("### 1 · Sketch", elem_classes="step-title")
            source = gr.Radio(choices=SOURCE_CHOICES, value=SketchSource.WEBCAM, show_label=False)
            webcam = gr.Image(label="Live webcam", sources=["webcam"], streaming=True, type="numpy",
                              webcam_options=gr.WebcamOptions(mirror=False), height=360, elem_id="webcam",
                              elem_classes=[f"rotate-{DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM]}"])
            upload = gr.Image(label="Sketch file", sources=["upload"], type="filepath", image_mode=None, visible=False,
                              height=360)

        return SketchCard(source=source, webcam=webcam, upload=upload)

    def _build_direction_card(self) -> DirectionCard:
        """
        Lay out step 2, the prompt, Generate, and collapsed settings.
        """
        with gr.Column(scale=2, elem_classes="card"):
            gr.Markdown("### 2 · Direction", elem_classes="step-title")
            prompt = gr.Textbox(label="Style prompt", placeholder=PROMPT_PLACEHOLDER, lines=4)
            generate = gr.Button("Generate", variant="primary", interactive=False, elem_id="generate")

            # Settings stay collapsed until needed.
            with gr.Accordion("Settings", open=False):
                candidates = gr.Slider(minimum=1, maximum=self.controller.max_candidates, step=1, value=self.default_candidates,
                                       label="Candidates per round")
                seed = gr.Number(label="Seed (blank for random)", value=self.default_seed, precision=0, minimum=0)
                rotation = gr.Dropdown(choices=list(Rotation), value=DEFAULT_ROTATION_BY_SOURCE[SketchSource.WEBCAM],
                                       label="Rotation (degrees clockwise)")
                raw = gr.Checkbox(label="Use the raw sketch (skip preprocessing and rotation)")

        return DirectionCard(prompt=prompt, generate=generate, candidates=candidates, seed=seed, rotation=rotation, raw=raw)

    def _build_choose_card(self) -> ChooseCard:
        """
        Lay out step 3, the status, sketch, candidates, and the choice.
        """
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 3 · Choose", elem_classes="step-title")
            status = gr.Markdown(NO_ROUNDS_STATUS, elem_id="status")

            # The received sketch beside the candidates.
            with gr.Row(equal_height=False):
                sketch_view = gr.Image(label="Sketch the model received", interactive=False, type="filepath", scale=1, height=320)
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
        source_outputs = [sketch.webcam, sketch.upload, direction.rotation]
        sketch.source.change(self.toggle_source, [sketch.source, direction.raw], source_outputs)
        direction.raw.change(self.toggle_raw, [direction.raw, sketch.source], [direction.rotation])
        direction.rotation.change(self.turn_preview, direction.rotation, [sketch.webcam], show_progress="hidden")
        stream_inputs = [sketch.webcam, sketch.source, sketch.upload, direction.prompt]
        sketch.webcam.stream(self.keep_frame, stream_inputs, [direction.generate], stream_every=self.capture_interval_seconds,
                             show_progress="hidden", time_limit=None)

        # Enable Generate once there is a sketch and a prompt.
        readiness_inputs = [sketch.source, sketch.upload, direction.prompt]
        for component in readiness_inputs:
            component.change(self.update_generate_button, readiness_inputs, [direction.generate])

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
        settings = [direction.candidates, direction.seed, direction.rotation, direction.raw]
        generate_inputs = [sketch.source, sketch.upload, direction.prompt, *settings]

        # Run session changes one at a time.
        direction.generate.click(self.run_generation, generate_inputs, round_outputs, concurrency_id="session",
                                 concurrency_limit=1).then(self.show_history, None, [history])
        choose.record.click(self.save_selection, [choose.round_state, choose.picks], [choose.status],
                            concurrency_id="session", concurrency_limit=1).then(self.show_history, None, [history])
        choose.new_session.click(self.start_session, None, [*round_outputs, history], concurrency_id="session",
                                 concurrency_limit=1)

        # Restore the latest round on load, even during a slow round.
        app.load(self.restore_view, None, [*round_outputs, history])

    def build(self) -> gr.Blocks:
        """
        Build the page and wire its events, without launching it.
        """
        with gr.Blocks(title="SketchLoop", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
            self._build_header()

            # Steps 1 and 2 side by side, then step 3 and the history.
            with gr.Row(equal_height=False):
                sketch = self._build_sketch_card()
                direction = self._build_direction_card()
            choose = self._build_choose_card()
            history = self._build_history()

            # Wire the events inside the Blocks context.
            self._wire_input_events(sketch, direction)
            self._wire_session_events(app, sketch, direction, choose, history)

        return app


def launch_app(controller: AppController, *, default_candidates: int, default_seed: int | None,
               capture_interval_seconds: float = 0.5) -> None:
    """
    Serve the page on this machine only, with no share link, the theme, CSS, and script, and access limited to the runs folder.

    Args:
        controller (AppController): Service holding the generator and the session.
        default_candidates (int): Candidates per round shown at start.
        default_seed (int | None): Seed shown at start, or None for random.
        capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.
    """
    page = SketchPage(controller, default_candidates=default_candidates, default_seed=default_seed,
                      capture_interval_seconds=capture_interval_seconds)
    app = page.build()
    head = f"<script>{SCRIPT_PATH.read_text(encoding='utf-8')}</script>"
    app.launch(server_name=SERVER_HOST, share=False, ssr_mode=False, inbrowser=True, theme=THEME, css_paths=[STYLESHEET_PATH],
               head=head, allowed_paths=[str(controller.store.root.resolve())], max_file_size="20mb", show_error=False)
