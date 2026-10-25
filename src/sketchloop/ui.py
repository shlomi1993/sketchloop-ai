import gradio as gr
import numpy as np

from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypeAlias

from sketchloop.app import AppController, HistoryRow, RoundView
from sketchloop.domain import SketchLoopError
from sketchloop.experiments import SketchSource


HISTORY_HEADERS: Final = ["Round", "Status", "Prompt", "Outcome"]
SOURCE_CHOICES: Final = [("Live webcam", "webcam"), ("Upload a file", "upload")]
ROTATION_CHOICES: Final = [0, 90, 180, 270]
PROMPT_PLACEHOLDER: Final = "Describe the style, for example: a sculptural oak lounge chair, soft studio light"

# A webcam facing the person drawing sees the paper upside down, so webcam sketches turn 180 degrees by default.
DEFAULT_ROTATION_BY_SOURCE: Final = {"webcam": 180, "upload": 0}

# The page's stylesheet and script, kept as real files next to this module and shipped as package data.
STATIC_DIR: Final = Path(__file__).parent / "static"
STYLESHEET_PATH: Final = STATIC_DIR / "ui.css"
SCRIPT_PATH: Final = STATIC_DIR / "ui.js"

# Events call the helpers that ui.js defines on window.sketchloop.
OPEN_WEBCAM_JS: Final = "() => window.sketchloop.openWebcam()"
ROTATE_PREVIEW_JS: Final = "(degrees) => window.sketchloop.rotatePreview(degrees)"

# Status lines guide the person through each step.
NO_ROUNDS_STATUS: Final = "Draw on paper, point the webcam at it, describe the style, and press **Generate**."
NEW_SESSION_STATUS: Final = "New session started. Its folder is created with the first round."
FAKE_BACKEND_PREFIX: Final = "**Fake backend:** these images are deterministic noise, not output from a real model."
PROGRESS_MESSAGE: Final = "Generating. The first round also loads the model."
SHARED_SESSION_NOTE: Final = "All browser tabs share one session, and reloading keeps it."

# A calm, neutral look with one accent color, so the sketches and candidates stand out.
THEME: Final = gr.themes.Default(primary_hue="indigo", neutral_hue="slate", radius_size="lg")

# Outputs that show a round: gallery, generation sketch, candidate picks, displayed iteration ID, and status.
RoundOutputs: TypeAlias = tuple[list[tuple[str, str]], str | None, gr.CheckboxGroup, str | None, str]


def describe_round_status(view: RoundView) -> str:
    """
    Say what the person can do next with a round, warning first when its images are fake.
    """
    # Say what to do next, warning first when the images are not from a real model.
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
    # Clear the round display when there is no round yet.
    if view is None:
        return [], None, gr.CheckboxGroup(choices=[], value=[]), None, empty_status

    gallery = [(str(candidate.image_path), candidate.label) for candidate in view.candidates]
    picks = gr.CheckboxGroup(choices=[(candidate.label, candidate.candidate_id) for candidate in view.candidates], value=[])
    return gallery, str(view.sketch_path), picks, view.iteration_id, describe_round_status(view)


def can_generate(choice: str, has_webcam_frame: bool, upload_path: str | None, prompt_text: str | None) -> bool:
    """
    Tell whether the chosen source has a sketch and the prompt is not blank, so generation can start.
    """
    has_sketch = has_webcam_frame if choice == "webcam" else upload_path is not None
    return has_sketch and bool((prompt_text or "").strip())


def pick_rotation(choice: str, use_raw: bool) -> int:
    """
    Pick the rotation the page offers, 0 for the raw sketch, which skips preprocessing and so cannot be turned.
    """
    return 0 if use_raw else DEFAULT_ROTATION_BY_SOURCE[choice]


def build_history_table(rows: Sequence[HistoryRow]) -> list[list[object]]:
    """
    Turn history rows into table rows, showing a missing prompt as not recorded.
    """
    return [[row.number, row.status, "not recorded" if row.prompt is None else row.prompt, row.outcome] for row in rows]


def build_app(controller: AppController, *, default_candidates: int, default_seed: int | None,
              capture_interval_seconds: float = 0.5) -> gr.Blocks:
    """
    Build the browser page around the controller, without launching it.

    Args:
        controller (AppController): Service holding the generator and the session.
        default_candidates (int): Candidates per round shown at start.
        default_seed (int | None): Seed shown at start, or None for random.
        capture_interval_seconds (float, optional): Seconds between kept webcam frames. Defaults to 0.5.

    Returns:
        gr.Blocks: The page.
    """
    # The latest streamed webcam frame in OpenCV channel order, shared by every tab like the session itself.
    latest_frames: dict[str, np.ndarray] = {}

    with gr.Blocks(title="SketchLoop", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        # Header with the backend in use and how tabs share the session.
        gr.Markdown(f"# SketchLoop\nSketch on paper, explore AI alternatives, and choose your direction.\n\n"
                    f"{controller.describe_backend()} {SHARED_SESSION_NOTE}", elem_id="header")

        # Step 1 and step 2 side by side: the sketch on the left, the direction and Generate on the right.
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="card"):
                gr.Markdown("### 1 · Sketch", elem_classes="step-title")
                source = gr.Radio(choices=SOURCE_CHOICES, value="webcam", show_label=False)
                webcam = gr.Image(label="Live webcam", sources=["webcam"], streaming=True, type="numpy",
                                  webcam_options=gr.WebcamOptions(mirror=False), height=360, elem_id="webcam")
                upload = gr.Image(label="Sketch file", sources=["upload"], type="filepath", image_mode=None, visible=False,
                                  height=360)

            # The prompt, the main action, and settings tucked away for later.
            with gr.Column(scale=2, elem_classes="card"):
                gr.Markdown("### 2 · Direction", elem_classes="step-title")
                prompt = gr.Textbox(label="Style prompt", placeholder=PROMPT_PLACEHOLDER, lines=4)
                generate = gr.Button("Generate", variant="primary", interactive=False, elem_id="generate")
                with gr.Accordion("Settings", open=False):
                    candidates = gr.Slider(minimum=1, maximum=controller.max_candidates, step=1, value=default_candidates,
                                           label="Candidates per round")
                    seed = gr.Number(label="Seed (blank for random)", value=default_seed, precision=0, minimum=0)
                    rotation = gr.Dropdown(choices=ROTATION_CHOICES, value=DEFAULT_ROTATION_BY_SOURCE["webcam"],
                                           label="Rotation (degrees clockwise)")
                    raw = gr.Checkbox(label="Use the raw sketch (skip preprocessing and rotation)")

        # Step 3: the status, the candidates beside the sketch the model received, and the explicit choice.
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 3 · Choose", elem_classes="step-title")
            status = gr.Markdown(NO_ROUNDS_STATUS, elem_id="status")
            with gr.Row(equal_height=False):
                sketch_view = gr.Image(label="Sketch the model received", interactive=False, type="filepath", scale=1, height=320)
                gallery = gr.Gallery(label="Candidates", interactive=False, format="png", columns=4, height=320,
                                     object_fit="contain", scale=3)
            picks = gr.CheckboxGroup(label="Candidates you choose")
            with gr.Row():
                record = gr.Button("Record my choice", variant="primary")
                new_session = gr.Button("Start a new session", variant="secondary")

        # The session history, and the iteration ID of the round this page shows.
        with gr.Accordion("Session history", open=False):
            history = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, show_label=False)
        round_state = gr.State(None)

        def toggle_source(choice: str, use_raw: bool) -> tuple[gr.Image, gr.Image, gr.Dropdown]:
            rotation_choice = gr.Dropdown(value=pick_rotation(choice, use_raw), interactive=not use_raw)
            return gr.Image(visible=choice == "webcam"), gr.Image(visible=choice == "upload"), rotation_choice

        def toggle_raw(use_raw: bool, choice: str) -> gr.Dropdown:
            return gr.Dropdown(value=pick_rotation(choice, use_raw), interactive=not use_raw)

        def keep_frame(frame: np.ndarray | None, choice: str, upload_path: str | None, prompt_text: str | None) -> gr.Button:
            # Keep only the newest frame, converted from the browser's RGB to OpenCV's BGR order.
            if frame is not None:
                latest_frames["webcam"] = np.ascontiguousarray(frame[:, :, ::-1])

            return gr.Button(interactive=can_generate(choice, "webcam" in latest_frames, upload_path, prompt_text))

        def update_generate_button(choice: str, upload_path: str | None, prompt_text: str | None) -> gr.Button:
            return gr.Button(interactive=can_generate(choice, "webcam" in latest_frames, upload_path, prompt_text))

        def run_generation(choice: SketchSource, upload_path: str | None, prompt_text: str, candidate_count: float,
                           seed_value: float | None, rotation_degrees: int, use_raw: bool,
                           progress: gr.Progress = gr.Progress(track_tqdm=True)) -> RoundOutputs:
            # Generate from the newest webcam frame or the uploaded file, showing the message of a refused or failed round.
            progress(None, desc=PROGRESS_MESSAGE)
            sketch = latest_frames.get("webcam") if choice == "webcam" else None if upload_path is None else Path(upload_path)
            seed_number = None if seed_value is None else int(seed_value)
            try:
                view = controller.run_round(sketch, choice, prompt_text, n_candidates=int(candidate_count), seed=seed_number,
                                            rotation=int(rotation_degrees or 0), use_raw=use_raw)
            except (SketchLoopError, OSError) as error:
                raise gr.Error(str(error)) from error

            return build_round_outputs(view)

        def save_selection(iteration_id: str | None, candidate_ids: list[str]) -> str:
            # A page without a displayed round never matches, so the person reloads to see the latest round.
            try:
                view = controller.record_selection(iteration_id, candidate_ids)
            except (SketchLoopError, OSError) as error:
                raise gr.Error(str(error)) from error

            return describe_round_status(view)

        def start_session() -> tuple[object, ...]:
            controller.start_new_session()
            return *build_round_outputs(None, NEW_SESSION_STATUS), build_history_table(controller.list_history())

        def restore_view() -> tuple[object, ...]:
            return *build_round_outputs(controller.latest_round()), build_history_table(controller.list_history())

        def show_history() -> list[list[object]]:
            return build_history_table(controller.list_history())

        # Switch the visible input and its default rotation, lock rotation at 0 for the raw sketch, and keep the newest
        # webcam frame at the chosen interval.
        source.change(toggle_source, [source, raw], [webcam, upload, rotation])
        raw.change(toggle_raw, [raw, source], [rotation])
        rotation.change(None, rotation, None, js=ROTATE_PREVIEW_JS)
        webcam.stream(keep_frame, [webcam, source, upload, prompt], [generate], stream_every=capture_interval_seconds,
                      show_progress="hidden", time_limit=None)

        # Enable Generate only once the chosen source has a sketch and the prompt is not blank.
        readiness_inputs = [source, upload, prompt]
        for component in readiness_inputs:
            component.change(update_generate_button, readiness_inputs, [generate])

        # Serialize every listener that changes the session, so a selection cannot interleave with a running round.
        round_outputs = [gallery, sketch_view, picks, round_state, status]
        generate_inputs = [source, upload, prompt, candidates, seed, rotation, raw]
        generate.click(run_generation, generate_inputs, round_outputs, concurrency_id="session",
                       concurrency_limit=1).then(show_history, None, [history])
        record.click(save_selection, [round_state, picks], [status], concurrency_id="session",
                     concurrency_limit=1).then(show_history, None, [history])
        new_session.click(start_session, None, [*round_outputs, history], concurrency_id="session", concurrency_limit=1)

        # Restore the latest round on page load, outside the session group so a reload during a slow round still renders.
        app.load(restore_view, None, [*round_outputs, history])

        # Open the webcam and start the frame stream on page load, since the webcam is the default source.
        app.load(None, None, None, js=OPEN_WEBCAM_JS)

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
    app = build_app(controller, default_candidates=default_candidates, default_seed=default_seed,
                    capture_interval_seconds=capture_interval_seconds)
    head = f"<script>{SCRIPT_PATH.read_text(encoding='utf-8')}</script>"
    app.launch(server_name="127.0.0.1", share=False, ssr_mode=False, inbrowser=True, theme=THEME, css_paths=[STYLESHEET_PATH],
               head=head, allowed_paths=[str(controller.store.root.resolve())], max_file_size="20mb", show_error=False)
