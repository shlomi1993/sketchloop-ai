"""
The ui subcommand, which creates the backend once and serves the browser page on localhost.
"""

import argparse
import os
import threading
import time

from rich.console import Console

from sketchloop.app import AppController
from sketchloop.cli.common import make_configured_generator
from sketchloop.components import make_evaluator, make_preprocessor, make_prompt_suggester, make_store, make_stroke_suggester
from sketchloop.domain import SketchLoopError
from sketchloop.settings import Settings
from sketchloop.stroke_suggestion import StrokeSuggester
from sketchloop.suggestion import PromptSuggester


def warm_up_suggester(suggester: PromptSuggester, console: Console) -> None:
    """
    Load the suggestion model in the background, printing why suggestions are unavailable if it cannot load.
    """
    try:
        suggester.warm_up()
    except SketchLoopError as error:
        console.print(f"Prompt suggestions are unavailable: {error}", style="yellow", markup=False, soft_wrap=True)


def warm_up_stroke_suggester(suggester: StrokeSuggester, width: int, height: int, console: Console) -> None:
    """
    Load and warm up the next-stroke suggester at the canvas size in the background, printing how long it took or why it failed.
    """
    started = time.perf_counter()
    try:
        suggester.load(width, height)
    except SketchLoopError as error:
        console.print(f"Next-stroke suggestions are unavailable: {error}", style="yellow", markup=False, soft_wrap=True)
        return

    console.print(f"Next-stroke suggestions are ready after {time.perf_counter() - started:.1f} s.", markup=False)


def make_controller(settings: Settings, console: Console) -> AppController:
    """
    Create the backend, suggester, and controller once, and refuse a default candidate count the backend cannot produce.

    Args:
        settings (Settings): Effective settings.
        console (Console): Console used for output.

    Returns:
        AppController: Controller holding the loaded generator and a new session.
    """
    store = make_store(settings.storage)
    generator, setup_seconds = make_configured_generator(settings, console)

    # Build the configured prompt suggester apart from the image backend, and load any model in the background.
    suggester = make_prompt_suggester(settings)
    threading.Thread(target=warm_up_suggester, args=(suggester, console), daemon=True).start()

    # Warm up the configured next-stroke suggester at the canvas size, so the first press does not wait for the model.
    drawing = settings.drawing
    stroke_suggester = make_stroke_suggester(settings)
    warm_up_arguments = (stroke_suggester, drawing.canvas_width, drawing.canvas_height, console)
    threading.Thread(target=warm_up_stroke_suggester, args=warm_up_arguments, daemon=True).start()
    controller = AppController(generator, store, make_evaluator(settings),
                               preprocessor=make_preprocessor(settings.preprocessing), suggester=suggester,
                               generator_setup_seconds=setup_seconds, stroke_suggester=stroke_suggester,
                               canvas_size=(drawing.canvas_width, drawing.canvas_height),
                               save_interval_seconds=drawing.save_interval_seconds)
    limit, candidates = controller.max_candidates, settings.ui.candidates
    if candidates > limit:
        raise SketchLoopError(f"{candidates} candidates exceed the limit of {limit}. Lower --candidates or ui.candidates.")

    return controller


def announce_study_mode(controller: AppController, console: Console) -> None:
    """
    Say that the page is in study mode, and warn about each suggester that stands in for a real one.
    """
    console.print("Serving the study page: participants see the sketch, the prompt, and AI help only in the AI condition.")
    for warning in controller.list_stand_in_warnings():
        console.print(warning, style="yellow", markup=False, soft_wrap=True)


def launch_ui(args: argparse.Namespace, console: Console) -> None:
    """
    Serve the browser interface on this machine with one loaded backend, until the person stops it with Ctrl+C.

    Args:
        args (argparse.Namespace): Parsed ui options.
        console (Console): Console used for output.
    """
    # Turn off Gradio and Hugging Face telemetry before Gradio is imported.
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # Import Gradio only for this command, so the other commands start without loading it.
    from sketchloop.ui import launch_app

    settings = args.settings
    controller = make_controller(settings, console)

    # Pass the study flag only when it is on, until the page takes it in the study page slice.
    study_options = {"study": True} if settings.study.enabled else {}
    if settings.study.enabled:
        announce_study_mode(controller, console)

    launch_app(controller, default_candidates=settings.ui.candidates, default_seed=settings.ui.seed,
               capture_interval_seconds=settings.ui.capture_interval_seconds, **study_options)
