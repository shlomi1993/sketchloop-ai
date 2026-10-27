"""
The ui subcommand, which creates the backend once and serves the browser page on localhost.
"""

import argparse
import os
import threading

from rich.console import Console

from sketchloop.app import AppController
from sketchloop.cli.common import make_configured_generator
from sketchloop.components import make_evaluator, make_preprocessor, make_store
from sketchloop.domain import GenerationBackend, SketchLoopError
from sketchloop.settings import Settings
from sketchloop.suggestion import FlorenceSuggester, PromptSuggester, TemplateSuggester


def warm_up_suggester(suggester: PromptSuggester, console: Console) -> None:
    """
    Load the suggestion model in the background, printing why suggestions are unavailable if it cannot load.
    """
    try:
        suggester.warm_up()
    except SketchLoopError as error:
        console.print(f"Prompt suggestions are unavailable: {error}", style="yellow", markup=False, soft_wrap=True)


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
    generator, setup_seconds = make_configured_generator(settings.generator, console)

    # Name the sketched object with the local model next to a real backend, and use the fixed styles with the fake.
    is_real = settings.generator.backend == GenerationBackend.DIFFUSERS
    suggester = FlorenceSuggester(settings.generator.diffusers.models_dir) if is_real else TemplateSuggester()
    threading.Thread(target=warm_up_suggester, args=(suggester, console), daemon=True).start()
    controller = AppController(generator, store, make_evaluator(settings.evaluator),
                               preprocessor=make_preprocessor(settings.preprocessing), suggester=suggester,
                               generator_setup_seconds=setup_seconds)
    limit, candidates = controller.max_candidates, settings.ui.candidates
    if candidates > limit:
        raise SketchLoopError(f"{candidates} candidates exceed the limit of {limit}. Lower --candidates or ui.candidates.")

    return controller


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
    launch_app(make_controller(settings, console), default_candidates=settings.ui.candidates, default_seed=settings.ui.seed,
               capture_interval_seconds=settings.ui.capture_interval_seconds)
