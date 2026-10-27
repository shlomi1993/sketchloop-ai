import argparse
import os

from rich.console import Console

from sketchloop.app import AppController
from sketchloop.cli.common import make_configured_generator
from sketchloop.components import make_evaluator, make_preprocessor, make_store
from sketchloop.domain import SketchLoopError


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

    # Create the backend once, and refuse a default candidate count the backend cannot produce.
    settings = args.settings
    store = make_store(settings.storage)
    generator, setup_seconds = make_configured_generator(settings.generator, console)
    controller = AppController(generator, store, make_evaluator(settings.evaluator),
                               preprocessor=make_preprocessor(settings.preprocessing), generator_setup_seconds=setup_seconds)
    limit, candidates = controller.max_candidates, settings.ui.candidates
    if candidates > limit:
        raise SketchLoopError(f"{candidates} candidates exceed the limit of {limit}. Lower --candidates or ui.candidates.")

    launch_app(controller, default_candidates=candidates, default_seed=settings.ui.seed,
               capture_interval_seconds=settings.ui.capture_interval_seconds)
