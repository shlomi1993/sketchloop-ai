"""
Serve the browser page with live reload, so saved edits to the page show up without reloading the model.
"""

import importlib
import os
import sys

from pathlib import Path
from rich.console import Console
from types import ModuleType

from sketchloop.cli.arguments import parse_args
from sketchloop.cli.serve import make_controller


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "sketchloop"
PAGE_MODULE = "sketchloop.ui"


def configure_reload() -> None:
    """
    Tell Gradio, before it is imported, to watch only the package and to send no telemetry.
    """
    os.environ.update({
        "GRADIO_WATCH_DIRS": str(PACKAGE_DIR),
        "GRADIO_WATCH_MODULE_NAME": "__main__",
        "GRADIO_WATCH_DEMO_NAME": "demo",
        "GRADIO_WATCH_DEMO_PATH": str(Path(__file__).resolve()),
        "GRADIO_ANALYTICS_ENABLED": "False",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })


def find_kept_modules() -> dict[str, ModuleType]:
    """
    Collect the loaded sketchloop modules except the page, which a reload must not replace.
    """
    return {name: module for name, module in sys.modules.items() if name.startswith("sketchloop") and name != PAGE_MODULE}


configure_reload()

# Gradio reruns this file on every saved change, so load the model only the first time.
if "controller" not in globals():
    settings = parse_args(["ui"]).settings
    controller = make_controller(settings, Console())
    kept_modules = find_kept_modules()

# Gradio drops every changed package module, so put back all but the page, keeping the controller's classes the page sees.
sys.modules.update(kept_modules)
page_module = importlib.import_module(PAGE_MODULE)
demo = page_module.SketchPage(controller, default_candidates=settings.ui.candidates, default_seed=settings.ui.seed,
                              capture_interval_seconds=settings.ui.capture_interval_seconds).build()

# Serve the page the first time, while later runs only hand Gradio its theme, CSS, and script.
page_module.serve_app(demo, controller)
