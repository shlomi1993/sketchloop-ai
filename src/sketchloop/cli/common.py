import time

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sketchloop.backends import GenerationBackend, make_generator
from sketchloop.backends.diffusers import GenerationMode
from sketchloop.cli.arguments import Command
from sketchloop.components import make_store
from sketchloop.domain import SketchLoopError
from sketchloop.experiments import ExperimentStore, RoundStage
from sketchloop.generation import Generator
from sketchloop.session import RoundOutcome
from sketchloop.settings import GeneratorSettings, StorageSettings


def make_announced_generator(backend: GenerationBackend, mode: GenerationMode | None, console: Console, *,
                             models_dir: Path) -> tuple[Generator, float]:
    """
    Create and time the chosen generation backend, then print which backend, device, and mode it uses and the setup time.

    Args:
        backend (GenerationBackend): Backend to create.
        mode (GenerationMode | None): Diffusers mode, or None for the device default.
        console (Console): Console used for output.
        models_dir (Path): Folder with the downloaded diffusers models.

    Returns:
        tuple[Generator, float]: The generation backend and the seconds creating it took.
    """
    # Time only the creation, which for diffusers imports torch and selects the device but loads no model yet.
    started = time.perf_counter()
    generator = make_generator(backend, mode, models_dir=models_dir)
    setup_seconds = time.perf_counter() - started
    console.print(generator.describe())
    console.print(f"Generator setup: {setup_seconds:.2f} s")
    return generator, setup_seconds


def make_configured_generator(settings: GeneratorSettings, console: Console) -> tuple[Generator, float]:
    """
    Create, time, and announce the backend the settings name, with its diffusers mode and models folder.
    """
    mode = settings.diffusers.mode
    return make_announced_generator(GenerationBackend(settings.backend), GenerationMode(mode) if mode else None, console,
                                    models_dir=settings.diffusers.models_dir)


def require_folder_store(settings: StorageSettings, command: Command) -> ExperimentStore:
    """
    Create the configured store, refusing one without a local folder, since the command prints saved file paths.

    Args:
        settings (StorageSettings): Storage settings naming the store.
        command (Command): Command that needs the folder, named in the error.

    Returns:
        ExperimentStore: The store, whose root is a folder.
    """
    store = make_store(settings)
    if store.root is None:
        raise SketchLoopError(f"sketchloop {command} shows saved file paths, so it needs a store that keeps files in a folder.")

    return store


def show_round(outcome: RoundOutcome, console: Console) -> None:
    """
    Show a round's sketches, preprocessing steps, timing, and candidates, with scores when the evaluator gave any.

    Args:
        outcome (RoundOutcome): The round to show.
        console (Console): Console used for output.
    """
    # Show where the raw and generation-input sketches are, which preprocessing steps ran, and how long generation took.
    sketch = outcome.iteration.request.sketch
    console.rule(f"Round {outcome.number}")
    console.print(f"Raw sketch: {(outcome.folder / outcome.raw_sketch.path).as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Generation sketch: {(outcome.folder / sketch.path).as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Preprocessing: {', '.join(step.name for step in outcome.steps) or 'skipped'}", soft_wrap=True)
    console.print(f"Generation time: {outcome.timings_seconds[RoundStage.GENERATION]:.1f} s")

    # Warn clearly when the images are not from a real model.
    if outcome.iteration.result.backend.is_fake:
        warning = "These images are deterministic noise, not output from a real model."
        console.print(Panel(warning, title="FAKE BACKEND", style="yellow"))

    # List the candidates, adding a score column only when the evaluator gave scores.
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("Candidate image", overflow="fold")
    if outcome.scores:
        table.add_column("Score", justify="right")

    for candidate in outcome.iteration.result.candidates:
        cells = [str(candidate.index + 1), (outcome.folder / candidate.image.path).as_posix()]
        if outcome.scores:
            score = outcome.scores.get(candidate.id)
            cells.append("" if score is None else f"{score:.3f}")

        table.add_row(*cells)
    console.print(table)
