"""
Helpers shared by subcommands for creating the backend, the store, and printing a round.
"""

import time

from collections.abc import Mapping
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Final

from sketchloop.backends import GenerationBackend, make_generator
from sketchloop.backends.diffusers import GenerationMode
from sketchloop.cli.arguments import Command
from sketchloop.components import make_store
from sketchloop.domain import SketchLoopError, Unavailable
from sketchloop.experiments import ExperimentStore, Hardware, RoundStage, format_value
from sketchloop.generation import Generator
from sketchloop.session import RoundOutcome
from sketchloop.settings import GeneratorSettings, StorageSettings


GIBIBYTE: Final = 1024**3


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


def format_timings(timings_seconds: Mapping[str, float], total_seconds: float | Unavailable) -> str:
    """
    Format the recorded stages in RoundStage order, then the total when it was measured, such as "capture 0.02 s, total 1.20 s".

    Args:
        timings_seconds (Mapping[str, float]): Duration by stage name.
        total_seconds (float | Unavailable): Round total, left out when unavailable.

    Returns:
        str: The timings on one line, or "none" when nothing was timed.
    """
    parts = [f"{stage.replace('_', ' ')} {timings_seconds[stage]:.2f} s" for stage in RoundStage if stage in timings_seconds]

    # Leave out an unavailable total, since show explains older rounds in a note of its own.
    if not isinstance(total_seconds, Unavailable):
        parts.append(f"total {total_seconds:.2f} s")

    return ", ".join(parts) or "none"


def format_machine(hardware: Hardware | Unavailable) -> str:
    """
    Format the CPU, logical cores, and memory, such as "Apple M2 Pro, 12 logical cores, 32.0 GiB memory".
    """
    # A whole unavailable record is shown with its reason, and otherwise each fact on its own.
    if isinstance(hardware, Unavailable):
        return format_value(hardware)

    cpu, cores, memory = hardware.cpu, hardware.logical_cores, hardware.memory_bytes
    cpu_text = f"CPU {format_value(cpu)}" if isinstance(cpu, Unavailable) else cpu
    cores_text = f"logical cores {format_value(cores)}" if isinstance(cores, Unavailable) else f"{cores} logical cores"
    memory_text = f"memory {format_value(memory)}" if isinstance(memory, Unavailable) else f"{memory / GIBIBYTE:.1f} GiB memory"
    return f"{cpu_text}, {cores_text}, {memory_text}"


def format_hardware(hardware: Hardware | Unavailable, device: object) -> str:
    """
    Format the machine and the device the backend generated on, such as "Apple M2 Pro, ..., generation on mps".

    Args:
        hardware (Hardware | Unavailable): Hardware the session recorded.
        device (object): Device the round's backend reported, or None when it reported none.

    Returns:
        str: The hardware on one line.
    """
    reported = device is not None and not isinstance(device, Unavailable)
    generation = f"generation on {device}" if reported else "generation device not reported"
    return f"{format_machine(hardware)}, {generation}"


def show_round(outcome: RoundOutcome, console: Console, hardware: Hardware | Unavailable) -> None:
    """
    Show a round's sketches, preprocessing steps, timings, hardware, and candidates, with scores when the evaluator gave any.

    Args:
        outcome (RoundOutcome): The round to show.
        console (Console): Console used for output.
        hardware (Hardware | Unavailable): Hardware the session recorded.
    """
    # Show where the raw and generation-input sketches are, which preprocessing steps ran, each stage's time, and the machine.
    sketch = outcome.iteration.request.sketch
    device = outcome.iteration.result.effective.controls.get("device")
    console.rule(f"Round {outcome.number}")
    console.print(f"Raw sketch: {(outcome.folder / outcome.raw_sketch.path).as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Generation sketch: {(outcome.folder / sketch.path).as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Preprocessing: {', '.join(step.name for step in outcome.steps) or 'skipped'}", soft_wrap=True)
    console.print(f"Timings: {format_timings(outcome.timings_seconds, outcome.total_seconds)}", soft_wrap=True)
    console.print(f"Hardware: {format_hardware(hardware, device)}", markup=False, soft_wrap=True)

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
