"""
The rerun subcommand, which repeats a saved round in a new linked session and compares the results.
"""

import argparse

from pathlib import Path
from rich.console import Console

from sketchloop.backends import RERUNNABLE_BACKENDS, GenerationBackend
from sketchloop.backends.diffusers import GenerationMode, TorchDevice, TorchPrecision, select_device
from sketchloop.cli.arguments import Command
from sketchloop.cli.common import make_announced_generator, require_folder_store, show_round
from sketchloop.cli.show import print_comparison
from sketchloop.components import make_evaluator
from sketchloop.domain import SketchLoopError
from sketchloop.experiments import (RerunLink, RoundRecord, RoundStatus, SavedSession, compare_rounds, load_session,
                                    rebuild_request)
from sketchloop.generation import Generator
from sketchloop.preprocessing import PreprocessedSketch
from sketchloop.session import SketchSession


def find_rerunnable_round(saved: SavedSession, round_number: int) -> RoundRecord:
    """
    Return a completed round whose files pass their integrity checks.

    Args:
        saved (SavedSession): Loaded session.
        round_number (int): Round to rerun.

    Returns:
        RoundRecord: The round.
    """
    completed = [record.number for record in saved.rounds if record.status == RoundStatus.COMPLETE]
    if round_number not in completed:
        listed = ", ".join(str(number) for number in completed) or "none"
        session_id = saved.session.session_id
        raise SketchLoopError(f"Session {session_id} has no completed round {round_number}. Completed rounds: {listed}.")

    # Refuse damaged inputs, since the rerun would not repeat the recorded conditions.
    if any(problem.startswith(f"round-{round_number}/") for problem in saved.problems):
        folder = saved.folder.as_posix()
        raise SketchLoopError(f"Round {round_number} fails integrity checks. Run sketchloop show {folder} for details.")

    return next(record for record in saved.rounds if record.number == round_number)


def make_rerun_generator(record: RoundRecord, console: Console, models_dir: Path) -> tuple[Generator, float]:
    """
    Recreate the backend a round used, with its recorded diffusers mode and precision.

    Args:
        record (RoundRecord): Completed round to rerun.
        console (Console): Console used for output.
        models_dir (Path): Folder with the downloaded diffusers models.

    Returns:
        tuple[Generator, float]: The recreated backend and the seconds creating it took.
    """
    adapter = record.iteration.result.backend.adapter
    if adapter not in RERUNNABLE_BACKENDS:
        raise SketchLoopError(f"Round {record.number} used backend {adapter}, which rerun cannot recreate.")

    # Pass the recorded mode and precision, since the diffusers defaults depend on the device.
    controls = record.iteration.result.effective.controls
    recorded_mode, recorded_precision = controls.get("mode"), controls.get("dtype")
    mode = GenerationMode(recorded_mode) if recorded_mode in list(GenerationMode) else None
    precision = TorchPrecision(recorded_precision) if recorded_precision in list(TorchPrecision) else None

    # A float16 round came from a GPU, and rerun keeps the recorded precision, so a CPU-only machine cannot repeat it.
    is_diffusers = RERUNNABLE_BACKENDS[adapter] == GenerationBackend.DIFFUSERS
    if is_diffusers and precision == TorchPrecision.FLOAT16 and select_device() == TorchDevice.CPU:
        raise SketchLoopError(f"Round {record.number} ran in float16 on a GPU, so it cannot be rerun on this CPU-only machine.")

    return make_announced_generator(RERUNNABLE_BACKENDS[adapter], mode, console, models_dir=models_dir, precision=precision)


def rerun_round(args: argparse.Namespace, console: Console) -> None:
    """
    Rerun a saved round as a new session linked to it, then compare the candidates and conditions.

    Args:
        args (argparse.Namespace): Parsed rerun options.
        console (Console): Console used for output.
    """
    # Take only the store, evaluator, and models folder from settings, and everything else from the record.
    settings = args.settings
    store = require_folder_store(settings.storage, Command.RERUN)
    original = load_session(args.session_folder)
    record = find_rerunnable_round(original, args.round_number)
    generator, setup_seconds = make_rerun_generator(record, console, settings.generator.diffusers.models_dir)

    # Read the stored sketches, reusing the processed one so generation gets the same input bytes.
    round_folder = original.folder / f"round-{record.number}"
    raw_payload = (round_folder / record.raw_sketch.path).read_bytes()
    sketch = record.request.sketch
    reuse = None
    if not record.use_raw:
        reuse = PreprocessedSketch(image=sketch, payload=(round_folder / sketch.path).read_bytes(), steps=record.steps)

    # Run one round in a new linked session with the recorded settings, without asking for a selection or timing a capture.
    request = rebuild_request(record, generator.capabilities)
    link = RerunLink(session_id=original.session.session_id, round_number=record.number, iteration_id=record.iteration.id)
    session = SketchSession(generator, store, make_evaluator(settings.evaluator), rerun_of=link,
                            generator_setup_seconds=setup_seconds)
    outcome = session.run_round(record.raw_sketch, raw_payload, request.guidance, source=record.source,
                                rotation=record.rotation, use_raw=record.use_raw, n_candidates=request.n_candidates,
                                seed=request.seed, reuse=reuse)
    show_round(outcome, console, session.session_record.environment.hardware)
    console.print(f"Rerun session folder: {session.folder.as_posix()}", markup=False, soft_wrap=True)
    print_comparison(compare_rounds(original, record.number, load_session(session.folder)), console)
