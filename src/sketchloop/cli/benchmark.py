"""
The benchmark subcommand, which times warm-up and measured rounds and prints per-stage statistics.
"""

import argparse

from rich.console import Console
from typing import Final

from sketchloop.capture import FileCapture
from sketchloop.cli.arguments import Command
from sketchloop.cli.common import format_hardware, make_configured_generator, require_folder_store
from sketchloop.cli.show import format_controls, format_seconds, print_benchmark_summary
from sketchloop.components import make_evaluator, make_preprocessor
from sketchloop.domain import Guidance
from sketchloop.experiments import BenchmarkPlan, BenchmarkSummary, SavedSession, format_value, load_session, summarize_benchmark
from sketchloop.session import SketchSession


# Benchmark pins a seed, unlike run, so repeated benchmarks generate the same images.
BENCHMARK_DEFAULT_SEED: Final = 0


def print_benchmark_conditions(saved: SavedSession, summary: BenchmarkSummary, console: Console) -> None:
    """
    Print the backend, settings, sketch sizes, generator setup, and machine of the first measured round.

    Args:
        saved (SavedSession): The loaded benchmark session.
        summary (BenchmarkSummary): Statistics of the session, naming its measured rounds.
        console (Console): Console used for output.
    """
    # Without a completed measured round there are no conditions to describe.
    console.rule("Benchmark conditions")
    if not summary.measured_rounds:
        console.print("No measured round completed.", style="yellow")
        return

    # Describe the backend and the settings it used, labeling a fake backend clearly.
    record = next(record for record in saved.rounds if record.number == summary.measured_rounds[0])
    request, result = record.request, record.iteration.result
    backend, controls = result.backend, result.effective.controls
    console.print(f"Backend: {backend.adapter} {backend.adapter_version}, model {format_value(backend.model_id)}", markup=False)
    if backend.is_fake:
        console.print("FAKE BACKEND: these timings measure deterministic noise, not a real model.", style="yellow")

    # Describe the inputs, the setup outside every round, and the machine.
    environment, raw_sketch, sketch = saved.session.environment, record.raw_sketch, request.sketch
    effective = {name: value for name, value in controls.items() if name != "seeds"}
    console.print(f"Effective: {format_controls(effective)}", markup=False, soft_wrap=True)
    console.print(f"Candidates: {request.n_candidates}, seed {format_value(request.seed)}")
    console.print(f"Sketch sizes: raw {raw_sketch.width}x{raw_sketch.height}, generation {sketch.width}x{sketch.height}")
    console.print(f"Preprocessing: {', '.join(step.name for step in record.steps) or 'skipped'}", soft_wrap=True)
    console.print(f"Generator setup: {format_seconds(saved.session.generator_setup_seconds)}, outside every round")
    console.print(f"Hardware: {format_hardware(environment.hardware, controls.get('device'))}", markup=False, soft_wrap=True)
    console.print(f"Python {environment.py_version} on {environment.platform}", markup=False)


def run_benchmark(args: argparse.Namespace, console: Console) -> None:
    """
    Run warm-up and measured rounds from one sketch file with one loaded backend, then print timing statistics.

    Args:
        args (argparse.Namespace): Parsed benchmark options.
        console (Console): Console used for output.
    """
    # Load the sketch first, so a bad file fails before the backend is created.
    settings = args.settings
    store = require_folder_store(settings.storage, Command.BENCHMARK)
    capture_source = FileCapture(args.sketch)
    first_capture = capture_source.capture()
    generator, setup_seconds = make_configured_generator(settings.generator, console)
    plan = BenchmarkPlan(warmup_rounds=args.warmup, measured_rounds=args.rounds)
    session = SketchSession(generator, store, make_evaluator(settings.evaluator),
                            preprocessor=make_preprocessor(settings.preprocessing), benchmark=plan,
                            generator_setup_seconds=setup_seconds)

    # Run every round with the same prompt, settings, and seed, loading the file again so capture is measured.
    preprocessing, run_settings = settings.preprocessing, settings.run
    seed = run_settings.seed if run_settings.seed is not None else BENCHMARK_DEFAULT_SEED
    round_count = plan.warmup_rounds + plan.measured_rounds
    for number in range(1, round_count + 1):
        captured = first_capture if number == 1 else capture_source.capture()
        outcome = session.run_round(captured.image, captured.payload, Guidance(prompt=args.prompt), source=capture_source.kind,
                                    rotation=preprocessing.rotation, use_raw=preprocessing.raw,
                                    n_candidates=run_settings.candidates, seed=seed, capture_seconds=captured.capture_seconds)
        role = "warm-up" if number <= plan.warmup_rounds else "measured"
        console.print(f"Round {number} of {round_count} ({role}): total {outcome.total_seconds:.2f} s")

    # Recompute the statistics from the saved rounds, as show and later analyses do.
    saved = load_session(session.folder)
    summary = summarize_benchmark(saved.rounds, plan)
    console.print(f"Benchmark session folder: {session.folder.as_posix()}", markup=False, soft_wrap=True)
    print_benchmark_conditions(saved, summary, console)
    print_benchmark_summary(summary, saved.rounds, console)
