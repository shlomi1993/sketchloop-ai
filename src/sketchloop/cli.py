import argparse
import time
import uuid

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sketchloop.capture import capture_from_camera, load_sketch_file
from sketchloop.diffusers_backend import DiffusersSketchGenerator, GenerationMode
from sketchloop.domain import (
    GenerationRequest, Guidance, Iteration, SelectionEvent, SketchLoopError, record_no_selection, select_candidates)
from sketchloop.fakes import FakeGenerator
from sketchloop.generation import Generator
from sketchloop.preprocessing import ROTATE_CODES, preprocess_sketch


class GenerationBackend(StrEnum):
    """
    Generation backends the command can use.
    """
    FAKE = "fake"
    DIFFUSERS = "diffusers"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse the sketchloop command-line options.

    Args:
        argv (list[str] | None): Arguments to parse, or None for the process arguments.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="sketchloop",
        description="Generate alternatives from a sketch and pick one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("sketch", type=Path, nargs="?", help="PNG or JPEG sketch image to start from")
    parser.add_argument("--camera", action="store_true", help="capture the sketch from a webcam instead of a file")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam to use with --camera, 0 for the default")
    parser.add_argument("--prompt", required=True, help="text guidance for generation")
    parser.add_argument("--candidates", type=int, default=4, help="number of candidates to generate")
    parser.add_argument("--seed", type=int, default=None, help="base seed for reproducible candidates")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="folder for run outputs")
    parser.add_argument("--raw", action="store_true", help="generate from your original sketch instead of the processed one")
    parser.add_argument("--rotate", type=int, choices=list(ROTATE_CODES), default=0, help="degrees to turn the sketch clockwise before processing")  # noqa: E501
    parser.add_argument("--backend", type=GenerationBackend, choices=list(GenerationBackend), default=GenerationBackend.FAKE, help="generation backend to use")  # noqa: E501
    parser.add_argument("--mode", type=GenerationMode, choices=list(GenerationMode), help="diffusers mode, by default fast on CPU and quality on a GPU")  # noqa: E501
    args = parser.parse_args(argv)

    # Take the sketch from exactly one source.
    if (args.sketch is None) == (not args.camera):
        parser.error("give exactly one of a sketch file or --camera")

    # Only the diffusers backend has modes.
    if args.mode is not None and args.backend != GenerationBackend.DIFFUSERS:
        parser.error(f"--mode needs --backend {GenerationBackend.DIFFUSERS}")

    # Rotation is part of preprocessing, which --raw skips.
    if args.rotate and args.raw:
        parser.error("--rotate cannot be combined with --raw, which skips preprocessing")

    return args


def make_generator(args: argparse.Namespace, console: Console) -> Generator:
    """
    Create the chosen generation backend and print which backend, device, and mode it uses.

    Args:
        args (argparse.Namespace): Parsed command-line options.
        console (Console): Console used for output.

    Returns:
        Generator: The generation backend.
    """
    if args.backend == GenerationBackend.FAKE:
        console.print(f"Using the {args.backend} backend, which makes test images, not real AI output.")
        return FakeGenerator()

    # Describe the real model, where it runs, and what the mode means, so slow CPU runs are expected.
    generator = DiffusersSketchGenerator(mode=args.mode)
    steps = next(spec.default for spec in generator.capabilities.controls if spec.name == "steps")
    model = "Stable Diffusion 1.5 and ControlNet scribble"
    device, mode = generator.device.upper(), generator.mode
    console.print(f"Using the {args.backend} backend with {model} on {device} in {mode} mode, running {steps} steps.")
    return generator


def ask_selection(iteration: Iteration, console: Console) -> SelectionEvent:
    """
    Ask the person which candidates they choose and record it explicitly.

    Args:
        iteration (Iteration): Iteration whose candidates were listed.
        console (Console): Console used for the prompt.

    Returns:
        SelectionEvent: The recorded selection, possibly of none.
    """
    # Read space-separated 1-based numbers, where an empty answer is an explicit choice of none.
    try:
        answer = console.input("[bold]Pick candidates by number (space separated), or press Enter for none:[/bold] ").split()
    except EOFError:
        raise SketchLoopError("No selection entered. Rerun and type numbers or press Enter.") from None

    if not answer:
        return record_no_selection(iteration)

    # Map each number to its candidate ID, rejecting anything outside the shown list.
    candidates = iteration.result.candidates
    all_numbers_valid = all(number.isdigit() and 1 <= int(number) <= len(candidates) for number in answer)
    if not all_numbers_valid:
        raise SketchLoopError(f"Invalid selection {' '.join(answer)!r}. Use numbers from 1 to {len(candidates)}.")

    return select_candidates(iteration, [candidates[int(number) - 1].id for number in answer])


def make_run_folder_name() -> str:
    """
    Name a run folder by local date and time plus a short random suffix, so folders sort by time and never clash.
    """
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def run(args: argparse.Namespace, console: Console) -> None:
    """
    Run one round: load or capture and preprocess the sketch, generate candidates, save files, and record the selection.

    Args:
        args (argparse.Namespace): Parsed command-line options.
        console (Console): Console used for output and the prompt.
    """
    # Load or capture the raw sketch and, unless skipped, preprocess it so generation uses the processed image.
    raw_sketch, raw_bytes = capture_from_camera(args.camera_index) if args.camera else load_sketch_file(args.sketch)
    sketch = raw_sketch
    files = {raw_sketch.path: raw_bytes}
    step_names = "skipped"
    if not args.raw:
        processed = preprocess_sketch(raw_bytes, args.rotate)
        sketch = processed.image
        files[sketch.path] = processed.payload
        step_names = ", ".join(step.name for step in processed.steps)

    # Generate candidates with the chosen backend and time the call, which includes a first model load.
    guidance = Guidance(prompt=args.prompt)
    request = GenerationRequest(sketch=sketch, guidance=guidance, n_candidates=args.candidates, seed=args.seed)
    generator = make_generator(args, console)
    started = time.perf_counter()
    output = generator.generate(request, files[sketch.path])
    elapsed_seconds = time.perf_counter() - started
    iteration = Iteration(parent_id=None, request=request, result=output.result)

    # Store the raw and processed sketches and each candidate image under a new, time-named run folder.
    run_dir: Path = args.runs_dir / make_run_folder_name()
    files_to_write: dict[str, bytes] = files | dict(output.payloads)
    for path, payload in files_to_write.items():
        target_file: Path = run_dir / path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(payload)

    # Show where the raw and generation-input sketches are, and which preprocessing steps ran.
    console.print(f"Raw sketch: {(run_dir / raw_sketch.path).as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Generation sketch: {(run_dir / sketch.path).as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Preprocessing: {step_names}", markup=False, soft_wrap=True)
    console.print(f"Generation time: {elapsed_seconds:.1f} s", markup=False)

    # Warn clearly when the images are not from a real model, then list them.
    if output.result.backend.is_fake:
        warning = "These images are deterministic noise, not output from a real model."
        console.print(Panel(warning, title="FAKE BACKEND", style="yellow"))

    table = Table()
    table.add_column("#", justify="right")
    table.add_column("Candidate image", overflow="fold")
    for candidate in iteration.result.candidates:
        table.add_row(str(candidate.index + 1), (run_dir / candidate.image.path).as_posix())
    console.print(table)

    # Record the person's explicit choice and summarize the run.
    selection = ask_selection(iteration, console)
    selected_ids = selection.selected_candidate_ids
    chosen = [str(candidate.index + 1) for candidate in iteration.result.candidates if candidate.id in selected_ids]
    console.print(f"Run folder: {run_dir.as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Selected: {', '.join(chosen) if chosen else 'none'}", style="green", markup=False)


def main(argv: list[str] | None = None) -> int:
    """
    Run one sketchloop round from the command line and report errors in one line.

    Args:
        argv (list[str] | None, optional): Command-line arguments. Defaults to the process arguments.

    Returns:
        int: Process exit code, 0 on success and 1 on a reported error.
    """
    args = parse_args(argv)
    try:
        run(args, Console())
    except (SketchLoopError, OSError) as error:
        Console(stderr=True).print(f"sketchloop error: {error}", style="red", markup=False, highlight=False, soft_wrap=True)
        return 1

    return 0
