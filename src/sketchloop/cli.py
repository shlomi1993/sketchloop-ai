import argparse

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sketchloop.capture import load_sketch_file
from sketchloop.domain import (
    GenerationRequest, Guidance, Iteration, SelectionEvent, SketchLoopError, record_no_selection, select_candidates)
from sketchloop.fakes import FakeGenerator


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sketchloop",
        description="Generate alternatives from a sketch and pick one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("sketch", type=Path, help="PNG sketch image to start from")
    parser.add_argument("--prompt", required=True, help="text guidance for generation")
    parser.add_argument("--candidates", type=int, default=4, help="number of candidates to generate")
    parser.add_argument("--seed", type=int, default=None, help="base seed for reproducible candidates")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="folder for run outputs")
    return parser.parse_args(argv)


def ask_selection(iteration: Iteration, console: Console) -> SelectionEvent:
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


def run(args: argparse.Namespace, console: Console) -> None:
    # Load the sketch and generate candidates with the fake backend.
    sketch, sketch_bytes = load_sketch_file(args.sketch)
    guidance = Guidance(prompt=args.prompt)
    request = GenerationRequest(sketch=sketch, guidance=guidance, n_candidates=args.candidates, seed=args.seed)
    output = FakeGenerator().generate(request)
    iteration = Iteration(parent_id=None, request=request, result=output.result)

    # Store the sketch and each candidate image under a new run folder named after the iteration.
    run_dir: Path = args.runs_dir / iteration.id
    sketch_file: Path = run_dir / sketch.path
    sketch_file.parent.mkdir(parents=True, exist_ok=True)
    sketch_file.write_bytes(sketch_bytes)
    for path, payload in output.payloads.items():
        candidate_file: Path = run_dir / path
        candidate_file.parent.mkdir(parents=True, exist_ok=True)
        candidate_file.write_bytes(payload)

    # Warn clearly that the images are not from a real model, then list them.
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
