import argparse

from enum import StrEnum
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sketchloop.capture import capture_from_camera, load_sketch_file
from sketchloop.diffusers_backend import DiffusersSketchGenerator, GenerationMode
from sketchloop.domain import Guidance, Iteration, SelectionEvent, SketchLoopError, record_no_selection, select_candidates
from sketchloop.fakes import FakeGenerator
from sketchloop.generation import Generator
from sketchloop.preprocessing import ROTATE_CODES
from sketchloop.session import RoundOutcome, SketchSession


class GenerationBackend(StrEnum):
    """
    Generation backends the command can use.
    """
    FAKE = "fake"
    DIFFUSERS = "diffusers"


class NextStep(StrEnum):
    """
    Menu choices after a round: what to change before the next one, or quit.
    """
    REPEAT = ""
    PROMPT = "p"
    RECAPTURE = "c"
    FILE = "f"
    QUIT = "q"


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


def describe_selection(outcome: RoundOutcome, selection: SelectionEvent) -> str:
    """
    Turn a selection into the candidate numbers shown in the table, such as "1, 3", or "none" when nothing was picked.
    """
    selected_ids = selection.selected_candidate_ids
    chosen = [str(candidate.index + 1) for candidate in outcome.iteration.result.candidates if candidate.id in selected_ids]
    return ", ".join(chosen) or "none"


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
    console.print(f"Generation time: {outcome.elapsed_seconds:.1f} s")

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


def read_answer(console: Console, question: str) -> str | None:
    """
    Read one stripped line of input, or None at the end of input.
    """
    try:
        return console.input(question).strip()
    except EOFError:
        return None


def ask_next_step(console: Console, can_recapture: bool) -> tuple[NextStep, str] | None:
    """
    Ask what to change for the next round, re-asking with a hint on invalid input.

    Args:
        console (Console): Console used for the prompt.
        can_recapture (bool): Whether the session uses a camera, so recapturing is offered.

    Returns:
        tuple[NextStep, str] | None: The change and its value (a prompt or file path), or None to quit.
    """
    recapture = "c = recapture, " if can_recapture else ""
    choices = f"Enter = again, p = new prompt, {recapture}f <path> = new file, q = quit"
    while True:
        # Quit on end of input or q.
        answer = read_answer(console, f"[bold]Next ({choices}):[/bold] ")
        if answer is None or answer == NextStep.QUIT:
            return None

        # Repeat, recapture, or load a file, when that choice is available.
        command, _, value = answer.partition(" ")
        is_available = answer == NextStep.REPEAT or (answer == NextStep.RECAPTURE and can_recapture)
        if is_available or (command == NextStep.FILE and value.strip()):
            return NextStep(command), value.strip()

        # Ask for the new prompt, quitting at the end of input and re-asking the menu when it is empty.
        if answer == NextStep.PROMPT:
            prompt = read_answer(console, "[bold]New prompt:[/bold] ")
            if prompt is None:
                return None

            if prompt:
                return NextStep.PROMPT, prompt

        console.print(f"Not a choice. Use: {choices}.")


def print_summary(session: SketchSession, console: Console) -> None:
    """
    Print the session folder, the number of rounds, and each round's selection.

    Args:
        session (SketchSession): Session to summarize.
        console (Console): Console used for output.
    """
    console.rule("Session summary")
    console.print(f"Session folder: {session.folder.as_posix()}", markup=False, soft_wrap=True)
    console.print(f"Rounds: {len(session.rounds)}")
    for outcome in session.rounds:
        console.print(f"Round {outcome.number} selected: {describe_selection(outcome, session.selections[outcome.iteration.id])}")


def run(args: argparse.Namespace, console: Console) -> None:
    """
    Run manual rounds until the person quits: generate, show candidates, record the selection, and apply the next change.

    Args:
        args (argparse.Namespace): Parsed command-line options, applied to every round.
        console (Console): Console used for output and prompts.
    """
    # Load or capture the first sketch, then create the backend once so a real model stays loaded between rounds.
    raw_sketch, raw_payload = capture_from_camera(args.camera_index) if args.camera else load_sketch_file(args.sketch)
    session = SketchSession(make_generator(args, console), args.runs_dir)
    prompt = args.prompt
    while True:

        # Run a round with the current sketch and prompt, then record the person's explicit choice.
        guidance = Guidance(prompt=prompt)
        outcome = session.run_round(raw_sketch, raw_payload, guidance, rotation=args.rotate, use_raw=args.raw,
                                    n_candidates=args.candidates, seed=args.seed)
        show_round(outcome, console)
        selection = ask_selection(outcome.iteration, console)
        session.record_selection(selection)
        console.print(f"Selected: {describe_selection(outcome, selection)}", style="green")

        # Stop when the person quits, otherwise apply the requested change before the next round.
        next_step = ask_next_step(console, args.camera)
        if next_step is None:
            break

        step, value = next_step
        if step == NextStep.PROMPT:
            prompt = value
        elif step == NextStep.RECAPTURE:
            raw_sketch, raw_payload = capture_from_camera(args.camera_index)
        elif step == NextStep.FILE:
            raw_sketch, raw_payload = load_sketch_file(Path(value))

    print_summary(session, console)


def main(argv: list[str] | None = None) -> int:
    """
    Run a sketchloop session from the command line and report errors in one line.

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
