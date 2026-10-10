import argparse
import os

from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Final

from sketchloop.capture import capture_from_camera, load_sketch_file
from sketchloop.diffusers_backend import DiffusersSketchGenerator, GenerationMode
from sketchloop.domain import (Guidance, ImageRef, Iteration, SelectionEvent, SketchLoopError, record_no_selection,
                               select_candidates)
from sketchloop.experiments import (SESSION_FILE, ExperimentRecordError, RerunComparison, RerunLink, RoundRecord,
                                    SavedSession, compare_rounds, format_value, load_session, rebuild_request)
from sketchloop.fakes import FakeGenerator
from sketchloop.generation import Generator
from sketchloop.preprocessing import ROTATE_CODES, PreprocessedSketch
from sketchloop.session import RoundOutcome, SketchSession


class GenerationBackend(StrEnum):
    """
    Generation backends the command can use.
    """
    FAKE = "fake"
    DIFFUSERS = "diffusers"


# Backends rerun can recreate, by the adapter name a round recorded.
RERUNNABLE_BACKENDS: Final = {"sketchloop.fake": GenerationBackend.FAKE, "sketchloop.diffusers": GenerationBackend.DIFFUSERS}
REPLAY_LIMITS: Final = "A rerun restores the recorded conditions. Equal seeds do not guarantee identical images across devices, library versions, or model revisions."  # noqa: E501


class Command(StrEnum):
    """
    Subcommands of the sketchloop command.
    """
    RUN = "run"
    SHOW = "show"
    RERUN = "rerun"


class NextStep(StrEnum):
    """
    Menu choices after a round: what to change before the next one, or quit.
    """
    REPEAT = ""
    PROMPT = "p"
    RECAPTURE = "c"
    FILE = "f"
    QUIT = "q"


def make_int_parser(minimum: int) -> Callable[[str], int]:
    """
    Make an argparse type that reads an integer of at least the minimum, so bad values fail before anything is saved.

    Args:
        minimum (int): Smallest accepted value.

    Returns:
        Callable[[str], int]: Function that parses the option text.
    """
    def parse_int(text: str) -> int:
        # Name the limit in the error, which argparse prints with the option name.
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")

        return value

    return parse_int


def check_run_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """
    Reject run option combinations that contradict each other.

    Args:
        parser (argparse.ArgumentParser): The run subcommand parser, used to report errors.
        args (argparse.Namespace): Parsed run options.
    """
    # Take the sketch from exactly one source.
    if (args.sketch is None) == (not args.camera):
        parser.error("give exactly one of a sketch file or --camera")

    # Only the diffusers backend has modes.
    if args.mode is not None and args.backend != GenerationBackend.DIFFUSERS:
        parser.error(f"--mode needs --backend {GenerationBackend.DIFFUSERS}")

    # Rotation is part of preprocessing, which --raw skips.
    if args.rotate and args.raw:
        parser.error("--rotate cannot be combined with --raw, which skips preprocessing")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse the sketchloop subcommand and its options.

    Args:
        argv (list[str] | None): Arguments to parse, or None for the process arguments.

    Returns:
        argparse.Namespace: Parsed options, with the subcommand in `command`.
    """
    # Let the environment move session data out of the repository, with --runs-dir still taking precedence.
    runs_dir = Path(os.environ.get("SKETCHLOOP_RUNS_DIR", "runs"))
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog="sketchloop", description="Run, inspect, and rerun sketch sessions.")
    commands = parser.add_subparsers(dest="command", required=True)

    # Run a new session.
    run_parser = commands.add_parser(Command.RUN, help="run a new session", description="Generate alternatives from a sketch and pick some.", formatter_class=formatter)  # noqa: E501
    run_parser.add_argument("sketch", type=Path, nargs="?", help="PNG or JPEG sketch image to start from")
    run_parser.add_argument("--camera", action="store_true", help="capture the sketch from a webcam instead of a file")
    run_parser.add_argument("--camera-index", type=make_int_parser(0), default=0, help="webcam to use with --camera, 0 for the default")  # noqa: E501
    run_parser.add_argument("--prompt", required=True, help="text guidance for generation")
    run_parser.add_argument("--candidates", type=make_int_parser(1), default=4, help="number of candidates to generate")
    run_parser.add_argument("--seed", type=make_int_parser(0), default=None, help="base seed for reproducible candidates")
    run_parser.add_argument("--runs-dir", type=Path, default=runs_dir, help="folder for run outputs, or set SKETCHLOOP_RUNS_DIR")  # noqa: E501
    run_parser.add_argument("--raw", action="store_true", help="generate from your original sketch instead of the processed one")  # noqa: E501
    run_parser.add_argument("--rotate", type=int, choices=list(ROTATE_CODES), default=0, help="degrees to turn the sketch clockwise before processing")  # noqa: E501
    run_parser.add_argument("--backend", type=GenerationBackend, choices=list(GenerationBackend), default=GenerationBackend.FAKE, help="generation backend to use")  # noqa: E501
    run_parser.add_argument("--mode", type=GenerationMode, choices=list(GenerationMode), help="diffusers mode, by default fast on CPU and quality on a GPU")  # noqa: E501

    # Show or rerun a saved session.
    show_parser = commands.add_parser(Command.SHOW, help="show a saved session without loading a model", description="Show a saved session and check its files.")  # noqa: E501
    show_parser.add_argument("session_folder", type=Path, help="session folder, such as runs/<session>")
    rerun_parser = commands.add_parser(Command.RERUN, help="rerun a saved round as a new linked session", description="Rerun a saved round with its recorded sketch and settings.", formatter_class=formatter)  # noqa: E501
    rerun_parser.add_argument("session_folder", type=Path, help="session folder, such as runs/<session>")
    rerun_parser.add_argument("--round", type=make_int_parser(1), required=True, dest="round_number", help="round number to rerun")  # noqa: E501
    rerun_parser.add_argument("--runs-dir", type=Path, default=runs_dir, help="folder for the new session, or set SKETCHLOOP_RUNS_DIR")  # noqa: E501
    args = parser.parse_args(argv)

    if args.command == Command.RUN:
        check_run_options(run_parser, args)

    return args


def make_generator(backend: GenerationBackend, mode: GenerationMode | None, console: Console) -> Generator:
    """
    Create the chosen generation backend and print which backend, device, and mode it uses.

    Args:
        backend (GenerationBackend): Backend to create.
        mode (GenerationMode | None): Diffusers mode, or None for the device default.
        console (Console): Console used for output.

    Returns:
        Generator: The generation backend.
    """
    if backend == GenerationBackend.FAKE:
        console.print(f"Using the {backend} backend, which makes test images, not real AI output.")
        return FakeGenerator()

    # Describe the real model, where it runs, and what the mode means, so slow CPU runs are expected.
    generator = DiffusersSketchGenerator(mode=mode)
    steps = next(spec.default for spec in generator.capabilities.controls if spec.name == "steps")
    model = "Stable Diffusion 1.5 and ControlNet scribble"
    device = generator.device.upper()
    console.print(f"Using the {backend} backend with {model} on {device} in {generator.mode} mode, running {steps} steps.")
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
    candidates = iteration.result.candidates
    while True:
        # Read space-separated 1-based numbers, where an empty answer is an explicit choice of none.
        try:
            answer = console.input("[bold]Pick candidates by number (space separated), or press Enter for none:[/bold] ").split()
        except EOFError:
            raise SketchLoopError("No selection entered. Rerun and type numbers or press Enter.") from None

        if not answer:
            return record_no_selection(iteration)

        # Accept only distinct numbers from the shown list, and ask again otherwise.
        numbers = [int(number) for number in answer if number.isdecimal()]
        all_numbers_valid = len(numbers) == len(answer) and all(1 <= number <= len(candidates) for number in numbers)
        if all_numbers_valid and len(set(numbers)) == len(numbers):
            return select_candidates(iteration, [candidates[number - 1].id for number in numbers])

        console.print(f"Use distinct numbers from 1 to {len(candidates)}, or press Enter for none.", style="yellow")


def describe_selection(iteration: Iteration, selection: SelectionEvent) -> str:
    """
    Turn a selection into the candidate numbers shown in the table, such as "1, 3", or "none" when nothing was picked.
    """
    selected_ids = selection.selected_candidate_ids
    chosen = [str(candidate.index + 1) for candidate in iteration.result.candidates if candidate.id in selected_ids]
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
    console.print(f"Generation time: {outcome.timings_seconds['generation']:.1f} s")

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


def ask_next_round(args: argparse.Namespace, console: Console, prompt: str,
                   sketch: tuple[ImageRef, bytes]) -> tuple[str, tuple[ImageRef, bytes]] | None:
    """
    Ask for the next change until one applies, so a wrong file path or a cancelled capture asks again.

    Args:
        args (argparse.Namespace): Parsed run options, for the camera settings.
        console (Console): Console used for prompts and errors.
        prompt (str): Current prompt.
        sketch (tuple[ImageRef, bytes]): Current raw sketch reference and bytes.

    Returns:
        tuple[str, tuple[ImageRef, bytes]] | None: The prompt and sketch for the next round, or None to quit.
    """
    while True:
        next_step = ask_next_step(console, args.camera)
        if next_step is None:
            return None

        # Load or capture a new sketch, and ask again when that fails.
        step, value = next_step
        try:
            if step == NextStep.RECAPTURE:
                return prompt, capture_from_camera(args.camera_index)
            if step == NextStep.FILE:
                return prompt, load_sketch_file(Path(value))
        except (SketchLoopError, OSError) as error:
            console.print(f"{error} Choose again.", style="red", markup=False, soft_wrap=True)
            continue

        return (value if step == NextStep.PROMPT else prompt), sketch


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
        selection = session.selections[outcome.iteration.id]
        console.print(f"Round {outcome.number} selected: {describe_selection(outcome.iteration, selection)}")


def run_session(args: argparse.Namespace, console: Console) -> None:
    """
    Run manual rounds until the person quits: generate, show candidates, record the selection, and apply the next change.

    Args:
        args (argparse.Namespace): Parsed command-line options, applied to every round.
        console (Console): Console used for output and prompts.
    """
    # Load or capture the first sketch, then create the backend once so a real model stays loaded between rounds.
    sketch = capture_from_camera(args.camera_index) if args.camera else load_sketch_file(args.sketch)
    session = SketchSession(make_generator(args.backend, args.mode, console), args.runs_dir)
    prompt = args.prompt
    while True:
        # Run a round with the current sketch and prompt, reporting a failure instead of ending the session.
        try:
            outcome = session.run_round(*sketch, Guidance(prompt=prompt), rotation=args.rotate, use_raw=args.raw,
                                        n_candidates=args.candidates, seed=args.seed)
        except (SketchLoopError, OSError) as error:
            console.print(f"Round failed: {error}", style="red", markup=False, soft_wrap=True)
        else:
            # Show the candidates and record the person's explicit choice.
            show_round(outcome, console)
            selection = ask_selection(outcome.iteration, console)
            session.record_selection(selection)
            console.print(f"Selected: {describe_selection(outcome.iteration, selection)}", style="green")

        # Stop when the person quits, otherwise apply the requested change before the next round.
        next_round = ask_next_round(args, console, prompt, sketch)
        if next_round is None:
            break

        prompt, sketch = next_round

    print_summary(session, console)


def format_controls(controls: Mapping[str, object]) -> str:
    """
    Format controls as name=value pairs on one line, or "none".
    """
    return ", ".join(f"{name}={format_value(value)}" for name, value in controls.items()) or "none"


def print_session_header(saved: SavedSession, console: Console) -> None:
    """
    Print a saved session's ID, creation time, Python, platform, installed packages, and rerun link.

    Args:
        saved (SavedSession): Loaded session.
        console (Console): Console used for output.
    """
    session, environment = saved.session, saved.session.environment
    console.rule(f"Session {session.session_id}")
    console.print(f"Created: {session.created_at.isoformat()}")
    console.print(f"Python {environment.py_version} on {environment.platform}", markup=False)
    installed = ", ".join(f"{name} {version}" for name, version in environment.packages.items() if isinstance(version, str))
    console.print(f"Packages: {installed or 'none'}", markup=False, soft_wrap=True)
    if session.rerun_of is not None:
        console.print(f"Rerun of: session {session.rerun_of.session_id} round {session.rerun_of.round_number}")


def print_candidates(record: RoundRecord, console: Console) -> None:
    """
    Print a completed round's candidates with seed, score, selected mark, and image path.

    Args:
        record (RoundRecord): Completed round.
        console (Console): Console used for output.
    """
    table = Table()
    for column in ("#", "Seed", "Score", "Selected", "Candidate image"):
        table.add_column(column)

    # Mark the selected candidates and show a score only where the evaluator gave one.
    selected_ids = record.selection.selected_candidate_ids if record.selection else ()
    for candidate in record.iteration.result.candidates:
        score = record.scores.get(candidate.id)
        table.add_row(str(candidate.index + 1), format_value(candidate.seed), "" if score is None else f"{score:.3f}",
                      "yes" if candidate.id in selected_ids else "", candidate.image.path)
    console.print(table)


def print_round_record(record: RoundRecord, console: Console) -> None:
    """
    Print a saved round's request, input, preprocessing, timings, and either its result and selection or its failure.

    Args:
        record (RoundRecord): Saved round.
        console (Console): Console used for output.
    """
    # Show what was asked, from which input, and how long each stage took.
    console.rule(f"Round {record.number} ({record.status})")
    if record.request is not None:
        guidance = record.request.guidance
        console.print(f"Prompt: {guidance.prompt}", markup=False, soft_wrap=True)
        console.print(f"Negative prompt: {guidance.negative_prompt or 'none'}", markup=False, soft_wrap=True)
        console.print(f"Requested: {format_controls(guidance.controls)}, seed {format_value(record.request.seed)}", markup=False)

    source = "raw sketch" if record.use_raw else f"processed sketch, rotation {record.rotation}"
    console.print(f"Input: {source}, from {record.raw_sketch.path}")
    console.print(f"Preprocessing: {', '.join(step.name for step in record.steps) or 'skipped'}", soft_wrap=True)
    timings = ", ".join(f"{stage} {seconds:.2f} s" for stage, seconds in record.timings_seconds.items())
    console.print(f"Timings: {timings or 'none'}")
    if record.failure is not None:
        failure = record.failure
        console.print(f"Failed during {failure.stage}: {failure.error}: {failure.message}", style="red", markup=False)
        return

    # Show the backend, the settings it used, the candidates, and the selection.
    backend = record.iteration.result.backend
    model = format_value(backend.model_id)
    console.print(f"Backend: {backend.adapter} {backend.adapter_version}, model {model}", markup=False, soft_wrap=True)
    if backend.is_fake:
        console.print("FAKE BACKEND: these images are deterministic noise, not output from a real model.", style="yellow")

    console.print(f"Effective: {format_controls(record.iteration.result.effective.controls)}", markup=False, soft_wrap=True)
    print_candidates(record, console)
    selection = describe_selection(record.iteration, record.selection) if record.selection else "not recorded"
    console.print(f"Selection: {selection}")


def print_comparison(comparison: RerunComparison, console: Console) -> None:
    """
    Print which candidates have identical bytes, each difference from the original round, and the replay limits.

    Args:
        comparison (RerunComparison): Comparison of a rerun round with its original.
        console (Console): Console used for output.
    """
    console.rule("Compared with the original round")
    identical = ", ".join(str(number) for number in comparison.identical_candidates) or "none"
    console.print(f"Candidates with identical bytes: {identical} of {comparison.candidate_count}")
    for difference in comparison.differences:
        console.print(difference, markup=False, soft_wrap=True)

    if not comparison.differences:
        console.print("No differences in backend, settings, or environment.")

    console.print(REPLAY_LIMITS)


def show_session(args: argparse.Namespace, console: Console) -> None:
    """
    Print a saved session's records and integrity problems without creating a generator.

    Args:
        args (argparse.Namespace): Parsed show options.
        console (Console): Console used for output.
    """
    saved = load_session(args.session_folder)
    print_session_header(saved, console)
    for record in saved.rounds:
        print_round_record(record, console)

    # Report missing or changed files, or confirm that all match.
    console.rule("Integrity")
    for problem in saved.problems:
        console.print(problem, style="red", markup=False)

    if not saved.problems:
        console.print(f"All {saved.checked_files} files match their recorded checksums.", style="green")

    # Compare a rerun with its original when the original session is still next to it.
    link = saved.session.rerun_of
    original_folder = saved.folder.parent / link.session_id if link else None
    if link is not None and (original_folder / SESSION_FILE).is_file():
        try:
            print_comparison(compare_rounds(load_session(original_folder), link.round_number, saved), console)
        except ExperimentRecordError as error:
            console.print(f"Cannot compare with the original session: {error}", style="yellow", markup=False)


def find_rerunnable_round(saved: SavedSession, round_number: int) -> RoundRecord:
    """
    Return a completed round whose files pass their integrity checks.

    Args:
        saved (SavedSession): Loaded session.
        round_number (int): Round to rerun.

    Returns:
        RoundRecord: The round.
    """
    completed = [record.number for record in saved.rounds if record.status == "complete"]
    if round_number not in completed:
        listed = ", ".join(str(number) for number in completed) or "none"
        session_id = saved.session.session_id
        raise SketchLoopError(f"Session {session_id} has no completed round {round_number}. Completed rounds: {listed}.")

    # Refuse damaged inputs, since the rerun would not repeat the recorded conditions.
    if any(problem.startswith(f"round-{round_number}/") for problem in saved.problems):
        folder = saved.folder.as_posix()
        raise SketchLoopError(f"Round {round_number} fails integrity checks. Run sketchloop show {folder} for details.")

    return next(record for record in saved.rounds if record.number == round_number)


def make_rerun_generator(record: RoundRecord, console: Console) -> Generator:
    """
    Recreate the backend a round used, with its recorded diffusers mode.

    Args:
        record (RoundRecord): Completed round to rerun.
        console (Console): Console used for output.

    Returns:
        Generator: The recreated backend.
    """
    adapter = record.iteration.result.backend.adapter
    if adapter not in RERUNNABLE_BACKENDS:
        raise SketchLoopError(f"Round {record.number} used backend {adapter}, which rerun cannot recreate.")

    # Pass the recorded mode, since the diffusers default depends on the device.
    recorded_mode = record.iteration.result.effective.controls.get("mode")
    mode = GenerationMode(recorded_mode) if recorded_mode in list(GenerationMode) else None
    return make_generator(RERUNNABLE_BACKENDS[adapter], mode, console)


def rerun_round(args: argparse.Namespace, console: Console) -> None:
    """
    Rerun a saved round as a new session linked to it, then compare the candidates and conditions.

    Args:
        args (argparse.Namespace): Parsed rerun options.
        console (Console): Console used for output.
    """
    original = load_session(args.session_folder)
    record = find_rerunnable_round(original, args.round_number)
    generator = make_rerun_generator(record, console)

    # Read the stored sketches, reusing the processed one so generation gets the same input bytes.
    round_folder = original.folder / f"round-{record.number}"
    raw_payload = (round_folder / record.raw_sketch.path).read_bytes()
    sketch = record.request.sketch
    reuse = None
    if not record.use_raw:
        reuse = PreprocessedSketch(image=sketch, payload=(round_folder / sketch.path).read_bytes(), steps=record.steps)

    # Run one round in a new linked session with the recorded settings, without asking for a selection.
    request = rebuild_request(record, generator.capabilities)
    link = RerunLink(session_id=original.session.session_id, round_number=record.number, iteration_id=record.iteration.id)
    session = SketchSession(generator, args.runs_dir, rerun_of=link)
    outcome = session.run_round(record.raw_sketch, raw_payload, request.guidance, rotation=record.rotation,
                                use_raw=record.use_raw, n_candidates=request.n_candidates, seed=request.seed, reuse=reuse)
    show_round(outcome, console)
    console.print(f"Rerun session folder: {session.folder.as_posix()}", markup=False, soft_wrap=True)
    print_comparison(compare_rounds(original, record.number, load_session(session.folder)), console)


def main(argv: list[str] | None = None) -> int:
    """
    Run a sketchloop subcommand from the command line and report errors in one line.

    Args:
        argv (list[str] | None, optional): Command-line arguments. Defaults to the process arguments.

    Returns:
        int: Process exit code, 0 on success and 1 on a reported error.
    """
    args = parse_args(argv)
    commands = {Command.RUN: run_session, Command.SHOW: show_session, Command.RERUN: rerun_round}
    try:
        commands[args.command](args, Console())
    except (SketchLoopError, OSError) as error:
        Console(stderr=True).print(f"sketchloop error: {error}", style="red", markup=False, highlight=False, soft_wrap=True)
        return 1

    return 0
