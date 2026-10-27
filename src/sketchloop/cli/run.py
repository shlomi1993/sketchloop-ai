import argparse

from enum import StrEnum
from pathlib import Path
from rich.console import Console

from sketchloop.app import describe_selection
from sketchloop.capture import CapturedSketch, CaptureSource, FileCapture
from sketchloop.cli.arguments import Command
from sketchloop.cli.common import make_configured_generator, require_folder_store, show_round
from sketchloop.components import make_capture_source, make_evaluator, make_preprocessor
from sketchloop.domain import (Guidance, Iteration, SelectionEvent, SketchLoopError, SketchSource, record_no_selection,
                               select_candidates)
from sketchloop.experiments import StoreError
from sketchloop.session import SketchSession
from sketchloop.settings import Settings


class NextStep(StrEnum):
    """
    Menu choices after a round: what to change before the next one, or quit.
    """
    REPEAT = ""
    PROMPT = "p"
    RECAPTURE = "c"
    FILE = "f"
    QUIT = "q"


class TerminalSession:
    """
    The interactive run loop: one loaded session, its capture source, and the current prompt and sketch.
    """

    def __init__(self, session: SketchSession, capture_source: CaptureSource, console: Console, *, settings: Settings,
                 prompt: str, sketch: CapturedSketch) -> None:
        """
        Hold the session, where sketches come from, the settings each round uses, and the first prompt and sketch.

        Args:
            session (SketchSession): Session that runs and saves the rounds.
            capture_source (CaptureSource): Source of the first sketch, used again to recapture from a camera.
            console (Console): Console used for output and prompts.
            settings (Settings): Effective settings, applied to every round.
            prompt (str): First prompt.
            sketch (CapturedSketch): First raw sketch, its bytes, and its capture time.
        """
        self.session = session
        self.capture_source = capture_source
        self.console = console
        self.settings = settings
        self.prompt = prompt
        self.sketch = sketch
        self.source: SketchSource = capture_source.kind

        # Only the first round after a capture records its time, since repeating a round reuses the sketch.
        self.capture_seconds: float | None = sketch.capture_seconds

    def read_answer(self, question: str) -> str | None:
        """
        Read one stripped line of input, or None at the end of input.
        """
        try:
            return self.console.input(question).strip()
        except EOFError:
            return None

    def ask_selection(self, iteration: Iteration) -> SelectionEvent:
        """
        Ask the person which candidates they choose and record it explicitly.

        Args:
            iteration (Iteration): Iteration whose candidates were listed.

        Returns:
            SelectionEvent: The recorded selection, possibly of none.
        """
        candidates = iteration.result.candidates
        while True:
            # Read space-separated 1-based numbers, where an empty answer is an explicit choice of none.
            try:
                answer = self.console.input("[bold]Pick candidates by number (space separated), or press Enter for none:[/bold] ").split()  # noqa: E501
            except EOFError:
                raise SketchLoopError("No selection entered. Rerun and type numbers or press Enter.") from None

            if not answer:
                return record_no_selection(iteration)

            # Accept only distinct numbers from the shown list, and ask again otherwise.
            numbers = [int(number) for number in answer if number.isdecimal()]
            all_numbers_valid = len(numbers) == len(answer) and all(1 <= number <= len(candidates) for number in numbers)
            if all_numbers_valid and len(set(numbers)) == len(numbers):
                return select_candidates(iteration, [candidates[number - 1].id for number in numbers])

            self.console.print(f"Use distinct numbers from 1 to {len(candidates)}, or press Enter for none.", style="yellow")

    def ask_next_step(self) -> tuple[NextStep, str] | None:
        """
        Ask what to change for the next round, re-asking with a hint on invalid input.

        Returns:
            tuple[NextStep, str] | None: The change and its value (a prompt or file path), or None to quit.
        """
        can_recapture = self.capture_source.kind == SketchSource.CAMERA
        recapture = "c = recapture, " if can_recapture else ""
        choices = f"Enter = again, p = new prompt, {recapture}f <path> = new file, q = quit"
        while True:
            # Quit on end of input or q.
            answer = self.read_answer(f"[bold]Next ({choices}):[/bold] ")
            if answer is None or answer == NextStep.QUIT:
                return None

            # Repeat, recapture, or load a file, when that choice is available.
            command, _, value = answer.partition(" ")
            is_available = answer == NextStep.REPEAT or (answer == NextStep.RECAPTURE and can_recapture)
            if is_available or (command == NextStep.FILE and value.strip()):
                return NextStep(command), value.strip()

            # Ask for the new prompt, quitting at the end of input and re-asking the menu when it is empty.
            if answer == NextStep.PROMPT:
                prompt = self.read_answer("[bold]New prompt:[/bold] ")
                if prompt is None:
                    return None

                if prompt:
                    return NextStep.PROMPT, prompt

            self.console.print(f"Not a choice. Use: {choices}.")

    def apply_next_change(self) -> bool:
        """
        Ask for the next change until one applies, so a wrong file path or a cancelled capture asks again.

        Returns:
            bool: True when the next round should run, False when the person quits.
        """
        while True:
            next_step = self.ask_next_step()
            if next_step is None:
                return False

            # Load or capture a new sketch, and ask again when that fails.
            step, value = next_step
            try:
                if step == NextStep.RECAPTURE:
                    self.sketch, self.source = self.capture_source.capture(), self.capture_source.kind
                elif step == NextStep.FILE:
                    self.sketch, self.source = FileCapture(Path(value)).capture(), SketchSource.FILE
            except (SketchLoopError, OSError) as error:
                self.console.print(f"{error} Choose again.", style="red", markup=False, soft_wrap=True)
                continue

            # A new sketch's capture time belongs to the next round.
            if step in (NextStep.RECAPTURE, NextStep.FILE):
                self.capture_seconds = self.sketch.capture_seconds

            # A new prompt replaces the current one, and repeating changes nothing.
            if step == NextStep.PROMPT:
                self.prompt = value

            return True

    def run_round(self) -> None:
        """
        Run one round with the current sketch and prompt, show it, and record the person's choice.
        """
        # Run the round, reporting a failure instead of ending the session, and count the capture time only once.
        preprocessing, run_settings = self.settings.preprocessing, self.settings.run
        capture_seconds, self.capture_seconds = self.capture_seconds, None
        try:
            outcome = self.session.run_round(self.sketch.image, self.sketch.payload, Guidance(prompt=self.prompt),
                                             source=self.source, rotation=preprocessing.rotation, use_raw=preprocessing.raw,
                                             n_candidates=run_settings.candidates, seed=run_settings.seed,
                                             capture_seconds=capture_seconds)
        except (SketchLoopError, OSError) as error:
            self.console.print(f"Round failed: {error}", style="red", markup=False, soft_wrap=True)
            return

        # Show the candidates and record the person's explicit choice.
        show_round(outcome, self.console)
        selection = self.ask_selection(outcome.iteration)

        # Report a selection the store could not save like a failed round, so the session continues.
        try:
            self.session.record_selection(selection)
        except StoreError as error:
            self.console.print(f"Selection not saved: {error}", style="red", markup=False, soft_wrap=True)
        else:
            self.console.print(f"Selected: {describe_selection(outcome.iteration, selection)}", style="green")

    def print_summary(self) -> None:
        """
        Print the session folder, the number of rounds, and each round's selection.
        """
        self.console.rule("Session summary")
        self.console.print(f"Session folder: {self.session.folder.as_posix()}", markup=False, soft_wrap=True)
        self.console.print(f"Rounds: {len(self.session.rounds)}")
        for outcome in self.session.rounds:
            selection = self.session.selections.get(outcome.iteration.id)
            choice = describe_selection(outcome.iteration, selection) if selection else "not saved"
            self.console.print(f"Round {outcome.number} selected: {choice}")

    def run(self) -> None:
        """
        Run rounds until the person quits, applying the requested change between rounds, then print a summary.
        """
        # Run a round, then stop when the person quits.
        while True:
            self.run_round()
            if not self.apply_next_change():
                break

        self.print_summary()


def run_session(args: argparse.Namespace, console: Console) -> None:
    """
    Run manual rounds until the person quits: generate, show candidates, record the selection, and apply the next change.

    Args:
        args (argparse.Namespace): Parsed command-line options, applied to every round.
        console (Console): Console used for output and prompts.
    """
    # Load or capture the first sketch, then create the backend once so a real model stays loaded between rounds.
    settings = args.settings
    store = require_folder_store(settings.storage, Command.RUN)
    capture_source = make_capture_source(settings.capture, args.sketch)
    sketch = capture_source.capture()
    generator, setup_seconds = make_configured_generator(settings.generator, console)
    session = SketchSession(generator, store, make_evaluator(settings.evaluator),
                            preprocessor=make_preprocessor(settings.preprocessing), generator_setup_seconds=setup_seconds)
    TerminalSession(session, capture_source, console, settings=settings, prompt=args.prompt, sketch=sketch).run()
