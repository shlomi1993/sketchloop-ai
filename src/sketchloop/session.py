import time
import uuid

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sketchloop.domain import GenerationRequest, Guidance, ImageRef, InvalidSelectionError, Iteration, SelectionEvent
from sketchloop.evaluation import Evaluator, NoOpEvaluator
from sketchloop.generation import Generator
from sketchloop.preprocessing import PreprocessingStep, preprocess_sketch


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundOutcome:
    """
    What one round produced: its folder, sketches, preprocessing steps, iteration, timing, and scores.
    """
    number: int  # Round number, starting at 1
    folder: Path  # Round folder holding the sketches and candidates
    raw_sketch: ImageRef  # Raw sketch as loaded or captured
    steps: tuple[PreprocessingStep, ...]  # Preprocessing steps applied, empty with --raw
    iteration: Iteration  # Request, result, and link to the previous round
    elapsed_seconds: float  # Generation time, including a first model load
    scores: Mapping[str, float]  # Evaluator scores by candidate ID, empty with the no-op evaluator


def make_session_folder_name() -> str:
    """
    Name a session folder by local date and time plus a short random suffix, so folders sort by time and never clash.
    """
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def write_files(folder: Path, files: Mapping[str, bytes]) -> None:
    """
    Write each payload to its relative path under the folder, creating subfolders as needed.

    Args:
        folder (Path): Folder the relative paths resolve against.
        files (Mapping[str, bytes]): Payload by relative POSIX path.
    """
    # Create each file's subfolder before writing it.
    for path, payload in files.items():
        target_file = folder / path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(payload)


class SketchSession:
    """
    Runs successive manual rounds with one generator, linking each round to the previous one.
    """

    def __init__(self, generator: Generator, runs_dir: Path, evaluator: Evaluator | None = None) -> None:
        self.generator = generator
        self.evaluator = evaluator or NoOpEvaluator()
        self.folder = runs_dir / make_session_folder_name()
        self.rounds: list[RoundOutcome] = []
        self.selections: dict[str, SelectionEvent] = {}

    def run_round(self, raw_sketch: ImageRef, raw_payload: bytes, guidance: Guidance, *, rotation: int = 0,
                  use_raw: bool = False, n_candidates: int = 1, seed: int | None = None) -> RoundOutcome:
        """
        Preprocess the sketch, generate and score candidates, save the files, and link the round to the previous one.

        Args:
            raw_sketch (ImageRef): Reference to the raw sketch as loaded or captured.
            raw_payload (bytes): Encoded raw sketch bytes.
            guidance (Guidance): Prompt and controls for generation.
            rotation (int, optional): Degrees to turn the sketch clockwise before processing. Defaults to 0.
            use_raw (bool, optional): Generate from the raw sketch and skip preprocessing. Defaults to False.
            n_candidates (int, optional): Number of candidates to generate. Defaults to 1.
            seed (int | None, optional): Base seed, or None for the backend default. Defaults to None.

        Returns:
            RoundOutcome: The round's folder, iteration, timing, and scores.
        """
        # Preprocess the raw sketch unless the person asked to generate from it directly.
        sketch, sketch_payload, steps = raw_sketch, raw_payload, ()
        if not use_raw:
            processed = preprocess_sketch(raw_payload, rotation)
            sketch, sketch_payload, steps = processed.image, processed.payload, processed.steps

        # Generate with the session's generator and time the call, which includes the model load in the first round.
        request = GenerationRequest(sketch=sketch, guidance=guidance, n_candidates=n_candidates, seed=seed)
        started = time.perf_counter()
        output = self.generator.generate(request, sketch_payload)
        elapsed_seconds = time.perf_counter() - started

        # Score separately from generation and selection, and link the iteration to the previous round.
        scores = self.evaluator.evaluate(output.result.candidates, output.payloads, request)
        parent_id = self.rounds[-1].iteration.id if self.rounds else None
        iteration = Iteration(parent_id=parent_id, request=request, result=output.result)

        # Store the raw and processed sketches and the candidates in this round's folder.
        number = len(self.rounds) + 1
        folder = self.folder / f"round-{number}"
        write_files(folder, {raw_sketch.path: raw_payload, sketch.path: sketch_payload} | dict(output.payloads))
        outcome = RoundOutcome(number=number, folder=folder, raw_sketch=raw_sketch, steps=steps, iteration=iteration,
                               elapsed_seconds=elapsed_seconds, scores=scores)
        self.rounds.append(outcome)
        return outcome

    def record_selection(self, selection: SelectionEvent) -> None:
        """
        Record the person's choice for the latest round.

        Args:
            selection (SelectionEvent): Selection made from the latest round's candidates.
        """
        # Accept a selection only for the latest round, and only once.
        latest_id = self.rounds[-1].iteration.id if self.rounds else None
        if selection.iteration_id != latest_id or latest_id in self.selections:
            raise InvalidSelectionError("A selection must be for the latest round and recorded once. Run a round first.")

        self.selections[latest_id] = selection
