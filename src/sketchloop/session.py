import time
import uuid

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from sketchloop.domain import (SCHEMA_VERSION, GenerationRequest, Guidance, ImageRef, InvalidSelectionError, Iteration,
                               SelectionEvent)
from sketchloop.evaluation import Evaluator, NoOpEvaluator
from sketchloop.experiments import (ROUND_FILE, SESSION_FILE, FailureStage, RerunLink, RoundRecord, SessionRecord,
                                    collect_environment, describe_failure, round_to_dict, session_to_dict, write_json_atomic)
from sketchloop.generation import Generator
from sketchloop.preprocessing import PreprocessedSketch, PreprocessingStep, preprocess_sketch


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundOutcome:
    """
    What one round produced: its folder, sketches, preprocessing steps, iteration, timings, and scores.
    """
    number: int  # Round number, starting at 1
    folder: Path  # Round folder holding the sketches and candidates
    raw_sketch: ImageRef  # Raw sketch as loaded or captured
    steps: tuple[PreprocessingStep, ...]  # Preprocessing steps applied, empty with --raw
    iteration: Iteration  # Request, result, and link to the previous round
    timings_seconds: Mapping[str, float]  # Duration by stage, where generation includes a first model load
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

    def __init__(self, generator: Generator, runs_dir: Path, evaluator: Evaluator | None = None, *,
                 rerun_of: RerunLink | None = None) -> None:
        self.generator = generator
        self.evaluator = evaluator or NoOpEvaluator()
        self.folder = runs_dir / make_session_folder_name()
        self.rerun_of = rerun_of
        self.rounds: list[RoundOutcome] = []
        self.selections: dict[str, SelectionEvent] = {}
        self.session_record: SessionRecord | None = None
        self._started_rounds = 0
        self._latest_record: RoundRecord | None = None

    def _write_session_record(self) -> None:
        # Write session.json once, before the first round's files.
        if self.session_record is not None:
            return

        self.session_record = SessionRecord(schema_version=SCHEMA_VERSION, session_id=self.folder.name,
                                            created_at=datetime.now(timezone.utc), environment=collect_environment(),
                                            rerun_of=self.rerun_of)
        self.folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.folder / SESSION_FILE, session_to_dict(self.session_record))

    def _prepare_sketch(self, raw_sketch: ImageRef, raw_payload: bytes, rotation: int, use_raw: bool,
                        reuse: PreprocessedSketch | None, timings: dict[str, float]) -> tuple[ImageRef, bytes, tuple[PreprocessingStep, ...]]:  # noqa: E501
        # Reuse a stored processed sketch for a rerun, or use the raw sketch when preprocessing is skipped.
        if reuse is not None:
            return reuse.image, reuse.payload, reuse.steps

        if use_raw:
            return raw_sketch, raw_payload, ()

        # Preprocess and time it.
        started = time.perf_counter()
        processed = preprocess_sketch(raw_payload, rotation)
        timings["preprocessing"] = time.perf_counter() - started
        return processed.image, processed.payload, processed.steps

    def run_round(self, raw_sketch: ImageRef, raw_payload: bytes, guidance: Guidance, *, rotation: int = 0,
                  use_raw: bool = False, n_candidates: int = 1, seed: int | None = None,
                  reuse: PreprocessedSketch | None = None) -> RoundOutcome:
        """
        Preprocess the sketch, generate and score candidates, save the files and record, and link to the previous round.

        Args:
            raw_sketch (ImageRef): Reference to the raw sketch as loaded or captured.
            raw_payload (bytes): Encoded raw sketch bytes.
            guidance (Guidance): Prompt and controls for generation.
            rotation (int, optional): Degrees to turn the sketch clockwise before processing. Defaults to 0.
            use_raw (bool, optional): Generate from the raw sketch and skip preprocessing. Defaults to False.
            n_candidates (int, optional): Number of candidates to generate. Defaults to 1.
            seed (int | None, optional): Base seed, or None for the backend default. Defaults to None.
            reuse (PreprocessedSketch | None, optional): Stored processed sketch to generate from, skipping preprocessing.
                Defaults to None.

        Returns:
            RoundOutcome: The round's folder, iteration, timings, and scores.
        """
        # Number the round and fix the fields its record has whether it completes or fails.
        self._write_session_record()
        self._started_rounds += 1
        number = self._started_rounds
        folder = self.folder / f"round-{number}"
        draft = partial(RoundRecord, schema_version=SCHEMA_VERSION, number=number, started_at=datetime.now(timezone.utc),
                        raw_sketch=raw_sketch, use_raw=use_raw, rotation=rotation)
        timings: dict[str, float] = {}
        stage: FailureStage = "preprocessing"
        steps: tuple[PreprocessingStep, ...] = ()
        request = None

        # Run each stage, tracking the active one so a failure record names it.
        try:
            sketch, sketch_payload, steps = self._prepare_sketch(raw_sketch, raw_payload, rotation, use_raw, reuse, timings)

            # Generate and time the call, which includes the model load in the first round, then link to the previous round.
            stage = "generation"
            request = GenerationRequest(sketch=sketch, guidance=guidance, n_candidates=n_candidates, seed=seed)
            started = time.perf_counter()
            output = self.generator.generate(request, sketch_payload)
            timings["generation"] = time.perf_counter() - started
            parent_id = self.rounds[-1].iteration.id if self.rounds else None
            iteration = Iteration(parent_id=parent_id, request=request, result=output.result)

            # Score separately from generation and selection.
            stage = "evaluation"
            started = time.perf_counter()
            scores = self.evaluator.evaluate(output.result.candidates, output.payloads, request)
            timings["evaluation"] = time.perf_counter() - started
        except Exception as error:
            # Keep the sketches made so far and a failed record, so the attempt stays visible, then let the caller report it.
            write_files(folder, {raw_sketch.path: raw_payload} | ({request.sketch.path: sketch_payload} if request else {}))
            failed = draft(status="failed", steps=steps, request=request, iteration=None, timings_seconds=timings,
                           evaluator=None, scores={}, selection=None, failure=describe_failure(stage, error))
            write_json_atomic(folder / ROUND_FILE, round_to_dict(failed))
            raise

        # Store the raw and processed sketches and the candidates, then the record without a selection.
        write_files(folder, {raw_sketch.path: raw_payload, sketch.path: sketch_payload} | dict(output.payloads))
        self._latest_record = draft(status="complete", steps=steps, request=request, iteration=iteration, timings_seconds=timings,
                                    evaluator=type(self.evaluator).__name__, scores=scores, selection=None, failure=None)
        write_json_atomic(folder / ROUND_FILE, round_to_dict(self._latest_record))
        outcome = RoundOutcome(number=number, folder=folder, raw_sketch=raw_sketch, steps=steps, iteration=iteration,
                               timings_seconds=timings, scores=scores)
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

        # Replace the round's record once to add the selection.
        self.selections[latest_id] = selection
        self._latest_record = replace(self._latest_record, selection=selection)
        write_json_atomic(self.rounds[-1].folder / ROUND_FILE, round_to_dict(self._latest_record))
