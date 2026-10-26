import time
import uuid

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from sketchloop.domain import (SCHEMA_VERSION, GenerationRequest, Guidance, ImageRef, InvalidSelectionError, Iteration,
                               Rotation, SelectionEvent, SketchSource, Unavailable)
from sketchloop.evaluation import Evaluator, NoOpEvaluator
from sketchloop.experiments import (ExperimentStore, RerunLink, RoundRecord, RoundStage, RoundStatus, SessionRecord,
                                    StoreError, collect_environment, describe_failure, require_known_source)
from sketchloop.generation import Generator
from sketchloop.preprocessing import PreprocessedSketch, PreprocessingStep, Preprocessor, StrokePreprocessor


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundOutcome:
    """
    What one round produced: its folder, sketches, preprocessing steps, iteration, timings, and scores.
    """
    number: int  # Round number, starting at 1
    folder: Path | None  # Round folder holding the sketches and candidates, None for a store without files
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


class SketchSession:
    """
    Runs successive manual rounds with one generator, linking each round to the previous one.
    """

    def __init__(self, generator: Generator, store: ExperimentStore, evaluator: Evaluator | None = None, *,
                 preprocessor: Preprocessor | None = None, rerun_of: RerunLink | None = None) -> None:
        self.generator = generator
        self.store = store
        self.evaluator = evaluator or NoOpEvaluator()
        self.preprocessor = preprocessor or StrokePreprocessor()
        self.session_id = make_session_folder_name()
        self.folder = store.locate_session(self.session_id)
        self.rerun_of = rerun_of
        self.rounds: list[RoundOutcome] = []
        self.selections: dict[str, SelectionEvent] = {}
        self.session_record: SessionRecord | None = None
        self.records: list[RoundRecord] = []
        self._started_rounds = 0

    def _write_session_record(self) -> None:
        # Save the session record once, before the first round's files.
        if self.session_record is not None:
            return

        record = SessionRecord(schema_version=SCHEMA_VERSION, session_id=self.session_id, created_at=datetime.now(timezone.utc),
                               environment=collect_environment(), rerun_of=self.rerun_of)

        # Remember the record only once it is saved, so a failed save is retried by the next round.
        self.store.save_session(record)
        self.session_record = record

    def _prepare_sketch(self, raw_sketch: ImageRef, raw_payload: bytes, rotation: int, use_raw: bool,
                        reuse: PreprocessedSketch | None, timings: dict[str, float]) -> tuple[ImageRef, bytes, tuple[PreprocessingStep, ...]]:  # noqa: E501
        # Reuse a stored processed sketch for a rerun, or use the raw sketch when preprocessing is skipped.
        if reuse is not None:
            return reuse.image, reuse.payload, reuse.steps

        if use_raw:
            return raw_sketch, raw_payload, ()

        # Preprocess and time it.
        started = time.perf_counter()
        processed = self.preprocessor.preprocess(raw_payload, rotation)
        timings[RoundStage.PREPROCESSING] = time.perf_counter() - started
        return processed.image, processed.payload, processed.steps

    def run_round(self, raw_sketch: ImageRef, raw_payload: bytes, guidance: Guidance, *, source: SketchSource | Unavailable,
                  rotation: int = Rotation.DEG_000, use_raw: bool = False, n_candidates: int = 1, seed: int | None = None,
                  reuse: PreprocessedSketch | None = None) -> RoundOutcome:
        """
        Preprocess the sketch, generate and score candidates, save the files and record, and link to the previous round.

        Args:
            raw_sketch (ImageRef): Reference to the raw sketch as loaded or captured.
            raw_payload (bytes): Encoded raw sketch bytes.
            guidance (Guidance): Prompt and controls for generation.
            source (SketchSource | Unavailable): Where the raw sketch bytes came from, such as file or webcam.
            rotation (int, optional): Degrees to turn the sketch clockwise before processing. Defaults to 0.
            use_raw (bool, optional): Generate from the raw sketch and skip preprocessing. Defaults to False.
            n_candidates (int, optional): Number of candidates to generate. Defaults to 1.
            seed (int | None, optional): Base seed, or None for the backend default. Defaults to None.
            reuse (PreprocessedSketch | None, optional): Stored processed sketch to generate from, skipping preprocessing.
                Defaults to None.

        Returns:
            RoundOutcome: The round's folder, iteration, timings, and scores.
        """
        # Refuse an unknown source before writing anything, since even the failure record could not hold it.
        require_known_source(source)

        # Number the round and fix the fields its record has whether it completes or fails.
        self._write_session_record()
        self._started_rounds += 1
        number = self._started_rounds
        draft = partial(RoundRecord, schema_version=SCHEMA_VERSION, number=number, started_at=datetime.now(timezone.utc),
                        raw_sketch=raw_sketch, source=source, use_raw=use_raw, rotation=rotation)
        timings: dict[str, float] = {}
        stage = RoundStage.PREPROCESSING
        steps: tuple[PreprocessingStep, ...] = ()
        request = None

        # Run each stage, tracking the active one so a failure record names it.
        try:
            sketch, sketch_payload, steps = self._prepare_sketch(raw_sketch, raw_payload, rotation, use_raw, reuse, timings)

            # Generate and time the call, which includes the model load in the first round, then link to the previous round.
            stage = RoundStage.GENERATION
            request = GenerationRequest(sketch=sketch, guidance=guidance, n_candidates=n_candidates, seed=seed)
            started = time.perf_counter()
            output = self.generator.generate(request, sketch_payload)
            timings[RoundStage.GENERATION] = time.perf_counter() - started
            parent_id = self.rounds[-1].iteration.id if self.rounds else None
            iteration = Iteration(parent_id=parent_id, request=request, result=output.result)

            # Score separately from generation and selection, and check the scores by building the record.
            stage = RoundStage.EVALUATION
            started = time.perf_counter()
            scores = self.evaluator.evaluate(output.result.candidates, output.payloads, request)
            timings[RoundStage.EVALUATION] = time.perf_counter() - started
            record = draft(status=RoundStatus.COMPLETE, steps=steps, request=request, iteration=iteration,
                           timings_seconds=timings, evaluator=type(self.evaluator).__name__, scores=scores, selection=None,
                           failure=None)

            # Store the raw and processed sketches and the candidates, then the record without a selection.
            stage = RoundStage.SAVING
            files = {raw_sketch.path: raw_payload, sketch.path: sketch_payload} | dict(output.payloads)
            self.store.save_round(self.session_id, record, files)
        except Exception as error:
            # Keep a failed record in the history even when it cannot be saved, so the attempt stays visible.
            sketches = {raw_sketch.path: raw_payload} | ({request.sketch.path: sketch_payload} if request else {})
            failed = draft(status=RoundStatus.FAILED, steps=steps, request=request, iteration=None, timings_seconds=timings,
                           evaluator=None, scores={}, selection=None, failure=describe_failure(stage, error))
            self.records.append(failed)

            # Save the failed record and the sketches made so far, keeping the stage's own error if that save fails too.
            try:
                self.store.save_round(self.session_id, failed, sketches)
            except StoreError as store_error:
                error.add_note(f"The failed round was not saved either: {store_error}")
            raise

        self.records.append(record)
        folder = self.store.locate_round(self.session_id, number)
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

        # Save the round's record with the selection first, so a failed save leaves the choice open for a retry.
        position = [record.iteration.id if record.iteration else None for record in self.records].index(latest_id)
        selected_record = replace(self.records[position], selection=selection)
        self.store.save_round(self.session_id, selected_record, {})
        self.records[position] = selected_record
        self.selections[latest_id] = selection
