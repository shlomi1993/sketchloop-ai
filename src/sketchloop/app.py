import numpy as np
import time

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sketchloop.capture import CapturedSketch, InvalidSketchError, encode_frame, load_sketch_file
from sketchloop.domain import (
    Guidance, InvalidSelectionError, Iteration, Rotation, SelectionEvent, SketchLoopError, SketchSource, Unavailable,
    record_no_selection, select_candidates)
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import (SETUP_NOT_MEASURED_REASON, SETUP_REUSED_REASON, ExperimentStore, RoundRecord, RoundStage,
                                    RoundStatus)
from sketchloop.generation import Generator
from sketchloop.preprocessing import Preprocessor
from sketchloop.session import RoundOutcome, SketchSession


NO_SKETCH_MESSAGE: Final = "No sketch yet. Take a webcam picture or upload a PNG or JPEG, then generate."
NO_FILES_MESSAGE: Final = "The browser interface shows saved image files, so it needs a store that keeps files in a folder."


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateView:
    """
    One candidate as the page shows it: its ID, caption, and stored image.
    """
    candidate_id: str  # Candidate ID sent back when the person selects it
    label: str  # Caption such as "2", "2 (fake backend)", or "2, score 0.812"
    image_path: Path  # Stored candidate image under the round folder


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundView:
    """
    A completed round as the page shows it, including the iteration ID a selection must name.
    """
    number: int  # Round number in the current session
    iteration_id: str  # Iteration ID a selection must name, so a stale page cannot record a choice for another round
    sketch_path: Path  # Sketch the generator received, processed or raw
    candidates: tuple[CandidateView, ...]  # Candidates in display order
    is_fake: bool  # Whether the fake backend produced the images
    generation_seconds: float  # Generation time, without a first model load, which is recorded separately
    total_seconds: float  # Capture plus the round's wall time until its files were saved
    selection: str | None  # "1, 3" or "none" once recorded, None while not recorded


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryRow:
    """
    One row of the session history: round, status, prompt, and outcome.
    """
    number: int  # Round number
    status: RoundStatus  # complete or failed
    prompt: str | None  # Requested prompt, None when preprocessing failed before a request existed
    outcome: str  # "selected 1, 3", "selected none", "not selected", or "failed during <stage>: <message>"


def describe_selection(iteration: Iteration, selection: SelectionEvent) -> str:
    """
    Turn a selection into the candidate numbers shown in the table, such as "1, 3", or "none" when nothing was picked.
    """
    selected_ids = selection.selected_candidate_ids
    chosen = [str(candidate.index + 1) for candidate in iteration.result.candidates if candidate.id in selected_ids]
    return ", ".join(chosen) or "none"


def label_candidate(number: int, is_fake: bool, score: float | None) -> str:
    """
    Caption a candidate with its 1-based number, a fake-backend note, and its score when there is one.
    """
    fake_note = " (fake backend)" if is_fake else ""
    score_note = "" if score is None else f", score {score:.3f}"
    return f"{number}{fake_note}{score_note}"


def build_round_view(outcome: RoundOutcome, selections: Mapping[str, SelectionEvent]) -> RoundView:
    """
    Turn a completed round into the view the page shows, with captions and the recorded selection.

    Args:
        outcome (RoundOutcome): The completed round.
        selections (Mapping[str, SelectionEvent]): Recorded selections by iteration ID.

    Returns:
        RoundView: The round as the page shows it.
    """
    iteration = outcome.iteration
    is_fake = iteration.result.backend.is_fake
    candidates = tuple(CandidateView(candidate_id=candidate.id, image_path=outcome.folder / candidate.image.path,
                                     label=label_candidate(candidate.index + 1, is_fake, outcome.scores.get(candidate.id)))
                       for candidate in iteration.result.candidates)
    selection = selections.get(iteration.id)
    return RoundView(number=outcome.number, iteration_id=iteration.id, sketch_path=outcome.folder / iteration.request.sketch.path,
                     candidates=candidates, is_fake=is_fake, generation_seconds=outcome.timings_seconds[RoundStage.GENERATION],
                     total_seconds=outcome.total_seconds,
                     selection=None if selection is None else describe_selection(iteration, selection))


def build_history_row(record: RoundRecord) -> HistoryRow:
    """
    Summarize a saved round for the history table, as its selection or its failure.

    Args:
        record (RoundRecord): Saved round, complete or failed.

    Returns:
        HistoryRow: The round's history row.
    """
    # Describe a failure by its stage and message, and a completed round by its selection.
    if record.failure is not None:
        outcome = f"failed during {record.failure.stage}: {record.failure.message}"
    elif record.selection is not None:
        outcome = f"selected {describe_selection(record.iteration, record.selection)}"
    else:
        outcome = "not selected"

    prompt = record.request.guidance.prompt if record.request is not None else None
    return HistoryRow(number=record.number, status=record.status, prompt=prompt, outcome=outcome)


def capture_browser_sketch(sketch: Path | np.ndarray) -> CapturedSketch:
    """
    Load an uploaded file, or encode a camera frame in OpenCV channel order, and time it as the capture stage.

    Args:
        sketch (Path | np.ndarray): Uploaded sketch file, or a webcam frame from the browser.

    Returns:
        CapturedSketch: The raw sketch, its bytes, and how long loading or encoding took.
    """
    started = time.perf_counter()
    image, payload = encode_frame(sketch) if isinstance(sketch, np.ndarray) else load_sketch_file(sketch)
    return CapturedSketch(image=image, payload=payload, capture_seconds=time.perf_counter() - started)


class AppController:
    """
    Framework-free service the browser UI calls: one loaded generator and the current session, shared by all tabs.
    """
    def __init__(self, generator: Generator, store: ExperimentStore, evaluator: Evaluator | None = None, *,
                 preprocessor: Preprocessor | None = None,
                 generator_setup_seconds: float | Unavailable = Unavailable(SETUP_NOT_MEASURED_REASON)) -> None:
        # The page shows candidates from their saved files, so refuse a store that keeps none.
        if store.root is None:
            raise SketchLoopError(NO_FILES_MESSAGE)

        self.generator = generator
        self.store = store
        self.evaluator = evaluator
        self.preprocessor = preprocessor
        self.session = SketchSession(generator, store, evaluator, preprocessor=preprocessor,
                                     generator_setup_seconds=generator_setup_seconds)

    @property
    def max_candidates(self) -> int:
        """
        Return the most candidates the backend can make in one round.
        """
        return self.generator.capabilities.max_candidates

    def describe_backend(self) -> str:
        """
        Return the backend's one-sentence description for the page header.
        """
        return self.generator.describe()

    def _check_round_inputs(self, sketch: Path | np.ndarray | None, source: SketchSource, prompt: str, rotation: int,
                            use_raw: bool, frame_age_seconds: float | None) -> None:
        # Refuse a missing sketch, a blank prompt, rotation settings that preprocessing cannot apply, and a misplaced age.
        if sketch is None:
            raise InvalidSketchError(NO_SKETCH_MESSAGE)

        if not prompt.strip():
            raise SketchLoopError("Enter a prompt, then generate.")

        if rotation not in list(Rotation):
            raise SketchLoopError(f"Rotation must be one of 0, 90, 180, or 270, got {rotation}.")

        if rotation and use_raw:
            raise SketchLoopError("Rotation cannot be combined with the raw sketch option, which skips preprocessing.")

        if frame_age_seconds is not None and source != SketchSource.WEBCAM:
            raise SketchLoopError(f"Only a streamed webcam frame has a frame age, got source {source}. Pass no frame age.")

    def run_round(self, sketch: Path | np.ndarray | None, source: SketchSource, prompt: str, *, n_candidates: int,
                  seed: int | None = None, rotation: int = Rotation.DEG_000, use_raw: bool = False,
                  frame_age_seconds: float | None = None) -> RoundView:
        """
        Check the inputs, then run a round from a sketch file or a camera frame and return its view.

        Args:
            sketch (Path | np.ndarray | None): Sketch file, camera frame in OpenCV channel order, or None when missing.
            source (SketchSource): Where the sketch came from, recorded with the round.
            prompt (str): Style prompt, recorded as typed.
            n_candidates (int): Number of candidates to generate.
            seed (int | None, optional): Base seed, or None for the backend default. Defaults to None.
            rotation (int, optional): Degrees to turn the sketch clockwise before processing. Defaults to 0.
            use_raw (bool, optional): Generate from the raw sketch and skip preprocessing. Defaults to False.
            frame_age_seconds (float | None, optional): Age of a streamed webcam frame when the round was taken, only for
                the webcam source. Defaults to None.

        Returns:
            RoundView: The completed round.
        """
        # Check the inputs before reading any file, so a bad request writes nothing.
        self._check_round_inputs(sketch, source, prompt, rotation, use_raw, frame_age_seconds)

        # Load and time a sketch file or a camera frame, keeping the prompt as typed.
        captured = capture_browser_sketch(sketch)
        outcome = self.session.run_round(captured.image, captured.payload, Guidance(prompt=prompt), source=source,
                                         rotation=rotation, use_raw=use_raw, n_candidates=n_candidates, seed=seed,
                                         capture_seconds=captured.capture_seconds, frame_age_seconds=frame_age_seconds)
        return build_round_view(outcome, self.session.selections)

    def record_selection(self, iteration_id: str | None, candidate_ids: Sequence[str]) -> RoundView:
        """
        Record the person's choice for the round the page showed, refusing a stale page or a second choice.

        Args:
            iteration_id (str | None): Iteration ID of the round the page showed, or None when it showed none.
            candidate_ids (Sequence[str]): Chosen candidate IDs, empty for an explicit choice of none.

        Returns:
            RoundView: The round with its recorded selection.
        """
        if not self.session.rounds:
            raise InvalidSelectionError("No round to select from yet. Generate first.")

        # Accept a selection only for the exact round the page showed, even across sessions, and only once.
        latest = self.session.rounds[-1]
        if iteration_id != latest.iteration.id:
            raise InvalidSelectionError(f"This page shows an older round. Reload the page to see round {latest.number}.")

        if latest.iteration.id in self.session.selections:
            raise InvalidSelectionError(f"Round {latest.number} already has a recorded selection. Generate the next round.")

        # Nothing ticked is an explicit choice of none.
        chosen_ids = list(candidate_ids)
        selection = select_candidates(latest.iteration, chosen_ids) if chosen_ids else record_no_selection(latest.iteration)
        self.session.record_selection(selection)
        return build_round_view(latest, self.session.selections)

    def latest_round(self) -> RoundView | None:
        """
        Return the latest completed round's view, or None before the first, so a reload shows it again.
        """
        return build_round_view(self.session.rounds[-1], self.session.selections) if self.session.rounds else None

    def list_history(self) -> tuple[HistoryRow, ...]:
        """
        List the current session's rounds, including failed ones, in order.
        """
        return tuple(build_history_row(record) for record in self.session.records)

    def start_new_session(self) -> None:
        """
        Start a fresh session folder with the same loaded generator, store, evaluator, and preprocessor.
        """
        # The generator is reused, so this session paid no setup, and repeating the first value would count it twice.
        self.session = SketchSession(self.generator, self.store, self.evaluator, preprocessor=self.preprocessor,
                                     generator_setup_seconds=Unavailable(SETUP_REUSED_REASON))
