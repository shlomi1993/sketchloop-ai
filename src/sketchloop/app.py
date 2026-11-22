"""
Framework-free controller and view records that the browser page calls to run and select rounds.
"""

import numpy as np
import time

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sketchloop.capture import CapturedSketch, InvalidSketchError, encode_frame, load_sketch_file
from sketchloop.domain import (
    ControlValue, Guidance, InvalidSelectionError, Iteration, Rotation, SelectionEvent, SketchLoopError, SketchSource,
    Unavailable, record_no_selection, select_candidates)
from sketchloop.drawing import GuidedDrawing
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import (SETUP_NOT_MEASURED_REASON, SETUP_REUSED_REASON, DrawingEnd, ExperimentStore, RoundRecord,
                                    RoundStage, RoundStatus, SuggestionRecord, SuggestionStage, check_experiment_name)
from sketchloop.generation import ControlKind, ControlSpec, Generator
from sketchloop.preprocessing import Preprocessor, make_paper_canvas
from sketchloop.session import RoundOutcome, SketchSession
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggester
from sketchloop.suggestion import PromptSuggester, TemplateSuggester, compute_sketch_signature, has_sketch_changed


NO_SKETCH_MESSAGE: Final = "No sketch yet. Take a webcam picture or upload a PNG or JPEG, then generate."
NO_FILES_MESSAGE: Final = "The browser interface shows saved image files, so it needs a store that keeps files in a folder."
NO_DRAWING_MESSAGE: Final = "No drawing is open. Enter an experiment name and start a drawing first."
NO_FRAME_MESSAGE: Final = "No sketch yet. Start the webcam or upload a PNG or JPEG, then try again."
NO_NEW_STROKES_MESSAGE: Final = "No new strokes to suggest. Keep drawing or press again for another try."
DEFAULT_CANVAS_SIZE: Final = (512, 360)  # Paper frame width and height, A4 or Letter in landscape on the latent grid
DEFAULT_SAVE_INTERVAL_SECONDS: Final = 2.0


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


@dataclass(frozen=True, slots=True, kw_only=True)
class SuggestionView:
    """
    A shown next-stroke suggestion as the page reports it.
    """
    drawing: int  # Drawing number in the current session
    number: int  # Suggestion number in the drawing
    part_count: int  # Parts drawn in the overlay, 0 when nothing new was suggested
    colors: tuple[str, ...]  # Overlay color of each part, by index
    is_fake: bool  # Whether the fake suggester drew the parts
    generation_seconds: float | None  # Seconds the suggester spent generating, None when it reported none
    status: str  # One line such as "Suggestion 2: 3 parts in 3.1 s" or the no-new-strokes message


@dataclass(frozen=True, slots=True, kw_only=True)
class DrawingView:
    """
    A drawing as the page reports it: its experiment, state, counts, and one line per suggestion.
    """
    number: int  # Drawing number in the current session
    experiment: str | None  # Experiment name of the session, None for a time-named session
    is_open: bool  # Whether the drawing still takes captures and suggestions
    capture_count: int  # Captures saved so far
    suggestion_count: int  # Suggestions so far, shown or failed
    suggestion_lines: tuple[str, ...]  # One line per suggestion, such as "Suggestion 1: 2 parts"


def count_parts(count: int) -> str:
    """
    Say a part count in words, such as "1 part" or "3 parts".
    """
    return f"{count} part" if count == 1 else f"{count} parts"


def build_suggestion_view(drawing_number: int, suggestion: SuggestionRecord) -> SuggestionView:
    """
    Turn a shown suggestion into the view the page reports, with a plain message when nothing new was suggested.

    Args:
        drawing_number (int): Number of the drawing the suggestion belongs to.
        suggestion (SuggestionRecord): The shown suggestion.

    Returns:
        SuggestionView: The suggestion as the page reports it.
    """
    result = suggestion.result
    seconds = suggestion.timings_seconds.get(SuggestionStage.GENERATION)
    timing = f" in {seconds:.1f} s" if seconds is not None else ""
    parts = len(result.parts)
    status = f"Suggestion {suggestion.number}: {count_parts(parts)}{timing}" if parts else NO_NEW_STROKES_MESSAGE
    return SuggestionView(drawing=drawing_number, number=suggestion.number, part_count=parts, colors=suggestion.part_colors,
                          is_fake=result.suggester.is_fake, generation_seconds=seconds, status=status)


def describe_suggestion(suggestion: SuggestionRecord) -> str:
    """
    Summarize a suggestion in one line: its part count, no new strokes, or its failure.
    """
    if suggestion.failure is not None:
        return f"Suggestion {suggestion.number}: failed during {suggestion.failure.stage}: {suggestion.failure.message}"

    parts = len(suggestion.result.parts)
    return f"Suggestion {suggestion.number}: {count_parts(parts) if parts else 'no new strokes'}"


def build_drawing_view(drawing: GuidedDrawing, experiment: str | None) -> DrawingView:
    """
    Turn a drawing into the view the page reports.

    Args:
        drawing (GuidedDrawing): The drawing.
        experiment (str | None): Experiment name of its session, or None.

    Returns:
        DrawingView: The drawing as the page reports it.
    """
    record = drawing.record
    return DrawingView(number=record.number, experiment=experiment, is_open=drawing.is_open, capture_count=len(record.captures),
                       suggestion_count=len(record.suggestions),
                       suggestion_lines=tuple(describe_suggestion(suggestion) for suggestion in record.suggestions))


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
                 preprocessor: Preprocessor | None = None, suggester: PromptSuggester | None = None,
                 generator_setup_seconds: float | Unavailable = Unavailable(SETUP_NOT_MEASURED_REASON),
                 stroke_suggester: StrokeSuggester | None = None, canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
                 save_interval_seconds: float = DEFAULT_SAVE_INTERVAL_SECONDS) -> None:
        # The page shows candidates from their saved files, so refuse a store that keeps none.
        if store.root is None:
            raise SketchLoopError(NO_FILES_MESSAGE)

        self.generator = generator
        self.store = store
        self.evaluator = evaluator
        self.preprocessor = preprocessor
        self.suggester = suggester or TemplateSuggester()
        self.stroke_suggester = stroke_suggester or FakeStrokeSuggester()
        self.canvas_size = canvas_size
        self.save_interval_seconds = save_interval_seconds
        self._suggestions: tuple[str, ...] = ()
        self._suggestion_index = 0
        self._suggested_signature: np.ndarray | None = None
        self.session = SketchSession(generator, store, evaluator, preprocessor=preprocessor,
                                     generator_setup_seconds=generator_setup_seconds)

    @property
    def max_candidates(self) -> int:
        """
        Return the most candidates the backend can make in one round.
        """
        return self.generator.capabilities.max_candidates

    @property
    def slider_controls(self) -> tuple[ControlSpec, ...]:
        """
        Return the backend's numeric controls that have both bounds, which the page offers as sliders.
        """
        numeric = (ControlKind.INT, ControlKind.FLOAT)
        controls = self.generator.capabilities.controls
        return tuple(spec for spec in controls if spec.kind in numeric and None not in (spec.minimum, spec.maximum))

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

    def preview_sketch(self, sketch: Path | np.ndarray | None, rotation: int, use_raw: bool) -> bytes | None:
        """
        Return the encoded sketch a round would give the generator now, preprocessed unless raw, without saving anything.

        Args:
            sketch (Path | np.ndarray | None): Sketch file, camera frame in OpenCV channel order, or None when missing.
            rotation (int): Degrees to turn the sketch clockwise before processing.
            use_raw (bool): Show the raw sketch, which skips preprocessing.

        Returns:
            bytes | None: The sketch's encoded bytes, or None when there is no sketch or preprocessing finds no drawing.
        """
        if sketch is None:
            return None

        # Encode or load the sketch as a round would, and preprocess it with the session's preprocessor.
        try:
            _, payload = encode_frame(sketch) if isinstance(sketch, np.ndarray) else load_sketch_file(sketch)
            return payload if use_raw else self.session.preprocessor.preprocess(payload, rotation).payload
        except (SketchLoopError, OSError):
            return None

    def suggest_prompt(self, sketch: Path | np.ndarray | None, rotation: int, use_raw: bool) -> str:
        """
        Suggest one style prompt for the sketch the next round would use, the next style each time for the same drawing.

        Args:
            sketch (Path | np.ndarray | None): Sketch file, camera frame in OpenCV channel order, or None when missing.
            rotation (int): Degrees to turn the sketch clockwise before processing.
            use_raw (bool): Suggest from the raw sketch, which skips preprocessing.

        Returns:
            str: The suggested prompt.
        """
        # Compare raw frames, since the crop and contrast steps make camera noise look like a changed drawing.
        raw_payload = self.preview_sketch(sketch, rotation, True)
        if raw_payload is None:
            raise InvalidSketchError(NO_SKETCH_MESSAGE)

        # Caption a changed drawing as the generator would see it, falling back to the raw sketch when preprocessing finds none.
        signature = compute_sketch_signature(raw_payload)
        if has_sketch_changed(self._suggested_signature, signature):
            payload = raw_payload if use_raw else self.preview_sketch(sketch, rotation, False) or raw_payload
            self._suggestions = self.suggester.suggest(payload)
            self._suggested_signature, self._suggestion_index = signature, 0
        else:
            self._suggestion_index = (self._suggestion_index + 1) % len(self._suggestions)

        return self._suggestions[self._suggestion_index]

    def run_round(self, sketch: Path | np.ndarray | None, source: SketchSource, prompt: str, *, n_candidates: int,
                  seed: int | None = None, rotation: int = Rotation.DEG_000, use_raw: bool = False,
                  frame_age_seconds: float | None = None, controls: Mapping[str, ControlValue] | None = None) -> RoundView:
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
            controls (Mapping[str, ControlValue] | None, optional): Requested backend controls, such as steps, or None
                for the backend defaults. Defaults to None.

        Returns:
            RoundView: The completed round.
        """
        # Check the inputs before reading any file, so a bad request writes nothing.
        self._check_round_inputs(sketch, source, prompt, rotation, use_raw, frame_age_seconds)

        # Load and time a sketch file or a camera frame, keeping the prompt as typed and recording the requested controls.
        captured = capture_browser_sketch(sketch)
        guidance = Guidance(prompt=prompt, controls=controls or {})
        outcome = self.session.run_round(captured.image, captured.payload, guidance, source=source,
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

    def describe_stroke_suggester(self) -> str:
        """
        Return the stroke suggester's one-sentence description for the page header.
        """
        return self.stroke_suggester.describe()

    def _find_open_drawing(self) -> GuidedDrawing | None:
        # Only the session's latest drawing can be open.
        drawings = self.session.drawings
        return drawings[-1] if drawings and drawings[-1].is_open else None

    def _end_open_drawing(self) -> None:
        # End an open drawing without a final capture, since the session it belongs to is being replaced.
        drawing = self._find_open_drawing()
        if drawing is not None:
            drawing.finish(None, end=DrawingEnd.NEW_SESSION)

    def _check_drawing_inputs(self, sketch: Path | np.ndarray | None, source: SketchSource, rotation: int,
                              frame_age_seconds: float | None) -> GuidedDrawing:
        # Refuse a missing drawing or frame, a rotation preprocessing cannot apply, and a misplaced age, before reading anything.
        drawing = self._find_open_drawing()
        if drawing is None:
            raise SketchLoopError(NO_DRAWING_MESSAGE)

        if sketch is None:
            raise InvalidSketchError(NO_FRAME_MESSAGE)

        if rotation not in list(Rotation):
            raise SketchLoopError(f"Rotation must be one of 0, 90, 180, or 270, got {rotation}.")

        if frame_age_seconds is not None and source != SketchSource.WEBCAM:
            raise SketchLoopError(f"Only a streamed webcam frame has a frame age, got source {source}. Pass no frame age.")

        return drawing

    def start_drawing(self, name: str) -> DrawingView:
        """
        Start a named experiment as a new session folder and open its first drawing.

        Args:
            name (str): Experiment name, such as participant1_car, which also names the session folder.

        Returns:
            DrawingView: The open drawing as the page shows it.
        """
        # Claim the name in a new session first, so an invalid or taken name leaves the current session and drawing as they are.
        name = check_experiment_name(name)
        # A session that saved nothing has not recorded the generator's setup yet, so the named session records it instead.
        unsaved = self.session.session_record is None
        setup_seconds = self.session.generator_setup_seconds if unsaved else Unavailable(SETUP_REUSED_REASON)
        session = SketchSession(self.generator, self.store, self.evaluator, preprocessor=self.preprocessor,
                                generator_setup_seconds=setup_seconds, name=name)

        # End the open drawing, then start drawing 1 of the named session with the configured suggester.
        self._end_open_drawing()
        self.session = session
        width, height = self.canvas_size
        drawing = session.start_drawing(self.stroke_suggester, canvas_width=width, canvas_height=height,
                                        save_interval_seconds=self.save_interval_seconds)
        return build_drawing_view(drawing, session.name)

    def suggest_strokes(self, sketch: Path | np.ndarray | None, source: SketchSource, prompt: str, *, rotation: int,
                        seed: int | None = None, frame_age_seconds: float | None = None) -> SuggestionView:
        """
        Suggest how to continue the open drawing from the current sketch and the prompt as typed.

        Args:
            sketch (Path | np.ndarray | None): Webcam frame or uploaded file.
            source (SketchSource): Where the sketch came from.
            prompt (str): The prompt box text exactly as typed, possibly empty.
            rotation (int): Degrees to turn the sketch clockwise.
            seed (int | None, optional): Seed for the suggestion, or None for random. Defaults to None.
            frame_age_seconds (float | None, optional): Age of a streamed webcam frame. Defaults to None.

        Returns:
            SuggestionView: The suggestion's status line, part count, and number.
        """
        # Check the inputs before reading any file, then suggest with the prompt box text exactly as typed, empty included.
        drawing = self._check_drawing_inputs(sketch, source, rotation, frame_age_seconds)
        captured = capture_browser_sketch(sketch)
        suggestion = drawing.suggest(captured, source=source, rotation=rotation, prompt=prompt, seed=seed,
                                     frame_age_seconds=frame_age_seconds)
        return build_suggestion_view(drawing.record.number, suggestion)

    def preview_guided_sketch(self, sketch: Path | np.ndarray | None, rotation: int) -> bytes | None:
        """
        Compose the live sketch in the paper frame with the open suggestion's overlay.

        Args:
            sketch (Path | np.ndarray | None): Webcam frame or uploaded file.
            rotation (int): Degrees to turn the sketch clockwise.

        Returns:
            bytes | None: PNG of the preview, or None when there is no open drawing, no frame, or no paper found.
        """
        # Without an open drawing or a frame, the page keeps its ordinary preview.
        drawing = self._find_open_drawing()
        if drawing is None or sketch is None:
            return None

        # Warp the frame to the paper canvas by its corners only, without alignment, since this runs every capture interval.
        try:
            _, payload = encode_frame(sketch) if isinstance(sketch, np.ndarray) else load_sketch_file(sketch)
            canvas = make_paper_canvas(payload, rotation, drawing.record.canvas_width, drawing.record.canvas_height)
            return drawing.compose_preview(canvas)
        except (SketchLoopError, OSError, ValueError):
            return None

    def finish_drawing(self, sketch: Path | np.ndarray | None, source: SketchSource, *, rotation: int,
                       frame_age_seconds: float | None = None) -> DrawingView:
        """
        Finish the open drawing, keeping the current sketch as its final capture when there is one.

        Args:
            sketch (Path | np.ndarray | None): Webcam frame or uploaded file, or None to finish without a capture.
            source (SketchSource): Where the sketch came from.
            rotation (int): Degrees to turn the sketch clockwise.
            frame_age_seconds (float | None, optional): Age of a streamed webcam frame. Defaults to None.

        Returns:
            DrawingView: The finished drawing as the page shows it.
        """
        # Finish with a final capture when a frame exists, and without one otherwise.
        drawing = self._find_open_drawing()
        if drawing is None:
            raise SketchLoopError(NO_DRAWING_MESSAGE)

        if sketch is not None:
            self._check_drawing_inputs(sketch, source, rotation, frame_age_seconds)
            drawing.finish(capture_browser_sketch(sketch), source=source, rotation=rotation, frame_age_seconds=frame_age_seconds)
        else:
            drawing.finish(None)

        return build_drawing_view(drawing, self.session.name)

    def latest_drawing(self) -> DrawingView | None:
        """
        Return the current session's latest drawing, open or finished, or None when it has none.
        """
        # Return the session's latest drawing, open or finished, so a reload shows it again.
        drawings = self.session.drawings
        return build_drawing_view(drawings[-1], self.session.name) if drawings else None

    def start_new_session(self) -> None:
        """
        Start a fresh session folder with the same loaded generator, store, evaluator, and preprocessor.
        """
        # End an open drawing first, since the session it belongs to is being replaced.
        self._end_open_drawing()

        # The generator is reused, so this session paid no setup, and repeating the first value would count it twice.
        self.session = SketchSession(self.generator, self.store, self.evaluator, preprocessor=self.preprocessor,
                                     generator_setup_seconds=Unavailable(SETUP_REUSED_REASON))
