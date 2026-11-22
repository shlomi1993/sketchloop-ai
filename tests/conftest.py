"""
Shared pytest fixtures and test doubles for evaluators, generators, clocks, and guided drawing frames.
"""

import cv2
import numpy as np
import pytest
import time

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CapturedSketch, encode_frame
from sketchloop.domain import Candidate, GenerationRequest, SketchSource
from sketchloop.drawing import GuidedDrawing
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import (SESSION_FILE, DrawingEnd, DrawingStudy, FolderStore, PromptSuggestionRecord,
                                    PromptSuggestionStatus, StudyCondition)
from sketchloop.generation import GenerationOutput
from sketchloop.session import SketchSession
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.suggestion import TemplateSuggester


class ConstantEvaluator(Evaluator):
    """
    Evaluator that gives every candidate a score of 0.5.
    """

    def evaluate(self, candidates: Sequence[Candidate], payloads: Mapping[str, bytes],
                 request: GenerationRequest) -> Mapping[str, float]:
        """
        Score every candidate 0.5.
        """
        return {candidate.id: 0.5 for candidate in candidates}

    def describe_scores(self) -> Mapping[str, str]:
        """
        Say that the score is a fixed test value.
        """
        return {"score": "0.5 for every candidate, a test value"}


class FirstLoadGenerator(FakeGenerator):
    """
    Test-only fake generator that reports a 5 s model load on its first call only, like a lazily loading backend.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loaded = False

    def generate(self, request: GenerationRequest, sketch_payload: bytes) -> GenerationOutput:
        """
        Generate fake candidates, reporting a load only on the first call.
        """
        output = super().generate(request, sketch_payload)
        load_seconds = None if self.loaded else 5.0
        self.loaded = True
        return replace(output, load_seconds=load_seconds)


def draw_sheet_frame() -> np.ndarray:
    """
    Draw a synthetic 640x480 webcam frame of a light landscape sheet on a dark table, with one rectangle drawn on it.
    """
    frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.rectangle(frame, (60, 50), (580, 430), (215, 215, 215), -1)
    cv2.rectangle(frame, (220, 150), (400, 290), (30, 30, 30), 4)
    return frame


def capture_frame(frame: np.ndarray) -> CapturedSketch:
    """
    Encode a frame as the PNG raw sketch the browser controller would capture, with a fixed 0.01 s capture time.
    """
    image, payload = encode_frame(frame)
    return CapturedSketch(image=image, payload=payload, capture_seconds=0.01)


class CountingClock:
    """
    Test clock for time.perf_counter that moves forward by a fixed step on every read, plus any explicit advance.
    """

    def __init__(self, step_seconds: float) -> None:
        self.step_seconds = step_seconds
        self.now = 0.0
        self.reads = 0

    def __call__(self) -> float:
        """
        Advance by one step and return the new time.
        """
        self.reads += 1
        self.now += self.step_seconds
        return self.now

    def advance(self, seconds: float) -> None:
        """
        Move the clock forward without a read, standing in for time that passes between reads.
        """
        self.now += seconds


@pytest.fixture
def counting_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace time.perf_counter with a clock that advances 7 s per read, longer than FirstLoadGenerator's 5 s load.
    """
    monkeypatch.setattr(time, "perf_counter", CountingClock(step_seconds=7.0))


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Run the test from its empty tmp_path, so a developer's own sketchloop.toml cannot change the settings it sees.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def constant_evaluator() -> Evaluator:
    """
    Provide an evaluator that scores every candidate 0.5, recorded under the class name ConstantEvaluator.
    """
    return ConstantEvaluator()


@pytest.fixture
def first_load_generator() -> FirstLoadGenerator:
    """
    Provide a fake generator that reports a 5 s model load on its first call only.
    """
    return FirstLoadGenerator()


@pytest.fixture
def sheet_frame() -> np.ndarray:
    """
    Provide a synthetic webcam frame of a sheet on a table with one rectangle drawn on it.
    """
    return draw_sheet_frame()


@pytest.fixture
def sheet_capture(sheet_frame: np.ndarray) -> CapturedSketch:
    """
    Provide the sheet frame with a rectangle as a captured PNG raw sketch.
    """
    return capture_frame(sheet_frame)


@pytest.fixture
def blank_capture() -> CapturedSketch:
    """
    Provide an evenly white frame, whose paper canvas has no strokes, as a captured PNG raw sketch.
    """
    return capture_frame(np.full((480, 640, 3), 255, dtype=np.uint8))



def draw_part_on_sheet(frame: np.ndarray, part_mask: bytes) -> np.ndarray:
    """
    Draw a suggested part's mask in dark pencil on the sheet frame, mapping canvas pixels onto the sheet's corners.
    """
    mask = cv2.imdecode(np.frombuffer(part_mask, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    drawn = frame.copy()
    on_sheet = cv2.resize(mask, (521, 381), interpolation=cv2.INTER_NEAREST) > 0
    drawn[50:431, 60:581][on_sheet] = 30
    return drawn


@pytest.fixture(name="draw_part_on_sheet")
def provide_part_drawer() -> Callable[[np.ndarray, bytes], np.ndarray]:
    """
    Provide a function that draws a suggested part's mask on the sheet frame, as a person following it would.
    """
    return draw_part_on_sheet


def build_prompt_suggestion(text: str, suggestion_id: str = "prompt-1", *, number: int = 1) -> PromptSuggestionRecord:
    # A shown template suggestion, pressed two seconds ago and shown one second ago, so a press made now comes after it.
    now = datetime.now(timezone.utc)
    return PromptSuggestionRecord(number=number, id=suggestion_id, status=PromptSuggestionStatus.SHOWN,
                                  requested_at=now - timedelta(seconds=2), shown_at=now - timedelta(seconds=1),
                                  suggester=TemplateSuggester().identity, text=text, list_position=0, ran_suggester=True,
                                  suggester_seconds=0.01, after_capture=None, failure=None)


@pytest.fixture(name="make_prompt_suggestion")
def provide_prompt_suggestion_maker() -> Callable[..., PromptSuggestionRecord]:
    return build_prompt_suggestion


def refuse_canvas(*_: object) -> None:
    # Stand in for a final frame whose sheet cannot be found.
    raise ValueError("no sheet found")


def start_study_drawing(runs_dir: Path, name: str | None,
                        condition: StudyCondition | None) -> tuple[SketchSession, GuidedDrawing]:
    # A named or time-named session holding one open drawing, with the fake suggester only in the AI condition.
    session = SketchSession(FakeGenerator(), FolderStore(runs_dir), name=name)
    study = DrawingStudy(condition=condition) if condition is not None else None
    suggester = FakeStrokeSuggester() if condition != StudyCondition.NO_AI else None
    drawing = session.start_drawing(suggester, canvas_width=512, canvas_height=360, save_interval_seconds=2.0, study=study)
    return session, drawing


def build_study_runs(runs_dir: Path, capture: CapturedSketch) -> Path:
    # Two exportable drawings, with and without AI, whose AI one used a prompt suggestion, then one per missing reason.
    _, with_ai = start_study_drawing(runs_dir, "p1_car", StudyCondition.AI)
    shown = with_ai.record_prompt_suggestion(build_prompt_suggestion("a car, studio render", "p-1"))
    with_ai.suggest(capture, source=SketchSource.UPLOAD, rotation=0, prompt="a car, studio render", prompt_suggestion_id=shown.id)
    with_ai.finish(capture, source=SketchSource.UPLOAD, prompt="a red car", prompt_suggestion_id=shown.id)
    start_study_drawing(runs_dir, "p2_car", StudyCondition.NO_AI)[1].finish(capture, source=SketchSource.UPLOAD, prompt="")
    start_study_drawing(runs_dir, "p3_car", StudyCondition.AI)[1].suggest(capture, source=SketchSource.UPLOAD, rotation=0,
                                                                          prompt="a car")
    start_study_drawing(runs_dir, "p4_car", StudyCondition.NO_AI)[1].finish(None, prompt="a car")
    start_study_drawing(runs_dir, "p5_old", None)[1].finish(capture, source=SketchSource.UPLOAD, prompt="a car")
    start_study_drawing(runs_dir, "p6_car", StudyCondition.NO_AI)[1].finish(None, end=DrawingEnd.NEW_SESSION)

    # The final frame of p7 cannot become a canvas, and the final canvas of p8 changes after the drawing ends.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("sketchloop.drawing.make_paper_canvas", refuse_canvas)
        start_study_drawing(runs_dir, "p7_car", StudyCondition.NO_AI)[1].finish(capture, source=SketchSource.UPLOAD, prompt="")
    session, changed = start_study_drawing(runs_dir, "p8_car", StudyCondition.AI)
    changed.finish(capture, source=SketchSource.UPLOAD, prompt="")
    (session.folder / "drawing-1" / changed.record.captures[-1].canvas.path).write_bytes(b"changed")

    # A broken session, a time-named session, and a folder that is not a session.
    (runs_dir / "p9_broken").mkdir()
    (runs_dir / "p9_broken" / SESSION_FILE).write_text("{", encoding="utf-8")
    start_study_drawing(runs_dir, None, StudyCondition.AI)[1].finish(capture, source=SketchSource.UPLOAD, prompt="")
    (runs_dir / "notes").mkdir()
    return runs_dir


@pytest.fixture
def study_runs(tmp_path: Path, sheet_capture: CapturedSketch) -> Path:
    return build_study_runs(tmp_path / "runs", sheet_capture)
