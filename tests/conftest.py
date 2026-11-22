"""
Shared pytest fixtures and test doubles for evaluators, generators, clocks, and guided drawing frames.
"""

import cv2
import numpy as np
import pytest
import time

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CapturedSketch, encode_frame
from sketchloop.domain import Candidate, GenerationRequest
from sketchloop.evaluation import Evaluator
from sketchloop.generation import GenerationOutput


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
