import ast
import cv2
import numpy as np
import pytest
import re
import subprocess
import sys

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final

import sketchloop
from sketchloop.app import AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CapturedSketch, CaptureSource, encode_frame, raw_image_ref
from sketchloop.domain import (Guidance, InvalidRecordError, SketchLoopError, SketchSource, Unavailable,
                               UnsupportedConfigurationError, record_no_selection, select_candidates)
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import SETUP_NOT_MEASURED_REASON, MemoryStore, RoundRecord, SessionRecord, StoreError
from sketchloop.generation import Generator
from sketchloop.preprocessing import PreprocessedSketch, PreprocessingStep, Preprocessor
from sketchloop.session import SketchSession


def list_imported_names(module_path: Path) -> set[str]:
    """
    List every module and name a module or package imports anywhere, including module.name pairs for from-imports.
    """
    files = sorted(module_path.glob("*.py")) if module_path.is_dir() else [module_path.with_suffix(".py")]
    names: set[str] = set()
    for node in (node for file in files for node in ast.walk(ast.parse(file.read_text(encoding="utf-8")))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(name for alias in node.names for name in (alias.name, f"{node.module}.{alias.name}"))

    return names


# Orchestration may import only protocols, records, and default components, never implementations it should receive.
ORCHESTRATION_FORBIDDEN: Final = ("FolderStore", "MemoryStore", "write_json_atomic", "preprocess_sketch", "sketchloop.backends",
                                  "sketchloop.components", "sketchloop.settings", "sketchloop.ui", "gradio")


@pytest.mark.parametrize(("module", "forbidden"), [
    ("session", (*ORCHESTRATION_FORBIDDEN, "write_files", "load_sketch_file", "capture_from_camera", "cv2")),
    ("app", ORCHESTRATION_FORBIDDEN),
    ("cli", ())
])
def test_import_boundaries(module: str, forbidden: tuple[str, ...]) -> None:
    """
    Session and controller import no store, preprocessing, capture, factory, or UI code, and no module loads gradio on import.
    """
    imported = list_imported_names(Path(sketchloop.__file__).parent / module)
    found = sorted(name for name in imported if any(name == banned or name.startswith(f"{banned}.") for banned in forbidden))
    code = "; ".join(["import sys", "sys.modules['gradio'] = None", f"import sketchloop.{module}",
                      "assert 'sketchloop.ui' not in sys.modules"])
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)

    # The module's source imports nothing orchestration must receive by injection.
    assert found == [], f"sketchloop.{module} imports {found}, which must be injected instead"

    # A subprocess imports the module without touching gradio or the UI module.
    assert result.returncode == 0, f"sketchloop.{module} imported gradio or sketchloop.ui: {result.stderr}"


class StillCamera(CaptureSource):
    """
    Capture source standing in for a camera, returning the same synthetic PNG on every capture.
    """

    @property
    def kind(self) -> SketchSource:
        """
        Record sketches from this source as camera input.
        """
        return SketchSource.CAMERA

    def capture(self) -> CapturedSketch:
        """
        Encode a synthetic light page with one dark stroke as the raw PNG sketch, reporting a fixed capture time.
        """
        page = np.full((96, 128, 3), 220, dtype=np.uint8)
        cv2.line(page, (10, 10), (110, 80), (20, 20, 20), 3)
        image, payload = encode_frame(page)
        return CapturedSketch(image=image, payload=payload, capture_seconds=0.25)


class IdentityPreprocessor(Preprocessor):
    """
    Preprocessor that passes the sketch through unchanged as sketch.png with one identity step.
    """

    def preprocess(self, payload: bytes, rotation: int) -> PreprocessedSketch:
        """
        Return the payload unchanged under the processed sketch name.
        """
        pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = replace(raw_image_ref(pixels, payload, ".png", "image/png"), path="sketch.png")
        return PreprocessedSketch(image=image, payload=payload, steps=(PreprocessingStep(name="identity", params={}),))


class FlakyMemoryStore(MemoryStore):
    """
    Memory store whose armed save methods each fail once with StoreError, standing in for a full disk.
    """

    def __init__(self) -> None:
        super().__init__()
        self.armed: set[str] = set()

    def fail_if_armed(self, method: str) -> None:
        """
        Raise StoreError once for an armed method.
        """
        if method in self.armed:
            self.armed.discard(method)
            raise StoreError(f"Cannot keep {method}: test failure. Retry.")

    def save_session(self, record: SessionRecord) -> None:
        """
        Fail once when armed, otherwise keep the session record.
        """
        self.fail_if_armed("save_session")
        super().save_session(record)

    def save_files(self, session_id: str, number: int, files: Mapping[str, bytes]) -> None:
        """
        Fail once when armed, otherwise keep the round's files.
        """
        self.fail_if_armed("save_files")
        super().save_files(session_id, number, files)

    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        Fail once when armed, otherwise keep the round's files and record.
        """
        self.fail_if_armed("save_round")
        super().save_round(session_id, record, files)


def test_session_runs_with_substituted_components(tmp_path: Path, constant_evaluator: Evaluator) -> None:
    """
    A session runs unchanged with a test camera, preprocessor, evaluator, and memory store, and a failed save loses nothing.
    """
    camera, store = StillCamera(), FlakyMemoryStore()
    session = SketchSession(FakeGenerator(), store, constant_evaluator, preprocessor=IdentityPreprocessor())
    captured = camera.capture()
    raw_sketch, raw_payload = captured.image, captured.payload

    # A failed session save consumes no round number, and the next round saves session.json again.
    store.armed = {"save_session"}
    with pytest.raises(StoreError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=camera.kind, n_candidates=2, seed=3)
    round1 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source=camera.kind, n_candidates=2, seed=3)

    # A failed selection save records the choice nowhere, so the same choice can be retried.
    selection = select_candidates(round1.iteration, [round1.iteration.result.candidates[0].id])
    store.armed = {"save_round"}
    with pytest.raises(StoreError):
        session.record_selection(selection)
    unsaved = (round1.iteration.id not in session.selections, session.records[-1].selection is None,
               store.rounds[(session.session_id, 1)].selection is None)
    assert unsaved == (True, True, True), f"A failed selection save must leave no selection, got {unsaved}"
    session.record_selection(selection)

    # A failed round save appends no outcome, then a raw round picking none, then a round failing in generation.
    store.armed = {"save_round"}
    with pytest.raises(StoreError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="lamp"), source=camera.kind)
    round3 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="lamp"), source=camera.kind, use_raw=True,
                               capture_seconds=captured.capture_seconds)
    session.record_selection(record_no_selection(round3.iteration))
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="vase"), source=camera.kind, n_candidates=99)

    # A failure whose record cannot be saved still raises its own error rather than the save error.
    store.armed = {"save_round"}
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="vase"), source=camera.kind, n_candidates=99)

    # The session history lists every attempt, and the failed save of round 2 is recorded as a saving failure.
    history = [(record.number, record.status) for record in session.records]
    assert history == [(1, "complete"), (2, "failed"), (3, "complete"), (4, "failed"), (5, "failed")], f"Bad history {history}"
    assert session.records[1].failure.stage == "saving", f"Expected a saving failure, got {session.records[1].failure}"

    session_id = session.session_id
    records = {number: record for (stored_id, number), record in store.rounds.items() if stored_id == session_id}
    files_by_round = {number: sorted(path for stored_id, file_round, path in store.files if file_round == number)
                      for number in records}

    # The store holds one session, round 2 as failed after its save failed, and no round 5, whose failed record was not saved.
    statuses = {number: record.status for number, record in records.items()}
    assert list(store.sessions) == [session_id], f"Expected one stored session record, got {list(store.sessions)}"
    assert statuses == {1: "complete", 2: "failed", 3: "complete", 4: "failed"}, f"Unexpected stored rounds {statuses}"
    assert [outcome.number for outcome in session.rounds] == [1, 3], "A failed save must append no outcome"

    # Stored selections equal the session's, and round 3 links to round 1 across the failed save.
    stored_selections = [records[outcome.number].selection for outcome in session.rounds]
    assert stored_selections == [session.selections[outcome.iteration.id] for outcome in session.rounds], "Selections differ"
    assert records[3].iteration.parent_id == records[1].iteration.id, "Round 3 must link to round 1"

    # Round 1 records the substitute preprocessor's step, the evaluator's class and scores, and the camera source.
    step_names = [step.name for step in records[1].steps]
    assert step_names == ["identity"] and records[1].evaluator == "ConstantEvaluator", f"Bad round 1 record {records[1]}"
    assert list(records[1].scores.values()) == [0.5, 0.5], f"Expected two scores of 0.5, got {records[1].scores}"
    assert {record.source for record in records.values()} == {"camera"}, "Every round must record the camera source"

    # Round 1 keeps both sketches and its candidates, the raw round has no sketch.png, and the failed round keeps its sketches.
    candidates = ["candidates/candidate-1.png", "candidates/candidate-2.png"]
    assert files_by_round[1] == [*candidates, "sketch-raw.png", "sketch.png"], f"Unexpected round 1 files {files_by_round[1]}"
    assert files_by_round[3] == [candidates[0], "sketch-raw.png"], f"Unexpected raw round files {files_by_round[3]}"
    assert files_by_round[4] == ["sketch-raw.png", "sketch.png"], f"Unexpected failed round files {files_by_round[4]}"
    assert store.files[(session_id, 1, "sketch-raw.png")] == raw_payload, "The raw sketch must be kept byte for byte"

    # Nothing has a folder, and nothing is written to disk.
    assert session.folder is None and all(outcome.folder is None for outcome in session.rounds), "Memory rounds have no folder"
    assert list(tmp_path.iterdir()) == [], f"A memory store must write no files, found {list(tmp_path.iterdir())}"

    # The browser controller refuses a store that keeps no files.
    message = "The browser interface shows saved image files, so it needs a store that keeps files in a folder."
    with pytest.raises(SketchLoopError, match=re.escape(message)):
        AppController(FakeGenerator(), MemoryStore())


@pytest.mark.usefixtures("counting_clock")
def test_session_records_stage_timings(first_load_generator: Generator) -> None:
    """
    Capture, a reported model load, saving, and the total are recorded, saved as shown, and kept up to a failure.
    """
    camera, store = StillCamera(), FlakyMemoryStore()
    session = SketchSession(first_load_generator, store, preprocessor=IdentityPreprocessor(), generator_setup_seconds=1.5)
    captured = camera.capture()
    guidance = Guidance(prompt="chair")
    first = session.run_round(captured.image, captured.payload, guidance, source=camera.kind,
                              capture_seconds=captured.capture_seconds)
    second = session.run_round(captured.image, captured.payload, guidance, source=camera.kind)

    # The generate call takes one 7 s clock step, of which the reported 5 s load is model_load and the rest generation.
    timings = first.timings_seconds
    stages = ["capture", "preprocessing", "model_load", "generation", "evaluation", "saving"]
    assert list(timings) == stages, f"Expected every stage in order, got {list(timings)}"
    assert timings["capture"] == 0.25 and timings["model_load"] == 5.0, f"Unexpected capture or load {timings}"
    assert timings["generation"] == pytest.approx(2.0), f"Generation must leave out the load, got {timings['generation']}"

    # The total adds the capture to the round's wall time, which covers every later stage, including the load.
    later_stages = sum(seconds for stage, seconds in timings.items() if stage != "capture")
    assert first.total_seconds >= later_stages + 0.25, f"Total {first.total_seconds} must cover the stages {timings}"

    # A round without a capture time and without a load records neither stage.
    second_stages = set(second.timings_seconds)
    assert second_stages == {"preprocessing", "generation", "evaluation", "saving"}, f"Unexpected stages {second_stages}"

    # The saved records hold exactly the shown timings and totals, and the session holds the setup time.
    saved = [store.rounds[(session.session_id, number)] for number in (1, 2)]
    shown = [(outcome.timings_seconds, outcome.total_seconds) for outcome in (first, second)]
    assert [(record.timings_seconds, record.total_seconds) for record in saved] == shown, "Saved timings differ from shown"
    assert session.session_record.generator_setup_seconds == 1.5, "The session must record the generator setup time"

    # A failed round keeps the stages it finished and a total, and a failed file save is a saving failure.
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(captured.image, captured.payload, guidance, source=camera.kind, n_candidates=99, capture_seconds=0.5)
    store.armed = {"save_files"}
    with pytest.raises(StoreError):
        session.run_round(captured.image, captured.payload, guidance, source=camera.kind)
    failed_generation, failed_saving = session.records[2], session.records[3]
    assert set(failed_generation.timings_seconds) == {"capture", "preprocessing"}, f"Bad partial {failed_generation}"
    assert failed_generation.total_seconds >= 0.5, f"A failed round must keep a total, got {failed_generation.total_seconds}"
    assert failed_saving.failure.stage == "saving", f"A failed file save must be a saving failure, got {failed_saving.failure}"

    # A library caller that measures no setup records why.
    default_session = SketchSession(FakeGenerator(), MemoryStore())
    default_session.run_round(captured.image, captured.payload, guidance, source=camera.kind)
    setup = default_session.session_record.generator_setup_seconds
    assert setup == Unavailable(SETUP_NOT_MEASURED_REASON), f"Expected the not-measured reason, got {setup}"


def test_session_refuses_load_longer_than_call(first_load_generator: Generator) -> None:
    """
    A backend that reports a model load longer than its whole call fails the round in generation instead of being clamped.
    """
    captured = StillCamera().capture()
    session = SketchSession(first_load_generator, MemoryStore())

    # The real clock makes the fake call far shorter than its reported 5 s load.
    with pytest.raises(InvalidRecordError, match="model load in a .* call. Fix its load_seconds."):
        session.run_round(captured.image, captured.payload, Guidance(prompt="chair"), source="file")
    failure = session.records[0].failure
    assert failure.stage == "generation", f"An impossible load must fail in generation, got {failure}"
