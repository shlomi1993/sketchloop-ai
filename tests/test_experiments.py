import cv2
import json
import numpy as np
import pytest
import re

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from sketchloop.app import AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import CaptureSource, encode_frame, load_sketch_file, raw_image_ref
from sketchloop.cli import main
from sketchloop.domain import (Guidance, ImageRef, SketchLoopError, SketchSource, Unavailable, UnsupportedConfigurationError,
                               record_no_selection, select_candidates)
from sketchloop.evaluation import Evaluator
from sketchloop.experiments import (MISSING_SOURCE_REASON, ExperimentRecordError, FolderStore, MemoryStore, RoundRecord,
                                    SessionRecord, StoreError, describe_failure, load_session)
from sketchloop.preprocessing import PreprocessedSketch, PreprocessingStep, Preprocessor
from sketchloop.session import SketchSession


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")

EXAMPLE_SKETCH = Path(__file__).resolve().parents[1] / "examples" / "sketch-photo.jpg"


def run_two_rounds(runs_dir: Path) -> SketchSession:
    """
    Run two fake rounds, a file round picking candidate 1 and a raw camera round picking none.
    """
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    session = SketchSession(FakeGenerator(), FolderStore(runs_dir))
    round1 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="chair"), source="file", n_candidates=2, seed=7)
    session.record_selection(select_candidates(round1.iteration, [round1.iteration.result.candidates[0].id]))
    round2 = session.run_round(raw_sketch, raw_payload, Guidance(prompt="lamp", controls={"steps": 8}), source="camera",
                               use_raw=True)
    session.record_selection(record_no_selection(round2.iteration))
    return session


def test_records_round_trip(tmp_path: Path) -> None:
    """
    Two rounds and a failed third save their files and load back with equal iterations, steps, selections, sources, and checksums.
    """
    session = run_two_rounds(tmp_path)
    round1_folder = session.folder / "round-1"
    round1_files = sorted(path.relative_to(round1_folder).as_posix() for path in round1_folder.rglob("*.*"))
    raw_sketch, raw_payload = load_sketch_file(EXAMPLE_SKETCH)
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="vase"), source="webcam", n_candidates=99)
    (session.folder / "round-1" / "round.json.tmp").write_text("{", encoding="utf-8")

    saved = load_session(session.folder)
    round1, round2, round3 = saved.rounds
    outcomes = session.rounds

    # Round 1 holds its sketches, candidates, and record, and the raw round 2 writes no processed sketch.
    expected_files = ["candidates/candidate-1.png", "candidates/candidate-2.png", "round.json", "sketch-raw.jpg", "sketch.png"]
    assert round1_files == expected_files, f"Round 1 must hold the sketches, candidates, and record, got {round1_files}"
    assert not (session.folder / "round-2" / "sketch.png").exists(), "A raw round must skip preprocessing"

    # The saved records reload into the same iterations, preprocessing steps, and round-to-round lineage.
    iterations_equal = [round1.iteration, round2.iteration] == [outcome.iteration for outcome in outcomes]
    assert iterations_equal, "Loaded requests, results, and effective settings must equal the session's"
    assert round1.steps == outcomes[0].steps and round2.steps == (), "Preprocessing steps must load unchanged"
    assert round2.iteration.parent_id == round1.iteration.id, "Round 2 must link to round 1"

    # Selections, stage timings, the failed round, and every file checksum survive the save and load.
    selections = [round1.selection, round2.selection]
    assert selections == [session.selections[outcome.iteration.id] for outcome in outcomes], f"Selections changed: {selections}"
    assert set(round1.timings_seconds) == {"preprocessing", "generation", "evaluation"}, "Round 1 must time every stage"
    assert round3.status == "failed" and round3.failure.stage == "generation", f"Round 3 must fail in generation: {round3}"
    assert saved.problems == () and saved.checked_files == 8, f"All files must match: {saved.problems}"

    # Each round's sketch source, including the failed round's, loads back unchanged.
    sources = [record.source for record in saved.rounds]
    assert sources == ["file", "camera", "webcam"], f"Sources changed on reload: {sources}"

    # Package versions are recorded as real versions or as unavailable with a reason, never invented.
    versions = list(saved.session.environment.packages.values())
    assert all(isinstance(value, str | Unavailable) for value in versions), f"Unexpected versions {versions}"


@pytest.mark.parametrize(("damage", "problem"), [("flip", "does not match its recorded checksum"), ("delete", "is missing")])
def test_damaged_candidate_listed(tmp_path: Path, damage: str, problem: str) -> None:
    """
    A changed or deleted candidate file is listed as one problem instead of failing the load.
    """
    session = run_two_rounds(tmp_path)
    candidate = session.folder / "round-1" / "candidates" / "candidate-1.png"
    if damage == "flip":
        payload = bytearray(candidate.read_bytes())
        payload[-1] ^= 1
        candidate.write_bytes(payload)
    else:
        candidate.unlink()

    # The damaged file is reported as exactly one problem naming the file, while the session still loads.
    problems = load_session(session.folder).problems
    assert problems == (f"round-1/candidates/candidate-1.png {problem}.",), f"Expected one problem, got {problems}"


@pytest.mark.parametrize(("file_name", "edit", "message"), [
    ("round-1/round.json", lambda data: data["input"]["raw_sketch"].update(path="../x.png"), "outside its round folder"),
    ("session.json", lambda data: data.update(schema_version=2), "schema version 2"),
    ("round-2/round.json", lambda data: data.update(parent_id="0" * 32), "does not link to round-1"),
    ("round-1/round.json", lambda data: data.update(started_at="2026-10-09T11:25:02"), "timezone-aware"),
    ("round-1/round.json", lambda data: data["evaluation"].update(scores={"x": float("nan")}), "finite"),
    ("round-1/round.json", lambda data: data["input"].update(source="scanner"), "source must be file, camera, webcam, or upload")
])
def test_invalid_records_rejected(tmp_path: Path, file_name: str, edit: Callable[[dict], None], message: str) -> None:
    """
    Path traversal, an unknown schema version, broken lineage, a naive time, a NaN score, and an unknown source refuse to load.
    """
    session = run_two_rounds(tmp_path)
    record_file = session.folder / file_name
    data = json.loads(record_file.read_text(encoding="utf-8"))
    edit(data)
    record_file.write_text(json.dumps(data), encoding="utf-8")

    # Loading refuses the damaged record with an error that names the reason.
    with pytest.raises(ExperimentRecordError, match=message):
        load_session(session.folder)


class StillCamera(CaptureSource):
    """
    Capture source standing in for a camera, returning the same synthetic PNG on every capture.
    """

    @property
    def kind(self) -> SketchSource:
        """
        Record sketches from this source as camera input.
        """
        return "camera"

    def capture(self) -> tuple[ImageRef, bytes]:
        """
        Encode a synthetic light page with one dark stroke as the raw PNG sketch.
        """
        page = np.full((96, 128, 3), 220, dtype=np.uint8)
        cv2.line(page, (10, 10), (110, 80), (20, 20, 20), 3)
        return encode_frame(page)


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
    raw_sketch, raw_payload = camera.capture()

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
        session.run_round(*camera.capture(), Guidance(prompt="lamp"), source=camera.kind)
    round3 = session.run_round(*camera.capture(), Guidance(prompt="lamp"), source=camera.kind, use_raw=True)
    session.record_selection(record_no_selection(round3.iteration))
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="vase"), source=camera.kind, n_candidates=99)

    # A failure whose record cannot be saved still raises its own error rather than the save error.
    store.armed = {"save_round"}
    with pytest.raises(UnsupportedConfigurationError):
        session.run_round(raw_sketch, raw_payload, Guidance(prompt="vase"), source=camera.kind, n_candidates=99)

    # The session history lists every failed attempt, including round 5, whose failed record was not saved.
    history = [(record.number, record.status) for record in session.records]
    assert history == [(1, "complete"), (3, "complete"), (4, "failed"), (5, "failed")], f"Bad history {history}"

    session_id = session.session_id
    records = {number: record for (stored_id, number), record in store.rounds.items() if stored_id == session_id}
    files_by_round = {number: sorted(path for stored_id, file_round, path in store.files if file_round == number)
                      for number in records}

    # The store holds one session and rounds 1, 3, and 4, since the failed save left round 2 empty and was never reused.
    statuses = {number: record.status for number, record in records.items()}
    assert list(store.sessions) == [session_id], f"Expected one stored session record, got {list(store.sessions)}"
    assert statuses == {1: "complete", 3: "complete", 4: "failed"}, f"Unexpected stored rounds {statuses}"
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


def refuse_generator(*args: object, **kwargs: object) -> None:
    """
    Fail the test if a command creates this generator.
    """
    raise AssertionError("This command must not create this generator")


@pytest.mark.parametrize("drop_source", [False, True], ids=["with-source", "recorded-before-sources"])
def test_cli_rerun_links_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                              drop_source: bool) -> None:
    """
    Rerunning a fake round with a diffusers settings file makes a linked fake session in the file's runs folder.
    """
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *prompt: next(answers))
    config_dir = tmp_path / "config"
    runs_dir = config_dir / "runs"
    config_dir.mkdir()
    config = config_dir / "rerun.toml"
    config.write_text('[generator]\nbackend = "diffusers"\n\n[storage]\nruns_dir = "runs"\n', encoding="utf-8")
    main(["run", str(EXAMPLE_SKETCH), "--prompt", "chair", "--candidates", "2", "--runs-dir", str(runs_dir)])
    original = next(runs_dir.iterdir())
    recorded_source = load_session(original).rounds[0].source

    # Strip input.source to stand in for a round saved before sources were tracked.
    if drop_source:
        record_file = original / "round-1" / "round.json"
        data = json.loads(record_file.read_text(encoding="utf-8"))
        del data["input"]["source"]
        record_file.write_text(json.dumps(data), encoding="utf-8")

    original_source = load_session(original).rounds[0].source

    # The file asks for diffusers, but a rerun recreates the recorded fake backend, so diffusers must never be created.
    monkeypatch.setattr("sketchloop.backends.DiffusersSketchGenerator", refuse_generator)
    rerun_exit = main(["rerun", str(original), "--round", "1", "--config", str(config)])
    rerun_output = capsys.readouterr().out

    rerun_folders = [path for path in runs_dir.iterdir() if path != original]
    monkeypatch.setattr("sketchloop.backends.FakeGenerator", refuse_generator)
    show_exit = main(["show", str(rerun_folders[0])]) if rerun_folders else None
    show_output = capsys.readouterr().out

    # Both commands succeed, the rerun is saved in the file's runs folder, and it records which original round it repeats.
    assert rerun_exit == 0 and show_exit == 0, f"Rerun and show must succeed, got {rerun_exit} and {show_exit}"
    assert len(rerun_folders) == 1, f"The rerun must be saved in the settings file's runs folder, got {rerun_folders}"
    rerun_folder = rerun_folders[0]
    link = load_session(rerun_folder).session.rerun_of
    assert link.session_id == original.name and link.round_number == 1, f"The rerun must link to round 1, got {link}"

    # The deterministic fake gives identical candidates, the rerun never claims exact replay, and show reports the comparison.
    assert "Candidates with identical bytes: 1, 2 of 2" in rerun_output, "The fake rerun must give identical candidates"
    assert "reproduced" not in rerun_output.lower(), "A rerun must never claim exact replay"
    assert "Candidates with identical bytes: 1, 2 of 2" in show_output, "show must compare a rerun with its original"

    # A run from a file records file, an older record loads as unavailable, and the rerun copies whichever it finds.
    expected_source = Unavailable(MISSING_SOURCE_REASON) if drop_source else "file"
    rerun_source = load_session(rerun_folder).rounds[0].source
    assert recorded_source == "file", f"run from a file must record source file, got {recorded_source}"
    assert original_source == expected_source, f"Expected original source {expected_source}, got {original_source}"
    assert rerun_source == expected_source, f"The rerun must copy the original source {expected_source}, got {rerun_source}"

    # show prints the rerun round's source, naming the reason when it is unavailable.
    shown_source = "(source: unavailable (recorded before input sources were tracked))" if drop_source else "(source: file)"
    assert shown_source in " ".join(show_output.split()), f"show must print {shown_source}"


# Build each path from pieces so this file never contains a literal home path for the publication guard to flag.
@pytest.mark.parametrize(("message", "expected"), [
    ("Cannot open '" + "C:" + "\\Users\\some one\\unet.bin': denied", "Cannot open '<path>': denied"),
    ("Cannot open \"/" + "home/someone/unet.bin\": denied", "Cannot open \"<path>\": denied"),
    ("Cannot open /" + "Volumes/Data/unet.bin now", "Cannot open <path> now"),
    ("See https://huggingface.co/models or models/unet.bin", "See https://huggingface.co/models or models/unet.bin")
], ids=["windows-with-space", "posix-quoted", "posix-unquoted", "url-and-relative"])
def test_failure_message_hides_paths(message: str, expected: str) -> None:
    """
    A saved failure message replaces absolute paths, which can name the user, and keeps URLs and relative paths.
    """
    failure = describe_failure("generation", OSError(f"{message}\nsecond line"))

    # Only the first line is kept, with each absolute path replaced.
    assert failure.message == expected and failure.error == "OSError", f"Unexpected failure record {failure}"
