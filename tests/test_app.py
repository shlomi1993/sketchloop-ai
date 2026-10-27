"""
Tests of the browser controller's rounds, selections, history, and new sessions.
"""

import cv2
import hashlib
import json
import numpy as np
import pytest
import re

from pathlib import Path

from sketchloop.app import NO_SKETCH_MESSAGE, AppController
from sketchloop.backends.fake import FakeGenerator
from sketchloop.capture import InvalidSketchError
from sketchloop.domain import InvalidSelectionError, SketchLoopError, Unavailable, UnsupportedConfigurationError
from sketchloop.experiments import SETUP_REUSED_REASON, FolderStore, load_session


def write_sketch_png(path: Path, channels: int) -> Path:
    """
    Write a synthetic light page with one dark rectangle as an RGB or RGBA PNG, like a browser webcam or upload file.
    """
    page = np.full((600, 900, channels), 210, dtype=np.uint8)
    cv2.rectangle(page, (500, 100), (700, 200), (30,) * channels, 4)

    # Keep an RGBA page fully opaque, as a browser upload of a scanned page would be.
    if channels == 4:
        page[..., 3] = 255

    path.write_bytes(cv2.imencode(".png", page)[1].tobytes())
    return path


@pytest.fixture
def sketch_file(tmp_path: Path) -> Path:
    """
    Provide a synthetic RGB PNG sketch standing in for the browser's file.
    """
    return write_sketch_png(tmp_path / "browser-sketch.png", channels=3)


@pytest.fixture
def controller(tmp_path: Path) -> AppController:
    """
    Provide a controller with the deterministic fake backend and a runs folder under tmp_path.
    """
    return AppController(FakeGenerator(), FolderStore(tmp_path / "runs"))


def read_round_files(controller: AppController) -> dict[str, bytes]:
    """
    Read every round.json of the controller's current session, keyed by round folder name.
    """
    return {path.parent.name: path.read_bytes() for path in sorted(controller.session.folder.glob("round-*/round.json"))}


@pytest.mark.parametrize(("source", "channels"), [("webcam", 3), ("upload", 4)])
def test_app_round_records_browser_source(tmp_path: Path, controller: AppController, source: str, channels: int) -> None:
    """
    A streamed webcam frame or an uploaded file gives fake-labeled candidates on disk, and records the source and raw bytes.
    """
    # The webcam streams pixels, which are saved as a lossless PNG, while an upload keeps the file's own bytes.
    sketch = write_sketch_png(tmp_path / f"{source}.png", channels)
    frame = cv2.imread(str(sketch))
    sketch_input = frame if source == "webcam" else sketch
    raw_bytes = cv2.imencode(".png", frame)[1].tobytes() if source == "webcam" else sketch.read_bytes()
    view = controller.run_round(sketch_input, source, "modern chair", n_candidates=3, seed=4)
    round_folder = controller.session.folder / "round-1"
    data = json.loads((round_folder / "round.json").read_text(encoding="utf-8"))
    saved = load_session(controller.session.folder)

    # The view is round 1 with the requested candidates, each stored under the round's candidates folder.
    candidate_folders = {candidate.image_path.resolve().parent for candidate in view.candidates}
    assert view.number == 1 and len(view.candidates) == 3, f"Expected round 1 with 3 candidates, got {view}"
    assert candidate_folders == {(round_folder / "candidates").resolve()}, f"Candidates stored elsewhere: {candidate_folders}"
    assert all(candidate.image_path.is_file() for candidate in view.candidates), "Every candidate image must exist"
    assert view.sketch_path.is_file() and view.selection is None, f"Expected a stored sketch and no selection, got {view}"

    # The fake backend is labeled on the view and on every caption.
    labels = [candidate.label for candidate in view.candidates]
    assert view.is_fake and labels == ["1 (fake backend)", "2 (fake backend)", "3 (fake backend)"], f"Bad labels {labels}"

    # The record names the browser source, and the raw sketch holds the frame's PNG or the uploaded file's bytes.
    expected_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    assert data["input"]["source"] == source, f"Expected input.source {source}, got {data['input'].get('source')}"
    assert data["input"]["raw_sketch"]["sha256"] == expected_sha256, "The raw sketch must hold the frame or file bytes"

    # The source loads back unchanged and every file matches its checksum.
    assert saved.rounds[0].source == source and saved.problems == (), f"Bad reload: {saved.rounds[0].source}, {saved.problems}"

    # The browser frame encode or upload load is the capture stage, and the view carries the saved total.
    assert "capture" in data["timings_seconds"], f"The round must time its capture, got {data['timings_seconds']}"
    assert view.total_seconds == data["total_seconds"], f"The view total {view.total_seconds} must equal the saved total"


@pytest.mark.parametrize(("sketch_kind", "source", "prompt", "options", "error", "message"), [
    (None, "upload", "chair", {}, InvalidSketchError, NO_SKETCH_MESSAGE),
    ("text", "upload", "chair", {}, InvalidSketchError, "Sketch notes.png is not a PNG or JPEG file."),
    ("png", "upload", "   ", {}, SketchLoopError, "Enter a prompt, then generate."),
    ("png", "upload", "chair", {"rotation": 45}, SketchLoopError, "Rotation must be one of 0, 90, 180, or 270, got 45."),
    ("png", "upload", "chair", {"rotation": 90, "use_raw": True}, SketchLoopError,
     "Rotation cannot be combined with the raw sketch option, which skips preprocessing."),
    ("png", "scanner", "chair", {}, SketchLoopError, "Round source must be file, camera, webcam, or upload, got 'scanner'.")
], ids=["no-sketch", "not-an-image", "blank-prompt", "bad-rotation", "rotation-with-raw", "unknown-source"])
def test_app_rejects_input_before_writing(tmp_path: Path, controller: AppController, sketch_file: Path, sketch_kind: str | None,
                                          source: str, prompt: str, options: dict[str, object], error: type[Exception],
                                          message: str) -> None:
    """
    A missing or non-image sketch, a blank prompt, a bad rotation, rotation with raw, or an unknown source saves nothing.
    """
    text_file = tmp_path / "notes.png"
    text_file.write_text("synthetic text, not an image", encoding="utf-8")
    sketch = {None: None, "text": text_file, "png": sketch_file}[sketch_kind]

    # The controller raises the documented one-line message.
    with pytest.raises(error, match=re.escape(message)):
        controller.run_round(sketch, source, prompt, n_candidates=2, **options)

    # No session folder is created for refused input.
    assert not controller.store.root.exists(), "Refused input must not create a session folder"


def test_app_selection_rules(controller: AppController, sketch_file: Path) -> None:
    """
    Candidates 1 and 3 are saved, none is explicit, and nothing to select, a repeat, a stale page, or a foreign ID is refused.
    """
    # Nothing can be selected before the first round.
    with pytest.raises(InvalidSelectionError, match=re.escape("No round to select from yet. Generate first.")):
        controller.record_selection(None, [])

    first = controller.run_round(sketch_file, "upload", "chair", n_candidates=3)
    picked = controller.record_selection(first.iteration_id, [first.candidates[0].candidate_id, first.candidates[2].candidate_id])
    round1_data = json.loads((controller.session.folder / "round-1" / "round.json").read_text(encoding="utf-8"))

    # The pick is shown as "1, 3" and saved with the two candidate IDs.
    expected_ids = [first.candidates[0].candidate_id, first.candidates[2].candidate_id]
    assert picked.selection == "1, 3", f"Expected selection 1, 3, got {picked.selection}"
    assert round1_data["selection"]["candidate_ids"] == expected_ids, f"Saved selection differs: {round1_data['selection']}"

    # A second selection for the same round is refused without touching its record.
    before = read_round_files(controller)
    repeat_message = "Round 1 already has a recorded selection. Generate the next round."
    with pytest.raises(InvalidSelectionError, match=re.escape(repeat_message)):
        controller.record_selection(first.iteration_id, [])
    assert read_round_files(controller) == before, "A refused repeat selection must not change any round.json"

    # A stale tab still showing round 1 cannot record a choice once round 2 exists.
    second = controller.run_round(sketch_file, "upload", "lamp", n_candidates=2)
    before = read_round_files(controller)
    stale_message = "This page shows an older round. Reload the page to see round 2."
    with pytest.raises(InvalidSelectionError, match=re.escape(stale_message)):
        controller.record_selection(first.iteration_id, [])
    assert read_round_files(controller) == before, "A stale selection must leave every round.json unchanged"

    # A candidate from round 1 cannot be selected in round 2.
    with pytest.raises(InvalidSelectionError):
        controller.record_selection(second.iteration_id, [first.candidates[0].candidate_id])
    assert read_round_files(controller) == before, "A foreign candidate ID must leave every round.json unchanged"

    # An empty pick is recorded as an explicit none.
    declined = controller.record_selection(second.iteration_id, [])
    round2_data = json.loads((controller.session.folder / "round-2" / "round.json").read_text(encoding="utf-8"))
    assert declined.selection == "none", f"An empty pick must show none, got {declined.selection}"
    assert round2_data["selection"]["candidate_ids"] == [], f"An explicit none must be saved, got {round2_data['selection']}"

    # A tab still showing round 1 of the old session cannot record none for round 1 of a new session.
    controller.start_new_session()
    controller.run_round(sketch_file, "upload", "stool", n_candidates=1)
    with pytest.raises(InvalidSelectionError, match=re.escape("Reload the page to see round 1.")):
        controller.record_selection(first.iteration_id, [])


def test_app_history_refresh_and_new_session(controller: AppController, sketch_file: Path) -> None:
    """
    Rounds link, a failed round joins the history but not the view, a reload view matches, and a new session keeps the backend.
    """
    generator = controller.session.generator
    empty_view, empty_history = controller.latest_round(), controller.list_history()
    first = controller.run_round(sketch_file, "webcam", "chair", n_candidates=2)
    picked = controller.record_selection(first.iteration_id, [first.candidates[1].candidate_id])
    reloaded_first = controller.latest_round()
    second = controller.run_round(sketch_file, "upload", "lamp", n_candidates=1, use_raw=True)
    with pytest.raises(UnsupportedConfigurationError):
        controller.run_round(sketch_file, "upload", "vase", n_candidates=99)

    history = controller.list_history()
    saved = load_session(controller.session.folder)
    old_folder = controller.session.folder

    # Before any round there is nothing to restore and no history.
    assert empty_view is None and empty_history == (), f"Expected no view or history, got {empty_view}, {empty_history}"

    # A reload shows the latest completed round with its recorded selection, even after a failed round.
    assert reloaded_first == picked and picked.selection == "2", f"A reload must show round 1 picked as 2, got {reloaded_first}"
    assert controller.latest_round() == second, f"A failed round must not replace round 2, got {controller.latest_round()}"

    # Round 2 links to round 1, and the history lists all three rounds, including the failure.
    rows = [(row.number, row.status, row.prompt) for row in history]
    assert saved.rounds[1].iteration.parent_id == saved.rounds[0].iteration.id, "Round 2 must link to round 1"
    assert rows == [(1, "complete", "chair"), (2, "complete", "lamp"), (3, "failed", "vase")], f"Unexpected history {rows}"
    outcomes = [row.outcome for row in history]
    assert outcomes[:2] == ["selected 2", "not selected"], f"Unexpected outcomes {outcomes}"
    assert outcomes[2].startswith("failed during generation: "), f"Round 3 must fail in generation, got {outcomes[2]}"

    # A new session starts empty, keeps the loaded generator, and saves its first round in a new unlinked folder.
    controller.start_new_session()
    new_history, new_view = controller.list_history(), controller.latest_round()
    restarted = controller.run_round(sketch_file, "webcam", "stool", n_candidates=1)
    new_saved = load_session(controller.session.folder)
    assert new_history == () and new_view is None, f"A new session must start empty, got {new_history}, {new_view}"
    assert controller.session.generator is generator, "A new session must reuse the loaded generator"
    assert controller.session.folder != old_folder and restarted.number == 1, "A new session must number rounds from 1 anew"
    assert new_saved.rounds[0].iteration.parent_id is None, "The first round of a new session must not link to the old one"


def test_app_records_frame_age_and_setup(tmp_path: Path, sketch_file: Path) -> None:
    """
    A webcam round records its frame age, an age with an upload is refused, and only the first session has the setup time.
    """
    controller = AppController(FakeGenerator(), FolderStore(tmp_path / "runs"), generator_setup_seconds=2.0)
    frame = cv2.imread(str(sketch_file))

    # An age with the upload source is refused before anything is read or saved.
    with pytest.raises(SketchLoopError, match="Only a streamed webcam frame has a frame age"):
        controller.run_round(sketch_file, "upload", "chair", n_candidates=1, frame_age_seconds=0.4)
    assert not controller.store.root.exists(), "A refused frame age must save nothing"

    # A webcam round records the age, and the first session the setup time.
    controller.run_round(frame, "webcam", "chair", n_candidates=1, frame_age_seconds=0.4)
    first = load_session(controller.session.folder)
    assert first.rounds[0].frame_age_seconds == 0.4, f"Expected frame age 0.4, got {first.rounds[0].frame_age_seconds}"
    assert first.session.generator_setup_seconds == 2.0, f"Expected setup 2.0, got {first.session.generator_setup_seconds}"

    # A new session reuses the generator, so it records the reused reason instead of the setup time again.
    controller.start_new_session()
    controller.run_round(sketch_file, "upload", "lamp", n_candidates=1)
    reused = load_session(controller.session.folder).session.generator_setup_seconds
    assert reused == Unavailable(SETUP_REUSED_REASON), f"A new session must record the reused reason, got {reused}"
