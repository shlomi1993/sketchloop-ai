"""
Tests that stores claim experiment names exclusively and save drawing records after their files.
"""

import json
import pytest

from datetime import datetime, timezone
from pathlib import Path

from sketchloop.experiments import (DRAWING_FILE, DrawingRecord, ExperimentNameTakenError, ExperimentStore, FolderStore,
                                    MemoryStore, StoreError)


def make_drawing_record(number: int) -> DrawingRecord:
    """
    Build an open drawing record with no captures, as a drawing has right after it starts.
    """
    return DrawingRecord(schema_version=1, number=number, drawing_id="d" * 32, started_at=datetime.now(timezone.utc),
                         ended_at=None, end=None, canvas_width=512, canvas_height=360, save_interval_seconds=2.0,
                         suggestions_enabled=True, suggester="FakeStrokeSuggester", captures=(), suggestions=(),
                         final_capture=None)


@pytest.mark.parametrize("taken_name", ["participant1_car", "Participant1_CAR"])
def test_folder_store_claims_a_name_once_ignoring_case(tmp_path: Path, taken_name: str) -> None:
    """
    FolderStore creates the named folder, then refuses the same name in any letter case and leaves the folder untouched.
    """
    store = FolderStore(tmp_path / "runs")
    store.claim_session("participant1_car")
    marker = tmp_path / "runs" / "participant1_car" / "keep.txt"
    marker.write_text("kept", encoding="utf-8")
    with pytest.raises(ExperimentNameTakenError) as caught:
        store.claim_session(taken_name)

    # The second claim names the experiment in one line, and the first folder is the only entry, unchanged.
    message = str(caught.value)
    expected = f"An experiment named {taken_name} already exists in the runs folder. Choose another name."
    assert message == expected, f"Expected one line naming the experiment, got {message!r}"
    assert [path.name for path in (tmp_path / "runs").iterdir()] == ["participant1_car"], "A refused claim must create nothing"
    assert marker.read_text(encoding="utf-8") == "kept", "A refused claim must leave the existing folder untouched"


def test_folder_store_refuses_a_name_matching_a_file(tmp_path: Path) -> None:
    """
    A file in the runs folder with the name, ignoring case, also takes it, since the folder could not be created.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "NOTES").write_text("x", encoding="utf-8")

    # The name of an existing file is taken too.
    with pytest.raises(ExperimentNameTakenError):
        FolderStore(runs_dir).claim_session("notes")


@pytest.mark.parametrize("taken_name", ["p1", "P1"])
def test_memory_store_claims_a_name_once_ignoring_case(taken_name: str) -> None:
    """
    MemoryStore refuses a name already claimed, in any letter case.
    """
    store = MemoryStore()
    store.claim_session("p1")

    # The second claim fails, and only the first name is held.
    with pytest.raises(ExperimentNameTakenError):
        store.claim_session(taken_name)
    assert store.claimed == {"p1"}, f"Expected only the first claim, got {store.claimed}"


def test_folder_store_saves_drawing_files_then_record(tmp_path: Path) -> None:
    """
    FolderStore writes a drawing's files under drawing-<n>/ and then drawing.json, and locates that folder.
    """
    store: ExperimentStore = FolderStore(tmp_path / "runs")
    record = make_drawing_record(2)
    store.save_drawing("session", record, {"captures/capture-0001-raw.png": b"raw"})
    folder = store.locate_drawing("session", 2)
    data = json.loads((folder / DRAWING_FILE).read_text(encoding="utf-8"))

    # The folder holds the file and a record of the drawing number.
    assert folder == tmp_path / "runs" / "session" / "drawing-2", f"Unexpected drawing folder {folder}"
    assert (folder / "captures" / "capture-0001-raw.png").read_bytes() == b"raw", "The capture file must be written"
    assert data["drawing"] == 2 and data["captures"] == [], f"Unexpected drawing.json {data}"


def test_folder_store_names_failed_drawing_file(tmp_path: Path) -> None:
    """
    A file that cannot be written raises one StoreError naming it relative to the runs folder, and drawing.json is not written.
    """
    store = FolderStore(tmp_path / "runs")
    (tmp_path / "runs" / "session" / "drawing-1").mkdir(parents=True)
    (tmp_path / "runs" / "session" / "drawing-1" / "captures").write_text("not a folder", encoding="utf-8")
    with pytest.raises(StoreError) as caught:
        store.save_drawing("session", make_drawing_record(1), {"captures/capture-0001-raw.png": b"raw"})

    # The message is relative and the record was not written after the failed file.
    message = str(caught.value)
    assert message.startswith("Cannot save session/drawing-1/captures/capture-0001-raw.png:"), f"Unexpected message {message!r}"
    assert str(tmp_path) not in message, f"The message must not hold an absolute path: {message!r}"
    assert not (tmp_path / "runs" / "session" / "drawing-1" / DRAWING_FILE).exists(), "drawing.json must follow its files"


def test_memory_store_keeps_drawings_without_folders() -> None:
    """
    MemoryStore keeps drawing records and file bytes and has no drawing folders.
    """
    store = MemoryStore()
    record = make_drawing_record(1)
    store.save_drawing("session", record, {"overlay.png": b"png"})

    # The record and bytes are kept by session and number, and no folder exists.
    assert store.drawings[("session", 1)] is record, "The drawing record must be kept"
    assert store.drawing_files[("session", 1, "overlay.png")] == b"png", "The drawing file must be kept"
    assert store.locate_drawing("session", 1) is None, "A memory store has no drawing folders"
