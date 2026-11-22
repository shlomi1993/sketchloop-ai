"""
Tests of the blinded study export: codes, the key, the export record, the refusals, and the atomic write.
"""

import csv
import cv2
import hashlib
import json
import numpy as np
import pytest
import struct
import zlib

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

from sketchloop.experiments import (DRAWINGS_FOLDER, EXPORT_CODE_ALPHABET, EXPORT_CODE_LENGTH, EXPORT_FILE, KEY_COLUMNS, KEY_FILE,
                                    ExportError, StudyExport, export_study_drawings, make_export_code)
from sketchloop.experiments.study_export import reencode_png


STUDY_EXPERIMENTS: Final = ("p1_car", "p2_car", "p3_car", "p4_car", "p6_car", "p7_car", "p8_car")
EXPECTED_MISSING: Final = {
    "p3_car": "not finished: the server stopped during it",
    "p4_car": "finished without a final frame",
    "p6_car": "ended by a new session",
    "p7_car": "final frame has no canvas: no sheet found",
    "p8_car": "final canvas is missing or changed",
}
TEXT_CHUNKS: Final = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}


def make_code_sequence(*codes: str) -> Callable[[], str]:
    # A code maker that returns the given codes in order, standing in for secrets.
    remaining: Iterator[str] = iter(codes)
    return lambda: next(remaining)


def read_key(folder: Path) -> list[dict[str, str]]:
    with (folder / KEY_FILE).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def list_png_chunks(payload: bytes) -> list[bytes]:
    # Walk the chunks after the 8-byte signature: a 4-byte length, a 4-byte type, the data, and a 4-byte CRC.
    chunks, position = [], 8
    while position < len(payload):
        length = struct.unpack(">I", payload[position:position + 4])[0]
        chunks.append(payload[position + 4:position + 8])
        position += 12 + length

    return chunks


def export_mixed(study_runs: Path, tmp_path: Path) -> StudyExport:
    return export_study_drawings(study_runs, tmp_path / "export", make_code=make_code_sequence("AAAAAA", "BBBBBB"))


def test_export_writes_one_png_per_final_drawing_and_a_key_with_missing_rows(study_runs: Path, tmp_path: Path) -> None:
    export = export_mixed(study_runs, tmp_path)
    folder = tmp_path / "export"

    # The drawings folder holds the two exportable drawings under their codes, beside the key and the record.
    assert sorted(path.name for path in folder.iterdir()) == [DRAWINGS_FOLDER, EXPORT_FILE, KEY_FILE], "Unexpected export layout"
    names = sorted(path.name for path in (folder / DRAWINGS_FOLDER).iterdir())
    assert names == ["AAAAAA.png", "BBBBBB.png"], f"Expected one PNG per exported drawing, got {names}"

    # The key has the design's columns, one row per study drawing by experiment, and every missing reason.
    rows = read_key(folder)
    with (folder / KEY_FILE).open(encoding="utf-8", newline="") as file:
        header = next(csv.reader(file))
    assert tuple(header) == KEY_COLUMNS, f"Expected the key columns {KEY_COLUMNS}, got {header}"
    assert [row["experiment"] for row in rows] == list(STUDY_EXPERIMENTS), f"Expected rows by experiment, got {rows}"
    exported = {row["experiment"]: (row["code"], row["condition"]) for row in rows if row["status"] == "exported"}
    assert exported == {"p1_car": ("AAAAAA", "ai"), "p2_car": ("BBBBBB", "no_ai")}, f"Unexpected exported rows {exported}"
    missing = {row["experiment"]: row["reason"] for row in rows if row["status"] == "missing"}
    assert missing == EXPECTED_MISSING, f"Unexpected missing reasons {missing}"
    assert all(row["code"] == "" for row in rows if row["status"] == "missing"), "A missing drawing must have no code"

    # The broken session is skipped with its error, and the time-named session, T09 drawing, and stray folder are passed over.
    assert len(export.skipped) == 1 and export.skipped[0].startswith("p9_broken: session.json is not valid JSON"), export.skipped
    assert export.passed_over == 3, f"Expected 3 passed-over entries, got {export.passed_over}"

    # The export record counts the drawings and keeps the match and the skipped line.
    record = json.loads((folder / EXPORT_FILE).read_text(encoding="utf-8"))
    counts = (record["schema_version"], record["exported"], record["missing"], record["passed_over"], record["match"])
    assert counts == (1, 2, 5, 3, "*"), f"Unexpected export record {record}"
    assert record["skipped"] == list(export.skipped), f"Expected the skipped line in {record}"


def test_key_checksums_name_the_recorded_canvas_and_the_exported_file(study_runs: Path, tmp_path: Path) -> None:
    export_mixed(study_runs, tmp_path)
    rows = [row for row in read_key(tmp_path / "export") if row["status"] == "exported"]

    # The source is the recorded canvas, whose bytes still match, and the exported checksum is the PNG handed over.
    for row in rows:
        source = study_runs / row["source"]
        assert row["source"].startswith(f"{row['experiment']}/drawing-1/captures/"), f"Unexpected source {row['source']}"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"], f"Source checksum differs in {row}"
        exported = tmp_path / "export" / DRAWINGS_FOLDER / f"{row['code']}.png"
        assert hashlib.sha256(exported.read_bytes()).hexdigest() == row["exported_sha256"], f"Exported checksum differs in {row}"
        assert row["final_capture"].isdigit(), f"Expected the final capture number in {row}"


def test_exported_pngs_keep_the_canvas_pixels_without_metadata(study_runs: Path, tmp_path: Path) -> None:
    export_mixed(study_runs, tmp_path)

    # Each PNG decodes to the recorded canvas's pixels and holds only image chunks.
    for row in (row for row in read_key(tmp_path / "export") if row["status"] == "exported"):
        payload = (tmp_path / "export" / DRAWINGS_FOLDER / f"{row['code']}.png").read_bytes()
        exported = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        recorded = cv2.imread(str(study_runs / row["source"]), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(exported, recorded), f"Exported pixels differ from the recorded canvas of {row['experiment']}"
        chunks = set(list_png_chunks(payload))
        assert not chunks & TEXT_CHUNKS, f"Expected no text or time chunks, got {chunks}"


def test_nothing_handed_over_or_in_the_record_names_an_experiment_or_condition(study_runs: Path, tmp_path: Path) -> None:
    export_mixed(study_runs, tmp_path)
    handed_over = [(path.name.encode(), path.read_bytes()) for path in (tmp_path / "export" / DRAWINGS_FOLDER).iterdir()]
    record = (tmp_path / "export" / EXPORT_FILE).read_bytes()

    # Neither file names, PNG bytes, nor the export record carry a study experiment's name or a condition.
    for label in (*STUDY_EXPERIMENTS, "no_ai", '"ai"', "condition"):
        leaks = [name for name, payload in handed_over if label.encode() in name or label.encode() in payload]
        assert not leaks, f"Drawings {leaks} reveal {label}"
        assert label.encode() not in record, f"export.json reveals {label}"


def test_match_limits_the_export_ignoring_case(study_runs: Path, tmp_path: Path) -> None:
    export = export_study_drawings(study_runs, tmp_path / "export", match="P1_*", make_code=make_code_sequence("AAAAAA"))

    # Only the matching experiment is exported, and nothing else is counted or skipped.
    experiments = [entry.experiment for entry in export.entries]
    assert experiments == ["p1_car"], f"Expected only p1_car, got {experiments}"
    assert (export.skipped, export.passed_over) == ((), 0), f"Expected no other entries, got {export}"


def test_a_code_collision_draws_again(study_runs: Path, tmp_path: Path) -> None:
    make_code = make_code_sequence("AAAAAA", "AAAAAA", "BBBBBB")
    export = export_study_drawings(study_runs, tmp_path / "export", make_code=make_code)

    # The second drawing gets the redrawn code, so codes stay unique.
    codes = sorted(entry.code for entry in export.entries if entry.code)
    assert codes == ["AAAAAA", "BBBBBB"], f"Expected a redraw after the collision, got {codes}"


def test_a_code_that_is_not_from_the_alphabet_is_refused(study_runs: Path, tmp_path: Path) -> None:
    # A code that could name a path outside the drawings folder is refused, and nothing is written.
    with pytest.raises(ExportError, match="is not 6 characters of the code alphabet"):
        export_study_drawings(study_runs, tmp_path / "export", make_code=lambda: "../../x")
    assert list(tmp_path.iterdir()) == [study_runs], f"Expected nothing written, got {list(tmp_path.iterdir())}"


def test_make_export_code_uses_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = {make_export_code() for _ in range(200)}

    # Codes have the fixed length and alphabet, and 200 draws do not repeat.
    assert len(codes) == 200, f"Expected 200 different codes, got {len(codes)}"
    valid = all(len(code) == EXPORT_CODE_LENGTH and set(code) <= set(EXPORT_CODE_ALPHABET) for code in codes)
    assert valid, f"Expected codes of the alphabet, got {sorted(codes)[:5]}"

    # Replacing secrets.choice makes the code deterministic, so it is the only source of randomness.
    monkeypatch.setattr("sketchloop.experiments.study_export.secrets.choice", lambda alphabet: alphabet[-1])
    assert make_export_code() == "ZZZZZZ", "make_export_code must draw every character with secrets.choice"


def test_an_empty_output_folder_is_used(study_runs: Path, tmp_path: Path) -> None:
    (tmp_path / "export").mkdir()
    export_mixed(study_runs, tmp_path)

    # The empty folder is replaced by the finished export.
    assert (tmp_path / "export" / KEY_FILE).is_file(), "An empty output folder must receive the export"


def make_unusable_output(study_runs: Path, tmp_path: Path, case: str) -> tuple[Path, Path]:
    # Return the runs folder and the output for each refused case.
    if case == "not empty":
        (tmp_path / "export").mkdir()
        (tmp_path / "export" / "earlier.txt").write_text("earlier", encoding="utf-8")

        return study_runs, tmp_path / "export"

    if case == "inside the runs folder":
        return study_runs, study_runs / "export"

    if case == "missing runs folder":
        return tmp_path / "no-runs", tmp_path / "export"

    return study_runs, tmp_path / "export"


@pytest.mark.parametrize(("case", "message"), [
    ("not empty", "is not empty"),
    ("inside the runs folder", "is inside the runs folder"),
    ("missing runs folder", "does not exist"),
])
def test_unusable_folders_are_refused_before_writing(study_runs: Path, tmp_path: Path, case: str, message: str) -> None:
    runs_dir, output = make_unusable_output(study_runs, tmp_path, case)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    # The export is refused with one line and nothing is added.
    with pytest.raises(ExportError, match=message):
        export_study_drawings(runs_dir, output, make_code=make_export_code)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before, f"Expected nothing written, got {set(after) - set(before)}"


def test_a_runs_folder_without_study_drawings_is_refused(tmp_path: Path) -> None:
    (tmp_path / "runs" / "notes").mkdir(parents=True)

    # No study drawing means nothing to export.
    with pytest.raises(ExportError, match="No study drawings in .* match '\\*'"):
        export_study_drawings(tmp_path / "runs", tmp_path / "export")
    assert not (tmp_path / "export").exists(), "No export folder may be created"


def test_a_runs_folder_without_a_usable_final_canvas_is_refused(study_runs: Path, tmp_path: Path) -> None:
    # Every matching drawing is missing, so the refusal lists each with its reason.
    with pytest.raises(ExportError, match="No study drawing has a usable final canvas: p3_car drawing 1: not finished"):
        export_study_drawings(study_runs, tmp_path / "export", match="p3_*")
    assert not (tmp_path / "export").exists(), "No export folder may be created"


def test_a_failed_write_leaves_neither_the_export_nor_a_temporary_folder(study_runs: Path, tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def fill_disk(*_: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("sketchloop.experiments.study_export.write_key_file", fill_disk)

    # The error names the step and the cause, and the partial folder is removed.
    with pytest.raises(ExportError, match="while writing the files: No space left on device"):
        export_mixed(study_runs, tmp_path)
    assert list(tmp_path.iterdir()) == [study_runs], f"Expected only the runs folder, got {list(tmp_path.iterdir())}"


def test_the_export_appears_only_when_complete(study_runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen_during_write = []
    original = Path.rename

    def record_and_rename(source: Path, target: Path) -> Path:
        # Look at the folders just before the final rename.
        seen_during_write.append((source.name, target.exists(), (source / EXPORT_FILE).is_file()))
        return original(source, target)

    monkeypatch.setattr(Path, "rename", record_and_rename)
    export_mixed(study_runs, tmp_path)

    # The files were written in a temporary sibling, renamed to the output only when complete, and nothing else is left.
    assert len(seen_during_write) == 1, f"Expected one rename, got {seen_during_write}"
    name, output_existed, complete = seen_during_write[0]
    assert name.startswith("export.partial-") and not output_existed and complete, f"Unexpected rename {seen_during_write}"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["export", "runs"], "No temporary folder may remain"


def insert_text_chunk(payload: bytes, text: bytes) -> bytes:
    # Put a tEXt chunk with its CRC right after the 33 bytes of signature and IHDR, as an image editor might.
    chunk = b"tEXt" + text
    return payload[:33] + struct.pack(">I", len(text)) + chunk + struct.pack(">I", zlib.crc32(chunk)) + payload[33:]


def test_reencoding_drops_text_chunks_and_keeps_pixels() -> None:
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    with_text = insert_text_chunk(cv2.imencode(".png", pixels)[1].tobytes(), b"Comment\x00p1_car no_ai")
    reencoded = reencode_png(with_text)

    # The text chunk is gone with its words, and the pixels are unchanged.
    assert b"tEXt" in list_png_chunks(with_text), "The test image must start with a text chunk"
    assert not set(list_png_chunks(reencoded)) & TEXT_CHUNKS and b"p1_car" not in reencoded, "Re-encoding must drop text"
    decoded = cv2.imdecode(np.frombuffer(reencoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded, pixels), "Re-encoding must keep the pixels"
