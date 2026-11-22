"""
Blinded export of study drawings: final canvases under random codes, with a key that stays on this machine.
"""

import csv
import cv2
import fnmatch
import hashlib
import json
import numpy as np
import secrets
import shutil

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Final

from sketchloop.domain import SketchLoopError, first_line
from sketchloop.experiments.drawings import DrawingEnd, DrawingRecord, StudyCondition
from sketchloop.experiments.loading import load_session
from sketchloop.experiments.provenance import read_package_version, redact_paths
from sketchloop.experiments.records import SESSION_FILE, ExperimentRecordError, SavedSession
from sketchloop.experiments.serialization import encode_value


EXPORT_CODE_ALPHABET: Final = "23456789ABCDEFGHJKMNPQRSTVWXYZ"  # No 0, 1, I, L, O, or U, so codes read aloud without confusion
EXPORT_CODE_LENGTH: Final = 6
MAX_CODE_ATTEMPTS: Final = 100
DRAWINGS_FOLDER: Final = "drawings"
KEY_FILE: Final = "key.csv"
EXPORT_FILE: Final = "export.json"
EXPORT_SCHEMA_VERSION: Final = 1
EXPORTED_IMAGE: Final = "final paper canvas, grayscale PNG re-encoded without metadata"
KEY_COLUMNS: Final = ("code", "experiment", "drawing", "condition", "status", "reason", "final_capture", "source",
                      "source_sha256", "exported_sha256")
NOT_FINISHED_REASON: Final = "not finished: the server stopped during it"
NEW_SESSION_REASON: Final = "ended by a new session"
NO_FINAL_FRAME_REASON: Final = "finished without a final frame"
CHANGED_CANVAS_REASON: Final = "final canvas is missing or changed"


class ExportError(SketchLoopError, RuntimeError):
    pass


class ExportStatus(StrEnum):
    EXPORTED = "exported"
    MISSING = "missing"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExportEntry:
    code: str | None  # Random code naming the exported file, None when missing
    experiment: str  # Experiment name, which is the session folder's name
    drawing: int  # Drawing number in the session
    condition: StudyCondition  # Condition recorded on the drawing
    status: ExportStatus  # exported or missing
    reason: str | None  # Why it is missing, None when exported
    final_capture: int | None  # Number of the final capture, None when there is none
    source: str | None  # Final canvas path relative to the runs folder, None when there is none
    source_sha256: str | None  # Recorded checksum of the final canvas, None when there is none
    exported_sha256: str | None  # Checksum of the exported PNG, None when missing


@dataclass(frozen=True, slots=True, kw_only=True)
class FoundStudyDrawings:
    drawings: tuple[tuple[SavedSession, DrawingRecord], ...]  # Study drawings of matching experiments, by experiment and drawing
    skipped: tuple[str, ...]  # One line per matching runs-folder entry that could not be loaded
    passed_over: int  # Matching runs-folder entries with no study drawing, such as time-named sessions or older drawings


@dataclass(frozen=True, slots=True, kw_only=True)
class StudyExport:
    folder: Path  # The export folder that was created
    entries: tuple[ExportEntry, ...]  # One per study drawing found, sorted by experiment and drawing
    skipped: tuple[str, ...]  # One line per runs-folder entry that could not be loaded
    passed_over: int  # Matching runs-folder entries with no study drawing


def make_export_code() -> str:
    # Draw each character from the operating system's random source, never from a name, time, or order.
    return "".join(secrets.choice(EXPORT_CODE_ALPHABET) for _ in range(EXPORT_CODE_LENGTH))


def matches_experiment(name: str, match: str) -> bool:
    # Compare ignoring case, as experiment names are claimed ignoring case.
    return fnmatch.fnmatchcase(name.lower(), match.lower())


def describe_load_error(error: Exception) -> str:
    # An OSError's text names the absolute path, so only its reason is kept.
    if isinstance(error, OSError):
        return error.strerror or type(error).__name__

    return redact_paths(first_line(error))


def find_study_drawings(runs_dir: Path, match: str = "*") -> FoundStudyDrawings:
    # A missing runs folder is a mistyped path, not an empty study.
    if not runs_dir.is_dir():
        raise ExportError(f"Runs folder {runs_dir.as_posix()} does not exist. Give the folder the study saved into with --runs-dir.")  # noqa: E501

    found, skipped, passed_over = [], [], 0
    for entry in sorted(runs_dir.iterdir()):
        # Look only at folders whose name, which is the experiment name, matches.
        if not entry.is_dir() or not matches_experiment(entry.name, match):
            continue

        # A folder without session.json is not a session, such as an earlier export placed there by hand.
        if not (entry / SESSION_FILE).is_file():
            passed_over += 1
            continue

        # Report a session that fails to load, and go on with the others.
        try:
            saved = load_session(entry)
        except (ExperimentRecordError, OSError) as error:
            skipped.append(f"{entry.name}: {describe_load_error(error)}")
            continue

        # Only named experiments hold study drawings, and time-named sessions or T09 drawings are passed over.
        study_drawings = [drawing for drawing in saved.drawings if drawing.study is not None]
        if saved.session.name is None or not study_drawings:
            passed_over += 1
            continue

        found += [(saved, drawing) for drawing in study_drawings]

    found.sort(key=lambda pair: (pair[0].session.name, pair[1].number))
    return FoundStudyDrawings(drawings=tuple(found), skipped=tuple(skipped), passed_over=passed_over)


def find_missing_reason(record: DrawingRecord) -> str | None:
    # Say why a drawing has no final canvas from its record alone, or None when the record names one.
    if record.end is None:
        return NOT_FINISHED_REASON

    if record.end == DrawingEnd.NEW_SESSION:
        return NEW_SESSION_REASON

    if record.final_capture is None:
        return NO_FINAL_FRAME_REASON

    final = record.captures[record.final_capture - 1]
    return f"final frame has no canvas: {final.failure.message}" if final.canvas is None else None


def plan_export_entry(saved: SavedSession, record: DrawingRecord) -> tuple[ExportEntry, bytes | None]:
    experiment, condition = saved.session.name, record.study.condition
    reason = find_missing_reason(record)
    final = record.captures[record.final_capture - 1] if record.final_capture else None
    canvas = final.canvas if final else None
    source = f"{saved.folder.name}/drawing-{record.number}/{canvas.path}" if canvas else None
    entry = ExportEntry(code=None, experiment=experiment, drawing=record.number, condition=condition, status=ExportStatus.MISSING,
                        reason=reason, final_capture=record.final_capture, source=source,
                        source_sha256=canvas.sha256 if canvas else None, exported_sha256=None)
    if reason is not None:
        return entry, None

    # Read the canvas only from inside its drawing folder, and only when it still matches its recorded checksum.
    drawing_folder = saved.folder / f"drawing-{record.number}"
    target = drawing_folder / canvas.path
    inside = target.resolve().is_relative_to(drawing_folder.resolve())
    payload = target.read_bytes() if inside and target.is_file() else None
    if payload is None or hashlib.sha256(payload).hexdigest() != canvas.sha256:
        return replace(entry, reason=CHANGED_CANVAS_REASON), None

    return replace(entry, status=ExportStatus.EXPORTED), payload


def check_output_folder(runs_dir: Path, output: Path) -> None:
    # An export never replaces or mixes with another one.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ExportError(f"Export folder {output.as_posix()} is not empty. Choose a new folder, since an export never replaces another.")  # noqa: E501

    # A folder in the runs folder would look like a session and block that experiment name.
    if output.resolve().is_relative_to(runs_dir.resolve()):
        raise ExportError(f"Export folder {output.as_posix()} is inside the runs folder, where it would look like a session. Choose a folder outside it, such as one under data/.")  # noqa: E501


def draw_unique_code(make_code: Callable[[], str], used: set[str]) -> str:
    # Draw again on a collision within this export, and refuse a code that could not name a file safely.
    for _ in range(MAX_CODE_ATTEMPTS):
        code = make_code()
        valid = isinstance(code, str) and len(code) == EXPORT_CODE_LENGTH and all(char in EXPORT_CODE_ALPHABET for char in code)
        if not valid:
            raise ExportError(f"Export code {code!r} is not {EXPORT_CODE_LENGTH} characters of the code alphabet. Use make_export_code.")  # noqa: E501

        if code not in used:
            used.add(code)
            return code

    raise ExportError(f"No unused export code after {MAX_CODE_ATTEMPTS} attempts. Use make_export_code, which draws random codes.")  # noqa: E501


def reencode_png(payload: bytes) -> bytes:
    # Decode the pixels and encode them again, so no chunk of the recorded file, such as text or a time, is carried over.
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("the final canvas cannot be decoded")

    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise ValueError("the final canvas cannot be encoded as PNG")

    return buffer.tobytes()


def entry_to_row(entry: ExportEntry) -> list[str]:
    # Write every value as text and a missing value as an empty cell.
    values = (entry.code, entry.experiment, entry.drawing, entry.condition, entry.status, entry.reason, entry.final_capture,
              entry.source, entry.source_sha256, entry.exported_sha256)
    return ["" if value is None else str(value) for value in values]


def write_key_file(path: Path, entries: tuple[ExportEntry, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(KEY_COLUMNS)
        writer.writerows(entry_to_row(entry) for entry in entries)


def build_export_record(entries: tuple[ExportEntry, ...], found: FoundStudyDrawings, match: str) -> dict[str, object]:
    exported = sum(entry.status == ExportStatus.EXPORTED for entry in entries)
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sketchloop_version": encode_value(read_package_version("sketchloop-ai")),
        "image": EXPORTED_IMAGE,
        "match": match,
        "exported": exported,
        "missing": len(entries) - exported,
        "passed_over": found.passed_over,
        "skipped": list(found.skipped),
    }


def write_export(folder: Path, entries: tuple[ExportEntry, ...], payloads: dict[str, bytes], record: dict[str, object]) -> None:
    # Write the drawings in code order, so the file order carries nothing of the experiments, then the key and the record.
    (folder / DRAWINGS_FOLDER).mkdir()
    for code in sorted(payloads):
        (folder / DRAWINGS_FOLDER / f"{code}.png").write_bytes(payloads[code])

    write_key_file(folder / KEY_FILE, entries)
    (folder / EXPORT_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")


def describe_write_error(step: str, error: Exception) -> str:
    # An OSError's text names the absolute path, so only its reason is kept.
    reason = describe_load_error(error)
    return f"Cannot write the export while {step}: {reason}. Check free disk space and write access to the export folder, then retry."  # noqa: E501


def code_and_encode(planned: list[tuple[ExportEntry, bytes | None]],
                    make_code: Callable[[], str]) -> tuple[tuple[ExportEntry, ...], dict[str, bytes]]:
    # Give each exportable drawing a fresh code and its re-encoded PNG, keeping missing ones as planned.
    entries, payloads, used = [], {}, set()
    for entry, canvas in planned:
        if canvas is None:
            entries.append(entry)
            continue

        code = draw_unique_code(make_code, used)
        payloads[code] = reencode_png(canvas)
        entries.append(replace(entry, code=code, exported_sha256=hashlib.sha256(payloads[code]).hexdigest()))

    return tuple(entries), payloads


def export_study_drawings(runs_dir: Path, output: Path, *, match: str = "*",
                          make_code: Callable[[], str] = make_export_code) -> StudyExport:
    # Refuse an unusable output folder and an export with nothing to hand over, before anything is written.
    found = find_study_drawings(runs_dir, match)
    check_output_folder(runs_dir, output)
    if not found.drawings:
        raise ExportError(f"No study drawings in {runs_dir.as_posix()} match {match!r}. Check --runs-dir and --match.")

    planned = [plan_export_entry(saved, record) for saved, record in found.drawings]
    if all(canvas is None for _, canvas in planned):
        missing = "; ".join(f"{entry.experiment} drawing {entry.drawing}: {entry.reason}" for entry, _ in planned)
        raise ExportError(f"No study drawing has a usable final canvas: {missing}.")

    # Write into a temporary sibling folder and rename it at the end, so a partial export never looks complete.
    temporary = output.with_name(f"{output.name}.partial-{secrets.token_hex(4)}")
    step = "re-encoding the drawings"
    try:
        entries, payloads = code_and_encode(planned, make_code)
        step = "creating the folder"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        step = "writing the files"
        write_export(temporary, entries, payloads, build_export_record(entries, found, match))

        # Replace an empty output folder, which Windows cannot rename onto.
        step = "renaming the folder"
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except BaseException as error:
        # Remove the partial folder, and turn an unexpected failure into one line.
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, ExportError) or not isinstance(error, OSError | ValueError | cv2.error):
            raise

        raise ExportError(describe_write_error(step, error)) from error

    return StudyExport(folder=output, entries=entries, skipped=found.skipped, passed_over=found.passed_over)
