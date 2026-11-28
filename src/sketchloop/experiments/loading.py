"""
Loading of saved sessions from their folders, with checksum checks of every recorded file.
"""

import hashlib
import json
import re

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, TypeVar

from sketchloop.domain import ImageRef
from sketchloop.experiments.drawings import DRAWING_FILE, DrawingRecord, drawing_from_dict, list_drawing_images
from sketchloop.experiments.prompts import PromptOrigin, PromptSuggestionStatus
from sketchloop.experiments.records import ROUND_FILE, SESSION_FILE, ExperimentRecordError, RoundRecord, SavedSession
from sketchloop.experiments.serialization import round_from_dict, session_from_dict


ROUND_FOLDER_PATTERN: Final = re.compile(r"round-([1-9][0-9]*)")
DRAWING_FOLDER_PATTERN: Final = re.compile(r"drawing-([1-9][0-9]*)")


RecordT = TypeVar("RecordT")


def read_json(path: Path, label: str) -> Mapping[str, object]:
    """
    Read a JSON object from a record file.

    Args:
        path (Path): Record file.
        label (str): Name used in error messages, relative to the session folder.

    Returns:
        Mapping[str, object]: The parsed object.
    """
    # Name the decoding error, which includes the line of a syntax error, so the person can find it.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExperimentRecordError(f"{label} is not valid JSON: {error}. Restore the file or inspect it by hand.") from None

    if not isinstance(data, dict):
        raise ExperimentRecordError(f"{label}: the top level is invalid: expected an object.")

    return data


def convert_record(label: str, data: Mapping[str, object], convert: Callable[[Mapping[str, object]], RecordT]) -> RecordT:
    """
    Convert record data, turning any validation fault into one ExperimentRecordError that names the file.

    Args:
        label (str): File name used in error messages.
        data (Mapping[str, object]): Parsed record.
        convert (Callable[[Mapping[str, object]], RecordT]): From-dict function.

    Returns:
        RecordT: The converted record.
    """
    try:
        return convert(data)
    except ExperimentRecordError as error:
        raise ExperimentRecordError(f"{label} {error}") from None
    except KeyError as error:
        raise ExperimentRecordError(f"{label}: {error.args[0]} is invalid: it is missing.") from None
    except (AttributeError, TypeError, ValueError) as error:
        raise ExperimentRecordError(f"{label}: a field is invalid: {error}") from None


def list_numbered_folders(folder: Path, pattern: re.Pattern[str], kind: str) -> list[tuple[int, Path]]:
    """
    List the numbered folders a pattern matches, by number, refusing gaps in the numbering.

    Args:
        folder (Path): Session folder.
        pattern (re.Pattern[str]): Folder name pattern whose first group is the number.
        kind (str): Folder kind for the error message, such as Round.

    Returns:
        list[tuple[int, Path]]: Numbers and folders, in order.
    """
    numbered = []
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_dir():
            numbered.append((int(match.group(1)), path))

    # Require 1 to N, since a missing folder means the records were edited or mixed.
    numbered.sort()
    numbers = [number for number, _ in numbered]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ExperimentRecordError(f"{kind} folders {numbers} have a gap. The records were edited or mixed.")

    return numbered


def list_round_folders(folder: Path) -> list[tuple[int, Path]]:
    """
    List round-<n> folders by number, refusing gaps in the numbering.
    """
    return list_numbered_folders(folder, ROUND_FOLDER_PATTERN, "Round")


def list_round_images(record: RoundRecord) -> list[ImageRef]:
    """
    List the raw sketch, the generation sketch, and the candidates a round recorded, without duplicates.
    """
    images = [record.raw_sketch] + ([record.request.sketch] if record.request else [])
    images += [candidate.image for candidate in record.iteration.result.candidates] if record.iteration else []
    return list({image.path: image for image in images}.values())


def verify_image_files(images: list[ImageRef], folder: Path, record_file: str) -> tuple[list[str], int]:
    """
    Compare each recorded image file in a round or drawing folder with its checksum.

    Args:
        images (list[ImageRef]): Images the record names, relative to the folder.
        folder (Path): Folder holding the files.
        record_file (str): Name of the record file, for messages.

    Returns:
        tuple[list[str], int]: Problems found, and the number of files that matched.
    """
    problems, matched = [], 0
    resolved_folder = folder.resolve()
    for image in images:
        # Refuse a link that resolves outside the folder, such as a planted symlink.
        target = folder / image.path
        if not target.resolve().is_relative_to(resolved_folder):
            raise ExperimentRecordError(f"{folder.name}/{record_file} references {image.path} outside its folder. Refusing to load it.")  # noqa: E501

        # List a missing or changed file as a problem, so the rest of the record can still be shown.
        label = f"{folder.name}/{image.path}"
        if not target.is_file():
            problems.append(f"{label} is missing.")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != image.sha256:
            problems.append(f"{label} does not match its recorded checksum.")
        else:
            matched += 1

    return problems, matched


def verify_round_files(record: RoundRecord, round_folder: Path) -> tuple[list[str], int]:
    """
    Compare each image file a round recorded with its checksum, returning the problems and the number that matched.
    """
    return verify_image_files(list_round_images(record), round_folder, ROUND_FILE)


def load_rounds(folder: Path) -> tuple[list[RoundRecord], list[str], int]:
    """
    Load and check every round of a session folder.

    Args:
        folder (Path): Session folder.

    Returns:
        tuple[list[RoundRecord], list[str], int]: The rounds, the artifact problems, and the number of matching files.
    """
    rounds, problems, matched = [], [], 0
    parent_id = None
    for number, round_folder in list_round_folders(folder):
        # A round folder without round.json was interrupted before its record was written.
        label = f"{round_folder.name}/{ROUND_FILE}"
        if not (round_folder / ROUND_FILE).is_file():
            problems.append(f"{round_folder.name} has no {ROUND_FILE}, so the round was interrupted.")
            continue

        # Check the number and the link to the previous complete round.
        record = convert_record(label, read_json(round_folder / ROUND_FILE, label), round_from_dict)
        if record.number != number:
            raise ExperimentRecordError(f"{label}: round is invalid: {record.number} does not match its folder.")
        if record.iteration is not None and record.iteration.parent_id != parent_id:
            raise ExperimentRecordError(f"round-{number} does not link to round-{number - 1}. The records were edited or mixed.")

        # Verify the files, then remember this round as the parent of the next complete one.
        round_problems, round_matched = verify_round_files(record, round_folder)
        problems += round_problems
        matched += round_matched
        parent_id = record.iteration.id if record.iteration else parent_id
        rounds.append(record)

    return rounds, problems, matched


def load_drawings(folder: Path) -> tuple[list[DrawingRecord], list[str], int]:
    """
    Load and check every drawing of a session folder.

    Args:
        folder (Path): Session folder.

    Returns:
        tuple[list[DrawingRecord], list[str], int]: The drawings, the artifact problems, and the number of matching files.
    """
    drawings, problems, matched = [], [], 0
    for number, drawing_folder in list_numbered_folders(folder, DRAWING_FOLDER_PATTERN, "Drawing"):
        # A drawing folder without drawing.json was interrupted before its first record was written.
        label = f"{drawing_folder.name}/{DRAWING_FILE}"
        if not (drawing_folder / DRAWING_FILE).is_file():
            problems.append(f"{drawing_folder.name} has no {DRAWING_FILE}, so the drawing was interrupted.")
            continue

        # Check the number, then every recorded file.
        record = convert_record(label, read_json(drawing_folder / DRAWING_FILE, label), drawing_from_dict)
        if record.number != number:
            raise ExperimentRecordError(f"{label}: drawing is invalid: {record.number} does not match its folder.")

        drawing_problems, drawing_matched = verify_image_files(list_drawing_images(record), drawing_folder, DRAWING_FILE)
        problems += drawing_problems
        matched += drawing_matched
        drawings.append(record)

    return drawings, problems, matched


def find_unknown_prompt_origins(rounds: list[RoundRecord], drawings: list[DrawingRecord]) -> list[str]:
    # A round's origin may name a suggestion of any round or drawing of the session, so it is checked against all of them.
    suggestions = [item for record in [*rounds, *drawings] for item in record.prompt_suggestions or ()]
    known = {item.id for item in suggestions if item.status == PromptSuggestionStatus.SHOWN}
    problems = []
    for record in rounds:
        origin = record.prompt_origin
        if isinstance(origin, PromptOrigin) and origin.prompt_suggestion is not None and origin.prompt_suggestion not in known:
            problems.append(f"round-{record.number} prompt origin names prompt suggestion {origin.prompt_suggestion}, which is not in this session.")  # noqa: E501

    return problems


def load_session(folder: Path) -> SavedSession:
    """
    Load a session folder's records, failing on structural faults and listing artifact problems.

    Args:
        folder (Path): Folder containing persisted session records.

    Returns:
        SavedSession: The session, its rounds and drawings, and any problems found.
    """
    if not (folder / SESSION_FILE).is_file():
        raise ExperimentRecordError(f"No {SESSION_FILE} in {folder.as_posix()}. Use a folder from sketchloop dev run.")

    # Load the rounds, then any drawings, which sessions saved before T09 do not have.
    session = convert_record(SESSION_FILE, read_json(folder / SESSION_FILE, SESSION_FILE), session_from_dict)
    rounds, round_problems, round_matched = load_rounds(folder)
    drawings, drawing_problems, drawing_matched = load_drawings(folder)
    problems = round_problems + drawing_problems + find_unknown_prompt_origins(rounds, drawings)
    return SavedSession(folder=folder, session=session, rounds=tuple(rounds), problems=tuple(problems),
                        checked_files=round_matched + drawing_matched, drawings=tuple(drawings))
