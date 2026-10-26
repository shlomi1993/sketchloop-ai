import json
import os

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from sketchloop.domain import first_line
from sketchloop.experiments.records import ROUND_FILE, SESSION_FILE, RoundRecord, SessionRecord, StoreError
from sketchloop.experiments.serialization import round_to_dict, session_to_dict


def write_json_atomic(path: Path, data: Mapping[str, object]) -> None:
    """
    Write JSON through a temporary file in the same folder, so a crash never leaves a half-written record.

    Args:
        path (Path): Target file.
        data (Mapping[str, object]): JSON-ready data.
    """
    # Flush to disk before the rename, so the replaced file is complete even after a power loss.
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)
    except BaseException:
        # Remove the partial temporary file, so a failed save leaves only the previous record.
        temporary.unlink(missing_ok=True)
        raise


def write_files(folder: Path, files: Mapping[str, bytes]) -> None:
    """
    Write each payload to its relative path under the folder, creating subfolders as needed.

    Args:
        folder (Path): Folder the relative paths resolve against.
        files (Mapping[str, bytes]): Payload by relative POSIX path.
    """
    # Create each file's subfolder before writing it.
    for path, payload in files.items():
        target_file = folder / path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(payload)


class ExperimentStore(ABC):
    """
    Write side of experiment storage: where sessions live, and saving session and round records with their files.
    """

    @property
    @abstractmethod
    def root(self) -> Path | None:
        """
        Return the local folder that holds all saved sessions, or None when files are not kept on disk.
        """
        ...

    @abstractmethod
    def locate_session(self, session_id: str) -> Path | None:
        """
        Return the session's folder, which may not exist yet, or None when files are not kept on disk.
        """
        ...

    @abstractmethod
    def locate_round(self, session_id: str, number: int) -> Path | None:
        """
        Return the round's folder, or None when files are not kept on disk.
        """
        ...

    @abstractmethod
    def save_session(self, record: SessionRecord) -> None:
        """
        Create the session and save its record, raising StoreError when it cannot.
        """
        ...

    @abstractmethod
    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        Save the files, then the record, replacing an earlier record of the same round number.

        Args:
            session_id (str): Session the round belongs to.
            record (RoundRecord): Round record to save.
            files (Mapping[str, bytes]): Payload by relative POSIX path, empty to replace only the record.
        """
        ...


class FolderStore(ExperimentStore):
    """
    Store that keeps each session in a folder under the runs folder, with session.json and round-<n>/round.json.
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir

    @property
    def root(self) -> Path | None:
        """
        Return the runs folder.
        """
        return self.runs_dir

    def locate_session(self, session_id: str) -> Path | None:
        """
        Return the session's folder under the runs folder.
        """
        return self.runs_dir / session_id

    def locate_round(self, session_id: str, number: int) -> Path | None:
        """
        Return the round's folder inside its session folder.
        """
        return self.runs_dir / session_id / f"round-{number}"

    @contextmanager
    def _report_failure(self, target: Path) -> Iterator[None]:
        # Name the file relative to the runs folder, which keeps user names out of the message.
        try:
            yield
        except OSError as error:
            label = target.relative_to(self.runs_dir).as_posix()
            reason = error.strerror or type(error).__name__
            message = f"Cannot save {label}: {reason}. Free disk space or check write access to the runs folder, then retry."
            raise StoreError(message) from error
        except (TypeError, ValueError) as error:
            label = target.relative_to(self.runs_dir).as_posix()
            raise StoreError(f"Cannot save {label}: {first_line(error)}. Check the record values, then retry.") from error

    def save_session(self, record: SessionRecord) -> None:
        """
        Create the session folder and atomically write session.json.

        Args:
            record (SessionRecord): Session record to save.
        """
        folder = self.runs_dir / record.session_id
        with self._report_failure(folder / SESSION_FILE):
            folder.mkdir(parents=True, exist_ok=True)
            write_json_atomic(folder / SESSION_FILE, session_to_dict(record))

    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        Write the round's files, then atomically write round.json, so the record never names an unsaved file.

        Args:
            session_id (str): Session the round belongs to.
            record (RoundRecord): Round record to save.
            files (Mapping[str, bytes]): Payload by relative POSIX path, empty to replace only the record.
        """
        # Write the files one at a time, so a failure names the file that could not be saved.
        round_folder = self.runs_dir / session_id / f"round-{record.number}"
        for path, payload in files.items():
            with self._report_failure(round_folder / path):
                write_files(round_folder, {path: payload})

        # Write the record last.
        with self._report_failure(round_folder / ROUND_FILE):
            round_folder.mkdir(parents=True, exist_ok=True)
            write_json_atomic(round_folder / ROUND_FILE, round_to_dict(record))


class MemoryStore(ExperimentStore):
    """
    Store that keeps records and file bytes in process memory, for tests and library callers that need no files.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, SessionRecord] = {}
        self.rounds: dict[tuple[str, int], RoundRecord] = {}
        self.files: dict[tuple[str, int, str], bytes] = {}

    @property
    def root(self) -> Path | None:
        """
        Return None, since no files are kept on disk.
        """
        return None

    def locate_session(self, session_id: str) -> Path | None:
        """
        Return None, since sessions have no folder.
        """
        return None

    def locate_round(self, session_id: str, number: int) -> Path | None:
        """
        Return None, since rounds have no folder.
        """
        return None

    def _check_encodable(self, label: str, encode: Callable[[], Mapping[str, object]]) -> None:
        # Encode as the folder store would, discarding the text, so a record it could not write fails here too.
        try:
            json.dumps(encode(), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise StoreError(f"Cannot keep {label}: {first_line(error)}. Check the record values, then retry.") from error

    def save_session(self, record: SessionRecord) -> None:
        """
        Keep the session record after checking that it encodes as JSON.

        Args:
            record (SessionRecord): Session record to keep.
        """
        self._check_encodable(f"{record.session_id}/{SESSION_FILE}", lambda: session_to_dict(record))
        self.sessions[record.session_id] = record

    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None:
        """
        Keep the round's files, then its record, replacing an earlier record of the same round number.

        Args:
            session_id (str): Session the round belongs to.
            record (RoundRecord): Round record to keep.
            files (Mapping[str, bytes]): Payload by relative POSIX path, empty to replace only the record.
        """
        self.files |= {(session_id, record.number, path): payload for path, payload in files.items()}
        self._check_encodable(f"{session_id}/round-{record.number}/{ROUND_FILE}", lambda: round_to_dict(record))
        self.rounds[(session_id, record.number)] = record
