"""
Loading and validation of the optional sketchloop.toml settings file.
"""

import math
import tomllib

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final

from sketchloop.domain import GenerationBackend, Rotation, SketchLoopError


SETTINGS_FILE_NAME: Final = "sketchloop.toml"
SECTIONS: Final = ("generator", "capture", "preprocessing", "evaluator", "storage", "run", "ui")


class SettingKind(StrEnum):
    """
    Kind of TOML value a settings key takes.
    """
    TEXT = "text"
    PATH = "path"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


# How a wrong-type message names each kind of value.
KIND_DESCRIPTIONS: Final[dict[SettingKind, str]] = {
    SettingKind.TEXT: "text",
    SettingKind.PATH: "non-empty text",
    SettingKind.INTEGER: "an integer",
    SettingKind.NUMBER: "a number",
    SettingKind.BOOLEAN: "true or false",
}


class SettingsError(SketchLoopError, ValueError):
    """
    A settings file or a combination of settings is invalid.
    """


@dataclass(frozen=True, slots=True)
class SettingRule:
    """
    The type and allowed values of one key in the settings file.
    """
    kind: SettingKind  # Kind of TOML value the key takes
    minimum: int | None = None  # Smallest accepted integer, None for no limit
    choices: tuple[int, ...] = ()  # Accepted integers, empty for any


# Every key the settings file accepts, by dotted name. Component names are checked against the registries later.
RULES: Final[dict[str, SettingRule]] = {
    "generator.backend": SettingRule(SettingKind.TEXT),
    "generator.diffusers.mode": SettingRule(SettingKind.TEXT),
    "generator.diffusers.models_dir": SettingRule(SettingKind.PATH),
    "capture.source": SettingRule(SettingKind.TEXT),
    "capture.camera_index": SettingRule(SettingKind.INTEGER, minimum=0),
    "preprocessing.name": SettingRule(SettingKind.TEXT),
    "preprocessing.rotation": SettingRule(SettingKind.INTEGER, choices=tuple(Rotation)),
    "preprocessing.raw": SettingRule(SettingKind.BOOLEAN),
    "evaluator.name": SettingRule(SettingKind.TEXT),
    "storage.store": SettingRule(SettingKind.TEXT),
    "storage.runs_dir": SettingRule(SettingKind.PATH),
    "run.candidates": SettingRule(SettingKind.INTEGER, minimum=1),
    "run.seed": SettingRule(SettingKind.INTEGER, minimum=0),
    "ui.candidates": SettingRule(SettingKind.INTEGER, minimum=1),
    "ui.seed": SettingRule(SettingKind.INTEGER, minimum=0),
    "ui.capture_interval_seconds": SettingRule(SettingKind.NUMBER),
}

# Tables nested inside a section, such as [generator.diffusers].
SUBTABLES: Final = ("generator.diffusers",)


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusersSettings:
    """
    Options used only by the diffusers backend.
    """
    mode: str | None = None  # fast or quality, None for the device default
    models_dir: Path = Path("models")  # Folder holding the downloaded model folders


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorSettings:
    """
    The generation backend and its options.
    """
    backend: str = GenerationBackend.FAKE  # Registry name of the generation backend
    diffusers: DiffusersSettings = DiffusersSettings()  # Options used only by the diffusers backend


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureSettings:
    """
    The image source for sketchloop run.
    """
    source: str = "file"  # Registry name of the capture source for sketchloop run
    camera_index: int = 0  # Webcam index for the camera source


@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessingSettings:
    """
    The preprocessor and the rotation and raw defaults of sketchloop run.
    """
    name: str = "strokes"  # Registry name of the preprocessor
    rotation: int = Rotation.DEG_000  # Clockwise degrees before preprocessing, for sketchloop run
    raw: bool = False  # Generate from the raw sketch and skip preprocessing, for sketchloop run


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluatorSettings:
    """
    The candidate scorer.
    """
    name: str = "none"  # Registry name of the evaluator


@dataclass(frozen=True, slots=True, kw_only=True)
class StorageSettings:
    """
    The experiment store and its folder.
    """
    store: str = "folder"  # Registry name of the experiment store
    runs_dir: Path = Path("runs")  # Folder for saved sessions


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSettings:
    """
    Defaults for sketchloop run.
    """
    candidates: int = 4  # Candidates per round for sketchloop run
    seed: int | None = None  # Base seed, None for random


@dataclass(frozen=True, slots=True, kw_only=True)
class UiSettings:
    """
    Defaults for sketchloop ui.
    """
    candidates: int = 2  # Default candidates per round on the page
    seed: int | None = None  # Default base seed, None for random
    capture_interval_seconds: float = 0.5  # Seconds between kept webcam frames


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """
    Every component choice and default, from built-in values, a settings file, and command-line flags.
    """
    generator: GeneratorSettings = GeneratorSettings()  # Generation backend and its options
    capture: CaptureSettings = CaptureSettings()  # Image source for sketchloop run
    preprocessing: PreprocessingSettings = PreprocessingSettings()  # Preprocessor and run defaults
    evaluator: EvaluatorSettings = EvaluatorSettings()  # Candidate scorer
    storage: StorageSettings = StorageSettings()  # Experiment store and its folder
    run: RunSettings = RunSettings()  # Defaults for sketchloop run
    ui: UiSettings = UiSettings()  # Defaults for sketchloop ui


def join_choices(choices: Sequence[object]) -> str:
    """
    Join choices for a message, such as "fake or diffusers" or "a, b, or c".
    """
    words = [str(choice) for choice in choices]
    return " or ".join(words) if len(words) < 3 else f"{', '.join(words[:-1])}, or {words[-1]}"


def format_toml_value(value: object) -> str:
    """
    Show a value as it would appear in TOML, quoting text and writing booleans in lower case.
    """
    if isinstance(value, bool):
        return str(value).lower()

    return f'"{value}"' if isinstance(value, str) else str(value)


def find_settings_file(explicit: Path | None) -> Path | None:
    """
    Choose the settings file: the explicit one, else sketchloop.toml in the working folder when present, else none.

    Args:
        explicit (Path | None): File named by --config, or None.

    Returns:
        Path | None: The settings file to read, or None for the built-in defaults.
    """
    # A file named explicitly must exist, since silently using defaults would hide a typo.
    if explicit is not None:
        if not explicit.is_file():
            raise SettingsError(f"Settings file {explicit.as_posix()} not found. Check the --config path.")

        return explicit

    default_file = Path(SETTINGS_FILE_NAME)
    return default_file if default_file.is_file() else None


def list_table_keys(table: str) -> list[str]:
    """
    List the keys and subtables a table accepts, in schema order.
    """
    names = [key.removeprefix(f"{table}.").split(".")[0] for key in RULES if key.startswith(f"{table}.")]
    return list(dict.fromkeys(names))


def check_value(key: str, value: object, label: str) -> None:
    """
    Check one value's type and range against its rule.

    Args:
        key (str): Dotted key, such as run.candidates.
        value (object): Value read from the file.
        label (str): Settings file name used in messages.
    """
    rule = RULES[key]
    is_integer = isinstance(value, int) and not isinstance(value, bool)
    type_checks = {
        SettingKind.TEXT: isinstance(value, str),
        SettingKind.PATH: isinstance(value, str) and bool(value.strip()),
        SettingKind.INTEGER: is_integer,
        SettingKind.NUMBER: is_integer or isinstance(value, float),
        SettingKind.BOOLEAN: isinstance(value, bool),
    }

    # Refuse the wrong kind of value, since TOML booleans and integers are distinct.
    if not type_checks[rule.kind]:
        raise SettingsError(f"{label}: {key} must be {KIND_DESCRIPTIONS[rule.kind]}, got {format_toml_value(value)}.")

    # Refuse a value outside the allowed set or range.
    if rule.choices and value not in rule.choices:
        raise SettingsError(f"{label}: {key} must be one of {join_choices(rule.choices)}, got {format_toml_value(value)}.")

    if rule.minimum is not None and value < rule.minimum:
        raise SettingsError(f"{label}: {key} must be at least {rule.minimum}, got {value}.")

    if rule.kind == SettingKind.NUMBER and not (value > 0 and math.isfinite(value)):
        raise SettingsError(f"{label}: {key} must be greater than 0, got {value}.")


def flatten_table(table: Mapping[str, object], prefix: str, label: str) -> dict[str, object]:
    """
    Check a table's keys and values and return them by dotted key, descending into subtables.

    Args:
        table (Mapping[str, object]): Parsed TOML table.
        prefix (str): Dotted name of the table, such as generator.
        label (str): Settings file name used in messages.

    Returns:
        dict[str, object]: Checked values by dotted key.
    """
    values: dict[str, object] = {}
    for name, value in table.items():
        key = f"{prefix}.{name}"

        # Descend into a known subtable, which must be a table itself.
        if key in SUBTABLES:
            if not isinstance(value, dict):
                raise SettingsError(f"{label}: {key} must be a table such as [{key}], got {format_toml_value(value)}.")

            values |= flatten_table(value, key, label)
            continue

        # Refuse a misspelled key, listing the ones this table accepts.
        if key not in RULES:
            raise SettingsError(f"{label}: unknown key {key}. Keys in [{prefix}]: {', '.join(list_table_keys(prefix))}.")

        check_value(key, value, label)
        values[key] = value

    return values


def read_settings_values(path: Path) -> dict[str, object]:
    """
    Read a settings file and return its checked values by dotted key.

    Args:
        path (Path): Settings file.

    Returns:
        dict[str, object]: Checked values by dotted key, with paths as written in the file.
    """
    label = path.as_posix()

    # Parse the file, naming the TOML error, which includes the line and column.
    try:
        with path.open("rb") as file:
            document = tomllib.load(file)
    except OSError as error:
        raise SettingsError(f"Cannot read settings file {label}: {error.strerror}. Check the file and retry.") from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise SettingsError(f"{label} is not valid TOML: {error}. Fix the file and retry.") from error

    # Check each section, which must be a known table.
    values: dict[str, object] = {}
    for section, table in document.items():
        # A known key written above every section belongs inside the table that accepts it, such as seed in [run].
        owner = next((key.rpartition(".")[0] for key in RULES if key.endswith(f".{section}")), None)
        if not isinstance(table, dict) and owner is not None:
            raise SettingsError(f"{label}: {section} must be inside a section such as [{owner}].")

        if section not in SECTIONS:
            raise SettingsError(f"{label}: unknown section [{section}]. Use {join_choices(SECTIONS)}.")

        if not isinstance(table, dict):
            raise SettingsError(f"{label}: {section} must be a table such as [{section}], got {format_toml_value(table)}.")

        values |= flatten_table(table, section, label)

    return values


def replace_setting(record: object, key: str, parts: Sequence[str], value: object) -> object:
    """
    Return a copy of a settings record with the value at the dotted path replaced.

    Args:
        record (object): Settings record at this level.
        key (str): Full dotted key, used in the error for an unknown key.
        parts (Sequence[str]): Remaining parts of the key.
        value (object): New value.

    Returns:
        object: The updated copy.
    """
    # An unknown key is a programming error, not a user error.
    name = parts[0]
    if not is_dataclass(record) or name not in {field.name for field in fields(record)}:
        raise KeyError(key)

    if len(parts) == 1:
        return replace(record, **{name: value})

    return replace(record, **{name: replace_setting(getattr(record, name), key, parts[1:], value)})


def override_settings(settings: Settings, overrides: Mapping[str, object]) -> Settings:
    """
    Return new settings with values replaced by dotted key, such as "run.candidates".

    Args:
        settings (Settings): Settings to start from.
        overrides (Mapping[str, object]): New value by dotted key.

    Returns:
        Settings: The updated settings.
    """
    for key, value in overrides.items():
        settings = replace_setting(settings, key, key.split("."), value)

    return settings


def load_settings(path: Path | None) -> Settings:
    """
    Read a settings file over the built-in defaults, resolving relative folders against the file's folder.

    Args:
        path (Path | None): Settings file, or None for the built-in defaults.

    Returns:
        Settings: The settings.
    """
    if path is None:
        return Settings()

    values = read_settings_values(path)
    for key, value in values.items():
        # Resolve relative folders against the file's folder, so the file works from any working folder.
        if RULES[key].kind == SettingKind.PATH:
            values[key] = path.parent / value
        elif RULES[key].kind == SettingKind.NUMBER:
            values[key] = float(value)

    return override_settings(Settings(), values)
