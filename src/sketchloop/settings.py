"""
Loading and validation of the optional sketchloop.toml settings file.
"""

import math
import tomllib

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Final, TypeVar, get_args

from sketchloop.domain import Rotation, SketchLoopError
from sketchloop.stroke_suggestion import CANVAS_GRID_PIXELS


SETTINGS_FILE_NAME: Final = "sketchloop.toml"
Record = TypeVar("Record")  # A settings dataclass, replaced by a copy of the same type

# Smallest accepted value of each integer key that has one.
MINIMUMS: Final[dict[str, int]] = {
    "capture.camera_index": 0,
    "run.candidates": 1,
    "run.seed": 0,
    "ui.candidates": 1,
    "ui.seed": 0,
}

# Paper canvas sides of guided sketching, which must fit the model's 8-pixel latent grid.
CANVAS_SIDE_KEYS: Final = ("drawing.canvas_width", "drawing.canvas_height")
CANVAS_SIDE_RANGE_PIXELS: Final = (256, 1024)

# Captures are saved less often than webcam frames arrive, which is every 0.5 s by default.
MIN_SAVE_INTERVAL_SECONDS: Final = 0.5


class SettingsError(SketchLoopError, ValueError):
    """
    A settings file or a combination of settings is invalid.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusersSettings:
    """
    Options shared by the diffusers and t2i-adapter backends.
    """
    mode: str | None = None  # fast or quality, None for the device default
    precision: str | None = None  # float16 or float32, None for float16 on a GPU and float32 on CPU


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorSettings:
    """
    The generation backend and its options.
    """
    backend: str = "fake"  # Registry name of the generation backend
    diffusers: DiffusersSettings = DiffusersSettings()  # Options shared by the diffusers and t2i-adapter backends


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelsSettings:
    """
    Where every downloaded model lives, shared by the backends, evaluators, and prompt suggester.
    """
    dir: Path = Path("models")  # Folder holding the downloaded model folders


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureSettings:
    """
    The image source for sketchloop dev run.
    """
    source: str = "file"  # Registry name of the capture source for sketchloop dev run
    camera_index: int = 0  # Webcam index for the camera source


@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessingSettings:
    """
    The preprocessor and the rotation and raw defaults of sketchloop dev run.
    """
    name: str = "strokes"  # Registry name of the preprocessor
    rotation: int = Rotation.DEG_000  # Clockwise degrees before preprocessing, for sketchloop dev run
    raw: bool = False  # Generate from the raw sketch and skip preprocessing, for sketchloop dev run


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluatorSettings:
    """
    The candidate scorer.
    """
    name: str = "none"  # Registry name of the evaluator


@dataclass(frozen=True, slots=True, kw_only=True)
class StorageSettings:
    """
    The experiment store, its folder, and where the blinded export writes.
    """
    store: str = "folder"  # Registry name of the experiment store
    runs_dir: Path = Path("runs")  # Folder for saved sessions
    export_dir: Path = Path("data/study-exports")  # Folder where sketchloop study export creates a new export folder


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSettings:
    """
    Defaults for sketchloop dev run.
    """
    candidates: int = 4  # Candidates per round for sketchloop dev run
    seed: int | None = None  # Base seed, None for random


@dataclass(frozen=True, slots=True, kw_only=True)
class UiSettings:
    """
    Defaults for sketchloop start.
    """
    candidates: int = 1  # Default candidates per round on the page
    seed: int | None = None  # Default base seed, None for random
    capture_interval_seconds: float = 0.5  # Seconds between kept webcam frames


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlNetSuggesterSettings:
    """
    Options of the ControlNet next-stroke suggester.
    """
    precision: str | None = None  # float16 or float32, None for float16 on a GPU and float32 on CPU


@dataclass(frozen=True, slots=True, kw_only=True)
class StrokeSuggesterSettings:
    """
    The next-stroke suggester and its options.
    """
    name: str = "fake"  # Registry name of the next-stroke suggester
    controlnet: ControlNetSuggesterSettings = ControlNetSuggesterSettings()  # Options of the ControlNet suggester


@dataclass(frozen=True, slots=True, kw_only=True)
class DrawingSettings:
    """
    The paper canvas and capture interval of guided sketching.
    """
    canvas_width: int = 512  # Paper frame width in pixels, a multiple of 8
    canvas_height: int = 360  # Paper frame height in pixels, so 512x360 is A4 or Letter in landscape on the latent grid
    save_interval_seconds: float = 2.0  # Seconds between saved captures while a drawing is open


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSuggesterSettings:
    """
    The prompt suggester behind "Suggest a prompt", chosen independently of the image backend.
    """
    name: str = "templates"  # Registry name of the prompt suggester


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """
    Every component choice and default, from built-in values, a settings file, and command-line flags.
    """
    generator: GeneratorSettings = GeneratorSettings()  # Generation backend and its options
    models: ModelsSettings = ModelsSettings()  # Folder holding every downloaded model
    capture: CaptureSettings = CaptureSettings()  # Image source for sketchloop dev run
    preprocessing: PreprocessingSettings = PreprocessingSettings()  # Preprocessor and run defaults
    evaluator: EvaluatorSettings = EvaluatorSettings()  # Candidate scorer
    storage: StorageSettings = StorageSettings()  # Experiment store and its folder
    run: RunSettings = RunSettings()  # Defaults for sketchloop dev run
    ui: UiSettings = UiSettings()  # Defaults for sketchloop start
    stroke_suggester: StrokeSuggesterSettings = StrokeSuggesterSettings()  # Next-stroke suggester and its options
    drawing: DrawingSettings = DrawingSettings()  # Paper canvas and capture interval of guided sketching
    prompt_suggester: PromptSuggesterSettings = PromptSuggesterSettings()  # Prompt suggester behind "Suggest a prompt"


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


def describe_type(value_type: type) -> str:
    """
    Name the kind of value a field type takes, for a wrong-type message.
    """
    names = {bool: "true or false", int: "an integer", float: "a number", Path: "non-empty text", str: "text"}
    return names[value_type]


def find_value_type(field_type: object) -> type:
    """
    Return the type a field holds, ignoring None in an optional field such as int | None.
    """
    return next(value_type for value_type in (get_args(field_type) or (field_type,)) if value_type is not type(None))


def check_range(key: str, value: object, label: str) -> None:
    """
    Check a value against the limits of its key, for the keys that have them.

    Args:
        key (str): Dotted key, such as run.candidates.
        value (object): Value read from the file, already of the right type.
        label (str): Settings file name used in messages.
    """
    if key in MINIMUMS and value < MINIMUMS[key]:
        raise SettingsError(f"{label}: {key} must be at least {MINIMUMS[key]}, got {value}.")

    if key == "preprocessing.rotation" and value not in tuple(Rotation):
        raise SettingsError(f"{label}: {key} must be one of {join_choices(tuple(Rotation))}, got {value}.")

    if key == "ui.capture_interval_seconds" and not (value > 0 and math.isfinite(value)):
        raise SettingsError(f"{label}: {key} must be greater than 0, got {value}.")

    smallest, largest = CANVAS_SIDE_RANGE_PIXELS
    grid = CANVAS_GRID_PIXELS
    if key in CANVAS_SIDE_KEYS and not (smallest <= value <= largest and value % grid == 0):
        raise SettingsError(f"{label}: {key} must be a multiple of {grid} from {smallest} to {largest}, got {value}.")

    if key == "drawing.save_interval_seconds" and not (value > MIN_SAVE_INTERVAL_SECONDS and math.isfinite(value)):
        raise SettingsError(f"{label}: {key} must be greater than {MIN_SAVE_INTERVAL_SECONDS}, got {value}.")


def convert_value(key: str, value: object, value_type: type, folder: Path, label: str) -> object:
    """
    Check one value's type and range, and convert it to the field's type.

    Args:
        key (str): Dotted key, such as run.candidates.
        value (object): Value read from the file.
        value_type (type): Type the field holds.
        folder (Path): Folder of the settings file, which relative paths are resolved against.
        label (str): Settings file name used in messages.

    Returns:
        object: The value as the field holds it.
    """
    # Refuse the wrong kind of value, since TOML booleans are not integers and paths must not be empty.
    is_integer = isinstance(value, int) and not isinstance(value, bool)
    type_matches = {
        bool: isinstance(value, bool),
        int: is_integer,
        float: is_integer or isinstance(value, float),
        Path: isinstance(value, str) and bool(value.strip()),
        str: isinstance(value, str),
    }
    if not type_matches[value_type]:
        raise SettingsError(f"{label}: {key} must be {describe_type(value_type)}, got {format_toml_value(value)}.")

    check_range(key, value, label)

    # Resolve a relative folder against the file's folder, so the file works from any working folder.
    if value_type is Path:
        return folder / value

    return float(value) if value_type is float else value


def build_record(record_type: type[Record], table: object, prefix: str, folder: Path, label: str) -> Record:
    """
    Build a settings record from a TOML table, checking every key against the record's fields.

    Args:
        record_type (type[Record]): Settings dataclass to build.
        table (object): Parsed TOML value, which must be a table.
        prefix (str): Dotted name of the table, such as generator.
        folder (Path): Folder of the settings file.
        label (str): Settings file name used in messages.

    Returns:
        Record: The record, with defaults for every key the table leaves out.
    """
    if not isinstance(table, dict):
        raise SettingsError(f"{label}: {prefix} must be a table such as [{prefix}], got {format_toml_value(table)}.")

    field_types = {field.name: field.type for field in fields(record_type)}
    values: dict[str, object] = {}
    for name, value in table.items():
        key = f"{prefix}.{name}" if prefix else name

        # Refuse a misspelled section or key, listing the ones this table accepts.
        if name not in field_types and not prefix:
            raise SettingsError(f"{label}: unknown section [{name}]. Use {join_choices(list(field_types))}.")

        if name not in field_types:
            raise SettingsError(f"{label}: unknown key {key}. Keys in [{prefix}]: {', '.join(field_types)}.")

        # Build a nested table, such as [generator.diffusers], as its own record.
        value_type = find_value_type(field_types[name])
        if is_dataclass(value_type):
            values[name] = build_record(value_type, value, key, folder, label)
        else:
            values[name] = convert_value(key, value, value_type, folder, label)

    return record_type(**values)


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

    label = path.as_posix()

    # Parse the file, naming the TOML error, which includes the line and column.
    try:
        with path.open("rb") as file:
            document = tomllib.load(file)
    except OSError as error:
        raise SettingsError(f"Cannot read settings file {label}: {error.strerror}. Check the file and retry.") from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise SettingsError(f"{label} is not valid TOML: {error}. Fix the file and retry.") from error

    return build_record(Settings, document, "", path.parent, label)


def replace_field(record: Record, parts: Sequence[str], value: object) -> Record:
    """
    Return a copy of a settings record with the value at the dotted path replaced.
    """
    if len(parts) == 1:
        return replace(record, **{parts[0]: value})

    return replace(record, **{parts[0]: replace_field(getattr(record, parts[0]), parts[1:], value)})


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
        settings = replace_field(settings, key.split("."), value)

    return settings
