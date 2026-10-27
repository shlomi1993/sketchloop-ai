"""
Factories that build the capture, preprocessing, evaluation, and storage components the settings name.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, TypeAlias

from sketchloop.backends import GenerationBackend
from sketchloop.backends.diffusers import GenerationMode
from sketchloop.capture import CameraCapture, CaptureSource, FileCapture
from sketchloop.evaluation import Evaluator, NoOpEvaluator
from sketchloop.experiments import ExperimentStore, FolderStore
from sketchloop.preprocessing import Preprocessor, StrokePreprocessor
from sketchloop.settings import (CaptureSettings, EvaluatorSettings, PreprocessingSettings, Settings, SettingsError,
                                 StorageSettings, join_choices)


NO_RUN_SKETCH_MESSAGE: Final = 'give a sketch file or --camera, or set capture.source = "camera" in the settings file'

CaptureFactory: TypeAlias = Callable[[CaptureSettings, Path | None], CaptureSource]


def make_file_capture(settings: CaptureSettings, sketch_file: Path | None) -> CaptureSource:
    """
    Create a source that reads the sketch file given on the command line.

    Args:
        settings (CaptureSettings): Capture settings, unused by the file source.
        sketch_file (Path | None): Sketch file from the command line, or None when none was given.

    Returns:
        CaptureSource: The file source.
    """
    if sketch_file is None:
        raise SettingsError(NO_RUN_SKETCH_MESSAGE)

    return FileCapture(sketch_file)


def make_camera_capture(settings: CaptureSettings, sketch_file: Path | None) -> CaptureSource:
    """
    Create a source that captures from the configured webcam, ignoring any sketch file.
    """
    return CameraCapture(settings.camera_index)


def make_stroke_preprocessor(settings: PreprocessingSettings) -> Preprocessor:
    """
    Create the default stroke preprocessor.
    """
    return StrokePreprocessor()


def make_no_op_evaluator(settings: EvaluatorSettings) -> Evaluator:
    """
    Create the evaluator that gives no scores.
    """
    return NoOpEvaluator()


def make_folder_store(settings: StorageSettings) -> ExperimentStore:
    """
    Create the folder store under the configured runs folder.
    """
    return FolderStore(settings.runs_dir)


# Components by the name a settings file uses. Tests and library callers may add entries, and nothing is discovered.
CAPTURE_SOURCES: Final[dict[str, CaptureFactory]] = {"file": make_file_capture, "camera": make_camera_capture}
PREPROCESSORS: Final[dict[str, Callable[[PreprocessingSettings], Preprocessor]]] = {"strokes": make_stroke_preprocessor}
EVALUATORS: Final[dict[str, Callable[[EvaluatorSettings], Evaluator]]] = {"none": make_no_op_evaluator}
STORES: Final[dict[str, Callable[[StorageSettings], ExperimentStore]]] = {"folder": make_folder_store}


def make_capture_source(settings: CaptureSettings, sketch_file: Path | None) -> CaptureSource:
    """
    Create the configured capture source.

    Args:
        settings (CaptureSettings): Capture settings naming the source.
        sketch_file (Path | None): Sketch file from the command line, or None.

    Returns:
        CaptureSource: The capture source.
    """
    return CAPTURE_SOURCES[settings.source](settings, sketch_file)


def make_preprocessor(settings: PreprocessingSettings) -> Preprocessor:
    """
    Create the configured preprocessor.
    """
    return PREPROCESSORS[settings.name](settings)


def make_evaluator(settings: EvaluatorSettings) -> Evaluator:
    """
    Create the configured evaluator.
    """
    return EVALUATORS[settings.name](settings)


def make_store(settings: StorageSettings) -> ExperimentStore:
    """
    Create the configured experiment store.
    """
    return STORES[settings.store](settings)


def check_component_names(settings: Settings, label: str) -> None:
    """
    Refuse a backend, mode, or component name that no registry knows, naming the first bad key.

    Args:
        settings (Settings): Settings to check.
        label (str): Settings file name used in messages.
    """
    choices_by_key: dict[str, tuple[str | None, Mapping[str, object] | list[str]]] = {
        "generator.backend": (settings.generator.backend, list(GenerationBackend)),
        "generator.diffusers.mode": (settings.generator.diffusers.mode, list(GenerationMode)),
        "capture.source": (settings.capture.source, CAPTURE_SOURCES),
        "preprocessing.name": (settings.preprocessing.name, PREPROCESSORS),
        "evaluator.name": (settings.evaluator.name, EVALUATORS),
        "storage.store": (settings.storage.store, STORES),
    }

    # An unset mode means the device default, so only a set one is checked.
    for key, (value, choices) in choices_by_key.items():
        if value is not None and value not in choices:
            raise SettingsError(f'{label}: {key} must be one of {join_choices(list(choices))}, got "{value}".')
