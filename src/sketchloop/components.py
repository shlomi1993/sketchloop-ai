"""
Factories that build the capture, preprocessing, evaluation, storage, and stroke suggestion components the settings name.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, TypeAlias

from sketchloop.backends.diffusers import DiffusersSketchGenerator, GenerationMode, ModelSource, TorchPrecision
from sketchloop.backends.fake import FakeGenerator
from sketchloop.backends.t2i_adapter import T2IAdapterSketchGenerator
from sketchloop.capture import CameraCapture, CaptureSource, FileCapture
from sketchloop.evaluation import Evaluator, NoOpEvaluator
from sketchloop.evaluators.clip import CLIP_B32_MODEL, CLIP_L14_MODEL, ClipEvaluator
from sketchloop.experiments import ExperimentStore, FolderStore
from sketchloop.generation import Generator
from sketchloop.preprocessing import MinimalPreprocessor, Preprocessor, StrokePreprocessor
from sketchloop.settings import (CaptureSettings, PreprocessingSettings, Settings, SettingsError, StorageSettings,
                                 join_choices)
from sketchloop.stroke_suggesters.controlnet import ControlNetStrokeSuggester
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggester


FAKE_BACKEND: Final = "fake"
NO_RUN_SKETCH_MESSAGE: Final = 'give a sketch file or --camera, or set capture.source = "camera" in the settings file'

GeneratorFactory: TypeAlias = Callable[[Settings], Generator]
CaptureFactory: TypeAlias = Callable[[CaptureSettings, Path | None], CaptureSource]
PreprocessorFactory: TypeAlias = Callable[[PreprocessingSettings], Preprocessor]
EvaluatorFactory: TypeAlias = Callable[[Settings], Evaluator]
StoreFactory: TypeAlias = Callable[[StorageSettings], ExperimentStore]
StrokeSuggesterFactory: TypeAlias = Callable[[Settings], StrokeSuggester]


def make_diffusers_generator(settings: Settings, generator_class: type[DiffusersSketchGenerator]) -> Generator:
    """
    Create an SD 1.5 backend of the given class with the configured mode, precision, and models folder.

    Args:
        settings (Settings): Settings whose generator.diffusers options and models folder apply.
        generator_class (type[DiffusersSketchGenerator]): Backend class to create, such as the ControlNet or adapter one.

    Returns:
        Generator: The backend, with no model loaded yet.
    """
    options = settings.generator.diffusers
    return generator_class(settings.models.dir, mode=GenerationMode(options.mode) if options.mode else None,
                           precision=TorchPrecision(options.precision) if options.precision else None)


def make_controlnet_stroke_suggester(settings: Settings) -> StrokeSuggester:
    precision = settings.stroke_suggester.controlnet.precision
    return ControlNetStrokeSuggester(settings.models.dir, precision=TorchPrecision(precision) if precision else None)


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


# Components by the name a settings file uses. Tests and library callers may add entries, and nothing is discovered.
GENERATORS: Final[dict[str, GeneratorFactory]] = {
    FAKE_BACKEND: lambda settings: FakeGenerator(),
    DiffusersSketchGenerator.BACKEND_NAME: lambda settings: make_diffusers_generator(settings, DiffusersSketchGenerator),
    T2IAdapterSketchGenerator.BACKEND_NAME: lambda settings: make_diffusers_generator(settings, T2IAdapterSketchGenerator),
}
CAPTURE_SOURCES: Final[dict[str, CaptureFactory]] = {
    "file": make_file_capture,
    "camera": lambda settings, sketch_file: CameraCapture(settings.camera_index),
}
PREPROCESSORS: Final[dict[str, PreprocessorFactory]] = {
    "strokes": lambda settings: StrokePreprocessor(),
    "minimal": lambda settings: MinimalPreprocessor(),
}
EVALUATORS: Final[dict[str, EvaluatorFactory]] = {
    "none": lambda settings: NoOpEvaluator(),
    "clip-b32": lambda settings: ClipEvaluator(CLIP_B32_MODEL, settings.models.dir),
    "clip-l14": lambda settings: ClipEvaluator(CLIP_L14_MODEL, settings.models.dir),
}
STORES: Final[dict[str, StoreFactory]] = {
    "folder": lambda settings: FolderStore(settings.runs_dir)
}
STROKE_SUGGESTERS: Final[dict[str, StrokeSuggesterFactory]] = {
    "fake": lambda settings: FakeStrokeSuggester(),
    "controlnet": make_controlnet_stroke_suggester,
}


# Weights by evaluator name, which scripts/download_models.py fetches only for the evaluators asked for.
EVALUATOR_MODELS: Final[dict[str, ModelSource]] = {
    "clip-b32": CLIP_B32_MODEL,
    "clip-l14": CLIP_L14_MODEL,
}

# Backend names by the adapter name a round recorded, so rerun can recreate the backend.
RERUNNABLE_BACKENDS: Final[dict[str, str]] = {
    FakeGenerator.ADAPTER_NAME: FAKE_BACKEND,
    DiffusersSketchGenerator.ADAPTER_NAME: DiffusersSketchGenerator.BACKEND_NAME,
    T2IAdapterSketchGenerator.ADAPTER_NAME: T2IAdapterSketchGenerator.BACKEND_NAME,
}


def find_recorded_evaluator(evaluator: str | None, score_details: Mapping[str, str]) -> str | None:
    """
    Find the registry name of the evaluator a round recorded, so rerun can score with the same one.

    Args:
        evaluator (str | None): Evaluator class name the round recorded.
        score_details (Mapping[str, str]): What the round recorded about its scores, such as the CLIP model.

    Returns:
        str | None: The evaluator's registry name, or None when the record does not name a known one.
    """
    if evaluator == NoOpEvaluator.__name__:
        return "none"

    # Tell the CLIP models apart by the repository the round recorded, since they share one class.
    if evaluator == ClipEvaluator.__name__:
        return next((name for name, model in EVALUATOR_MODELS.items() if model.repo_id == score_details.get("model")), None)

    return None


def make_generator(settings: Settings) -> Generator:
    """
    Create the configured generation backend, which reads the models folder from the settings, without loading any model.
    """
    return GENERATORS[settings.generator.backend](settings)


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


def make_evaluator(settings: Settings) -> Evaluator:
    """
    Create the configured evaluator, which reads the models folder from the settings, without loading any model.
    """
    return EVALUATORS[settings.evaluator.name](settings)


def make_store(settings: StorageSettings) -> ExperimentStore:
    """
    Create the configured experiment store.
    """
    return STORES[settings.store](settings)


def make_stroke_suggester(settings: Settings) -> StrokeSuggester:
    """
    Create the configured next-stroke suggester, which reads the models folder from the settings, without loading any model.
    """
    return STROKE_SUGGESTERS[settings.stroke_suggester.name](settings)


def check_component_names(settings: Settings, label: str) -> None:
    """
    Refuse a backend, mode, precision, or component name that no registry knows, naming the first bad key.

    Args:
        settings (Settings): Settings to check.
        label (str): Settings file name used in messages.
    """
    choices_by_key: dict[str, tuple[str | None, Mapping[str, object] | list[str]]] = {
        "generator.backend": (settings.generator.backend, GENERATORS),
        "generator.diffusers.mode": (settings.generator.diffusers.mode, list(GenerationMode)),
        "generator.diffusers.precision": (settings.generator.diffusers.precision, list(TorchPrecision)),
        "capture.source": (settings.capture.source, CAPTURE_SOURCES),
        "preprocessing.name": (settings.preprocessing.name, PREPROCESSORS),
        "evaluator.name": (settings.evaluator.name, EVALUATORS),
        "storage.store": (settings.storage.store, STORES),
        "stroke_suggester.name": (settings.stroke_suggester.name, STROKE_SUGGESTERS),
        "stroke_suggester.controlnet.precision": (settings.stroke_suggester.controlnet.precision, list(TorchPrecision)),
    }

    # An unset mode or precision means the device default, so only a set one is checked.
    for key, (value, choices) in choices_by_key.items():
        if value is not None and value not in choices:
            raise SettingsError(f'{label}: {key} must be one of {join_choices(list(choices))}, got "{value}".')
