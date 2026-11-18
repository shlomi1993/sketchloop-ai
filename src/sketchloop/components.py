"""
Factories that build the capture, preprocessing, evaluation, and storage components the settings name.
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
from sketchloop.settings import (CaptureSettings, GeneratorSettings, PreprocessingSettings, Settings, SettingsError,
                                 StorageSettings, join_choices)


FAKE_BACKEND: Final = "fake"
NO_RUN_SKETCH_MESSAGE: Final = 'give a sketch file or --camera, or set capture.source = "camera" in the settings file'

GeneratorFactory: TypeAlias = Callable[[GeneratorSettings], Generator]
CaptureFactory: TypeAlias = Callable[[CaptureSettings, Path | None], CaptureSource]
PreprocessorFactory: TypeAlias = Callable[[PreprocessingSettings], Preprocessor]
EvaluatorFactory: TypeAlias = Callable[[Settings], Evaluator]
StoreFactory: TypeAlias = Callable[[StorageSettings], ExperimentStore]


def make_diffusers_generator(settings: GeneratorSettings, generator_class: type[DiffusersSketchGenerator]) -> Generator:
    """
    Create an SD 1.5 backend of the given class with the configured mode, precision, and models folder.

    Args:
        settings (GeneratorSettings): Generator settings whose diffusers options apply.
        generator_class (type[DiffusersSketchGenerator]): Backend class to create, such as the ControlNet or adapter one.

    Returns:
        Generator: The backend, with no model loaded yet.
    """
    options = settings.diffusers
    return generator_class(options.models_dir, mode=GenerationMode(options.mode) if options.mode else None,
                           precision=TorchPrecision(options.precision) if options.precision else None)


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
    "diffusers": lambda settings: make_diffusers_generator(settings, DiffusersSketchGenerator),
    "t2i-adapter": lambda settings: make_diffusers_generator(settings, T2IAdapterSketchGenerator),
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
    "clip-b32": lambda settings: ClipEvaluator(CLIP_B32_MODEL, settings.generator.diffusers.models_dir),
    "clip-l14": lambda settings: ClipEvaluator(CLIP_L14_MODEL, settings.generator.diffusers.models_dir),
}
STORES: Final[dict[str, StoreFactory]] = {
    "folder": lambda settings: FolderStore(settings.runs_dir)
}


# Weights by evaluator name, which scripts/download_models.py fetches only for the evaluators asked for.
EVALUATOR_MODELS: Final[dict[str, ModelSource]] = {
    "clip-b32": CLIP_B32_MODEL,
    "clip-l14": CLIP_L14_MODEL,
}

# Backend names by the adapter name a round recorded, so rerun can recreate the backend.
RERUNNABLE_BACKENDS: Final[dict[str, str]] = {
    FakeGenerator.ADAPTER_NAME: FAKE_BACKEND,
    DiffusersSketchGenerator.ADAPTER_NAME: "diffusers",
    T2IAdapterSketchGenerator.ADAPTER_NAME: "t2i-adapter",
}


def make_generator(settings: GeneratorSettings) -> Generator:
    """
    Create the configured generation backend, without loading any model yet.
    """
    return GENERATORS[settings.backend](settings)


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
    Create the configured evaluator, which reads the models folder from the generator settings, without loading any model.
    """
    return EVALUATORS[settings.evaluator.name](settings)


def make_store(settings: StorageSettings) -> ExperimentStore:
    """
    Create the configured experiment store.
    """
    return STORES[settings.store](settings)


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
    }

    # An unset mode or precision means the device default, so only a set one is checked.
    for key, (value, choices) in choices_by_key.items():
        if value is not None and value not in choices:
            raise SettingsError(f'{label}: {key} must be one of {join_choices(list(choices))}, got "{value}".')
