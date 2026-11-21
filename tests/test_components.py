"""
Tests of the evaluator registry, which builds evaluators from the whole settings so a model evaluator finds its weights.
"""

import pytest
import re

from pathlib import Path

from sketchloop.backends.diffusers import DiffusersSketchGenerator, ModelSource
from sketchloop.backends.t2i_adapter import T2IAdapterSketchGenerator
from sketchloop.components import (EVALUATOR_MODELS, EVALUATORS, STROKE_SUGGESTERS, check_component_names,
                                   find_recorded_evaluator, make_evaluator, make_generator, make_preprocessor,
                                   make_stroke_suggester)
from sketchloop.evaluation import NoOpEvaluator
from sketchloop.evaluators.clip import CLIP_B32_MODEL, CLIP_L14_MODEL, ClipEvaluator
from sketchloop.preprocessing import MinimalPreprocessor, StrokePreprocessor
from sketchloop.settings import (ControlNetSuggesterSettings, DiffusersSettings, EvaluatorSettings, GeneratorSettings,
                                 ModelsSettings, PreprocessingSettings, Settings, SettingsError, SuggesterSettings)
from sketchloop.stroke_suggesters.fake import FakeStrokeSuggester


def make_settings(evaluator_name: str, models_dir: Path) -> Settings:
    """
    Build settings naming an evaluator and a models folder.
    """
    return Settings(models=ModelsSettings(dir=models_dir), evaluator=EvaluatorSettings(name=evaluator_name))


@pytest.mark.parametrize(("name", "model"), [("clip-b32", CLIP_B32_MODEL), ("clip-l14", CLIP_L14_MODEL)])
def test_clip_evaluators_registered(tmp_path: Path, name: str, model: ModelSource) -> None:
    """
    Each CLIP name builds the CLIP evaluator for its model in the configured models folder, without loading it.
    """
    evaluator = make_evaluator(make_settings(name, tmp_path))
    assert isinstance(evaluator, ClipEvaluator), f"{name} must build a ClipEvaluator, got {type(evaluator).__name__}"
    assert evaluator.model == model, f"{name} must use {model}, got {evaluator.model}"
    assert evaluator.model_folder == tmp_path / model.folder, f"Expected the configured folder, got {evaluator.model_folder}"
    assert EVALUATOR_MODELS[name] == model, f"The download list must name {name}'s model"


def test_none_evaluator_still_default() -> None:
    """
    The default settings still build the no-op evaluator, and every evaluator with weights is registered by the same name.
    """
    assert isinstance(make_evaluator(Settings()), NoOpEvaluator), "The default evaluator must stay the no-op one"
    assert set(EVALUATOR_MODELS) <= set(EVALUATORS), f"Download names {set(EVALUATOR_MODELS)} must all be evaluators"


def test_unknown_evaluator_name_refused(tmp_path: Path) -> None:
    """
    A CLIP name passes the settings check, and an unknown name is refused with every choice listed.
    """
    check_component_names(make_settings("clip-l14", tmp_path), "sketchloop.toml")
    with pytest.raises(SettingsError, match='evaluator.name must be one of none, clip-b32, or clip-l14, got "clip"'):
        check_component_names(make_settings("clip", tmp_path), "sketchloop.toml")


@pytest.mark.parametrize(("name", "expected_type"), [("strokes", StrokePreprocessor), ("minimal", MinimalPreprocessor)])
def test_preprocessors_registered(name: str, expected_type: type) -> None:
    """
    Each preprocessor name in the settings builds its preprocessor.
    """
    preprocessor = make_preprocessor(PreprocessingSettings(name=name))
    assert isinstance(preprocessor, expected_type), f"{name} built {type(preprocessor).__name__}"



@pytest.mark.parametrize(("backend", "expected_type"), [
    ("diffusers", DiffusersSketchGenerator),
    ("t2i-adapter", T2IAdapterSketchGenerator)
])
def test_real_backends_read_models_dir(tmp_path: Path, backend: str, expected_type: type) -> None:
    """
    Both real backends read their weights from the [models] folder, without loading any model.
    """
    generator_settings = GeneratorSettings(backend=backend, diffusers=DiffusersSettings(precision="float32"))
    generator = make_generator(Settings(generator=generator_settings, models=ModelsSettings(dir=tmp_path)))
    assert isinstance(generator, expected_type), f"{backend} built {type(generator).__name__}"
    assert generator._model_dir == tmp_path, f"{backend} must read the [models] folder, got {generator._model_dir}"


@pytest.mark.parametrize(("evaluator", "score_details", "expected"), [
    ("NoOpEvaluator", {}, "none"),
    ("ClipEvaluator", {"model": CLIP_B32_MODEL.repo_id, "revision": "older"}, "clip-b32"),
    ("ClipEvaluator", {"model": CLIP_L14_MODEL.repo_id}, "clip-l14"),
    ("ClipEvaluator", {"model": "someone/other-clip"}, None),
    ("ClipEvaluator", {}, None),
    ("ConstantEvaluator", {}, None),
    (None, {}, None)
], ids=["none", "clip-b32", "clip-l14", "unknown-clip", "clip-without-model", "unknown-class", "unrecorded"])
def test_find_recorded_evaluator(evaluator: str | None, score_details: dict[str, str], expected: str | None) -> None:
    """
    A recorded evaluator maps back to its registry name, CLIP by its recorded repository, and anything else to None.
    """
    found = find_recorded_evaluator(evaluator, score_details)
    assert found == expected, f"Expected {expected} for {evaluator} with {score_details}, got {found}"


def test_stroke_suggester_registry_builds_the_fake() -> None:
    """
    The default settings build the fake stroke suggester, the only one registered so far.
    """
    suggester = make_stroke_suggester(Settings())
    assert isinstance(suggester, FakeStrokeSuggester), f"Expected the fake stroke suggester, got {type(suggester).__name__}"
    assert set(STROKE_SUGGESTERS) == {"fake"}, f"Unexpected stroke suggesters {sorted(STROKE_SUGGESTERS)}"


@pytest.mark.parametrize(("suggester", "message"), [
    (SuggesterSettings(name="magic"), 'suggester.name must be one of fake, got "magic"'),
    (SuggesterSettings(controlnet=ControlNetSuggesterSettings(precision="fp16")),
     'suggester.controlnet.precision must be one of float16 or float32, got "fp16"')
], ids=["unknown-name", "unknown-precision"])
def test_unknown_stroke_suggester_settings_refused(suggester: SuggesterSettings, message: str) -> None:
    """
    An unknown stroke suggester name or precision is refused with the key and every choice.
    """
    with pytest.raises(SettingsError, match=re.escape(message)):
        check_component_names(Settings(suggester=suggester), "sketchloop.toml")
