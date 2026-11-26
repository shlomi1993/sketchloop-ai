"""
Tests of the evaluator registry, which builds evaluators from the whole settings so a model evaluator finds its weights.
"""

import pytest
import re

from pathlib import Path

from sketchloop.backends.diffusers import DiffusersSketchGenerator, ModelSource
from sketchloop.backends.t2i_adapter import T2IAdapterSketchGenerator
from sketchloop.components import (EVALUATOR_MODELS, PROMPT_SUGGESTERS, check_component_names, find_recorded_evaluator,
                                   make_evaluator, make_generator, make_preprocessor, make_prompt_suggester,
                                   make_stroke_suggester)
from sketchloop.evaluators.clip import CLIP_B32_MODEL, CLIP_L14_MODEL, ClipEvaluator
from sketchloop.preprocessing import MinimalPreprocessor
from sketchloop.settings import (ControlNetSuggesterSettings, DiffusersSettings, EvaluatorSettings, GeneratorSettings,
                                 ModelsSettings, PreprocessingSettings, PromptSuggesterSettings, Settings, SettingsError,
                                 StrokeSuggesterSettings)
from sketchloop.stroke_suggesters.controlnet import ControlNetStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggestionError
from sketchloop.suggestion import FLORENCE_MODEL, FlorenceSuggester, SuggestionError, TemplateSuggester


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


def test_minimal_preprocessor_registered() -> None:
    """
    The minimal preprocessor name in the settings builds the minimal preprocessor.
    """
    preprocessor = make_preprocessor(PreprocessingSettings(name="minimal"))
    assert isinstance(preprocessor, MinimalPreprocessor), f"minimal built {type(preprocessor).__name__}"


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
    ("ClipEvaluator", {"model": "someone/other-clip"}, None),
    ("ClipEvaluator", {}, None),
    (None, {}, None)
], ids=["none", "clip-b32", "unknown-clip", "clip-without-model", "unrecorded"])
def test_find_recorded_evaluator(evaluator: str | None, score_details: dict[str, str], expected: str | None) -> None:
    """
    A recorded evaluator maps back to its registry name, CLIP by its recorded repository, and anything else to None.
    """
    found = find_recorded_evaluator(evaluator, score_details)
    assert found == expected, f"Expected {expected} for {evaluator} with {score_details}, got {found}"


def test_stroke_suggester_registry_builds_the_controlnet_suggester(tmp_path: Path) -> None:
    """
    Naming controlnet builds the real suggester with the models folder and precision from the settings, loading nothing.
    """
    options = StrokeSuggesterSettings(name="controlnet", controlnet=ControlNetSuggesterSettings(precision="float32"))
    settings = Settings(models=ModelsSettings(dir=tmp_path), stroke_suggester=options)
    suggester = make_stroke_suggester(settings)

    # The suggester runs in the configured precision.
    assert isinstance(suggester, ControlNetStrokeSuggester), f"Expected the ControlNet suggester, got {type(suggester).__name__}"
    assert suggester.precision == "float32", f"Expected float32, got {suggester.precision}"

    # It looks for the weights in the configured folder, which is empty here.
    with pytest.raises(StrokeSuggestionError, match=re.escape(f"{tmp_path.as_posix()}/stable-diffusion-v1-5 is missing")):
        suggester.load(512, 360)


@pytest.mark.parametrize(("settings", "message"), [
    (Settings(evaluator=EvaluatorSettings(name="clip")), 'evaluator.name must be one of none, clip-b32, or clip-l14, got "clip"'),
    (Settings(stroke_suggester=StrokeSuggesterSettings(controlnet=ControlNetSuggesterSettings(precision="fp16"))),
     'stroke_suggester.controlnet.precision must be one of float16 or float32, got "fp16"'),
    (Settings(prompt_suggester=PromptSuggesterSettings(name="llava")),
     'prompt_suggester.name must be one of templates or florence, got "llava"')
], ids=["unknown-evaluator", "unknown-precision", "unknown-prompt-suggester"])
def test_unknown_component_settings_refused(settings: Settings, message: str) -> None:
    """
    An unknown component name or suggester precision is refused with the key and every choice.
    """
    with pytest.raises(SettingsError, match=re.escape(message)):
        check_component_names(settings, "sketchloop.toml")


def test_prompt_suggester_registry(tmp_path: Path) -> None:
    """
    The templates are the default prompt suggester, and florence reads its weights from the models folder.
    """
    default = make_prompt_suggester(Settings())
    florence = make_prompt_suggester(Settings(models=ModelsSettings(dir=tmp_path),
                                              prompt_suggester=PromptSuggesterSettings(name="florence")))

    # The templates need no model and are the default, and florence is built with the fake image backend, loading nothing.
    assert isinstance(default, TemplateSuggester), f"Expected the templates by default, got {type(default).__name__}"
    assert isinstance(florence, FlorenceSuggester), f"Expected Florence-2, got {type(florence).__name__}"
    assert set(PROMPT_SUGGESTERS) == {"templates", "florence"}, f"Unexpected prompt suggesters {sorted(PROMPT_SUGGESTERS)}"

    # Florence-2 looks for its weights in the configured models folder, which is empty here.
    with pytest.raises(SuggestionError, match=re.escape(f"{(tmp_path / FLORENCE_MODEL.folder).as_posix()} is missing")):
        florence.warm_up()
