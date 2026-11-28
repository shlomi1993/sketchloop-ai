"""
Tests of the component registries, which build components from the whole settings so model components find their weights.
"""

import pytest
import re

from pathlib import Path

from sketchloop.backends.diffusers import DiffusersSketchGenerator
from sketchloop.backends.t2i_adapter import T2IAdapterSketchGenerator
from sketchloop.components import (EVALUATOR_MODELS, PROMPT_SUGGESTERS, RERUNNABLE_BACKENDS, check_component_names,
                                   find_recorded_evaluator, make_evaluator, make_generator, make_preprocessor,
                                   make_prompt_suggester, make_stroke_suggester)
from sketchloop.evaluators.clip import CLIP_B32_MODEL, CLIP_L14_MODEL, ClipEvaluator
from sketchloop.preprocessing import MinimalPreprocessor
from sketchloop.settings import (ControlNetSuggesterSettings, DiffusersSettings, EvaluatorSettings, GeneratorSettings,
                                 ModelsSettings, PreprocessingSettings, PromptSuggesterSettings, Settings, SettingsError,
                                 StrokeSuggesterSettings)
from sketchloop.stroke_suggesters.controlnet import ControlNetStrokeSuggester
from sketchloop.stroke_suggestion import StrokeSuggestionError
from sketchloop.suggestion import FLORENCE_MODEL, FlorenceSuggester, SuggestionError, TemplateSuggester


def test_registries_build_components_without_loading(tmp_path: Path) -> None:
    """
    Each registered evaluator, preprocessor, and real backend is built from the settings with the models folder, loading nothing.
    """
    models = ModelsSettings(dir=tmp_path)
    b32 = make_evaluator(Settings(models=models, evaluator=EvaluatorSettings(name="clip-b32")))
    l14 = make_evaluator(Settings(models=models, evaluator=EvaluatorSettings(name="clip-l14")))
    diffusers = DiffusersSettings(mode="quality", precision="float32")
    generators = [make_generator(Settings(generator=GeneratorSettings(backend=backend, diffusers=diffusers), models=models))
                  for backend in ("diffusers", "t2i-adapter")]

    # Each CLIP name builds the CLIP evaluator for its model in the models folder, which the download list also names.
    clip = [(type(evaluator), evaluator.model, evaluator.model_folder) for evaluator in (b32, l14)]
    expected_clip = [(ClipEvaluator, model, tmp_path / model.folder) for model in (CLIP_B32_MODEL, CLIP_L14_MODEL)]
    assert clip == expected_clip, f"Unexpected CLIP evaluators {clip}"
    assert (EVALUATOR_MODELS["clip-b32"], EVALUATOR_MODELS["clip-l14"]) == (CLIP_B32_MODEL, CLIP_L14_MODEL), "Bad download list"

    # The minimal preprocessor name builds the minimal preprocessor.
    preprocessor = make_preprocessor(PreprocessingSettings(name="minimal"))
    assert isinstance(preprocessor, MinimalPreprocessor), f"minimal built {type(preprocessor).__name__}"

    # Both real backends read the [models] folder and share the diffusers options.
    built = [(type(generator), generator._model_dir, generator.mode, generator.precision) for generator in generators]
    expected = [(kind, tmp_path, "quality", "float32") for kind in (DiffusersSketchGenerator, T2IAdapterSketchGenerator)]
    assert built == expected, f"Unexpected real backends {built}"

    # Rerun recreates each real backend from its own adapter name.
    adapter_name = T2IAdapterSketchGenerator.ADAPTER_NAME
    assert adapter_name != DiffusersSketchGenerator.ADAPTER_NAME, "The two real backends need distinct adapter names"
    assert RERUNNABLE_BACKENDS[adapter_name] == "t2i-adapter", "Rerun must map the adapter name to t2i-adapter"


@pytest.mark.parametrize(("evaluator", "score_details", "expected"), [
    ("ClipEvaluator", {"model": CLIP_B32_MODEL.repo_id, "revision": "older"}, "clip-b32"),
    ("ClipEvaluator", {"model": "someone/other-clip"}, None),
], ids=["clip-b32", "unknown-clip"])
def test_find_recorded_evaluator(evaluator: str | None, score_details: dict[str, str], expected: str | None) -> None:
    """
    A recorded evaluator maps back to its registry name, CLIP by its recorded repository, and anything else to None.
    """
    found = find_recorded_evaluator(evaluator, score_details)
    assert found == expected, f"Expected {expected} for {evaluator} with {score_details}, got {found}"


@pytest.mark.parametrize(("settings", "message"), [
    (Settings(evaluator=EvaluatorSettings(name="clip")), 'evaluator.name must be one of none, clip-b32, or clip-l14, got "clip"'),
    (Settings(stroke_suggester=StrokeSuggesterSettings(controlnet=ControlNetSuggesterSettings(precision="fp16"))),
     'stroke_suggester.controlnet.precision must be one of float16 or float32, got "fp16"'),
], ids=["unknown-evaluator", "unknown-precision"])
def test_unknown_component_settings_refused(settings: Settings, message: str) -> None:
    """
    An unknown component name or suggester precision is refused with the key and every choice.
    """
    with pytest.raises(SettingsError, match=re.escape(message)):
        check_component_names(settings, "sketchloop.toml")


def test_suggester_registries_read_the_models_folder(tmp_path: Path) -> None:
    """
    The ControlNet and Florence-2 suggesters are built in the configured precision and look for weights in the models folder.
    """
    models = ModelsSettings(dir=tmp_path)
    options = StrokeSuggesterSettings(name="controlnet", controlnet=ControlNetSuggesterSettings(precision="float32"))
    stroke_suggester = make_stroke_suggester(Settings(models=models, stroke_suggester=options))
    default = make_prompt_suggester(Settings())
    florence = make_prompt_suggester(Settings(models=models, prompt_suggester=PromptSuggesterSettings(name="florence")))

    # The stroke suggester runs in the configured precision, the templates are the default, and florence is registered.
    assert isinstance(stroke_suggester, ControlNetStrokeSuggester), f"Got {type(stroke_suggester).__name__}"
    assert stroke_suggester.precision == "float32", f"Expected float32, got {stroke_suggester.precision}"
    assert isinstance(default, TemplateSuggester), f"Expected the templates by default, got {type(default).__name__}"
    assert isinstance(florence, FlorenceSuggester), f"Expected Florence-2, got {type(florence).__name__}"
    assert set(PROMPT_SUGGESTERS) == {"templates", "florence"}, f"Unexpected prompt suggesters {sorted(PROMPT_SUGGESTERS)}"

    # Both look for their weights in the configured models folder, which is empty here.
    with pytest.raises(StrokeSuggestionError, match=re.escape(f"{tmp_path.as_posix()}/stable-diffusion-v1-5 is missing")):
        stroke_suggester.load(512, 360)
    with pytest.raises(SuggestionError, match=re.escape(f"{(tmp_path / FLORENCE_MODEL.folder).as_posix()} is missing")):
        florence.warm_up()
