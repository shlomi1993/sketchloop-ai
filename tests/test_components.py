"""
Tests of the evaluator registry, which builds evaluators from the whole settings so a model evaluator finds its weights.
"""

import pytest

from pathlib import Path

from sketchloop.backends.diffusers import ModelSource
from sketchloop.components import EVALUATOR_MODELS, EVALUATORS, check_component_names, make_evaluator
from sketchloop.evaluation import NoOpEvaluator
from sketchloop.evaluators.clip import CLIP_B32_MODEL, CLIP_L14_MODEL, ClipEvaluator
from sketchloop.settings import DiffusersSettings, EvaluatorSettings, GeneratorSettings, Settings, SettingsError


def make_settings(evaluator_name: str, models_dir: Path) -> Settings:
    """
    Build settings naming an evaluator and a models folder.
    """
    generator = GeneratorSettings(diffusers=DiffusersSettings(models_dir=models_dir))
    return Settings(generator=generator, evaluator=EvaluatorSettings(name=evaluator_name))


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
