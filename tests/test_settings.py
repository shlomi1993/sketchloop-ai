"""
Tests of the settings file loader for the shared models folder.
"""

import pytest
import re

from pathlib import Path

from sketchloop.settings import Settings, SettingsError, load_settings


def test_models_dir_resolves_against_the_file(tmp_path: Path) -> None:
    """
    A relative [models] dir is read relative to the settings file's folder, and the default is the models folder.
    """
    config = tmp_path / "config" / "sketchloop.toml"
    config.parent.mkdir()
    config.write_text('[models]\ndir = "weights"\n', encoding="utf-8")
    settings = load_settings(config)

    # The file's folder joins the relative path, and leaving the section out keeps the default.
    assert settings.models.dir == tmp_path / "config" / "weights", f"Expected weights next to the file, got {settings.models.dir}"
    assert Settings().models.dir == Path("models"), f"Expected the default models folder, got {Settings().models.dir}"


def test_old_models_dir_key_refused(tmp_path: Path) -> None:
    """
    The removed generator.diffusers.models_dir key is refused as an unknown key, naming the keys the table accepts.
    """
    config = tmp_path / "sketchloop.toml"
    config.write_text('[generator.diffusers]\nmodels_dir = "models"\n', encoding="utf-8")
    expected = "unknown key generator.diffusers.models_dir. Keys in [generator.diffusers]: mode, precision."
    with pytest.raises(SettingsError, match=re.escape(expected)):
        load_settings(config)


def test_guided_sketching_defaults_and_file_values(tmp_path: Path) -> None:
    """
    The backend and suggester default to the fakes on a 512x360 canvas, and a file sets every key.
    """
    # Without a file, the fakes need no weights, and the canvas is A4 in landscape on the latent grid.
    defaults = Settings()
    suggester = defaults.stroke_suggester
    assert defaults.generator.backend == "fake", f"Expected the fake backend by default, got {defaults.generator.backend!r}"
    assert suggester.name == "fake", f"Expected the fake suggester by default, got {suggester.name!r}"
    assert suggester.controlnet.precision is None, "The precision must default to the device default"
    drawing = (defaults.drawing.canvas_width, defaults.drawing.canvas_height, defaults.drawing.save_interval_seconds)
    assert drawing == (512, 360, 2.0), f"Expected a 512x360 canvas saved every 2 s, got {drawing}"

    # A file sets each key, and an integer interval becomes seconds as a float.
    config = tmp_path / "sketchloop.toml"
    config.write_text('[stroke_suggester]\nname = "fake"\n\n[stroke_suggester.controlnet]\nprecision = "float32"\n\n'
                      "[drawing]\ncanvas_width = 1024\ncanvas_height = 256\nsave_interval_seconds = 3\n", encoding="utf-8")
    settings = load_settings(config)
    precision = settings.stroke_suggester.controlnet.precision
    assert precision == "float32", f"Got precision {precision!r}"
    drawing = (settings.drawing.canvas_width, settings.drawing.canvas_height, settings.drawing.save_interval_seconds)
    assert drawing == (1024, 256, 3.0), f"Expected the file's drawing values, got {drawing}"
    assert isinstance(settings.drawing.save_interval_seconds, float), "The interval must be held as a float"


@pytest.mark.parametrize(("text", "message"), [
    ("[drawing]\ncanvas_width = 500\n", "drawing.canvas_width must be a multiple of 8 from 256 to 1024, got 500."),
    ("[drawing]\ncanvas_height = 248\n", "drawing.canvas_height must be a multiple of 8 from 256 to 1024, got 248."),
    ("[drawing]\ncanvas_width = 512.0\n", "drawing.canvas_width must be an integer, got 512.0."),
    ("[drawing]\nsave_interval_seconds = 0.5\n", "drawing.save_interval_seconds must be greater than 0.5, got 0.5."),
    ("[stroke_suggester]\ngoal = \"a car\"\n", "unknown key stroke_suggester.goal. Keys in [stroke_suggester]: name, controlnet.")
], ids=["width-500", "height-248", "float-width", "interval-0.5", "goal-key"])
def test_guided_sketching_values_refused(tmp_path: Path, text: str, message: str) -> None:
    """
    A canvas side off the latent grid or out of range, a save interval at or below 0.5 s, or an unknown key is refused.
    """
    config = tmp_path / "sketchloop.toml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(SettingsError, match=re.escape(message)):
        load_settings(config)


def test_study_and_prompt_suggester_settings(tmp_path: Path) -> None:
    defaults = Settings()

    # Study mode is off, exports go under the ignored data folder, and prompts come from the templates.
    assert not defaults.study.enabled, "Study mode must be off by default"
    assert defaults.study.export_dir == Path("data/study-exports"), f"Unexpected export folder {defaults.study.export_dir}"
    assert defaults.prompt_suggester.name == "templates", f"Unexpected prompt suggester {defaults.prompt_suggester.name}"

    # A file turns study mode on, chooses Florence-2, and places the export folder next to itself.
    config = tmp_path / "config" / "sketchloop.toml"
    config.parent.mkdir()
    study_text = '[study]\nenabled = true\nexport_dir = "exports"\n\n[prompt_suggester]\nname = "florence"\n'
    config.write_text(study_text, encoding="utf-8")
    settings = load_settings(config)
    assert settings.study.enabled and settings.prompt_suggester.name == "florence", f"Unexpected settings {settings}"
    assert settings.study.export_dir == tmp_path / "config" / "exports", f"Unexpected folder {settings.study.export_dir}"


@pytest.mark.parametrize(("text", "message"), [
    ('[study]\nenabled = "yes"\n', 'study.enabled must be true or false, got "yes".'),
    ("[study]\nexport_dir = \"\"\n", 'study.export_dir must be non-empty text, got "".'),
    ("[study]\ntask = \"car\"\n", "unknown key study.task. Keys in [study]: enabled, export_dir.")
], ids=["text-enabled", "empty-folder", "task-key"])
def test_study_values_refused(tmp_path: Path, text: str, message: str) -> None:
    config = tmp_path / "sketchloop.toml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(SettingsError, match=re.escape(message)):
        load_settings(config)
