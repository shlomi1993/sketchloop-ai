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
    The suggester and drawing sections default to the fake suggester on a 512x360 canvas, and a file sets every key.
    """
    # Without a file, the fake suggester needs no weights, and the canvas is A4 in landscape on the latent grid.
    defaults = Settings()
    assert defaults.suggester.name == "fake", f"Expected the fake suggester by default, got {defaults.suggester.name!r}"
    assert defaults.suggester.controlnet.precision is None, "The precision must default to the device default"
    drawing = (defaults.drawing.canvas_width, defaults.drawing.canvas_height, defaults.drawing.save_interval_seconds)
    assert drawing == (512, 360, 2.0), f"Expected a 512x360 canvas saved every 2 s, got {drawing}"

    # A file sets each key, and an integer interval becomes seconds as a float.
    config = tmp_path / "sketchloop.toml"
    config.write_text('[suggester]\nname = "fake"\n\n[suggester.controlnet]\nprecision = "float32"\n\n'
                      "[drawing]\ncanvas_width = 1024\ncanvas_height = 256\nsave_interval_seconds = 3\n", encoding="utf-8")
    settings = load_settings(config)
    assert settings.suggester.controlnet.precision == "float32", f"Got precision {settings.suggester.controlnet.precision!r}"
    drawing = (settings.drawing.canvas_width, settings.drawing.canvas_height, settings.drawing.save_interval_seconds)
    assert drawing == (1024, 256, 3.0), f"Expected the file's drawing values, got {drawing}"
    assert isinstance(settings.drawing.save_interval_seconds, float), "The interval must be held as a float"


@pytest.mark.parametrize(("text", "message"), [
    ("[drawing]\ncanvas_width = 500\n", "drawing.canvas_width must be a multiple of 8 from 256 to 1024, got 500."),
    ("[drawing]\ncanvas_height = 248\n", "drawing.canvas_height must be a multiple of 8 from 256 to 1024, got 248."),
    ("[drawing]\ncanvas_width = 1032\n", "drawing.canvas_width must be a multiple of 8 from 256 to 1024, got 1032."),
    ("[drawing]\ncanvas_width = 512.0\n", "drawing.canvas_width must be an integer, got 512.0."),
    ("[drawing]\nsave_interval_seconds = 0.5\n", "drawing.save_interval_seconds must be greater than 0.5, got 0.5."),
    ("[drawing]\nsave_interval_seconds = inf\n", "drawing.save_interval_seconds must be greater than 0.5, got inf."),
    ("[suggester]\ngoal = \"a car\"\n", "unknown key suggester.goal. Keys in [suggester]: name, controlnet.")
], ids=["width-500", "height-248", "width-1032", "float-width", "interval-0.5", "infinite-interval", "goal-key"])
def test_guided_sketching_values_refused(tmp_path: Path, text: str, message: str) -> None:
    """
    A canvas side off the latent grid or out of range, a save interval at or below 0.5 s, or an unknown key is refused.
    """
    config = tmp_path / "sketchloop.toml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(SettingsError, match=re.escape(message)):
        load_settings(config)
