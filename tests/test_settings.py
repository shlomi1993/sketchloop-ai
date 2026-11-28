"""
Tests of the settings file loader: values, relative folders, and refused mistakes.
"""

import pytest
import re

from pathlib import Path

from sketchloop.settings import SettingsError, load_settings


def test_file_values_resolve_against_the_file(tmp_path: Path) -> None:
    """
    A file sets the guided sketching and study keys, with relative folders read next to the file.
    """
    config = tmp_path / "config" / "sketchloop.toml"
    config.parent.mkdir()
    lines = ["[models]", 'dir = "weights"', "[stroke_suggester.controlnet]", 'precision = "float32"', "[drawing]",
             "canvas_width = 1024", "canvas_height = 256", "save_interval_seconds = 3", "[study]", "enabled = true",
             'export_dir = "exports"', "[prompt_suggester]", 'name = "florence"', ""]
    config.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(config)

    # Relative folders join the file's folder.
    folders = (settings.models.dir, settings.study.export_dir)
    assert folders == (tmp_path / "config" / "weights", tmp_path / "config" / "exports"), f"Unexpected folders {folders}"

    # Each key takes the file's value, and an integer interval becomes seconds as a float.
    drawing = (settings.drawing.canvas_width, settings.drawing.canvas_height, settings.drawing.save_interval_seconds)
    assert drawing == (1024, 256, 3.0), f"Expected the file's drawing values, got {drawing}"
    assert isinstance(settings.drawing.save_interval_seconds, float), "The interval must be held as a float"
    chosen = (settings.stroke_suggester.controlnet.precision, settings.study.enabled, settings.prompt_suggester.name)
    assert chosen == ("float32", True, "florence"), f"Unexpected suggester and study values {chosen}"


@pytest.mark.parametrize(("text", "message"), [
    ("[drawing]\ncanvas_width = 500\n", "drawing.canvas_width must be a multiple of 8 from 256 to 1024, got 500."),
    ("[drawing]\nsave_interval_seconds = 0.5\n", "drawing.save_interval_seconds must be greater than 0.5, got 0.5."),
    ('[stroke_suggester]\ngoal = "a car"\n', "unknown key stroke_suggester.goal. Keys in [stroke_suggester]: name, controlnet."),
    ('[study]\nenabled = "yes"\n', 'study.enabled must be true or false, got "yes".'),
    ('[generator.diffusers]\nmodels_dir = "models"\n',
     "unknown key generator.diffusers.models_dir. Keys in [generator.diffusers]: mode, precision."),
], ids=["width-500", "interval-0.5", "goal-key", "text-enabled", "removed-models-dir"])
def test_settings_mistakes_refused(tmp_path: Path, text: str, message: str) -> None:
    """
    A canvas off the latent grid, a too-short save interval, a task key, a text boolean, or a removed key is refused.
    """
    config = tmp_path / "sketchloop.toml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(SettingsError, match=re.escape(message)):
        load_settings(config)
