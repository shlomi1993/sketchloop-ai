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
