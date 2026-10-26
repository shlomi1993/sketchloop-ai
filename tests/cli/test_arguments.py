import os
import pytest
import re
import sys
import types

from pathlib import Path

import sketchloop
from sketchloop.app import AppController
from sketchloop.cli import main


# Run every test from tmp_path, so a sketchloop.toml in the repository cannot change CLI settings.
pytestmark = pytest.mark.usefixtures("isolated_cwd")


EXAMPLE_SKETCH = Path(__file__).resolve().parents[2] / "examples" / "sketch-photo.jpg"


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AppController, dict[str, object]]]:
    """
    Replace sketchloop.ui with a stub whose launch_app records its arguments and the telemetry settings, without gradio.
    """
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    calls: list[tuple[AppController, dict[str, object]]] = []

    def record_launch(controller: AppController, **options: object) -> None:
        """
        Record a launch instead of starting a server.
        """
        telemetry = {name: os.environ.get(name) for name in ("GRADIO_ANALYTICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY")}
        calls.append((controller, options | telemetry))

    stub = types.ModuleType("sketchloop.ui")
    stub.launch_app = record_launch
    monkeypatch.setitem(sys.modules, "sketchloop.ui", stub)
    monkeypatch.setattr(sketchloop, "ui", stub, raising=False)
    return calls


@pytest.mark.usefixtures("launches")
@pytest.mark.parametrize(("command", "option", "settings_text", "error_pattern"), [
    ("run", ["--candidates", "0"], None, r"must be at least 1, got 0"),
    ("run", ["--seed", "-1"], None, r"must be at least 0, got -1"),
    ("ui", ["--candidates", "0"], None, r"must be at least 1, got 0"),
    ("ui", ["--mode", "fast"], None, r"--mode needs --backend diffusers"),
    ("run", [], '[generator]\nbackend = "difusers"\n', r'sketchloop\.toml: generator\.backend must be one of .+"difusers"\.'),
    ("ui", [], "[ui]\ncandidates = true\n", r"sketchloop\.toml: ui\.candidates must be an integer, got (?i:true)\."),
    ("run", [], '[generator]\nmdoe = "fast"\n', r"sketchloop\.toml: unknown key generator\.mdoe\. Keys in \[generator\]: "),
    ("run", [], '[generator\nbackend = "fake"\n', r"sketchloop\.toml is not valid TOML: .+\. Fix the file and retry\."),
    ("run", [], "[preprocessing]\nrotation = 90\nraw = true\n",
     re.escape("preprocessing.rotation 90 cannot be combined with preprocessing.raw, which skips preprocessing. Change one in sketchloop.toml or pass --rotate 0 or --no-raw.")),  # noqa: E501
    ("run", [], '[storage]\nstore = "memory"\n', re.escape('storage.store must be one of folder, got "memory".')),
    ("rerun", ["--config", "missing.toml"], None, re.escape("Settings file missing.toml not found. Check the --config path.")),
    ("ui", ["--capture-interval", "inf"], None, r"must be greater than 0 seconds, got inf"),
    ("ui", [], "[ui]\ncapture_interval_seconds = inf\n", r"capture_interval_seconds must be greater than 0, got inf\."),
    ("run", [], "seed = 7\n", re.escape("sketchloop.toml: seed must be inside a section such as [run]."))
], ids=["candidates-0", "negative-seed", "ui-candidates-0", "mode-without-diffusers", "misspelled-backend", "boolean-candidates",
        "unknown-key", "broken-toml", "rotation-with-raw", "memory-store", "missing-config", "infinite-interval-flag",
        "infinite-interval-file", "top-level-key"])
def test_cli_rejects_invalid_options(tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, option: list[str],
                                     settings_text: str | None, error_pattern: str) -> None:
    """
    Bad flags, a bad settings file, or a missing --config file are rejected with a one-line reason before anything is saved.
    """
    runs_option = ["--runs-dir", str(tmp_path / "runs")]
    base = {
        "run": ["run", str(EXAMPLE_SKETCH), "--prompt", "chair", *runs_option],
        "ui": ["ui", *runs_option],
        "rerun": ["rerun", str(tmp_path / "runs" / "earlier"), "--round", "1", *runs_option],
    }[command]

    # A settings file in the working folder is found without --config.
    if settings_text is not None:
        (tmp_path / "sketchloop.toml").write_text(settings_text, encoding="utf-8")

    # Argparse exits with its usage error code.
    with pytest.raises(SystemExit) as raised:
        main([*base, *option])
    errors = capsys.readouterr().err

    # The error names the reason, and nothing is saved for the rejected command.
    assert raised.value.code == 2, f"Expected the usage error code 2, got {raised.value.code}"
    assert re.search(error_pattern, errors), f"Expected an error matching {error_pattern!r}, got {errors!r}"
    assert not (tmp_path / "runs").exists(), "No session folder may be created for invalid options"
