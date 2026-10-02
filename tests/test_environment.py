"""Fast installation checks; no camera, model, credentials, or network required."""

from importlib.metadata import version
from pathlib import Path

import sketchloop


def test_editable_package_import() -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path(sketchloop.__file__).resolve() == root / "src/sketchloop/__init__.py", "Package must import from src/"
    assert version("sketchloop-ai"), "Package metadata must be installed"


def test_temporary_artifact_round_trip(tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic.txt"
    artifact.write_text("synthetic readiness check", encoding="utf-8")
    assert artifact.read_text(encoding="utf-8") == "synthetic readiness check", "Temporary file must round-trip"
