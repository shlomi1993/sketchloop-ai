"""
Fast installation checks; no camera, model, credentials, or network required.
"""

from importlib.metadata import version
from pathlib import Path

import sketchloop


def test_editable_package_import() -> None:
    """
    An editable install must resolve the package to this checkout's src/ folder, not a stale copy.
    """
    root = Path(__file__).resolve().parents[1]
    assert Path(sketchloop.__file__).resolve() == root / "src/sketchloop/__init__.py", "Package must import from src/"
    assert version("sketchloop-ai"), "Package metadata must be installed"


def test_temporary_artifact_round_trip(tmp_path: Path) -> None:
    """
    Pytest's temporary folder must be writable, since later tests store artifacts there.
    """
    artifact = tmp_path / "synthetic.txt"
    artifact.write_text("synthetic readiness check", encoding="utf-8")
    assert artifact.read_text(encoding="utf-8") == "synthetic readiness check", "Temporary file must round-trip"
