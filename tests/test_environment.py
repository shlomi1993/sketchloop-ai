from importlib.metadata import version
from pathlib import Path

import sketchloop


def test_editable_package_import() -> None:
    """
    An editable install must resolve the package to this checkout's src/ folder, not a stale copy.
    """
    root = Path(__file__).resolve().parents[1]

    # The package imports from this checkout's src folder and has installed metadata.
    assert Path(sketchloop.__file__).resolve() == root / "src/sketchloop/__init__.py", "Package must import from src/"
    assert version("sketchloop-ai"), "Package metadata must be installed"
