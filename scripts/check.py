from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib

from urllib.parse import unquote, urlsplit

from privacy_check import ROOT, main as privacy_main, public_paths


def main() -> int:
    """
    Run the privacy guard, file checks, Ruff lint, and pytest in order, stopping at the first failure.

    Returns:
        int: Process exit code, 0 when every check passes.
    """
    # Stop before any other check if publishable content leaks private data.
    if privacy_main():
        return 1

    # Collect every file problem first so one run reports all of them.
    failures = []
    for name in public_paths():
        path = ROOT / name

        # Skip directories, deleted entries, and symlinks, which the privacy guard already reviews.
        if not path.is_file() or path.is_symlink():
            continue

        # Catch syntax errors cheaply, even in files pytest never imports.
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=name)
            except SyntaxError as exc:
                failures.append(f"Python syntax: {name}:{exc.lineno}")

        # Verify that relative Markdown links resolve inside the repository.
        elif path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                parsed = urlsplit(target)

                # External URLs and in-page anchors are not checked offline.
                if parsed.scheme or target.startswith("#"):
                    continue

                # Reject targets that are missing or escape the repository root.
                destination = (path.parent / unquote(parsed.path)).resolve()
                if not destination.is_relative_to(ROOT) or not destination.exists():
                    failures.append(f"Broken/nonportable Markdown target in {name}: {target}")

    # Parse pyproject.toml so a malformed config fails here rather than inside pip or pytest.
    with (ROOT / "pyproject.toml").open("rb") as stream:
        tomllib.load(stream)

    # Report all file problems together before running the slower tools.
    if failures:
        print("\n".join(failures))
        return 1

    # Flush so this line appears before the subprocess output.
    print("Python syntax, TOML, and local Markdown targets passed.", flush=True)
    if subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT).returncode:
        return 1

    # Run the test suite last and pass its exit code through.
    result = subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
