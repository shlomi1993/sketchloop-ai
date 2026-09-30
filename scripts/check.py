from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib

from urllib.parse import unquote, urlsplit

from privacy_check import ROOT, main as privacy_main, public_paths


def main() -> int:
    """Run the privacy guard, file checks, Ruff lint, and pytest in order, stopping at the first failure.

    Returns:
        int: Process exit code, 0 when every check passes.
    """
    if privacy_main():
        return 1

    failures = []
    for name in public_paths():
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            continue

        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=name)
            except SyntaxError as exc:
                failures.append(f"Python syntax: {name}:{exc.lineno}")

        elif path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith("#"):
                    continue

                destination = (path.parent / unquote(parsed.path)).resolve()
                if not destination.is_relative_to(ROOT) or not destination.exists():
                    failures.append(f"Broken/nonportable Markdown target in {name}: {target}")

    with (ROOT / "pyproject.toml").open("rb") as stream:
        tomllib.load(stream)

    if failures:
        print("\n".join(failures))
        return 1

    print("Python syntax, TOML, and local Markdown targets passed.", flush=True)
    if subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT).returncode:
        return 1

    result = subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
