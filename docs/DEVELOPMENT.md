# Development environment

## Baseline

Python 3.11+ runs the scaffold and its repository checks and pytest tests. Python 3.11 and 3.12 are configured in CI; local validation may use a newer interpreter. Compatibility of future ML packages must be evaluated separately before fixing the application runtime. No runtime dependencies, provider credentials, model downloads, camera access, or GPU are needed yet.

From the repository root:

```sh
python3 install.py && . .venv/bin/activate
python scripts/check.py
```

The installer uses the Python interpreter that launches it to create `.venv`. It also works when invoked by path from another directory. Existing valid environments are reused; existing incomplete directories are not deleted or overwritten. To change Python versions, move the old environment aside and rerun with the desired interpreter. Installation stops on any failed command and reports success only after `pip check` and two readiness tests pass (editable package import and temporary artifact read/write). It does not run the full suite or download models. Initial dependency installation requires network access or cached packages. If creation fails because `venv`/`ensurepip` is unavailable, install your operating system's Python venv support and move the partial environment aside before retrying.

On Windows, run `py -3 install.py` and activate with `.venv\Scripts\Activate.ps1`. Alternatively invoke `.venv/bin/python` or `.venv\Scripts\python.exe` directly. Checks run offline once the development dependencies are installed.

For a library-only editable install without testing tools (requires the build backend unless cached):

```sh
python -m pip install -e .
python -c "import sketchloop"
```

The package is an intentionally empty namespace scaffold. An import succeeding does not mean the application exists. Declare future runtime dependencies in `pyproject.toml`, keep hardware/backend extras optional, and commit a resolved lock or constraints file with the chosen compatible runtime when dependencies are introduced. Do not freeze an unrelated global environment or commit machine-specific paths.

## Commands

| Command | Purpose |
| --- | --- |
| `python3 install.py` | Create/reuse `.venv`, install development dependencies, and run two readiness tests |
| `python3 scripts/check.py` | Privacy/publication guard, local Markdown target checks, Python syntax checks, pytest tests |
| `python3 scripts/privacy_check.py` | Scan publishable working files and staged blobs; reject private artifact paths and suspicious data patterns |
| `python3 -m pytest` | Run offline tests only |
| `make check` | Convenience alias for the full check (override `PYTHON` if needed) |

Default verification is deliberately lightweight. As application code grows, add selected lint/type tools with reproducible versions and meaningful contract/integration tests. Keep hardware/network tests separately invoked and accurately report skipped coverage.

## Repository layout

- `src/sketchloop/`: Python library; implement modules as their slices begin.
- `tests/`: synthetic, offline tests; never use copied private documents as fixtures.
- `docs/`: durable project knowledge, decisions, and handoff.
- `scripts/`: repository maintenance checks.
- `.github/`: CI and review template.
- `runs/`, `data/`, `models/`, `private/`, `tmp/`: ignored local-only artifacts, created as needed.

CI runs the installer, then runs the full offline checks using its prepared environment on pushes and pull requests. It uses read-only repository permissions, does not request secrets, and does not execute real inference. See the [official checkout action](https://github.com/actions/checkout), [Python setup action](https://github.com/actions/setup-python), and [PyPA packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) for the configuration conventions used here.

## Delivering a change

Select a roadmap task, define its observable outcome, implement a complete slice, run focused verification plus repository checks, review the diff, and update the handoff. Record important design changes in an ADR. Include requirement IDs and actual validation in PR descriptions. Do not conflate a passing offline test with a successful camera/model demonstration.

## Test conventions

Use pytest test functions, plain assertions, `tmp_path` for temporary files, and parametrization for related cases. Discovery is restricted to `tests/`; the repository scripts directory is configured as an import path in `pyproject.toml`, so tests need no manual `sys.path` edits. The pytest version is pinned in the `dev` extra. See [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html) for the supported convention.

## Code style

Use a 120-character line limit and keep calls and expressions on one line when they fit and remain readable. `AGENTS.md` records the agent formatting preferences. Ruff configuration in `pyproject.toml` uses the same limit and permits collapsing short expressions with trailing commas. Ruff is not currently installed or enforced by CI; these settings apply when using a Ruff-enabled editor or formatter. See the [Ruff formatting settings](https://docs.astral.sh/ruff/settings/#format_skip-magic-trailing-comma).

Add a blank line after `return` or `raise` when another statement follows, including after a guard clause before the outer block continues. No extra blank line is needed solely at the end of a function or file.

Do not add file-level (module) docstrings unless explicitly requested by the user.

Prefer concise error messages and single-line `raise` statements within 120 characters. Remove redundant wording instead of splitting messages across adjacent string literals, while preserving the essential cause and recovery action.

Always annotate function parameters and return types, including helpers and tests. Use `-> None` when appropriate; implicit `self` and `cls` parameters need no annotation.

The installer's `run(label: str, command: str) -> None` accepts a POSIX-quoted command string, parsed with `shlex.split` and executed without a shell on all platforms. Quote dynamic arguments with `shlex.quote`; shell operators and variable expansion are not supported.

## Import ordering

Use three import blocks, separated by one blank line:

1. Plain `import package` statements for standard-library and third-party packages.
2. `from package import name` statements for standard-library and third-party packages.
3. Imports from project-local modules, including relative imports.

Sort each block alphabetically by module/package path, and sort multiple imported names alphabetically.
Keep one plain import per line. Omit empty blocks. Required `from __future__` imports remain first,
in their own block. Keep project-local plain imports in the project-local block as well.

```python
import os
import subprocess

from collections.abc import Callable
from pathlib import Path

from sketchloop.domain import Candidate, Iteration
```


The environment stays in `.venv` and uses the shell prompt name `sketchloop-ai`. Running `python3 install.py` alone prepares it but cannot activate it in the calling terminal. Run `. .venv/bin/activate` afterward, or use the combined command above. If Conda shows `(base)`, run `conda deactivate` before activating the project environment to avoid stacked prompts. Check the selected interpreter with `python -c "import sys; print(sys.prefix)"`, which should point to this repository's `.venv`. Rerunning the installer refreshes the prompt name of an existing environment without removing its packages.
