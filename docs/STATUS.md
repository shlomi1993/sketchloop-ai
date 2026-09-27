# Session handoff

## Current state

Development environment established. The complete seven-page source proposal has been read, visually reviewed, and synthesized into privacy-safe project documentation. The source PDF is ignored and is not a development dependency.

Implemented: clone-and-install setup via `python3 install.py` (creates/reuses `.venv`, installs development dependencies, checks compatibility, and runs two readiness tests), agent agreement, requirements and source mapping, architecture guidance, experiment contract, roadmap, decision/handoff templates, minimal Python package metadata, offline repository checks, pytest privacy guard tests, and GitHub Actions configuration. Tests use plain assertions, parametrization, and the `tmp_path` fixture; the pinned pytest development extra is installed before local and CI checks.

Not implemented: capture, preprocessing, real or fake generation adapters, orchestration, storage/replay, UI, or application evaluation. Proposed interfaces in the design documents are not existing APIs.

Code style: 120-character lines; prefer single-line calls and expressions when they fit. Add a blank line after `return` or `raise` when another statement follows. Agent guidance in `AGENTS.md` and optional Ruff formatting settings in `pyproject.toml` record these preferences.

Installer command helper now accepts a quoted string and executes it without a shell. All function signatures must include parameter and return type hints. The updated installer tests cover quoted arguments and failure propagation; all 18 tests and repository checks passed. No files have been staged or committed during setup.

The environment prompt is now `sketchloop-ai` while storage stays in `.venv`. The installer refreshes activation scripts for existing environments and explains that activation must happen in the calling shell. The existing local environment was updated. All 19 tests pass, including activation and interpreter selection in a temporary environment with a spaced path.

## Next action

Start T01a in [ROADMAP.md](ROADMAP.md): typed domain contracts and synthetic adapter contract tests. Read [PROJECT.md](PROJECT.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [EXPERIMENTS.md](EXPERIMENTS.md) first.

## Open questions

Hardware, first real model, UI toolkit, storage implementation, and distribution license remain undecided. They do not block T01a. See the roadmap for when each answer becomes necessary.

## Validation

Verified during setup on Python 3.14: publication guard, Python syntax, TOML parsing, local Markdown targets, and all five privacy regression tests passed. A fresh public-only copy also passed without the PDF or local virtual environment. Editable package installation and import passed in `.venv`. The source PDF is ignored and untracked; existing Git history contains only the original README and ignore file.

After the pytest migration, `.venv/bin/python scripts/check.py` passed all repository checks and 14 parametrized test cases on Python 3.14 with pytest 9.0.2. The development extra installed successfully. See [ADR 0002](decisions/0002-pytest.md).

The installer was verified in a fresh public-only copy invoked from outside its directory: dependency installation, `pip check`, and both readiness tests passed. A second installer run successfully reused the environment. An invalid existing `.venv` was rejected without deleting its contents. The full local suite now has 16 passing tests. Windows execution has not been locally tested.

Run `python3 install.py`, activate `.venv`, and run `python scripts/check.py` in the activated environment to reproduce offline checks. GitHub-hosted CI is configured for Python 3.11/3.12 but has not yet run. No application, camera, or model behavior has been tested because those components are not implemented.

## Handoff discipline

Replace this state summary after completing a task. Keep it short and factual; place long-lived decisions in ADRs and detailed requirements in the specification. Use [the session template](templates/SESSION.md) when a longer handoff is needed. Never put private source text or local absolute paths here.
