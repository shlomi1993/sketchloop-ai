# Session handoff

## Current state

Development environment and project documentation only. No application code exists yet: capture, preprocessing, generation, orchestration, storage, and UI are all unimplemented, and interfaces in the design documents are proposals.

In place:
- Installer (`install.py`), offline checks (`scripts/check.py`: publication guard, syntax, TOML, Markdown links, Ruff lint, pytest), and GitHub Actions CI for Python 3.11/3.12.
- Project docs: specification, architecture, experiment contract, roadmap, privacy policy, references with citations, research notes area, and a thesis outline.
- Claude Code environment ([ADR 0003](decisions/0003-claude-code-environment.md)): `CLAUDE.md`, shared permissions, six role subagents, and the skills `/start-task`, `/handoff`, `/adr`, `/commit`, `/bisect`, `/readability`, and `/audit`. The subagents have not yet been exercised on an application task.
- Research workspace and Ruff lint ([ADR 0004](decisions/0004-research-workspace-and-tooling.md)).

Known hardware: Windows laptop without a GPU, laptop camera or USB webcam. A GPU or Colab may become available later.

## Next action

Start T01a in [ROADMAP.md](ROADMAP.md): typed domain contracts and synthetic adapter contract tests. In parallel, the researcher can compare MLflow with a filesystem store. The foundational paper is summarized in [research/sde-sketching-paper.md](research/sde-sketching-paper.md).

## Open questions

First real model (CPU-feasible or Colab-hosted), UI toolkit, store format, and software license. None blocks T01a. See the roadmap for when each is needed.

## Validation

On Windows 11 with Python 3.12, `.venv\Scripts\python.exe scripts/check.py` passes all checks and 19 tests. The installer was verified earlier on Python 3.14 in a fresh copy, but not rerun on Windows. Hosted CI has not run yet. No camera or model behavior has been tested.

## Handoff discipline

Replace this summary after each task rather than appending. Keep long-lived decisions in ADRs and requirements in the specification. Use [the session template](templates/SESSION.md) for longer handoffs. Never put private source text or local absolute paths here.
