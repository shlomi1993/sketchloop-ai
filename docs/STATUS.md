# Session handoff

## Current state

T01a is done: typed domain records, the generator contract with capability validation, and a labeled deterministic fake adapter ([design](design/t01a-contracts.md), [ADR 0005](decisions/0005-domain-records-and-generator-contract.md)). T01 is not fully done. Capture, preprocessing, real generation, orchestration, storage, and UI are unimplemented, and no real model has run.

In place:
- Installer (`install.py`), offline checks (`scripts/check.py`: publication guard, syntax, TOML, Markdown links, Ruff lint, pytest), and GitHub Actions CI for Python 3.11/3.12.
- Project docs: specification, architecture, experiment contract, roadmap, privacy policy, references with citations, research notes area, and a thesis outline.
- Claude Code environment ([ADR 0003](decisions/0003-claude-code-environment.md)): `CLAUDE.md`, shared permissions, six role subagents, and the skills `/start-task`, `/handoff`, `/adr`, `/commit`, `/bisect`, `/readability`, and `/audit`. The subagents have not yet been exercised on an application task.
- Research workspace and Ruff lint ([ADR 0004](decisions/0004-research-workspace-and-tooling.md)).

Known hardware: Windows laptop without a GPU, laptop camera or USB webcam. A GPU or Colab may become available later.

## Next action

Finish T01 (T01b in [ROADMAP.md](ROADMAP.md)): record research findings and the hardware, first-model, and UI decisions. In parallel, the researcher can compare MLflow with a filesystem store. The foundational paper is summarized in [research/sde-sketching-paper.md](research/sde-sketching-paper.md).

## Open questions

First real model (CPU-feasible or Colab-hosted), UI toolkit, store format, and software license. None blocks T01b. See the roadmap for when each is needed.

## Validation

On Windows 11 with Python 3.12, `.venv\Scripts\python.exe scripts/check.py` passes all checks and 29 tests. The installer was verified earlier on Python 3.14 in a fresh copy, but not rerun on Windows. Hosted CI has not run yet. No camera or model behavior has been tested.

## Handoff discipline

Replace this summary after each task rather than appending. Keep long-lived decisions in ADRs and requirements in the specification. Use [the session template](templates/SESSION.md) for longer handoffs. Never put private source text or local absolute paths here.
