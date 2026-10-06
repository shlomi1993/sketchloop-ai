# Session handoff

## Current state

T01a is done: typed domain records, the generator contract with capability validation, and a labeled deterministic fake adapter ([design](design/t01a-contracts.md), [ADR 0005](decisions/0005-domain-records-and-generator-contract.md)). The T01b runnable increment is implemented: `sketchloop examples/sketch.png --prompt "..."` loads a PNG sketch, generates fake candidates, saves them under `runs/<iteration-id>/`, and records a terminal selection (no persisted record yet). T01 is done: T01b also recorded research notes and the first-model, store, and UI decisions ([ADR 0006](decisions/0006-first-backend-store-and-ui.md)). T02a is implemented: the command loads PNG or JPEG with OpenCV, keeps the raw file as `sketch-raw.<ext>`, preprocesses it (grayscale, crop to strokes, contrast normalization, resize to 512 px) into `sketch.png` with ordered step records, and generates from it unless `--raw` is given. Perspective correction is deferred. Webcam capture, real generation, orchestration, storage, and UI are unimplemented, and no real model has run.

In place:
- Installer (`install.py`), offline checks (`scripts/check.py`: publication guard, syntax, TOML, Markdown links, Ruff lint, pytest), and GitHub Actions CI for Python 3.11/3.12.
- Project docs: specification, architecture, experiment contract, roadmap, privacy policy, references with citations, research notes area, and a thesis outline.
- Claude Code environment ([ADR 0003](decisions/0003-claude-code-environment.md)): `CLAUDE.md`, shared permissions, six role subagents, and the skills `/start-task`, `/handoff`, `/adr`, `/commit`, `/bisect`, `/readability`, and `/audit`. The subagents have not yet been exercised on an application task.
- Research workspace and Ruff lint ([ADR 0004](decisions/0004-research-workspace-and-tooling.md)).

Known hardware: Windows laptop without a GPU, laptop camera or USB webcam. A GPU or Colab may become available later.

## Next action

Start T02b in [ROADMAP.md](ROADMAP.md): capture a sketch from the webcam with `sketchloop --camera` through the same input path.

## Open questions

Software license for the repository. The CPU speed target in ADR 0006 is a starting point to adjust after T03 measurements.

## Validation

On Windows 11 with Python 3.12, `.venv\Scripts\python.exe scripts/check.py` passes all checks and 32 tests. The installer was verified earlier on Python 3.14 in a fresh copy, but not rerun on Windows. Hosted CI has not run yet. No camera or model behavior has been tested.

## Handoff discipline

Replace this summary after each task rather than appending. Keep long-lived decisions in ADRs and requirements in the specification. Use [the session template](templates/SESSION.md) for longer handoffs. Never put private source text or local absolute paths here.
