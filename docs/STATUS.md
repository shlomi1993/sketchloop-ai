# Session handoff

## Current state

T01a is done: typed domain records, the generator contract with capability validation, and a labeled deterministic fake adapter ([design](design/t01a-contracts.md), [ADR 0005](decisions/0005-domain-records-and-generator-contract.md)). The T01b runnable increment is implemented: `sketchloop examples/sketch.png --prompt "..."` loads a PNG sketch, generates fake candidates, saves them under `runs/<iteration-id>/`, and records a terminal selection (no persisted record yet). T01 is done: T01b also recorded research notes and the first-model, store, and UI decisions ([ADR 0006](decisions/0006-first-backend-store-and-ui.md)). T02a is implemented: the command loads PNG or JPEG with OpenCV, keeps the raw file as `sketch-raw.<ext>`, preprocesses it with grayscale, crop to strokes, contrast normalization, and resize to 512 px into `sketch.png` with ordered step records, and generates from it unless `--raw` is given. T02b is implemented and verified with the laptop webcam: `sketchloop --camera` shows a preview (Space captures, Esc cancels), saves the frame as `sketch-raw.png`, and preprocessing now corrects the paper's perspective. The dependency is now `opencv-python` (GUI build, ADR 0008 amended). T03 is done on CPU: `--backend diffusers` generates real images with SD 1.5 + ControlNet scribble, in `fast` (LCM-LoRA, 4 steps) or `quality` mode, from weights fetched by `scripts/download_models.py` ([ADR 0009](decisions/0009-diffusers-backend.md)). Image quality and CPU speed still need work, and the Apple MPS path has not run. T04 is done with the fake backend: `SketchSession` runs successive rounds with one loaded backend, links each iteration to the previous round, saves rounds under `runs/<session>/round-<n>/`, and calls a no-op `Evaluator` separately from generation and selection ([ADR 0010](decisions/0010-sessions-and-evaluator.md)). After each pick, a menu repeats, changes the prompt, recaptures, loads a new file, or quits with a summary. T05 is done with the fake backend ([design](design/t05-records.md), [ADR 0011](decisions/0011-session-records-and-subcommands.md)): the command now has subcommands. `sketchloop run` keeps the earlier behavior and also atomically writes `session.json`, with Python, platform, and package versions, and `round-<n>/round.json`, with request, effective settings, seeds, checksums, stage timings, and selection or failure. `sketchloop show <session>` prints the records and lists missing or changed files without loading a model. `sketchloop rerun <session> --round N` repeats a round from the stored `sketch.png` in a new linked session and reports identical candidates and environment differences. Sessions saved before T05 cannot be shown. The UI is unimplemented.

In place:
- Installer (`install.py`), offline checks in `scripts/check.py` for publication guard, syntax, TOML, Markdown links, Ruff lint, and pytest, and GitHub Actions CI for Python 3.11/3.12.
- Project docs: specification, architecture, experiment contract, roadmap, privacy policy, references with citations, research notes area, and a thesis outline.
- Claude Code environment ([ADR 0003](decisions/0003-claude-code-environment.md)): `CLAUDE.md`, shared permissions, six role subagents, and the skills `/start-task`, `/handoff`, `/adr`, `/commit`, `/bisect`, `/readability`, and `/audit`. The subagents have not yet been exercised on an application task.
- Research workspace and Ruff lint ([ADR 0004](decisions/0004-research-workspace-and-tooling.md)).

Known hardware: Windows laptop without a GPU, laptop camera or USB webcam. A GPU or Colab may become available later.

## Next action

Rerun a diffusers round with `sketchloop rerun` on the laptop to see the reported differences on a real backend. Then start T06 in [ROADMAP.md](ROADMAP.md). Later, improve conditioning for thin pencil strokes and run the diffusers backend on the Apple M2 Pro.

## Open questions

Software license for the repository. Follow-up: make the data folder configurable instead of `runs/` in the repository. The one-minute CPU target in ADR 0006 is not met: one fast-mode image takes about 35-60 s of sampling and about 100 s in total on the laptop CPU.

## Validation

On Windows 11 with Python 3.12, `.venv\Scripts\python.exe scripts/check.py` passes all checks and 47 tests, and `python -m pytest -m model` passes the real-model test in about 75 s. The installer was verified earlier on Python 3.14 in a fresh copy, but not rerun on Windows. The owner captured a real sketch with the laptop webcam through `sketchloop --camera` on 2026-10-08, and the full flow completed. The owner ran the diffusers backend on the laptop CPU on 2026-10-09.

## Handoff discipline

Replace this summary after each task rather than appending. Keep long-lived decisions in ADRs and requirements in the specification. Use [the session template](templates/SESSION.md) for longer handoffs. Never put private source text or local absolute paths here.
