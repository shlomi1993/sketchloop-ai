# Session handoff

## Current state

T01a is done: typed domain records, the generator contract with capability validation, and a labeled deterministic fake adapter ([design](design/t01a-contracts.md), [ADR 0005](decisions/0005-domain-records-and-generator-contract.md)). The T01b runnable increment is implemented: `sketchloop examples/sketch.png --prompt "..."` loads a PNG sketch, generates fake candidates, saves them under `runs/<iteration-id>/`, and records a terminal selection (no persisted record yet). T01 is done: T01b also recorded research notes and the first-model, store, and UI decisions ([ADR 0006](decisions/0006-first-backend-store-and-ui.md)). T02a is implemented: the command loads PNG or JPEG with OpenCV, keeps the raw file as `sketch-raw.<ext>`, preprocesses it with grayscale, crop to strokes, contrast normalization, and resize to 512 px into `sketch.png` with ordered step records, and generates from it unless `--raw` is given. T02b is implemented and verified with the laptop webcam: `sketchloop --camera` shows a preview (Space captures, Esc cancels), saves the frame as `sketch-raw.png`, and preprocessing now corrects the paper's perspective. The dependency is now `opencv-python` (GUI build, ADR 0008 amended). T03 is done on CPU: `--backend diffusers` generates real images with SD 1.5 + ControlNet scribble, in `fast` (LCM-LoRA, 4 steps) or `quality` mode, from weights fetched by `scripts/download_models.py` ([ADR 0009](decisions/0009-diffusers-backend.md)). Image quality and CPU speed still need work, and the Apple MPS path has not run. T04 is done with the fake backend: `SketchSession` runs successive rounds with one loaded backend, links each iteration to the previous round, saves rounds under `runs/<session>/round-<n>/`, and calls a no-op `Evaluator` separately from generation and selection ([ADR 0010](decisions/0010-sessions-and-evaluator.md)). After each pick, a menu repeats, changes the prompt, recaptures, loads a new file, or quits with a summary. T05 is done with the fake backend ([design](design/t05-records.md), [ADR 0011](decisions/0011-session-records-and-subcommands.md)): the command now has subcommands. `sketchloop run` keeps the earlier behavior and also atomically writes `session.json`, with Python, platform, and package versions, and `round-<n>/round.json`, with request, effective settings, seeds, checksums, stage timings, and selection or failure. `sketchloop show <session>` prints the records and lists missing or changed files without loading a model. `sketchloop rerun <session> --round N` repeats a round from the stored `sketch.png` in a new linked session and reports identical candidates and environment differences. Sessions saved before T05 cannot be shown.

T06 is done (R07), and the owner verified the browser UI on 2026-10-12 ([design](design/t06-ui.md), [ADR 0012](decisions/0012-gradio-ui-and-input-source.md)):
- **Command:** `sketchloop ui` serves a Gradio page on 127.0.0.1 only, with analytics off and no share link.
- **Page:** three steps (sketch, direction, choose) with a live webcam or an upload, a prompt, a Generate button enabled only when a sketch and prompt exist, a candidate gallery, an explicit choice, a new-session button, and the current session's history.
- **Webcam:** the browser streams frames, and the server keeps the newest one every `--capture-interval` seconds, 0.5 by default, so Generate needs no snapshot. The camera starts on page load, and the preview turns with the rotation setting, 180 degrees by default for the webcam.
- **Library:** `sketchloop.app.AppController` holds one loaded generator and the session and never imports Gradio. A selection must name the iteration the page showed, so a stale tab cannot record a choice. `sketchloop.backends` creates backends for the CLI and UI, and each backend describes itself.
- **Records:** each round records `input.source` as file, camera, webcam, or upload. Older rounds load it as unavailable, and the schema stays at version 1.
- **Dependencies:** `pip install -e .` now installs the model and UI libraries, and `[dev]` is the only extra ([ADR 0013](decisions/0013-default-dependencies.md)). Linux CI installs CPU-only torch.
- **Preprocessing:** a tilted sheet is now straightened by its corners or, when partly off-frame, by its visible edges, and the removed tilt is recorded. This is verified on synthetic images only.

T07 is implemented and tested offline (R08, R10), and the owner has not yet tried it by hand ([design](design/t07-modularity.md), [ADR 0014](decisions/0014-replaceable-components-and-settings.md)):
- **Interfaces:** capture (`FileCapture`, `CameraCapture`), preprocessing (`StrokePreprocessor`), and storage (`FolderStore`, and `MemoryStore` for tests) sit behind protocols next to the existing `Generator` and `Evaluator`. `SketchSession` and `AppController` depend only on these protocols, and a test runs a whole session with every component substituted.
- **Settings:** an optional `sketchloop.toml` in the working folder, or `--config PATH`, chooses the backend and mode, image source, preprocessor, evaluator, store, runs folder, models folder, and run and UI defaults. Flags override the file, and bad settings exit with one line naming the key before anything is created. `rerun` takes only storage, the evaluator, and the models folder from settings. `examples/sketchloop.toml` shows every key.
- **Saving:** a failed save raises a one-line `StoreError` without absolute paths. `session.json` is retried on the next round, a failed selection save can be retried, and an interrupted atomic write leaves no `.tmp` file. Records keep schema version 1.
- **Models folder:** `generator.diffusers.models_dir` sets where the weights are, and `scripts/download_models.py` reads the same settings file.

Standalone tools resolve paths from their own files without requiring an application install.

The publication guard uses working and staged Git ignore rules to catch ignored tracked or force-staged files (R10).
It retains environment-file checks and scans UTF-8 and BOM-marked UTF-16 text. See [PRIVACY.md](PRIVACY.md).

In place:
- Installer (`install.py`), offline checks in `scripts/check.py` for publication guard, syntax, TOML, Markdown links, Ruff lint, and pytest, and GitHub Actions CI for Python 3.11/3.12.
- Project docs: specification, architecture, experiment contract, roadmap, privacy policy, references with citations, research notes area, and a thesis outline.
- Claude Code environment ([ADR 0003](decisions/0003-claude-code-environment.md)): `CLAUDE.md`, shared permissions, six role subagents, and the skills `/start-task`, `/handoff`, `/adr`, `/commit`, `/bisect`, `/readability`, and `/audit`. T06 used the researcher, architect, python-developer, qa-engineer, code-reviewer, and privacy-reviewer subagents.
- Research workspace and Ruff lint ([ADR 0004](decisions/0004-research-workspace-and-tooling.md)).

Known hardware: Windows laptop without a GPU, laptop camera or USB webcam. A GPU or Colab may become available later.

## Next action

The owner reviews T06 and T07 and requests adjustments, including a manual try of a `sketchloop.toml` with `run` and `ui`. Then start T08 in [ROADMAP.md](ROADMAP.md) with `/start-task`. Also try the tilt correction with a real diagonal sheet under the webcam, and rerun a diffusers round with `sketchloop rerun` to see the reported differences on a real backend. Later, improve conditioning for thin pencil strokes and run the diffusers backend on the Apple M2 Pro.

## Open questions

Software license for the repository. Saving the requested prompt and settings when preprocessing fails is a deferred record change. The page script that starts the webcam stream clicks Gradio's own controls, so it must be rechecked when Gradio is upgraded. Streamed webcam frames arrive as JPEG, so check whether faint pencil strokes survive. The one-minute CPU target in ADR 0006 is not met: one fast-mode image takes about 35-60 s of sampling and about 100 s in total on the laptop CPU.

## Validation

Privacy tests are consolidated into 12 cases covering private artifacts, public article exceptions, renamed PDFs,
sensitive patterns, staged ignore rules despite unstaged edits, and staged secrets in UTF-8 and both UTF-16 byte orders.
They share temporary-repository setup and check publication behavior instead of helper details. No open questions were
added by this change. The full repository check passed with 83 tests and the real-model test deselected.

After restoring standalone root resolution, the full repository check again passed with 83 tests. With site-packages
disabled and the working directory outside the checkout, the privacy guard ran successfully and the article script
resolved its target without importing the application or downloading files.

On Windows 11 with Python 3.12, `.venv\Scripts\python.exe scripts/check.py` passes all checks and 88 tests, and `python -m pytest -m model` passed the real-model test in about 75 s before T06. The T06 and T07 tests use the fake backend offline, and the model download through the settings file was not run. The owner checked the browser UI by hand on 2026-10-12, and this record does not say whether that run used the real model. The installer was verified earlier on Python 3.14 in a fresh copy, but not rerun on Windows. The owner captured a real sketch with the laptop webcam through `sketchloop --camera` on 2026-10-08, and the full flow completed. The owner ran the diffusers backend on the laptop CPU on 2026-10-09.

## Handoff discipline

Replace this summary after each task rather than appending. Keep long-lived decisions in ADRs and requirements in the specification. Use [the session template](templates/SESSION.md) for longer handoffs. Never put private source text or local absolute paths here.
