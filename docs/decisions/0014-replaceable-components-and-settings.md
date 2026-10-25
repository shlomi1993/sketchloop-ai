# ADR 0014: Replaceable components, an experiment store protocol, and a settings file

Status: accepted

## Context

T07 (R08, R10) must make capture, preprocessing, generation, evaluation, and storage replaceable through documented interfaces, with tests showing that orchestration does not change, and must let a person switch the model, image source, or storage through a setting without code changes. Generators and evaluators are already injected into `SketchSession` and `AppController` ([ADR 0005](0005-domain-records-and-generator-contract.md), [ADR 0010](0010-sessions-and-evaluator.md)). `SketchSession` still calls `preprocess_sketch` and writes files and records itself. [ADR 0011](0011-session-records-and-subcommands.md) deferred a store protocol to T07, and the folder store of [ADR 0006](0006-first-backend-store-and-ui.md) is the only persistent store. The CLI chooses components only through flags.

The owner decided that the setting is an optional TOML file read with the standard library plus overriding flags, that T07 adds no real model or scorer, and that storage gets an interface with the folder store as default and an in-memory store for tests. Reviewing the design, the owner also decided to keep the automatic `./sketchloop.toml` lookup, to add no `--random-seed` flag and no page defaults for source or rotation, to add a models-folder setting shared with the download script, and to add a typed store error.

## Decision

Adopt [the T07 design](../design/t07-modularity.md):

- **Protocols:** `CaptureSource` (`kind`, `capture()`) in `capture`, `Preprocessor` (`preprocess(payload, rotation)`) in `preprocessing`, and a write-side `ExperimentStore` (`root`, `locate_session`, `locate_round`, `save_session`, `save_round`) in `experiments`. `Generator` and `Evaluator` are unchanged. Existing functions become implementations: `FileCapture` and `CameraCapture` wrap `load_sketch_file` and `capture_from_camera`, `StrokePreprocessor` wraps `preprocess_sketch`, and `FolderStore` holds the writes that `SketchSession` did. `MemoryStore` keeps records and bytes in memory and checks that records encode as JSON.
- **Store errors:** both stores raise `StoreError`, a `SketchLoopError`, for anything they cannot save, with one line naming the file relative to the runs folder and the recovery, such as freeing disk space. The CLI and UI already report `SketchLoopError` messages, so they need no change. `SketchSession` marks `session.json` as saved only after it is saved, so the next round retries.
- **Orchestration:** `SketchSession(generator, store, evaluator=None, *, preprocessor=None, rerun_of=None)` and `AppController(generator, store, evaluator=None, *, preprocessor=None)`. Orchestration imports only protocols, records, and the default `NoOpEvaluator` and `StrokePreprocessor`. `RoundOutcome.folder` and `SketchSession.folder` become `Path | None`. `AppController` refuses a store without a local folder.
- **Registries:** a new `sketchloop.components` maps names to factories in `CAPTURE_SOURCES` (`file`, `camera`), `PREPROCESSORS` (`strokes`), `EVALUATORS` (`none`), and `STORES` (`folder`). Generators keep `GenerationBackend` and `make_generator`. Component-specific options live in a subtable named after the component, starting with `[generator.diffusers]` with `mode` and `models_dir`. `make_generator` gains a `models_dir` keyword.
- **Settings:** a new standard-library `sketchloop.settings` reads `--config PATH`, or else `./sketchloop.toml` when present, into frozen settings records with sections `generator`, `capture`, `preprocessing`, `evaluator`, `storage`, `run`, and `ui`. Precedence is flag, then file, then default. Unknown sections and keys, wrong types, and unknown names are rejected with one line naming the file and key. Resolution happens in `parse_args`, so every configuration error exits with code 2 before anything is created. The command prints which file it used, and the local file is ignored by git. A relative `storage.runs_dir` or `models_dir` resolves against the file's folder. Removing `seed` from the file means a random seed. `rerun` takes only storage, evaluator, and the models folder from settings and recreates everything else from the record.
- **Model download:** `scripts/download_models.py` gains `--config PATH`, uses the same lookup, and downloads into `generator.diffusers.models_dir`, so the script and the backend always agree. A separate `--models-dir` flag was rejected as a second way to name the same folder.
- **Memory store:** not selectable in settings, because `run` and the browser show saved files, `show` and `rerun` read folders, and a session lost at exit would undermine R06.
- **Records:** unchanged, at schema version 1. Components are already identified by backend identity, `input.source`, preprocessing step names, and the evaluator class name.

Alternatives considered:

- A full store protocol with loading, checks, and comparison would make `show` and `rerun` store-agnostic but needs a second implementation of integrity checks with no persistent user.
- A path-keyed byte sink would keep the folder layout inside orchestration.
- Resolving settings in `main` would report file errors with exit code 1 and change the existing `--mode` exit code.
- A flat `generator.mode` conflicts whenever the backend changes and the mode stays.
- Selecting the memory store in settings needs every front end to handle missing files.
- A dictionary registry for generators would duplicate `GenerationBackend`, which the CLI choices and `RERUNNABLE_BACKENDS` already use.
- Pydantic or another configuration library would add a dependency without a concrete need, since `tomllib` is in the standard library from Python 3.11.

## Consequences

Library callers must pass a store, for example `SketchSession(generator, FolderStore(Path("runs")))`, in place of a runs folder. Existing tests change only that call. Command-line behavior, messages, exit codes, and records stay the same when no settings file exists. A settings file in the working directory changes defaults silently except for the printed `Settings:` line, so the CLI tests run in a temporary working directory. The local `sketchloop.toml` is ignored by git, and `examples/sketchloop.toml` is the committed example. Replaceability is shown by a test that runs a session with a substitute capture source, preprocessor, evaluator, and memory store through unchanged orchestration, and by an import-boundary test. A full-disk test checks the one-line `StoreError` message through the CLI. `show` and `rerun` remain folder-only. Write failures now show `StoreError` text instead of the raw `OSError` text. A failed save is not recorded in `round.json`, because the store that failed cannot write it. Amended after T07: a failed round stays in the session history even when its record cannot be saved, and it still reports the stage's original error. With no settings file, the download script writes to `models` in the working folder instead of the repository folder, which is the same place when it runs from the repository root as documented.
