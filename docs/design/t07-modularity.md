# T07 design: replaceable components and a settings file

Status: accepted in [ADR 0014](../decisions/0014-replaceable-components-and-settings.md) and implemented in T07. Serves T07 in [ROADMAP.md](../ROADMAP.md): R08 for documented, replaceable component interfaces and R10 for a reusable integration layer. Runnable increment: switch the model, image source, or storage through a setting without code changes.

Impact: High in breadth and low in data risk. It touches the `capture`, `preprocessing`, `experiments`, orchestration (`session`, `app`), and `cli` boundaries and adds `settings` and `components`. It changes the `SketchSession` and `AppController` constructors. It does not change `round.json`, `session.json`, or `SCHEMA_VERSION`, and `run_round` keeps its signature and behavior.

## Current state

- **Already injected:** `SketchSession` and `AppController` receive a `Generator` and an optional `Evaluator`. `backends.make_generator` creates a generator by name.
- **Not injected:** `SketchSession` calls `preprocess_sketch` directly and writes files and records itself through `write_json_atomic`, `write_files`, `session_to_dict`, and `round_to_dict`. Capture happens in the callers. `cli.run_session` picks `load_sketch_file` or `capture_from_camera` from `--camera`, and `AppController` encodes browser input.
- **Configuration:** only command-line flags, with defaults repeated in the `run` and `ui` parsers.

## Owner decisions this design applies

1. The setting is an optional TOML file read with the standard library `tomllib`, and command-line flags override it, for `run`, `ui`, and `rerun`. Precedence is flag, then file, then built-in default.
2. T07 adds no new real model or scorer. Replaceability is proven with test substitutes.
3. Storage gets an interface with today's JSON folder store as the default and an in-memory store that tests use. This design decides that the memory store is not selectable in settings (see Options).

Answers to the design's open questions:

4. Keep the automatic `./sketchloop.toml` lookup next to `--config PATH`, print which file is used, and ignore the local file in git.
5. No `--random-seed` flag in T07. Removing `seed` from the file means a random seed.
6. No `[ui]` defaults for the page's source or rotation in T07.
7. Add `[generator.diffusers] models_dir`, used by the diffusers backend and by `scripts/download_models.py`, and resolve a relative path against the settings file's folder like `runs_dir`.
8. Add a `StoreError`, a `SketchLoopError` raised by the stores with a one-line message naming the file and the recovery, shown like other errors in the CLI and UI.

## Options

### Store interface shape

| Option | Description | Tradeoffs |
| --- | --- | --- |
| A, recommended | A write-side `ExperimentStore` protocol injected into `SketchSession`: save the session record, save a round's files and record, and say where files live locally. Loading, checking, and comparing stay as today's folder functions for `show` and `rerun`. | Small. Orchestration stops touching the filesystem. `show` and `rerun` stay folder-specific, which is honest because only one persistent store exists. |
| B | A full store protocol that also loads sessions, verifies checksums, and rebuilds requests, so `show` and `rerun` work with any store | `load_session`, integrity checks, and `compare_rounds` would need a second implementation for memory with no user. Builds for a second persistent store before one is needed. |
| C | A byte sink keyed by path, such as `write(relative_path, payload)` | Tiny, but `SketchSession` would keep building `session.json` and `round-<n>/` paths, so the folder layout stays in orchestration. |

### Where settings are resolved

| Option | Tradeoffs |
| --- | --- |
| In `parse_args`, recommended | Every configuration problem, from a flag or the file, is reported through `parser.error` with exit code 2 before anything is loaded or saved. Existing exit codes and the `--mode` check stay the same. |
| In `main`, after parsing | File errors would exit with code 1 like runtime errors, and the `--mode` check against a file-chosen backend would change from exit 2 to exit 1. |

### Backend-specific options

| Option | Tradeoffs |
| --- | --- |
| A subtable per backend, `[generator.diffusers] mode = "fast"`, recommended | Switching `backend` never conflicts with another backend's options, so `--backend fake` works with a file written for diffusers. New backends add their own subtable. |
| A flat `generator.mode` | Needs a cross-check that fails whenever the backend changes and the mode stays. |

### Memory store in settings

Not selectable. `run` prints saved file paths, the browser shows saved files through `allowed_paths`, and `show` and `rerun` read folders. A session that silently disappears when the process ends would also undermine R06 provenance. The memory store is a library and test implementation, and the registry still allows tests to select substitutes by name.

## Component contracts

Each protocol lives in the module of its boundary, next to its implementations. Orchestration imports only the protocols, the record types, and the two defaults it already had or gains (`NoOpEvaluator`, `StrokePreprocessor`).

| Component | Protocol (module) | Implementations | Registry key in settings | Registry values |
| --- | --- | --- | --- | --- |
| Generator | `Generator` (`generation`), unchanged | `FakeGenerator`, `DiffusersSketchGenerator` | `generator.backend` | `fake`, `diffusers` (the existing `GenerationBackend` and `make_generator`) |
| Capture | `CaptureSource` (`capture`), new | `FileCapture`, `CameraCapture` | `capture.source` | `file`, `camera` |
| Preprocessing | `Preprocessor` (`preprocessing`), new | `StrokePreprocessor` | `preprocessing.name` | `strokes` |
| Evaluation | `Evaluator` (`evaluation`), unchanged | `NoOpEvaluator` | `evaluator.name` | `none` |
| Storage | `ExperimentStore` (`experiments`), new | `FolderStore`, `MemoryStore` (not registered) | `storage.store` | `folder` |

### `SketchSource` moves to `domain`

`SketchSource = Literal["file", "camera", "webcam", "upload"]` moves from `experiments` to `domain`, so `capture` can name its kind without importing `experiments`. `experiments` imports it from `domain`, so `from sketchloop.experiments import SketchSource` keeps working. `require_known_source` stays in `experiments`. No value is added.

### `CaptureSource` (in `sketchloop.capture`)

```python
class CaptureSource(Protocol):
    @property
    def kind(self) -> SketchSource: ...  # Value recorded as input.source for sketches from this source

    def capture(self) -> tuple[ImageRef, bytes]: ...  # Raw sketch reference and encoded bytes, or InvalidSketchError


class FileCapture:
    def __init__(self, path: Path) -> None: ...  # kind "file", capture() returns load_sketch_file(path)


class CameraCapture:
    def __init__(self, camera_index: int, open_camera: Callable[[int], VideoSource] = open_webcam) -> None: ...
    # kind "camera", capture() returns capture_from_camera(camera_index, open_camera)
```

- `load_sketch_file`, `capture_from_camera`, and `encode_frame` stay public and unchanged. The classes are thin wrappers.
- `capture()` may be called again for a recapture. A file source rereads the file.
- Failures stay `InvalidSketchError` for an unreadable, cancelled, or missing capture.
- The browser is the capture device for the UI. `AppController.run_round` keeps receiving a file path or a frame and the source value from the page. T07 does not route browser input through `CaptureSource`.

### `Preprocessor` (in `sketchloop.preprocessing`)

```python
class Preprocessor(Protocol):
    def preprocess(self, payload: bytes, rotation: int) -> PreprocessedSketch: ...


class StrokePreprocessor:
    def preprocess(self, payload: bytes, rotation: int) -> PreprocessedSketch: ...  # returns preprocess_sketch(payload, rotation)
```

- Invariants: the result's `image.sha256` matches its `payload`, `image.path` is a relative path such as `sketch.png`, and `steps` lists every applied operation in order with its effective parameters. Step names identify the operations in `round.json`, so a new preprocessor must use step names that say what it did.
- `rotation` is one of 0, 90, 180, or 270. A preprocessor that cannot rotate must raise rather than ignore it.
- `use_raw` stays a per-round option of `run_round`. It skips the preprocessor entirely, as today.

### `ExperimentStore` (in `sketchloop.experiments`)

```python
class ExperimentStore(Protocol):
    @property
    def root(self) -> Path | None: ...  # Local folder that holds all saved sessions, None when files are not kept on disk

    def locate_session(self, session_id: str) -> Path | None: ...  # Session folder, which may not exist yet, or None

    def locate_round(self, session_id: str, number: int) -> Path | None: ...  # Round folder, or None

    def save_session(self, record: SessionRecord) -> None: ...  # Create the session and save its record

    def save_round(self, session_id: str, record: RoundRecord, files: Mapping[str, bytes]) -> None: ...
    # Save the files, then the record, replacing an earlier record of the same round number
```

- **Ordering:** `save_round` saves every file before the record, so a record never names a file that was not saved. A second call for the same round, with `files={}`, replaces only the record. `SketchSession` uses that to add the selection.
- **Failures:** a store raises `StoreError` for anything it cannot save, never a raw `OSError`. See the next section.
- **Paths:** keys of `files` are the `ImageRef.path` values, which the domain already restricts to safe relative POSIX paths.

`FolderStore(runs_dir: Path)` is today's behavior moved out of `SketchSession`:

| Member | Behavior |
| --- | --- |
| `root` | `runs_dir` |
| `locate_session(id)` | `runs_dir / id` |
| `locate_round(id, n)` | `runs_dir / id / f"round-{n}"` |
| `save_session(record)` | Creates the session folder with parents, then `write_json_atomic(folder / SESSION_FILE, session_to_dict(record))` |
| `save_round(id, record, files)` | Creates the round folder, `write_files(round_folder, files)`, then `write_json_atomic(round_folder / ROUND_FILE, round_to_dict(record))` |

`write_files` moves from `session.py` to `experiments.py` unchanged. `load_session`, `rebuild_request`, and `compare_rounds` stay folder functions.

### `StoreError` (in `sketchloop.experiments`)

```python
class StoreError(SketchLoopError):
    """
    A store could not save a session record, a round record, or a file.
    """
```

- `FolderStore` wraps every `OSError` from creating folders, writing files, and the atomic record write in `StoreError`, chained with `from error`. The message is one line that names the file relative to the runs folder and the recovery: `Cannot save <session-id>/round-2/round.json: No space left on device. Free disk space or check write access to the runs folder, then retry.` It uses `error.strerror`, or the class name when that is empty. A path relative to the runs folder keeps user names out of the message.
- `MemoryStore` raises `StoreError` when a record cannot be encoded: `Cannot keep <session-id>/round-2/round.json: <first line of the error>. Check the record values, then retry.`
- `SketchSession` does not catch it. A failed save of a complete round propagates out of `run_round` and appends no outcome, as a write failure does today. A failed save inside the failure path propagates chained to the original error. `_write_session_record` sets `session_record` only after `save_session` succeeds, so the next round retries saving `session.json` instead of skipping it.
- The CLI already reports `SketchLoopError` as `Round failed: <message>` inside a session and as `sketchloop error: <message>` elsewhere, and the UI turns it into `gr.Error`. No front-end change is needed. `OSError` stays in those `except` clauses for reads, such as a missing sketch file.
- `load_session` keeps raising `ExperimentRecordError` for structural faults. `StoreError` is only for saving.

`MemoryStore()` keeps everything in process memory for tests and library callers that need no files:

```python
class MemoryStore:
    sessions: dict[str, SessionRecord]  # Session record by session ID
    rounds: dict[tuple[str, int], RoundRecord]  # Latest round record by session ID and round number
    files: dict[tuple[str, int, str], bytes]  # File bytes by session ID, round number, and relative path
```

- `root`, `locate_session`, and `locate_round` return `None`.
- Before keeping a record, it encodes it with `session_to_dict` or `round_to_dict` and `json.dumps(..., allow_nan=False)`, discarding the text, so a record the folder store cannot write fails here the same way.
- It is not registered in `STORES`.

### Orchestration changes

```python
class SketchSession:
    def __init__(self, generator: Generator, store: ExperimentStore, evaluator: Evaluator | None = None, *,
                 preprocessor: Preprocessor | None = None, rerun_of: RerunLink | None = None) -> None: ...
    session_id: str  # make_session_folder_name(), unchanged format
    folder: Path | None  # store.locate_session(session_id), set at construction as today


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundOutcome:
    folder: Path | None  # store.locate_round(session_id, number), None for a store without files
    # other fields unchanged
```

- `store` replaces `runs_dir` as the second positional parameter and has no default, so storage is always an explicit choice. `preprocessor` defaults to `StrokePreprocessor()`, as `evaluator` defaults to `NoOpEvaluator()`.
- `_write_session_record` calls `store.save_session(record)` once, before the first round, as today. A complete round calls `store.save_round(session_id, record, files)` with the raw sketch, the generation sketch, and the candidates. A failed round calls it with the sketches made so far and the failed record, then re-raises. `record_selection` calls `store.save_round(session_id, updated_record, {})`.
- `_prepare_sketch` calls `self.preprocessor.preprocess(raw_payload, rotation)` in place of `preprocess_sketch`.
- `session.py` no longer imports `write_json_atomic`, `session_to_dict`, `round_to_dict`, `SESSION_FILE`, `ROUND_FILE`, or `preprocess_sketch`. Round numbering, lineage, timings, evaluator naming by class name, failure records, and selection rules are unchanged.

```python
class AppController:
    def __init__(self, generator: Generator, store: ExperimentStore, evaluator: Evaluator | None = None, *,
                 preprocessor: Preprocessor | None = None) -> None: ...
    runs_dir: Path  # store.root, kept so sketchloop.ui.launch_app is unchanged
```

- The constructor raises `SketchLoopError("The browser interface shows saved image files, so it needs a store that keeps files in a folder.")` when `store.root is None`.
- `start_new_session` passes the same store, evaluator, and preprocessor to the new `SketchSession`.
- `sketchloop.ui` does not change.

## Registries (new module `sketchloop.components`)

The module maps setting names to factories and checks names. The CLI and UI build components only through it, and library callers inject objects directly. Registries are plain module dictionaries, so a test or a library caller can add an entry, but there is no plugin discovery.

```python
CaptureFactory: TypeAlias = Callable[[CaptureSettings, Path | None], CaptureSource]

CAPTURE_SOURCES: Final[dict[str, CaptureFactory]] = {"file": ..., "camera": ...}
PREPROCESSORS: Final[dict[str, Callable[[PreprocessingSettings], Preprocessor]]] = {"strokes": ...}
EVALUATORS: Final[dict[str, Callable[[EvaluatorSettings], Evaluator]]] = {"none": ...}
STORES: Final[dict[str, Callable[[StorageSettings], ExperimentStore]]] = {"folder": ...}  # FolderStore(settings.runs_dir)

def make_capture_source(settings: CaptureSettings, sketch_file: Path | None) -> CaptureSource: ...
def make_preprocessor(settings: PreprocessingSettings) -> Preprocessor: ...
def make_evaluator(settings: EvaluatorSettings) -> Evaluator: ...
def make_store(settings: StorageSettings) -> ExperimentStore: ...
def check_component_names(settings: Settings, label: str) -> None: ...  # SettingsError naming the first bad key
```

- Every factory receives its own settings section, so a component can gain options later in a subtable named after it, such as `[evaluator.clip]`, the same way `[generator.diffusers]` works.
- The `file` capture factory raises `SettingsError` when `sketch_file` is None (see the messages table). The `camera` factory ignores `sketch_file`.
- Generators keep `GenerationBackend` and `RERUNNABLE_BACKENDS` in `sketchloop.backends`. `make_generator` gains one keyword, `make_generator(backend, mode=None, *, models_dir: Path = Path("models"))`, passed as `model_dir` to `DiffusersSketchGenerator` and ignored by the fake. `check_component_names` validates `generator.backend` against `GenerationBackend` and `generator.diffusers.mode` against `GenerationMode`.
- The diffusers backend's missing-folder message becomes `Model folder <folder> is missing. Run python scripts/download_models.py from the folder with your settings file and retry.`, so a person who moved `models_dir` downloads to the same place.

### Model download script

`scripts/download_models.py` reads the same settings with `find_settings_file` and `load_settings`. It gains one option, `--config PATH`, with the same lookup as the CLI: `--config`, else `./sketchloop.toml`, else defaults. It downloads into `settings.generator.diffusers.models_dir` and prints `Settings: <path>` and the target folder first. With no settings file, the default `models` folder resolves against the working directory, which matches where the backend reads by default. Today's script writes to the repository's `models/` regardless of the working directory, so both agree when run from the repository root, as the README instructs.

A separate `--models-dir` flag was rejected. It would be a second way to name the folder, and the script and the backend could disagree.

## Settings (new module `sketchloop.settings`)

Standard library only: `tomllib`, `dataclasses`, and `pathlib`. It imports `SketchLoopError` from `domain` and nothing heavier, so it validates shape and types but not component names, which `components` checks.

### File name and lookup

1. `--config PATH` on `run`, `ui`, and `rerun` names the file. A missing file is an error.
2. Otherwise `sketchloop.toml` in the current working directory is used when it exists.
3. Otherwise built-in defaults apply.

When a file is used, the command prints `Settings: <path as given>` before anything else, so an implicit file is never a surprise. A relative `storage.runs_dir` or `generator.diffusers.models_dir` in the file resolves against the file's folder, and a relative `--runs-dir` resolves against the working directory as today. `show` takes no settings. The local `sketchloop.toml` is ignored by git, because it may hold local paths, and `examples/sketchloop.toml` is the committed example.

### Schema

| Key | Type and allowed values | Default | Used by | Flag that overrides it |
| --- | --- | --- | --- | --- |
| `generator.backend` | `fake` or `diffusers` | `fake` | run, ui (rerun uses the recorded backend) | `--backend` |
| `generator.diffusers.mode` | `fast` or `quality` | absent, meaning the device default | run, ui, when the backend is diffusers | `--mode` |
| `generator.diffusers.models_dir` | non-empty path text, relative to the settings file | `models` | run, ui, rerun, when the backend is diffusers, and `scripts/download_models.py` | none |
| `capture.source` | `file` or `camera` | `file` | run | the sketch argument sets `file`, `--camera` sets `camera` |
| `capture.camera_index` | integer, at least 0 | 0 | run | `--camera-index` |
| `preprocessing.name` | `strokes` | `strokes` | run, ui | none |
| `preprocessing.rotation` | 0, 90, 180, or 270 | 0 | run | `--rotate` |
| `preprocessing.raw` | boolean | false | run | `--raw` or `--no-raw` |
| `evaluator.name` | `none` | `none` | run, ui, rerun | none |
| `storage.store` | `folder` | `folder` | run, ui, rerun | none |
| `storage.runs_dir` | non-empty path text | `runs` | run, ui, rerun | `--runs-dir` |
| `run.candidates` | integer, at least 1 | 4 | run | `--candidates` |
| `run.seed` | integer, at least 0 | absent, meaning random | run | `--seed` |
| `ui.candidates` | integer, at least 1 | 2 | ui | `--candidates` |
| `ui.seed` | integer, at least 0 | absent, meaning random | ui | `--seed` |
| `ui.capture_interval_seconds` | number greater than 0 | 0.5 | ui | `--capture-interval` |

- A key a command does not use is still validated and then ignored, so one file serves every command. The page keeps its own rotation and raw controls, so `ui` ignores `preprocessing.rotation` and `preprocessing.raw`.
- `rerun` takes `storage.*`, `evaluator.name`, and `generator.diffusers.models_dir` from settings. The models folder says where the weights are, not which weights, and the recorded model revisions still reveal a difference. It always recreates the backend, mode, controls, seed, rotation, and raw choice from the recorded round and reuses the stored processed sketch, so `generator.backend`, `generator.diffusers.mode`, `[capture]`, and `[preprocessing]` never affect a rerun.
- Component names with a single value (`preprocessing.name`, `evaluator.name`, `storage.store`) get no flag until a second value exists.
- TOML has no null, so an absent `seed` or `mode` means unset. Removing `seed` from the file gives a random seed. There is no flag to clear a seed set in the file, by the owner's decision.
- Integers reject booleans. `ui.capture_interval_seconds` accepts an integer or a float.

### Records

```python
class SettingsError(SketchLoopError, ValueError): ...  # A settings file or a combination of settings is invalid


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusersSettings:
    mode: str | None = None  # fast or quality, None for the device default
    models_dir: Path = Path("models")  # Folder holding the downloaded model folders

@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorSettings:
    backend: str = "fake"  # Registry name of the generation backend
    diffusers: DiffusersSettings = DiffusersSettings()  # Options used only by the diffusers backend

@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureSettings:
    source: str = "file"  # Registry name of the capture source for sketchloop run
    camera_index: int = 0  # Webcam index for the camera source

@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessingSettings:
    name: str = "strokes"  # Registry name of the preprocessor
    rotation: int = 0  # Clockwise degrees before preprocessing, for sketchloop run
    raw: bool = False  # Generate from the raw sketch and skip preprocessing, for sketchloop run

@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluatorSettings:
    name: str = "none"  # Registry name of the evaluator

@dataclass(frozen=True, slots=True, kw_only=True)
class StorageSettings:
    store: str = "folder"  # Registry name of the experiment store
    runs_dir: Path = Path("runs")  # Folder for saved sessions

@dataclass(frozen=True, slots=True, kw_only=True)
class RunSettings:
    candidates: int = 4  # Candidates per round for sketchloop run
    seed: int | None = None  # Base seed, None for random

@dataclass(frozen=True, slots=True, kw_only=True)
class UiSettings:
    candidates: int = 2  # Default candidates per round on the page
    seed: int | None = None  # Default base seed, None for random
    capture_interval_seconds: float = 0.5  # Seconds between kept webcam frames

@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    generator: GeneratorSettings = GeneratorSettings()  # Generation backend and its options
    capture: CaptureSettings = CaptureSettings()  # Image source for sketchloop run
    preprocessing: PreprocessingSettings = PreprocessingSettings()  # Preprocessor and run defaults
    evaluator: EvaluatorSettings = EvaluatorSettings()  # Candidate scorer
    storage: StorageSettings = StorageSettings()  # Experiment store and its folder
    run: RunSettings = RunSettings()  # Defaults for sketchloop run
    ui: UiSettings = UiSettings()  # Defaults for sketchloop ui


def find_settings_file(explicit: Path | None) -> Path | None: ...  # explicit, else ./sketchloop.toml if present, else None
def load_settings(path: Path | None) -> Settings: ...  # Defaults for None, SettingsError for any file problem
def override_settings(settings: Settings, overrides: Mapping[str, object]) -> Settings: ...  # Dotted key to value
```

`override_settings` takes keys such as `"run.candidates"` and returns a new `Settings`. An unknown key is a programming error and raises `KeyError`. The developer may implement validation with one small rule table instead of per-field code.

### Example file (`examples/sketchloop.toml`)

```toml
# SketchLoop settings. Copy to sketchloop.toml in the folder you run sketchloop from, or pass --config PATH.
# Command-line flags override these values. Remove a line to use the built-in default.

[generator]
backend = "diffusers"  # fake or diffusers

[generator.diffusers]
mode = "fast"  # fast or quality, omit for the device default
models_dir = "models"  # relative to this file, also used by scripts/download_models.py

[capture]
source = "camera"  # file or camera, for sketchloop run
camera_index = 0

[preprocessing]
name = "strokes"
rotation = 0  # 0, 90, 180, or 270, for sketchloop run
raw = false

[evaluator]
name = "none"

[storage]
store = "folder"
runs_dir = "runs"  # relative to this file

[run]
candidates = 4
# seed = 7  # omit for a random seed

[ui]
candidates = 2
capture_interval_seconds = 0.5
```

## CLI changes

- `run`, `ui`, and `rerun` gain `--config PATH`. `run` gains `--no-raw` through `argparse.BooleanOptionalAction`.
- Overridable options use `default=argparse.SUPPRESS`, so an option that was not given is absent from the namespace and the help formatter shows no misleading default. Each help text names its settings key and built-in default, for example `number of candidates to generate (settings run.candidates, default 4)`.
- `parse_args` parses, runs today's flag-only checks, finds and loads the file, applies the given flags with `override_settings`, runs `check_component_names` and the combination checks, and stores the result in `args.settings`. Any `SettingsError` becomes `parser.error`, which exits with code 2 before anything is created.
- Command functions read component choices only from `args.settings`:
  - `run_session` builds the capture source with `make_capture_source(settings.capture, args.sketch)`, captures the first sketch, and records `capture_source.kind`. The `c` menu choice is offered when `kind == "camera"` and recaptures through the same source. `f <path>` keeps calling `load_sketch_file` and records `file`. It builds `SketchSession(generator, make_store(...), make_evaluator(...), preprocessor=make_preprocessor(...))`, with rotation, raw, candidates, and seed from settings.
  - `launch_ui` builds `AppController` the same way, with `ui.*` defaults for `launch_app`.
  - `make_announced_generator` passes `settings.generator.diffusers.models_dir` to `make_generator` for `run`, `ui`, and `rerun`.
  - `rerun_round` builds the store and evaluator from settings, uses the settings' models folder, and keeps everything else from the record.
  - `run` and `rerun` raise `SketchLoopError("sketchloop <command> shows saved file paths, so it needs a store that keeps files in a folder.")` when the built store's `root` is None. Only test-registered stores can reach this.
- `show` is unchanged.

## Failure messages

`<file>` is the settings path as given. File messages are one line.

| Situation | Message |
| --- | --- |
| `--config` file missing | `Settings file <file> not found. Check the --config path.` |
| Invalid TOML | `<file> is not valid TOML: <tomllib message>. Fix the file and retry.` |
| Unknown section | `<file>: unknown section [<name>]. Use generator, capture, preprocessing, evaluator, storage, run, or ui.` |
| Section is not a table | `<file>: <section> must be a table such as [<section>], got <value>.` |
| Unknown key | `<file>: unknown key <section>.<key>. Keys in [<section>]: <comma-separated keys>.` |
| Wrong type | `<file>: <key> must be <an integer, a number, true or false, or text>, got <value>.` |
| Out of range | `<file>: <key> must be at least <minimum>, got <value>.` or `must be greater than 0` |
| Unknown name or value | `<file>: <key> must be one of <choices>, got "<value>".` (backend, mode, capture source, preprocessor, evaluator, store, rotation) |
| `--mode` with a non-diffusers effective backend | `--mode needs --backend diffusers` (unchanged) |
| Rotation with raw, both flags | `--rotate cannot be combined with --raw, which skips preprocessing` (unchanged) |
| Rotation with raw, either from the file | `preprocessing.rotation <n> cannot be combined with preprocessing.raw, which skips preprocessing. Change one in <file> or pass --rotate 0 or --no-raw.` |
| Sketch argument and `--camera` | `give exactly one of a sketch file or --camera` (unchanged) |
| No sketch and the effective source is `file` | `give a sketch file or --camera, or set capture.source = "camera" in the settings file` |
| Store without files in `run` or `rerun` | `sketchloop <command> shows saved file paths, so it needs a store that keeps files in a folder.` |
| `AppController` given a store without files | `The browser interface shows saved image files, so it needs a store that keeps files in a folder.` |
| Folder store cannot save | `Cannot save <session-id>/<file>: <strerror>. Free disk space or check write access to the runs folder, then retry.` |
| Memory store cannot encode a record | `Cannot keep <session-id>/<file>: <first line>. Check the record values, then retry.` |
| Diffusers model folder missing | `Model folder <folder> is missing. Run python scripts/download_models.py from the folder with your settings file and retry.` |

## File plan for the python-developer

Implement in this order, keeping each step green with `scripts/check.py`. Every step except 7 to 9 is a behavior-preserving refactor.

1. `src/sketchloop/domain.py`, `experiments.py`: move `SketchSource` to `domain` and import it in `experiments`.
2. `src/sketchloop/capture.py`: add `CaptureSource`, `FileCapture`, and `CameraCapture`.
3. `src/sketchloop/preprocessing.py`: add `Preprocessor` and `StrokePreprocessor`.
4. `src/sketchloop/experiments.py`: add `ExperimentStore` and `StoreError`, move `write_files` from `session.py`, and add `FolderStore` and `MemoryStore`, both raising `StoreError`.
5. `src/sketchloop/session.py`: inject `store` and `preprocessor`, add `session_id`, type `folder` and `RoundOutcome.folder` as `Path | None`, remove direct file writes and `preprocess_sketch`, and set `session_record` only after it is saved.
6. `src/sketchloop/app.py`: take `store` and `preprocessor`, refuse a store without `root`, and set `runs_dir = store.root`. Update existing constructor calls in tests to pass `FolderStore(...)`, with no other test edits in this step.
7. `src/sketchloop/settings.py`: settings records, `SettingsError`, `find_settings_file`, `load_settings`, and `override_settings`.
8. `src/sketchloop/components.py`: registries, `make_*`, and `check_component_names`. `src/sketchloop/backends.py`: the `models_dir` keyword of `make_generator`. `src/sketchloop/diffusers_backend.py`: the missing-folder message.
9. `src/sketchloop/cli.py`: `--config`, `--no-raw`, suppressed defaults, settings resolution in `parse_args`, the `Settings:` line, the models folder, and component construction through `components`.
10. `scripts/download_models.py`: `--config`, the settings lookup, and downloads into `models_dir`.
11. `examples/sketchloop.toml`, `/sketchloop.toml` in `.gitignore`, and a short README section on the settings file, its precedence, and `models_dir`.

No new dependency. `tomllib` is in the standard library from Python 3.11, which `pyproject.toml` already requires.

## Tests for the qa-engineer

Offline, with the fake backend and `tmp_path`. Prefer extending existing tests. Two kinds of new tests are justified, because they cover new behavior with no existing home.

**Isolation first.** Add a `tests/conftest.py` fixture `isolated_cwd` that changes to `tmp_path`, and apply it with `pytestmark = pytest.mark.usefixtures("isolated_cwd")` in `test_cli.py` and `test_experiments.py`. Otherwise a developer's own `sketchloop.toml` in the repository root would change test results.

1. **New: `test_session_runs_with_substituted_components`** in `test_experiments.py` (R08, R10). It uses a test `CaptureSource` that returns a synthetic PNG with kind `camera`, a test `Preprocessor` that returns the payload unchanged as `sketch.png` with one step named `identity`, `FakeGenerator`, a test evaluator that gives every candidate 0.5, and `MemoryStore`. It runs two rounds, the second with `use_raw=True`, records a selection and an explicit none, then a third round with `n_candidates=99` that fails in generation. It asserts:
   - The memory store holds one session record and three round records with statuses complete, complete, and failed, and their selections equal `session.selections`.
   - Round 1 has steps `(identity,)`, evaluator `"<test class name>"`, scores 0.5, and source `camera`, and its files are the raw sketch, `sketch.png`, and the candidates. Round 2 has no `sketch.png`. The failed round keeps the sketches made before generation.
   - `session.folder` and every outcome's `folder` are None, and `tmp_path` stays empty.
   - `AppController(FakeGenerator(), MemoryStore())` raises `SketchLoopError` with the folder-store message.
2. **Extend `test_cli_and_app_never_import_gradio` into an import-boundary test** (R08), parametrized by module. It checks with `ast` that `session.py` imports none of `write_json_atomic`, `write_files`, `FolderStore`, `MemoryStore`, `preprocess_sketch`, `load_sketch_file`, `capture_from_camera`, `cv2`, `sketchloop.backends`, `sketchloop.components`, `sketchloop.settings`, or `gradio`, and that `app.py` imports none of the store implementations, `write_json_atomic`, `preprocess_sketch`, `sketchloop.backends`, `sketchloop.components`, `sketchloop.settings`, or `gradio`. Together with test 1, this shows orchestration runs every substitute without changes.
3. **Extend `test_records_round_trip`** only by building the session with `FolderStore(tmp_path)`. It already proves the folder store keeps today's files, records, and checksums byte for byte (R06).
4. **Extend `test_cli_rounds_track_source` into the settings switch test** (R08, runnable increment). Write `tmp_path/sketchloop.toml` with `capture.source = "camera"`, `evaluator.name = "constant"`, `storage.store = "counting"`, `storage.runs_dir = "file-runs"`, `run.candidates = 3`, and `run.seed = 4`. Register a substitute camera source, a constant evaluator, and a counting `FolderStore` subclass with `monkeypatch.setitem` on the registries. Run `main(["run", "--prompt", "chair", "--candidates", "2"])` without `--camera` or `--runs-dir`, and assert:
   - Output starts with `Settings: sketchloop.toml`, and the session is under `tmp_path/file-runs`.
   - Sources are camera, camera, and file, and the backend is created once (today's assertions).
   - Each round has 2 candidates, so the flag beat the file, and round 1's first seed is 4, so the file beat the default.
   - Scores and the evaluator name are recorded, and the counting store saw every save.
5. **Extend `test_cli_ui_launches_controller`** (model switch). Add a call with `--config` naming a file with `generator.backend = "diffusers"`, `generator.diffusers.mode = "quality"`, `generator.diffusers.models_dir = "weights"`, `ui.candidates = 3`, and a `storage.runs_dir`. Replace `sketchloop.backends.DiffusersSketchGenerator` with a stub that records its mode and models folder and returns a `FakeGenerator`. Assert the stub got `quality` and `weights` under the file's folder, `default_candidates` is 3, and `controller.runs_dir` is under the file's folder. A further call with the same file and `--backend fake` must not call the stub, which shows the flag wins without a mode conflict.
6. **Extend `test_cli_rejects_invalid_options`** with a settings-text column and an expected message, asserting exit code 2, the message in stderr, and no runs folder. Cases:
   - A misspelled backend, a boolean `ui.candidates`, an unknown key `generator.mdoe`, and broken TOML.
   - Rotation 90 with `raw = true` from the file, and `storage.store = "memory"`, which must name `folder` as the only choice.
   - A missing `--config` path for `rerun`.
   - The existing `ui --mode fast` case without a file stays.
7. **Extend `test_cli_rerun_links_back`** (R06). Pass `--config` with a file setting `generator.backend = "diffusers"` and the runs folder. Replace `DiffusersSketchGenerator` with a stub that fails the test if called. The rerun must still recreate the fake backend from the record and save under the file's runs folder.
8. **Extend `test_camera_capture`** to go through `CameraCapture(0, open_camera=...)`, asserting `kind == "camera"` and the same bytes as before (R01).
9. **Failure path, inside test 1:** a `MemoryStore` subclass whose `save_round` raises `StoreError` once makes `run_round` raise `StoreError`, appends no outcome, and leaves the next round numbered after it, so a failed save never overwrites an earlier round.
10. **Extend `test_cli_session_survives_errors`** with a parameter for a full disk (persistence failure). Replace `os.replace` in `sketchloop.experiments` with a function that raises `OSError(errno.ENOSPC, "No space left on device")`, so saving `session.json` fails. Assert that the output shows `Round failed: Cannot save` with `session.json` and `Free disk space` on one line, that the message does not contain `tmp_path`, and that the session still reaches the menu and summary with exit code 0. The existing candidate-limit failure stays as the other parameter. The UI needs no new test, because it already turns any `SketchLoopError` into `gr.Error`.
11. **No test for the download script.** It needs the network. Its settings lookup reuses `find_settings_file` and `load_settings`, which tests 4 to 6 cover.

## Out of scope

- A new real model, scorer, or preprocessor, and a second persistent store.
- `show` and `rerun` for stores other than the folder store, and a store-agnostic loader.
- Routing browser input through `CaptureSource`.
- Plugin discovery through entry points, and settings in a user configuration folder.
- Recording the settings file or registry names in `session.json`. Each component is already identified by backend identity, `input.source`, preprocessing step names, and the evaluator class name.
- A `--random-seed` flag, and `[ui]` defaults for the page's source and rotation, by the owner's decision.
- Recording a failed save in `round.json`, which the store that failed cannot write.

## Open questions

None for T07. The owner answered all five design questions. They are listed under the owner decisions above.
