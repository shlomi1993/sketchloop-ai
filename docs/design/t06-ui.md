# T06 design: local Gradio interface and sketch input source

Status: accepted in [ADR 0012](../decisions/0012-gradio-ui-and-input-source.md) and implemented in T06. Serves T06 in [ROADMAP.md](../ROADMAP.md): R07 for the interface, R06 for the new `input.source` record field, and R04, R05, and R08 for selection, successive rounds, and the framework-free controller. Impact: Medium. It adds the `app` and `ui` boundaries and one additive field to `round.json`, and it changes no existing contract.

T06a is done: `sketchloop.backends` holds `GenerationBackend`, `RERUNNABLE_BACKENDS`, and `make_generator`, and `Generator.describe()` exists. Guidance for Gradio 6.30.0 is in [the Gradio implementation note](../research/gradio-ui-implementation.md). This design follows it except where noted.

## Owner decisions this design applies

1. Round records say where the raw sketch came from: `file`, `camera`, `webcam`, or `upload`.
2. A browser refresh keeps the same session. One controller per running app holds the loaded generator and the current `SketchSession`.
3. The UI history covers only the current session's rounds with prompt, status, and selection. `show` and `rerun` stay in the CLI.

## Record change: `input.source`

```json
"input": {"source": "webcam", "raw_sketch": {"path": "sketch-raw.png", "...": "..."}, "use_raw": false, "rotation": 0,
          "preprocessing": ["..."]}
```

| Value | Meaning | Written by |
| --- | --- | --- |
| `file` | A PNG or JPEG file read from disk | `sketchloop run <file>` and the `f <path>` menu choice |
| `camera` | A frame captured by OpenCV on the machine running sketchloop | `sketchloop run --camera` and the `c` menu choice |
| `webcam` | The newest browser webcam frame, streamed every `--capture-interval` seconds and saved by the server as a lossless PNG | The UI webcam input |
| `upload` | A file the person uploaded in the browser, with its bytes unchanged | The UI upload input |

- **Name and place:** the JSON key is `input.source`, first in the `input` block, because it describes the raw sketch. The record attribute is `RoundRecord.source: SketchSource | Unavailable`, after `raw_sketch`. `SketchSource = Literal["file", "camera", "webcam", "upload"]` lives in `sketchloop.experiments` next to `FailureStage`, because it is a persisted record value and `experiments` already imports nothing heavier than the domain.
- **Older sessions:** a `round.json` without `input.source` loads with `Unavailable("recorded before input sources were tracked")`. `SCHEMA_VERSION` stays 1. The change is additive: today's reader already ignores an unknown key in `input`, so new files open in the T05 code, and the new reader gives old files an explicit unavailable value. A default of `"file"` is rejected because T05 sessions include webcam captures, so it would fabricate provenance. A version bump is rejected because it would force a migration and make every existing session unreadable by `show` until one exists, for no gain in meaning. The rule, recorded in ADR 0012: adding an optional field whose absence has a documented unavailable value keeps the version. Renaming, removing, or changing the meaning of a field bumps it.
- **Rerun:** a rerun round records the original round's `source`, including `Unavailable` for an original without one. The field describes where the raw sketch bytes came from, and a rerun reuses those exact bytes. The rerun itself is already recorded once per session in `session.json` `rerun_of`, so a separate `rerun` value would duplicate it and would hide the physical origin from per-round analysis. `compare_rounds` does not compare `source`.
- **Validation:** `RoundRecord.__post_init__` requires `isinstance(source, Unavailable) or source in get_args(SketchSource)`. An unknown string on load raises `ExperimentRecordError` through the existing `convert_record` path: `round-<n>/round.json: a field is invalid: Round source must be file, camera, webcam, or upload, got 'scanner'.`
- **Writing:** `round_to_dict` writes `encode_value(record.source)`, so a copied unavailable value is written as `{"unavailable": "<reason>"}`.

`SketchSession.run_round` gains a required keyword-only `source: SketchSource | Unavailable`. It is required, without a default, so a new caller cannot silently mislabel a capture. The three existing calls in tests change accordingly.

## Module `sketchloop.app` (new, never imports gradio)

The application service for interactive front ends. It owns one generator and the current session and returns plain values. It is not thread-safe by itself. The UI serializes calls that change state (see Concurrency).

```python
NO_SKETCH_MESSAGE: Final = "No sketch yet. Take a webcam picture or upload a PNG or JPEG, then generate."

@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateView:
    candidate_id: str  # Candidate ID sent back when the person selects it
    label: str  # Caption such as "2", "2 (fake backend)", or "2, score 0.812"
    image_path: Path  # Stored candidate image under the round folder

@dataclass(frozen=True, slots=True, kw_only=True)
class RoundView:
    number: int  # Round number in the current session
    iteration_id: str  # Iteration ID a selection must name, so a stale page cannot record a choice for another round
    sketch_path: Path  # Sketch the generator received, processed or raw
    candidates: tuple[CandidateView, ...]  # Candidates in display order
    is_fake: bool  # Whether the fake backend produced the images
    generation_seconds: float  # Generation time, including the model load in the first round
    selection: str | None  # "1, 3" or "none" once recorded, None while not recorded

@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryRow:
    number: int  # Round number
    status: RoundStatus  # complete or failed
    prompt: str | None  # Requested prompt, None when preprocessing failed before a request existed
    outcome: str  # "selected 1, 3", "selected none", "not selected", or "failed during <stage>: <message>"

def describe_selection(iteration: Iteration, selection: SelectionEvent) -> str  # moved unchanged from cli.py

class AppController:
    def __init__(self, generator: Generator, runs_dir: Path, evaluator: Evaluator | None = None) -> None
    runs_dir: Path  # Parent folder of session folders
    session: SketchSession  # Current session, replaced by start_new_session
    @property
    def max_candidates(self) -> int  # generator.capabilities.max_candidates
    def describe_backend(self) -> str  # generator.describe()
    def run_round(self, sketch: Path | np.ndarray | None, source: SketchSource, prompt: str, *, n_candidates: int,
                  seed: int | None = None, rotation: int = 0, use_raw: bool = False) -> RoundView
    def record_selection(self, iteration_id: str | None, candidate_ids: Sequence[str]) -> RoundView
    def latest_round(self) -> RoundView | None
    def list_history(self) -> tuple[HistoryRow, ...]
    def start_new_session(self) -> None
```

Behavior:

- `__init__` stores the generator and evaluator and creates `SketchSession(generator, runs_dir, evaluator)`. No folder is written until the first round, as today.
- `run_round` checks, in order, before reading any file: a missing path raises `InvalidSketchError(NO_SKETCH_MESSAGE)`. A blank prompt raises `SketchLoopError("Enter a prompt, then generate.")`. A rotation outside `ROTATE_CODES` raises `SketchLoopError(f"Rotation must be one of 0, 90, 180, or 270, got {rotation}.")`. A rotation with `use_raw` raises `SketchLoopError("Rotation cannot be combined with the raw sketch option, which skips preprocessing.")`. It then calls `load_sketch_file(sketch_path)`, then `session.run_round(..., Guidance(prompt=prompt), source=source, ...)`, and returns the view. The prompt is recorded as given, not stripped. Errors from loading or from the round propagate unchanged. Input errors write nothing, and round errors leave the failed `round.json` that `SketchSession` already writes.
- `record_selection` enforces that the IDs belong to the round the person saw. With no completed round it raises `InvalidSelectionError("No round to select from yet. Generate first.")`. When `iteration_id` is not the latest completed round's iteration it raises `InvalidSelectionError(f"This page shows an older round. Reload the page to see round {latest}.")`. Comparing iteration IDs rather than round numbers stops a stale tab from recording "none" for a round it never showed, even after another tab starts a new session whose round has the same number. When that round already has a selection it raises `InvalidSelectionError(f"Round {latest} already has a recorded selection. Generate the next round.")`. An empty list records `record_no_selection`, otherwise `select_candidates`, which still rejects foreign or repeated IDs. It returns the updated view.
- `latest_round` returns the view of `session.rounds[-1]` with its recorded selection, or None before the first completed round. The UI calls it on page load, which is how a refresh keeps the session.
- `list_history` maps `session.records` (below) to rows, in round order, including failed rounds.
- `start_new_session` replaces `session` with a new `SketchSession` on the same generator and evaluator, so the loaded model is kept.
- Candidate labels: the 1-based number, then `" (fake backend)"` when fake, then `f", score {score:.3f}"` when the evaluator gave a score.

### `SketchSession` change for history

Replace the private `_latest_record` with a public `records: list[RoundRecord]` that holds every started round, complete or failed, in order. `run_round` appends the failed record before re-raising and the complete record after writing it. `record_selection` finds the record whose iteration is the latest completed round, replaces it with the selection, and rewrites its `round.json` as today. Nothing else in the write order changes.

## Module `sketchloop.ui` (new, the only module that imports gradio)

```python
def build_app(controller: AppController, *, default_candidates: int, default_seed: int | None) -> gr.Blocks
def launch_app(controller: AppController, *, default_candidates: int, default_seed: int | None) -> None
```

`build_app` builds `gr.Blocks(title="SketchLoop", analytics_enabled=False, delete_cache=(3600, 3600))` with:

| Component | Settings |
| --- | --- |
| Header `gr.Markdown` | `controller.describe_backend()`, and "All browser tabs share one session. Reloading keeps it." |
| Source `gr.Radio` | Choices `[("Webcam", "webcam"), ("Upload", "upload")]`, value `"webcam"`. Its `change` event toggles the two images' `visible` |
| Webcam `gr.Image` | `sources=["webcam"]`, `streaming=True`, `type="numpy"`, `webcam_options=gr.WebcamOptions(mirror=False)`. Its `.stream` event keeps only the newest frame, every `capture_interval_seconds` |
| Upload `gr.Image` | `sources=["upload"]`, `type="filepath"`, `image_mode=None`, `visible=False` |
| Prompt `gr.Textbox` | One line |
| Candidates `gr.Slider` | 1 to `controller.max_candidates`, step 1, value `default_candidates` |
| Seed `gr.Number` | `precision=0`, `minimum=0`, value `default_seed`, label "Seed (blank for the backend default)" |
| Rotation `gr.Dropdown` | `[0, 90, 180, 270]`, value 0 |
| Raw `gr.Checkbox` | "Use the raw sketch (skip preprocessing)" |
| Generate `gr.Button` | `variant="primary"` |
| Status `gr.Markdown` | One line per state, below |
| Generation sketch `gr.Image` | Not interactive, shows `RoundView.sketch_path` |
| Candidates `gr.Gallery` | `interactive=False`, `format="png"`, `columns=4`, values `(str(path), label)` |
| Picks `gr.CheckboxGroup` | Choices `(label, candidate_id)`, reset to `value=[]` for each new round |
| Record selection `gr.Button` | |
| New session `gr.Button` | |
| History `gr.Dataframe` | `interactive=False`, headers Round, Status, Prompt, Outcome. A missing prompt shows "not recorded" |
| Round `gr.State` | The displayed round number, per tab |

Two `gr.Image` inputs with a radio replace the note's single image with two sources, because a single component passes only a file path and cannot say whether it came from the webcam or an upload, which owner decision 1 needs.

Events, where every state-changing listener has `concurrency_id="session"` and `concurrency_limit=1`:

- `generate.click(run_generation, [source, webcam, upload, prompt, candidates, seed, rotation, raw], [gallery, sketch_view, picks, round_state, status])`, with `progress: gr.Progress = gr.Progress(track_tqdm=True)` and `progress(None, desc="Generating. The first round also loads the model.")`. It picks the path of the image the radio names, converts the seed with `None if seed is None else int(seed)`, and calls `controller.run_round`. Then `.then(show_history, None, [history])`, so a failed round still appears.
- `record.click(save_selection, [round_state, picks], [status])`, then `.then(show_history, None, [history])`.
- `new_session.click(start_session, None, [gallery, sketch_view, picks, round_state, status, history])`.
- `app.load(restore_view, None, [gallery, sketch_view, picks, round_state, status, history])`, not in the session concurrency group, so a reload during a slow round still renders.

Every handler converts `(SketchLoopError, OSError)` to `raise gr.Error(str(error)) from error`. Other exceptions reach Gradio, which shows a generic error with `show_error=False`. The session has still written the failed record. One shared helper turns a `RoundView | None` into the gallery, sketch, checkbox, state, and status outputs for generate, load, and new session.

Status lines:

| State | Text |
| --- | --- |
| No round yet | `No rounds yet. Take or upload a sketch, enter a prompt, and generate.` |
| Round ready | `Round {n} is ready after {seconds:.1f} s. Tick the candidates you like, or none, and record the selection.` |
| Fake backend | Prefix `**FAKE BACKEND:** these images are deterministic noise, not output from a real model.` |
| Selection recorded | `Round {n} selected: {selection}. Change the sketch, prompt, or settings and generate the next round.` |
| New session | `New session started. Its folder is created with the first round.` |

`launch_app` calls `build_app(...).launch(server_name="127.0.0.1", share=False, ssr_mode=False, inbrowser=True, allowed_paths=[str(controller.runs_dir.resolve())], max_file_size="20mb", show_error=False)`.

### Concurrency and refresh

One controller per process, created before `build_app` and closed over by the handlers. Never in `gr.State`, and never a module global. Shared `concurrency_id="session"` with a limit of 1 keeps a selection or new session from interleaving with a running round. A reload calls `restore_view`, which shows the latest completed round, its recorded selection, and the history. Two tabs share the session. A stale tab's selection is refused by the round-number check.

## CLI changes

- `Command.UI = "ui"` and a `ui` subparser with the same help texts and defaults as `run`: `--backend` (default fake), `--mode`, `--runs-dir` (default `runs`), `--candidates` (`make_int_parser(1)`, default 2, since a CPU round takes about 100 s per candidate), and `--seed` (`make_int_parser(0)`, default None), `--capture-interval` (seconds above 0, default 0.5). No `--share`, `--port`, or camera options.
- Move the `--mode` check out of `check_run_options` into `check_backend_options(parser, args)` and call it for `run` and `ui`.
- `launch_ui(args, console)`, in order:
  1. Set `os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"` and `os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"`, overriding any user value.
  2. `from sketchloop.ui import launch_app`, imported only here so the other commands start without gradio, which ADR 0012 later made a core dependency.
  3. `controller = AppController(make_announced_generator(args.backend, args.mode, console), args.runs_dir)`.
  4. If `args.candidates > controller.max_candidates`, raise `SketchLoopError(f"--candidates {args.candidates} is above the backend limit of {limit}. Use {limit} or fewer.")`.
  5. `launch_app(controller, default_candidates=args.candidates, default_seed=args.seed)`, which blocks until Ctrl+C.
- `run_session` tracks the source with the sketch: `"camera"` with `--camera`, otherwise `"file"`. `ask_next_round` takes and returns it as a third value: `c` gives `"camera"`, `f <path>` gives `"file"`, and other choices keep it.
- `rerun_round` passes `source=record.source`.
- `print_round_record` adds the source: `Input: processed sketch, rotation 0, from sketch-raw.png (source: webcam)`, using `format_value`, so an old record prints `(source: unavailable (recorded before input sources were tracked))`.
- `describe_selection` is imported from `sketchloop.app`.

`gradio` is not added to `TRACKED_PACKAGES`. Its version does not change generated output, because uploads keep their bytes and the browser encodes webcam PNGs. `input.source` already marks UI rounds.

## Failure messages

| Case | Message, shown by `gr.Error` in the UI or `sketchloop error:` in the CLI |
| --- | --- |
| Too many default candidates | `--candidates 9 is above the backend limit of 8. Use 8 or fewer.` |
| No sketch | `No sketch yet. Take a webcam picture or upload a PNG or JPEG, then generate.` |
| Blank prompt | `Enter a prompt, then generate.` |
| Rotation with raw | `Rotation cannot be combined with the raw sketch option, which skips preprocessing.` |
| Not PNG or JPEG, unreadable | Existing `InvalidSketchError` texts, which name only the file name |
| Unsupported controls or too many candidates | Existing `UnsupportedConfigurationError` text. The round is recorded as failed |
| Backend failure | Existing `GenerationBackendError` text, recorded as failed |
| Nothing to select | `No round to select from yet. Generate first.` |
| Stale tab | `This page shows an older round. Reload the page to see round 2.` |
| Selected twice | `Round 2 already has a recorded selection. Generate the next round.` |
| Unknown source on load | `round-1/round.json: a field is invalid: Round source must be file, camera, webcam, or upload, got 'scanner'.` |

## File plan for the python-developer

1. `src/sketchloop/experiments.py`: `SketchSource`, `MISSING_SOURCE_REASON`, `RoundRecord.source` with validation, writing in `round_to_dict`, and reading with the unavailable default in `round_from_dict`.
2. `src/sketchloop/session.py`: the required `source` keyword in `run_round`, and `records` replacing `_latest_record`.
3. `src/sketchloop/app.py`: `NO_SKETCH_MESSAGE`, the three view records, `describe_selection` moved from `cli.py`, view and row builders as module functions above the class, and `AppController`.
4. `src/sketchloop/cli.py`: source tracking in `run`, `rerun`, and `show`, `check_backend_options`, the `ui` subcommand, and `launch_ui`. Import `describe_selection` from `app`.
5. `src/sketchloop/ui.py`: `build_app` and `launch_app`.
6. `pyproject.toml`: `ui = ["gradio==6.30.0"]`. `.gitignore`: `.gradio/`.
7. README "Try it" paragraph for `sketchloop ui`, the progress table, traceability, and `docs/STATUS.md`.

## Tests for the qa-engineer

Default tests use `FakeGenerator`, `tmp_path`, and a PNG written with OpenCV to stand in for the browser's file. They need no camera or network.

1. R07, R06 (parametrized over `webcam` and `upload`): `AppController.run_round` returns a view with the requested candidate count, existing image paths under `<runs>/<session>/round-1/candidates/`, labels ending in "(fake backend)", and `is_fake`. `round.json` has `input.source` equal to the parameter, and the raw sketch's sha256 equals the input file's.
2. R07: a None path, a blank prompt, and rotation with raw each raise with the message above, and no session folder exists afterwards.
3. R04, R07: selecting candidates 1 and 3 returns a view with selection "1, 3" and writes it to `round.json`. An empty list records an explicit none. A second selection, a stale round number, and an ID from another round each raise `InvalidSelectionError`, and the stale case leaves `round.json` unchanged.
4. R05, R06: two rounds link round 2's `parent_id` to round 1. A third round with 99 candidates raises `UnsupportedConfigurationError`, `list_history` then has three rows with the third failed during generation, and `latest_round` is still round 2.
5. R07 (refresh): `latest_round` is None before a round and equals the returned view after it, including a recorded selection. `start_new_session` gives an empty history and a new folder, and the session keeps the same generator object.
6. R06: each source value round-trips through `load_session`. A `round.json` with `input.source` deleted loads with `Unavailable(MISSING_SOURCE_REASON)`, and `main(["show", folder])` returns 0 and prints "source: unavailable". Add `data["input"].update(source="scanner")` with match "source" to `test_invalid_records_rejected`.
7. R06: `run` from a file records `file`. `run --camera` with a monkeypatched capture records `camera`, and the `f <path>` menu choice makes round 2 `file`. `rerun` copies the original source, and copies `Unavailable` when the original lacks one.
8. R07: with a stub `sketchloop.ui` module in `sys.modules`, `main(["ui", "--runs-dir", tmp, "--seed", "5"])` calls the stub's `launch_app` with a controller for that runs folder, `default_candidates=2`, and `default_seed=5`, and both telemetry variables are set. Use `monkeypatch.delenv` first so they are restored. `--candidates 9` returns 1 with the limit message, and `ui --mode fast` without `--backend diffusers` exits with code 2.
9. R08: a subprocess running `import sys, sketchloop.cli, sketchloop.app` then `assert "gradio" not in sys.modules` succeeds.
10. R07: after a fixture that refuses non-local network connections, `build_app` with a fake-backed controller returns a `gr.Blocks` whose `analytics_enabled` is false. It never calls `launch`.

Manual checks on the laptop, recorded in `docs/STATUS.md`: the webcam frame is not mirrored, an uploaded RGBA PNG keeps its checksum, the Diffusers step progress appears in the browser, a 100 s round finishes without a timeout, a reload keeps the gallery and history, the radio toggles the two images, the history updates after a failed round, a blank seed field reaches the handler as None, and a second tab's stale selection is refused.

## Out of scope

`show` and `rerun` in the UI, past sessions in history, a share link or Colab mode, `--port`, server-side OpenCV preview in the UI, negative prompt and per-control inputs (the CLI has none either), a selected candidate fed back as the next sketch, multiple users, cancellation, streaming, and a schema migration. Saving the requested prompt and settings when preprocessing fails is a later record change.
