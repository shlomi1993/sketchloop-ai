# T05 design: session records, show, and rerun

Status: accepted in [ADR 0011](../decisions/0011-session-records-and-subcommands.md) and implemented in T05. Serves R06 and the record-integrity and reconstruction parts of R09.

Today `SketchSession` ([ADR 0010](../decisions/0010-sessions-and-evaluator.md)) writes `runs/<YYYYMMDD-HHMMSS-6hex>/round-<n>/` with the raw sketch, `sketch.png`, and `candidates/candidate-<i>.png`, but keeps requests, settings, timings, and selections in memory only. T05 adds two JSON files to that layout, a loader, and the `run`, `show`, and `rerun` subcommands. Standard library only ([ADR 0006](../decisions/0006-first-backend-store-and-ui.md)).

## JSON files

Common rules: UTF-8, `indent=2`, `allow_nan=False`, keys in the fixed order the to-dict functions write. Times are ISO 8601 UTC. An unknown value is `{"unavailable": "<reason>"}`, which is unambiguous because no control or setting value is an object. `ImageRef` is `{"path", "width", "height", "mode", "media_type", "sha256"}` with a relative POSIX path inside the round folder.

`session.json`, written once before the first round's files:

```json
{"schema_version": 1, "session_id": "20261009-142233-a1b2c3", "created_at": "2026-10-09T11:22:33+00:00",
 "environment": {"python": "3.12.7", "platform": "Windows 11 AMD64",
                 "packages": {"sketchloop": "0.1.0", "diffusers": "0.40.0", "torch": {"unavailable": "not installed"}}},
 "rerun_of": null}
```

- `session_id` is the folder name. `rerun_of` is `{"session_id", "round", "iteration_id"}` for a rerun session.
- `packages` covers a fixed list of sketchloop, numpy, opencv-python, rich, diffusers, torch, transformers, accelerate, peft, safetensors, and huggingface_hub through `importlib.metadata`. `platform` uses `platform.system()`, `release()`, and `machine()` only, never host or user names.

`round-<n>/round.json`:

```json
{"schema_version": 1, "round": 2, "status": "complete", "started_at": "2026-10-09T11:25:02+00:00",
 "input": {"raw_sketch": {"path": "sketch-raw.png", "...": "..."}, "use_raw": false, "rotation": 0,
           "preprocessing": [{"name": "rotate", "params": {"degrees_clockwise": 0}}, "..."]},
 "iteration_id": "9f2e...", "parent_id": "3c41...",
 "request": {"sketch": {"path": "sketch.png", "...": "..."}, "guidance": {"prompt": "a lamp", "negative_prompt": null,
             "controls": {}}, "n_candidates": 4, "seed": null},
 "result": {"backend": {"adapter": "sketchloop.diffusers", "adapter_version": "1", "execution": "in_process", "model_id": "..."},
            "effective": {"prompt": "a lamp", "negative_prompt": null, "controls": {"steps": 4, "mode": "fast", "device": "cpu"}},
            "candidates": [{"id": "a1b2...", "index": 0, "seed": 1234, "image": {"path": "candidates/candidate-1.png", "...": "..."}}]},
 "timings_seconds": {"preprocessing": 0.41, "generation": 98.2, "evaluation": 0.0},
 "evaluation": {"evaluator": "NoOpEvaluator", "scores": {}},
 "selection": {"id": "77d0...", "candidate_ids": ["a1b2..."], "created_at": "2026-10-09T11:27:40+00:00"},
 "failure": null}
```

- `request` is what was asked. `result.effective` and the candidate seeds are what the backend reports it used. Both stay.
- `parent_id` is null in round 1. A rerun session's round 1 also has a null parent and links back through `rerun_of`.
- `selection`: null means none recorded, because the session stopped before a choice or because it is a rerun, which does not ask. `"candidate_ids": []` is an explicit no-selection.
- `evaluation.scores` is empty when the evaluator gives none. `evaluator` is its class name.
- `timings_seconds.generation` in a session's first round includes the lazy model load. `preprocessing` is absent when skipped (`--raw`) or reused (rerun).
- A failed round has `status: "failed"`, `failure: {"stage": "preprocessing" | "generation" | "evaluation", "error": "<class name>", "message": "<one line>"}` with absolute paths replaced by `<path>`, null `iteration_id`, `result`, `evaluation`, and `selection`, and a null `request` if preprocessing failed. Its raw sketch is still saved.

## Module `sketchloop.experiments` (new)

```python
class ExperimentRecordError(SketchLoopError, ValueError): ...
RerunLink(session_id: str, round: int, iteration_id: str)
Environment(py_version: str, platform: str, packages: Mapping[str, str | Unavailable])
SessionRecord(schema_version: int, session_id: str, created_at: datetime, environment: Environment, rerun_of: RerunLink | None)
RoundFailure(stage: Literal["preprocessing", "generation", "evaluation"], error: str, message: str)
RoundRecord(schema_version: int, number: int, status: Literal["complete", "failed"], started_at: datetime, raw_sketch: ImageRef,
            use_raw: bool, rotation: int, steps: tuple[PreprocessingStep, ...], request: GenerationRequest | None,
            iteration: Iteration | None, timings_seconds: Mapping[str, float], evaluator: str | None,
            scores: Mapping[str, float], selection: SelectionEvent | None, failure: RoundFailure | None)
SavedSession(folder: Path, session: SessionRecord, rounds: tuple[RoundRecord, ...], problems: tuple[str, ...], checked_files: int)
RerunComparison(identical_candidates: tuple[int, ...], candidate_count: int, differences: tuple[str, ...])

def collect_environment() -> Environment
def session_to_dict(record: SessionRecord) -> dict[str, object]      # and session_from_dict
def round_to_dict(record: RoundRecord) -> dict[str, object]          # and round_from_dict
def write_json_atomic(path: Path, data: Mapping[str, object]) -> None
def load_session(folder: Path) -> SavedSession
def rebuild_request(record: RoundRecord, capabilities: GeneratorCapabilities) -> GenerationRequest
def compare_rounds(original: SavedSession, round_number: int, rerun: SavedSession) -> RerunComparison
```

All records are frozen keyword-only dataclasses. Invariants: `status == "complete"` exactly when `iteration` is set and `failure` is None, `request is iteration.request` when complete, and selection IDs belong to the iteration. From-dict functions rebuild domain records through their constructors, so existing invariants run on load. `write_json_atomic` writes `<name>.tmp` in the same folder, flushes and fsyncs, then calls `os.replace`.

Write order in `SketchSession`: `session.json` before the first round, then per round the artifacts, then `round.json` with a null selection. `record_selection` replaces that `round.json` once with the selection. This is the only rewrite. A crash keeps every earlier round. On an exception during preprocessing, generation, or evaluation the session saves the raw sketch and a failed `round.json`, then re-raises.

`load_session` raises `ExperimentRecordError` for structural faults: missing or invalid `session.json` or `round.json`, an unknown schema version, a field that fails validation, a round number that does not match its folder, a gap in round numbers, a `parent_id` that does not match the previous complete round, a selection outside its round, or an image path that resolves outside its round folder. It collects artifact faults in `problems` instead, so `show` can still display the record: a missing file, a checksum mismatch, or a round folder without `round.json` (interrupted). It ignores `*.tmp` files.

## Subcommands

- `sketchloop run <sketch> | --camera ...`: today's behavior and flags unchanged, now also writing the records.
- `sketchloop show <session-folder>`: loads without constructing a generator or importing torch. Prints the session header with ID, time, Python, platform, installed generation packages, and `rerun_of`. For each round it prints the status, prompt, negative prompt, requested controls and seed, raw or processed input with rotation, preprocessing step names, backend and model ID (with the fake warning), effective settings on one line, timings, a candidate table with seed, score, and a selected mark, the selection ("1, 3", "none", or "not recorded"), and the failure. It ends with the problems, or "All N files match their recorded checksums." For a rerun session whose original still loads, it also prints the comparison. Exit code 0 when the records load, even with problems.
- `sketchloop rerun <session-folder> --round N [--runs-dir runs]`, in order:
  1. Load the session. Refuse a missing round, a failed round, or a round with problems.
  2. Recreate the generator from `result.backend.adapter`: `sketchloop.fake` gives `FakeGenerator()`, and `sketchloop.diffusers` gives `DiffusersSketchGenerator(mode=<effective "mode">)`. The mode is passed explicitly because its default depends on the device.
  3. `rebuild_request` keeps the sketch, prompt, negative prompt, and candidate count. It pins the effective value of every control the new generator declares, so changed defaults cannot leak in, and pins `seed` to the first candidate's effective seed, or to None with a difference line when unavailable.
  4. Run one round in a new session with `rerun_of` set. `run_round` gains `reuse: PreprocessedSketch | None`, built from the stored `sketch.png` bytes and the original steps, so generation gets byte-identical input. No selection prompt.
  5. Print `compare_rounds`: "Candidates with identical bytes: 1, 2 of 4", then one line per difference in backend identity, effective prompt and settings, covering device, dtype, scheduler, seeds, and versions, Python, platform, and packages, such as "device: cuda -> cpu". Always end with "A rerun restores the recorded conditions. Equal seeds do not guarantee identical images across devices, library versions, or model revisions." Never print "reproduced".

## Failure messages

| Case | Error (one line, through the existing `sketchloop error:` handler) |
| --- | --- |
| No `session.json` | `No session.json in <folder>. Give a session folder created by sketchloop run.` |
| Invalid JSON | `<file> is not valid JSON: <decoding error with line>. Restore the file or inspect it by hand.` |
| Schema version | `<file> has schema version <v>, but this sketchloop reads 1. Upgrade sketchloop.` |
| Invalid field | `<file>: <field> is invalid: <reason>.` |
| Path escapes folder | `<file> references <path> outside its round folder. Refusing to load it.` |
| Broken lineage | `round-<n> does not link to round-<n-1>. The records were edited or mixed.` |
| Missing or changed file (problem) | `round-<n>/<path> is missing.` and `round-<n>/<path> does not match its recorded checksum.` |
| Round absent or failed | `Session <id> has no completed round <n>. Completed rounds: <list>.` |
| Round has problems | `Round <n> fails integrity checks. Run sketchloop show <folder> for details.` |
| Unknown adapter | `Round <n> used backend <adapter>, which rerun cannot recreate.` |
| Write failure | Existing `OSError` path. The temporary file may remain and is ignored on load. |

## File plan for the python-developer

1. `src/sketchloop/experiments.py`: errors, records, environment collection, to-dict and from-dict functions, atomic write, `load_session`, `rebuild_request`, `compare_rounds`.
2. `src/sketchloop/session.py`: optional `rerun_of`, per-stage timings in `RoundOutcome` (replacing `elapsed_seconds`), the `reuse` parameter, record writes, and failed-round recording.
3. `src/sketchloop/cli.py`: `add_subparsers(required=True)` with `run`, `show`, and `rerun`. Move today's options under `run`, and keep printing and prompts in the CLI.
4. README usage, the `experiments` row and `ExperimentStore` line in `docs/ARCHITECTURE.md`, traceability, and `docs/STATUS.md`. Update existing CLI tests to the `run` subcommand.

## Tests for the qa-engineer (fake backend, `tmp_path`, offline)

1. R06: two rounds with a selection, then `load_session` returns equal requests, effective settings, seeds, steps, selection, timings keys, and a round-2 parent equal to round 1's iteration ID. Environment values are strings or `Unavailable`.
2. R06, R09 (parametrized): a flipped candidate byte or a deleted candidate gives one entry in `problems`. An `ImageRef` path of `../x.png`, `schema_version: 2`, or an edited `parent_id` raises `ExperimentRecordError`.
3. R06: a generator stub that fails in round 2 leaves a loadable round 1 and a failed round 2 with stage `generation`. A stray `round.json.tmp` is ignored.
4. R06, R09: `main(["run", ...])`, then `main(["rerun", folder, "--round", "1"])` creates a new session whose `rerun_of` names the original and whose candidates match the fake's bytes. `main(["show", folder])` returns 0 without a generator.

## Out of scope

Retries within one session, schema migrations (only version 1 exists), a store protocol (T07), MLflow export, sessions saved before T05, storing the conditioning image, concurrent writers, and feeding a selected candidate back as a sketch.
