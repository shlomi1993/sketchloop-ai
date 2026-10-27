# T08 design: stage timings, hardware facts, and a benchmark command

Status: accepted in [ADR 0017](../decisions/0017-stage-timings-hardware-and-benchmark.md) and implemented. Serves the first slice of T08 in [ROADMAP.md](../ROADMAP.md) for R09, with record changes under R06 and interface changes under R01 and R08. Runnable increment: every run prints how long each stage took, with the hardware it ran on.

Impact: Medium. It touches `capture`, `generation`, `backends.diffusers`, `experiments`, orchestration (`session`, `app`), `cli`, and `ui`. Two component interfaces change: `CaptureSource.capture()` returns a record, and `ExperimentStore` gains `save_files`. Every record change is additive, so `SCHEMA_VERSION` stays 1 and older sessions still load.

The "Measured latency target?" decision stays open. This slice measures a baseline and makes no real-time claim. Optimizing bottlenecks and judging whether continuous operation is feasible are later T08 slices that use the benchmark results.

## Owner decisions this design applies

1. Both interface changes are approved: `CaptureSource.capture()` returns `CapturedSketch`, and `ExperimentStore` gains `save_files`.
2. Benchmark defaults are 5 measured rounds and 1 warm-up round. When no seed is given, benchmark pins seed 0, unlike `run`, so repeated benchmarks generate the same images. The fewer-than-20-rounds warning stays.
3. Each session records how long creating its generator took before round 1, which is separate from the lazy pipeline load in the `model_load` stage.
4. A round from a streamed webcam frame records how old the frame was when the round started.

## Current state

- `SketchSession.run_round` times `preprocessing`, `generation`, and `evaluation` into `RoundRecord.timings_seconds`. Generation in the first diffusers round includes the model load, because `DiffusersSketchGenerator._generate_candidates` loads the pipeline lazily.
- `RoundStage.SAVING` exists only as a failure stage. Capture happens before `run_round`: `capture_source.capture()` in `cli/run.py`, and `load_sketch_file` or `encode_frame` in `AppController.run_round`. No capture time and no round total are recorded.
- `Environment` holds Python, platform, and package versions. Nothing records the CPU, core count, or memory. The diffusers backend records `device` and `dtype` in its effective controls.
- `show_round` prints only the generation time, `show` prints every recorded stage, and `RoundView.generation_seconds` feeds the page status line.

## Options

### How capture time reaches the record

| Option | Description | Tradeoffs |
| --- | --- | --- |
| A, recommended | `CaptureSource.capture()` returns a `CapturedSketch` with the image, bytes, and `capture_seconds`, and each source times its own work. `run_round` takes `capture_seconds`. | The camera source can start its clock when Space is pressed, so the person's preview time is never counted as latency. It changes one interface and its two implementations and test substitutes. |
| B | Callers time the `capture()` call and pass `capture_seconds`. | No interface change, but a camera capture would include however long the person looked at the preview, which is not system latency. |
| C | As B, but leave the CLI camera untimed. | Smallest, but `run --camera`, the main R01 path, would miss a stage, which breaks "every run prints how long each stage took". |

### How saving time gets into the record it describes

| Option | Description | Tradeoffs |
| --- | --- | --- |
| A, recommended | Add `ExperimentStore.save_files`. The session times the sketch and candidate writes, then writes `round.json` once with that time and the total. | The record keeps one write per round plus the selection, as in ADR 0011. The final `round.json` write, a few milliseconds, is outside the measurement and documented as such. |
| B | Keep the store interface, time `save_round`, then rewrite `round.json` to add the time. | Measures the first record write too, but adds a second record write and fsync to every round and a new failure point after the files are saved. |
| C | Keep saving time in memory only and print it. | The printed and saved timings would differ, which undermines R06. |

### Model load time

| Option | Description | Tradeoffs |
| --- | --- | --- |
| A, recommended | The backend reports the load it did during a call as `GenerationOutput.load_seconds`, and the session records it as stage `model_load` and subtracts it from `generation`. | The backend says what it actually did, and a remote backend can report its own load or none. |
| B | Add a `Generator.load()` method that the session calls and times in every round. | The session cannot tell a real load from a no-op, so every round would record a near-zero load. |
| C | Load eagerly when the backend is created. | Changes startup for `run` and `ui`, and the load would happen outside any round record. |

### Benchmark results for T09

| Option | Description | Tradeoffs |
| --- | --- | --- |
| A, recommended | Benchmark rounds are ordinary saved rounds, and `session.json` gains a `benchmark` plan naming the warm-up and measured round counts. A pure function recomputes the statistics, and `benchmark` and `show` print them. | No derived data to keep in sync, and T09 reuses `load_session` plus `summarize_benchmark`. |
| B | Also write a derived `benchmark.json` summary. | Easier to read outside Python, but it duplicates the rounds and needs a new store method. It can be added later as an export. |

## Contracts

### Timings (`sketchloop.experiments.records`, `sketchloop.session`)

`RoundStage` gains two members, in this order, which is also the print order:

```python
class RoundStage(StrEnum):
    CAPTURE = "capture"
    PREPROCESSING = "preprocessing"
    MODEL_LOAD = "model_load"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    SAVING = "saving"
```

Stage meanings, all measured with `time.perf_counter()` and recorded in seconds:

| Key | Covers | Absent when |
| --- | --- | --- |
| `capture` | From the trigger to encoded raw bytes: file read and decode, or camera frame encode after Space including camera release, or browser frame encode or upload load | The caller passes no capture time, as `rerun` does when it reuses the stored sketch |
| `preprocessing` | `Preprocessor.preprocess` | `--raw` or a rerun reusing the processed sketch |
| `model_load` | The load the backend reports in `GenerationOutput.load_seconds` | The backend loaded nothing during the call, which is every round after the first |
| `generation` | `Generator.generate` minus `model_load` | Generation did not start |
| `evaluation` | `Evaluator.evaluate` | Evaluation did not start |
| `saving` | `ExperimentStore.save_files` for the raw sketch, processed sketch, and candidates | Saving did not start. The final `round.json` write is not included, since that record holds the time |

`RoundRecord` gains one field:

```python
total_seconds: float | Unavailable  # Capture time plus wall time of run_round until the files were saved
```

Invariants:

- Every `timings_seconds` value and a float `total_seconds` are finite and at least 0, checked in `__post_init__` with `require`.
- `total_seconds` starts at the first statement of `run_round` and ends after `save_files` returns, then adds `capture_seconds` when given. It is at least the sum of the stages after capture, and the remainder is bookkeeping such as writing `session.json` in round 1. Time between capture and the `run_round` call, such as creating the backend before round 1 in `run`, is outside the round.
- A failed round records the stages that finished and `total_seconds` up to the failure. A model load that fails is a `generation` failure, as today.
- The record is still built and validated at the evaluation stage, so a bad score still fails as `evaluation`. After `save_files` the session replaces its timings and total with `dataclasses.replace` and writes `round.json` once.

`SketchSession.run_round` gains one keyword, and `RoundOutcome` gains one field:

```python
def run_round(self, raw_sketch: ImageRef, raw_payload: bytes, guidance: Guidance, *, source: SketchSource | Unavailable,
              rotation: int = Rotation.DEG_000, use_raw: bool = False, n_candidates: int = 1, seed: int | None = None,
              reuse: PreprocessedSketch | None = None, capture_seconds: float | None = None) -> RoundOutcome: ...

total_seconds: float  # Capture plus the round's wall time until its files were saved
```

`round.json` adds `"total_seconds"` right after `"timings_seconds"`, encoded with `encode_value`. A record without the key loads with `Unavailable(MISSING_TOTAL_REASON)`, where `MISSING_TOTAL_REASON = "recorded before total round time was measured"`. Older records keep their `timings_seconds` as written, so their `generation` may include a model load. `show` says so for any round whose total is unavailable.

### Capture (`sketchloop.capture`)

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class CapturedSketch:
    image: ImageRef  # Raw sketch reference with size and checksum
    payload: bytes  # Encoded raw sketch bytes
    capture_seconds: float  # Time from the trigger to encoded bytes, excluding any wait for the person
```

- `CaptureSource.capture(self) -> CapturedSketch` replaces the tuple return. `FileCapture` times `load_sketch_file`.
- `capture_from_camera(...) -> CapturedSketch` starts its clock when `preview_until_capture` returns and stops it after `encode_frame`, so it includes camera release and window closing but not the preview.
- `load_sketch_file` and `encode_frame` keep their tuple returns. `AppController` times them in a helper `capture_browser_sketch(sketch: Path | np.ndarray) -> CapturedSketch` in `app.py`. A streamed webcam frame's age is recorded separately, as described below.
- `TerminalSession.sketch` becomes a `CapturedSketch`, and a new file chosen from the menu loads through `FileCapture(path).capture()`.

### Model load (`sketchloop.generation`, `sketchloop.backends.diffusers`)

```python
# GenerationOutput gains, validated as None or finite and >= 0:
load_seconds: float | None = None  # Seconds this call spent loading models, None when it loaded none
```

`DiffusersSketchGenerator.generate` moves the lazy load out of `_generate_candidates`: when `self._runner is None`, it times `self._load_runner()` and passes the time to `GenerationOutput`. An injected runner means no load and `load_seconds=None`. `FakeGenerator` reports `None`.

The diffusers effective controls gain runtime facts read once in `_load_runner` through a helper `describe_torch_runtime(torch_module: Any, device: TorchDevice) -> dict[str, ControlValue]`:

- `torch_threads`: `torch.get_num_threads()`, on every device, since it dominates CPU latency.
- `accelerator`: `torch.cuda.get_device_name(0)` on CUDA only. On MPS the session's CPU name names the chip, and on CPU there is no accelerator.

With an injected runner these facts are absent. A rerun of a round recorded before T08 will report them as new differences, which is accurate.

### Hardware (`sketchloop.experiments.records`, new `sketchloop.experiments.hardware`)

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Hardware:
    cpu: str | Unavailable  # Processor model name, such as "Apple M2 Pro"
    logical_cores: int | Unavailable  # Logical CPU count the operating system reports
    memory_bytes: int | Unavailable  # Total physical memory
```

`Environment` gains `hardware: Hardware | Unavailable  # CPU, cores, and memory of the machine`. The accelerator the backend actually used stays per round in the effective controls (`device`, and `accelerator` on CUDA), because it is the backend's choice and can differ between rounds of different sessions on one machine.

Readers, all standard library, each returning `Unavailable` with a fixed reason instead of raising, and never including exception text that could contain a path:

| Fact | macOS | Linux | Windows |
| --- | --- | --- | --- |
| `cpu` | `subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)` | First `model name` line of `/proc/cpuinfo` | `winreg` value `ProcessorNameString` under `HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0` |
| `logical_cores` | `os.cpu_count()` | `os.cpu_count()` | `os.cpu_count()` |
| `memory_bytes` | `sysctl -n hw.memsize` | `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")` | `ctypes.windll.kernel32.GlobalMemoryStatusEx` with a `MEMORYSTATUSEX` structure, field `ullTotalPhys` |

Functions in `sketchloop/experiments/hardware.py`:

```python
HARDWARE_COMMAND_TIMEOUT_SECONDS: Final = 5
def read_sysctl(name: str) -> str | Unavailable: ...
def read_linux_cpu_name(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | Unavailable: ...
def read_windows_cpu_name() -> str | Unavailable: ...
def read_windows_memory_bytes() -> int | Unavailable: ...
def read_cpu_name() -> str | Unavailable: ...  # Dispatches on sys.platform
def read_memory_bytes() -> int | Unavailable: ...  # Dispatches on sys.platform
def collect_hardware() -> Hardware: ...
```

`collect_environment()` calls `collect_hardware()`. The CPU name is stripped, has whitespace collapsed, and is cut to 120 characters. Other platforms get an `Unavailable` that names which fact this operating system does not read.

Privacy rule: read only these three facts. Never call `platform.node()`, `socket.gethostname()`, `getpass`, `platform.uname()` as a whole, or `sysctl -a`, and never read serial numbers, model identifiers, or MAC addresses.

`session.json` stores `environment.hardware` as `{"cpu": ..., "logical_cores": ..., "memory_bytes": ...}` with each value through `encode_value`. Because `decode_value` treats any dict as unavailable, `hardware_from_dict` checks for the `"unavailable"` key first. A session without the key loads with `Unavailable(MISSING_HARDWARE_REASON)`, where `MISSING_HARDWARE_REASON = "recorded before hardware was tracked"`. `describe_conditions` in `replay.py` adds `cpu`, `logical cores`, and `memory bytes`, so a rerun on another machine lists them as differences.

### Generator setup time (`sketchloop.experiments.records`, `sketchloop.session`, `cli.common`, `app`)

Creating a backend is session setup, not part of any round. For diffusers it is `DiffusersSketchGenerator.__init__`, whose `select_device` imports torch and probes CUDA and MPS. The diffusers, transformers, and weight loading in `_load_runner` stays the `model_load` stage of the first round.

`SessionRecord` gains one field, and `SketchSession.__init__` gains the matching keyword:

```python
generator_setup_seconds: float | Unavailable  # Time to create the generator this session started with, before round 1

def __init__(self, generator: Generator, store: ExperimentStore, evaluator: Evaluator | None = None, *,
             preprocessor: Preprocessor | None = None, rerun_of: RerunLink | None = None, benchmark: BenchmarkPlan | None = None,
             generator_setup_seconds: float | Unavailable = Unavailable(SETUP_NOT_MEASURED_REASON)) -> None: ...
```

- Reasons: `SETUP_NOT_MEASURED_REASON = "not measured by the caller"` for library callers, `SETUP_REUSED_REASON = "generator reused from an earlier session"`, and `MISSING_SETUP_REASON = "recorded before generator setup was timed"` for older `session.json` files.
- A float value is finite and at least 0. `session.json` stores `"generator_setup_seconds"` after `"environment"` through `encode_value`, and a missing key loads with `MISSING_SETUP_REASON`.
- Measurement: `make_announced_generator` in `cli/common.py` times only the `make_generator` call with `time.perf_counter()`, then prints the backend description and `Generator setup: 3.21 s`. It and `make_configured_generator` return `tuple[Generator, float]`. `make_generator` itself stays untimed, so library callers are unaffected.
- `run`, `benchmark`, and `rerun` (through `make_rerun_generator`) pass the time to their one `SketchSession`. `cli/serve.py` passes it to `AppController(..., generator_setup_seconds=...)`. The controller gives it to its first session and `Unavailable(SETUP_REUSED_REASON)` to every session from `start_new_session`, since those sessions paid no setup and a repeated value would be counted twice.
- `show` prints `Generator setup: 3.21 s` or `unavailable (reason)` in the session header. The benchmark report lists it among the conditions with the note that it is outside every round.

### Webcam frame age (`sketchloop.experiments.records`, `sketchloop.session`, `app`, `ui`)

The page streams webcam frames and Generate uses the newest kept one, so a round can start from a frame up to `STALE_FRAME_INTERVALS` times `capture_interval_seconds` old. The age is a wait before the round, not a stage, so it is not in `timings_seconds` and not in `total_seconds`.

`RoundRecord` gains one field, stored in `round.json` as `input.frame_age_seconds` through `encode_value`:

```python
frame_age_seconds: float | Unavailable  # Seconds from the webcam frame's arrival to the start of the round
```

- Reasons: `NOT_STREAMED_REASON = "not a streamed webcam frame"` for file, camera, and upload sources, `FRAME_AGE_NOT_MEASURED_REASON = "not measured by the caller"` for a webcam round without an age, and `MISSING_FRAME_AGE_REASON = "recorded before frame age was tracked"` for older records.
- Invariant in `RoundRecord.__post_init__`: a float age is finite and at least 0 and only allowed when `source` is `webcam`.
- `SketchPage` adds a `threading.Lock`. `keep_frame` sets `latest_frame` and `latest_frame_time` together under it, and a new `take_fresh_frame(self) -> tuple[np.ndarray, float] | None` reads both under it and returns the frame with `time.monotonic() - latest_frame_time`, or `None` when the frame is stale, using the existing staleness limit. It replaces `get_fresh_frame`, and the Generate button uses it too. `run_generation` calls `take_fresh_frame` for a webcam round, so the age is read when the round is taken, and the frame encode that follows is the `capture` stage.
- `AppController.run_round` gains `frame_age_seconds: float | None = None` and passes it on. A value with a source other than `webcam` raises `SketchLoopError` before anything is read.
- `SketchSession.run_round` gains `frame_age_seconds: float | None = None`. It records the float for a webcam round, `FRAME_AGE_NOT_MEASURED_REASON` for a webcam round without one, and `NOT_STREAMED_REASON` for other sources.
- `show` adds `frame age 0.42 s` to the `Input:` line of a webcam round. `run` and `benchmark` never use the webcam, so they print nothing for it.

### Benchmark plan (`sketchloop.experiments.records`) and statistics (new `sketchloop.experiments.benchmark`)

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkPlan:
    warmup_rounds: int  # Rounds run first and left out of the statistics, such as the one that loads the model
    measured_rounds: int  # Rounds the statistics cover, after the warm-up
    # __post_init__: ints, not bools, warmup_rounds >= 0, measured_rounds >= 1

# SessionRecord gains:
benchmark: BenchmarkPlan | None  # Warm-up and measured round counts for a benchmark session, otherwise None
```

`session.json` adds `"benchmark": {"warmup_rounds": W, "measured_rounds": N}` or `null` after `"rerun_of"`. A missing key loads as `None`, and an invalid plan raises `ExperimentRecordError`. `SketchSession.__init__` gains the keyword `benchmark: BenchmarkPlan | None = None` and writes it into the session record.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class StageStatistics:
    count: int  # Measured rounds that timed this stage
    median_seconds: float  # Median duration
    p95_seconds: float  # 95th percentile, interpolated between measured rounds
    min_seconds: float  # Shortest duration
    max_seconds: float  # Longest duration

@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkSummary:
    plan: BenchmarkPlan  # The plan the session recorded
    warmup_rounds: tuple[int, ...]  # Round numbers left out as warm-up
    measured_rounds: tuple[int, ...]  # Completed round numbers the statistics cover
    failed_rounds: tuple[int, ...]  # Round numbers that failed, left out of the statistics
    stages: Mapping[str, StageStatistics]  # By stage name in RoundStage order, then "total"
    varying_settings: tuple[str, ...]  # Effective control names, other than seeds, that differ across measured rounds

def compute_statistics(durations_seconds: Sequence[float]) -> StageStatistics: ...
def summarize_benchmark(records: Sequence[RoundRecord], plan: BenchmarkPlan) -> BenchmarkSummary: ...
```

- Round numbers up to `warmup_rounds` are warm-up, and later complete rounds are measured. Failed rounds are listed and left out.
- `compute_statistics` uses `statistics.median` and, for two or more values, `statistics.quantiles(durations, n=20, method="inclusive")[18]`. A single value is its own p95, because Python 3.11 and 3.12 reject one data point.
- A stage gets statistics only from the measured rounds that timed it, so `count` can be smaller than the number of measured rounds. `total` comes from float `total_seconds` values.

### Printing (`sketchloop.cli.common`, `cli.show`, `app`)

Two formatters in `cli/common.py`:

```python
def format_timings(timings_seconds: Mapping[str, float], total_seconds: float | Unavailable) -> str: ...
def format_hardware(hardware: Hardware | Unavailable, device: object) -> str: ...
```

`show_round(outcome: RoundOutcome, console: Console, hardware: Hardware | Unavailable) -> None` replaces its "Generation time" line with two lines, printing only the stages present, in `RoundStage` order:

```text
Timings: capture 0.02 s, preprocessing 0.31 s, model load 41.20 s, generation 38.40 s, evaluation 0.00 s, saving 0.06 s, total 80.12 s
Hardware: Apple M2 Pro, 12 logical cores, 32.0 GiB memory, generation on mps
```

When the backend reports no device, the end reads "generation device not reported". When `hardware` is unavailable, the start reads "unavailable (reason)". `run` and `rerun` pass `session.session_record.environment.hardware`.

`show` adds a `Hardware:` line after the Python line, appends the total to each round's `Timings:` line through `format_timings`, and prints "Recorded before T08: generation may include the model load, and capture, saving, and total were not measured." for a round whose total is unavailable. For a session with a `benchmark` plan it prints the benchmark summary after the rounds.

`RoundView` gains `total_seconds: float`, and the page status line says "ready after {total_seconds:.1f} s" instead of using `generation_seconds`, which no longer includes the first model load.

### The `benchmark` subcommand (`sketchloop.cli.benchmark`, `cli.arguments`)

```text
sketchloop benchmark SKETCH --prompt TEXT [--rounds N] [--warmup W] [--candidates C] [--seed S]
                 [--raw | --no-raw] [--rotate DEG] [--backend B] [--mode M] [--runs-dir DIR] [--config PATH]
```

- `Command.BENCHMARK = "benchmark"`. `--rounds` uses `make_int_parser(1)` with default 5, and `--warmup` uses `make_int_parser(0)` with default 1. Neither is a setting. The other flags map to the same settings keys as `run` through `FLAG_SETTINGS[Command.BENCHMARK]`, and `collect_flag_overrides` always sets `capture.source` to `file`, so a settings file naming the camera does not apply. The rotation and raw check in `check_settings` applies to `benchmark` too. A camera benchmark needs a person for every round and is out of scope.
- `run_benchmark(args: argparse.Namespace, console: Console) -> None` creates the folder store, one generator through `make_configured_generator`, and one `SketchSession(..., benchmark=BenchmarkPlan(...))`. It runs `W + N` rounds through `run_round`, each loading the file again through `FileCapture` so capture is measured, with the same prompt, settings, and seed. The seed is `--seed`, else `run.seed` from the settings file, else `BENCHMARK_DEFAULT_SEED = 0`, so repeated benchmarks generate the same images, unlike `run`, whose default is random. It records no selection, because selection is a human event. After each round it prints one line such as `Round 2 of 6 (measured): total 39.81 s`.
- A failing round ends the benchmark through the usual one-line error with exit code 1. The rounds already saved stay, and `show` summarizes them.
- The report prints the benchmark session folder, then:
  - Conditions from the first measured round: backend adapter and model, effective controls without seeds through `format_controls`, candidates, seed, raw and generation sketch sizes, preprocessing step names, the generator setup time, the `Hardware:` line, and the Python and platform line. When `varying_settings` is not empty, a warning names them.
  - Round counts: warm-up round numbers, measured round numbers, and failed round numbers.
  - A table with columns Stage, Median, p95, Min, Max, and N, with one row per stage and a final `total` row, in seconds with two decimals.
  - Each warm-up round's `format_timings` line, shown separately.
  - When fewer than 20 rounds were measured: "p95 is interpolated from N rounds, so treat it as indicative."
- Results live where every session lives, `runs/<session>/` or `--runs-dir`. T09 reuses them with `summarize_benchmark(load_session(folder).rounds, plan)`.

### Out of scope

Optimizing bottlenecks, a latency target or real-time claim, a continuous mode, a derived `benchmark.json` export, a camera benchmark, GPU memory, physical core counts, and a page view of all stages.

## File plan for the python-developer

1. `src/sketchloop/experiments/records.py`: `RoundStage.CAPTURE` and `MODEL_LOAD`, `Hardware`, `BenchmarkPlan`, `Environment.hardware`, `SessionRecord.benchmark` and `generator_setup_seconds`, `RoundRecord.total_seconds` and `frame_age_seconds`, the timing and frame age checks, and the unavailable reasons.
2. `src/sketchloop/experiments/serialization.py`: `hardware_to_dict`, `hardware_from_dict`, and the `total_seconds`, `input.frame_age_seconds`, `benchmark`, and `generator_setup_seconds` keys, with the defaults for older files.
3. `src/sketchloop/experiments/hardware.py` and `provenance.collect_environment`.
4. `src/sketchloop/generation.py`: `GenerationOutput.load_seconds`.
5. `src/sketchloop/backends/diffusers.py`: timed lazy load in `generate` and `describe_torch_runtime`.
6. `src/sketchloop/experiments/stores.py`: abstract `save_files(session_id: str, number: int, files: Mapping[str, bytes]) -> None`, implemented by `FolderStore` and `MemoryStore`, with `save_round` calling it for its own files.
7. `src/sketchloop/capture.py`: `CapturedSketch` and the new `capture()` returns.
8. `src/sketchloop/session.py`: `capture_seconds`, `model_load`, `saving`, `total_seconds`, `RoundOutcome.total_seconds`, `frame_age_seconds`, and the `benchmark` and `generator_setup_seconds` keywords.
9. `src/sketchloop/experiments/benchmark.py`, `experiments/replay.describe_conditions`, and `experiments/__init__.py` exports.
10. `src/sketchloop/cli/common.py` with the timed `make_announced_generator`, `cli/run.py`, `cli/rerun.py`, `cli/serve.py`, and `cli/show.py`.
11. `src/sketchloop/cli/arguments.py`, a new `cli/benchmark.py` with `BENCHMARK_DEFAULT_SEED`, and `cli/__init__.py`.
12. `src/sketchloop/app.py`: `capture_browser_sketch`, `RoundView.total_seconds`, the `generator_setup_seconds` keyword, and `frame_age_seconds` in `run_round`. The `ui-developer` adds the frame lock and `take_fresh_frame` in `ui.py`, passes the age from `run_generation`, and changes the status line.
13. README usage for `benchmark`, and `docs/STATUS.md` through `/handoff`.

## Tests for the qa-engineer

All offline with the fake backend or stubs, with a fake clock where a duration is asserted.

| File | Behaviors and failure paths | Requirements |
| --- | --- | --- |
| `tests/test_capture.py` | `FileCapture` returns a `CapturedSketch` with a non-negative time. With a patched clock and a preview that advances it by 100 s, camera `capture_seconds` stays below 100 s. Cancel and missing-camera errors are unchanged. | R01, R09 |
| `tests/test_session.py` | Given `capture_seconds`, `capture` is recorded, and without it the key is absent. A test generator reporting `load_seconds=5` yields `model_load` 5 and `generation` without it. `saving` is recorded and `total_seconds` is at least the sum of stages after capture plus capture. The saved record equals the outcome timings. A failed round keeps partial timings and a total. Arming `save_files` in the flaky store gives a `saving` failure. The setup time reaches `session.json`, and the default is the not-measured reason. A webcam round records its frame age, a file round records the not-streamed reason, and a webcam round without an age records the not-measured reason. The substituted-components test uses the new capture return. | R06, R08, R09 |
| `tests/experiments/test_loading.py` | Round trip of `total_seconds`, `hardware`, and `benchmark`. Round trip of `generator_setup_seconds` and `input.frame_age_seconds`. Records with those keys deleted load as `Unavailable` with the stated reasons or `None`. A negative or infinite timing, a negative setup time, a frame age on a file round, and a benchmark plan with `warmup_rounds: -1` raise `ExperimentRecordError`. | R06 |
| `tests/experiments/test_hardware.py` (new) | Each reader with a patched `sys.platform`, `subprocess.run`, a temporary cpuinfo file, and `os.sysconf`. A failing command, a timeout, a missing `model name` line, and an unknown platform give `Unavailable` and never raise. With `platform.node` patched to a sentinel, the encoded `session.json` does not contain it. A test on the real machine, on every system, gets each fact as a value or a reason and a real memory size on macOS, Linux, and Windows. | R09, R10 privacy |
| `tests/experiments/test_benchmark.py` (new) | `compute_statistics` for one and five values gives the expected median, inclusive p95, min, and max. `summarize_benchmark` leaves out warm-up and failed rounds, gives `model_load` no row when only warm-up loaded, includes `total`, and lists a varying setting. | R09 |
| `tests/backends/test_diffusers.py` | With an injected runner, `load_seconds` is `None`. With `_load_runner` patched to return a stub and a fake clock, the first `generate` reports the load and the second reports `None`. `describe_torch_runtime` with a stub torch module gives `torch_threads` on CPU and adds `accelerator` on CUDA. | R02, R09 |
| `tests/backends/test_fake.py` | `load_seconds` is `None`. | R02 |
| `tests/cli/test_run.py` | Output has a `Generator setup:` line, a `Timings:` line with capture and total, and a `Hardware:` line. `round.json` has `total_seconds` and the not-streamed frame age, and `session.json` has the setup time. | R09 |
| `tests/cli/test_show.py` (new) | `show` prints the generator setup, the hardware line, totals, and the frame age of a webcam round. A session edited to drop `hardware` and `total_seconds` prints the unavailable hardware and the pre-T08 note. A benchmark session prints its summary. | R06, R09 |
| `tests/cli/test_rerun.py` | A rerun records no `capture`, records its own generator setup time, and a comparison lists a changed `cpu`. | R06 |
| `tests/cli/test_benchmark.py` (new) | `benchmark sketch.png --prompt x --rounds 3 --warmup 1 --candidates 1 --seed 7` in a temporary working folder exits 0, saves four rounds with no selection and a `benchmark` plan, and prints a table with every stage and `total`, the warm-up round, the generator setup, and the p95 note. Without `--seed` and without `run.seed`, every round records seed 0, and a settings seed applies when given. `--rounds 0` and `--warmup -1` exit 2. A settings file with `capture.source = "camera"` still uses the file. `--candidates 99` exits 1 with one line and leaves a failed round. | R09 |
| `tests/test_app.py` | Upload and webcam rounds record `capture` and `RoundView.total_seconds`. A webcam round with an age records it, and an age with the upload source raises `SketchLoopError` and writes nothing. The first session records the setup time, and a session from `start_new_session` records the reused reason. | R07, R09 |
| `tests/test_ui.py` | With `time.monotonic` patched, a frame kept at 10 s and a Generate at 10.4 s passes an age of 0.4 s to the controller. A stale frame is still refused. An upload round passes no age. | R07, R09 |
