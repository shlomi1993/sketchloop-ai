# ADR 0017: Stage timings, hardware facts, and a benchmark command

Status: accepted

## Context

T08 (R09) asks for stage and end-to-end measurements with documented conditions, and its runnable increment is that every run prints how long each stage took, with the hardware it ran on. [EXPERIMENTS.md](../EXPERIMENTS.md) lists what to measure and report: capture, processing, generation, evaluation, persistence, and total time, plus image size, batch size, configuration, hardware and runtime, warm-up treatment, number of runs, and a distribution such as median and p95.

Facts from the code after T07: `SketchSession.run_round` times preprocessing, generation, and evaluation. Generation in the first diffusers round includes the lazy model load. Capture happens before `run_round` in the CLI and in `AppController`, and is not timed. `RoundStage.SAVING` exists only as a failure stage. `Environment` records Python, platform, and packages but no CPU, cores, or memory. Records are at schema version 1 ([ADR 0011](0011-session-records-and-subcommands.md)), and stores write through `ExperimentStore` ([ADR 0014](0014-replaceable-components-and-settings.md), [ADR 0015](0015-abstract-base-classes-for-components.md)). The measured latency target is an open owner decision, and nothing here sets one.

The owner approved both interface changes below, set the benchmark defaults to 5 measured rounds and 1 warm-up round with seed 0 when none is given, and asked for the generator setup time per session and the webcam frame age per round.

## Decision

Adopt [the T08 design](../design/t08-latency.md):

- **Stages:** `RoundStage` gains `capture` and `model_load`, and `saving` becomes a timed stage. `timings_seconds` stays keyed by stage name and holds only stages that ran and were timed.
- **Capture:** `CaptureSource.capture()` returns a `CapturedSketch` with the image, bytes, and `capture_seconds`, and each source times only its own work, so a camera's clock starts when Space is pressed. `run_round` gains `capture_seconds`, and `AppController` times the browser frame encode or upload load itself.
- **Model load:** `GenerationOutput` gains `load_seconds`, reported by the backend for a call that loaded models. The session records it as `model_load` and subtracts it from `generation`. The diffusers backend also records `torch_threads`, and on CUDA the `accelerator` name, in its effective controls.
- **Saving and total:** `ExperimentStore` gains `save_files`. The session times the file writes as `saving`, then writes `round.json` once with `RoundRecord.total_seconds`, which is capture time plus the round's wall time until the files are saved.
- **Hardware:** `Environment` gains `hardware`, a `Hardware` record with CPU model name, logical core count, and total memory, read with the standard library on macOS (`sysctl`), Linux (`/proc/cpuinfo`, `os.sysconf`), and Windows (`winreg`, `ctypes`). An unreadable fact is `Unavailable` with a fixed reason. No machine name, user name, serial number, model identifier, or network address is read.
- **Generator setup:** `SessionRecord` gains `generator_setup_seconds`, the time `make_generator` took before round 1, including the torch import and device selection in the diffusers backend's constructor. `make_announced_generator` times it and prints it, and `run`, `rerun`, `benchmark`, and `ui` pass it to the session. A session started from the page's new-session button records it as unavailable because the generator was reused.
- **Webcam frame age:** `RoundRecord` gains `frame_age_seconds`, stored as `input.frame_age_seconds`, from the frame's arrival in `SketchPage.keep_frame` to the round taking it. It passes through `AppController.run_round` to `SketchSession.run_round`. Other sources record it as unavailable, and it is neither a stage nor part of the total.
- **Benchmark:** a `sketchloop benchmark SKETCH --prompt TEXT --rounds N --warmup W` subcommand, with defaults 5 and 1, runs `W + N` rounds through one `SketchSession` with one loaded backend from a file, uses seed 0 unless a seed is given, records no selection, and prints the conditions, warm-up rounds separately, and per-stage and total median, p95, min, and max over the measured rounds, using the `statistics` module. `SessionRecord` gains a `benchmark` plan, so the statistics are recomputed from saved rounds by `summarize_benchmark`, and `show` prints them too.
- **Compatibility:** schema version stays 1. Older `round.json` files load `total_seconds` as unavailable ("recorded before total round time was measured"), older `session.json` files load `hardware` as unavailable ("recorded before hardware was tracked"), `generator_setup_seconds` as unavailable ("recorded before generator setup was timed"), and `benchmark` as none, older rounds load `input.frame_age_seconds` as unavailable ("recorded before frame age was tracked"), and `show` notes that their generation time may include the model load.

Alternatives considered:

- Timing the whole `capture()` call in callers needs no interface change but would count the person's camera preview as latency, and leaving the CLI camera untimed would break the runnable increment.
- Rewriting `round.json` after timing `save_round` measures the first record write too, but adds a second write and fsync per round and a failure point after the files are saved.
- A `Generator.load()` method called every round cannot tell a real load from a no-op, and loading at backend creation would move the load outside any round record.
- A derived `benchmark.json` summary would duplicate the rounds and need a new store method, so it is left as a later export.
- Recording the frame age or the generator setup as entries of `timings_seconds` would make sums over stages count waits and setup as round latency.
- Per-stage `Unavailable` values inside `timings_seconds` would change its type for every consumer, so absence means a stage did not run or was not timed.

## Consequences

Two component interfaces change: implementations and test substitutes of `CaptureSource` return `CapturedSketch`, and those of `ExperimentStore` implement `save_files`. Records stay readable in both directions within version 1, but older records cannot be given a model load, capture, saving, or total time after the fact. The final `round.json` write is outside every stage, and backend creation is recorded once per session rather than in a round. `make_announced_generator` and `make_configured_generator` now return the setup time with the generator. The page guards its newest frame with a lock so the frame and its arrival time are read together. Benchmark defaults to seed 0, so its images differ from a `run` with a random seed. Reruns of rounds recorded before T08 report the new hardware and runtime facts as differences. p95 from fewer than 20 measured rounds is interpolated and printed with that warning. Optimizing bottlenecks, a latency target, and the continuous-operation assessment follow in later T08 slices using benchmark results on the owner's hardware. Validation is offline with the fake backend and stubs, plus a benchmark run with the diffusers backend on the owner's Apple M2 Pro, which the owner chose over the laptop CPU, recorded with its conditions before the slice is called done.
