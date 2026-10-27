# T08 latency baseline

First stage timings for SketchLoop (roadmap task T08, requirement R09), measured with `sketchloop benchmark` as specified in [the T08 design](../design/t08-latency.md) and [ADR 0017](../decisions/0017-stage-timings-hardware-and-benchmark.md). This is a baseline, not a latency target, which stays open in [ROADMAP.md](../ROADMAP.md).

## Conditions

| Condition | Value |
| --- | --- |
| Hardware | Apple M2 Pro, 12 logical cores, 16 GiB unified memory, generation on the Apple GPU (MPS) |
| Runtime | Python 3.14.0, macOS (Darwin 27.0.0, arm64), float32, 8 torch threads |
| Model | Stable Diffusion 1.5 with ControlNet v1.1 scribble, plus LCM-LoRA in fast mode |
| Sketch | `examples/sketch-photo.jpg`, 1200 x 900 raw, preprocessed to 357 x 512 and generated at 360 x 512 |
| Request | 2 candidates per round, seed 0, prompt "a sculptural oak lounge chair, soft studio light" |
| Plan | 1 warm-up round and 5 measured rounds per mode, in one session with one loaded backend |
| Machine load | Ordinary desktop apps open, with no effort to free memory |

## Results

Times are in seconds. Median and p95 are over the measured rounds, and p95 from 5 rounds is interpolated, so treat it as indicative.

| Mode | Steps | Rounds used | Round total, median | Round total, min to max | Generation share | Model load (first round) | Generator setup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| quality | 20 | 5 of 5 | 26.17 | 25.85 to 26.47 | 99.8% | 10.23 | 1.04 |
| fast | 4 | 3 of 5 | 6.92 | 6.13 to 7.00 | 99.4% | 17.02 | 1.17 |

Capture, preprocessing, evaluation with the no-op evaluator, and saving took about 0.05 s per round together in both modes. Per candidate, generation took about 13 s in quality mode and about 3.5 s in fast mode. The fast mode's model load is longer because it also loads and fuses the LCM-LoRA weights.

## Memory pressure

In fast mode, rounds 5 and 6, the fourth and fifth measured rounds, took 141 s and 217 s instead of about 7 s. A rerun on an otherwise idle machine took 320 s for its warm-up round and was stopped. During that run the system used 9.7 GB of its 11 GB swap, and the benchmark process kept about 9 MB of its model in memory, so the slow rounds measured paging, not generation. The float32 model on the Apple GPU needs several GB of the 16 GB shared between CPU and GPU, so with ordinary apps open it can be paged out partway through a session. The two fast rounds affected are left out of the fast row above.

## Findings

- **Bottleneck:** generation is over 99% of every round, so the other stages need no optimization now.
- **Memory:** on this machine the next limit is memory, not compute. Running the model in float16 on the Apple GPU halves its memory and is the next T08 slice, measured with heavy apps closed and compared for image similarity.
- **Continuous operation:** continuous suggestions while drawing are not feasible at these speeds, since a round takes about 7 s even in fast mode. Generating when the person pauses, or on a manual trigger as now, fits the measured times.

## Not measured yet

The laptop CPU from STATUS, a GPU with CUDA, more than 5 measured rounds for a meaningful p95, and fast mode on an unloaded machine. The saved sessions are under the ignored `runs/` folder, so `sketchloop show` and the benchmark summary can be recomputed from them.
