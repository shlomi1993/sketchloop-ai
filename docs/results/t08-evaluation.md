# T08 system evaluation

On 2026-11-18 the full flow, record integrity, and replay were checked with both real backends on the owner's Apple M2 Pro (12 logical cores, 16 GiB, MPS, float16), with pinned library versions and other apps open (R05, R06, R09). Every run used `examples/sketch-photo.jpg`, fast mode, 2 candidates per round, seed 0, the `clip-b32` evaluator, and `[models] dir` pointing at the downloaded weights. The latency distribution is in [the T08 baseline](t08-latency-baseline.md). This note checks that the flow works end to end and that its records hold up.

## Full flow

| Path | Backend | Rounds | Round 1 total | Round 2 total |
| --- | --- | --- | --- | --- |
| Terminal, `sketchloop run` from a file | `diffusers` | 2 | 26.7 s, of which 20.1 s model load | 6.7 s |
| Terminal, `sketchloop run` from a file | `t2i-adapter` | 2 | 21.2 s, of which 16.3 s model load | 4.3 s |
| Page controller, webcam-like frame | `t2i-adapter` | 1 | 18.7 s, of which 13.6 s model load | none |

- **Terminal:** each session ran two rounds, with the prompt changed from "modern chair" to "modern armchair" in round 2 and an explicit selection in each round. Round 2 links to round 1 through its `parent_id`, and model load appears only in round 1, so later rounds are generation, scoring, and saving.
- **Page path:** the run drove `AppController`, the service the page calls, with a frame in memory as the browser webcam delivers it. "Suggest a prompt" returned "a chair, product design render, studio lighting, white background" from Florence-2, which then generated the round, and the selection was recorded. The browser itself was not part of this run, and the owner last verified the page by hand after the UI session.
- **Scores:** CLIP gave scores between 0.245 and 0.275, recorded with the model and revision that produced them. Evaluation took 0.05 to 0.5 s per round once loaded.

## Record integrity

- **Saved sessions:** `sketchloop show` reported no missing or changed files for any of the three sessions.
- **Tampering:** in a copy of the `diffusers` session, one byte was appended to a round 1 candidate and round 2's `sketch.png` was deleted. `show` reported "round-1/candidates/candidate-1.png does not match its recorded checksum." and "round-2/sketch.png is missing.", and named nothing else.

## Replay

| Session | Rerun of round 1 | Byte-identical candidates | Reported differences |
| --- | --- | --- | --- |
| Terminal, `diffusers` | New linked session | 2 of 2 | None in backend, settings, or environment |
| Terminal, `t2i-adapter` | New linked session | 2 of 2 | None |
| Page, `t2i-adapter` | New linked session | 2 of 2 | None |

Each rerun recreated the recorded backend, mode, precision, and evaluator from the round record, and still stated that equal seeds do not guarantee identical images in general.

## Limits

- One machine, one example sketch, and fast mode only. Quality mode, the CPU path on the Windows laptop, and CUDA were not part of this run.
- Replay was checked on the same machine and library versions. A rerun on other hardware or versions is expected to report differences and may not be byte-identical.
- The browser page, the real webcam, and the terminal camera capture need a person and were not exercised here.
