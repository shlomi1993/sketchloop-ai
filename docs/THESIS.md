# Thesis outline

Working outline for the final written report, mapped to the repository evidence each chapter needs. The report itself may be written outside this repository. Update the evidence column as work lands.

| Chapter | Content | Evidence in the repository |
| --- | --- | --- |
| 1. Introduction | Goal, motivation, and contributions of a modular research infrastructure | [PROJECT.md](PROJECT.md) |
| 2. Background | Sketching in design and the SDE method, latent diffusion, ControlNet conditioning, CLIP similarity, related tools | [REFERENCES.md](REFERENCES.md), [research notes](research/README.md) |
| 3. Requirements | R01-R10, scope, and extensions | [PROJECT.md](PROJECT.md) |
| 4. System design | Component boundaries, contracts, experiment records, and key decisions | [ARCHITECTURE.md](ARCHITECTURE.md), [EXPERIMENTS.md](EXPERIMENTS.md), [decisions](decisions/) |
| 5. Implementation | Capture and preprocessing, generation adapter, iteration loop, experiment store, UI | `src/sketchloop/`, roadmap T02-T07 |
| 6. Evaluation | End-to-end flow, stage and total latency with stated hardware, record integrity, replay of earlier runs, the guided-sketching user study, limitations | Tests, benchmark results, the T08 report, T09 to T11 in [ROADMAP.md](ROADMAP.md) and [ADR 0021](decisions/0021-guided-sketching-and-user-study.md) |
| 7. Discussion | What worked, reproducibility limits, lessons for human-AI iterative design | Research notes, STATUS history |
| 8. Future work | Real-time suggestions, model comparisons, preference adaptation, comparing professional artists with inexperienced participants, collaboration | [PROJECT.md](PROJECT.md) scope boundaries, [supervisor meeting 1](meetings/2026-11-supervisor-meeting-1.md) |
| Appendix | Installation, usage and extension examples, experiment record schema | [DEVELOPMENT.md](DEVELOPMENT.md), usage examples |

Keep figures and measurements reproducible: every reported number should point to a saved run or a script that regenerates it.
