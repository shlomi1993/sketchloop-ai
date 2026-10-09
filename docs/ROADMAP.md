# Roadmap and task queue

All application tasks below are pending. Environment setup is complete; task completion is recorded in [STATUS.md](STATUS.md). Month numbers preserve the proposal's tentative relative schedule, not deadlines or estimates guaranteed by the repository.

Development is incremental. From T01b on, every task ends with a runnable increment of the same program that the owner can try, and each task upgrades it rather than starting something new. The program runs as the `sketchloop` command, installed by `pyproject.toml` as a console script starting in T01b.

| Task | Proposal stage | Dependencies | Concrete completion evidence | Requirements | Runnable increment |
| --- | --- | --- | --- | --- | --- |
| T01 | Month 1: literature, requirements, architecture | None | Read foundational research; record findings separately from proposal summaries. Define typed image/guidance/candidate/iteration contracts and fake adapters; document hardware/model/UI questions and decisions. Offline contract tests pass. Done: T01a (contracts and fake adapter) and T01b (the `sketchloop` command, research notes, and [ADR 0006](decisions/0006-first-backend-store-and-ui.md)). | R02, R03, R08 | `sketchloop examples/sketch.png` loads an example sketch image, makes fake variations, saves them under `runs/`, and lets the person pick one in the terminal (T01b) |
| T02a | Month 2: sketch input and preprocessing | T01 | Load a sketch image file; preserve the raw image; demonstrate selected preprocessing operations (crop, perspective, contrast, resize). Invalid-image behavior verified. Done: PNG/JPEG loading, grayscale, crop, contrast, and resize with recorded steps, and `--raw`. T02b added perspective correction. | R01 | The same command shows the original and preprocessed sketch before generating |
| T02b | Month 2: webcam capture | T02a | Capture a real physical sketch from the laptop or USB webcam through the same input interface; camera disconnect behavior verified. Implemented: `--camera` preview with Space to capture and Esc to cancel, `--camera-index`, a one-line error for a missing camera, and perspective correction of the paper. The owner captured a real sketch with the webcam on 2026-10-08. Esc and unplugging during preview are not yet checked by hand. | R01 | `sketchloop --camera` takes the sketch from the webcam instead of a file |
| T03 | Month 2: first generative baseline | T01, T02a (T02b for camera demonstration) | Choose and document one real sketch-conditioned backend; generate/display an output from camera input. Record model/version/settings and timing; distinguish real and fake execution. Done on CPU ([ADR 0009](decisions/0009-diffusers-backend.md)): `--backend diffusers` with fast and quality modes. Image quality, CPU speed, and the Apple MPS run remain to improve. | R02, R03, R09 | The same command makes real AI images from the loaded sketch or webcam capture |
| T04 | Month 3: iterative workflow | T03 | Multiple alternatives, explicit human selection, changed sketch/prompt/parameters, and successive manual iterations. Evaluation interface independent of generation; no-op scorer allowed. | R04, R05 | Several rounds in one session: pick, change the sketch, prompt, or settings, and run again |
| T05 | Month 4: experiment management/replay | T04 | Durable record/artifact storage; reopen earlier run; restore effective conditions; rerun as linked attempt. Verify integrity, lineage, partial failures, and determinism limits. | R06 | Reopen an earlier session from `runs/` and rerun it as a linked attempt |
| T06 | Month 5: user interface | T04, T05 | Camera preview, guidance entry, manual trigger, progress/error states, candidate gallery, selection, next iteration, and history access through library services. Before the UI picks a backend, move `GenerationBackend` from `cli.py` to `generation.py` with a shared `make_generator(backend, mode)` factory, so the CLI and UI create backends the same way. | R07 | A simple window replaces the terminal: preview, prompt, generate, gallery, select, next round |
| T07 | Month 6: modularity and extension | T05, T06 | Replace capture, processing, generator, evaluator, and storage through documented interfaces; tests show orchestration remains unchanged. A second real model is optional. | R08, R10 | Switch the model, image source, or storage through a setting without code changes |
| T08 | Month 7: latency/interactivity | T07 | Stage and end-to-end measurements with documented conditions; optimize measured bottlenecks. Record whether continuous operation is feasible; no mandatory streaming implementation. | R09 | Every run prints how long each stage took, with the hardware it ran on |
| T09 | Month 8: evaluation/documentation/demos | T07, T08 | Full workflow and replay evaluation; usage and extension examples; demonstration, evaluation report, technical write-up and presentation material; assess readiness for future studies. | R09, R10 | A full recorded demo and the evaluation report |
| T10 | Months 9-10: contingency | As needed | Resolve delayed work, unexpected defects, and final adjustments; no new mandatory feature scope. | All | Whatever was delayed, still as a runnable increment |

Model/library research can proceed alongside offline contracts. Introduce enough in-memory metadata early to avoid losing provenance, while deferring the complete durable store to T05. Early display can be minimal; T06 delivers the integrated UI.

## Next bounded task

T04: several rounds in one session (pick, change the sketch, prompt, or settings, and run again), keeping the model loaded between rounds.

## Open decisions

| Question | Needed by | Safe progress meanwhile |
| --- | --- | --- |
| ~~Target OS, camera, compute?~~ Answered: Windows laptop with no GPU, laptop camera or USB webcam. A GPU or Colab may become available later. | Real adapters, T02/T03 | Core contracts, synthetic image fixtures, orchestration tests |
| ~~First model?~~ Answered in [ADR 0006](decisions/0006-first-backend-store-and-ui.md): SD 1.5 with ControlNet scribble and an LCM-LoRA CPU mode, about one minute per round as a starting target. | T03 | Capability abstraction; researcher comparison of CPU-feasible and Colab-hosted options |
| ~~UI toolkit?~~ Answered in [ADR 0006](decisions/0006-first-backend-store-and-ui.md): Gradio as an optional extra. | T06 | Framework-independent application service |
| ~~Store format?~~ Answered in [ADR 0006](decisions/0006-first-backend-store-and-ui.md): filesystem JSON records and artifacts under `runs/`. | T05 | Define record semantics and artifact lifecycle. During T01, have the researcher compare MLflow tracking with a filesystem store before building one |
| Measured latency target? | T08 | Instrument stages; no invented real-time claim |
| Public software license? | Distribution | Develop without asserting an unchosen license |

Optional backlog: live suggestions, multi-model studies, richer constraints, preference learning, prompt optimization, participant studies, presentation experiments, and collaborative drawing. Activate these only after a deliberate scope decision.
