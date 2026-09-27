# Roadmap and task queue

All application tasks below are pending. Environment setup is complete; task completion is recorded in [STATUS.md](STATUS.md). Month numbers preserve the proposal's tentative relative schedule, not deadlines or estimates guaranteed by the repository.

| Task | Proposal stage | Dependencies | Concrete completion evidence | Requirements |
| --- | --- | --- | --- | --- |
| T01 | Month 1: literature, requirements, architecture | None | Read foundational research; record findings separately from proposal summaries. Define typed image/guidance/candidate/iteration contracts and fake adapters; document hardware/model/UI questions and decisions. Offline contract tests pass. | R02, R03, R08 |
| T02 | Month 2: camera and preprocessing | T01 | Capture a real physical sketch; preserve raw image; demonstrate selected preprocessing operations and replaceable file-input test source. Camera disconnect and invalid-image behavior verified. | R01 |
| T03 | Month 2: first generative baseline | T01, T02 for camera demonstration | Choose and document one real sketch-conditioned backend; generate/display an output from camera input. Record model/version/settings and timing; distinguish real and fake execution. | R02, R03, R09 |
| T04 | Month 3: iterative workflow | T03 | Multiple alternatives, explicit human selection, changed sketch/prompt/parameters, and successive manual iterations. Evaluation interface independent of generation; no-op scorer allowed. | R04, R05 |
| T05 | Month 4: experiment management/replay | T04 | Durable record/artifact storage; reopen earlier run; restore effective conditions; rerun as linked attempt. Verify integrity, lineage, partial failures, and determinism limits. | R06 |
| T06 | Month 5: user interface | T04, T05 | Camera preview, guidance entry, manual trigger, progress/error states, candidate gallery, selection, next iteration, and history access through library services. | R07 |
| T07 | Month 6: modularity and extension | T05, T06 | Replace capture, processing, generator, evaluator, and storage through documented interfaces; tests show orchestration remains unchanged. A second real model is optional. | R08, R10 |
| T08 | Month 7: latency/interactivity | T07 | Stage and end-to-end measurements with documented conditions; optimize measured bottlenecks. Record whether continuous operation is feasible; no mandatory streaming implementation. | R09 |
| T09 | Month 8: evaluation/documentation/demos | T07, T08 | Full workflow and replay evaluation; usage and extension examples; demonstration, evaluation report, technical write-up and presentation material; assess readiness for future studies. | R09, R10 |
| T10 | Months 9-10: contingency | As needed | Resolve delayed work, unexpected defects, and final adjustments; no new mandatory feature scope. | All |

Model/library research can proceed alongside offline contracts. Introduce enough in-memory metadata early to avoid losing provenance, while deferring the complete durable store to T05. Early display can be minimal; T06 delivers the integrated UI.

## Next bounded task

T01a: define image/artifact, guidance, generation result, and iteration contracts with explicit capability validation. Write synthetic contract tests that reject unsupported controls and invalid candidate selection. Implement only a deterministic test adapter, label it clearly, and preserve the real-model requirement in T03. Do not select hardware-dependent packages until their suitability is known.

## Open decisions

| Question | Needed by | Safe progress meanwhile |
| --- | --- | --- |
| Target OS, camera, compute/accelerator and memory? | Real adapters, T02/T03 | Core contracts, synthetic image fixtures, orchestration tests |
| Local inference or remote model? Model identity, license, cost and reproducibility constraints? | T03 | Capability abstraction; backend comparison criteria |
| UI toolkit and intended local deployment? | T06 | Framework-independent application service |
| Durable store format and optional experiment tracker? | T05 | Define record semantics and artifact lifecycle |
| Measured latency target? | T08 | Instrument stages; no invented real-time claim |
| Public software license? | Distribution | Develop without asserting an unchosen license |

Optional backlog: live suggestions, multi-model studies, richer constraints, preference learning, prompt optimization, participant studies, presentation experiments, and collaborative drawing. Activate these only after a deliberate scope decision.
