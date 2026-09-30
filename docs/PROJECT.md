# Project specification

## Purpose and source

Create independent, reusable research infrastructure for design processes integrating hand-drawn sketches and generative AI. The system expands the designer's alternatives while retaining human direction across repeated rounds. It must enable systematic experiments, component replacement, and reconstruction of past run conditions.

This is a sanitized English synthesis of all seven pages of the original Hebrew proposal. Page and section references below provide provenance without requiring the source file. Personal, administrative, and institutional details are deliberately omitted. The proposal's research and technical content is preserved here and in the linked companion documents. Implementation choices are explicitly separated from source requirements.

## Research context

The conceptual foundation is the iterative method discussed in *Rethinking sketching: Integrating hand drawings, digital tools, and AI in modern design*. Sketching supports exploration, selection of directions, and refinement early in design. The referenced method places Style Analysis before Exploration, Definition, Refinement, and Selection within Stylistic Design Engineering (SDE). AI helps produce variations, preserve structural and stylistic features, and move from initial concepts toward more detailed digital representations.

The proposal describes the paper's use of Stable Diffusion 1.5, SDXL, and FLUX through Krita AI Diffusion; latent diffusion; ControlNet conditioning such as edges and depth; image-to-image generation; textual guidance; and CLIP-based similarity filtering followed by human selection. These are background examples, not mandated implementations. The project does not reproduce the paper's tooling or restrict itself to its vehicle-design case study. [References](REFERENCES.md) preserve the full reading list as standard citations.

## Required workflow

1. A person draws freely on a physical surface with a drawing medium of their choice.
2. A camera captures a still image (or a sequence supported by the capture component).
3. Preprocessing prepares the sketch for the selected generative model.
4. The person supplies visual/artistic style guidance and supported controls.
5. The model produces alternatives or completions based on the sketch and guidance.
6. The system presents candidates; optional automated ranking/filtering assists explicit human choice.
7. The person changes the physical sketch, prompt, generation parameters, or conditioning, and manually triggers another iteration.
8. The system retains experimental inputs, configuration, results, timing, and iteration relationships for later inspection and replay.

A selected generated image does not automatically replace the physical sketch. Any future digital-feedback path must be explicit and recorded.

## Requirements and acceptance criteria

Acceptance examples operationalize the proposal; they are repository planning decisions, not claims that the source specified exact test procedures.

| ID | Requirement and source | Observable acceptance |
| --- | --- | --- |
| R01 | Independent capture and preprocessing (pp. 2, §3.1) | A real camera captures a physical sketch. Interchangeable processing can locate the drawing region, correct perspective, resize, enhance contrast, reduce noise, and derive conditioning representations as needed. Preserve original and processed images; document which operations are applicable. |
| R02 | Uniform generative model layer; one model first (pp. 2, §3.2; p. 5, §5) | One real backend generates sketch-conditioned outputs through a documented interface. A second test adapter can be substituted without editing orchestration or UI; additional real models are optional. |
| R03 | Extensible guidance and constraints (p. 3, §3.3) | A user controls desired style through text and available model controls. The interface can represent negative prompts, text-described style/functional constraints, references, or other supported conditioning, including diffusion steps where relevant. Unsupported controls produce explicit feedback. Not every backend must support every example. |
| R04 | Alternatives, optional ranking, and human selection (p. 3, §3.4) | An iteration can produce multiple candidates; generation is independent of scoring/filtering and selection. Show candidates and record the person's choice. A replaceable evaluation hook permits later similarity scoring such as CLIP. |
| R05 | Repeated human-AI interaction (p. 3, §3.5) | Demonstrate successive manually triggered rounds with changed sketch/guidance/parameters and preserved lineage. Stable manual interaction precedes any continuous mode. |
| R06 | Experiment management and replay (p. 3, §3.6) | Store captured images, sketch versions, prompts, model/version, effective parameters, outputs, response times, and iteration links where available. Reopen a run and reconstruct its conditions; explicitly report missing provenance and nondeterminism. |
| R07 | User interface (p. 4, §3.7) | Present camera input, guidance entry, generation trigger, candidate display, selection, and continuation to the next round. Begin with user-initiated operation. |
| R08 | Modular Python library (p. 5, §§5-6) | Documented component interfaces permit replacement/extension without rewriting other components. A library caller can use the workflow independently of UI. |
| R09 | Evaluation (pp. 4-5, §5) | Verify the full capture-to-display flow, runtime measurements, record integrity, and reconstruction of a previous run. Report conditions and limitations. Broader comparisons and user evaluation depend on resources and progress. |
| R10 | Reusable research deliverables (p. 5, §6) | Deliver library, working camera-based system, model/component integration layer, experiment management, UI, technical documentation, usage examples, and demonstrations supporting future research. |

## Scope boundaries

The required outcome is a working, documented research system with one real model and manual iterations, plus extensible components and experiment reconstruction. No specific model, camera, UI framework, hosting service, latency target, or compute budget is mandated. Do not infer these from the bibliography.

The following are optional future extensions (p. 6, §8):

- Continuous, real-time suggestions while drawing.
- Systematic comparison of multiple generative models.
- Additional guidance and constraints, especially richer stylistic constraints.
- Adaptation to preferences inferred from a user's selections.
- Automatic prompt improvement.
- User studies across groups such as age and design-experience levels.
- Alternative ways to present suggestions and integrate them into creative work.
- Multiple people collaborating in one design process.

The core should support later investigations of sketch detail, model differences, user characteristics, and response latency (p. 4, §4). This does not require collecting participant demographics now.

## Delivery and evaluation method

Study the literature and relevant visual-generation technology; define modular interfaces; implement one end-to-end baseline; add iterative choice, persistence/replay, and UI; demonstrate replaceability; measure behavior and latency; document and demonstrate the system. The original tentative plan spans two semesters, eight working months plus months 9-10 as contingency. [ROADMAP.md](ROADMAP.md) preserves each stage without inventing calendar deadlines.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the proposed record contract and [ARCHITECTURE.md](ARCHITECTURE.md) for implementation guidance. The current implementation state is maintained only in [STATUS.md](STATUS.md).
