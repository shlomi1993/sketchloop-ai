# Foundational paper: Rethinking Sketching (SDE + generative AI)

Donnici, Galiè, and Frizziero, *Rethinking Sketching: Integrating Hand Drawings, Digital Tools, and AI in Modern Design*, Designs 9(5), 119 (2025), [doi:10.3390/designs9050119](https://doi.org/10.3390/designs9050119). Open access under CC BY 4.0. Local copy: [docs/articles/Rethinking Sketching.pdf](../articles/Rethinking%20Sketching.pdf) (20 pages). Read in full for this note.

**Use this note instead of the PDF.** Open the PDF only for a detail missing here, and cite it by page.

## Question and decision informed

What the paper's method implies for SketchLoop's pipeline, recorded parameters, evaluation, and thesis background (T01, T03, T04, T09).

## Relevant findings (verified in the source)

The paper's generative loop (pp. 5-6, Algorithm 1) maps onto the planned components:

| Paper step | SketchLoop component | Details from the paper |
| --- | --- | --- |
| Input preprocessing | `preprocessing` | Deskew, normalize to 1024×1024, contrast enhancement, optional binarization. Edge or normal-map ControlNet guidance when strokes are dense |
| Conditioned generation | `generation` | Image-to-image with classifier-free guidance. 25-50 steps, guidance scale 7-9, fixed seed. SD 1.5, SDXL, and Flux via Krita AI Diffusion. ControlNet depth, normals, style, sketch, segmentation |
| Guidance | `domain` guidance records | Prompts combine style cues with functional constraints, for example "maintain wheelbase and roofline". Negative prompts are used |
| Variants | `generation` | Batches of 8-12 variants per cycle |
| Selection | `evaluation` plus human selection | CLIP similarity filtering followed by manual review |
| Iterative loop | `orchestration` | The person adjusts prompt, ControlNet settings, or sketch, and repeats until an acceptance threshold (expert rating ≥4.5/5) |

Other points that matter here:
- **Input quality drives results** (pp. 12, 16-17). The model does best with a moderate number of lines, high contrast, and clear separation between elements. Dense overlapping strokes confuse it. This supports preprocessing and user guidance about sketch style.
- **Nondeterminism** (p. 17). Outputs vary "even with fixed seeds", and dataset bias shifts style. This supports the project's rule against claiming exact replay.
- **Constraints** (pp. 8-9, 16). The paper separates technical constraints (feasibility) from stylistic ones (identity, continuity or disruption). Both are expressed through prompts, ControlNet depth, and filtering. This is a useful structure for R03 guidance records.
- **Sketch stages** (pp. 7, 10-15). Exploration (D1: many ideas), Definition (D2: top 2-3 ideas), Refinement (D3: details), Selection (D4: final). AI helps most in D1-D2. Iteration records could optionally be tagged with a stage.
- **Evaluation used** (p. 9, Table 1). One automotive case study with a comparison against a manual workflow: 50 h versus 36 h, 9 versus 15 variants in D1, and a coherence rating of 4.3 versus 4.6 from a panel of three designers.
- **Future work** (pp. 17-18). Novice usability, collaboration, custom-trained models, real-time or VR/AR, and automated prompt optimization. These match the proposal's extensions.

## Out of scope for this project

Automotive specifics and drawing techniques ("holding boxes", engine layouts), the IDeS/QFD management framework, style analysis from brand history, CAD vectorization and 3D modeling, rendering and AR, and design-education discussion. Mention them in the thesis background only.

## Implications (inference)

- Record at least: model and revision, ControlNet type and weight, conditioning image, steps, guidance scale, seed, strength, resolution, batch size, prompt and negative prompt, and the CLIP model used for filtering. This fits [EXPERIMENTS.md](../EXPERIMENTS.md).
- With no local GPU, SD 1.5 plus a scribble or lineart ControlNet at 512 px is the most plausible CPU baseline. SDXL and Flux likely need Colab. This must be measured, not assumed (T03).
- Default to a batch of about 4 on CPU and 8 on GPU, and let the user change it.
- The paper's evaluation is a single case study with three raters. The thesis can position SketchLoop as infrastructure that makes such comparisons systematic and repeatable (R06, R09).

## Open questions

- Whether to support a stage tag (D1-D4) in iteration records.
- Which ControlNet variant works best for camera-captured pencil sketches. Researcher follow-up during T03.
