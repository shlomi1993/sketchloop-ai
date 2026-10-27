# ADR 0020: Continuing the next round from a selected candidate

Status: accepted

## Context

During the UI session after T08, the owner asked to use a selected candidate as the base for the next round, as requirement U2 in [ROADMAP.md](../ROADMAP.md). [PROJECT.md](../PROJECT.md) says that a selected generated image does not automatically replace the physical sketch, and that any digital-feedback path must be explicit and recorded. R05 asks for repeated rounds with preserved lineage, and R06 asks that each round's inputs can be reconstructed.

Facts: every round records each candidate's seed and the iteration's parent round ([ADR 0010](0010-sessions-and-evaluator.md), [ADR 0011](0011-session-records-and-subcommands.md)). The diffusers backend runs `StableDiffusionControlNetPipeline` with the sketch as the only image input ([ADR 0009](0009-diffusers-backend.md)), and diffusers also offers a ControlNet image-to-image pipeline for SD 1.5. Its exact arguments and behavior with LCM-LoRA in fast mode are not yet verified.

## Decision

- **Explicit action:** the page offers "Continue from this candidate" on a selected candidate, and nothing continues from a candidate on its own.
- **Image-to-image:** the default way to continue uses ControlNet image-to-image, with the candidate as the starting image, the current sketch as the ControlNet input, and a strength from 0 to 1 that sets how far the result may move from the candidate.
- **Same seed:** a cheaper option keeps the sketch as the only input and reuses the candidate's recorded seed, so a changed prompt gives a similar-looking variation.
- **Records:** the round records the candidate it continued from, by iteration and candidate ID, the method, the strength, and that its starting image was digital, and the starting image is stored with its checksum so the round can be rerun.
- **Backend contract:** a starting image is a new optional input that a backend declares in its capabilities. A backend without it reports the request as unsupported, as with other controls.

Turning a candidate into a new line sketch was rejected as the default, because it drops colors and materials and moves the loop away from the physical drawing. It can be added later as another method.

## Consequences

The generator contract, the request record, and `rerun` gain an optional starting image, within schema version 1, which older rounds load as having none. The diffusers backend loads a second pipeline or switches pipelines, which costs memory and load time that must be measured like T08 stages. A rerun of a continued round needs the stored starting image. The pipeline's arguments, its fit with fast mode, and the memory cost are verified by the `researcher` and `architect` before implementation.
