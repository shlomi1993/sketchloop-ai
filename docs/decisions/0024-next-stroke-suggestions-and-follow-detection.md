# ADR 0024: Next-stroke suggestions and follow detection

Status: accepted

## Context

[ADR 0021](0021-guided-sketching-and-user-study.md) made guided sketching the primary objective: on request, the system suggests how to continue the sketch (S01), and whether the person followed a suggestion is inferred from what they draw next (S02). The [research note](../research/next-stroke-suggestions.md) compared four kinds of suggestion method and several follow-detection approaches, and prototyped the best of each on the owner's Apple M2 Pro.

Facts from the note: generating a line drawing by image-to-image from the person's own sketch, with the existing SD 1.5, scribble ControlNet, and LCM-LoRA weights, gave a plausible new part in about 9 of 12 test cases. Examples include chair legs, car wheels or a cabin from a rectangle, and wings or a tail from a fuselage. It needs no new model or dependency and took about 3 s per suggestion at 512x384 in float16, after a one-time load of 18 to 25 s. Generating from noise or from a photo prompt with edge extraction was usable in about 4 of 12 cases. SketchAgent needs a cloud API, pretrained SketchRNN models lack cars and airplanes, and newer sketch completion models have no confirmed weights. Florence-2 named parts correctly for the car and chair but not for the airplane. For follow detection, aligning each later capture to the one at suggestion time and scoring the new strokes per suggested part separated followed from drawn elsewhere with an AUC of 0.98 in 240 simulated trials, while a different shape drawn in the same place was often counted as followed. Assumptions: the 12 cases came from four synthetic or edited sketches, so usefulness on participants' sketches is unverified.

## Decision

- **Suggestions:** on a button press, run `StableDiffusionControlNetImg2ImgPipeline` with the person's sketch in the paper frame as the starting image and the scribble ControlNet as guidance, at strength about 0.85 with LCM-LoRA. Keep only dark lines away from the existing strokes, and group them into up to four colored parts shown as an overlay on the live sketch. The canvas is the paper frame, not the cropped `sketch.png`, so captures stay comparable and new parts have room.
- **One suggestion per press:** each press shows one suggestion with up to four parts, and each part is scored separately.
- **Followed means the same part in the same place:** a part counts as followed when the person draws in its place, judged by coverage and precision of the new strokes against the dilated part. Matching the exact shape is not required, which is why a different shape in the same place may count as followed.
- **Time window:** a suggestion is checked from the moment it is shown until the next suggestion or the end of the task. Drawing nothing in that window gives "undetermined", not "not followed".
- **Records:** every capture's raw alignment and per-part scores, at several tolerances, are recorded so the threshold can be tuned later against hand labels from the pilot, which the owner provides. The provisional rule is coverage at 16 px of at least 0.35 and precision at 16 px of at least 0.4.
- **Part names:** Florence-2 part names are recorded as metadata only, and tasks have no fixed part lists.
- **Hardware:** the study runs on the owner's M2 Pro. The system keeps working on Windows and Linux so others can clone it and run experiments, but suggestion speed on a CPU-only machine is not a study requirement.
- **Component:** suggestions sit behind a replaceable component like the generator, so another method can replace this one through the settings.

## Consequences

T09's build tasks follow this design, and the `architect` designs the suggestion component and the records for suggestions, captures, and follow scores within schema version 1 before implementation. The suggestion pipeline is a second SD 1.5 pipeline beside the generator, which costs memory and load time that must be measured. Part labels are often wrong and some suggestions land near the frame edges, so the overlay should not depend on labels. On a CPU, a suggestion is expected to take 35 to 60 s, which is acceptable only outside the study. The threshold, the usefulness of suggestions on real participants' sketches, and the behavior with real hands and lighting stay unverified until the pilot in T10.
