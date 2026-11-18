# ADR 0021: Guided sketching and a user study as the primary objective

Status: accepted

## Context

At the first supervisor meeting ([notes](../meetings/2026-11-supervisor-meeting-1.md)), the supervisor redirected the project from AI-generated designs to human-AI interaction. The system should suggest how to continue the sketch, infer whether a suggestion was accepted from whether the person follows it, and support a user study: four design tasks, five participants, each task with and without the system, 40 drawings for expert evaluation, with a submission target in early 2027.

Facts: the system captures a live webcam sketch, preprocesses it, and generates complete images with SD 1.5 and ControlNet scribble in about 5 s per fast round on the Apple M2 Pro ([baseline](../results/t08-latency-baseline.md)). [PROJECT.md](../PROJECT.md) listed participant studies and continuous suggestions as optional extensions, and its required workflow already allows "alternatives or completions" (step 5). [AGENTS.md](../../AGENTS.md) required human selection to be explicit. Continuous suggestions while drawing were measured as not feasible ([ADR 0018](0018-float16-on-the-apple-gpu.md)), and an automatic trigger on drawing change was tried and reverted because camera noise made it unreliable ([ADR 0019](0019-offline-prompt-suggestions.md)).

Assumptions to verify in T09: a local model can propose useful next strokes from a partial pencil sketch within a few seconds, and new strokes in later captures can be matched against a suggestion reliably enough to infer acceptance.

## Decision

- **Primary objective:** the implementation plan's main goal is a working system that supports the user study and collects its data. Tasks that the study needs come first in [ROADMAP.md](../ROADMAP.md), and other requested work follows.
- **Requirements:** the meeting adds S01 to S05 in [PROJECT.md](../PROJECT.md), meaning guided sketching, implicit acceptance, study conditions, process and outcome data, and local study data. They refine R03 to R05 and R09 and do not replace R01 to R10.
- **Guided sketching:** on request, the system shows visual suggestions for continuing the drawing toward the stated goal, overlaid on the live sketch, and never replaces the physical sketch. Suggestions are requested by a button, because continuous and automatic triggering were not feasible. The suggestion method and its model are chosen in T09 by the `researcher` before any code is written.
- **Implicit acceptance:** a suggestion is recorded as followed, not followed, or undetermined, inferred by comparing the strokes drawn after it with the suggested strokes. The record keeps the method, score, threshold, and time window, so the inference can be recomputed. Inferred acceptance is a research signal recorded separately from explicit selections, and the invariant in AGENTS.md now says so.
- **Study conditions:** the condition without AI runs the same capture and records with suggestions turned off, so both conditions produce comparable data.
- **Study data:** participants are identified only by pseudonymous codes. Captures, records, and exports stay in ignored local storage, as [PRIVACY.md](../PRIVACY.md) requires for participant data.
- **Existing features:** complete-image generation, the candidate gallery, and explicit selection stay. U2, prompt suggestion records, and the new-model modularity test remain required but come after the study's critical path.

Keeping complete-image generation as the main interaction was rejected because the supervisor asked to study guidance rather than generation. Explicit accept and reject buttons were rejected as the main signal because the supervisor asked for acceptance to be inferred from the drawing, though the records can still hold explicit choices.

## Consequences

The roadmap gains three tasks: T09 for guided sketching, T10 for the study mode and export, and T11 for running the study. The T07 and T08 refinements come before them, and the project documentation follows as T12. The records gain suggestion events, a drawing timeline, inferred acceptance, and study metadata within schema version 1, designed by the `architect` before implementation. A new suggestion component sits beside the generator. The study mode is configured rather than fixed, so any tasks, order of conditions, and prompts can be used and other studies can reuse it. Whether suggestions are useful and whether acceptance can be inferred reliably stay unverified until T09 and the pilot in T10. The submission target leaves little time, so work off the critical path waits until the study data is collected.
