# ADR 0026: Study mode, the condition record, prompt suggestion records, and the blinded export

Status: accepted

## Context

T10 in [ROADMAP.md](../ROADMAP.md) builds the study mode for the user study that [ADR 0021](0021-guided-sketching-and-user-study.md) made the primary objective, serving S03, S04, and S05 in [PROJECT.md](../PROJECT.md), with records under R06 and the page under R07. The owner also moved U1's prompt suggestion records into T10, which [ADR 0019](0019-offline-prompt-suggestions.md) required but left unbuilt.

Facts:

- **Drawings:** [ADR 0025](0025-drawings-as-the-unit-of-guided-sketching.md) records each drawing in `drawing-<n>/drawing.json` inside a session folder named by the experiment name, the only study label the system stores. It reserved `suggestions_enabled` and a study object for T10, and `SketchSession.start_drawing(None, ...)` already gives a drawing with suggestions off.
- **Final drawing:** each finished drawing keeps its final capture, a PNG frame plus a grayscale paper canvas at 512x360 by default.
- **Prompt suggestions:** "Suggest a prompt" returns a string and records nothing. The prompt suggester has no identity, and it is chosen with the backend.
- **Page:** there is one layout, which shows generation controls, model descriptions, and the experiment name to whoever sits at it.

The owner decided, as final:

1. **Prompt suggestions:** "Suggest a prompt" stays in the AI condition and every use is recorded, including whether each used prompt was typed, suggested, or an edited suggestion. It is hidden in the no-AI condition, so the conditions differ only in AI help.
2. **Study page:** one page, `sketchloop ui --study` or a settings key, shows participants only the sketch, the prompt, the AI buttons, and the status, with experimenter controls in a collapsed panel.
3. **No-AI condition:** the Suggest next strokes button is removed, and the experimenter chooses the condition per experiment, recorded on the drawing.
4. **Export:** a blinded export of one final drawing per experiment under random codes, with a local key, missing drawings reported, a configurable folder, and a whole runs folder as input.
5. **Free study:** any tasks, order, and prompts, and the system never knows the task.

Assumptions to verify in the pilot: a 512x360 canvas is readable enough for expert evaluation, and the webcam keeps streaming while the experimenter panel is closed.

## Decision

[The T10 design](../design/t10-study-mode.md) holds the full contracts. In short:

- **Condition record:** `drawing.json` gains `"study": {"condition": "ai" | "no_ai"}`, null outside study mode. A study drawing's condition must agree with `suggestions_enabled`, which now means all AI help, both next-stroke and prompt suggestions. No task, participant, or order field is added.
- **Prompt suggestion records:** each press of "Suggest a prompt" is a `PromptSuggestionRecord` with an ID, times, the suggester's `BackendIdentity`, the text, its list position, whether the model ran, its time, the latest capture, and any failure. It is stored in the open drawing's `prompt_suggestions`, or outside a drawing in the next round's `prompt_suggestions`. `PromptSuggester` gains an abstract `identity` property.
- **Prompt origins:** the page keeps the ID of the suggestion it wrote into its prompt box and sends it with each stroke suggestion, generation, and finish. A pure `classify_prompt_origin` gives `typed`, `suggested`, or `edited` from the used text and that suggestion's text, recorded as `prompt_origin` on each stroke suggestion and round. A drawing checks that its origins name its own earlier suggestions and agree with the stored texts. An unknown ID is recorded as unavailable and never blocks the participant.
- **Final prompt:** each drawing records the prompt box text at finish and its origin, so no-AI drawings also hold the participant's prompt.
- **Prompt suggester setting:** a `[prompt_suggester]` settings table with `name` set to `templates`, the default, or `florence` chooses the prompt suggester independently of the image backend, which amends [ADR 0019](0019-offline-prompt-suggestions.md).
- **Compatibility:** all additions are optional within schema version 1. Older drawings and rounds load with no study, no prompt suggestions recorded, and unavailable origins.
- **Study page:** `[study] enabled` or `--study` serves a page whose participant area holds only a title, the paper canvas with any overlay, the prompt, the two AI buttons while an AI drawing is open, and a status line worded the same in both conditions. Generation, selection, and session controls are not built. A collapsed "Experimenter" panel holds the name, a condition radio without a default, Start and Finish, the camera controls, the model descriptions with stand-in warnings, and the saving notice. Start needs both a name and a condition, then clears both and closes the panel.
- **Blinded export:** `sketchloop study export` loads every session of a runs folder, filtered by a name glob. It writes each finished study drawing's final canvas, re-encoded without metadata, as `drawings/<code>.png`, with a random six-character code from `secrets`. Beside the drawings it writes `key.csv`, mapping codes to experiment, drawing, condition, status, reason, and checksums, and `export.json` with provenance and counts. The default folder is a new `data/study-exports/export-<time>`. The output is written to a temporary folder and renamed, and the command refuses a folder that is not empty, lies inside the runs folder, or is not ignored by Git.
- **Summary:** `sketchloop study summary` prints one row per study drawing for the experimenter and writes nothing.

Alternatives considered, detailed in the design:

- **A session-level prompt suggestion file:** rejected, because it splits a study drawing's record across files.
- **Tracking the last suggestion in the controller:** rejected, because reloads and second tabs make typed prompts look edited.
- **Exporting raw camera frames:** rejected, because they show hands and the room and are not straightened.
- **A plan file of names and conditions:** deferred until the pilot shows typing errors.
- **A separate key folder:** deferred.
- **Keeping the prompt suggester tied to the backend:** rejected by the owner, because the study would have needed a real image backend only to get Florence-2 prompts.

## Consequences

The page gains a second layout that the `ux-reviewer` and a pilot must check, and the webcam's behavior inside a closed accordion is unverified. The study's settings name `florence` for prompt suggestions explicitly, since the default is the model-free template list.

The owner accepted the recorded 512x360 canvas for the pilot. A larger image for the experts would need a re-warp of the raw final frame, which stays a noted option that can be added without record changes. Prompt suggestions made on the generation page outside a drawing are lost if no round follows. Codes change on every export, so a second export cannot be joined to the first one's expert scores. With the owner's approval, S03 in PROJECT.md no longer mentions a design task, and the T10 row in ROADMAP.md names `sketchloop ui --study` instead of a `sketchloop study` runner.

Amended by owner decision: the page has one mode, the study layout, so `--study`, `study.enabled`, and the separate generation page are removed, and image generation stays in the command line.
