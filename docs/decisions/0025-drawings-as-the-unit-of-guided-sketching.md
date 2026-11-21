# ADR 0025: Drawings as the unit of guided sketching records, with named experiments

Status: accepted

## Context

[ADR 0024](0024-next-stroke-suggestions-and-follow-detection.md) decided how next strokes are suggested and how following is detected. It left the records to the `architect`, within schema version 1. S01, S02, and S04 in [PROJECT.md](../PROJECT.md) need suggestion events, captures of the drawing over time, per-part follow scores, and a decision for each suggestion. These must be recorded separately from explicit selections, and the inference must be recomputable. T10 adds study conditions with or without AI, and each task ends with a final drawing. The owner asked that a whole experiment's results be saved under a name the experimenter enters, such as `participant1_car`, and that the system never be told the task: the participant receives the task, writes the prompt, and makes the sketch.

Facts:

- **Rounds:** today a session holds numbered rounds. Each round is one generation iteration with candidates and lineage checks, written as `round-<n>/round.json`. `session.json` is written once.
- **Loading:** `load_session` reads only `round-<n>` folders and ignores any other folder.
- **Generator:** the diffusers generator loads its pipeline lazily on the first generation.
- **Measurements:** the research note measured about 3 s per suggestion on the M2 Pro, a 3.6 GB peak with the suggestion pipeline and Florence-2 loaded, and about 2.3 GB that sharing weights with the generator would save.

Assumptions:

- **Drawing length:** a study drawing lasts about 15 minutes.
- **Storage:** at a 2 s interval, the captures take about 75 MB per drawing when interval frames are JPEG, which the pilot must confirm.
- **Neutral suggestions:** suggestions from an empty prompt were not tested in the research note.

## Decision

- **Unit:** suggestions and captures belong to a new **drawing**, from its start to its end, stored as `drawing-<n>/drawing.json` in the session folder beside `round-<n>/`, with its files under `captures/` and `suggestions/suggestion-<k>/`. A session can hold rounds, drawings, or both, and at most one drawing is open at a time. T10 adds the study fields to the drawing and session without moving anything, and its no-AI condition is a drawing with suggestions off.
- **One record file:** `drawing.json` holds the canvas size, capture interval, suggester, follow method and parameters, rule, every capture with its alignment and per-part scores at 4, 8, 16, and 24 px, every suggestion with its request, effective settings, model identity, parts, overlay, timings, window, and decision, and the final capture. It is rewritten atomically after each change, files first, like `round.json`.
- **Recomputable decisions:** each decision stores its method, version, and rule, and a pure function decides again from the stored scores. Open windows are left undecided in the file, and `show` labels anything it computes.
- **Selections stay separate:** no field of a drawing is a selection, and rounds keep explicit selections.
- **Compatibility:** schema version stays 1. Older sessions load with no drawings, and older code ignores drawing folders.
- **Named experiments:** the experimenter enters a name when starting a drawing on the page. The name names the **session**, not the drawing, so the folder is `runs/<name>/` and holds the drawing and any rounds generated during the experiment, `show` finds it by name, and uniqueness is one exclusive folder creation. Names are 1 to 64 ASCII letters, digits, `_`, or `-`, start with a letter or digit, are not Windows device names, and must differ, ignoring case, from every entry of the runs folder. An invalid or taken name is refused with one line before anything is saved. `session.json` gains an optional `name`, null for time-named sessions and older ones. The page hints that names must be codes and never real names, and the name is the only study label the system stores.
- **No task in the system:** there is no goal input. A suggestion is conditioned on the prompt box text as the participant typed it, which may be empty, in which case a neutral line-drawing prompt is used. The request records the typed text unchanged, and the effective settings record the exact text used, the template, the token count, and whether it was truncated.
- **Capture format:** an interval capture every 2 s keeps its raw frame as JPEG at quality 90, and the captures at each suggestion and at finish keep PNG. Every capture also keeps its paper-frame canvas as PNG.
- **Separate pipeline:** the real suggester owns its own img2img pipeline instead of sharing the generator's through `from_pipe`. This keeps it independent of the backend and mode, at about 2.3 GB more memory only when generation also runs in the same process.
- **Fixed canvas:** the paper frame is warped to a fixed canvas size from the settings, 512x360 by default, for A4 in landscape, so captures are comparable and the suggester can warm up once at that size.
- **Settings:** `[suggester] name` chooses `fake`, the default, or `controlnet`. `[suggester.controlnet] precision` and `[drawing]` with `canvas_width`, `canvas_height`, and `save_interval_seconds` are optional.

Alternatives considered are recorded in [the T09 design](../design/t09-guided-sketching.md):

- **Suggestions inside rounds:** rejected, because a round is a generation iteration and a no-AI drawing has none.
- **A session-level timeline:** rejected, because a session can hold rounds and drawings, and T10 may still put several drawings in one session.
- **Naming the drawing instead of the session:** rejected, because a drawing folder is found only by searching every session, and rounds generated during the experiment would sit in a time-named folder.
- **A goal field:** rejected by the owner, because the system should not know the task.
- **Append-only JSON Lines:** deferred until rewrites are measurably slow.
- **Change-triggered captures:** rejected, because camera noise made change detection unreliable in [ADR 0019](0019-offline-prompt-suggestions.md).

## Consequences

- **Additions:** `ExperimentStore` gains `claim_session`, `locate_drawing`, and `save_drawing`, `SessionRecord` gains `name`, `SavedSession` gains `drawings`, and `show` prints the experiment name and a table per drawing.
- **T10:** builds on named sessions. It may compose names from codes and the order of conditions, and its blinded export must map names to random codes with the key kept local, since names may encode a participant code and a task. Its roadmap row, which has the experimenter enter the task, is reconciled with the name being the only stored study label when T10 is designed.
- **Wording:** S01 in PROJECT.md says "toward the stated goal". The goal is now stated only through the participant's prompt, which the handoff can reflect in S01. `rerun` is unchanged, and drawings are inspected rather than rerun.
- **Storage:** each drawing writes hundreds of image files, so the capture interval and raw format set the disk use. The owner chose 2 s with JPEG for interval frames, and both remain settings.
- **Rule and labels:** the provisional rule is a code constant, so changing it after the pilot is a recorded decision, and older drawings keep the rule they were decided with. Part labels are reserved but null until a labeler and its vocabulary are decided.
- **Memory:** the memory of both pipelines together, and the time to rewrite `drawing.json` late in a long drawing, are measured in T09's model test and the T10 pilot.
