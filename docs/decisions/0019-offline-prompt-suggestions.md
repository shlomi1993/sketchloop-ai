# ADR 0019: Offline style prompt suggestions from the sketch

Status: accepted, amended in the UI session (Florence-2-base, suggestions on request only, chosen with the backend)

## Context

While planning the UI session after T08, the owner asked the page to suggest style prompts, and asked that it work without a cloud model. [PROJECT.md](../PROJECT.md) lists automatic prompt improvement as an optional extension (p. 6, §8), and [ROADMAP.md](../ROADMAP.md) activates extensions only after a deliberate scope decision. R03 asks that a person controls style through text and model controls, and R04 keeps human choice explicit.

Facts: the page is Gradio over `AppController` ([ADR 0012](0012-gradio-ui-and-input-source.md), [ADR 0016](0016-keep-gradio-page-and-folder-store.md)). Generation on the owner's Apple M2 Pro takes about 5 to 23 s per round with SD 1.5 in float16 ([ADR 0018](0018-float16-on-the-apple-gpu.md)), and the machine has 16 GB shared between CPU and GPU. Assumption to verify: a small vision-language or captioning model can run on MPS next to SD 1.5 within that memory and propose useful prompts from a line sketch.

## Decision

- **Scope:** activate offline prompt suggestions as owner requirement U1 in [ROADMAP.md](../ROADMAP.md).
- **Offline only:** no cloud model, API key, network call, or paid service. A cloud implementation is out of scope unless the owner decides otherwise later.
- **Interface:** a `PromptSuggester` abstract base class, a replaceable component like `Evaluator`, chosen through the settings file and independent of generation, evaluation, and selection.
- **Implementations:** a fixed list of style suggestions that needs no model and is always available, and one small local vision model that looks at the processed sketch and proposes full prompts. The `researcher` agent chooses the model by license, size, memory on MPS next to SD 1.5, and load time before any code is written.
- **Human control:** a suggestion only fills or extends the prompt box. The person edits it and clicks Generate, and nothing is generated automatically.
- **Records:** the round records which suggester was used and which suggestion, if any, the prompt came from, so suggestions stay traceable.

A cloud vision model would give richer suggestions but was rejected by the owner, and suggestions without the sketch alone would miss the request to base them on the drawing.

## Consequences

The page gains a suggestion step and a second local model with its own download, memory use, and load time, which must be measured like T08 stages. Default checks stay offline through a fake suggester. The record change adds fields within schema version 1 that older rounds load as unavailable. The model choice, its quality on pencil sketches, and its memory cost are unverified until the research note and a measured run.

The UI session implemented this decision with three changes. The [research note](../research/prompt-suggestion-models.md) chose Florence-2-base (MIT, about 463 MB), which names the sketched object, and the page fills five style templates with it, since a captioning model describes the drawing rather than proposing a style. The page first wrote a suggestion automatically whenever the drawing changed, but camera noise, a hand passing over the paper, and the wait for a still frame made that unreliable and hard to follow, so the owner reverted it, and a suggestion is now written only when the person presses "Suggest a prompt". The suggester is chosen with the backend, Florence-2 next to diffusers and the templates next to the fake, rather than through the settings file, and the round does not yet record the suggester or the suggestion. The model loads and captions a blank image when the page starts, so the first suggestion takes about 0.3 s on the M2 Pro instead of about 6 s.
