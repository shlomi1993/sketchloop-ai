# ADR 0023: CLIP evaluators and recorded score details

Status: accepted

## Context

The supervisor asked to test the system's modularity by adding components ([meeting notes](../meetings/2026-11-supervisor-meeting-1.md)), and R04 asks for a replaceable evaluation hook that permits similarity scoring such as CLIP. Until now the only evaluator was `NoOpEvaluator` ([ADR 0010](0010-sessions-and-evaluator.md)), its factory saw only `[evaluator]` settings, and a round identified its evaluator only by class name, which cannot tell two CLIP models apart.

Facts from the [research note](../research/model-clip.md): `openai/clip-vit-base-patch32` at revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` has only pickled weights, about 0.6 GB, and `openai/clip-vit-large-patch14` at `32bd64288804d66eefd0ccbe215aa642df71cc41` has safetensors, about 1.7 GB. The OpenAI CLIP code is MIT, and the model cards place deployed use out of scope. CLIP center-crops images to 224 px, so the edges of wide candidates are ignored.

## Decision

- **Evaluators:** the owner chose both models, selectable as `clip-b32` and `clip-l14` in `[evaluator]`, beside `none`. One `ClipEvaluator` in the new `sketchloop.evaluators` package scores each candidate by the cosine similarity of its image and the prompt, loading the model on first use.
- **Factories:** evaluator factories receive the whole settings, so a model-backed evaluator reads the shared models folder in `generator.diffusers.models_dir`, where every model already lives.
- **Weights:** ViT-B/32's `.bin` is converted once to safetensors on download, as in [ADR 0022](0022-t2i-adapter-sketch-backend.md). The download script fetches an evaluator's model only when the settings file names it or `--evaluator` asks for it.
- **Records:** `Evaluator.describe_scores()` returns what the scores mean, and each round stores it as `evaluation.score_details` within schema version 1. CLIP records its model, revision, and score meaning. Older rounds load it as empty, which matches the no-op evaluator they used.
- **Use of scores:** scores are shown and recorded as a research signal. They never filter, rank, or select candidates.

A top-level models setting would read better than a generator-named section, but it was left out to keep the settings unchanged.

## Consequences

Adding an evaluator touched the registry, the evaluator factories and their four callers, and the records, which feeds the T07 modularity evaluation. The first round's evaluation time includes CLIP's model load, so few-round benchmarks overstate it. `rerun` scores with the evaluator in the current settings, not the recorded one. On the M2 Pro, ViT-L/14 next to the T2I-Adapter backend peaked at 3.3 GB of process memory with no new swapping, and warm scoring took about 0.1 s per round. Whether CLIP scores track what participants prefer is unknown and is a research question, not a property of the system.

Follow-up from the T07 modularity results: the shared folder moved to a top-level `[models] dir` setting, `rerun` now reuses the evaluator a round recorded when it can and otherwise says which settings evaluator it uses, and an evaluator's load is timed as `model_load` before its first scoring rather than as evaluation.
