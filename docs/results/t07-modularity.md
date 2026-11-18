# T07 modularity validation

The supervisor asked to test the system's modularity by integrating a new model ([meeting notes](../meetings/2026-11-supervisor-meeting-1.md)). In November 2026, three components were added through the existing interfaces and the settings file, and one prerequisite change was made first (R02, R04, R08). This note records what each addition had to touch, as evidence of how replaceable the components really are.

## Summary

| Addition | Setting | New code | Existing source files changed | Orchestration, UI, or records changed |
| --- | --- | --- | --- | --- |
| Backend registry (prerequisite) | `generator.backend` | None | 7 | No |
| T2I-Adapter sketch backend ([ADR 0022](../decisions/0022-t2i-adapter-sketch-backend.md)) | `backend = "t2i-adapter"` | 1 module | 4 | No |
| CLIP evaluators ([ADR 0023](../decisions/0023-clip-evaluators.md)) | `name = "clip-b32"` or `"clip-l14"` | 1 package | 12 | Records and session, by one field |
| Minimal preprocessor | `name = "minimal"` | 1 class | 1 | No |

Counts exclude tests and documentation. Every addition kept `SketchSession`, `AppController`, and the page unchanged, and all of them are covered by offline tests.

## What each addition touched

**Backend registry.** Backend names were a fixed enum that seven files branched on: `domain`, `backends`, `settings`, the argument parser, the shared CLI helpers, `serve`, and `rerun`. Adding a model would have meant editing all of them. Backends are now registered by name in `components`, like the other components, and the CLI builds its `--backend` choices, name checks, and rerun mapping from that registry.

**T2I-Adapter backend.** One new module subclasses the ControlNet backend. The base class first needed three hooks, for building the pipeline, fusing LCM-LoRA, and passing the conditioning image, because those steps were written as one ControlNet-only block, and its adapter name had to become overridable. The registry gained one entry, the download script gained the source and a one-time conversion to safetensors, and `pillow` became a declared dependency. Its images follow the sketch, and rounds took 4.4 s in fast mode and 18.6 s in quality mode on the M2 Pro, about 15% to 20% faster than ControlNet.

**CLIP evaluators.** One new package holds the evaluator. Evaluator factories received only `[evaluator]` settings, so they could not find the models folder, and now receive all settings, which changed four CLI callers by one line each. A round identified its evaluator only by class name, which cannot tell two CLIP models apart, so the evaluator interface gained `describe_scores()` and the round record gained `score_details` within schema version 1. `show` prints it, and the download script fetches an evaluator's model on request.

**Minimal preprocessor.** One function and one class beside the stroke preprocessor, and one registry entry. Nothing else changed.

## Findings

- **Easy:** a component that needs no model and no new record data, such as a preprocessor, takes one class and one registry entry.
- **Moderate:** a new model of an existing kind takes a class, a registry entry, and a download source, once the base class has hooks for the steps that differ.
- **Harder than it should be:**
  1. **Models folder:** every model lives in the folder named by `generator.diffusers.models_dir`, a generator-specific key that evaluators and the prompt suggester also read.
     - **Fixed:** a top-level `[models] dir` replaces it, and the backends, evaluators, prompt suggester, `rerun`, and the download script all read it. Generator factories now receive the whole settings, like evaluator factories.
  2. **Backend wording:** the T2I-Adapter backend says "Using the diffusers backend" when it starts, and messages and help text name "diffusers" for options both backends share.
     - **Fixed:** each backend announces the name the person typed, from a `BACKEND_NAME` the registry also uses, and the `--mode` and `--precision` help and the load messages no longer imply one backend.
  3. **Evaluator load time:** a model-backed evaluator loads during the first round, so its load time is counted as evaluation time, while generators report model loading separately.
     - **Fixed:** `Evaluator.load()` lets the session load an evaluator once before its first scoring and add that time to the round's `model_load`, so `evaluation` is only scoring. The records keep schema version 1.
  4. **Rerun evaluator:** `rerun` scores with the evaluator in the current settings rather than the one the round recorded.
     - **Fixed:** `rerun` recreates the recorded evaluator, telling the CLIP models apart by the recorded model, and otherwise uses the settings evaluator and prints one line saying so.

These four items were fixed in the next step, "Fix what was hard", as each item's Fixed line says.
