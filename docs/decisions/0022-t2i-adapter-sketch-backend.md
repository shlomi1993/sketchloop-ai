# ADR 0022: T2I-Adapter sketch as a second generation backend

Status: accepted

## Context

The supervisor asked to test the system's modularity by choosing a new model and integrating it ([meeting notes](../meetings/2026-11-supervisor-meeting-1.md)), which the roadmap places in T07 (R02, R08). Until now the only real backend was SD 1.5 with ControlNet scribble ([ADR 0009](0009-diffusers-backend.md)), and backend names were a fixed enum that seven files branched on. That enum was replaced by a name registry in `sketchloop.components`, so a new backend needs a class and one registry entry.

Facts from the [research note](../research/model-t2i-adapter-sketch.md): the sketch adapter `TencentARC/t2iadapter_sketch_sd15v2` at revision `a73a564e35b0956deffd164e1ee463f967c3f952` is Apache 2.0, has 77M parameters, runs once per image rather than every step, and loads beside the existing SD 1.5 weights through `StableDiffusionAdapterPipeline`. It expects white strokes on black, like ControlNet scribble. Its weights exist only as a `.bin` file, and the pipeline has no `load_lora_weights`, so LCM-LoRA needs a lower-level loading path. Estimates, not yet measured: about 0.15 GB more in float16 and rounds at 70 to 80% of the ControlNet time.

## Decision

- **Model:** add SD 1.5 with T2I-Adapter sketch as the backend `t2i-adapter`, beside `fake` and `diffusers`. Its paper was already on the reading list, and it reuses the downloaded SD 1.5 base.
- **Settings:** it shares `[generator.diffusers]`, since mode, precision, and the models folder mean the same for both backends. No settings keys are added.
- **Weights:** `scripts/download_models.py` downloads the pinned `.bin` and converts it once to safetensors, so the backends still load only safetensors as ADR 0009 requires.
- **Conditioning:** the existing `conditioning_scale` control maps to the adapter's `adapter_conditioning_scale`, with a default of 1.0, the model card's default.
- **Modes:** quality mode mirrors the ControlNet backend. Fast mode fuses LCM-LoRA into the UNet through the loader functions the research note verified.

SDXS-512 sketch, a one-step model that might help the Windows laptop, was considered and deferred because quality evidence is thin. SD-Turbo with ControlNet was rejected over its license restrictions and unverified ControlNet support. Loading the `.bin` directly would have saved the conversion but broken the safetensors-only rule.

## Consequences

Adding the backend validates the registry from the T07 refinement, and the files it touched are recorded for the modularity evaluation in T07. The download grows by about 0.3 GB, and disk use by about 0.6 GB, since the `.bin` is kept so a later download skips it. Rounds record the adapter name, revision, and effective controls like the ControlNet backend, so `rerun` can recreate them. Image quality across varied pencil sketches, the CPU path on the Windows laptop, and memory use stay unverified. A first real run on the M2 Pro in float16 gave images that clearly follow the sketch in both modes, and `sketchloop benchmark` with 2 candidates measured median rounds of 4.4 s in fast mode and 18.6 s in quality mode, about 15% and 20% faster than the ControlNet backend, with other open apps unchecked.
