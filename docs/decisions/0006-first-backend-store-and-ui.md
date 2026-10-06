# ADR 0006: First backend, experiment store, and UI toolkit

Status: accepted

## Context

T01b needs the first-model, storage, and UI decisions (R02, R06, R07, R09). The development machine is a Windows laptop without a GPU, with Colab possible later. Options were compared in [first-model options](../research/first-model-options.md), [experiment store options](../research/experiment-store-options.md), and [UI toolkit options](../research/ui-toolkit-options.md).

## Decision

- **First real backend (T03):** Stable Diffusion 1.5 with the ControlNet v1.1 scribble model through Diffusers, with an LCM-LoRA fast mode for CPU. Fallbacks, in order: T2I-Adapter sketch on the same base, then SDXS sketch. The owner accepts the CreativeML OpenRAIL-M license for research use.
- **Speed target:** about one minute per round on the laptop CPU in fast mode at small resolution. This is a development target, not a requirement, and is adjusted from real measurements in T03 and T08. Colab is used when the target cannot be met.
- **Experiment store (T05):** a standard-library filesystem store of versioned JSON records and checksummed artifacts under `runs/`. MLflow is not adopted. It can be added later as an optional exporter.
- **UI (T06):** Gradio as an optional extra with analytics disabled. NiceGUI is the fallback if live camera preview is awkward in Gradio.

Exact package versions are pinned when each dependency is added, in the task that adds it.

## Consequences

T02 and T03 can proceed without further decisions. Heavy ML and UI dependencies stay in optional extras, so default checks remain offline and light. Negative prompts are unavailable in the LCM fast mode, which the backend's capabilities must report. CPU latency is unmeasured, so the backend choice is revisited if T03 measurements are far from the target.
