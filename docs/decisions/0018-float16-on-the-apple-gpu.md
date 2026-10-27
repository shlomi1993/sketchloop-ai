# ADR 0018: float16 on the Apple GPU and a precision option

Status: accepted

## Context

T08 (R09) asks to optimize the measured bottleneck. [The T08 baseline](../results/t08-latency-baseline.md) showed that generation is over 99% of every round on the owner's Apple M2 Pro, and that the float32 model was paged to swap with ordinary apps open, so memory was the next limit. [ADR 0009](0009-diffusers-backend.md) ran float16 only on CUDA and float32 on CPU and MPS, since SD 1.5 in float16 on MPS was reported to produce NaN (black) images. The precision was fixed by the device and could not be chosen.

Measured facts, with heavy apps closed, seed 0, 2 candidates at 360 x 512, 1 warm-up and 5 measured rounds per run, `diffusers==0.40.0` and `torch==2.14.1`:

- Quality mode (20 steps): round median 25.85 s in float32 and 23.27 s in float16, 10% faster.
- Fast mode (4 steps): round median 6.17 s in float32 and 5.16 s in float16, 16% faster.
- float16 images differ from float32 images of the same seed by a mean of about 0.1 of 255 per pixel, a PSNR of 55 to 59 dB, and none was black. Within each precision, repeated rounds with the same seed were identical.
- After loading in float16, the U-Net, ControlNet, VAE, and text encoder all report `torch.float16`.

The process's memory use was not measured, and half the weight memory is the expected effect of float16, not a measurement.

## Decision

- **Option:** the diffusers backend takes a precision, `float16` or `float32`, from `--precision` on `run`, `ui`, and `benchmark`, or from `generator.diffusers.precision` in the settings file, and `--precision` needs `--backend diffusers` like `--mode`.
- **Default:** float16 on any GPU, CUDA or Apple MPS, and float32 on the CPU. This replaces the device line of ADR 0009 for MPS.
- **CPU:** float16 on the CPU is refused when the backend is created, with one line, because the CPU has no fast float16 kernels.
- **Records:** the effective `dtype` control records the precision used, as before, and `rerun` recreates the backend with the recorded precision, as it does with the mode.

Keeping float32 as the MPS default was rejected because the measured speed and the image comparison favor float16 and float32 stays one flag away. Lower step counts or a smaller resolution would change the images, not only their cost, so they are left for an owner decision.

## Consequences

New MPS sessions record `dtype` float16, so a rerun of an older float32 round on the same machine uses float32 and a comparison between old and new sessions shows the dtype difference. A round recorded in float16 on a GPU cannot be rerun on a CPU-only machine without editing its precision, and that rerun fails with the one-line CPU message instead of silently changing precision. The NaN risk is checked only for this model, these library versions, and the M2 Pro, so a black image after an upgrade should be checked against float32 first. Continuous suggestions while drawing remain infeasible at about 5 s per fast round.
