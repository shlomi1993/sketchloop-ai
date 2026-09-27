# Experiment records and reproducibility

Status: proposed implementation contract derived from R06; exact serialization is decided during T01/T05. This document adds engineering detail to the source requirement and does not describe an implemented store.

## Minimum information

| Record | Fields to preserve |
| --- | --- |
| Experiment | Schema version, opaque experiment ID, creation time, research configuration, code revision and dirty-state flag, environment/dependency snapshot |
| Iteration | Opaque iteration ID, parent iteration ID or null, attempt/retry relationship, status, start/end time, stage durations and units |
| Capture | Raw image artifact reference and checksum, capture timestamp, non-identifying acquisition settings, sketch version |
| Processing | Processed artifacts/checksums, ordered operation names/versions, effective parameters, crop/perspective transforms when used |
| Guidance | Prompt, negative prompt when supported, style/functional descriptions, reference/conditioning artifact references, validated requested controls |
| Generation | Adapter/version, model identity and revision, effective settings and defaults, seed(s) when supported, scheduler/steps/strength where applicable, device/runtime details relevant to reproducibility |
| Candidates | Stable IDs, output artifact references/checksums, per-candidate generation metadata |
| Evaluation | Evaluator identity/version/configuration, scores and meaning, filter decisions, or explicit absence of evaluation |
| Selection | Selected candidate IDs or explicit no-selection, iteration link, event time; separate from automated ranking |
| Failure | Failed stage, sanitized error category, completion state, retained partial artifacts; no credentials or raw private paths |

Record unavailable values as unavailable with a reason; do not fabricate seeds, versions, or timings. Requested configuration and effective configuration can differ and must remain distinguishable. Capture only environment details that matter; omit usernames, machine names, home directories, and credentials.

## Persistence behavior

Keep raw input and all generated alternatives, including filtered ones, so later analysis is possible. Do not overwrite artifacts on retry. Use stable relative artifact paths within an experiment directory; reject path traversal when loading manifests. Validate schema and checksums. Commit metadata atomically after artifact writes, and distinguish pending, complete, failed, and cancelled states as needed by the implementation. Define migrations before changing a persisted schema.

Default local artifacts belong under ignored `runs/`; exports must be deliberately sanitized before being copied into a public directory. Synthetic test fixtures are the only default public experiment examples.

## Three distinct operations

1. **Inspect:** load saved inputs, outputs, parameters, timings, scores, selections, and history without invoking a model.
2. **Reconstruct:** validate stored artifacts and recover effective configuration and model prerequisites. Report unavailable dependencies or model versions before running.
3. **Rerun:** generate a new linked attempt from reconstructed conditions. Record any differences and determinism limits; preserve the original record.

Restoring conditions is required. Bit-for-bit regeneration is not universally guaranteed across hardware, kernels, remote providers, or model revisions. Seeds alone are not a reproducibility guarantee. Test exact output equality with the deterministic fake adapter; characterize the real backend honestly.

## Evaluation plan

Use synthetic or authorized sketches to exercise capture/preprocessing, multiple candidates, selection, revised input and a second iteration, save/load, and rerun. Test missing/corrupt artifacts, unsupported controls, failed generation, interrupted persistence, and invalid lineage. Separate camera/model integration tests from default offline checks.

Measure capture, processing, generation, evaluation, persistence, and total response time. State image size, batch size, configuration, hardware/runtime, warm-up treatment, number of runs, and distribution (for example median and p95). The brief gives no fixed latency threshold. Establish a measured baseline before setting targets or claiming real-time behavior.

Human interpretation and automated similarity measure different things. Record score semantics; do not use CLIP similarity as a universal design-quality score. Model comparisons and formal user studies remain optional extensions.
