# Experiment store: MLflow tracking or a filesystem store

## Question and decision informed

Should durable runs, iterations, selection events, and artifacts (R06, R09) live in MLflow tracking or in a plain filesystem store of versioned JSON plus artifacts under `runs/`? Informs the T05 store ADR. The T01b runnable increment already saves outputs under `runs/`.

## Sources (read 2026-10-02)

- PyPI JSON metadata: `mlflow` 3.16.1 and `mlflow-skinny` 3.16.1 (2026-09-16), Apache-2.0, Python >=3.10, "OS Independent".
- [MLflow backend stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) and [tracking overview](https://mlflow.org/docs/latest/ml/tracking/) (latest docs).
- [MLflow filesystem backend deprecation notice](https://github.com/mlflow/mlflow/issues/18534) (GitHub issue).
- [MLflow usage tracking](https://mlflow.org/docs/latest/community/usage-tracking/) (latest docs).
- [mlflow-skinny on PyPI](https://pypi.org/project/mlflow-skinny/).
- Python standard library: `json`, `hashlib`, `os.replace`, and exclusive-create mode `open(path, "x")`.

## Findings

- Verified: full `mlflow` 3.16.1 requires, among others, Flask, SQLAlchemy, Alembic, pandas, pyarrow, scipy, scikit-learn, matplotlib, docker, gunicorn, and waitress, plus `mlflow-skinny` and `mlflow-tracing`.
- Verified: `mlflow-skinny` excludes "SQL storage, server, UI, or data science dependencies", but still requires FastAPI, uvicorn, pydantic, OpenTelemetry, protobuf, GitPython, and databricks-sdk.
- Verified: SQLite (`sqlite:///mlflow.db`) is the default backend store. The file backend is "in maintenance mode and will not receive further updates" and is marked to be deprecated. The notice issue places the switch in 3.7.
- Verified: MLflow sends opt-out usage telemetry. Disable it with `MLFLOW_DISABLE_TELEMETRY=true` or `DO_NOT_TRACK=true`.
- Verified (GitHub issues): MLflow run params are immutable ("Changing param values is not allowed"). Tags can change and metrics append.
- Inference: the MLflow docs do not describe artifact checksums, so SHA-256 integrity (already in `ImageRef`) would remain project code either way.
- Inference: MLflow models runs, params, and metrics. SketchLoop's core records (iteration lineage, candidate sets, explicit selection events, requested versus effective settings) would become nested runs and JSON artifacts, so the project schema exists anyway and MLflow adds a second representation.
- Inference: a filesystem store needs only the standard library, works offline and on Windows, is easy to inspect and diff, and fits immutable write-once records. Its weak points are concurrent writers and querying across many runs, neither needed for one local user.

## Options compared

| Criterion | Filesystem store (JSON + artifacts) | MLflow (full) | mlflow-skinny |
| --- | --- | --- | --- |
| Install weight | None (stdlib) | Heavy (server, ORM, data science stack) | Medium, needs a server or deprecated file backend |
| Windows and offline | Yes | Yes, telemetry must be disabled | Yes, same telemetry note |
| Immutable records | By design (exclusive create) | Params only | Params only |
| Artifact checksums | Project code, already in `ImageRef` | Not documented | Not documented |
| Lineage and selection | Native in project schema | Encoded as tags or JSON artifacts | Same |
| Browsing UI and comparison | None (files, or the T06 history view) | Built-in UI | None |
| Default tests offline | Trivial with `tmp_path` | Possible with a local SQLite URI | Same |

## Recommendation and confidence

Use a filesystem store as the authoritative `ExperimentStore`. Consider an optional MLflow exporter later, behind the store interface, only if cross-run comparison in its UI becomes a concrete need (for example T08 or T09 latency tables). Confidence: high for one local user. Revisit for multi-user or large-scale studies.

## Implementation guidance

- Target module: `sketchloop.experiments`. Standard library only, no new dependency.
- Layout sketch for the architect: `runs/<run_id>/run.json`, `runs/<run_id>/iterations/<iteration_id>.json`, `runs/<run_id>/selections/<event_id>.json`, and artifacts under `runs/<run_id>/sources/` and `candidates/`, referenced by relative `ImageRef` paths with SHA-256.
- Every JSON record carries `schema_version`. Write to a temporary file in the same directory, then `os.replace` to a name created with exclusive mode. Never overwrite an existing record. Append new events instead of editing old ones.
- Loading verifies each artifact's SHA-256 and size and reports missing or mismatched files explicitly, rather than failing silently or "repairing".
- Record environment provenance once per run (package versions, platform summary without host names or user paths) and per-iteration effective settings, timings, and backend identity.
- Pitfalls: keep paths relative and POSIX-style in JSON so runs move between Windows and Colab. Sort JSON keys for stable diffs. Store floats as written, and do not round-trip seeds through floats.
- Tests: round-trip each record type through `tmp_path`, check that a second write of the same record raises, check that tampering with an artifact byte is reported on load, and simulate an interrupted write (temporary file left behind) being ignored.

## Open questions for the owner

- Should `runs/` live in the repository directory (git-ignored) or in a configurable data directory?
- Is MLflow's UI wanted for the thesis evaluation, or are generated tables and plots from the JSON records enough?
