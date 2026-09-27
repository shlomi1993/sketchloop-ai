# ADR 0002: Use pytest for testing

Status: accepted. Supersedes the standard-library test-runner choice in ADR 0001.

## Context

The project owner selected pytest as the testing framework.

## Decision

Use pytest test functions, plain assertions, parametrized cases, and built-in temporary-path fixtures. Configure discovery and script imports in `pyproject.toml`. Pin pytest in the development extra; keep application runtime dependencies empty. Local checks, Make targets, and CI all invoke pytest. Install development dependencies before running checks; execution remains offline.

## Consequences

The original privacy regression coverage is retained with individually reported parameter cases. Development now requires installing pytest, while the standalone publication guard still uses only the standard library. The direct test-runner version is pinned; transitive dependencies are resolved by pip and are not yet fully locked.
