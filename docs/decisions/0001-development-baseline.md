# ADR 0001: Offline-first development scaffold

Status: accepted for environment setup. Application architecture remains proposed.

## Context

The proposal requires a modular Python research library, a camera-driven iterative workflow, replaceable generation, and experiment reconstruction. The repository initially contained only a short README and Python ignore rules. Hardware, first model, and UI implementation are unspecified.

## Decision

Maintain sanitized project knowledge in `docs/` and a repository-wide `AGENTS.md`. Use a `src/sketchloop` Python package scaffold, standard-library checks and unit tests, and GitHub Actions. Require Python 3.11+ for the scaffold and configure CI on 3.11/3.12. Use setuptools as the packaging backend with no runtime dependencies yet. Keep names and contact metadata out of the package.

Defer hardware/ML/UI dependencies until an implementation task evaluates their compatibility. Default tests must not download models, access cameras, call APIs, or require credentials. Keep source PDFs, extracts, model weights, experiment artifacts, and secrets out of Git and add a publication guard. The guard supplements human review.

## Consequences

New sessions can begin without the proposal or special hardware. Tests are useful for repository safety now but do not validate application behavior. There is no dependency lock yet because runtime dependencies are empty; selecting dependencies requires a compatible runtime and reproducible resolution. Hosted CI and dependency installation still require network access. This scaffold does not choose a software license or promise a real-time system.
