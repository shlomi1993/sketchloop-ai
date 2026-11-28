# ADR 0028: Explicit configuration files

Status: accepted

## Context

The owner requested that `--config` have no default file. Previously the parser defaulted to None, but the shared settings helper automatically discovered `sketchloop.toml` in the working directory. This made invocation behavior depend on an unrequested file. R08 still requires configurable components, without requiring automatic discovery.

## Decision

Read a settings file only when an explicit path is supplied. With no path, use built-in defaults and any command-line overrides. Apply this rule to both the application and model downloader through their existing shared helper. Explicit missing files still fail, and file-relative paths and override precedence stay unchanged.

This amends automatic discovery in ADR 0014. Existing configuration files and keys remain valid when explicitly selected. No library API, storage schema, dependency, or UI layout changes are needed.

## Consequences

Users who relied on discovery must add `--config sketchloop.toml`. The study guide already supplies `--config study.toml`. Update existing tests and current usage documentation without adding tests, as requested. Historical designs retain the original decision. Internal library docstrings describe responsibilities independently of outer CLI command names.
