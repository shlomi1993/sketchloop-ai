# ADR 0007: Rich for terminal output

Status: accepted

## Context

The `sketchloop` command (T01b) printed its banner and candidate list with plain `print` calls and hand-drawn separators. The owner wants readable terminal output without writing formatting code by hand.

## Decision

Use Rich (`rich==15.0.0`, MIT license, depends on `markdown-it-py` and `pygments`) as the first runtime dependency, for panels, tables, colored messages, and the selection prompt in `cli.py`. Library modules (`domain`, `generation`, `fakes`, `capture`) do not import it, so it stays a UI concern. Alternatives were plain `print` (no dependency, but manual formatting) and Textual (a full terminal UI framework, more than needed before T06).

## Consequences

Default installs now pull Rich and its two small dependencies. Output adapts to the terminal and falls back to plain text when piped. Error lines are printed without wrapping or markup so they stay one line. The pin is updated deliberately, like the dev tools.
