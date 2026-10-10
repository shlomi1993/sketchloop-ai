---
name: readability
description: Review SketchLoop Python code for human readability, especially self-explanatory names, and report concrete renames and simplifications. Use when reviewing a diff, or when asked whether code is easy to read.
argument-hint: "[optional files or diff range]"
---

# Readability review

Scope: $ARGUMENTS, or by default the current `git diff` and staged changes

Read the code as a newcomer to the project would. The question is whether someone can understand what each part does without reading its callers.

## Names

- Every variable, function, class, and parameter name says what it holds or does. Flag single letters, except conventional loop indices like `i` or short comprehension variables. Also flag abbreviations such as `cfg`, `img`, `res`, and `tmp`, and vague names such as `data`, `info`, `obj`, `result`, `handle`, and `process`.
- Functions start with verbs (`load_manifest`, `validate_selection`). Booleans read as questions (`is_complete`, `has_seed`). Collections are plural.
- Include units and formats where they matter (`timeout_seconds`, `width_px`, `created_at_utc`).
- Use the project's domain terms consistently: sketch, capture, preprocessing, guidance, candidate, iteration, selection, experiment, artifact. Do not mix synonyms for the same concept.

## Structure

- Functions do one thing and fit on a screen. Nesting stays shallow, and guard clauses with early returns are preferred.
- Magic numbers and strings become named constants.
- Single-line statements are preferred, but split a line that is hard to read. Compact code does not justify clever code.
- Comments explain why, not what. A comment that restates the code signals that a better name is needed.

## Report

List each finding as `file:line`, the current code or name, and a concrete suggestion, for example `res` → `generation_result`. Put the few most confusing spots first. Skip anything that is already clear, and say plainly if the code reads well.
