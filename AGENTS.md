# Agent working agreement

## Start every session

1. Read `docs/STATUS.md` for current state and the next unblocked task.
2. Read `docs/PROJECT.md` for the authoritative, sanitized project requirements.
3. Read `docs/ARCHITECTURE.md` and `docs/EXPERIMENTS.md` before changing interfaces or persistence.
4. Select one bounded item from `docs/ROADMAP.md`. Check the worktree before editing; preserve unrelated work.
5. Read relevant decisions in `docs/decisions/`. Explicit user instructions take precedence over this agreement.

The original proposal is private and is not needed to develop this project. Never require it in a future session. Requirements R01-R10 come from the proposal; architecture and tooling choices are repository decisions, not additional source requirements.

## Project invariants

- Build a modular Python research library and a thin interface for a human-led, iterative physical-sketch workflow.
- Keep capture, preprocessing, generation, evaluation, human selection, orchestration, persistence, and UI separate.
- Start with manual triggering and one real generative model. A fake backend supports tests but does not satisfy the working-system milestone.
- Preserve the source sketch, effective configuration, outputs, timings, and iteration lineage. Never claim exact replay when a backend cannot guarantee it.
- Human selection remains explicit. Similarity scores are research signals, not proof of design quality or satisfaction of functional constraints.
- Default checks must work without a camera, GPU, network, model weights, or credentials. Hardware/model tests are separate and explicitly invoked.
- Continuous generation, collaborative use, personalization, and participant studies are extensions, not MVP blockers.

## Work cycle

State the intended outcome, implement the smallest complete slice, and validate the behavior affected. Use typed public interfaces, explicit data contracts, dependency injection, and clear errors at boundaries. Avoid introducing a framework, service, provider lock-in, or heavyweight dependency without a concrete need. Verify current upstream APIs before adopting dependencies, then record compatible versions and a reproducible lock/constraints strategy.

Prefer the smallest diff that fully solves the problem. Small changes are easier to review, fix, and revert. When two solutions behave the same, choose the one that changes fewer lines and files. Do not refactor, rename, reformat, or add abstractions beyond what the task needs. Split large work into separate small commits.

Run repository checks with the project interpreter: `.venv/bin/python scripts/check.py` on Linux/macOS, `.venv\Scripts\python.exe scripts/check.py` on Windows, or `python scripts/check.py` in an activated environment. `make check` is a POSIX convenience alias. On Windows, `python3` may be a Microsoft Store stub and `make` is often unavailable. Use pytest with plain test functions, fixtures, and parametrization for meaningful unit and integration tests as behavior is implemented. Install the development extra (`python -m pip install -e ".[dev]"`) before running checks. Before a model/camera milestone is declared complete, run the corresponding real integration and record hardware and limitations. Do not silently replace a failed backend with synthetic output.

For durable interface, storage, dependency, or scope decisions, add an ADR using `docs/templates/DECISION.md`. Update requirements traceability and `docs/STATUS.md` when completing a task. At session end record what changed, checks and results, unresolved questions, and the exact next action; do not copy transcripts or local machine details.

## Communication style

- Avoid semicolons in user-facing messages. Use separate sentences or natural connecting words instead.

## Code style

- Type hints on every function and method parameter and return value, including helpers and tests. Use `-> None` when nothing is returned. Implicit `self` and `cls` need no annotation.
- Target 120 characters per line. Up to 130 is fine when it keeps a statement on one line and the file shorter.
- Prefer single-line statements, calls, and collections unless one line is hard to read. Do not wrap short calls or add trailing commas just to force multiline formatting.
- Keep log strings and error messages on one line. Shorten redundant wording instead of splitting a message across string literals, but keep the cause and recovery action.
- Add a blank line after `return` or `raise` when another statement follows, including after a guard clause. Add a blank line before a comment unless it is the first line of a block. No extra blank line is needed at the end of a function or file.
- In tests, give every `assert` a short message that says what went wrong, for example `assert seeds == [7, 8], f"Expected seed + index, got {seeds}"`.
- Use self-explanatory names. Avoid single letters and abbreviations except conventional loop indices, and include units where they matter (for example `timeout_seconds`).

```python
if privacy_main():
    return 1

# Scan index blobs too, so a cleaned worktree cannot hide a staged secret.
failures = []
raise RuntimeError("Existing .venv is incomplete. Move it aside and rerun the installer.")
```

## Docstrings

Do not write docstrings during development. Add them before committing (the `/commit` skill does this) for functions, methods, and classes in the change, except tests. Never add a file-level (module) docstring. Keep them short: one to three lines of explanation (one is best), an `Args:` section with one line per argument, and a `Returns:` section. Do not document raised exceptions.

```python
def filter_candidates(candidates: list[Candidate], threshold: float = 0.5) -> list[Candidate]:
    """Keep candidates whose similarity score reaches the threshold.

    Args:
        candidates (list[Candidate]): Candidates to filter.
        threshold (float, optional): Minimum similarity score to keep. Defaults to 0.5.

    Returns:
        list[Candidate]: Candidates at or above the threshold, in original order.
    """
```

## Import ordering

Use three blocks separated by one blank line, sorted alphabetically within each block. Never use wildcard (`*`) imports.

1. `import package` for standard-library and third-party packages, one per line.
2. `from package import name` for standard-library and third-party packages.
3. Project-local imports (`sketchloop` and relative imports), including plain local imports.

Omit empty blocks. Sort multiple imported names alphabetically. `from __future__` imports come first in their own block.

```python
import os
import subprocess

from collections.abc import Callable
from pathlib import Path

from sketchloop.domain import Candidate, Iteration
```

## Privacy and publishing

Never copy proposal identities, identifiers, affiliations, contact information, signatures, raw extracted text, screenshots, or PDF metadata into repository files, examples, test fixtures, issues, or commit messages. Do not encode private values in a denylist. Use synthetic data and relative paths; keep credentials and research artifacts in ignored local storage. See `docs/PRIVACY.md`.

Run the checks and review the full diff before committing. Ignoring a file does not remove it from the index or history. Do not push, publish, upload participant data, or invoke paid services without authorization for that action. Routine local edits and tests need no extra confirmation.

## Definition of done

A task has working behavior (or the specified document), focused verification, updated documentation/traceability, no private artifacts in the publishable diff, and a usable handoff. Distinguish implemented, tested, proposed, and blocked work. Do not mark the overall application complete while only the environment exists.
