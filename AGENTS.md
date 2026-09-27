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

Use `python3 scripts/check.py` (or `make check`) for repository checks. Use pytest with plain test functions, fixtures, and parametrization for meaningful unit and integration tests as behavior is implemented. Install the development extra (`python -m pip install -e ".[dev]"`) before running checks. Before a model/camera milestone is declared complete, run the corresponding real integration and record hardware and limitations. Do not silently replace a failed backend with synthetic output.

For durable interface, storage, dependency, or scope decisions, add an ADR using `docs/templates/DECISION.md`. Update requirements traceability and `docs/STATUS.md` when completing a task. At session end record what changed, checks and results, unresolved questions, and the exact next action; do not copy transcripts or local machine details.

## Communication style

- Avoid semicolons in user-facing messages. Use separate sentences or natural connecting words instead.

## Code style

- Always include type hints for every function parameter and return value, including helpers and tests.
- Use `-> None` for functions that return no value; implicit `self` and `cls` parameters need no annotation.

- Do not add file-level (module) docstrings unless explicitly requested by the user.

- Use a maximum line length of 120 characters, including indentation.
- Prefer a single line for calls, expressions, and collections when they fit within 120 characters and remain readable.
- Do not wrap a short call merely to put its arguments on separate lines.
- Prefer concise error messages and single-line `raise` statements within the 120-character limit.
- Shorten redundant wording instead of splitting an error message across adjacent string literals.
- Preserve the essential cause and recovery action; wrap only when that information cannot fit clearly on one line.
- Add a blank line after a `return` or `raise` statement when another statement follows, including after a guard clause before execution continues at an outer indentation level.
- Avoid trailing commas that exist only to force multiline formatting. Wrap when the line exceeds the limit or clarity requires it.

Preferred:

```python
result = subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT)
```

Preferred error-message style:

```python
raise RuntimeError("Existing .venv is incomplete. Move it aside and rerun the installer.")
```

Preferred guard-clause spacing:

```python
if privacy_main():
    return 1

failures = []
```

Apply the same spacing after `raise`. No extra blank line is needed solely at the end of a function or file.

## Import ordering

Use three import blocks, separated by one blank line:

1. Plain `import package` statements for standard-library and third-party packages.
2. `from package import name` statements for standard-library and third-party packages.
3. Imports from project-local modules, including relative imports.

Sort each block alphabetically by module/package path, and sort multiple imported names alphabetically.
Keep one plain import per line. Omit empty blocks. Required `from __future__` imports remain first,
in their own block. Keep project-local plain imports in the project-local block as well.

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
