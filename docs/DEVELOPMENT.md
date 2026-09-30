# Development environment

## Baseline

Python 3.11+ runs the scaffold and its repository checks and pytest tests. Python 3.11 and 3.12 are configured in CI; local validation may use a newer interpreter. Compatibility of future ML packages must be evaluated separately before fixing the application runtime. No runtime dependencies, provider credentials, model downloads, camera access, or GPU are needed yet.

From the repository root:

```sh
python3 install.py && . .venv/bin/activate
python scripts/check.py
```

The installer uses the Python interpreter that launches it to create `.venv`. It also works when invoked by path from another directory. Existing valid environments are reused; existing incomplete directories are not deleted or overwritten. To change Python versions, move the old environment aside and rerun with the desired interpreter. Installation stops on any failed command and reports success only after `pip check` and two readiness tests pass (editable package import and temporary artifact read/write). It does not run the full suite or download models. Initial dependency installation requires network access or cached packages. If creation fails because `venv`/`ensurepip` is unavailable, install your operating system's Python venv support and move the partial environment aside before retrying.

On Windows, run `py -3 install.py` and activate with `.venv\Scripts\Activate.ps1`. Alternatively invoke `.venv/bin/python` or `.venv\Scripts\python.exe` directly. Checks run offline once the development dependencies are installed.

For a library-only editable install without testing tools (requires the build backend unless cached):

```sh
python -m pip install -e .
python -c "import sketchloop"
```

The package is an intentionally empty namespace scaffold. An import succeeding does not mean the application exists. Declare future runtime dependencies in `pyproject.toml`, keep hardware/backend extras optional, and commit a resolved lock or constraints file with the chosen compatible runtime when dependencies are introduced. Do not freeze an unrelated global environment or commit machine-specific paths.

## Commands

Run these from the repository root in the activated environment. Without activation, replace `python` with `.venv/bin/python` (Linux/macOS) or `.venv\Scripts\python.exe` (Windows). On Windows, `python3` may resolve to a Microsoft Store stub, so avoid it there.

| Command | Purpose |
| --- | --- |
| `python3 install.py` (`py -3 install.py` on Windows) | Create/reuse `.venv`, install development dependencies, and run two readiness tests |
| `python scripts/check.py` | Privacy/publication guard, local Markdown target checks, Python syntax checks, Ruff lint, pytest tests |
| `python scripts/privacy_check.py` | Scan publishable working files and staged blobs; reject private artifact paths and suspicious data patterns |
| `python -m pytest` | Run offline tests only |
| `make check` | POSIX convenience alias for the full check (override `PYTHON` if needed) |

The environment stays in `.venv` and uses the shell prompt name `sketchloop-ai`. Running the installer alone prepares it but cannot activate it in the calling terminal. If Conda shows `(base)`, run `conda deactivate` before activating the project environment to avoid stacked prompts. Check the selected interpreter with `python -c "import sys; print(sys.prefix)"`, which should point to this repository's `.venv`. Rerunning the installer refreshes the prompt name of an existing environment without removing its packages.

Default verification is deliberately lightweight. As application code grows, add selected lint/type tools with reproducible versions and meaningful contract/integration tests. Keep hardware/network tests separately invoked and accurately report skipped coverage.

## Repository layout

- `src/sketchloop/`: Python library; implement modules as their slices begin.
- `tests/`: synthetic, offline tests; never use copied private documents as fixtures.
- `docs/`: durable project knowledge, decisions, and handoff.
- `scripts/`: repository maintenance checks.
- `.github/`: CI.
- `notebooks/`: exploratory research notebooks, not library code.
- `CLAUDE.md`, `.claude/`: Claude Code entry point, skills, subagents, and shared permissions.
- `runs/`, `data/`, `models/`, `private/`, `tmp/`: ignored local-only artifacts, created as needed.

CI runs the installer, then runs the full offline checks using its prepared environment on pushes and pull requests. It uses read-only repository permissions, does not request secrets, and does not execute real inference. See the [official checkout action](https://github.com/actions/checkout), [Python setup action](https://github.com/actions/setup-python), and [PyPA packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) for the configuration conventions used here.

## Delivering a change

Select a roadmap task, define its observable outcome, implement a complete slice, run focused verification plus repository checks, review the diff, and update the handoff. Record important design changes in an ADR. Include requirement IDs and actual validation in PR descriptions. Do not conflate a passing offline test with a successful camera/model demonstration.

## Test conventions

Use pytest test functions, plain assertions, `tmp_path` for temporary files, and parametrization for related cases. Discovery is restricted to `tests/`; the repository scripts directory is configured as an import path in `pyproject.toml`, so tests need no manual `sys.path` edits. The pytest version is pinned in the `dev` extra. See [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html) for the supported convention.

## Code style

`AGENTS.md` is the single source for code style, docstrings, and import ordering. `scripts/check.py` runs `ruff check` (pinned in the `dev` extra) for pyflakes errors, wildcard imports, missing type hints (`ANN`), and the 130-character hard limit. The 120-character target, blank-line rules, import blocks, and naming are not machine-checked and rely on review. The Ruff formatter is not enforced. See [ADR 0004](decisions/0004-research-workspace-and-tooling.md).

The installer's `run(label: str, command: str) -> None` accepts a POSIX-quoted command string, parsed with `shlex.split` and executed without a shell on all platforms. Quote dynamic arguments with `shlex.quote`; shell operators and variable expansion are not supported.

## Coding agents

`AGENTS.md` is the single working agreement for all coding agents. Tool-specific entry points only import it and add tool mechanics, so shared rules are never duplicated. See [ADR 0003](decisions/0003-claude-code-environment.md).

- Claude Code reads `CLAUDE.md`, which imports `AGENTS.md`, `docs/STATUS.md`, and `docs/PROJECT.md`. Shared configuration in `.claude/` provides the `/start-task`, `/handoff`, `/adr`, `/commit`, `/bisect`, `/readability`, and `/audit` skills, a permission policy in `.claude/settings.json`, and six role subagents: `architect`, `researcher`, `python-developer`, `qa-engineer`, `code-reviewer`, and `privacy-reviewer`. `CLAUDE.md` describes their scope and the typical task flow. Research notes from the `researcher` go in `docs/research/`. That policy allows all shell commands and edits within the repository. It asks before push, merge, rebase, and pull. It denies force pushes, destructive deletes outside the repository tree, privilege escalation, disk tools, and reading ignored private material. Personal overrides go in the ignored `.claude/settings.local.json`.
- Agents that read `AGENTS.md` natively need no extra files.
