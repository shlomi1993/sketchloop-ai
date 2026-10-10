# ADR 0003: Claude Code agent environment

Status: accepted

## Context

The agent environment was first written around `AGENTS.md`, which Claude Code does not load on its own. Its commands also assumed `python3` and `make`, which fail on a default Windows setup where `python3` can be a Microsoft Store stub. The project owner asked for a Claude Code environment in the repository that lets Claude do anything in the repository without prompting, except `git push`, `merge`, and `rebase`, while declining clearly harmful commands. The owner also wants Claude able to edit its own settings.

## Decision

Keep `AGENTS.md` as the single tool-neutral agreement. Add `CLAUDE.md` that imports it along with `docs/STATUS.md` and `docs/PROJECT.md`, plus a short list of Claude Code specifics. Track shared configuration in `.claude/`:

- `settings.json` allows every Bash and PowerShell command, all reads, and edits anywhere under the repository root, including `.claude/` settings. It asks before `git push`, `merge`, `rebase`, and `pull`, because pull performs a merge or rebase. It denies force pushes, recursive or forced deletes aimed at the filesystem root, home, parent directories, or absolute paths, `sudo`/`su`, `dd`, `mkfs`, recursive `chmod`/`chown`, shutdown and reboot, disk-formatting cmdlets, and reading `private/` and `.env` files. PDFs and the `data/`, `runs/`, and `models/` directories were read-blocked at first, but those rules were removed at the owner's request so agents can inspect articles and run outputs.
- Skills: `start-task` and `handoff` encode the session-start and handoff steps, `adr` creates numbered decision records from the template, `commit` runs checks and a privacy review before committing and can only be invoked by the user, `bisect` reproduces an issue and locates the commit that introduced it in a separate worktree, and `readability` reviews naming and clarity and is preloaded by `code-reviewer`, and `audit` reviews the whole repository, unpushed commits, and history for exposure and simplification opportunities. The `commit` skill also adds docstrings, since the style rules defer them until commit time.
- Six role subagents in `.claude/agents/`: `architect` designs contracts, drafts ADRs, and edits `docs/` only. `researcher` covers papers, official docs, and model and dependency evaluation, with notes in `docs/research/`. `python-developer` does test-first implementation. `qa-engineer` writes test plans against acceptance criteria and tests, and diagnoses failures. `code-reviewer` is read-only and reports prioritized findings. `privacy-reviewer` does a read-only diff review before commits.

The subagents adapt public best practice rather than inventing a format. From the [official subagent guide](https://code.claude.com/docs/en/sub-agents): focused single-purpose agents, "use when" descriptions, and least-privilege tool lists. From [VoltAgent/awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents): review checklists and the research method of objective, source evaluation, cross-checking, and gaps. From [wshobson/agents](https://github.com/wshobson/agents): response-approach sequences, including impact rating and ADRs for architecture and the red-green-refactor cycle for testing. Generic capability catalogs, JSON messaging protocols, and references to agents this project does not have were dropped. Each prompt instead encodes this project's invariants: provenance, honest fake versus real labeling, explicit human selection, offline default tests, privacy, and the `AGENTS.md` style rules. Agents inherit the session model, and none uses persistent agent memory, so durable knowledge stays in `docs/`.

Personal settings stay in the ignored `.claude/settings.local.json`. Check commands in shared docs now name the project interpreter per OS.

Alternatives considered: duplicating `AGENTS.md` content into `CLAUDE.md`, rejected because the copies would drift, and a symlink, rejected because the publication guard rejects symlinks and Windows symlinks need extra privileges.

## Consequences

Claude Code sessions start with the working agreement, current status, and specification already in context. Prompts are limited to publishing and history-integrating git operations. The deny list is a guardrail against obvious accidents, not a sandbox. Pattern rules match command text, so wrapped or indirect forms such as `git -C . push`, a script that deletes files, or `python -c` are not caught, and shell commands can write outside the repository. Allowed operations such as `git reset --hard`, `git clean`, and package installs run without prompting, so agent instructions still require restraint. Read-deny rules reduce accidental exposure of private material but are not a privacy boundary, so `docs/PRIVACY.md` still applies. Enabling the Claude Code sandbox would confine shell writes to the repository if stronger isolation is needed.
