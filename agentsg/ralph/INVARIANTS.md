# Ralph loop invariants (never violate)

These guardrails apply to every iteration. If a task cannot be completed without
breaking one of them, stop, revert, and write `BLOCKED.md`.

## Test suite

- The full suite must stay green. Baseline: **3255 passed**.
- Never delete, skip, or weaken an existing test to make the suite pass.
- New tests for a task are allowed and encouraged when the backlog asks for them.
- Run verification via `bash ralph/verify.sh` from the `agentsg/` package directory
  (the directory that contains `pyproject.toml` and `ralph/`).

## Exact / numeric boundary

- Preserve the package split: exact-rational symmetry lives in `agentsg.*`;
  numeric cell math lives in `agentsg.cell.*`.
- The metric bridge is `agentsg.cell.constraints` (`W^T G W = G`). Other cell
  modules may import space-group symbols/operators when a workflow needs them
  (ambiguity, Selling, primitive, PDB); do not invent new *metric* crossings.
- Do not introduce new runtime dependencies.

## Dependencies

- No new **runtime** dependencies. `[project].dependencies` in `pyproject.toml`
  must remain `[]`.
- Test-only extras (`pytest`, `gemmi`, `spglib`) and the optional `db` extra
  (`duckdb`) may stay as they are; do not add new ones unless a backlog task
  explicitly requires a test oracle already listed.

## API / behavior

- Behavior-preserving refactors must keep public results identical (same return
  values, same exceptions for valid inputs).
- Deprecations and back-compat aliases must remain importable for at least one
  release cycle (do not break existing `from agentsg... import _foo` call sites
  the backlog says to keep).
- Do not expand scope into MCP/skill packaging (R1–R6). That is out of scope.

## Lint / hygiene

- After edits, ensure touched Python files have no new syntax errors.
- Prefer `python -m py_compile` on touched modules; if `ruff` or `pyflakes` is
  available in the environment, run it on touched files and leave no new issues.
- Do not leave `print` / debug scaffolding in library code.

## Git

- Commit only after `verify.sh` exits 0.
- One backlog task → one commit, message form: `ralph: <task-id> <short summary>`.
- On failure after retries: `git checkout -- .` (and `git clean -fd` only for
  untracked files you created this iteration), then append to `BLOCKED.md`.
- Working tree for commits is the **git repo root** (parent of this package;
  `git rev-parse --show-toplevel`). Package work and `pytest` run from `agentsg/`.
