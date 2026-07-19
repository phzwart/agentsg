# Ralph loop — quick start

Autonomous code-quality backlog runner for `agentsg`.

## Prerequisites

1. Git repo at `/Users/phzwart/Projects/agentsg` (parent of this package).
2. `cursor-agent` on PATH (`~/.local/bin` after `curl https://cursor.com/install -fsS | bash`).
3. **Authenticated:** `cursor-agent login` (once). Without login, use the manual
   protocol in `PROMPT.md` or set `CURSOR_API_KEY`.
4. Package venv with test deps: `pip install -e ".[test]"` inside `agentsg/`.

## Commands

```bash
cd agentsg   # this package directory

bash ralph/run.sh --dry     # show next open task (no agent)
bash ralph/verify.sh        # compile + full pytest gate
MAX_ITERS=40 bash ralph/run.sh   # autonomous loop
```

Stop conditions: `RALPH-DONE` (backlog empty), `RALPH-BLOCKED` (see `BLOCKED.md`),
or `RALPH-CAP` (hit `MAX_ITERS`).

## Files

| file | role |
|------|------|
| `PROMPT.md` | Fixed per-iteration instruction fed to cursor-agent |
| `BACKLOG.md` | Ordered `[ ]` / `[x]` tasks (durable state) |
| `INVARIANTS.md` | Never-violate guardrails |
| `verify.sh` | Targeted then full pytest + compile gate |
| `run.sh` | While-loop driver |
| `BLOCKED.md` | Written on halt (clear to resume) |
| `logs/` | Per-iteration agent transcripts |
