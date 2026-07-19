#!/usr/bin/env bash
# ralph/run.sh — Ralph loop: feed PROMPT.md to cursor-agent until done/blocked/cap.
#
# Usage (from agentsg/ package directory):
#   bash ralph/run.sh              # autonomous loop
#   bash ralph/run.sh --dry        # print first unchecked task; do not invoke agent
#   MAX_ITERS=5 bash ralph/run.sh  # custom iteration cap (default 40)
#
# Requires: cursor-agent (or agent) on PATH; git repo at parent of this package.
set -euo pipefail

RALPH_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(cd "$RALPH_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
MAX_ITERS="${MAX_ITERS:-40}"
PROMPT_FILE="$RALPH_DIR/PROMPT.md"
BACKLOG_FILE="$RALPH_DIR/BACKLOG.md"
LOG_DIR="$RALPH_DIR/logs"
mkdir -p "$LOG_DIR"

export PATH="${HOME}/.local/bin:${PATH}"

if command -v cursor-agent >/dev/null 2>&1; then
  AGENT=(cursor-agent)
elif command -v agent >/dev/null 2>&1; then
  AGENT=(agent)
else
  echo "error: cursor-agent / agent not found on PATH" >&2
  exit 1
fi

first_open_task() {
  # Print first unchecked task id (e.g. A1) or empty if none.
  # Lines look like: ### [ ] A1 — Title...
  sed -n 's/^### \[ \] \([A-Z][0-9][0-9]*\).*/\1/p' "$BACKLOG_FILE" | head -1
}

if [[ "${1:-}" == "--dry" ]]; then
  id="$(first_open_task || true)"
  if [[ -z "${id:-}" ]]; then
    echo "RALPH-DONE (no open tasks)"
    exit 0
  fi
  echo "DRY: would run task $id"
  echo "agent: ${AGENT[*]}"
  echo "workspace: $REPO_ROOT"
  echo "prompt: $PROMPT_FILE"
  exit 0
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "error: missing $PROMPT_FILE" >&2
  exit 1
fi

cd "$PKG_DIR"

PROMPT_TEXT="$(cat "$PROMPT_FILE")"
# Append a short pointer so the agent always knows cwd / backlog path.
PROMPT_TEXT+=$'\n\n## Paths for this run\n'
PROMPT_TEXT+=$'- Package cwd should be: '"$PKG_DIR"$'\n'
PROMPT_TEXT+=$'- Git root: '"$REPO_ROOT"$'\n'
PROMPT_TEXT+=$'- Backlog: ralph/BACKLOG.md\n'
PROMPT_TEXT+=$'- Invariants: ralph/INVARIANTS.md\n'

for ((i = 1; i <= MAX_ITERS; i++)); do
  if [[ -f "$RALPH_DIR/BLOCKED.md" ]] && [[ -s "$RALPH_DIR/BLOCKED.md" ]]; then
    echo "RALPH-BLOCKED: ralph/BLOCKED.md is non-empty; clear it to continue."
    exit 2
  fi

  open_id="$(first_open_task || true)"
  if [[ -z "${open_id:-}" ]]; then
    echo "RALPH-DONE: backlog empty"
    exit 0
  fi

  stamp="$(date +%Y%m%d-%H%M%S)"
  log="$LOG_DIR/iter-${i}-${open_id}-${stamp}.log"
  echo "==> iteration $i/$MAX_ITERS  task=$open_id  log=$log"

  set +e
  "${AGENT[@]}" -p --force --trust \
    --workspace "$REPO_ROOT" \
    --output-format text \
    "$PROMPT_TEXT" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  if grep -q '^RALPH-DONE$' "$log" 2>/dev/null || grep -q 'RALPH-DONE' "$log" 2>/dev/null; then
    # Agent may finish the last task and print DONE, or find empty backlog.
    if [[ -z "$(first_open_task || true)" ]]; then
      echo "RALPH-DONE"
      exit 0
    fi
  fi

  if grep -q 'RALPH-BLOCKED' "$log" 2>/dev/null; then
    echo "RALPH-BLOCKED (see ralph/BLOCKED.md and $log)"
    exit 2
  fi

  if [[ $rc -ne 0 ]]; then
    echo "error: agent exited $rc (see $log)" >&2
    exit $rc
  fi

  # Progress check: the previously open task should now be checked, or backlog empty.
  if grep -q "^### \\[ \\] ${open_id} " "$BACKLOG_FILE" 2>/dev/null; then
    echo "warning: task $open_id still open after iteration $i — continuing carefully" >&2
  else
    echo "==> task $open_id closed (or renamed); continuing"
  fi
done

echo "RALPH-CAP: hit MAX_ITERS=$MAX_ITERS with open tasks remaining"
exit 3
