#!/usr/bin/env bash
# ralph/verify.sh — gate every Ralph iteration.
# Usage (from agentsg/ package dir):
#   bash ralph/verify.sh [optional targeted test paths...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
  PY="$ROOT/../.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

echo "==> using interpreter: $PY"

REPO_ROOT="$(cd "$ROOT/.." && pwd)"
FILES=()
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if [[ -f "$REPO_ROOT/$f" ]]; then
      FILES+=("$REPO_ROOT/$f")
    elif [[ -f "$ROOT/$f" ]]; then
      FILES+=("$ROOT/$f")
    fi
  done < <(
    {
      git -C "$REPO_ROOT" diff --name-only --diff-filter=ACMR HEAD -- 'agentsg/src' 'agentsg/tests' 2>/dev/null || true
      git -C "$REPO_ROOT" status --porcelain -- 'agentsg/src' 'agentsg/tests' 2>/dev/null | awk '{print $NF}' || true
    } | sort -u
  )
fi

if ((${#FILES[@]})); then
  echo "==> py_compile ${#FILES[@]} touched file(s)"
  "$PY" -m py_compile "${FILES[@]}"
  if "$PY" -c "import ruff" 2>/dev/null; then
    echo "==> ruff check (touched)"
    "$PY" -m ruff check "${FILES[@]}" || true
  elif "$PY" -c "import pyflakes" 2>/dev/null; then
    echo "==> pyflakes (touched)"
    "$PY" -m pyflakes "${FILES[@]}" || true
  fi
else
  echo "==> no touched Python files detected for compile gate (ok)"
fi

if (($#)); then
  echo "==> targeted pytest: $*"
  "$PY" -m pytest -q --tb=short "$@"
fi

echo "==> full pytest -q"
"$PY" -m pytest -q --tb=line
echo "==> verify.sh OK"
