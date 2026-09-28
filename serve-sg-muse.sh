#!/usr/bin/env bash
# Launch the sg-muse agentsg API: 127.0.0.1:9876, public URL
# https://sg-muse.mxagents.org. Secrets stay in agentsg/serve.local.env
# (gitignored). Point cloudflared at the front door (./serve-tunnel-front.sh
# on 127.0.0.1:9880), which forwards this host here and sg-mcp to FastMCP.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_FILE="$ROOT/agentsg/serve.local.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  set +a
fi

export AGENTSG_DB="${AGENTSG_DB:-$ROOT/data/pdb_cells.duckdb}"
export AGENTSG_HOST="${AGENTSG_HOST:-127.0.0.1}"
export AGENTSG_PORT="${AGENTSG_PORT:-9876}"
export AGENTSG_PUBLIC_URL="${AGENTSG_PUBLIC_URL:-https://sg-muse.mxagents.org}"
export AGENTSG_API_NAME="${AGENTSG_API_NAME:-sg-muse}"

if [[ ! -f "$AGENTSG_DB" ]]; then
  echo "PDB database not found: $AGENTSG_DB" >&2
  echo "Download it as described in AGENTS.md." >&2
  exit 1
fi

if [[ -z "${AGENTSG_TOKEN:-}" ]]; then
  echo "AGENTSG_TOKEN is not set. Add it to agentsg/serve.local.env (not committed)." >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "python3 not found. Create .venv and pip install -e '.[db,plot]'." >&2
  exit 1
fi

echo "agentsg sg-muse on http://${AGENTSG_HOST}:${AGENTSG_PORT} (${AGENTSG_API_NAME}, ${AGENTSG_PUBLIC_URL})" >&2
exec "$PYTHON" -m agentsg.serve
