#!/usr/bin/env bash
# Launch the sg-mcp FastMCP server: 127.0.0.1:9877, public URL
# https://sg-mcp.mxagents.org/mcp. No API key.
# The Cloudflare tunnel should target the front door (serve-tunnel-front.sh),
# which sends this hostname here and leaves sg-muse on the HTTP API.
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
export AGENTSG_ACCESS_LOG="${AGENTSG_ACCESS_LOG:-$ROOT/data/access.jsonl}"
export AGENTSG_MCP_HOST="${AGENTSG_MCP_HOST:-127.0.0.1}"
export AGENTSG_MCP_PORT="${AGENTSG_MCP_PORT:-9877}"
export AGENTSG_MCP_PUBLIC_URL="${AGENTSG_MCP_PUBLIC_URL:-https://sg-mcp.mxagents.org}"
unset AGENTSG_TOKEN

if [[ ! -f "$AGENTSG_DB" ]]; then
  echo "PDB database not found: $AGENTSG_DB" >&2
  echo "Download it as described in AGENTS.md." >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "python3 not found. Create .venv and pip install -e '.[db,plot,mcp]'." >&2
  exit 1
fi

echo "agentsg sg-mcp on http://${AGENTSG_MCP_HOST}:${AGENTSG_MCP_PORT}/mcp (${AGENTSG_MCP_PUBLIC_URL}/mcp, no API key)" >&2
echo "playbook: tool playbook  (numeric gates: concept numeric_gate)" >&2
exec "$PYTHON" -m agentsg.serve.mcp_app
