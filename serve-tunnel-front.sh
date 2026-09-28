#!/usr/bin/env bash
# Single origin for the Cloudflare tunnel. cloudflared --url can point at
# only one local port; this router sends sg-mcp.mxagents.org to the MCP
# server and every other hostname (sg-muse) to the HTTP API.
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

export AGENTSG_FRONT_HOST="${AGENTSG_FRONT_HOST:-127.0.0.1}"
export AGENTSG_FRONT_PORT="${AGENTSG_FRONT_PORT:-9880}"
export AGENTSG_MCP_PORT="${AGENTSG_MCP_PORT:-9877}"
export AGENTSG_PORT="${AGENTSG_PORT:-9876}"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "python3 not found." >&2
  exit 1
fi

echo "agentsg tunnel front on http://${AGENTSG_FRONT_HOST}:${AGENTSG_FRONT_PORT}" >&2
exec "$PYTHON" -m agentsg.serve.front
