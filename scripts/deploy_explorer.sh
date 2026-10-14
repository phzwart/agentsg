#!/usr/bin/env bash
# Build the component explorer and publish it to Cloudflare Pages.
# Reads CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" || -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
  echo "Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID." >&2
  exit 1
fi
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "python3 not found." >&2
  exit 1
fi
export PYTHONPATH="agentsg/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" linear_embed/build_explorer_site.py --out site/
npx --yes wrangler pages deploy site --project-name cell-embed --commit-dirty=true
