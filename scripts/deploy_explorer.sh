#!/usr/bin/env bash
# Build the static cell plane and publish it to Cloudflare Pages.
# Reads CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" || -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
  echo "Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID." >&2
  exit 1
fi
export PYTHONPATH="agentsg/src${PYTHONPATH:+:$PYTHONPATH}"
python -m agentsg.explorer.build_static --out site/ --k 25
npx --yes wrangler pages deploy site --project-name cell-embed --commit-dirty=true
