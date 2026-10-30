#!/usr/bin/env bash
# Publish the MX Agents landing page to Cloudflare Pages.
# Reads CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" || -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
  echo "Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID." >&2
  exit 1
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -R landing/. "$stage/"
cp agentsg/src/agentsg/serve/index.html "$stage/index.html"

npx --yes wrangler pages project create mxagents --production-branch main \
  || true
npx --yes wrangler pages deploy "$stage" --project-name mxagents --commit-dirty=true

for name in mxagents.org www.mxagents.org; do
  echo "registering ${name}"
  curl -sS -X POST \
    "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/pages/projects/mxagents/domains" \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"name\":\"${name}\"}"
  echo
done

cat <<'EOF'

Uploaded. https://mxagents.pages.dev is the new page.

In DNS for mxagents.org, edit only these two rows. Replace the target that
ends in cfargotunnel.com with:

    mxagents.pages.dev

  name www          (the www.mxagents.org row)
  name @            (the mxagents.org row, sometimes shown as mxagents.org)

Leave the type as CNAME and leave the orange cloud on.
Do not edit sg-muse or sg-mcp.
EOF
