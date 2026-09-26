# Run agentsg for Muse (`sg-muse.mxagents.org`)

1. Create `agentsg/serve.local.env` (not committed):

   ```
   AGENTSG_TOKEN=<random hex>
   AGENTSG_DB=/Users/phzwart/Projects/agentsg/data/pdb_cells.duckdb
   ```

2. Start the API (binds localhost only):

   ```
   cd /Users/phzwart/Projects/agentsg/agentsg
   set -a && source serve.local.env && set +a
   PYTHONPATH=src python -m agentsg.serve --db "$AGENTSG_DB" --host 127.0.0.1 --port 9876 --token "$AGENTSG_TOKEN"
   ```

3. Named Cloudflare tunnel (once):

   ```
   brew install cloudflare/cloudflare/cloudflared
   cloudflared tunnel login          # pick mxagents.org
   cloudflared tunnel create agentsg
   cloudflared tunnel route dns agentsg sg-muse.mxagents.org
   ```

   Config `~/.cloudflared/config.yml`:

   ```yaml
   tunnel: agentsg
   credentials-file: /Users/YOU/.cloudflared/<TUNNEL-UUID>.json
   ingress:
     - hostname: sg-muse.mxagents.org
       service: http://127.0.0.1:9876
     - service: http_status:404
   ```

   Then: `cloudflared tunnel run agentsg`

4. Phone-check `https://sg-muse.mxagents.org/health` once Universal SSL is issued
   (new zones often serve HTTP immediately and HTTPS a few minutes later).
   Then paste the prompt from `GET /docs/muse.md`.
   If Muse is already connected but answers space groups from memory,
   paste the **Already connected** block from that page so it re-reads
   `/skill.md` and starts calling `GET /v1/space-group?sg=…`.

Optional keep-alive (after filling the token in the serve plist):

```
launchctl load ~/Library/LaunchAgents/org.mxagents.serve.plist
launchctl load ~/Library/LaunchAgents/com.cloudflare.agentsg-tunnel.plist
```

Copies of those plists live in `agentsg/deploy/`.
