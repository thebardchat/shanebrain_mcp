# ShaneBrain MCP Server — Claude Code Guide

## What this repo is

Production MCP server for shanebrain-1 (Pi 5). 37 tools, 16 groups.
Runs as Docker container on port 8100 with `--network host`.

**Live server:** `/mnt/shanebrain-raid/shanebrain-core/mcp-server/`
**This repo:** `~/shanebrain_mcp/` — keep in sync with live server.

## Key files

| File | Purpose |
|------|---------|
| `shanebrain_mcp.py` | All 37 tools — mirror of live `server.py` |
| `weaviate_bridge.py` | DockerWeaviateHelper subclass |
| `health.py` | Weaviate / Ollama / Gateway health checks |
| `scripts/weaviate_helpers.py` | Base WeaviateHelper (shared with shanebrain-core) |
| `docker-compose.yml` | Full deployment spec — authoritative |
| `.env.example` | All env vars with defaults |

## After any code change

1. Mirror the change to the live server:
   ```
   cp shanebrain_mcp.py /mnt/shanebrain-raid/shanebrain-core/mcp-server/server.py
   ```
2. Rebuild and restart the container:
   ```
   cd /mnt/shanebrain-raid/shanebrain-core
   docker build -t shanebrain-mcp -f mcp-server/Dockerfile .
   docker stop shanebrain-mcp && docker rm shanebrain-mcp
   # Use env_from_vault to get secrets, then docker run per docker-compose.yml
   ```
3. Verify: `curl http://localhost:8100/health`
4. Run smoke tests: `python3 test_smoke.py`

## Adding a tool

1. Add `class XxxInput(BaseModel)` with Pydantic v2 fields
2. Add `@mcp.tool(name="shanebrain_xxx", annotations={...})` + function
3. Update the tool count in the docstring at the top of `shanebrain_mcp.py`
4. Add an entry to `TOOLS.md`
5. Add a test case to `test_smoke.py`

## Environment variables

All defined in `.env.example`. Required: `GMAIL_APP_PASSWORD`.
Secrets are stored in Weaviate PersonalDoc (category `credentials`).
Rebuild env file: `python3 scripts/env_from_vault.py`

## Infra cutover (Phase 3)

`.claude/infra.env` holds the variables that flip when the active Weaviate
host changes (e.g. Pi 5 → neworleans). Scaffold lives at
`.claude/infra.env.example` (committed). The actual `.claude/infra.env` is
gitignored — per-host values only.

Source before running scripts:
```
set -a; source .claude/infra.env; set +a
```

Cutover = edit `.claude/infra.env` and re-source. No code changes.
Variables that are host-invariant (MCP_PORT, GCAL_*) stay in
`.env` / `docker-compose.yml`.

## Google Calendar setup

First-time auth:
```
cd /mnt/shanebrain-raid/shanebrain-core/scripts
python3 google_calendar_setup.py   # follow prompts → generates gcal_token.json
```
Token is mounted at `/app/gcal_token.json` (rw so refreshes persist).

## Common pitfalls

- **stdout pollution** — MCP uses stdout. All debug output must go to stderr.
  Use `logging` (already configured) or `print(..., file=sys.stderr)`.
- **Planning dir not mounted** — If container starts without the planning volume,
  plans write to ephemeral overlay FS and are lost on restart. Check docker-compose.yml.
- **gcal_token.json read-only** — Production mounts it rw now. Token refresh
  will log a warning but still work in-memory if somehow mounted ro again.
- **Weaviate GRPC** — `weaviate.connect_to_custom()` uses both HTTP (8080) and
  gRPC (50051). Both ports must be reachable from inside the container.
