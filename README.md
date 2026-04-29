<div align="center"><img src=".github/assets/banner.png" alt="ShaneBrain MCP" width="100%"></div>

> **Try Claude free for 2 weeks** — the AI powering this ecosystem. [Start your free trial →](https://claude.ai/referral/4fAMYN9Ing)

![social card](assets/social-card.jpg)

[![Constitution](https://img.shields.io/badge/Constitution-ShaneTheBrain-blue)](https://github.com/thebardchat/constitution)

# ShaneBrain MCP Server v2.5

> **Try Claude free for 2 weeks** — the AI behind this entire ecosystem. [Start your free trial →](https://claude.ai/referral/4fAMYN9Ing)

---



> Production-grade Model Context Protocol server for the ShaneBrain Pi 5 stack.
> 37 tools across 16 groups. FastMCP + Weaviate RAG + Ollama + Google Calendar + Planning System.

---

## Stack

- **Raspberry Pi 5** (16 GB) running all services
- **Weaviate** vector database (17 collections, nomic-embed-text 768-dim)
- **Ollama** local LLM inference (shanebrain-3b, llama3.2:3b)
- **Planning System** markdown-based project management on disk

---

## Quick Start

### Docker (recommended)

The server runs as the `shanebrain-mcp` container via docker-compose:

```bash
docker compose up -d shanebrain-mcp
```

### Direct / systemd

```bash
pip install "mcp[fastmcp]" httpx pydantic weaviate-client ollama --break-system-packages

# Default: streamable-http on port 8100
python3 shanebrain_mcp.py

# Switch transport
python3 shanebrain_mcp.py --transport sse
python3 shanebrain_mcp.py --transport stdio
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEAVIATE_HOST` | `localhost` | Weaviate hostname |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `shanebrain-3b:latest` | Default model for generation |
| `PLANNING_DIR` | `/mnt/shanebrain-raid/shanebrain-core/planning-system` | Planning files root |
| `MCP_PORT` | `8100` | Server listen port |
| `GMAIL_APP_PASSWORD` | *(required for email)* | Gmail App Password for SMTP send/reply |
| `GATEWAY_HOST` | `http://host.docker.internal:4200` | Angel Cloud Gateway base URL |
| `GCAL_TOKEN_PATH` | `/app/gcal_token.json` | Path to Google Calendar OAuth token file |
| `GCAL_CALENDAR_ID` | `primary` | Default Google Calendar ID |

---

## Docker Run

The live container runs with `--network host` so it can reach Weaviate (8080),
Ollama (11434), and the Angel Cloud Gateway (4200) on the Pi's localhost:

```bash
docker build -t shanebrain-mcp .
docker run -d \
  --name shanebrain-mcp \
  --network host \
  -e GMAIL_APP_PASSWORD="your-app-password-here" \
  -v /path/to/gcal_token.json:/app/gcal_token.json:ro \
  shanebrain-mcp
```

The server binds to `0.0.0.0:8100` inside the container. With `--network host`
that exposes it on the Pi's LAN/Tailscale interface directly.

---

## Claude Code Integration

```bash
claude mcp add --scope user shanebrain --transport http http://localhost:8100/mcp
```

Or add to `~/.claude/mcp_servers.json`:

```json
{
  "shanebrain": {
    "type": "streamable_http",
    "url": "http://100.67.120.6:8100/mcp"
  }
}
```

---

## Tools Reference (37 tools, 16 groups)

### Group 1 -- Knowledge (2)

| Tool | Description |
|------|-------------|
| `shanebrain_search_knowledge` | Semantic search across LegacyKnowledge (RAG, wisdom, BGKPJR) |
| `shanebrain_add_knowledge` | Store new knowledge with category and source |

### Group 2 -- Chat (3)

| Tool | Description |
|------|-------------|
| `shanebrain_search_conversations` | Semantic search across conversation history |
| `shanebrain_log_conversation` | Log a message to the Conversation collection |
| `shanebrain_get_conversation_history` | Retrieve conversation by session ID |

### Group 3 -- RAG Chat (1)

| Tool | Description |
|------|-------------|
| `shanebrain_chat` | Full RAG pipeline: search knowledge + Ollama generate |

### Group 4 -- Social (2)

| Tool | Description |
|------|-------------|
| `shanebrain_search_friends` | Semantic search across FriendProfile |
| `shanebrain_get_top_friends` | List friends sorted by interaction count |

### Group 5 -- Vault (3)

| Tool | Description |
|------|-------------|
| `shanebrain_vault_search` | Search PersonalDoc (medical, legal, financial, personal, work) |
| `shanebrain_vault_add` | Store a new vault document with category and tags |
| `shanebrain_vault_list_categories` | List all vault categories with document counts |

### Group 6 -- Notes (3)

| Tool | Description |
|------|-------------|
| `shanebrain_daily_note_add` | Add a journal entry, todo, reminder, or reflection |
| `shanebrain_daily_note_search` | Semantic search across DailyNote |
| `shanebrain_daily_briefing` | Generate today's briefing from recent notes via Ollama |

### Group 7 -- Drafts (2)

| Tool | Description |
|------|-------------|
| `shanebrain_draft_create` | AI-generate a draft (email, post, letter, etc.) with vault context |
| `shanebrain_draft_search` | Search saved drafts by topic or type |

### Group 8 -- Security (3)

| Tool | Description |
|------|-------------|
| `shanebrain_security_log_search` | Semantic search across SecurityLog |
| `shanebrain_security_log_recent` | Chronological recent security events, filterable by severity |
| `shanebrain_privacy_audit_search` | Search PrivacyAudit records |

### Group 9 -- Weaviate Admin (2)

| Tool | Description |
|------|-------------|
| `shanebrain_rag_list_classes` | List all Weaviate collections with object counts |
| `shanebrain_rag_delete` | Delete a Weaviate object by collection + UUID |

### Group 10 -- Ollama (2)

| Tool | Description |
|------|-------------|
| `shanebrain_ollama_generate` | Prompt the local LLM directly with optional system prompt |
| `shanebrain_ollama_list_models` | List downloaded Ollama models with sizes |

### Group 11 -- Planning (3)

| Tool | Description |
|------|-------------|
| `shanebrain_plan_list` | List markdown files in a planning subfolder |
| `shanebrain_plan_read` | Read a planning file's full content |
| `shanebrain_plan_write` | Create or append to a planning file (path-traversal protected) |

### Group 12 -- System (1)

| Tool | Description |
|------|-------------|
| `shanebrain_system_health` | Ping Weaviate, Ollama, and Gateway; return latency dashboard |

### Group 13 -- Email (2)

| Tool | Description |
|------|-------------|
| `shanebrain_send_email` | Send an email from Shane's Gmail via SMTP (requires GMAIL_APP_PASSWORD env) |
| `shanebrain_reply_email` | Reply to an email with proper thread headers (In-Reply-To, References) |

### Group 14 -- Google Calendar (5)

| Tool | Description |
|------|-------------|
| `shanebrain_calendar_list` | List upcoming Google Calendar events for the next N days |
| `shanebrain_calendar_get` | Get a specific event by Google Calendar event ID |
| `shanebrain_calendar_create` | Create a new calendar event with auto-duration (1-hour default) |
| `shanebrain_calendar_update` | Patch an existing event — only provided fields are changed |
| `shanebrain_calendar_delete` | Permanently delete an event by ID |

**Calendar setup:** Requires `gcal_token.json` at `GCAL_TOKEN_PATH` (default `/app/gcal_token.json`).
Run `scripts/google_calendar_setup.py` to authenticate and generate the token.

### Group 15 -- Context Snapshot (1)

| Tool | Description |
|------|-------------|
| `shanebrain_context_snapshot` | Rich session-start snapshot — sobriety, mood, last 3 sessions, active projects, family context |

### Group 16 -- Weaviate Session Tools (2)

Persist and retrieve full Claude session transcripts so memory survives context resets.

| Tool | Description |
|------|-------------|
| `weaviate_log_conversation` | Store a full session transcript in `Conversation` (timestamp, source, 200-char summary) |
| `weaviate_get_context` | Return 5 most recent transcripts + context snapshot, formatted for CLAUDE.md injection |

---

## Transport Options

| Flag | Protocol | Use case |
|------|----------|----------|
| `--transport streamable_http` | Streamable HTTP (default) | Docker, remote clients |
| `--transport sse` | Server-Sent Events | Legacy MCP clients |
| `--transport stdio` | Standard I/O | Direct subprocess integration |

The server listens on `0.0.0.0:8100` by default. The `/mcp` endpoint handles MCP
JSON-RPC, and `/health` returns service status (HTTP 200 healthy, 503 degraded).

---

## Quality

- Pydantic v2 `BaseModel` with `Field` constraints on every tool input
- MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) on all 42 tools
- Actionable error messages with next-step suggestions
- Logging to stderr only (never pollutes MCP stdout)
- Async throughout, no blocking I/O
- Lifespan manager validates Weaviate + Ollama connectivity at startup
- Path traversal protection on filesystem tools (Planning)
- `shanebrain_` prefix on all tool names to prevent conflicts

---

## Smoke Test

```bash
# Server must be running on localhost:8100
python3 test_smoke.py

# Custom URL
python3 test_smoke.py http://192.168.1.50:8100
```

The smoke test initializes an MCP session, calls each tool group, and reports
pass/fail counts. Slow tools (RAG chat, briefing, draft generation) are skipped.

---

## Files

| File | Purpose |
|------|---------|
| `shanebrain_mcp.py` | Main server (all 37 tools + health endpoint + lifespan) |
| `weaviate_bridge.py` | DockerWeaviateHelper — Weaviate connection via Docker host network |
| `health.py` | Health check functions for Weaviate, Ollama, and Gateway |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build (standalone; see note about weaviate_helpers.py) |
| `test_smoke.py` | Smoke test suite for all tool groups |

**Note:** `weaviate_bridge.py` imports `scripts.weaviate_helpers.WeaviateHelper` which
lives in the shanebrain-core monorepo at `scripts/weaviate_helpers.py`. The production
Dockerfile copies it from there. For standalone deployment, copy it manually into a
`scripts/` subdirectory alongside `weaviate_bridge.py`.


## Built With

<table>
  <tr>
    <td align="center" width="200">
      <b>Claude by Anthropic</b><br/>
      <sub>AI partner and co-builder.</sub><br/><br/>
      <a href="https://claude.ai"><code>claude.ai</code></a>
    </td>
    <td align="center" width="200">
      <b>Raspberry Pi 5</b><br/>
      <sub>Local AI compute node.</sub><br/><br/>
      <a href="https://www.raspberrypi.com"><code>raspberrypi.com</code></a>
    </td>
    <td align="center" width="200">
      <b>Pironman 5-MAX</b><br/>
      <sub>NVMe RAID 1 chassis by Sunfounder.</sub><br/><br/>
      <a href="https://www.sunfounder.com"><code>sunfounder.com</code></a>
    </td>
  </tr>
</table>

---

## Support This Work

If what I'm building matters to you — local AI for real people, tools for the left-behind — here's how to help:

- **[Sponsor me on GitHub](https://github.com/sponsors/thebardchat)**
- **[Buy the book](https://www.amazon.com/Probably-Think-This-Book-About/dp/B0GT25R5FD)** — *You Probably Think This Book Is About You*
- **Star the repos** — visibility matters for projects like this

Built by **Shane Brazelton** · Co-built with **Claude** (Anthropic) · Hazel Green, Alabama

---

<div align="center">

*Part of the [ShaneBrain Ecosystem](https://github.com/thebardchat) · Built under the [Constitution](https://github.com/thebardchat/constitution)*

</div>
