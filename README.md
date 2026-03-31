<div align="center"><img src=".github/assets/banner.png" alt="ShaneBrain MCP" width="100%"></div>

[![Constitution](https://img.shields.io/badge/Constitution-ShaneTheBrain-blue)](https://github.com/thebardchat/constitution)

# ShaneBrain MCP Server v2.0

> Production-grade Model Context Protocol server for the ShaneBrain Pi 5 stack.
> 42 tools across 12 groups. FastMCP + Weaviate RAG + Ollama + Planning System.

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

## Tools Reference (42 tools, 12 groups)

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
| `shanebrain_mcp.py` | Main server (all 42 tools + health endpoint + lifespan) |
| `test_smoke.py` | Smoke test suite for all tool groups |

Note: The production deployment also uses `weaviate_bridge.py` and `health.py`
as local imports. Those modules live in the Pi deployment at
`/mnt/shanebrain-raid/shanebrain-core/mcp-server/`.


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
