# ShaneBrain MCP Server v2.0

> Production-grade Model Context Protocol server for the ShaneBrain Pi 5 stack.  
> FastMCP · Streamable HTTP · Weaviate RAG · Ollama · Planning System · Health Monitor

---

## 🚀 Quick Deploy

```bash
# 1. Copy server to Pi
scp shanebrain_mcp.py shane@100.67.120.6:/mnt/shanebrain-raid/shanebrain-core/

# 2. Install/update dependencies
pip install "mcp[fastmcp]" httpx pydantic --break-system-packages

# 3. Test run (check stderr for startup logs)
python3 shanebrain_mcp.py --transport streamable_http --port 8008

# 4. Install as systemd service
sudo cp shanebrain-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable shanebrain-mcp
sudo systemctl start shanebrain-mcp

# 5. Check status
sudo systemctl status shanebrain-mcp
journalctl -u shanebrain-mcp -f
```

---

## 🔧 Environment Variables (.env)

```env
WEAVIATE_URL=http://localhost:8080
OLLAMA_URL=http://localhost:11434
MONGO_URL=mongodb://localhost:27017
PLANNING_DIR=/mnt/shanebrain-raid/shanebrain-core/planning-system
OLLAMA_DEFAULT_MODEL=llama3.2:1b
MCP_PORT=8008
```

---

## 🛠️ Tools Reference

### GROUP 1 — Weaviate RAG
| Tool | What it does |
|------|-------------|
| `shanebrain_rag_search` | Semantic nearText search across any Weaviate class |
| `shanebrain_rag_store` | Vectorize and store a document/chunk |
| `shanebrain_rag_delete` | Permanently delete an object by UUID |
| `shanebrain_rag_list_classes` | Show all schema classes and property counts |
| `shanebrain_rag_answer` | **Full RAG pipeline**: search → inject → Ollama generate |

### GROUP 2 — Ollama Local Inference
| Tool | What it does |
|------|-------------|
| `shanebrain_ollama_generate` | Prompt the local LLM directly |
| `shanebrain_ollama_list_models` | List downloaded models with sizes |

### GROUP 3 — Planning System
| Tool | What it does |
|------|-------------|
| `shanebrain_plan_list` | List .md files in a planning subfolder |
| `shanebrain_plan_read` | Read a planning file's full content |
| `shanebrain_plan_write` | Create or append to a planning file |

### GROUP 4 — System Health
| Tool | What it does |
|------|-------------|
| `shanebrain_system_health` | Ping all services, return latency dashboard |

---

## 🔌 Claude Code / claude.ai Config

Add to your `~/.claude/mcp_servers.json` on Pulsar0100:

```json
{
  "shanebrain": {
    "type": "streamable_http",
    "url": "http://100.67.120.6:8008/mcp"
  }
}
```

Or via Tailscale hostname:
```json
{
  "shanebrain": {
    "type": "streamable_http",
    "url": "http://shanebrain-1:8008/mcp"
  }
}
```

---

## ✅ Quality Checklist (vs MCP Best Practices)

- [x] Tool names use `shanebrain_` prefix to prevent conflicts
- [x] All inputs use Pydantic v2 BaseModel with Field constraints
- [x] Annotations set (readOnly, destructive, idempotent, openWorld) on every tool
- [x] Dual format responses (markdown + json) on read tools
- [x] Pagination metadata on list operations
- [x] Actionable error messages — not generic stack traces
- [x] Path traversal protection on filesystem tools
- [x] Lifespan manager validates service connectivity at startup
- [x] Logging to stderr only (never stdout)
- [x] Async throughout — no blocking I/O
- [x] DRY helpers: `_weaviate_get/post`, `_ollama_get/post`, `_format_error`
- [x] Streamable HTTP transport (not deprecated SSE)
- [x] `--host 0.0.0.0` for Tailscale accessibility
- [x] `shanebrain_rag_answer` = full chained RAG pipeline in one call

---

## 🔥 v2.0 vs Previous Server — What's New

| Feature | Old | New |
|---------|-----|-----|
| Transport | SSE (deprecated) | Streamable HTTP ✅ |
| Tool annotations | Missing | All 4 annotations on every tool ✅ |
| Error messages | Generic | Actionable with next steps ✅ |
| Pydantic validation | Loose | v2 with Field constraints ✅ |
| RAG pipeline | Search only | Full RAG answer (search→LLM) ✅ |
| Planning system | Not exposed | 3 tools: list/read/write ✅ |
| Health check | None | Full service dashboard ✅ |
| Startup validation | None | Lifespan checks all services ✅ |
| Response formats | Text only | Markdown + JSON toggle ✅ |
| Path security | None | Traversal attack protection ✅ |
