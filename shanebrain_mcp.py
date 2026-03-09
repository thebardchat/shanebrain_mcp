"""
ShaneBrain Core MCP Server v2.0
================================
Production-grade MCP server exposing Weaviate RAG, Ollama inference,
planning system, MongoDB memory, and system health monitoring.

Stack: FastMCP + Streamable HTTP | Pi 5 @ /mnt/shanebrain-raid/shanebrain-core/
Transport: Streamable HTTP (port 8008) — multi-client, Tailscale-accessible
"""

import json
import os
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, ConfigDict, field_validator

# ---------------------------------------------------------------------------
# Logging — stderr only (never pollute stdout for MCP)
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shanebrain_mcp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEAVIATE_URL   = os.getenv("WEAVIATE_URL",   "http://localhost:8080")
OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
MONGO_URL      = os.getenv("MONGO_URL",      "mongodb://localhost:27017")
PLANNING_DIR   = Path(os.getenv("PLANNING_DIR", "/mnt/shanebrain-raid/shanebrain-core/planning-system"))
DEFAULT_MODEL  = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:1b")
MCP_PORT       = int(os.getenv("MCP_PORT", "8008"))

WEAVIATE_KNOWLEDGE_CLASS = "SocialKnowledge"
WEAVIATE_MEMORY_CLASS    = "ConversationMemory"
WEAVIATE_FRIEND_CLASS    = "FriendProfile"

# ---------------------------------------------------------------------------
# Response format enum (reused across tools)
# ---------------------------------------------------------------------------
class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON     = "json"

# ---------------------------------------------------------------------------
# Shared HTTP client helpers
# ---------------------------------------------------------------------------

async def _weaviate_get(path: str, params: Optional[Dict] = None) -> Dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{WEAVIATE_URL}{path}", params=params)
        r.raise_for_status()
        return r.json()

async def _weaviate_post(path: str, body: Dict) -> Dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{WEAVIATE_URL}{path}",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.json()

async def _weaviate_delete(path: str) -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.delete(f"{WEAVIATE_URL}{path}")
        r.raise_for_status()

async def _ollama_post(path: str, body: Dict) -> Dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{OLLAMA_URL}{path}", json=body)
        r.raise_for_status()
        return r.json()

async def _ollama_get(path: str) -> Dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{OLLAMA_URL}{path}")
        r.raise_for_status()
        return r.json()

# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def _format_error(e: Exception, context: str = "") -> str:
    prefix = f"[{context}] " if context else ""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 404:
            return f"{prefix}Error 404: Resource not found. Check class name or object ID."
        elif code == 422:
            return f"{prefix}Error 422: Invalid request body. Check your schema and field names."
        elif code == 503:
            return f"{prefix}Error 503: Service unavailable. Is the target service running?"
        return f"{prefix}Error {code}: {e.response.text[:200]}"
    elif isinstance(e, httpx.TimeoutException):
        return f"{prefix}Timeout: Request took too long. Large models or heavy RAG queries may need more time."
    elif isinstance(e, httpx.ConnectError):
        return f"{prefix}Connection refused. Verify the service is running: {e}"
    return f"{prefix}Unexpected error ({type(e).__name__}): {str(e)}"

# ---------------------------------------------------------------------------
# Lifespan — validate connectivity at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan():
    logger.info("ShaneBrain MCP server starting...")
    for name, url, path in [
        ("Weaviate", WEAVIATE_URL, "/v1/.well-known/ready"),
        ("Ollama",   OLLAMA_URL,   "/api/version"),
    ]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{url}{path}")
                r.raise_for_status()
            logger.info("✅  %s reachable at %s", name, url)
        except Exception as exc:
            logger.warning("⚠️  %s NOT reachable (%s) — tools will return errors until fixed.", name, exc)

    PLANNING_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("active-projects", "templates", "completed", "logs"):
        (PLANNING_DIR / sub).mkdir(exist_ok=True)
    logger.info("📁  Planning dir ready: %s", PLANNING_DIR)

    yield {}

    logger.info("ShaneBrain MCP server shutting down.")

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("shanebrain_mcp", lifespan=lifespan)

# ===========================================================================
# TOOLS — GROUP 1: Weaviate RAG
# ===========================================================================

class RagSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Natural language search query (e.g., 'Angel Cloud crisis detection logic')",
        min_length=1,
        max_length=500,
    )
    class_name: str = Field(
        default=WEAVIATE_KNOWLEDGE_CLASS,
        description=f"Weaviate class to search. Common: '{WEAVIATE_KNOWLEDGE_CLASS}', '{WEAVIATE_MEMORY_CLASS}', '{WEAVIATE_FRIEND_CLASS}'",
    )
    limit: int = Field(default=5, ge=1, le=50, description="Max results to return (1–50)")
    certainty: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum similarity threshold (0.0–1.0). Higher = stricter match.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="'markdown' for readable output, 'json' for structured data",
    )

@mcp.tool(
    name="shanebrain_rag_search",
    annotations={
        "title": "Search ShaneBrain Knowledge Base (RAG)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def shanebrain_rag_search(params: RagSearchInput, ctx: Context) -> str:
    """Semantic vector search across ShaneBrain's local Weaviate knowledge base.

    Uses nearText to find conceptually similar documents — not keyword matching.
    Ideal for: retrieving past decisions, project context, conversation memory,
    crisis patterns, or any knowledge stored in the ShaneBrain RAG pipeline.

    Args:
        params (RagSearchInput): Validated search parameters including query,
            class_name, limit, certainty threshold, and response_format.

    Returns:
        str: Ranked results with content, certainty scores, and metadata.
             Returns empty result set if no matches exceed certainty threshold.
    """
    await ctx.report_progress(0.1, "Querying Weaviate...")
    try:
        gql = {
            "query": f"""
            {{
              Get {{
                {params.class_name}(
                  nearText: {{ concepts: ["{params.query}"], certainty: {params.certainty} }}
                  limit: {params.limit}
                ) {{
                  _additional {{ id certainty }}
                  content
                  source
                  timestamp
                  tags
                }}
              }}
            }}
            """
        }
        data = await _weaviate_post("/v1/graphql", gql)
        await ctx.report_progress(0.9, "Formatting results...")
        results = data.get("data", {}).get("Get", {}).get(params.class_name, [])

        if not results:
            return f"No results found in '{params.class_name}' for query: '{params.query}' (certainty ≥ {params.certainty}). Try lowering certainty or broadening the query."

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"class": params.class_name, "query": params.query, "results": results}, indent=2)

        lines = [f"## RAG Search: `{params.query}`", f"**Class:** {params.class_name} | **Results:** {len(results)}\n"]
        for i, r in enumerate(results, 1):
            score = r.get("_additional", {}).get("certainty", 0)
            obj_id = r.get("_additional", {}).get("id", "—")
            content = (r.get("content") or "")[:600]
            source = r.get("source") or "unknown"
            ts = r.get("timestamp") or ""
            tags = ", ".join(r.get("tags") or []) or "none"
            lines += [
                f"### Result {i} — Certainty: {score:.2%}",
                f"**ID:** `{obj_id}`  **Source:** {source}  **Date:** {ts}",
                f"**Tags:** {tags}",
                f"```\n{content}\n```\n",
            ]
        return "\n".join(lines)

    except Exception as e:
        return _format_error(e, "shanebrain_rag_search")


class RagStoreInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    content: str = Field(..., description="Text content to store and vectorize", min_length=1, max_length=50000)
    class_name: str = Field(
        default=WEAVIATE_KNOWLEDGE_CLASS,
        description=f"Weaviate class to store into. Common: '{WEAVIATE_KNOWLEDGE_CLASS}', '{WEAVIATE_MEMORY_CLASS}'",
    )
    source: str = Field(default="manual", description="Origin of this knowledge (e.g., 'angel_cloud', 'dispatch', 'conversation')")
    tags: List[str] = Field(default_factory=list, description="Topic tags for filtering (e.g., ['crisis', 'angel-cloud', 'planning'])", max_length=20)
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp. Defaults to current UTC time if omitted.",
    )

@mcp.tool(
    name="shanebrain_rag_store",
    annotations={
        "title": "Store Knowledge in ShaneBrain RAG",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def shanebrain_rag_store(params: RagStoreInput, ctx: Context) -> str:
    """Store a document or knowledge chunk into Weaviate for semantic retrieval.

    Weaviate automatically generates the vector embedding. The stored object
    becomes immediately searchable via shanebrain_rag_search. Use this to
    persist: conversation summaries, project decisions, crisis patterns,
    code snippets, or any knowledge the AI should recall later.

    Args:
        params (RagStoreInput): Content, class, source, tags, and optional timestamp.

    Returns:
        str: Confirmation with the newly created object ID for future reference.
    """
    await ctx.report_progress(0.2, "Storing to Weaviate...")
    try:
        ts = params.timestamp or datetime.now(timezone.utc).isoformat()
        obj = {
            "class": params.class_name,
            "properties": {
                "content":   params.content,
                "source":    params.source,
                "tags":      params.tags,
                "timestamp": ts,
            },
        }
        result = await _weaviate_post("/v1/objects", obj)
        obj_id = result.get("id", "unknown")
        await ctx.report_progress(1.0, "Stored.")
        return (
            f"✅ Stored in `{params.class_name}`\n"
            f"**ID:** `{obj_id}`\n"
            f"**Source:** {params.source}\n"
            f"**Tags:** {', '.join(params.tags) or 'none'}\n"
            f"**Timestamp:** {ts}\n"
            f"**Content preview:** {params.content[:120]}..."
        )
    except Exception as e:
        return _format_error(e, "shanebrain_rag_store")


class RagDeleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str = Field(..., description="Weaviate object UUID to delete (get from search results)")
    class_name: str = Field(..., description="Weaviate class the object belongs to")

@mcp.tool(
    name="shanebrain_rag_delete",
    annotations={
        "title": "Delete Object from Weaviate",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def shanebrain_rag_delete(params: RagDeleteInput) -> str:
    """Permanently delete a specific object from Weaviate by UUID.

    Use with caution — deletion is irreversible. Get the object ID first
    using shanebrain_rag_search and inspecting the _additional.id field.

    Args:
        params (RagDeleteInput): object_id (UUID) and class_name.

    Returns:
        str: Confirmation of deletion or error if not found.
    """
    try:
        await _weaviate_delete(f"/v1/objects/{params.class_name}/{params.object_id}")
        return f"🗑️ Deleted object `{params.object_id}` from class `{params.class_name}`."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Object `{params.object_id}` not found in class `{params.class_name}`. Already deleted?"
        return _format_error(e, "shanebrain_rag_delete")
    except Exception as e:
        return _format_error(e, "shanebrain_rag_delete")


class RagListClassesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

@mcp.tool(
    name="shanebrain_rag_list_classes",
    annotations={
        "title": "List Weaviate Schema Classes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def shanebrain_rag_list_classes(params: RagListClassesInput) -> str:
    """List all classes (schemas) in the local Weaviate instance with object counts.

    Useful for discovering what data is stored, checking schema health, and
    understanding which class_name to use in search/store operations.

    Returns:
        str: Table of class names, property counts, and object counts.
    """
    try:
        schema = await _weaviate_get("/v1/schema")
        classes = schema.get("classes", [])
        if not classes:
            return "No classes found in Weaviate schema. Run your schema setup scripts first."

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(classes, indent=2)

        lines = ["## Weaviate Schema Classes\n", "| Class | Properties | Vectorizer |", "|-------|-----------|------------|"]
        for c in classes:
            name = c.get("class", "?")
            props = len(c.get("properties", []))
            vectorizer = c.get("vectorizer", "none")
            lines.append(f"| `{name}` | {props} | {vectorizer} |")
        return "\n".join(lines)
    except Exception as e:
        return _format_error(e, "shanebrain_rag_list_classes")


# ===========================================================================
# TOOLS — GROUP 2: Ollama Local Inference
# ===========================================================================

class OllamaGenerateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompt: str = Field(..., description="The prompt or question to send to the local model", min_length=1, max_length=8000)
    model: str = Field(
        default=DEFAULT_MODEL,
        description=f"Ollama model name (e.g., '{DEFAULT_MODEL}', 'llama3.2:3b', 'mistral'). Use shanebrain_ollama_list_models to see available.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to set model behavior/persona",
        max_length=2000,
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0,
        description="Sampling temperature. 0.0 = deterministic, 2.0 = very creative.",
    )
    max_tokens: int = Field(
        default=512, ge=1, le=4096,
        description="Max tokens to generate. Keep low on 1b model for speed.",
    )

@mcp.tool(
    name="shanebrain_ollama_generate",
    annotations={
        "title": "Generate Text with Local Ollama Model",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def shanebrain_ollama_generate(params: OllamaGenerateInput, ctx: Context) -> str:
    """Generate text using a locally-running Ollama model. Zero cloud dependency.

    The default model (llama3.2:1b) is optimized for Pi 5 memory constraints.
    For better quality, use llama3.2:3b if RAM allows. For code, use codellama.

    Args:
        params (OllamaGenerateInput): prompt, model, system_prompt, temperature, max_tokens.

    Returns:
        str: Generated text response from the local model with metadata footer.
    """
    await ctx.report_progress(0.1, f"Sending to {params.model}...")
    try:
        body: Dict[str, Any] = {
            "model": params.model,
            "prompt": params.prompt,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "num_predict": params.max_tokens,
            },
        }
        if params.system_prompt:
            body["system"] = params.system_prompt

        await ctx.report_progress(0.4, "Model generating...")
        result = await _ollama_post("/api/generate", body)
        await ctx.report_progress(0.9, "Formatting response...")

        response_text = result.get("response", "").strip()
        eval_count = result.get("eval_count", 0)
        total_duration_s = result.get("total_duration", 0) / 1e9

        footer = (
            f"\n\n---\n*Model: `{params.model}` | "
            f"Tokens: {eval_count} | "
            f"Time: {total_duration_s:.1f}s | "
            f"Local inference — zero cloud cost*"
        )
        return response_text + footer

    except Exception as e:
        return _format_error(e, "shanebrain_ollama_generate")


class OllamaListModelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

@mcp.tool(
    name="shanebrain_ollama_list_models",
    annotations={
        "title": "List Available Ollama Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def shanebrain_ollama_list_models(params: OllamaListModelsInput) -> str:
    """List all Ollama models currently downloaded on this Pi.

    Use this before calling shanebrain_ollama_generate to confirm a model is
    available. Shows model name, size, and last modified date.

    Returns:
        str: Table of available local models with sizes.
    """
    try:
        data = await _ollama_get("/api/tags")
        models = data.get("models", [])
        if not models:
            return "No models found. Pull a model: `ollama pull llama3.2:1b`"

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(models, indent=2)

        lines = ["## Local Ollama Models\n", "| Model | Size | Modified |", "|-------|------|----------|"]
        for m in models:
            name = m.get("name", "?")
            size_gb = m.get("size", 0) / 1e9
            modified = m.get("modified_at", "")[:10]
            lines.append(f"| `{name}` | {size_gb:.1f} GB | {modified} |")
        return "\n".join(lines)
    except Exception as e:
        return _format_error(e, "shanebrain_ollama_list_models")


# ===========================================================================
# TOOLS — GROUP 3: Planning System (Markdown files)
# ===========================================================================

class PlanListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subfolder: str = Field(
        default="active-projects",
        description="Subfolder to list: 'active-projects', 'templates', 'completed', 'logs'",
    )

@mcp.tool(
    name="shanebrain_plan_list",
    annotations={
        "title": "List Planning System Files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def shanebrain_plan_list(params: PlanListInput) -> str:
    """List markdown planning files in the ShaneBrain planning system directory.

    The planning system uses persistent markdown files for multi-session project
    continuity. Use this to discover existing plans before reading or writing.

    Args:
        params (PlanListInput): subfolder to list.

    Returns:
        str: File listing with sizes and modification dates.
    """
    try:
        folder = PLANNING_DIR / params.subfolder
        if not folder.exists():
            return f"Subfolder '{params.subfolder}' does not exist under {PLANNING_DIR}."

        files = sorted(folder.glob("*.md"))
        if not files:
            return f"No markdown files found in '{params.subfolder}'."

        lines = [f"## Planning Files: `{params.subfolder}`\n"]
        for f in files:
            stat = f.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- `{f.name}` — {size_kb:.1f} KB — modified {mtime}")
        return "\n".join(lines)
    except Exception as e:
        return _format_error(e, "shanebrain_plan_list")


class PlanReadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str = Field(..., description="Filename to read (e.g., 'angel-cloud-phase2.md'). Include .md extension.", min_length=1)
    subfolder: str = Field(default="active-projects", description="Subfolder containing the file")

    @field_validator("filename")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Filename must not contain path separators or '..'")
        return v

@mcp.tool(
    name="shanebrain_plan_read",
    annotations={
        "title": "Read a Planning File",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def shanebrain_plan_read(params: PlanReadInput) -> str:
    """Read the full content of a markdown planning file.

    Use shanebrain_plan_list first to discover available files. Planning files
    contain project context, task tracking, error logs, and session continuity
    notes that enable multi-session ADHD-friendly project management.

    Args:
        params (PlanReadInput): filename and subfolder.

    Returns:
        str: Full file content with metadata header.
    """
    try:
        path = PLANNING_DIR / params.subfolder / params.filename
        if not path.exists():
            return f"File not found: `{params.subfolder}/{params.filename}`. Use shanebrain_plan_list to see available files."

        content = path.read_text(encoding="utf-8")
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"## {params.filename}\n*Path: {path} | Modified: {mtime} | Size: {stat.st_size/1024:.1f} KB*\n\n---\n\n{content}"
    except Exception as e:
        return _format_error(e, "shanebrain_plan_read")


class PlanWriteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str = Field(..., description="Filename to create/overwrite (e.g., 'logibot-sprint1.md'). Must end in .md.", min_length=1)
    content: str = Field(..., description="Full markdown content to write", min_length=1, max_length=100000)
    subfolder: str = Field(default="active-projects", description="Subfolder: 'active-projects', 'templates', 'completed', 'logs'")
    append: bool = Field(default=False, description="If True, append to existing file. If False (default), overwrite.")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Filename must not contain path separators or '..'")
        if not v.endswith(".md"):
            raise ValueError("Filename must end with .md")
        return v

@mcp.tool(
    name="shanebrain_plan_write",
    annotations={
        "title": "Write or Append to a Planning File",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def shanebrain_plan_write(params: PlanWriteInput) -> str:
    """Create or update a markdown planning file in the ShaneBrain planning system.

    Supports both full overwrite and append modes. Use append=True to add session
    notes, error logs, or progress updates without losing prior content.

    Args:
        params (PlanWriteInput): filename, content, subfolder, append flag.

    Returns:
        str: Confirmation with path and size written.
    """
    try:
        folder = PLANNING_DIR / params.subfolder
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / params.filename

        mode = "a" if params.append else "w"
        with open(path, mode, encoding="utf-8") as f:
            if params.append:
                f.write(f"\n\n<!-- Appended {datetime.now(timezone.utc).isoformat()} -->\n\n")
            f.write(params.content)

        size_kb = path.stat().st_size / 1024
        action = "Appended to" if params.append else "Wrote"
        return (
            f"✅ {action} `{params.subfolder}/{params.filename}`\n"
            f"**Path:** {path}\n"
            f"**Size:** {size_kb:.1f} KB"
        )
    except Exception as e:
        return _format_error(e, "shanebrain_plan_write")


# ===========================================================================
# TOOLS — GROUP 4: System Health
# ===========================================================================

class HealthCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verbose: bool = Field(default=False, description="Include version/detail info in output")

@mcp.tool(
    name="shanebrain_system_health",
    annotations={
        "title": "Check ShaneBrain Service Health",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def shanebrain_system_health(params: HealthCheckInput, ctx: Context) -> str:
    """Check the health status of all ShaneBrain infrastructure services.

    Pings Weaviate, Ollama, and the planning filesystem. Returns status for
    each with latency. Use this to diagnose connectivity issues before running
    other tools, or to confirm services are up after a Pi reboot.

    Args:
        params (HealthCheckInput): verbose flag for extra detail.

    Returns:
        str: Health dashboard with status, latency, and version info.
    """
    await ctx.report_progress(0.1, "Checking services...")
    results = {}

    # Weaviate
    try:
        import time
        t0 = time.monotonic()
        wdata = await _weaviate_get("/v1/.well-known/ready")
        latency = (time.monotonic() - t0) * 1000
        results["Weaviate"] = {"status": "✅ UP", "latency_ms": round(latency, 1), "detail": str(wdata)}
    except Exception as e:
        results["Weaviate"] = {"status": "❌ DOWN", "latency_ms": None, "detail": str(e)[:100]}

    # Ollama
    try:
        import time
        t0 = time.monotonic()
        odata = await _ollama_get("/api/version")
        latency = (time.monotonic() - t0) * 1000
        version = odata.get("version", "unknown")
        results["Ollama"] = {"status": "✅ UP", "latency_ms": round(latency, 1), "detail": f"v{version}"}
    except Exception as e:
        results["Ollama"] = {"status": "❌ DOWN", "latency_ms": None, "detail": str(e)[:100]}

    # Planning filesystem
    try:
        files = list(PLANNING_DIR.rglob("*.md"))
        results["Planning FS"] = {"status": "✅ UP", "latency_ms": 0, "detail": f"{len(files)} .md files at {PLANNING_DIR}"}
    except Exception as e:
        results["Planning FS"] = {"status": "❌ DOWN", "latency_ms": None, "detail": str(e)[:100]}

    await ctx.report_progress(0.9, "Building report...")
    lines = ["## ShaneBrain System Health\n"]
    lines.append(f"*Checked: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n")
    lines.append("| Service | Status | Latency | Info |")
    lines.append("|---------|--------|---------|------|")
    for service, info in results.items():
        lat = f"{info['latency_ms']} ms" if info["latency_ms"] is not None else "—"
        detail = info["detail"] if params.verbose else ""
        lines.append(f"| **{service}** | {info['status']} | {lat} | {detail} |")

    all_up = all(v["status"].startswith("✅") for v in results.values())
    lines.append(f"\n{'🟢 All systems operational.' if all_up else '🔴 One or more services down — check stderr logs.'}")
    return "\n".join(lines)


# ===========================================================================
# TOOLS — GROUP 5: RAG Semantic Summary (Power tool — RAG + Ollama chained)
# ===========================================================================

class RagSummarizeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(
        ...,
        description="Question to answer using the knowledge base (e.g., 'What is the current Angel Cloud architecture?')",
        min_length=5, max_length=500,
    )
    class_name: str = Field(default=WEAVIATE_KNOWLEDGE_CLASS, description="Weaviate class to search")
    num_chunks: int = Field(default=3, ge=1, le=10, description="Number of RAG chunks to retrieve and synthesize")
    model: str = Field(default=DEFAULT_MODEL, description="Ollama model for synthesis")
    certainty: float = Field(default=0.65, ge=0.0, le=1.0)

@mcp.tool(
    name="shanebrain_rag_answer",
    annotations={
        "title": "Answer Question Using RAG + Local LLM",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def shanebrain_rag_answer(params: RagSummarizeInput, ctx: Context) -> str:
    """Answer a question by retrieving relevant context from Weaviate and
    synthesizing a response using the local Ollama LLM. This is the full
    RAG pipeline — Retrieve → Augment → Generate — running 100% locally.

    Pipeline: nearText search → inject context → Ollama generate → return answer.
    No cloud calls. Complete privacy. Works offline over Tailscale.

    Args:
        params (RagSummarizeInput): question, class_name, num_chunks, model, certainty.

    Returns:
        str: Synthesized answer grounded in retrieved knowledge base content.
    """
    await ctx.report_progress(0.1, "Retrieving context from Weaviate...")
    try:
        # Step 1: RAG retrieval
        gql = {
            "query": f"""
            {{
              Get {{
                {params.class_name}(
                  nearText: {{ concepts: ["{params.question}"], certainty: {params.certainty} }}
                  limit: {params.num_chunks}
                ) {{
                  _additional {{ certainty }}
                  content
                  source
                  timestamp
                }}
              }}
            }}
            """
        }
        data = await _weaviate_post("/v1/graphql", gql)
        chunks = data.get("data", {}).get("Get", {}).get(params.class_name, [])

        if not chunks:
            return (
                f"No relevant context found in '{params.class_name}' for: '{params.question}' "
                f"(certainty ≥ {params.certainty}). "
                "Try storing relevant documents first with shanebrain_rag_store, "
                "or lower the certainty threshold."
            )

        await ctx.report_progress(0.5, "Sending to local LLM...")
        # Step 2: Build augmented prompt
        context_block = "\n\n---\n\n".join(
            f"[Source: {c.get('source', '?')} | {c.get('timestamp', '')[:10]}]\n{c.get('content', '')}"
            for c in chunks
        )
        prompt = (
            f"You are ShaneBrain, a local AI assistant. "
            f"Answer the question using ONLY the provided context. "
            f"If the context doesn't contain enough information, say so clearly.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {params.question}\n\n"
            f"ANSWER:"
        )

        # Step 3: Generate
        result = await _ollama_post("/api/generate", {
            "model": params.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        })
        await ctx.report_progress(0.95, "Done.")

        answer = result.get("response", "").strip()
        sources = ", ".join({c.get("source", "?") for c in chunks})
        return (
            f"## RAG Answer: {params.question}\n\n"
            f"{answer}\n\n"
            f"---\n"
            f"*Grounded in {len(chunks)} chunks from `{params.class_name}` | "
            f"Sources: {sources} | Model: `{params.model}` | Local inference*"
        )

    except Exception as e:
        return _format_error(e, "shanebrain_rag_answer")


# ===========================================================================
# ENTRYPOINT
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShaneBrain MCP Server")
    parser.add_argument("--transport", choices=["streamable_http", "stdio"], default="streamable_http")
    parser.add_argument("--port", type=int, default=MCP_PORT)
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (use 127.0.0.1 for local-only)")
    args = parser.parse_args()

    logger.info("Starting ShaneBrain MCP | transport=%s | port=%s", args.transport, args.port)

    if args.transport == "streamable_http":
        mcp.run(transport="streamable_http", port=args.port, host=args.host)
    else:
        mcp.run()  # stdio
