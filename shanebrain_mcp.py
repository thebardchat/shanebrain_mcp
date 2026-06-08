#!/usr/bin/env python3
"""
ShaneBrain MCP Server v2.6
===========================
42 tools across 18 groups — merged from Pi deployment + GitHub quality patterns.

Groups: Knowledge (2), Chat (3), RAG Chat (1), Social (2), Vault (3),
        Notes (3), Drafts (2), Security (3), Weaviate Admin (2),
        Ollama (2), Planning (3), System (1), Email (2), Calendar (5),
        Weaviate Session (2), Node Bus (3)

Transport: streamable-http on port 8100 (Docker), switchable to sse/stdio via --transport
Quality:   Pydantic v2 validation, MCP annotations, actionable errors, stderr logging
"""

import concurrent.futures
import json
import logging
import os
import socket
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

import anthropic as anthropic_lib
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse
from weaviate.classes.query import Filter

from health import check_gateway, check_weaviate
from weaviate_bridge import DockerWeaviateHelper

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
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
PLANNING_DIR = Path(os.environ.get("PLANNING_DIR", "/app/planning"))
MCP_PORT = int(os.environ.get("MCP_PORT", "8100"))
RAG_CHUNK_LIMIT = 5
NODE_BUS_WEBHOOK = os.environ.get(
    "NODE_BUS_WEBHOOK",
    "https://discord.com/api/webhooks/1512754438429741118/5c7fGicvxD1yvbY-j00Gq8ZMzdJcXY9QNsQYiUR4YS8W1wNZH1unW4JxIPEhYjnxyaYw",
)

# Daily briefing constants
SOBER_SINCE = date(2023, 11, 27)
WEDDING_DATE = date(2026, 5, 2)
BOOK2_PATH = Path(os.environ.get("BOOK2_PATH", "/app/book2/volume-two/compiled/draft-001.md"))
CLUSTER_NODES = ["shanebrain", "alaska", "mexico", "bullfrog", "gulfshores", "neworleans", "pulsar00100", "biloxi"]
CLUSTER_NODE_PORTS = {"biloxi": 8080}  # nodes without SSH use a different health-check port
NODE_BUS_PATH = Path(os.environ.get("NODE_BUS_PATH", "/app/node-bus/bus.db"))
_VERSES = [
    ("Psalm 27:11", "Teach me your way, O Lord; lead me in a straight path."),
    ("Proverbs 3:5-6", "Trust in the Lord with all your heart and lean not on your own understanding."),
    ("Isaiah 40:31", "Those who hope in the Lord will renew their strength. They will soar on wings like eagles."),
    ("Philippians 4:13", "I can do all things through Christ who strengthens me."),
    ("Joshua 1:9", "Be strong and courageous. Do not be afraid; do not be discouraged."),
    ("Romans 8:28", "And we know that in all things God works for the good of those who love him."),
    ("Proverbs 16:3", "Commit to the Lord whatever you do, and he will establish your plans."),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


def _weaviate():
    """Get a Weaviate helper as a context manager (auto-connects and closes)."""
    return DockerWeaviateHelper()


def _claude_generate(system: str, user: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Generate text via Claude Haiku. Returns the response string."""
    client = anthropic_lib.Anthropic(api_key=ANTHROPIC_API_KEY)
    result = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return result.content[0].text.strip()


def _check_node(hostname: str) -> tuple[str, str]:
    """TCP-ping a cluster node. Returns (hostname, status)."""
    port = CLUSTER_NODE_PORTS.get(hostname, 22)
    try:
        socket.create_connection((hostname, port), timeout=2.0).close()
        return hostname, "green"
    except ConnectionRefusedError:
        return hostname, "green"  # host up, service just refused
    except OSError:
        return hostname, "red"


def _cluster_health() -> dict[str, str]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(_check_node, CLUSTER_NODES)
    return dict(results)


def _bus_conn() -> sqlite3.Connection:
    """Open (and initialize) the node-bus SQLite database."""
    NODE_BUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(NODE_BUS_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            from_node TEXT NOT NULL,
            to_node   TEXT NOT NULL DEFAULT 'all',
            content   TEXT NOT NULL,
            tags      TEXT,
            ts        TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _ensure_thoughts_collection(h) -> None:
    if h.collection_exists("Thoughts"):
        return
    import weaviate.classes.config as wcc
    h.client.collections.create(
        name="Thoughts",
        vectorizer_config=wcc.Configure.Vectorizer.text2vec_transformers(),
        properties=[
            wcc.Property(name="content", data_type=wcc.DataType.TEXT),
            wcc.Property(name="timestamp", data_type=wcc.DataType.DATE),
            wcc.Property(name="mood", data_type=wcc.DataType.TEXT),
        ],
    )


def _format_error(e: Exception, context: str = "") -> str:
    """Format errors with actionable hints."""
    prefix = f"[{context}] " if context else ""
    msg = str(e)
    if "connect" in msg.lower():
        return f"{prefix}Connection error: {msg}. Is the service running?"
    if "timeout" in msg.lower():
        return f"{prefix}Timeout: {msg}. Large model or heavy query — try again."
    if "not found" in msg.lower() or "404" in msg:
        return f"{prefix}Not found: {msg}. Check collection/object name."
    return f"{prefix}{type(e).__name__}: {msg}"


def _get_system_prompt():
    """Build the ShaneBrain system prompt with family info."""
    sobriety_days = (datetime.now() - datetime(2023, 11, 27)).days
    sobriety_years = sobriety_days // 365
    sobriety_months = (sobriety_days % 365) // 30

    return f"""You are ShaneBrain - Shane Brazelton's AI, built to serve his family for generations.

CRITICAL RULES:
1. BE BRIEF: 2-4 sentences MAX unless asked for more
2. NEVER HALLUCINATE: If you don't know, say "I don't know that yet"
3. NO FLUFF: Never say "certainly", "I'd be happy to", "great question"
4. FACTS ONLY: Only state what you know for certain

FAMILY (Shane is the FATHER of all 5 sons):
- Shane Brazelton: Father, Creator of ShaneBrain
- Tiffany Brazelton: Wife, Mother
- Gavin Brazelton: Eldest son, married to Angel
- Kai Brazelton: Second son
- Pierce Brazelton: Third son, has ADHD like Shane, wrestler
- Jaxton Brazelton: Fourth son, wrestler
- Ryker Brazelton: Youngest son
- Angel Brazelton: Daughter-in-law, married to Gavin

SOBRIETY: Shane has been sober since November 27, 2023 ({sobriety_years} years, {sobriety_months} months)

Be direct. Be brief. Be accurate."""


# ---------------------------------------------------------------------------
# Lifespan — validate connectivity at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastMCP):
    logger.info("ShaneBrain MCP v2.5 starting — 39 tools, 17 groups")
    # Check Weaviate
    try:
        with _weaviate() as h:
            if h.is_ready():
                logger.info("Weaviate: reachable")
            else:
                logger.warning("Weaviate: NOT ready")
    except Exception as e:
        logger.warning("Weaviate: NOT reachable (%s)", e)
    logger.info("Inference: Claude Haiku (%s)", CLAUDE_MODEL)
    # Ensure planning dirs
    PLANNING_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("active-projects", "templates", "completed", "logs"):
        (PLANNING_DIR / sub).mkdir(exist_ok=True)
    logger.info("Planning dir: %s", PLANNING_DIR)
    yield {}
    logger.info("ShaneBrain MCP shutting down.")


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "ShaneBrain",
    instructions=(
        "ShaneBrain AI tools — knowledge, chat, RAG, social, vault, notes, "
        "drafts, security, admin, ollama, planning, and system health."
    ),
    host="0.0.0.0",
    port=MCP_PORT,
    lifespan=lifespan,
)


# ===========================================================================
# GROUP 1: Knowledge (LegacyKnowledge) — 2 tools
# ===========================================================================

class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for (e.g. 'family values', 'sobriety journey')", min_length=1, max_length=500)
    category: Optional[str] = Field(default=None, description="Filter: family, faith, technical, philosophy, general, wellness")
    limit: int = Field(default=5, ge=1, le=50, description="Max results")
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON)


@mcp.tool(
    name="shanebrain_search_knowledge",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_search_knowledge(params: SearchKnowledgeInput) -> str:
    """Semantic search across ShaneBrain's legacy knowledge base (LegacyKnowledge).

    Finds conceptually similar entries — not keyword matching. Covers family,
    faith, technical decisions, philosophy, and wellness knowledge.
    """
    try:
        with _weaviate() as h:
            results = h.search_knowledge(params.query, category=params.category, limit=params.limit)
            if not results:
                return json.dumps({"results": [], "message": f"No matches for '{params.query}'."})
            if params.response_format == ResponseFormat.MARKDOWN:
                lines = [f"## Knowledge Search: {params.query}", f"**Results:** {len(results)}\n"]
                for i, r in enumerate(results, 1):
                    title = r.get("title", "Untitled")
                    dist = r.get("_distance", "N/A")
                    content = (r.get("content") or "")[:400]
                    lines.append(f"### {i}. {title} (distance: {dist})\n{content}\n")
                return "\n".join(lines)
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_search_knowledge")


class AddKnowledgeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    content: str = Field(..., description="The knowledge content", min_length=1, max_length=50000)
    category: str = Field(..., description="Category: family, faith, technical, philosophy, general, wellness")
    source: str = Field(default="mcp", description="Where this came from")
    title: Optional[str] = Field(default=None, description="Optional title")


@mcp.tool(
    name="shanebrain_add_knowledge",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_add_knowledge(params: AddKnowledgeInput) -> str:
    """Add an entry to ShaneBrain's legacy knowledge base (LegacyKnowledge).

    Stored content is auto-vectorized and becomes immediately searchable.
    """
    try:
        with _weaviate() as h:
            uid = h.add_knowledge(params.content, params.category, source=params.source, title=params.title)
            if uid:
                return json.dumps({"success": True, "uuid": uid, "preview": params.content[:120]})
            return json.dumps({"success": False, "error": "Failed — collection may not exist."})
    except Exception as e:
        return _format_error(e, "shanebrain_add_knowledge")


# ===========================================================================
# GROUP 2: Chat (Conversation) — 3 tools
# ===========================================================================

class SearchConversationsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for", min_length=1, max_length=500)
    mode: Optional[str] = Field(default=None, description="Filter: CHAT, MEMORY, WELLNESS, SECURITY, DISPATCH, CODE")
    limit: int = Field(default=10, ge=1, le=50)


@mcp.tool(
    name="shanebrain_search_conversations",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_search_conversations(params: SearchConversationsInput) -> str:
    """Search past ShaneBrain conversations semantically."""
    try:
        with _weaviate() as h:
            results = h.search_conversations(params.query, mode=params.mode, limit=params.limit)
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_search_conversations")


class LogConversationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    message: str = Field(..., description="Message content", min_length=1)
    role: str = Field(..., description="Role: user, assistant, system")
    mode: str = Field(default="CHAT", description="Mode: CHAT, MEMORY, WELLNESS, SECURITY, DISPATCH, CODE")
    session_id: Optional[str] = Field(default=None, description="Session ID (auto-generated if omitted)")


@mcp.tool(
    name="shanebrain_log_conversation",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_log_conversation(params: LogConversationInput) -> str:
    """Log a message to ShaneBrain's conversation history."""
    try:
        with _weaviate() as h:
            uid = h.log_conversation(params.message, params.role, mode=params.mode, session_id=params.session_id)
            if uid:
                return json.dumps({"success": True, "uuid": uid})
            return json.dumps({"success": False, "error": "Failed to log."})
    except Exception as e:
        return _format_error(e, "shanebrain_log_conversation")


class GetConversationHistoryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    session_id: str = Field(..., description="The session identifier")
    limit: int = Field(default=50, ge=1, le=200)


@mcp.tool(
    name="shanebrain_get_conversation_history",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_get_conversation_history(params: GetConversationHistoryInput) -> str:
    """Get conversation history for a session by ID."""
    try:
        with _weaviate() as h:
            results = h.get_conversation_history(params.session_id, limit=params.limit)
            return json.dumps({"messages": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_get_conversation_history")


# ===========================================================================
# GROUP 3: RAG Chat — 1 tool
# ===========================================================================

class ChatInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    message: str = Field(..., description="Your message to ShaneBrain", min_length=1, max_length=2000)
    model: str = Field(default="", description="Unused — inference is Claude Haiku")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=100, ge=1, le=4096)


@mcp.tool(
    name="shanebrain_chat",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_chat(params: ChatInput) -> str:
    """Full RAG chat — searches knowledge base, then generates via local Ollama.

    Pipeline: semantic search LegacyKnowledge -> inject context -> Ollama generate.
    100% local, zero cloud. Uses ShaneBrain persona with family knowledge.
    """
    try:
        with _weaviate() as h:
            # RAG retrieval
            chunks = []
            results = h.search_knowledge(params.message, limit=RAG_CHUNK_LIMIT)
            for r in results:
                content = r.get("content", "")
                title = r.get("title", "")
                if content:
                    chunks.append(f"[{title}]\n{content}" if title else content)

            # Build prompt
            system = _get_system_prompt()
            if chunks:
                context = "\n\n---\n\n".join(chunks)
                system += f"\n\nRELEVANT KNOWLEDGE FROM MEMORY:\n{context}\n\nUse this knowledge to answer. If it doesn't help, say you don't know."

            # Generate via Claude Haiku (Ollama decommissioned 2026-04-30)
            response_text = _claude_generate(
                system=system,
                user=params.message,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
            )

            return json.dumps({
                "response": response_text,
                "knowledge_chunks_used": len(chunks),
                "model": CLAUDE_MODEL,
            })
    except Exception as e:
        return _format_error(e, "shanebrain_chat")


# ===========================================================================
# GROUP 4: Social (FriendProfile) — 2 tools
# ===========================================================================

class SearchFriendsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for (e.g. a person's name or topic)", min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


@mcp.tool(
    name="shanebrain_search_friends",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_search_friends(params: SearchFriendsInput) -> str:
    """Search ShaneBrain's friend profiles semantically."""
    try:
        with _weaviate() as h:
            results = h.search_friends(params.query, limit=params.limit)
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_search_friends")


class GetTopFriendsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=10, ge=1, le=50, description="Max results")


@mcp.tool(
    name="shanebrain_get_top_friends",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_get_top_friends(params: GetTopFriendsInput) -> str:
    """Get friend profiles ranked by relationship strength (highest first)."""
    try:
        with _weaviate() as h:
            results = h.get_top_friends(limit=params.limit)
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_get_top_friends")


# ===========================================================================
# GROUP 5: Vault (PersonalDoc) — 3 tools
# ===========================================================================

class VaultSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for", min_length=1, max_length=500)
    category: Optional[str] = Field(default=None, description="Optional category filter")
    limit: int = Field(default=10, ge=1, le=50)


@mcp.tool(
    name="shanebrain_vault_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_vault_search(params: VaultSearchInput) -> str:
    """Search Shane's personal vault (PersonalDoc) semantically."""
    try:
        with _weaviate() as h:
            filters = None
            if params.category:
                filters = Filter.by_property("category").equal(params.category)
            results = h._generic_near_text("PersonalDoc", params.query, filters=filters, limit=params.limit)
            if not results and not h.collection_exists("PersonalDoc"):
                return json.dumps({"results": [], "message": "PersonalDoc collection does not exist yet."})
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_vault_search")


class VaultAddInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    content: str = Field(..., description="Document content", min_length=1, max_length=50000)
    category: str = Field(..., description="Category: medical, legal, financial, personal, work")
    title: Optional[str] = Field(default=None, description="Optional title")
    tags: Optional[str] = Field(default=None, description="Comma-separated tags")


@mcp.tool(
    name="shanebrain_vault_add",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_vault_add(params: VaultAddInput) -> str:
    """Add a document to Shane's personal vault (PersonalDoc)."""
    try:
        with _weaviate() as h:
            data = {
                "content": params.content,
                "category": params.category,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if params.title:
                data["title"] = params.title
            if params.tags:
                data["tags"] = [t.strip() for t in params.tags.split(",")]
            uid = h._generic_insert("PersonalDoc", data)
            if uid:
                return json.dumps({"success": True, "uuid": uid})
            return json.dumps({"success": False, "error": "PersonalDoc collection may not exist."})
    except Exception as e:
        return _format_error(e, "shanebrain_vault_add")


class VaultListCategoriesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=100, ge=1, le=500)


@mcp.tool(
    name="shanebrain_vault_list_categories",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_vault_list_categories(params: VaultListCategoriesInput) -> str:
    """List document counts per category in the PersonalDoc vault."""
    try:
        with _weaviate() as h:
            if not h.collection_exists("PersonalDoc"):
                return json.dumps({"error": "PersonalDoc collection does not exist yet.", "categories": {}})
            docs = h._generic_fetch("PersonalDoc", limit=params.limit)
            counts = {}
            for d in docs:
                cat = d.get("category", "uncategorized")
                counts[cat] = counts.get(cat, 0) + 1
            return json.dumps({"categories": counts, "total": len(docs)})
    except Exception as e:
        return _format_error(e, "shanebrain_vault_list_categories")


# ===========================================================================
# GROUP 6: Notes (DailyNote) — 3 tools
# ===========================================================================

class DailyNoteAddInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    content: str = Field(..., description="Note content", min_length=1)
    note_type: str = Field(default="journal", description="Type: journal, todo, reminder, reflection")
    mood: Optional[str] = Field(default=None, description="Mood tag: grateful, tired, focused, anxious, etc.")


@mcp.tool(
    name="shanebrain_daily_note_add",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_daily_note_add(params: DailyNoteAddInput) -> str:
    """Add a daily note (journal, todo, reminder, or reflection)."""
    try:
        with _weaviate() as h:
            data = {
                "content": params.content,
                "note_type": params.note_type,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if params.mood:
                data["mood"] = params.mood
            uid = h._generic_insert("DailyNote", data)
            if uid:
                return json.dumps({"success": True, "uuid": uid})
            return json.dumps({"success": False, "error": "DailyNote collection does not exist yet."})
    except Exception as e:
        return _format_error(e, "shanebrain_daily_note_add")


class DailyNoteSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for", min_length=1, max_length=500)
    note_type: Optional[str] = Field(default=None, description="Filter: journal, todo, reminder, reflection")
    limit: int = Field(default=10, ge=1, le=50)


@mcp.tool(
    name="shanebrain_daily_note_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_daily_note_search(params: DailyNoteSearchInput) -> str:
    """Search daily notes semantically."""
    try:
        with _weaviate() as h:
            filters = None
            if params.note_type:
                filters = Filter.by_property("note_type").equal(params.note_type)
            results = h._generic_near_text("DailyNote", params.query, filters=filters, limit=params.limit)
            if not results and not h.collection_exists("DailyNote"):
                return json.dumps({"results": [], "message": "DailyNote collection does not exist yet."})
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_daily_note_search")


@mcp.tool(
    name="shanebrain_daily_briefing",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    meta={"ui": {"resourceUri": "ui://shanebrain/briefing"}},
)
def shanebrain_daily_briefing() -> str:
    """Personal daily briefing — sobriety, verse, book word count, cluster health, recent thoughts."""
    try:
        today = date.today()
        sober_days = (today - SOBER_SINCE).days
        wedding_days = (today - WEDDING_DATE).days  # days since wedding (honeymoon counter)
        dow = today.weekday()  # 0=Mon
        verse_ref, verse_text = _VERSES[dow % 7]

        # Book word count
        book_words = 0
        try:
            book_words = len(BOOK2_PATH.read_text(encoding="utf-8", errors="ignore").split())
        except OSError:
            pass

        # Cluster health
        cluster = _cluster_health()

        # Last 3 Thoughts
        thoughts = []
        try:
            with _weaviate() as h:
                _ensure_thoughts_collection(h)
                thoughts = h._generic_fetch("Thoughts", limit=3)
        except Exception:
            pass

        payload = {
            "sober_days": sober_days,
            "today": today.strftime("%A, %B %-d %Y"),
            "verse_ref": verse_ref,
            "verse_text": verse_text,
            "book2_words": book_words,
            "honeymoon_days": max(0, wedding_days),
            "cluster": cluster,
            "thoughts": thoughts,
        }
        return json.dumps(payload, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_daily_briefing")


# ===========================================================================
# GROUP 7: Drafts (PersonalDraft) — 2 tools
# ===========================================================================

class DraftCreateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    prompt: str = Field(..., description="What to write about", min_length=1, max_length=2000)
    draft_type: str = Field(default="general", description="Type: email, message, post, letter, general")
    use_vault_context: bool = Field(default=True, description="Search PersonalDoc for context (default True)")


@mcp.tool(
    name="shanebrain_draft_create",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_draft_create(params: DraftCreateInput) -> str:
    """Generate a writing draft with optional vault context via Ollama.

    Searches PersonalDoc for relevant context, then generates in Shane's voice.
    Saves the result to PersonalDraft collection.
    """
    try:
        with _weaviate() as h:
            context_chunks = []
            if params.use_vault_context and h.collection_exists("PersonalDoc"):
                results = h._generic_near_text("PersonalDoc", params.prompt, limit=3)
                for r in results:
                    content = r.get("content", "")
                    if content:
                        context_chunks.append(content)

            system = "You are ShaneBrain, helping Shane draft content. Match his voice: direct, warm, no fluff."
            if context_chunks:
                system += f"\n\nRelevant context from vault:\n" + "\n---\n".join(context_chunks)

            draft_text = _claude_generate(
                system=system,
                user=f"Draft a {params.draft_type}: {params.prompt}",
                max_tokens=150,
                temperature=0.5,
            )

            saved_uuid = None
            if h.collection_exists("PersonalDraft"):
                saved_uuid = h._generic_insert("PersonalDraft", {
                    "content": draft_text,
                    "prompt": params.prompt,
                    "draft_type": params.draft_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            return json.dumps({
                "draft": draft_text,
                "draft_type": params.draft_type,
                "saved": saved_uuid is not None,
                "uuid": saved_uuid,
                "vault_context_used": len(context_chunks),
            })
    except Exception as e:
        return _format_error(e, "shanebrain_draft_create")


class DraftSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for", min_length=1, max_length=500)
    draft_type: Optional[str] = Field(default=None, description="Filter: email, message, post, letter, general")
    limit: int = Field(default=10, ge=1, le=50)


@mcp.tool(
    name="shanebrain_draft_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_draft_search(params: DraftSearchInput) -> str:
    """Search saved writing drafts (PersonalDraft) semantically."""
    try:
        with _weaviate() as h:
            filters = None
            if params.draft_type:
                filters = Filter.by_property("draft_type").equal(params.draft_type)
            results = h._generic_near_text("PersonalDraft", params.query, filters=filters, limit=params.limit)
            if not results and not h.collection_exists("PersonalDraft"):
                return json.dumps({"results": [], "message": "PersonalDraft collection does not exist yet."})
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_draft_search")


# ===========================================================================
# GROUP 8: Security (SecurityLog, PrivacyAudit) — 3 tools
# ===========================================================================

class SecurityLogSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for (e.g. 'failed login', 'unusual activity')", min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


@mcp.tool(
    name="shanebrain_security_log_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_security_log_search(params: SecurityLogSearchInput) -> str:
    """Search SecurityLog entries semantically."""
    try:
        with _weaviate() as h:
            results = h._generic_near_text("SecurityLog", params.query, limit=params.limit)
            if not results and not h.collection_exists("SecurityLog"):
                return json.dumps({"results": [], "message": "SecurityLog collection does not exist yet."})
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_security_log_search")


class SecurityLogRecentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    severity: str = Field(default="", description="Filter: low, medium, high, critical. Empty for all.")
    limit: int = Field(default=20, ge=1, le=100)


@mcp.tool(
    name="shanebrain_security_log_recent",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_security_log_recent(params: SecurityLogRecentInput) -> str:
    """Get recent security log entries (chronological, not semantic)."""
    try:
        with _weaviate() as h:
            if not h.collection_exists("SecurityLog"):
                return json.dumps({"results": [], "message": "SecurityLog collection does not exist yet."})
            collection = h.client.collections.get("SecurityLog")
            from weaviate.classes.query import Sort
            sort = Sort.by_creation_time(ascending=False)
            if params.severity:
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("severity").equal(params.severity),
                    sort=sort,
                    limit=params.limit,
                )
            else:
                response = collection.query.fetch_objects(sort=sort, limit=params.limit)
            results = [obj.properties for obj in response.objects]
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_security_log_recent")


class PrivacyAuditSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="What to search for", min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


@mcp.tool(
    name="shanebrain_privacy_audit_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_privacy_audit_search(params: PrivacyAuditSearchInput) -> str:
    """Search PrivacyAudit entries semantically."""
    try:
        with _weaviate() as h:
            results = h._generic_near_text("PrivacyAudit", params.query, limit=params.limit)
            if not results and not h.collection_exists("PrivacyAudit"):
                return json.dumps({"results": [], "message": "PrivacyAudit collection does not exist yet."})
            return json.dumps({"results": results, "count": len(results)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_privacy_audit_search")


# ===========================================================================
# GROUP 9: Weaviate Admin — 2 tools (NEW from GitHub)
# ===========================================================================

class RagDeleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_name: str = Field(..., description="Weaviate collection the object belongs to")
    object_id: str = Field(..., description="Weaviate object UUID to delete (get from search results)")


@mcp.tool(
    name="shanebrain_rag_delete",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_rag_delete(params: RagDeleteInput) -> str:
    """Permanently delete a specific object from Weaviate by UUID.

    Use with caution — deletion is irreversible. Get the object ID first
    using a search tool and inspecting the results.
    """
    try:
        with _weaviate() as h:
            if not h.collection_exists(params.collection_name):
                return json.dumps({"error": f"Collection '{params.collection_name}' does not exist."})
            collection = h.client.collections.get(params.collection_name)
            collection.data.delete_by_id(params.object_id)
            return json.dumps({"success": True, "deleted": params.object_id, "collection": params.collection_name})
    except Exception as e:
        return _format_error(e, "shanebrain_rag_delete")


class RagListClassesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


@mcp.tool(
    name="shanebrain_rag_list_classes",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_rag_list_classes(params: RagListClassesInput) -> str:
    """List all Weaviate collections with object counts.

    Useful for discovering what data is stored, checking health, and
    understanding which collections to search.
    """
    try:
        with _weaviate() as h:
            counts = h.get_all_collection_counts()
            total = sum(counts.values())
            if params.response_format == ResponseFormat.MARKDOWN:
                lines = ["## Weaviate Collections\n", "| Collection | Objects |", "|------------|---------|"]
                for name, count in counts.items():
                    lines.append(f"| `{name}` | {count} |")
                lines.append(f"\n**Total:** {total} objects across {len(counts)} collections")
                return "\n".join(lines)
            return json.dumps({"collections": counts, "total": total})
    except Exception as e:
        return _format_error(e, "shanebrain_rag_list_classes")


# ===========================================================================
# GROUP 10: Ollama Local Inference — 2 tools (NEW from GitHub)
# ===========================================================================

class OllamaGenerateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    prompt: str = Field(..., description="Prompt or question for the local model", min_length=1, max_length=8000)
    model: str = Field(default="", description="Unused — inference is Claude Haiku")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt", max_length=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)


@mcp.tool(
    name="shanebrain_ollama_generate",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_ollama_generate(params: OllamaGenerateInput) -> str:
    """Generate text using Claude Haiku (Ollama decommissioned 2026-04-30).

    For RAG-grounded answers use shanebrain_chat instead. This tool is for
    direct generation without knowledge base context.
    """
    try:
        system = params.system_prompt or "You are ShaneBrain, Shane Brazelton's personal AI. Be direct and brief."
        text = _claude_generate(
            system=system,
            user=params.prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
        )
        return json.dumps({
            "response": text,
            "model": CLAUDE_MODEL,
        })
    except Exception as e:
        return _format_error(e, "shanebrain_ollama_generate")


class OllamaListModelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


@mcp.tool(
    name="shanebrain_ollama_list_models",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_ollama_list_models(params: OllamaListModelsInput) -> str:
    """Ollama decommissioned 2026-04-30. Returns status message."""
    return "Ollama decommissioned 2026-04-30. All inference now routes through Claude Haiku (claude-haiku-4-5-20251001). No local models running."


# ===========================================================================
# GROUP 11: Planning System — 3 tools (NEW from GitHub)
# ===========================================================================

class PlanListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subfolder: str = Field(
        default="active-projects",
        description="Subfolder: 'active-projects', 'templates', 'completed', 'logs'",
    )


@mcp.tool(
    name="shanebrain_plan_list",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_plan_list(params: PlanListInput) -> str:
    """List markdown planning files in the ShaneBrain planning system.

    The planning system uses persistent markdown files for multi-session
    project continuity. Discover files before reading or writing.
    """
    try:
        folder = PLANNING_DIR / params.subfolder
        if not folder.exists():
            return f"Subfolder '{params.subfolder}' does not exist under {PLANNING_DIR}."

        files = sorted(folder.glob("*.md"))
        if not files:
            return f"No markdown files in '{params.subfolder}'."

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
    filename: str = Field(..., description="Filename to read (e.g. 'angel-cloud-phase2.md')", min_length=1)
    subfolder: str = Field(default="active-projects")

    @field_validator("filename")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Filename must not contain path separators or '..'")
        return v


@mcp.tool(
    name="shanebrain_plan_read",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_plan_read(params: PlanReadInput) -> str:
    """Read the full content of a markdown planning file.

    Use shanebrain_plan_list first to discover available files.
    """
    try:
        path = PLANNING_DIR / params.subfolder / params.filename
        if not path.exists():
            return f"File not found: `{params.subfolder}/{params.filename}`. Use shanebrain_plan_list to see available files."

        content = path.read_text(encoding="utf-8")
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"## {params.filename}\n*Modified: {mtime} | Size: {stat.st_size / 1024:.1f} KB*\n\n---\n\n{content}"
    except Exception as e:
        return _format_error(e, "shanebrain_plan_read")


class PlanWriteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    filename: str = Field(..., description="Filename (must end in .md)", min_length=1)
    content: str = Field(..., description="Full markdown content to write", min_length=1, max_length=100000)
    subfolder: str = Field(default="active-projects")
    append: bool = Field(default=False, description="Append to existing file instead of overwrite")

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
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_plan_write(params: PlanWriteInput) -> str:
    """Create or update a markdown planning file.

    Supports overwrite and append modes. Use append=True to add session
    notes or progress updates without losing prior content.
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
        return json.dumps({"success": True, "action": action, "file": f"{params.subfolder}/{params.filename}", "size_kb": round(size_kb, 1)})
    except Exception as e:
        return _format_error(e, "shanebrain_plan_write")


# ===========================================================================
# GROUP 12: System Health — 1 tool
# ===========================================================================

@mcp.tool(
    name="shanebrain_system_health",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_system_health() -> str:
    """Check ShaneBrain system health — Weaviate, Gateway + all collection counts."""
    try:
        with _weaviate() as h:
            weaviate_status = check_weaviate(h)
            gateway_status = check_gateway()
            counts = h.get_all_collection_counts()

            return json.dumps({
                "services": {
                    "weaviate": weaviate_status,
                    "inference": {"status": "ok", "service": "claude-haiku", "model": CLAUDE_MODEL},
                    "gateway": gateway_status,
                },
                "collections": counts,
                "total_objects": sum(counts.values()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_system_health")


# ===========================================================================
# GROUP 13: Email — 2 tools (send + reply via Gmail SMTP)
# ===========================================================================

GMAIL_USER = "brazeltonshane@gmail.com"
# Set GMAIL_APP_PASSWORD environment variable — never hardcode credentials
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")


class EmailSendInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    to: str = Field(..., description="Recipient email address", min_length=5)
    subject: str = Field(..., description="Email subject line", min_length=1)
    body: str = Field(..., description="Email body (plain text or HTML)", min_length=1)
    html: bool = Field(default=False, description="Set True to send as HTML email")


class EmailReplyInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    to: str = Field(..., description="Recipient email address to reply to", min_length=5)
    subject: str = Field(..., description="Email subject (usually 'Re: ...')", min_length=1)
    body: str = Field(..., description="Reply body (plain text or HTML)", min_length=1)
    html: bool = Field(default=False, description="Set True to send as HTML email")
    in_reply_to: Optional[str] = Field(default=None, description="Message-ID header of the email being replied to")
    references: Optional[str] = Field(default=None, description="References header for threading")


def _send_email(to: str, subject: str, body: str, html: bool = False,
                in_reply_to: str = None, references: str = None) -> dict:
    """Internal helper to send email via Gmail SMTP."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    if not GMAIL_APP_PASSWORD:
        raise RuntimeError("GMAIL_APP_PASSWORD environment variable is not set.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = to
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    content_type = "html" if html else "plain"
    msg.attach(MIMEText(body, content_type))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, to, msg.as_string())

    return {"sent": True, "to": to, "subject": subject}


@mcp.tool(
    name="shanebrain_send_email",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def shanebrain_send_email(params: EmailSendInput) -> str:
    """Send an email from Shane's Gmail (brazeltonshane@gmail.com)."""
    try:
        result = _send_email(params.to, params.subject, params.body, params.html)
        return json.dumps(result)
    except Exception as e:
        return _format_error(e, "shanebrain_send_email")


@mcp.tool(
    name="shanebrain_reply_email",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def shanebrain_reply_email(params: EmailReplyInput) -> str:
    """Reply to an email from Shane's Gmail. Provide in_reply_to Message-ID for proper threading."""
    try:
        result = _send_email(
            params.to, params.subject, params.body, params.html,
            params.in_reply_to, params.references,
        )
        result["reply"] = True
        return json.dumps(result)
    except Exception as e:
        return _format_error(e, "shanebrain_reply_email")


# ===========================================================================
# GROUP 14: Google Calendar — 5 tools (list, get, create, update, delete)
# ===========================================================================

GCAL_TOKEN_PATH = Path(os.environ.get("GCAL_TOKEN_PATH", "/app/gcal_token.json"))
GCAL_CALENDAR_ID = os.environ.get("GCAL_CALENDAR_ID", "primary")


def _gcal_service():
    """Build an authenticated Google Calendar API service, auto-refreshing the token."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    if not GCAL_TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"gcal_token.json not found at {GCAL_TOKEN_PATH}. "
            "Run scripts/google_calendar_setup.py to authenticate."
        )

    data = json.loads(GCAL_TOKEN_PATH.read_text())
    creds = Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes", ["https://www.googleapis.com/auth/calendar.events"]),
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        data["token"] = creds.token
        try:
            GCAL_TOKEN_PATH.write_text(json.dumps(data, indent=2))
        except (PermissionError, OSError) as write_err:
            logger.warning("Could not persist refreshed token (%s). Mount rw to persist.", write_err)

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


class CalendarListInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    days: int = Field(default=7, ge=1, le=90, description="Number of upcoming days to fetch (1-90)")
    max_results: int = Field(default=20, ge=1, le=100, description="Max events to return")
    calendar_id: Optional[str] = Field(default=None, description="Calendar ID (default: primary)")


class CalendarGetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    event_id: str = Field(..., description="Google Calendar event ID", min_length=1)
    calendar_id: Optional[str] = Field(default=None, description="Calendar ID (default: primary)")


class CalendarCreateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    summary: str = Field(..., description="Event title", min_length=1)
    start: str = Field(..., description="Start datetime in ISO 8601 (e.g. 2026-04-25T14:00:00) or date (2026-04-25)")
    end: str = Field(default="", description="End datetime in ISO 8601 or date. Leave blank for 1-hour duration.")
    description: Optional[str] = Field(default=None, description="Event description / notes")
    location: Optional[str] = Field(default=None, description="Event location")
    attendees: Optional[List[str]] = Field(default=None, description="List of attendee email addresses")
    calendar_id: Optional[str] = Field(default=None, description="Calendar ID (default: primary)")
    timezone: str = Field(default="America/Chicago", description="Timezone for the event")


class CalendarUpdateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    event_id: str = Field(..., description="Google Calendar event ID to update", min_length=1)
    summary: Optional[str] = Field(default=None, description="New event title")
    start: Optional[str] = Field(default=None, description="New start datetime ISO 8601")
    end: Optional[str] = Field(default=None, description="New end datetime ISO 8601")
    description: Optional[str] = Field(default=None, description="New description")
    location: Optional[str] = Field(default=None, description="New location")
    calendar_id: Optional[str] = Field(default=None, description="Calendar ID (default: primary)")
    timezone: str = Field(default="America/Chicago", description="Timezone for updated times")


class CalendarDeleteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    event_id: str = Field(..., description="Google Calendar event ID to delete", min_length=1)
    calendar_id: Optional[str] = Field(default=None, description="Calendar ID (default: primary)")


def _parse_gcal_datetime(dt_str: str, tz: str, duration_hours: int = 1) -> tuple[dict, dict]:
    """Return (start_body, end_body) for the Google Calendar API from an ISO string.

    Always generates an end that is duration_hours after start. The caller
    overrides end_body if the user supplied an explicit end parameter.
    """
    from datetime import timedelta, datetime as _dt

    is_date_only = len(dt_str) == 10  # YYYY-MM-DD
    if is_date_only:
        # All-day events: end date is the next day (exclusive)
        from datetime import date, timedelta as _td
        start_date = date.fromisoformat(dt_str)
        end_date = start_date + _td(days=1)
        return {"date": dt_str}, {"date": end_date.isoformat()}

    # Parse datetime string, handling both naive and aware variants
    try:
        start_dt = _dt.fromisoformat(dt_str)
    except ValueError:
        # Fallback: treat as-is and let Google reject invalid input
        start_body = {"dateTime": dt_str, "timeZone": tz}
        return start_body, start_body

    end_dt = start_dt + timedelta(hours=duration_hours)

    if "+" in dt_str or dt_str.endswith("Z"):
        # Timezone-aware string — preserve offset, don't add timeZone field
        start_body = {"dateTime": dt_str}
        end_body = {"dateTime": end_dt.isoformat()}
    else:
        # Naive string — attach the provided timezone
        start_body = {"dateTime": dt_str, "timeZone": tz}
        end_body = {"dateTime": end_dt.isoformat(), "timeZone": tz}

    return start_body, end_body


@mcp.tool(
    name="shanebrain_calendar_list",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
def shanebrain_calendar_list(params: CalendarListInput) -> str:
    """List upcoming Google Calendar events for the next N days."""
    try:
        from datetime import timedelta
        service = _gcal_service()
        cal_id = params.calendar_id or GCAL_CALENDAR_ID
        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=params.days)
        result = service.events().list(
            calendarId=cal_id,
            timeMin=now.isoformat(),
            timeMax=time_max.isoformat(),
            maxResults=params.max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        simplified = []
        for e in events:
            simplified.append({
                "id": e["id"],
                "summary": e.get("summary", "(no title)"),
                "start": e["start"].get("dateTime", e["start"].get("date")),
                "end": e["end"].get("dateTime", e["end"].get("date")),
                "location": e.get("location"),
                "description": e.get("description"),
            })
        return json.dumps({"events": simplified, "count": len(simplified), "days": params.days})
    except Exception as e:
        return _format_error(e, "shanebrain_calendar_list")


@mcp.tool(
    name="shanebrain_calendar_get",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
def shanebrain_calendar_get(params: CalendarGetInput) -> str:
    """Get a specific Google Calendar event by ID."""
    try:
        service = _gcal_service()
        cal_id = params.calendar_id or GCAL_CALENDAR_ID
        event = service.events().get(calendarId=cal_id, eventId=params.event_id).execute()
        return json.dumps({
            "id": event["id"],
            "summary": event.get("summary", "(no title)"),
            "start": event["start"].get("dateTime", event["start"].get("date")),
            "end": event["end"].get("dateTime", event["end"].get("date")),
            "location": event.get("location"),
            "description": event.get("description"),
            "attendees": [a["email"] for a in event.get("attendees", [])],
            "htmlLink": event.get("htmlLink"),
            "status": event.get("status"),
        })
    except Exception as e:
        return _format_error(e, "shanebrain_calendar_get")


@mcp.tool(
    name="shanebrain_calendar_create",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def shanebrain_calendar_create(params: CalendarCreateInput) -> str:
    """Create a new Google Calendar event. Returns the created event ID and link."""
    try:
        service = _gcal_service()
        cal_id = params.calendar_id or GCAL_CALENDAR_ID

        start_body, auto_end = _parse_gcal_datetime(params.start, params.timezone)

        if params.end and params.end.strip():
            # Caller supplied explicit end — build the body directly
            if "T" in params.end:
                if "+" in params.end or params.end.endswith("Z"):
                    end_body = {"dateTime": params.end}
                else:
                    end_body = {"dateTime": params.end, "timeZone": params.timezone}
            else:
                end_body = {"date": params.end}
        else:
            end_body = auto_end

        body = {
            "summary": params.summary,
            "start": start_body,
            "end": end_body,
        }
        if params.description:
            body["description"] = params.description
        if params.location:
            body["location"] = params.location
        if params.attendees:
            body["attendees"] = [{"email": a} for a in params.attendees]

        created = service.events().insert(calendarId=cal_id, body=body).execute()
        return json.dumps({
            "created": True,
            "id": created["id"],
            "summary": created.get("summary"),
            "start": created["start"].get("dateTime", created["start"].get("date")),
            "end": created["end"].get("dateTime", created["end"].get("date")),
            "htmlLink": created.get("htmlLink"),
        })
    except Exception as e:
        return _format_error(e, "shanebrain_calendar_create")


@mcp.tool(
    name="shanebrain_calendar_update",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
def shanebrain_calendar_update(params: CalendarUpdateInput) -> str:
    """Update an existing Google Calendar event. Only provided fields are changed."""
    try:
        service = _gcal_service()
        cal_id = params.calendar_id or GCAL_CALENDAR_ID

        # Fetch current event to patch
        event = service.events().get(calendarId=cal_id, eventId=params.event_id).execute()

        if params.summary:
            event["summary"] = params.summary
        if params.description is not None:
            event["description"] = params.description
        if params.location is not None:
            event["location"] = params.location
        if params.start:
            if "T" in params.start:
                event["start"] = {"dateTime": params.start, "timeZone": params.timezone}
            else:
                event["start"] = {"date": params.start}
        if params.end:
            if "T" in params.end:
                event["end"] = {"dateTime": params.end, "timeZone": params.timezone}
            else:
                event["end"] = {"date": params.end}

        updated = service.events().update(calendarId=cal_id, eventId=params.event_id, body=event).execute()
        return json.dumps({
            "updated": True,
            "id": updated["id"],
            "summary": updated.get("summary"),
            "start": updated["start"].get("dateTime", updated["start"].get("date")),
            "end": updated["end"].get("dateTime", updated["end"].get("date")),
            "htmlLink": updated.get("htmlLink"),
        })
    except Exception as e:
        return _format_error(e, "shanebrain_calendar_update")


@mcp.tool(
    name="shanebrain_calendar_delete",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
def shanebrain_calendar_delete(params: CalendarDeleteInput) -> str:
    """Delete a Google Calendar event by ID. This is permanent."""
    try:
        service = _gcal_service()
        cal_id = params.calendar_id or GCAL_CALENDAR_ID
        service.events().delete(calendarId=cal_id, eventId=params.event_id).execute()
        return json.dumps({"deleted": True, "event_id": params.event_id})
    except Exception as e:
        return _format_error(e, "shanebrain_calendar_delete")


# ===========================================================================
# HTTP Health Endpoint (for Docker healthcheck / monitoring)
# ===========================================================================

@mcp.custom_route("/health", methods=["GET"])
async def http_health(request: Request) -> JSONResponse:
    """HTTP health endpoint for Docker healthcheck and monitoring."""
    try:
        with _weaviate() as h:
            weaviate_ok = h.is_ready()
    except Exception:
        weaviate_ok = False

    healthy = weaviate_ok

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "healthy" if healthy else "degraded",
            "weaviate": "ok" if weaviate_ok else "down",
            "ollama": "disabled",
            "version": "2.3.0",
        },
    )


# ===========================================================================
# GROUP 15: Context Snapshot — 1 tool
# Returns a rich Shane-context object for session start injection
# ===========================================================================

@mcp.tool(
    name="shanebrain_context_snapshot",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_context_snapshot() -> str:
    """
    Pull a rich context snapshot of Shane's current state from Weaviate.
    Call at the start of every session so Claude walks in knowing Shane — not cold.
    Returns: sobriety, recent mood, last 3 sessions, active projects, family state.
    """
    from datetime import datetime

    snapshot = {}

    # --- Sobriety ---
    sober_since = datetime(2023, 11, 27)
    days = (datetime.now() - sober_since).days
    years = days // 365
    months = (days % 365) // 30
    snapshot["sobriety"] = {
        "days": days,
        "label": f"{years} years, {months} months sober (since Nov 27, 2023)"
    }

    try:
        with _weaviate() as h:

            # --- Recent mood (last 5 daily notes) ---
            notes = h._generic_fetch("DailyNote", limit=5)
            mood_entries = []
            for n in notes:
                mood_entries.append({
                    "date": n.get("date", ""),
                    "content": n.get("content", "")[:200],
                    "note_type": n.get("note_type", "note"),
                })
            snapshot["recent_mood"] = mood_entries

            # --- Last 3 Claude Code sessions ---
            recent_sessions = h.search_conversations(
                "Claude Code session Shane built worked on", mode="CODE", limit=3
            )
            session_summaries = []
            for s in recent_sessions:
                msg = s.get("message", "")
                # Extract first line (the session header) + first request
                lines = [l.strip() for l in msg.split("\n") if l.strip()]
                session_summaries.append({
                    "date": lines[0] if lines else "",
                    "preview": " | ".join(lines[2:5]) if len(lines) > 2 else msg[:200],
                    "session_id": s.get("session_id", ""),
                })
            snapshot["recent_sessions"] = session_summaries

            # --- Active projects from knowledge ---
            projects = h.search_knowledge("active project building working on", limit=5)
            snapshot["active_projects"] = [
                {
                    "title": p.get("title", "")[:80],
                    "content": p.get("content", "")[:200],
                    "source": p.get("source", ""),
                }
                for p in projects
            ]

            # --- Shane profile (hardcoded — near_text search over PersonalDoc is unsafe
            # because credential docs match semantically and must never leak into snapshots) ---
            snapshot["shane_profile"] = (
                "Shane Brazelton — SRM Concrete dispatcher, Hazel Green AL. "
                "Faith, family, sobriety (since Nov 27 2023), local AI. "
                "Building for the ~800M people Big Tech is about to leave behind. "
                "Wife Tiffany (home recovering — Chiari malformation + shunts). "
                "Sons: Gavin (m. Angel), Kai, Pierce, Jaxton, Ryker (5)."
            )

            # --- Family state from recent notes ---
            family_notes = h.search_conversations("Tiffany family health surgery kids", limit=2)
            snapshot["family_context"] = [
                n.get("message", "")[:200] for n in family_notes
            ]

    except Exception as e:
        snapshot["error"] = f"Partial snapshot — Weaviate issue: {e}"

    snapshot["instructions"] = (
        "Use this snapshot to walk into the session knowing Shane. "
        "Reference sobriety milestone if relevant. Check recent_sessions "
        "to avoid re-doing work. Check family_context before discussing Tiffany."
    )

    return json.dumps(snapshot, default=str, indent=2)


# ===========================================================================
# GROUP 16: Weaviate Session Tools — 2 tools
# weaviate_log_conversation: store a full session transcript
# weaviate_get_context: retrieve recent sessions for CLAUDE.md injection
# ===========================================================================

class WeaviateLogConversationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    session_transcript: str = Field(..., description="Full session transcript to store", min_length=1)
    source: str = Field(default="claude.ai", description="Origin of the session: 'claude.ai', 'claude-code', etc.")


@mcp.tool(
    name="weaviate_log_conversation",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def weaviate_log_conversation(params: WeaviateLogConversationInput) -> str:
    """Log a full session transcript to Weaviate's Conversation collection.

    Vectorized via text2vec-transformers (MiniLM-L6-v2). Stores transcript, source,
    timestamp, and 200-char summary. Designed to be called from claude.ai via MCP
    at session end.
    """
    try:
        with _weaviate() as h:
            now = datetime.now(timezone.utc)
            data = {
                "message": params.session_transcript,
                "role": "transcript",
                "mode": "TRANSCRIPT",
                "session_id": f"session-{now.strftime('%Y%m%dT%H%M%S')}",
                "timestamp": now.isoformat(),
                "source": params.source,
                "summary": params.session_transcript[:200],
            }
            uid = h._generic_insert("Conversation", data)
            if uid:
                return json.dumps({"success": True, "uuid": uid, "summary": data["summary"]})
            return json.dumps({"success": False, "error": "Conversation collection may not exist."})
    except Exception as e:
        return _format_error(e, "weaviate_log_conversation")


@mcp.tool(
    name="weaviate_get_context",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def weaviate_get_context() -> str:
    """Return recent session context formatted for CLAUDE.md injection.

    Fetches the 5 most recent Conversation entries (newest first) plus
    the context snapshot. Output is plain text suitable for direct append
    to CLAUDE.md at session start.
    """
    try:
        from weaviate.classes.query import Sort

        lines = ["=== RECENT SESSION CONTEXT ===\n"]

        with _weaviate() as h:
            if h.collection_exists("Conversation"):
                col = h.client.collections.get("Conversation")
                response = col.query.fetch_objects(
                    sort=Sort.by_creation_time(ascending=False),
                    limit=5,
                )
                for obj in response.objects:
                    props = obj.properties
                    ts = props.get("timestamp", "")
                    source = props.get("source") or props.get("mode") or props.get("role") or "unknown"
                    summary = props.get("summary") or (props.get("message") or "")[:300]
                    lines.append(f"[{ts}] [{source}]\n{summary}\n\n")

        # Append context snapshot if available
        try:
            lines.append("=== CONTEXT SNAPSHOT ===\n")
            lines.append(shanebrain_context_snapshot())
        except Exception:
            pass

        return "".join(lines)
    except Exception as e:
        return _format_error(e, "weaviate_get_context")


# ===========================================================================
# MCP App resource — Daily Briefing UI
# ===========================================================================

BRIEFING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ShaneBrain Daily Briefing</title>
<style>
  :root{--bg:#0a0e1a;--cyan:#00f0ff;--magenta:#ff006e;--text:#e8eef5;--dim:#7a8a9a;--green:#00ff88;--red:#ff3355;--yellow:#ffd700}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Courier New',monospace;height:100vh;overflow:hidden;display:flex;align-items:center;justify-content:center}
  #panel{width:100%;max-width:640px;max-height:600px;padding:24px;border:1px solid var(--cyan);box-shadow:0 0 20px rgba(0,240,255,.15);display:flex;flex-direction:column;gap:16px}
  .headline{text-align:center;font-size:2.6rem;font-weight:700;color:var(--cyan);letter-spacing:4px;text-shadow:0 0 12px var(--cyan)}
  .sub{text-align:center;color:var(--dim);font-size:.75rem;letter-spacing:2px;text-transform:uppercase}
  .divider{border:none;border-top:1px solid rgba(0,240,255,.2)}
  .row{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
  .label{color:var(--dim);font-size:.7rem;text-transform:uppercase;letter-spacing:1px;white-space:nowrap}
  .value{color:var(--text);font-size:.85rem;text-align:right}
  .verse-block{border-left:2px solid var(--magenta);padding-left:10px}
  .verse-ref{color:var(--magenta);font-size:.7rem;letter-spacing:1px;margin-bottom:3px}
  .verse-text{color:var(--text);font-size:.8rem;font-style:italic;line-height:1.5}
  .cluster{display:flex;flex-wrap:wrap;gap:6px}
  .node{display:flex;align-items:center;gap:4px;font-size:.7rem;color:var(--dim)}
  .dot{width:7px;height:7px;border-radius:50%}
  .dot-green{background:var(--green);box-shadow:0 0 6px var(--green)}
  .dot-red{background:var(--red);box-shadow:0 0 6px var(--red)}
  .dot-unknown{background:var(--yellow)}
  .thoughts{display:flex;flex-direction:column;gap:6px;overflow:hidden}
  .thought{font-size:.75rem;color:var(--dim);border-left:1px solid rgba(255,0,110,.3);padding-left:8px;line-height:1.4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .loading{text-align:center;color:var(--dim);font-size:.8rem;letter-spacing:2px;animation:pulse 1.4s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
</style>
</head>
<body>
<div id="panel">
  <div class="loading" id="loading">INITIALIZING SHANEBRAIN...</div>
  <div id="content" style="display:none;display:flex;flex-direction:column;gap:16px"></div>
</div>
<script>
(function(){
  var buf=null, initialized=false;

  function render(d){
    var el=document.getElementById('content');
    el.style.display='flex';
    document.getElementById('loading').style.display='none';
    var wdays=(d.honeymoon_days||0)+' days in';
    var cluster='';
    if(d.cluster){Object.keys(d.cluster).forEach(function(n){
      var s=d.cluster[n];
      cluster+='<span class="node"><span class="dot dot-'+(s==='green'?'green':'red')+'"></span>'+n+'</span>';
    });}
    var thoughts='';
    if(d.thoughts&&d.thoughts.length){d.thoughts.forEach(function(t){
      thoughts+='<div class="thought">'+(t.content||'').slice(0,90)+'</div>';
    });}else{thoughts='<div class="thought" style="font-style:italic">No thoughts yet.</div>';}
    el.innerHTML=
      '<div class="headline">'+d.sober_days+'</div>'+
      '<div class="sub">days sober &nbsp;&#9679;&nbsp; '+d.today+'</div>'+
      '<hr class="divider"/>'+
      '<div class="verse-block">'+
        '<div class="verse-ref">'+d.verse_ref+'</div>'+
        '<div class="verse-text">'+d.verse_text+'</div>'+
      '</div>'+
      '<hr class="divider"/>'+
      '<div class="row"><span class="label">Book II words</span><span class="value">'+d.book2_words.toLocaleString()+'</span></div>'+
      '<div class="row"><span class="label">Gavin &amp; Angel Honeymoon</span><span class="value">'+wdays+'</span></div>'+
      '<hr class="divider"/>'+
      '<div class="label" style="margin-bottom:4px">Cluster</div>'+
      '<div class="cluster">'+cluster+'</div>'+
      (d.thoughts&&d.thoughts.length?'<hr class="divider"/><div class="label" style="margin-bottom:4px">Recent thoughts</div><div class="thoughts">'+thoughts+'</div>':'');
  }

  function processResult(payload){
    try{
      var data=typeof payload==='string'?JSON.parse(payload):payload;
      if(data.sober_days!==undefined){render(data);return;}
      if(data.content&&Array.isArray(data.content)){
        for(var i=0;i<data.content.length;i++){
          if(data.content[i].type==='text'){render(JSON.parse(data.content[i].text));return;}
        }
      }
    }catch(e){document.getElementById('loading').textContent='Error: '+e.message;}
  }

  window.addEventListener('message',function(ev){
    if(ev.source!==window.parent)return;
    var msg=ev.data;
    if(!msg||!msg.method)return;
    if(msg.method==='ui/initialize'){
      initialized=true;
      window.parent.postMessage({jsonrpc:'2.0',method:'ui/notifications/initialized',params:{}}, '*');
      if(buf!==null){processResult(buf);buf=null;}
    }
    if(msg.method==='ui/notifications/tool-result'){
      if(!initialized){buf=msg.params;return;}
      processResult(msg.params);
    }
  });

  // handshake kick-off
  window.parent.postMessage({jsonrpc:'2.0',method:'ui/initialize',params:{
    appInfo:{name:'ShaneBrain Daily Briefing',version:'1.0.0'},
    appCapabilities:{},
    protocolVersion:'2026-01-26'
  }}, '*');
})();
</script>
</body>
</html>"""


@mcp.resource("ui://shanebrain/briefing", mime_type="text/html;profile=mcp-app")
def briefing_ui() -> str:
    """MCP App UI resource for the daily briefing."""
    return BRIEFING_HTML


# ===========================================================================
# GROUP 17: Pulsar Sentinel — 2 tools
# ===========================================================================

SENTINEL_BASE = os.environ.get("SENTINEL_URL", "http://localhost:8250")
SENTINEL_SERVICE_KEY = os.environ.get("PULSAR_SERVICE_KEY", "shanebrain-internal-2026")


@mcp.tool(
    name="shanebrain_sentinel_health",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_sentinel_health() -> str:
    """Check Pulsar Sentinel is alive. No auth required. Use for uptime monitoring.
    Returns: status (healthy/down), pqc_available (bool — real ML-KEM loaded), timestamp."""
    try:
        resp = requests.get(f"{SENTINEL_BASE}/api/v1/health", timeout=5)
        resp.raise_for_status()
        return json.dumps(resp.json(), default=str)
    except requests.exceptions.ConnectionError:
        return json.dumps({"status": "down", "error": "connection refused — is pulsar-sentinel service running?"})
    except Exception as e:
        return _format_error(e, "shanebrain_sentinel_health")


@mcp.tool(
    name="shanebrain_sentinel_status",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def shanebrain_sentinel_status() -> str:
    """Full Pulsar Sentinel status: pqc_available, user role, tier, trust score (PTS).
    Uses internal service key — no MetaMask needed. Flag pqc_available=false in daily briefing."""
    try:
        headers = {"Authorization": f"Bearer {SENTINEL_SERVICE_KEY}"}
        resp = requests.get(f"{SENTINEL_BASE}/api/v1/status", headers=headers, timeout=5)
        resp.raise_for_status()
        return json.dumps(resp.json(), default=str)
    except requests.exceptions.ConnectionError:
        return json.dumps({"status": "down", "error": "connection refused — is pulsar-sentinel service running?"})
    except Exception as e:
        return _format_error(e, "shanebrain_sentinel_status")


# ===========================================================================
# Node Bus — cross-session, cross-node messaging via shared SQLite
# ===========================================================================

class NodePostInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_node: str = Field(..., description="Sender node name (e.g. 'shanebrain', 'biloxi')")
    content: str = Field(..., description="Message content")
    to_node: str = Field("all", description="Recipient node name or 'all'")
    tags: Optional[str] = Field(None, description="Optional comma-separated tags")


class NodeReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to_node: Optional[str] = Field(None, description="Filter by recipient ('all' or specific node). Omit to read everything.")
    limit: int = Field(30, ge=1, le=100, description="Max messages to return (newest first)")
    since_id: Optional[int] = Field(None, description="Only return messages with id > since_id")


@mcp.tool(
    description="Post a message to the cross-node session bus. Any Claude Code session on any cluster node can read it.",
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
def shanebrain_node_post(params: NodePostInput) -> str:
    try:
        conn = _bus_conn()
        ts = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO messages (from_node, to_node, content, tags, ts) VALUES (?,?,?,?,?)",
            (params.from_node, params.to_node, params.content, params.tags, ts),
        )
        conn.commit()
        msg_id = cur.lastrowid
        conn.close()

        # Fire Discord webhook — non-blocking, best-effort
        try:
            to_label = f"→ **{params.to_node}**" if params.to_node != "all" else "→ **all**"
            discord_body = json.dumps({
                "username": params.from_node,
                "content": f"{to_label}  `#{msg_id}`\n{params.content}"
            }).encode()
            req = urllib.request.Request(
                NODE_BUS_WEBHOOK,
                data=discord_body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "DiscordBot (https://shanebrain.local, 1.0)",
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass  # webhook failure never breaks the bus

        return json.dumps({"ok": True, "id": msg_id, "ts": ts})
    except Exception as e:
        return _format_error(e, "shanebrain_node_post")


@mcp.tool(
    description="Read messages from the cross-node session bus. Filter by recipient or read all.",
    annotations={"readOnlyHint": True},
)
def shanebrain_node_read(params: NodeReadInput) -> str:
    try:
        conn = _bus_conn()
        clauses, args = [], []
        if params.to_node:
            clauses.append("(to_node = ? OR to_node = 'all')")
            args.append(params.to_node)
        if params.since_id is not None:
            clauses.append("id > ?")
            args.append(params.since_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(params.limit)
        rows = conn.execute(
            f"SELECT id, from_node, to_node, content, tags, ts FROM messages {where} ORDER BY id DESC LIMIT ?",
            args,
        ).fetchall()
        conn.close()
        messages = [
            {"id": r[0], "from": r[1], "to": r[2], "content": r[3], "tags": r[4], "ts": r[5]}
            for r in rows
        ]
        return json.dumps({"messages": messages, "count": len(messages)}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_node_read")


@mcp.tool(
    description="Show which cluster nodes have posted to the session bus and when — live heartbeat view.",
    annotations={"readOnlyHint": True},
)
def shanebrain_node_status() -> str:
    try:
        conn = _bus_conn()
        rows = conn.execute(
            "SELECT from_node, MAX(ts) as last_seen, COUNT(*) as posts FROM messages GROUP BY from_node ORDER BY last_seen DESC"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()
        nodes = [{"node": r[0], "last_seen": r[1], "posts": r[2]} for r in rows]
        return json.dumps({"active_nodes": nodes, "total_messages": total}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_node_status")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import warnings

    # Suppress benign ResourceWarning from anyio/MCP framework
    warnings.filterwarnings("ignore", category=ResourceWarning, module="anyio")

    parser = argparse.ArgumentParser(description="ShaneBrain MCP Server v2.5")
    parser.add_argument("--transport", choices=["sse", "streamable-http", "stdio"], default="streamable-http")
    args = parser.parse_args()

    logger.info("Starting ShaneBrain MCP v2.6 | transport=%s | port=%s | 42 tools", args.transport, MCP_PORT)

    mcp.run(transport=args.transport)
