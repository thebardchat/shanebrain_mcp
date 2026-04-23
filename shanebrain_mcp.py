#!/usr/bin/env python3
"""
ShaneBrain MCP Server v2.3
===========================
34 tools across 14 groups — merged from Pi deployment + GitHub quality patterns.

Groups: Knowledge (2), Chat (3), RAG Chat (1), Social (2), Vault (3),
        Notes (3), Drafts (2), Security (3), Weaviate Admin (2),
        Ollama (2), Planning (3), System (1), Email (2), Calendar (5)

Transport: SSE on port 8100 (Docker), switchable to streamable_http via --transport
Quality:   Pydantic v2 validation, MCP annotations, actionable errors, stderr logging
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

import ollama as ollama_lib
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse
from weaviate.classes.query import Filter

from health import check_gateway, check_ollama, check_weaviate
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
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "shanebrain-3b:latest")
PLANNING_DIR = Path(os.environ.get("PLANNING_DIR", "/app/planning"))
MCP_PORT = int(os.environ.get("MCP_PORT", "8100"))
RAG_CHUNK_LIMIT = 5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


def _weaviate():
    """Get a Weaviate helper as a context manager (auto-connects and closes)."""
    return DockerWeaviateHelper()


def _ollama_client():
    """Get an Ollama client with configured host."""
    return ollama_lib.Client(host=OLLAMA_HOST, timeout=600)


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
    logger.info("ShaneBrain MCP v2.3 starting — 34 tools, 14 groups")
    # Check Weaviate
    try:
        with _weaviate() as h:
            if h.is_ready():
                logger.info("Weaviate: reachable")
            else:
                logger.warning("Weaviate: NOT ready")
    except Exception as e:
        logger.warning("Weaviate: NOT reachable (%s)", e)
    # Check Ollama
    try:
        status = check_ollama()
        if status.get("status") == "ok":
            logger.info("Ollama: reachable (%d models)", len(status.get("models", [])))
        else:
            logger.warning("Ollama: NOT ready")
    except Exception as e:
        logger.warning("Ollama: NOT reachable (%s)", e)
    # Ensure planning dirs — MUST be mounted as a volume in Docker, not just a host path
    PLANNING_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("active-projects", "templates", "completed", "logs"):
        (PLANNING_DIR / sub).mkdir(exist_ok=True)
    logger.info("Planning dir: %s", PLANNING_DIR)
    if str(PLANNING_DIR).startswith("/mnt/"):
        logger.warning(
            "PLANNING_DIR is a host path (%s) — plans will be LOST on container restart "
            "unless this path is mounted via -v. See docker-compose.yml.",
            PLANNING_DIR,
        )
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
    model: str = Field(default="", description="Ollama model override (default: OLLAMA_MODEL env)")
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

            # Generate
            model = params.model or OLLAMA_MODEL
            client = _ollama_client()
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": params.message},
                ],
                options={"temperature": params.temperature, "num_predict": params.max_tokens},
                keep_alive="10m",
            )

            return json.dumps({
                "response": response["message"]["content"],
                "knowledge_chunks_used": len(chunks),
                "model": model,
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
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_daily_briefing() -> str:
    """AI-generated daily briefing summarizing recent notes via Ollama."""
    try:
        with _weaviate() as h:
            if not h.collection_exists("DailyNote"):
                return json.dumps({"error": "DailyNote collection does not exist yet."})

            notes = h._generic_fetch("DailyNote", limit=20)
            if not notes:
                return json.dumps({"briefing": "No daily notes found.", "note_count": 0})

            note_texts = []
            for n in notes:
                ntype = n.get("note_type", "note")
                content = n.get("content", "")
                date = n.get("date", "")
                note_texts.append(f"[{date} - {ntype}] {content}")

            client = _ollama_client()
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are ShaneBrain. Summarize these daily notes into a brief daily briefing. Be concise — bullet points preferred."},
                    {"role": "user", "content": f"Here are recent notes:\n\n" + "\n".join(note_texts) + "\n\nGive me a daily briefing."},
                ],
                options={"temperature": 0.3, "num_predict": 100},
                keep_alive="10m",
            )
            return json.dumps({
                "briefing": response["message"]["content"],
                "note_count": len(notes),
            })
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

            client = _ollama_client()
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Draft a {params.draft_type}: {params.prompt}"},
                ],
                options={"temperature": 0.5, "num_predict": 150},
                keep_alive="10m",
            )

            draft_text = response["message"]["content"]

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
    model: str = Field(default="", description="Ollama model name (default: OLLAMA_MODEL env)")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt", max_length=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)


@mcp.tool(
    name="shanebrain_ollama_generate",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def shanebrain_ollama_generate(params: OllamaGenerateInput) -> str:
    """Generate text using a locally-running Ollama model. Zero cloud dependency.

    For RAG-grounded answers use shanebrain_chat instead. This tool is for
    direct generation without knowledge base context.
    """
    try:
        model = params.model or OLLAMA_MODEL
        client = _ollama_client()
        response = client.generate(
            model=model,
            prompt=params.prompt,
            system=params.system_prompt or "",
            options={"temperature": params.temperature, "num_predict": params.max_tokens},
            keep_alive="10m",
        )
        text = response.get("response", "").strip()
        eval_count = response.get("eval_count", 0)
        total_ns = response.get("total_duration", 0)
        return json.dumps({
            "response": text,
            "model": model,
            "tokens": eval_count,
            "duration_s": round(total_ns / 1e9, 1) if total_ns else 0,
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
    """List all Ollama models currently downloaded on this Pi."""
    try:
        client = _ollama_client()
        data = client.list()
        models = data.get("models", [])
        if not models:
            return "No models found. Pull one: `ollama pull llama3.2:1b`"

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = ["## Local Ollama Models\n", "| Model | Size | Modified |", "|-------|------|----------|"]
            for m in models:
                name = m.get("name", "?")
                size_gb = m.get("size", 0) / 1e9
                modified = str(m.get("modified_at", ""))[:10]
                lines.append(f"| `{name}` | {size_gb:.1f} GB | {modified} |")
            return "\n".join(lines)
        return json.dumps({"models": models}, default=str)
    except Exception as e:
        return _format_error(e, "shanebrain_ollama_list_models")


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
    """Check ShaneBrain system health — Weaviate, Ollama, Gateway + all collection counts."""
    try:
        with _weaviate() as h:
            weaviate_status = check_weaviate(h)
            ollama_status = check_ollama()
            gateway_status = check_gateway()
            counts = h.get_all_collection_counts()

            return json.dumps({
                "services": {
                    "weaviate": weaviate_status,
                    "ollama": ollama_status,
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
            # Token file may be mounted read-only — refresh succeeded in memory,
            # API calls will work but next startup will refresh again.
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

    ollama_ok = check_ollama().get("status") == "ok"
    healthy = weaviate_ok and ollama_ok

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "healthy" if healthy else "degraded",
            "weaviate": "ok" if weaviate_ok else "down",
            "ollama": "ok" if ollama_ok else "down",
            "version": "2.3.0",
        },
    )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import warnings

    # Suppress benign ResourceWarning from anyio/MCP framework
    warnings.filterwarnings("ignore", category=ResourceWarning, module="anyio")

    parser = argparse.ArgumentParser(description="ShaneBrain MCP Server v2.3")
    parser.add_argument("--transport", choices=["sse", "streamable-http", "stdio"], default="streamable-http")
    args = parser.parse_args()

    logger.info("Starting ShaneBrain MCP v2.3 | transport=%s | port=%s | 34 tools", args.transport, MCP_PORT)

    mcp.run(transport=args.transport)
