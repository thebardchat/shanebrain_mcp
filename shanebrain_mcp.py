#!/usr/bin/env python3
"""
ShaneBrain MCP Server v2.0
===========================
27 tools across 12 groups — merged from Pi deployment + GitHub quality patterns.

Groups: Knowledge (2), Chat (3), RAG Chat (1), Social (2), Vault (3),
        Notes (3), Drafts (2), Security (3), Weaviate Admin (2),
        Ollama (2), Planning (3), System (1)

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
PLANNING_DIR = Path(os.environ.get("PLANNING_DIR", "/mnt/shanebrain-raid/shanebrain-core/planning-system"))
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
    logger.info("ShaneBrain MCP v2.0 starting — 27 tools, 12 groups")
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
            if params.severity:
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("severity").equal(params.severity),
                    limit=params.limit,
                )
            else:
                response = collection.query.fetch_objects(limit=params.limit)
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
            "version": "2.0.0",
        },
    )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShaneBrain MCP Server v2.0")
    parser.add_argument("--transport", choices=["sse", "streamable-http", "stdio"], default="streamable-http")
    args = parser.parse_args()

    logger.info("Starting ShaneBrain MCP v2.0 | transport=%s | port=%s | 27 tools", args.transport, MCP_PORT)

    mcp.run(transport=args.transport)
