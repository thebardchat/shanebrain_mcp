# ShaneBrain MCP Tools Reference

**Version:** v2.6
**Tool count:** 32 tools across 14 groups
**Transport:** Streamable HTTP on port 8100
**Endpoints:** `/mcp` (MCP JSON-RPC), `/health` (HTTP status)

All tool names are prefixed with `shanebrain_` (or `weaviate_` for Group 14) to prevent conflicts.

Embeddings are produced server-side by Weaviate's `text2vec-transformers` vectorizer.
There is no inference layer — Claude (the consumer) does any synthesis after retrieval.

---

## Group 1: Knowledge (2 tools)

### `shanebrain_search_knowledge`

Semantic search across ShaneBrain's legacy knowledge base (LegacyKnowledge). Finds conceptually similar entries — not keyword matching. Covers family, faith, technical decisions, philosophy, and wellness knowledge.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What to search for (1–500 chars) |
| `category` | string | no | null | Filter: family, faith, technical, philosophy, general, wellness |
| `limit` | integer | no | 5 | Max results (1–50) |
| `response_format` | string | no | `json` | `json` or `markdown` |

**Returns:** JSON with `results` array and `count`, or markdown table.

---

### `shanebrain_add_knowledge`

Add an entry to ShaneBrain's legacy knowledge base (LegacyKnowledge). Stored content is auto-vectorized and becomes immediately searchable.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `content` | string | yes | — | Knowledge content (1–50000 chars) |
| `category` | string | yes | — | family, faith, technical, philosophy, general, wellness |
| `source` | string | no | `mcp` | Where this came from |
| `title` | string | no | null | Optional title |

**Returns:** JSON with `success`, `uuid`, and `preview`.

---

## Group 2: Chat (3 tools)

### `shanebrain_search_conversations`

Search past ShaneBrain conversations semantically.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What to search for (1–500 chars) |
| `mode` | string | no | null | Filter: CHAT, MEMORY, WELLNESS, SECURITY, DISPATCH, CODE |
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

### `shanebrain_log_conversation`

Log a message to ShaneBrain's conversation history.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `message` | string | yes | — | Message content |
| `role` | string | yes | — | user, assistant, or system |
| `mode` | string | no | `CHAT` | CHAT, MEMORY, WELLNESS, SECURITY, DISPATCH, CODE |
| `session_id` | string | no | null | Session ID (auto-generated if omitted) |

**Returns:** JSON with `success` and `uuid`.

---

### `shanebrain_get_conversation_history`

Get conversation history for a session by ID.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `session_id` | string | yes | — | The session identifier |
| `limit` | integer | no | 50 | Max messages (1–200) |

**Returns:** JSON with `messages` array and `count`.

---

## Group 3: Social (2 tools)

### `shanebrain_search_friends`

Search ShaneBrain's friend profiles semantically.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Name or topic to search (1–500 chars) |
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

### `shanebrain_get_top_friends`

Get friend profiles ranked by relationship strength (highest first).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

## Group 4: Vault (3 tools)

### `shanebrain_vault_search`

Search Shane's personal vault (PersonalDoc) semantically.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What to search for (1–500 chars) |
| `category` | string | no | null | Optional category filter |
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

### `shanebrain_vault_add`

Add a document to Shane's personal vault (PersonalDoc).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `content` | string | yes | — | Document content (1–50000 chars) |
| `category` | string | yes | — | medical, legal, financial, personal, work |
| `title` | string | no | null | Optional title |
| `tags` | string | no | null | Comma-separated tags |

**Returns:** JSON with `success` and `uuid`.

---

### `shanebrain_vault_list_categories`

List document counts per category in the PersonalDoc vault.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `limit` | integer | no | 100 | Max docs to scan (1–500) |

**Returns:** JSON with `categories` dict and `total` count.

---

## Group 5: Notes (2 tools)

### `shanebrain_daily_note_add`

Add a daily note (journal, todo, reminder, or reflection).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `content` | string | yes | — | Note content |
| `note_type` | string | no | `journal` | journal, todo, reminder, reflection |
| `mood` | string | no | null | Mood tag: grateful, tired, focused, anxious, etc. |

**Returns:** JSON with `success` and `uuid`.

---

### `shanebrain_daily_note_search`

Search daily notes semantically.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What to search for (1–500 chars) |
| `note_type` | string | no | null | Filter: journal, todo, reminder, reflection |
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

## Group 6: Drafts (1 tool)

### `shanebrain_draft_search`

Search saved writing drafts (PersonalDraft) semantically.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What to search for (1–500 chars) |
| `draft_type` | string | no | null | Filter: email, message, post, letter, general |
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

## Group 7: Security (3 tools)

### `shanebrain_security_log_search`

Search SecurityLog entries semantically.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What to search for (1–500 chars) |
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

### `shanebrain_security_log_recent`

Get recent security log entries (chronological, not semantic).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `severity` | string | no | `""` | Filter: low, medium, high, critical. Empty for all. |
| `limit` | integer | no | 20 | Max results (1–100) |

**Returns:** JSON with `results` array and `count`.

---

### `shanebrain_privacy_audit_search`

Search PrivacyAudit entries semantically.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What to search for (1–500 chars) |
| `limit` | integer | no | 10 | Max results (1–50) |

**Returns:** JSON with `results` array and `count`.

---

## Group 8: Weaviate Admin (2 tools)

### `shanebrain_rag_list_classes`

List all Weaviate collections with object counts. Useful for discovering data, checking health.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `response_format` | string | no | `markdown` | `json` or `markdown` |

**Returns:** Collection name -> count dict, or markdown table.

---

### `shanebrain_rag_delete`

Permanently delete a specific object from Weaviate by UUID. **Destructive — irreversible.**

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `collection_name` | string | yes | — | Weaviate collection name |
| `object_id` | string | yes | — | UUID of object to delete |

**Returns:** JSON with `success`, `deleted`, `collection`.

---

## Group 9: Planning (3 tools)

### `shanebrain_plan_list`

List markdown planning files in a planning subfolder.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `subfolder` | string | no | `active-projects` | active-projects, templates, completed, logs |

**Returns:** Markdown list of files with sizes and modification times.

---

### `shanebrain_plan_read`

Read the full content of a markdown planning file.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `filename` | string | yes | — | Filename to read (e.g. `angel-cloud-phase2.md`) |
| `subfolder` | string | no | `active-projects` | Subfolder to look in |

**Returns:** File content with metadata header. Path traversal protected.

---

### `shanebrain_plan_write`

Create or update a markdown planning file. Supports overwrite and append modes.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `filename` | string | yes | — | Filename (must end in `.md`) |
| `content` | string | yes | — | Full markdown content (1–100000 chars) |
| `subfolder` | string | no | `active-projects` | Subfolder to write in |
| `append` | boolean | no | false | Append instead of overwrite |

**Returns:** JSON with `success`, `action`, `file`, `size_kb`. Path traversal protected.

---

## Group 10: System (1 tool)

### `shanebrain_system_health`

Check ShaneBrain system health — Weaviate, Gateway + all collection counts.

**Parameters:** None

**Returns:** JSON with `services` status dict, `collections` counts, `total_objects`, `timestamp`.

---

## Group 11: Email (2 tools)

### `shanebrain_send_email`

Send an email from Shane's Gmail (brazeltonshane@gmail.com) via SMTP. Requires `GMAIL_APP_PASSWORD` environment variable.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `to` | string | yes | — | Recipient email address |
| `subject` | string | yes | — | Email subject line |
| `body` | string | yes | — | Email body (plain text or HTML) |
| `html` | boolean | no | false | Set true to send as HTML |

**Returns:** JSON with `sent`, `to`, `subject`.

---

### `shanebrain_reply_email`

Reply to an email from Shane's Gmail with proper thread headers. Requires `GMAIL_APP_PASSWORD` environment variable.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `to` | string | yes | — | Recipient email address |
| `subject` | string | yes | — | Reply subject (usually `Re: ...`) |
| `body` | string | yes | — | Reply body |
| `html` | boolean | no | false | Send as HTML |
| `in_reply_to` | string | no | null | Message-ID of email being replied to |
| `references` | string | no | null | References header for threading |

**Returns:** JSON with `sent`, `to`, `subject`, `reply: true`.

---

## Group 12: Google Calendar (5 tools)

Requires Google Calendar OAuth2 token at `GCAL_TOKEN_PATH` (default `/app/gcal_token.json`).
Token auto-refreshes when expired using the stored refresh token.

### `shanebrain_calendar_list`

List upcoming Google Calendar events for the next N days.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `days` | integer | no | 7 | Number of upcoming days to fetch (1–90) |
| `max_results` | integer | no | 20 | Max events to return (1–100) |
| `calendar_id` | string | no | null | Calendar ID (default: primary) |

**Returns:** JSON with `events` array, `count`, `days`.

---

### `shanebrain_calendar_get`

Get a specific Google Calendar event by ID.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `event_id` | string | yes | — | Google Calendar event ID |
| `calendar_id` | string | no | null | Calendar ID (default: primary) |

**Returns:** JSON with event details: id, summary, start, end, location, description, attendees, htmlLink, status.

---

### `shanebrain_calendar_create`

Create a new Google Calendar event. Auto-generates a 1-hour end time if `end` is omitted.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `summary` | string | yes | — | Event title |
| `start` | string | yes | — | Start datetime ISO 8601 (e.g. `2026-04-25T14:00:00`) or date (`2026-04-25`) |
| `end` | string | no | `""` | End datetime ISO 8601 or date. Omit for 1-hour auto-duration. |
| `description` | string | no | null | Event description / notes |
| `location` | string | no | null | Event location |
| `attendees` | list[string] | no | null | Attendee email addresses |
| `calendar_id` | string | no | null | Calendar ID (default: primary) |
| `timezone` | string | no | `America/Chicago` | Timezone for naive datetimes |

**Returns:** JSON with `created: true`, `id`, `summary`, `start`, `end`, `htmlLink`.

---

### `shanebrain_calendar_update`

Update an existing Google Calendar event. Only provided fields are changed (patch semantics).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `event_id` | string | yes | — | Google Calendar event ID |
| `summary` | string | no | null | New event title |
| `start` | string | no | null | New start datetime ISO 8601 |
| `end` | string | no | null | New end datetime ISO 8601 |
| `description` | string | no | null | New description |
| `location` | string | no | null | New location |
| `calendar_id` | string | no | null | Calendar ID (default: primary) |
| `timezone` | string | no | `America/Chicago` | Timezone for updated naive datetimes |

**Returns:** JSON with `updated: true`, `id`, `summary`, `start`, `end`, `htmlLink`.

---

### `shanebrain_calendar_delete`

Permanently delete a Google Calendar event by ID. **Destructive — irreversible.**

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `event_id` | string | yes | — | Google Calendar event ID |
| `calendar_id` | string | no | null | Calendar ID (default: primary) |

**Returns:** JSON with `deleted: true`, `event_id`.

---

## Group 13: Context Snapshot (1 tool)

### `shanebrain_context_snapshot`

Pull a rich context snapshot of Shane's current state from Weaviate. Call at the start of every session so Claude walks in knowing Shane — not cold. Returns sobriety, recent mood (last 5 daily notes), last 3 Claude Code sessions, active projects, Shane profile, and family context.

**Parameters:** None

**Returns:** JSON with `sobriety`, `recent_mood`, `recent_sessions`, `active_projects`, `shane_profile`, `family_context`, `instructions`. On Weaviate failure, returns a partial snapshot with an `error` field.

---

## Group 14: Weaviate Session Tools (2 tools)

Persist and retrieve full Claude session transcripts so memory survives context resets.

### `weaviate_log_conversation`

Log a full session transcript to Weaviate's `Conversation` collection. Vectorized via text2vec-transformers. Stores transcript, source, timestamp, and a 200-char summary. Designed to be called from claude.ai via MCP at session end.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `session_transcript` | string | yes | — | Full session transcript (min 1 char) |
| `source` | string | no | `claude.ai` | Origin: `claude.ai`, `claude-code`, etc. |

**Returns:** JSON with `success`, `uuid`, and `summary` on success; `success: false` with `error` on failure (e.g. Conversation collection missing).

---

### `weaviate_get_context`

Return recent session context formatted for CLAUDE.md injection. Fetches the 5 most recent Conversation entries (newest first) plus the context snapshot. Output is plain text suitable for direct append to CLAUDE.md at session start.

**Parameters:** None

**Returns:** Plain text — `=== RECENT SESSION CONTEXT ===` block (5 most recent transcripts with timestamp + source) followed by `=== CONTEXT SNAPSHOT ===` (output of `shanebrain_context_snapshot`).

---

## Notes

- All tools use Pydantic v2 validation with Field constraints
- All tools carry MCP annotations: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`
- Errors include actionable hints (connection errors, timeouts, not-found)
- Logging goes to stderr only — never pollutes MCP stdout
- Embeddings are produced by Weaviate's `text2vec-transformers` vectorizer; no inference dependency
- Email tools require `GMAIL_APP_PASSWORD` env var — never hardcoded
- Calendar tools require `gcal_token.json` at `GCAL_TOKEN_PATH` — run `scripts/google_calendar_setup.py` once to authenticate

---

Built by Shane Brazelton + Claude (Anthropic) · Hazel Green, Alabama
