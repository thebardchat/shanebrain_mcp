#!/usr/bin/env python3
"""
Smoke tests for ShaneBrain MCP Server v2.0
Hits each tool via the streamable-http endpoint and verifies no crashes.

Usage:
    python3 test_smoke.py                  # default: http://localhost:8100
    python3 test_smoke.py http://host:port # custom base URL
"""

import json
import sys
import time
import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
MCP_URL = f"{BASE_URL}/mcp"
HEALTH_URL = f"{BASE_URL}/health"

# Track results
passed = 0
failed = 0
errors = []


SESSION_ID = None


def mcp_request(method: str, params: dict = None) -> dict:
    """Send a JSON-RPC request to the MCP server."""
    global SESSION_ID
    payload = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params:
        payload["params"] = params
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if SESSION_ID:
        headers["Mcp-Session-Id"] = SESSION_ID
    with httpx.Client(timeout=120.0) as client:
        r = client.post(MCP_URL, json=payload, headers=headers)
        r.raise_for_status()
        # Capture session ID from response
        if "mcp-session-id" in r.headers:
            SESSION_ID = r.headers["mcp-session-id"]
        ct = r.headers.get("content-type", "")
        if "text/event-stream" in ct:
            for line in r.text.strip().split("\n"):
                if line.startswith("data: "):
                    return json.loads(line[6:])
            return {"error": "No data in SSE response"}
        return r.json()


def initialize_session():
    """Initialize the MCP session (required before tool calls)."""
    result = mcp_request("initialize", {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "smoke-test", "version": "1.0"},
    })
    # Send initialized notification
    global SESSION_ID
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if SESSION_ID:
        headers["Mcp-Session-Id"] = SESSION_ID
    with httpx.Client(timeout=10.0) as client:
        client.post(MCP_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers)
    return result


def call_tool(name: str, args: dict) -> dict:
    """Call an MCP tool via streamable-http and return the result."""
    return mcp_request("tools/call", {"name": name, "arguments": args})


def test(name: str, args: dict, expect_key: str = None):
    """Run a single tool test."""
    global passed, failed
    try:
        result = call_tool(name, args)
        # Check for JSON-RPC error
        if "error" in result:
            errors.append(f"  FAIL {name}: JSON-RPC error: {result['error']}")
            failed += 1
            return
        # If we got a result, try to parse the tool output
        content = result.get("result", {})
        if isinstance(content, list) and content:
            text = content[0].get("text", "")
        elif isinstance(content, dict):
            text = content.get("content", [{}])[0].get("text", "") if "content" in content else str(content)
        else:
            text = str(content)

        # Check for tool-level errors
        if text and expect_key:
            try:
                parsed = json.loads(text)
                if "error" in parsed and "not exist" not in parsed.get("error", ""):
                    errors.append(f"  WARN {name}: {parsed['error']}")
            except (json.JSONDecodeError, TypeError):
                pass

        print(f"  OK   {name}")
        passed += 1
    except Exception as e:
        errors.append(f"  FAIL {name}: {type(e).__name__}: {e}")
        failed += 1


def main():
    global passed, failed
    print(f"\nShaneBrain MCP Smoke Tests — {BASE_URL}")
    print("=" * 60)

    # Health check first
    print("\n[Health Check]")
    try:
        r = httpx.get(HEALTH_URL, timeout=10)
        health = r.json()
        status = health.get("status", "unknown")
        version = health.get("version", "?")
        print(f"  OK   /health — {status} (v{version})")
        passed += 1
        if status != "healthy":
            print(f"  WARN Services degraded: {health}")
    except Exception as e:
        print(f"  FAIL /health — {e}")
        failed += 1
        print("\nServer unreachable. Aborting.")
        sys.exit(1)

    # Initialize MCP session
    print("\n[MCP Session]")
    try:
        init = initialize_session()
        server_name = init.get("result", {}).get("serverInfo", {}).get("name", "?")
        print(f"  OK   initialized (server: {server_name}, session: {SESSION_ID[:16]}...)")
        passed += 1
    except Exception as e:
        print(f"  FAIL initialize — {e}")
        failed += 1
        print("\nCannot initialize MCP session. Aborting.")
        sys.exit(1)

    # Group 1: Knowledge
    print("\n[Group 1: Knowledge]")
    test("shanebrain_search_knowledge", {"query": "family", "limit": 2})
    test("shanebrain_add_knowledge", {"content": "Smoke test entry — delete me", "category": "technical", "source": "test"})

    # Group 2: Chat
    print("\n[Group 2: Chat]")
    test("shanebrain_search_conversations", {"query": "hello", "limit": 2})
    test("shanebrain_log_conversation", {"message": "smoke test", "role": "system", "mode": "CODE"})
    test("shanebrain_get_conversation_history", {"session_id": "test-smoke-000", "limit": 5})

    # Group 3: RAG Chat (skip — slow Ollama call)
    print("\n[Group 3: RAG Chat]")
    print("  SKIP shanebrain_chat (requires Ollama inference, ~30s)")

    # Group 4: Social
    print("\n[Group 4: Social]")
    test("shanebrain_search_friends", {"query": "friend", "limit": 2})
    test("shanebrain_get_top_friends", {"limit": 3})

    # Group 5: Vault
    print("\n[Group 5: Vault]")
    test("shanebrain_vault_search", {"query": "document", "limit": 2})
    test("shanebrain_vault_add", {"content": "Smoke test vault entry", "category": "personal"})
    test("shanebrain_vault_list_categories", {"limit": 50})

    # Group 6: Notes
    print("\n[Group 6: Notes]")
    test("shanebrain_daily_note_add", {"content": "Smoke test note", "note_type": "todo"})
    test("shanebrain_daily_note_search", {"query": "test", "limit": 2})
    # Skip daily_briefing — requires Ollama
    print("  SKIP shanebrain_daily_briefing (requires Ollama inference)")

    # Group 7: Drafts
    print("\n[Group 7: Drafts]")
    test("shanebrain_draft_search", {"query": "test draft", "limit": 2})
    # Skip draft_create — requires Ollama
    print("  SKIP shanebrain_draft_create (requires Ollama inference)")

    # Group 8: Security
    print("\n[Group 8: Security]")
    test("shanebrain_security_log_search", {"query": "login", "limit": 2})
    test("shanebrain_security_log_recent", {"severity": "", "limit": 5})
    test("shanebrain_privacy_audit_search", {"query": "account", "limit": 2})

    # Group 9: Weaviate Admin
    print("\n[Group 9: Weaviate Admin]")
    test("shanebrain_rag_list_classes", {"response_format": "json"})
    # Skip delete — destructive
    print("  SKIP shanebrain_rag_delete (destructive, needs real UUID)")

    # Group 10: Ollama
    print("\n[Group 10: Ollama]")
    test("shanebrain_ollama_list_models", {"response_format": "json"})
    # Skip generate — slow
    print("  SKIP shanebrain_ollama_generate (requires inference, ~30s)")

    # Group 11: Planning
    print("\n[Group 11: Planning]")
    test("shanebrain_plan_list", {"subfolder": "active-projects"})
    test("shanebrain_plan_write", {"filename": "smoke-test.md", "content": "# Smoke Test\nThis file can be deleted.", "subfolder": "logs"})
    test("shanebrain_plan_read", {"filename": "smoke-test.md", "subfolder": "logs"})

    # Group 12: System
    print("\n[Group 12: System]")
    test("shanebrain_system_health", {})

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\nIssues:")
        for e in errors:
            print(e)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
