"""
Service health checks for ShaneBrain MCP server.
Checks Weaviate, Ollama, and Angel Cloud Gateway.
"""

import os
import requests


GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "http://host.docker.internal:4200")


def check_weaviate(helper) -> dict:
    """Check Weaviate readiness."""
    try:
        ready = helper.is_ready()
        return {"status": "ok" if ready else "down", "service": "weaviate"}
    except Exception as e:
        return {"status": "down", "service": "weaviate", "error": str(e)}


def check_gateway() -> dict:
    """Check Angel Cloud Gateway."""
    try:
        resp = requests.get(f"{GATEWAY_HOST}/", timeout=5)
        if resp.status_code == 200:
            return {"status": "ok", "service": "gateway"}
        return {"status": "down", "service": "gateway", "http_status": resp.status_code}
    except Exception as e:
        return {"status": "unreachable", "service": "gateway", "error": str(e)}
