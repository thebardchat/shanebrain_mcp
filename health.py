"""
Service health checks for ShaneBrain MCP server.
Checks Weaviate, Ollama, and Angel Cloud Gateway.
"""

import os
import requests


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "http://host.docker.internal:4200")


def check_weaviate(helper) -> dict:
    """Check Weaviate readiness."""
    try:
        ready = helper.is_ready()
        return {"status": "ok" if ready else "down", "service": "weaviate"}
    except Exception as e:
        return {"status": "down", "service": "weaviate", "error": str(e)}


def check_ollama() -> dict:
    """Check Ollama API."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"status": "ok", "service": "ollama", "models": models}
        return {"status": "down", "service": "ollama", "http_status": resp.status_code}
    except Exception as e:
        return {"status": "down", "service": "ollama", "error": str(e)}


def check_gateway() -> dict:
    """Check Angel Cloud Gateway."""
    try:
        resp = requests.get(f"{GATEWAY_HOST}/", timeout=5)
        if resp.status_code == 200:
            return {"status": "ok", "service": "gateway"}
        return {"status": "down", "service": "gateway", "http_status": resp.status_code}
    except Exception as e:
        return {"status": "unreachable", "service": "gateway", "error": str(e)}
