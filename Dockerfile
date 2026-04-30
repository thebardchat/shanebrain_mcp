FROM python:3.13-slim

WORKDIR /app

# Build context: this repo's root (standalone). For monorepo builds, run
# `docker build -f mcp-server/Dockerfile mcp-server/` so the same paths apply.

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy weaviate_helpers (subclassed by weaviate_bridge.py)
COPY scripts/weaviate_helpers.py /app/scripts/weaviate_helpers.py
RUN touch /app/scripts/__init__.py

# Copy MCP server code
COPY shanebrain_mcp.py .
COPY weaviate_bridge.py .
COPY health.py .

EXPOSE 8100

CMD ["python", "shanebrain_mcp.py"]
