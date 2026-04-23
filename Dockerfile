FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# weaviate_helpers.py is included in scripts/ in this repo.
# Production deploys from shanebrain-core root use the live Dockerfile which
# copies from scripts/ directly — same result either way.
COPY scripts/weaviate_helpers.py /app/scripts/weaviate_helpers.py
RUN touch /app/scripts/__init__.py

# Copy MCP server code
COPY shanebrain_mcp.py server.py
COPY weaviate_bridge.py .
COPY health.py .

# Planning dir created at runtime via PLANNING_DIR env var
# Mount the host planning-system dir as a volume — see docker-compose.yml
RUN mkdir -p /app/planning

EXPOSE 8100

CMD ["python", "server.py", "--transport", "streamable-http"]
