FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: Production deployment builds from the shanebrain-core root context
# because weaviate_helpers.py is shared from scripts/.
# This standalone Dockerfile expects you to copy weaviate_helpers.py manually:
#
#   cp /mnt/shanebrain-raid/shanebrain-core/scripts/weaviate_helpers.py scripts/
#
# Or build from shanebrain-core root with the production Dockerfile that does:
#   COPY scripts/weaviate_helpers.py /app/scripts/weaviate_helpers.py

# Copy weaviate_helpers (must exist at scripts/weaviate_helpers.py)
COPY scripts/weaviate_helpers.py /app/scripts/weaviate_helpers.py
RUN touch /app/scripts/__init__.py

# Copy MCP server code
COPY shanebrain_mcp.py server.py
COPY weaviate_bridge.py .
COPY health.py .

EXPOSE 8100

CMD ["python", "server.py"]
