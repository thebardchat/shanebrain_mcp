FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY mcp-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy weaviate_helpers from scripts/ (reused via subclass)
COPY scripts/weaviate_helpers.py /app/scripts/weaviate_helpers.py
RUN touch /app/scripts/__init__.py

# Copy MCP server code
COPY mcp-server/server.py .
COPY mcp-server/weaviate_bridge.py .
COPY mcp-server/health.py .

EXPOSE 8100

CMD ["python", "server.py"]
