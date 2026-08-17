FROM python:3.13-slim

WORKDIR /app

# arp-scan — powers shanebrain_network_scan (LAN recon tool)
RUN apt-get update && apt-get install -y --no-install-recommends arp-scan \
 && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy weaviate_helpers from scripts/ (reused via subclass)
COPY scripts/weaviate_helpers.py /app/scripts/weaviate_helpers.py
RUN touch /app/scripts/__init__.py

# Copy MCP server code
COPY shanebrain_mcp.py .
COPY weaviate_bridge.py .
COPY health.py .

EXPOSE 8100

CMD ["python", "shanebrain_mcp.py"]
