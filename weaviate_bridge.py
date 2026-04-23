"""
Docker-aware WeaviateHelper subclass for MCP server.
Connects via Docker network hostname instead of localhost.
"""

import os
import weaviate
from scripts.weaviate_helpers import WeaviateHelper


class DockerWeaviateHelper(WeaviateHelper):
    """WeaviateHelper that connects via Docker network."""

    def __init__(self):
        host = os.environ.get("WEAVIATE_HOST", "localhost")
        port = int(os.environ.get("WEAVIATE_PORT", "8080"))
        grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
        self._docker_host = host
        self._docker_port = port
        self._docker_grpc_port = grpc_port
        super().__init__(url=f"{host}:{port}")

    def connect(self):
        """Connect to Weaviate via Docker network hostname."""
        if self._client is None:
            self._client = weaviate.connect_to_custom(
                http_host=self._docker_host,
                http_port=self._docker_port,
                http_secure=False,
                grpc_host=self._docker_host,
                grpc_port=self._docker_grpc_port,
                grpc_secure=False,
            )
        return self._client

    # --- Phase 3 collection helpers (generic semantic search + insert) ---

    def _generic_near_text(self, collection_name: str, query: str,
                           filters=None, limit: int = 10):
        """Semantic search on any collection."""
        if not self.client.collections.exists(collection_name):
            return []
        col = self.client.collections.get(collection_name)
        try:
            from weaviate.classes.query import MetadataQuery
            kwargs = dict(query=query, limit=limit,
                          return_metadata=MetadataQuery(distance=True))
            if filters:
                kwargs["filters"] = filters
            response = col.query.near_text(**kwargs)
            results = []
            for obj in response.objects:
                entry = obj.properties.copy()
                if obj.metadata and obj.metadata.distance is not None:
                    entry["_distance"] = obj.metadata.distance
                results.append(entry)
            return results
        except Exception:
            return []

    def _generic_insert(self, collection_name: str, data: dict):
        """Insert into any collection. Returns UUID string or None."""
        if not self.client.collections.exists(collection_name):
            return None
        col = self.client.collections.get(collection_name)
        try:
            result = col.data.insert(data)
            return str(result)
        except Exception as e:
            print(f"Error inserting into {collection_name}: {e}")
            return None

    def _generic_fetch(self, collection_name: str, filters=None, limit: int = 50):
        """Fetch objects from any collection."""
        if not self.client.collections.exists(collection_name):
            return []
        col = self.client.collections.get(collection_name)
        try:
            kwargs = dict(limit=limit)
            if filters:
                kwargs["filters"] = filters
            response = col.query.fetch_objects(**kwargs)
            return [obj.properties for obj in response.objects]
        except Exception:
            return []

    def get_all_collection_counts(self) -> dict:
        """Get object counts for all known collections."""
        names = [
            # Core collections (text2vec-ollama)
            "LegacyKnowledge", "Conversation", "FriendProfile",
            "SocialKnowledge", "CrisisLog",
            "PersonalDoc", "DailyNote", "PersonalDraft",
            "SecurityLog", "PrivacyAudit",
            # Training module collections (vectorizer: none)
            "BrainDoc", "BusinessDoc", "Document",
            "DraftTemplate", "MessageLog", "MyBrain",
        ]
        counts = {}
        for name in names:
            counts[name] = self.get_collection_count(name)
        return counts
