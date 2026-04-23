#!/usr/bin/env python3
"""
Weaviate Helper Functions for ShaneBrain Core
Provides reusable functions for interacting with Weaviate collections.
Compatible with weaviate-client v4.
"""

import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import uuid


class WeaviateHelper:
    """Helper class for Weaviate operations in ShaneBrain."""

    def __init__(self, url: str = "localhost:8080"):
        """Initialize connection to Weaviate."""
        self._client = None
        self._url = url

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Establish connection to Weaviate."""
        if self._client is None:
            self._client = weaviate.connect_to_local()
        return self._client

    def close(self):
        """Close the Weaviate connection."""
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self):
        """Get the Weaviate client, connecting if necessary."""
        if self._client is None:
            self.connect()
        return self._client

    def is_ready(self) -> bool:
        """Check if Weaviate is ready."""
        try:
            return self.client.is_ready()
        except Exception:
            return False

    # =========================================================================
    # Conversation Operations
    # =========================================================================

    def log_conversation(
        self,
        message: str,
        role: str,
        mode: str = "CHAT",
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[str]:
        """
        Log a conversation message to Weaviate.

        Args:
            message: The message content
            role: Role (user, assistant, system)
            mode: Agent mode (CHAT, MEMORY, WELLNESS, SECURITY, DISPATCH, CODE)
            session_id: Session identifier (generated if not provided)
            timestamp: Message timestamp (now if not provided)

        Returns:
            The UUID of the created object, or None on error
        """
        if not self.client.collections.exists("Conversation"):
            return None

        collection = self.client.collections.get("Conversation")

        data = {
            "message": message,
            "role": role,
            "mode": mode,
            "session_id": session_id or str(uuid.uuid4()),
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat()
        }

        try:
            result = collection.data.insert(data)
            return str(result)
        except Exception as e:
            print(f"Error logging conversation: {e}")
            return None

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: The session identifier
            limit: Maximum messages to return

        Returns:
            List of conversation messages
        """
        if not self.client.collections.exists("Conversation"):
            return []

        collection = self.client.collections.get("Conversation")

        try:
            response = collection.query.fetch_objects(
                filters=Filter.by_property("session_id").equal(session_id),
                limit=limit
            )
            return [obj.properties for obj in response.objects]
        except Exception as e:
            print(f"Warning: Failed to get conversation history: {e}")
            return []

    def search_conversations(
        self,
        query: str,
        mode: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversations semantically.

        Args:
            query: Search query
            mode: Optional mode filter
            limit: Maximum results

        Returns:
            List of matching messages
        """
        if not self.client.collections.exists("Conversation"):
            return []

        collection = self.client.collections.get("Conversation")

        try:
            if mode:
                response = collection.query.near_text(
                    query=query,
                    filters=Filter.by_property("mode").equal(mode),
                    limit=limit
                )
            else:
                response = collection.query.near_text(
                    query=query,
                    limit=limit
                )
            return [obj.properties for obj in response.objects]
        except Exception as e:
            print(f"Warning: Failed to search conversations: {e}")
            return []

    # =========================================================================
    # Legacy Knowledge Operations
    # =========================================================================

    def add_knowledge(
        self,
        content: str,
        category: str,
        source: str = "manual",
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Add knowledge to LegacyKnowledge collection.

        Args:
            content: The knowledge content
            category: Category (family, faith, technical, philosophy, general, wellness)
            source: Source of the knowledge
            title: Optional title

        Returns:
            UUID of created object or None
        """
        if not self.client.collections.exists("LegacyKnowledge"):
            return None

        collection = self.client.collections.get("LegacyKnowledge")

        data = {
            "content": content,
            "category": category,
            "source": source,
        }
        if title:
            data["title"] = title

        try:
            result = collection.data.insert(data)
            return str(result)
        except Exception as e:
            print(f"Error adding knowledge: {e}")
            return None

    def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search legacy knowledge semantically.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of matching knowledge entries
        """
        if not self.client.collections.exists("LegacyKnowledge"):
            return []

        collection = self.client.collections.get("LegacyKnowledge")

        try:
            if category:
                response = collection.query.near_text(
                    query=query,
                    filters=Filter.by_property("category").equal(category),
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                )
            else:
                response = collection.query.near_text(
                    query=query,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                )

            results = []
            for obj in response.objects:
                entry = obj.properties.copy()
                if obj.metadata and obj.metadata.distance is not None:
                    entry["_distance"] = obj.metadata.distance
                results.append(entry)
            return results
        except Exception as e:
            print(f"Warning: Failed to search legacy knowledge: {e}")
            return []

    # =========================================================================
    # Crisis Log Operations
    # =========================================================================

    def log_crisis(
        self,
        input_text: str,
        severity: str,
        session_id: Optional[str] = None,
        response_given: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[str]:
        """
        Log a crisis detection event.

        Args:
            input_text: The triggering user input
            severity: Severity level (low, medium, high, critical)
            session_id: Session identifier
            response_given: The response provided to the user
            timestamp: Event timestamp

        Returns:
            UUID of created log or None
        """
        if not self.client.collections.exists("CrisisLog"):
            return None

        collection = self.client.collections.get("CrisisLog")

        data = {
            "input_text": input_text,
            "severity": severity,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat()
        }
        if session_id:
            data["session_id"] = session_id
        if response_given:
            data["response_given"] = response_given

        try:
            result = collection.data.insert(data)
            return str(result)
        except Exception as e:
            print(f"Error logging crisis: {e}")
            return None

    def get_recent_crises(
        self,
        severity: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent crisis logs.

        Args:
            severity: Optional severity filter
            limit: Maximum results

        Returns:
            List of crisis log entries
        """
        if not self.client.collections.exists("CrisisLog"):
            return []

        collection = self.client.collections.get("CrisisLog")

        try:
            if severity:
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("severity").equal(severity),
                    limit=limit
                )
            else:
                response = collection.query.fetch_objects(limit=limit)
            return [obj.properties for obj in response.objects]
        except Exception as e:
            print(f"Warning: Failed to get crisis logs: {e}")
            return []

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of objects in a collection."""
        if not self.client.collections.exists(collection_name):
            return 0

        try:
            collection = self.client.collections.get(collection_name)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count
        except Exception as e:
            print(f"Warning: Failed to get collection count for {collection_name}: {e}")
            return 0

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        return self.client.collections.exists(name)


# =========================================================================
# CLI Demo
# =========================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("     ShaneBrain Weaviate Helper Demo")
    print("="*60 + "\n")

    with WeaviateHelper() as helper:
        if not helper.is_ready():
            print("Weaviate is not ready. Make sure it's running.")
            sys.exit(1)

        print("✓ Connected to Weaviate\n")

        # Show collection counts
        print("Collection counts:")
        for name in ['Conversation', 'LegacyKnowledge', 'CrisisLog']:
            count = helper.get_collection_count(name)
            exists = helper.collection_exists(name)
            status = f"{count} records" if exists else "not created"
            print(f"  • {name}: {status}")

        # Demo: Search knowledge
        print("\nDemo: Searching LegacyKnowledge for 'family'...")
        results = helper.search_knowledge("family", limit=2)
        if results:
            for r in results:
                title = r.get('title', 'Untitled')[:40]
                dist = r.get('_distance', 'N/A')
                print(f"  • {title} (distance: {dist})")
        else:
            print("  (no results or collection not set up)")

    print("\n" + "="*60 + "\n")
