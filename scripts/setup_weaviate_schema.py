#!/usr/bin/env python3
"""
Weaviate Schema Setup for ShaneBrain Core
Sets up all required classes for conversations, legacy knowledge, and crisis logs.
Compatible with weaviate-client v4.
"""

import weaviate
from weaviate.classes.config import Property, DataType, Configure
import time
import sys

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def wait_for_weaviate(max_attempts=5):
    """Wait for Weaviate to be ready."""
    print(f"{BLUE}Connecting to Weaviate...{RESET}")

    for i in range(max_attempts):
        try:
            client = weaviate.connect_to_local()
            if client.is_ready():
                print(f"{GREEN}✓ Weaviate is ready!{RESET}")
                return client
        except Exception as e:
            print(f"{YELLOW}Waiting for Weaviate... attempt {i+1}/{max_attempts}{RESET}")
            if i < max_attempts - 1:
                time.sleep(3)

    print(f"{RED}✗ Could not connect to Weaviate after {max_attempts} attempts{RESET}")
    sys.exit(1)


def setup_schema(client):
    """Create all required schema classes."""

    schemas = [
        {
            "name": "Conversation",
            "description": "Chat history for ShaneBrain and Angel Cloud",
            "properties": [
                Property(name="message", data_type=DataType.TEXT, description="The message content"),
                Property(name="role", data_type=DataType.TEXT, description="Role: user, assistant, system"),
                Property(name="mode", data_type=DataType.TEXT, description="Agent mode: CHAT, MEMORY, WELLNESS, etc."),
                Property(name="timestamp", data_type=DataType.DATE, description="When the message was sent"),
                Property(name="session_id", data_type=DataType.TEXT, description="Session identifier"),
            ]
        },
        {
            "name": "LegacyKnowledge",
            "description": "Shane's RAG data and family legacy wisdom",
            "properties": [
                Property(name="content", data_type=DataType.TEXT, description="The knowledge content"),
                Property(name="category", data_type=DataType.TEXT, description="Category: family, faith, technical, philosophy, general"),
                Property(name="source", data_type=DataType.TEXT, description="Source file or origin"),
                Property(name="title", data_type=DataType.TEXT, description="Optional title or header"),
            ]
        },
        {
            "name": "CrisisLog",
            "description": "Logs for high-severity wellness detections",
            "properties": [
                Property(name="input_text", data_type=DataType.TEXT, description="The triggering input"),
                Property(name="severity", data_type=DataType.TEXT, description="Severity level: low, medium, high, critical"),
                Property(name="timestamp", data_type=DataType.DATE, description="When the crisis was detected"),
                Property(name="session_id", data_type=DataType.TEXT, description="Session identifier"),
                Property(name="response_given", data_type=DataType.TEXT, description="Response provided to user"),
            ]
        },
        {
            "name": "SocialKnowledge",
            "description": "Knowledge harvested from Facebook interactions",
            "vectorizer": Configure.Vectorizer.text2vec_ollama(
                model="llama3.2:1b",
                api_endpoint="http://host.docker.internal:11434"
            ),
            "properties": [
                Property(name="content", data_type=DataType.TEXT, description="The interaction content"),
                Property(name="author_name", data_type=DataType.TEXT, description="Name of the person"),
                Property(name="author_id", data_type=DataType.TEXT, description="Facebook user ID"),
                Property(name="interaction_type", data_type=DataType.TEXT, description="Type: comment, reaction, share, post"),
                Property(name="sentiment", data_type=DataType.TEXT, description="Sentiment: positive, negative, neutral, supportive"),
                Property(name="context", data_type=DataType.TEXT, description="Context about the interaction"),
                Property(name="source_post_id", data_type=DataType.TEXT, description="Facebook post ID this came from"),
                Property(name="timestamp", data_type=DataType.DATE, description="When the interaction occurred"),
                Property(name="relationship_tags", data_type=DataType.TEXT_ARRAY, description="Tags like friend, family, supporter"),
                Property(name="knowledge_extracted", data_type=DataType.TEXT, description="Key knowledge extracted from this interaction"),
            ]
        },
        {
            "name": "FriendProfile",
            "description": "Living profiles of people who interact on Facebook",
            "vectorizer": Configure.Vectorizer.text2vec_ollama(
                model="llama3.2:1b",
                api_endpoint="http://host.docker.internal:11434"
            ),
            "properties": [
                Property(name="name", data_type=DataType.TEXT, description="Person's display name"),
                Property(name="facebook_id", data_type=DataType.TEXT, description="Facebook user ID"),
                Property(name="interaction_count", data_type=DataType.INT, description="Total interactions"),
                Property(name="first_seen", data_type=DataType.DATE, description="First interaction timestamp"),
                Property(name="last_seen", data_type=DataType.DATE, description="Most recent interaction"),
                Property(name="sentiment_profile", data_type=DataType.TEXT, description="Overall sentiment pattern"),
                Property(name="topics_discussed", data_type=DataType.TEXT_ARRAY, description="Topics this person engages with"),
                Property(name="relationship_strength", data_type=DataType.NUMBER, description="0.0-1.0 relationship strength score"),
                Property(name="summary", data_type=DataType.TEXT, description="AI-generated summary of this person"),
            ]
        }
    ]

    print(f"\n{BLUE}=== Setting Up Schema Classes ==={RESET}\n")

    created = 0
    skipped = 0
    failed = 0

    for schema in schemas:
        class_name = schema["name"]
        try:
            # Check if class already exists
            if client.collections.exists(class_name):
                print(f"{YELLOW}○ Class {class_name} already exists - skipping{RESET}")
                skipped += 1
                continue

            # Use custom vectorizer if specified, otherwise default to text2vec-ollama
            vectorizer = schema.get("vectorizer", Configure.Vectorizer.text2vec_ollama(
                model="llama3.2:1b",
                api_endpoint="http://host.docker.internal:11434"
            ) if class_name != "CrisisLog" else None)

            client.collections.create(
                name=class_name,
                description=schema["description"],
                properties=schema["properties"],
                vectorizer_config=vectorizer
            )
            print(f"{GREEN}✓ Created class: {class_name}{RESET}")
            created += 1

        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg:
                print(f"{YELLOW}○ Class {class_name} already exists - skipping{RESET}")
                skipped += 1
            else:
                print(f"{RED}✗ Error creating {class_name}: {e}{RESET}")
                failed += 1

    print(f"\n{BLUE}=== Summary ==={RESET}")
    print(f"  Created: {created}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")

    return failed == 0


def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}     ShaneBrain Core - Weaviate Schema Setup{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    client = None
    try:
        client = wait_for_weaviate()
        success = setup_schema(client)

        print(f"\n{BLUE}{'='*60}{RESET}")
        if success:
            print(f"{GREEN}[OK] Schema setup complete!{RESET}")
        else:
            print(f"{RED}[ERROR] Some classes failed to create{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")

        return 0 if success else 1

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
