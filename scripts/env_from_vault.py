#!/usr/bin/env python3
"""
Rebuild .env from vault secrets + .env.template defaults.

Pulls all ENV_SECRET entries from the PersonalDoc vault (category=credentials)
and merges them into the .env.template to produce a working .env file.

Usage:
    python scripts/env_from_vault.py              # rebuild .env
    python scripts/env_from_vault.py --dry-run     # preview without writing
    python scripts/env_from_vault.py --diff         # show what would change

Requires: weaviate-client, python-dotenv
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import weaviate
    from weaviate.classes.query import Filter
except ImportError:
    print("ERROR: weaviate-client not installed. pip install weaviate-client")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
ENV_FILE = ROOT / ".env"
TEMPLATE_FILE = ROOT / ".env.template"
SECRET_PREFIX = "ENV_SECRET:"


def fetch_vault_secrets() -> dict:
    """Pull all ENV_SECRET entries from PersonalDoc (category=credentials)."""
    secrets = {}
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get("PersonalDoc")
        results = collection.query.fetch_objects(
            filters=Filter.by_property("category").equal("credentials"),
            limit=50,
        )
        for obj in results.objects:
            content = obj.properties.get("content", "")
            if not content.startswith(SECRET_PREFIX):
                continue
            pair = content[len(SECRET_PREFIX):]
            eq = pair.find("=")
            if eq == -1:
                continue
            key = pair[:eq].strip()
            value = pair[eq + 1:].strip()
            secrets[key] = value
    finally:
        client.close()
    return secrets


def load_template() -> str:
    """Load .env.template as base."""
    if not TEMPLATE_FILE.exists():
        print(f"ERROR: {TEMPLATE_FILE} not found")
        sys.exit(1)
    return TEMPLATE_FILE.read_text()


def load_existing_env() -> dict:
    """Load current .env key=value pairs."""
    env = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eq = line.find("=")
        if eq == -1:
            continue
        env[line[:eq].strip()] = line[eq + 1:].strip()
    return env


def build_env(template: str, secrets: dict, existing: dict) -> str:
    """
    Merge template + existing .env + vault secrets.

    Priority: vault secrets > existing .env > template defaults
    """
    # Combine existing values with vault secrets (vault wins)
    merged = {**existing, **secrets}

    lines = []
    for line in template.splitlines():
        stripped = line.strip()

        # Preserve comments and blank lines
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue

        eq = stripped.find("=")
        if eq == -1:
            lines.append(line)
            continue

        key = stripped[:eq].strip()

        if key in merged:
            lines.append(f"{key}={merged[key]}")
        else:
            lines.append(line)

    # Append keys that exist in .env/vault but not in template
    template_keys = set()
    for line in template.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            eq = stripped.find("=")
            if eq != -1:
                template_keys.add(stripped[:eq].strip())

    extra_keys = sorted(set(merged.keys()) - template_keys)
    if extra_keys:
        lines.append("")
        lines.append("# --- ADDITIONAL CONFIG (not in template) ---")
        for key in extra_keys:
            lines.append(f"{key}={merged[key]}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Rebuild .env from vault secrets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--diff", action="store_true", help="Show what would change")
    args = parser.parse_args()

    print("Fetching secrets from vault...")
    secrets = fetch_vault_secrets()
    print(f"  Found {len(secrets)} vault secrets: {', '.join(sorted(secrets.keys()))}")

    template = load_template()
    existing = load_existing_env()
    print(f"  Existing .env has {len(existing)} keys")

    output = build_env(template, secrets, existing)

    if args.diff:
        if ENV_FILE.exists():
            current = ENV_FILE.read_text()
            current_lines = current.splitlines()
            output_lines = output.splitlines()
            import difflib
            diff = difflib.unified_diff(current_lines, output_lines, lineterm="",
                                         fromfile=".env (current)", tofile=".env (rebuilt)")
            diff_text = "\n".join(diff)
            if diff_text:
                print("\nChanges:\n")
                print(diff_text)
            else:
                print("\nNo changes - .env is already in sync with vault.")
        else:
            print("\nNo existing .env - would create new file.")
        return

    if args.dry_run:
        # Redact secret values in preview
        for line in output.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                eq = stripped.find("=")
                if eq != -1:
                    key = stripped[:eq]
                    val = stripped[eq + 1:]
                    if key in secrets and len(val) > 8:
                        print(f"{key}={val[:4]}...{val[-4:]}")
                        continue
            print(line)
        return

    # Write .env
    ENV_FILE.write_text(output)
    os.chmod(ENV_FILE, 0o600)
    print(f"\n  Wrote {ENV_FILE} ({len(output.splitlines())} lines, mode 0600)")
    print("  Done.")


if __name__ == "__main__":
    main()
