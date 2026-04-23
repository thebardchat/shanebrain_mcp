#!/usr/bin/env python3
"""
Complete Weaviate Setup for ShaneBrain Core
Orchestrates all Weaviate setup operations in the correct order.

Usage:
    python scripts/setup_all_weaviate.py              # Full setup
    python scripts/setup_all_weaviate.py --no-import  # Skip RAG import
    python scripts/setup_all_weaviate.py --test       # Run tests after setup
"""

import subprocess
import sys
import time
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def run_script(script_name: str, args: list = None, description: str = None) -> bool:
    """Run a Python script and return success status."""
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"{RED}[X] Script not found: {script_name}{RESET}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    desc = description or script_name
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}Running: {desc}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"{RED}Error running {script_name}: {e}{RESET}")
        return False


def check_weaviate_running() -> bool:
    """Check if Weaviate is responding."""
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:8080/v1/.well-known/ready", timeout=5)
        return req.status == 200
    except:
        return False


def wait_for_weaviate(max_wait: int = 60) -> bool:
    """Wait for Weaviate to be ready."""
    print(f"\n{BLUE}Checking Weaviate status...{RESET}")

    if check_weaviate_running():
        print(f"{GREEN}[OK] Weaviate is already running{RESET}")
        return True

    print(f"{YELLOW}Weaviate not responding. Waiting up to {max_wait} seconds...{RESET}")

    for i in range(max_wait):
        if check_weaviate_running():
            print(f"\n{GREEN}[OK] Weaviate is now ready!{RESET}")
            return True
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"  Still waiting... ({i + 1}s)")

    print(f"\n{RED}[X] Weaviate did not become ready within {max_wait} seconds{RESET}")
    return False


def find_rag_file() -> str:
    """Find the RAG.md file."""
    search_paths = [
        Path(__file__).parent.parent / "RAG.md",
        Path.cwd() / "RAG.md",
        Path(__file__).parent.parent / "data" / "RAG.md",
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    return "RAG.md"  # Default, let import script handle error


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Complete Weaviate setup for ShaneBrain')
    parser.add_argument('--no-import', action='store_true', help='Skip RAG.md import')
    parser.add_argument('--no-json-schemas', action='store_true', help='Skip loading JSON schemas')
    parser.add_argument('--test', action='store_true', help='Run integration tests after setup')
    parser.add_argument('--clear', action='store_true', help='Clear existing RAG data before import')
    args = parser.parse_args()

    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}     ShaneBrain Core - Complete Weaviate Setup{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    # Track results
    steps = []

    # Step 1: Check Weaviate is running
    if not wait_for_weaviate():
        print(f"\n{RED}[ABORT] Weaviate must be running first{RESET}")
        print(f"\nStart Weaviate with:")
        print(f"  cd weaviate-config && docker-compose up -d")
        print(f"\nOr on Windows:")
        print(f"  START-SHANEBRAIN.bat")
        return 1

    # Step 2: Setup core schema (Conversation, LegacyKnowledge, CrisisLog)
    success = run_script(
        "setup_weaviate_schema.py",
        description="Core Schema Setup (Conversation, LegacyKnowledge, CrisisLog)"
    )
    steps.append(("Core Schema Setup", success))

    # Step 3: Load JSON schemas (ShanebrainMemory, AngelCloudConversation, PulsarSecurityEvent)
    if not args.no_json_schemas:
        success = run_script(
            "load_json_schemas.py",
            description="JSON Schema Loading (Memory, AngelCloud, Pulsar)"
        )
        steps.append(("JSON Schema Loading", success))
    else:
        steps.append(("JSON Schema Loading", None))  # Skipped

    # Step 4: Import RAG.md
    if not args.no_import:
        rag_file = find_rag_file()
        import_args = [rag_file]
        if args.clear:
            import_args.append("--clear")

        success = run_script(
            "import_rag_to_weaviate.py",
            args=import_args,
            description=f"RAG.md Import ({rag_file})"
        )
        steps.append(("RAG.md Import", success))
    else:
        steps.append(("RAG.md Import", None))  # Skipped

    # Step 5: Verify setup
    success = run_script(
        "verify_weaviate.py",
        description="Verification"
    )
    steps.append(("Verification", success))

    # Step 6: Run tests (optional)
    if args.test:
        success = run_script(
            "test_weaviate_integration.py",
            description="Integration Tests"
        )
        steps.append(("Integration Tests", success))

    # Final Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}Setup Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    all_passed = True
    for step_name, result in steps:
        if result is None:
            print(f"  {YELLOW}[-]{RESET} {step_name} (skipped)")
        elif result:
            print(f"  {GREEN}[OK]{RESET} {step_name}")
        else:
            print(f"  {RED}[X]{RESET} {step_name}")
            all_passed = False

    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_passed:
        print(f"{GREEN}[SUCCESS] Weaviate setup complete!{RESET}")
        print(f"\nNext steps:")
        print(f"  • Query knowledge: python scripts/query_legacy.py")
        print(f"  • Run agent: python langchain-chains/shanebrain_agent.py")
    else:
        print(f"{YELLOW}[WARNING] Some steps failed - check output above{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
