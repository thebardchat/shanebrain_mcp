#!/usr/bin/env python3
"""
One-time Google Calendar OAuth setup — manual code paste.
Works headless over SSH.
"""
import json
import sys
import urllib.parse
from pathlib import Path

CREDS_FILE = Path(__file__).parent / "gcal_credentials.json"
TOKEN_FILE = Path(__file__).parent / "gcal_token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
REDIRECT_URI = "http://localhost:8090/"


def setup():
    if not CREDS_FILE.exists():
        print(f"ERROR: {CREDS_FILE} not found.")
        sys.exit(1)

    creds_data = json.loads(CREDS_FILE.read_text())["installed"]
    client_id = creds_data["client_id"]
    client_secret = creds_data["client_secret"]
    token_uri = creds_data["token_uri"]

    # Build auth URL manually
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)

    print("\n=== Google Calendar Authorization ===\n")
    print("1. Open this URL in your browser:\n")
    print(auth_url)
    print("\n2. Sign in and click 'Allow'")
    print("3. It will redirect to localhost and FAIL — that's OK!")
    print("4. Copy the ENTIRE URL from your browser's address bar")
    print("   (it will look like: http://localhost:8090/?code=4/xxxxx&scope=...)\n")

    redirect_url = input("Paste the full redirect URL here: ").strip()

    # Extract the code from the URL
    parsed = urllib.parse.urlparse(redirect_url)
    query = urllib.parse.parse_qs(parsed.query)
    code = query.get("code", [None])[0]

    if not code:
        print("ERROR: Could not find authorization code in that URL.")
        print("Make sure you pasted the full URL from the address bar.")
        sys.exit(1)

    # Exchange code for token
    import urllib.request
    token_data = urllib.parse.urlencode({
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode()

    req = urllib.request.Request(token_uri, data=token_data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        tokens = json.loads(resp.read())

    result = {
        "token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token"),
        "token_uri": token_uri,
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": SCOPES,
    }
    TOKEN_FILE.write_text(json.dumps(result, indent=2))
    print(f"\nToken saved to {TOKEN_FILE}")
    print("The dashboard will now show real Google Calendar events.")


if __name__ == "__main__":
    setup()
