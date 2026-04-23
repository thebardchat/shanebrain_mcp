#!/usr/bin/env python3
"""
gcal_refresh.py — Refresh Google Calendar cache for the dashboard.

Two modes:
1. OAuth token mode (preferred): Uses gcal_token.json for direct API access
2. Fallback: Reads from gcal_cache.json (seeded by external tools like Claude MCP)

Run via cron every 15 minutes:
  */15 * * * * python3 /mnt/shanebrain-raid/shanebrain-core/scripts/gcal_refresh.py
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCRIPTS_DIR = Path("/mnt/shanebrain-raid/shanebrain-core/scripts")
TOKEN_FILE = SCRIPTS_DIR / "gcal_token.json"
CACHE_FILE = SCRIPTS_DIR / "gcal_cache.json"


def fetch_via_oauth():
    """Fetch events using stored OAuth token."""
    if not TOKEN_FILE.exists():
        return None

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    token_data = json.loads(TOKEN_FILE.read_text())
    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri"),
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes"),
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_data["token"] = creds.token
        TOKEN_FILE.write_text(json.dumps(token_data, indent=2))

    service = build("calendar", "v3", credentials=creds)
    now = datetime.now(timezone.utc).isoformat()
    week_later = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

    result = service.events().list(
        calendarId="primary",
        timeMin=now,
        timeMax=week_later,
        maxResults=20,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = []
    for ev in result.get("items", []):
        start = ev.get("start", {})
        dt_str = start.get("dateTime", start.get("date", ""))
        all_day = "date" in start and "dateTime" not in start

        time_str = ""
        date_str = ""
        day_name = ""
        if dt_str:
            try:
                if "T" in dt_str:
                    dt = datetime.fromisoformat(dt_str)
                    time_str = dt.strftime("%-I:%M %p")
                    date_str = dt.strftime("%b %d")
                    day_name = dt.strftime("%a")
                else:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d")
                    date_str = dt.strftime("%b %d")
                    day_name = dt.strftime("%a")
                    time_str = "ALL DAY"
            except Exception:
                pass

        events.append({
            "title": ev.get("summary", "(No title)"),
            "description": (ev.get("description") or "")[:200],
            "time": time_str,
            "date": date_str,
            "day": day_name,
            "allDay": all_day,
            "location": ev.get("location", ""),
            "link": ev.get("htmlLink", ""),
            "status": ev.get("status", ""),
        })

    return events


def main():
    events = fetch_via_oauth()
    if events is not None:
        cache = {
            "events": events,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
        print(f"Refreshed: {len(events)} events cached via OAuth")
    else:
        print("No OAuth token available. Cache not refreshed.")
        print(f"Run: python3 {SCRIPTS_DIR}/google_calendar_setup.py")


if __name__ == "__main__":
    main()
