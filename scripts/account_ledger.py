#!/usr/bin/env python3
"""
Account-level (cross-topic) post ledger for the single-account platforms.

Bluesky runs one account per topic, but Instagram and Threads each publish to a
SINGLE account that now spans every topic. That makes the per-topic dedup files
(topics/<name>/<platform>/bills_used.json) insufficient on their own:

  * Cross-topic dedup — a bill matched by two topics (say a measure that is both
    "healthcare" and "reproductive_rights") must not be posted twice to the same
    account.
  * Global daily cap — 13 topics each posting independently would flood one feed,
    so the account needs a ceiling on how many posts go out per day in total.

This module stores a shared ledger per platform at
account_state/<platform>/posted.json:

    {
      "posted": ["<dedup_key>", ...],   # every key published to the account
      "daily":  {"YYYY-MM-DD": <count>},# posts published per UTC date
      "last_run": "<iso8601>"
    }

The per-topic dedup files are still written as before; this ledger is an
additional, account-wide guard layered on top. Because the daily posters loop
over topics sequentially in one job, each topic's process reads the ledger the
previous topic just saved, so the cap and dedup hold across the whole run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep the daily counter from growing without bound; only recent dates matter
# for the cap, the rest is just history.
_MAX_DAILY_ENTRIES = 30


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class AccountLedger:
    """Read/modify/save the cross-topic ledger for one platform
    ("instagram" or "meta-threads")."""

    def __init__(self, platform: str):
        self.platform = platform
        self.path = ROOT / "account_state" / platform / "posted.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                d.setdefault("posted", [])
                d.setdefault("daily", {})
                return d
            except json.JSONDecodeError:
                pass
        return {"posted": [], "daily": {}}

    @property
    def seen(self) -> set[str]:
        """Every dedup_key already published to this account (any topic)."""
        return set(self.data.get("posted", []))

    def posted_today(self, today: str | None = None) -> int:
        return int(self.data.get("daily", {}).get(today or _today(), 0))

    def remaining_today(self, daily_cap: int, today: str | None = None) -> int:
        """Posts still allowed on the account today, given the global cap."""
        return max(0, daily_cap - self.posted_today(today))

    def record(self, dedup_keys, today: str | None = None) -> None:
        """Mark one published bill: add its dedup_key (and any same-day sibling
        keys) to the cross-topic set and bump today's count by exactly one."""
        today = today or _today()
        keys = {k for k in dedup_keys if k}
        if not keys:
            return
        self.data["posted"] = sorted(set(self.data.get("posted", [])) | keys)
        daily = self.data.setdefault("daily", {})
        daily[today] = int(daily.get(today, 0)) + 1
        # Prune old date buckets so the file doesn't grow forever.
        if len(daily) > _MAX_DAILY_ENTRIES:
            for stale in sorted(daily)[:-_MAX_DAILY_ENTRIES]:
                daily.pop(stale, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["last_run"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.data, indent=2))
