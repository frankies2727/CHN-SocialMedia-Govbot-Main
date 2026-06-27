#!/usr/bin/env python3
"""
Account-level (cross-topic) post ledger for the single-account platforms.

Bluesky runs one account per topic, but Instagram and Threads each publish to a
SINGLE account that now spans every topic. That makes the per-topic dedup files
(topics/<name>/<platform>/bills_used.json) insufficient on their own:

  * Cross-topic dedup — a bill matched by two topics (say a measure that is both
    "healthcare" and "reproductive_rights") must not be posted twice to the same
    account.
  * Per-run cap — 13 topics each posting independently would flood one feed, so
    the account needs a ceiling on how many posts go out per workflow run in
    total (across every topic combined).

This module stores a shared ledger per platform at
account_state/<platform>/posted.json:

    {
      "posted": ["<dedup_key>", ...],   # every key published to the account
      "runs":   {"<run_id>": <count>},  # posts published per run
      "last_run": "<iso8601>"
    }

The per-topic dedup files are still written as before; this ledger is an
additional, account-wide guard layered on top. Because the daily posters loop
over topics sequentially in one job, each topic's process reads the ledger the
previous topic just saved, so the cap and dedup hold across the whole run.

A "run" is one execution of the topic loop. All of that run's per-topic
processes share a single run id (RUN_ID, or GitHub Actions' run id + attempt),
so the cap is enforced across every topic in the run yet starts fresh on the
next run. When no run id is available (an ad-hoc local invocation), the run is
just that single process and the cap is enforced within it alone — nothing
run-scoped is persisted, so repeated local runs are never blocked by stale
state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep the per-run counter from growing without bound; only the current run's
# bucket matters for the cap, the rest is just recent history.
_MAX_RUN_ENTRIES = 30


def _run_id() -> str | None:
    """Stable identifier shared by every per-topic process in one run, or None
    when there's no run-scoped context (an ad-hoc local invocation). Prefers an
    explicit RUN_ID, then GitHub Actions' run id + attempt."""
    rid = (os.environ.get("RUN_ID") or "").strip()
    if rid:
        return rid
    gh = (os.environ.get("GITHUB_RUN_ID") or "").strip()
    if gh:
        attempt = (os.environ.get("GITHUB_RUN_ATTEMPT") or "1").strip()
        return f"{gh}-{attempt}"
    return None


class AccountLedger:
    """Read/modify/save the cross-topic ledger for one platform
    ("instagram" or "meta-threads")."""

    def __init__(self, platform: str):
        self.platform = platform
        self.path = ROOT / "account_state" / platform / "posted.json"
        self.data = self._load()
        self.run_id = _run_id()
        # Track this process's own posts so the per-run cap still holds for an
        # ad-hoc local run that has no shared run id to persist against.
        self._local_count = 0

    def _load(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                d.setdefault("posted", [])
                d.setdefault("runs", {})
                d.pop("daily", None)  # legacy per-day counter, no longer used
                return d
            except json.JSONDecodeError:
                pass
        return {"posted": [], "runs": {}}

    @property
    def seen(self) -> set[str]:
        """Every dedup_key already published to this account (any topic)."""
        return set(self.data.get("posted", []))

    def posted_this_run(self) -> int:
        """Posts published on the account so far during the current run."""
        if not self.run_id:
            return self._local_count
        return int(self.data.get("runs", {}).get(self.run_id, 0))

    def remaining_this_run(self, run_cap: int) -> int:
        """Posts still allowed on the account this run, given the per-run cap."""
        return max(0, run_cap - self.posted_this_run())

    def record(self, dedup_keys) -> None:
        """Mark one published bill: add its dedup_key (and any same-day sibling
        keys) to the cross-topic set and bump this run's count by exactly one."""
        keys = {k for k in dedup_keys if k}
        if not keys:
            return
        self.data["posted"] = sorted(set(self.data.get("posted", [])) | keys)
        if not self.run_id:
            self._local_count += 1
            return
        runs = self.data.setdefault("runs", {})
        runs[self.run_id] = int(runs.get(self.run_id, 0)) + 1
        # Prune the oldest run buckets so the file doesn't grow forever. Dicts
        # preserve insertion order, so the earliest-seen runs sort to the front.
        if len(runs) > _MAX_RUN_ENTRIES:
            for stale in list(runs)[:-_MAX_RUN_ENTRIES]:
                runs.pop(stale, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["last_run"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.data, indent=2))
