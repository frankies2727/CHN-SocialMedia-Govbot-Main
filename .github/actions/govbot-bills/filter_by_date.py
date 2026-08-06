#!/usr/bin/env python3
"""Stream-filter `govbot logs` JSONL on stdin, keeping only records whose action
is within a recency window, and write them to stdout.

This is what lets the govbot-bills action run `govbot logs --filter none
--limit none` (every routine action, no per-state cap) without ballooning
bills.jsonl: govbot streams the full log line-by-line and this drops everything
older than CUTOFF_EPOCH before it lands on disk. Kept deliberately dependency-free
and line-at-a-time so memory stays flat no matter how large the upstream log is.

Env:
  CUTOFF_EPOCH  Unix seconds (UTC). Records dated on/after this are kept.

A record is kept when its date is >= the cutoff, OR when it has no parseable date
(rare — govbot records carry a filename-derived `timestamp`), so a malformed date
never silently drops a bill from the feed.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


def _record_epoch(rec: dict) -> int | None:
    """Best-effort UTC-midnight epoch for a record: prefer the log action's
    date, fall back to the filename-derived timestamp. None if neither parses."""
    action = (rec.get("log") or {}).get("action") or {}
    day = (action.get("date") or "")[:10]
    if day:
        try:
            return int(datetime.strptime(day, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    # govbot stamps each log record with the source filename's "YYYYMMDDT..." ts.
    ts = rec.get("timestamp") or ""
    if len(ts) >= 8:
        try:
            return int(datetime.strptime(ts[:8], "%Y%m%d")
                       .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    return None


def main() -> int:
    cutoff = int(os.environ["CUTOFF_EPOCH"])
    kept = dropped = undated = 0
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip unparseable lines, same as load_bills()
        epoch = _record_epoch(rec)
        if epoch is None:
            undated += 1
            out.write(line + "\n")  # keep: never silently lose a bill
        elif epoch >= cutoff:
            kept += 1
            out.write(line + "\n")
        else:
            dropped += 1
    sys.stderr.write(
        f"filter_by_date: kept {kept} in-window "
        f"({undated} undated kept), dropped {dropped} older records.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
