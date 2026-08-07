#!/usr/bin/env python3
"""Stream-filter `govbot logs` JSONL on stdin, keeping only records whose action
is within a recency window, and write them to stdout.

This is what lets the govbot-bills action run `govbot logs --filter none
--limit none` (every routine action, no per-state cap) without ballooning
bills.jsonl: govbot streams the full log line-by-line and this drops everything
older than CUTOFF_EPOCH before it lands on disk. Kept deliberately dependency-free
and line-at-a-time so memory stays flat no matter how large the upstream log is.

Because it already sees EVERY action of EVERY bill in the clone (the stream is
unbounded — only stdout is date-filtered), it doubles as the clone-coverage
probe: it tallies, per state, how many records the clone holds and the newest
action date it carries, then flags any state whose newest action lags the rest
of the corpus. That is the signal that would have caught Hawaii silently sitting
on stale 2025 data while every other state advanced into 2026 — `govbot clone
all` exits 0 even when a repo fails or lags, so nothing else surfaces it.

Env:
  CUTOFF_EPOCH        Unix seconds (UTC). Records dated on/after this are kept.
  TODAY_EPOCH         Unix seconds (UTC) for "now". Future-dated actions (e.g.
                      2027 effective dates) are ignored when computing a state's
                      newest date so they can't mask staleness. Optional; if
                      unset, no future cap is applied.
  STALE_LAG_DAYS      A state is flagged when its newest action lags the corpus
                      frontier (the freshest any state reaches) by more than this
                      many days. Default 30.
  COVERAGE_FLAGS_FILE If set, flagged states are written here one per line so the
                      calling shell step can echo GitHub `::warning::` annotations
                      (this script's stdout is bills.jsonl, so it can't emit them).

A record is kept when its date is >= the cutoff, OR when it has no parseable date
(rare — govbot records carry a filename-derived `timestamp`), so a malformed date
never silently drops a bill from the feed.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

# Mirrors detect_state() in post_to_bluesky.py: govbot tags every record with an
# OCD jurisdiction like ".../state:hi/..." ("state:usa" for federal). Searching
# the raw JSON line is far cheaper than walking the parsed record and is just as
# reliable — the tag is identical everywhere it appears in a record.
_STATE_TAG_RE = re.compile(r"\bstate:([a-z]{2,3})\b", re.IGNORECASE)
_DAY = 86400


def _state_of(line: str) -> str:
    m = _STATE_TAG_RE.search(line)
    if not m:
        return "??"
    code = m.group(1).upper()
    return "US" if code == "USA" else code


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


def _fmt(epoch: int | None) -> str:
    if epoch is None:
        return "  (none)  "
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d")


def _report_coverage(counts: dict, newest: dict, today: int | None, stale_lag_days: int) -> None:
    """Print a per-state coverage table to stderr and flag states lagging the
    corpus frontier. Writes flagged states to COVERAGE_FLAGS_FILE if set."""
    if not counts:
        sys.stderr.write("coverage: no records seen in the clone stream.\n")
        return

    # Frontier = the freshest newest-action any state reaches (capped at today so
    # a stray future date can't inflate it). States are stale RELATIVE to this, so
    # a uniformly old clone doesn't false-alarm — only a state left behind does.
    dated = [e for e in newest.values() if e is not None]
    frontier = max(dated) if dated else None

    rows = sorted(counts, key=lambda s: (newest.get(s) is not None, newest.get(s) or 0))
    sys.stderr.write(
        f"\n=== clone coverage: {len(counts)} state(s), "
        f"corpus frontier {_fmt(frontier)} ===\n"
        f"{'state':<6}{'records':>9}   {'newest action':<14}{'lag(days)':>10}\n"
    )
    flagged = []
    for st in rows:
        n = newest.get(st)
        lag = ""
        is_stale = False
        if frontier is not None and n is not None:
            lag_days = (frontier - n) // _DAY
            lag = str(lag_days)
            if lag_days > stale_lag_days:
                is_stale = True
        elif n is None:
            lag = "n/a"
        mark = "  <== STALE" if is_stale else ""
        sys.stderr.write(f"{st:<6}{counts[st]:>9}   {_fmt(n):<14}{lag:>10}{mark}\n")
        if is_stale:
            flagged.append((st, counts[st], _fmt(n), lag))

    if flagged:
        sys.stderr.write(
            f"\ncoverage WARNING: {len(flagged)} state(s) lag the corpus by "
            f">{stale_lag_days}d — the clone may be serving stale data for them "
            f"(bills there will be invisible to the posters):\n"
        )
        for st, n, nd, lag in flagged:
            sys.stderr.write(f"  {st}: newest action {nd} ({lag}d behind, {n} records)\n")
        flags_file = os.environ.get("COVERAGE_FLAGS_FILE")
        if flags_file:
            try:
                with open(flags_file, "w", encoding="utf-8") as fh:
                    for st, n, nd, lag in flagged:
                        fh.write(f"{st} is {lag}d behind the corpus (newest action {nd}); "
                                 f"clone likely stale — its bills are invisible to the posters.\n")
            except OSError as e:
                sys.stderr.write(f"coverage: could not write flags file: {e}\n")
    else:
        sys.stderr.write("coverage: no state lags the corpus frontier — coverage looks healthy.\n")


def main() -> int:
    cutoff = int(os.environ["CUTOFF_EPOCH"])
    today = int(os.environ["TODAY_EPOCH"]) if os.environ.get("TODAY_EPOCH") else None
    stale_lag_days = int(os.environ.get("STALE_LAG_DAYS", "30"))

    counts: dict[str, int] = {}
    newest: dict[str, int | None] = {}
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

        # --- coverage tally over the FULL stream (before date-bounding) ---
        st = _state_of(line)
        counts[st] = counts.get(st, 0) + 1
        epoch = _record_epoch(rec)
        # Ignore future-dated actions (effective dates) when tracking freshness.
        if epoch is not None and (today is None or epoch <= today):
            cur = newest.get(st)
            if cur is None or epoch > cur:
                newest[st] = epoch
        elif st not in newest:
            newest[st] = None

        # --- date filter for stdout (bills.jsonl) ---
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
    _report_coverage(counts, newest, today, stale_lag_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
