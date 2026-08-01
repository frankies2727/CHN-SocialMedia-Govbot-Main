#!/usr/bin/env python3
"""
Stage the govbot clone's full bill text so it survives the shared per-day cache.

The govbot-bills action shares one bills.jsonl across every run on the same UTC
day and, on a cache hit, SKIPS the multi-state clone. But full bill text lives
*inside* that clone: each bill dir carries a ``metadata.json`` plus govbot's
pre-extracted ``files/<bill_id>_text_extracted.txt``. So on any cache-hit run
(every run after the day's first) the clone is gone,
``bill_text.resolve_metadata_path`` finds nothing, and every post falls back to
the short abstract — the regression this script fixes. The parallel Bluesky post
shards (which never clone) hit the same wall.

Re-caching / re-shipping the whole clone is a non-starter (>14 GB, dominated by
``.git`` and PDFs). Instead this runs once on the day's clone (cache miss) to
copy the *small, useful subset* — ``metadata.json`` and the pre-extracted text
file — for exactly the bills a run could ever call ``_get_full_text`` on: those
that match at least one topic and fall inside the freshness window
(STAGE_MAX_AGE_DAYS). That set is a few thousand tiny text files, not the nation.
The govbot-bills action tars the staging tree into one file cached ALONGSIDE
bills.jsonl (a single file also dodges upload-artifact@v4's rejection of ``:`` in
paths — govbot dir names like ``country:us`` are full of them — for the Bluesky
shards that ship it as an artifact). A cache-hit run untars it and points
``GOVBOT_DATA_ROOT`` at the result, so ``resolve_metadata_path`` resolves against
the staged subset exactly as it would against a full clone — no code change to
bill_text.py.

For a bill with no pre-extracted text, only ``metadata.json`` is copied; the
shard still has poppler + network, so the existing PDF-download fallback in
bill_text.py handles it from the document link in that metadata.json.

Usage (BOT_TOPIC must be set so the shared modules import; any topic works):
    BOT_TOPIC=<topic> python scripts/stage_bill_text.py <stage_dir>
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import bill_text
from post_to_bluesky import MAX_ACTION_AGE_DAYS, load_normalized_bills
from topic import Topic, list_topics

# How far back to stage full text. Must cover every consumer of the shared
# snapshot: the daily posters gate at MAX_ACTION_AGE_DAYS (32) and the weekly
# digests widen their lookback up to 30 days, so a margin over both guarantees
# the staged subset is a superset of what any run can ask for. Configurable so a
# future longer-lookback consumer can widen it without a code change.
STAGE_MAX_AGE_DAYS = int(
    os.environ.get("STAGE_MAX_AGE_DAYS", str(max(MAX_ACTION_AGE_DAYS, 45))))


def _fresh(action_date: str, cutoff) -> bool:
    """Keep only bills whose action is within STAGE_MAX_AGE_DAYS of today (undated
    → dropped). Deliberately a touch wider than the daily poster's own freshness
    gate so the shared snapshot also covers the weekly digests' longer windows."""
    try:
        d = datetime.strptime(action_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    return (cutoff - d).days <= STAGE_MAX_AGE_DAYS


def _matched_fresh_sources(bills: list[dict]) -> set[str]:
    """Union, across every topic, of the ``sources_bill`` paths for bills that a
    shard could load full text for: topic-matched AND inside the freshness
    window. This is a superset of what any single run actually fetches (the draw
    + gate walk only a prefix of it), so staging it guarantees the shards never
    miss text for a bill they end up considering."""
    cutoff = datetime.now(timezone.utc).date()
    fresh = [b for b in bills if _fresh(b.get("action_date", ""), cutoff)]

    topics: list[Topic] = []
    for name in list_topics():
        try:
            topics.append(Topic.load(name))
        except (FileNotFoundError, ValueError) as e:
            print(f"  skipping topic {name}: {e}", file=sys.stderr)
    if not topics:
        return set()

    wanted: set[str] = set()
    for b in fresh:
        sb = b.get("sources_bill") or ""
        if not sb:
            continue
        if any(t.matches(b) for t in topics):
            wanted.add(sb)
    return wanted


def stage(stage_dir: Path) -> tuple[int, int, int]:
    """Copy metadata.json (+ pre-extracted text when present) for every wanted
    bill into ``stage_dir`` under the bill's ``sources_bill`` relative path, so a
    shard can set GOVBOT_DATA_ROOT=<stage_dir> and resolve them unchanged.

    Returns (wanted, with_text, meta_only)."""
    bills = load_normalized_bills()
    if not bills:
        print("  no normalized bills loaded; nothing to stage.")
        return (0, 0, 0)

    wanted = _matched_fresh_sources(bills)
    print(f"  {len(wanted)} topic-matched bill(s) within {STAGE_MAX_AGE_DAYS}d "
          f"to stage full text for.")

    with_text = 0
    meta_only = 0
    missing = 0
    for sources_bill in sorted(wanted):
        meta_src = bill_text.resolve_metadata_path(sources_bill)
        if meta_src is None:
            missing += 1
            continue
        rel = Path(sources_bill.lstrip("/"))  # e.g. il-legislation/.../SB1/metadata.json
        dest_meta = stage_dir / rel
        dest_meta.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(meta_src, dest_meta)
        except OSError as e:
            print(f"  ! failed to copy metadata for {sources_bill}: {e}", file=sys.stderr)
            missing += 1
            continue

        extracted = bill_text.find_extracted_text_file(meta_src)
        if extracted is not None:
            dest_text = dest_meta.parent / "files" / extracted.name
            dest_text.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(extracted, dest_text)
                with_text += 1
            except OSError as e:
                print(f"  ! failed to copy text for {sources_bill}: {e}", file=sys.stderr)
                meta_only += 1
        else:
            # No pre-extracted text in the clone; the metadata.json carries the
            # document link, so the shard's PDF-download fallback handles it.
            meta_only += 1

    if missing:
        print(f"  {missing} bill(s) had no resolvable metadata in the clone (skipped).")
    print(f"  staged {with_text} with pre-extracted text, {meta_only} metadata-only.")
    return (len(wanted), with_text, meta_only)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: BOT_TOPIC=<topic> python scripts/stage_bill_text.py <stage_dir>",
              file=sys.stderr)
        return 2
    stage_dir = Path(argv[0])
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage(stage_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
