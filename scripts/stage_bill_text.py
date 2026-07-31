#!/usr/bin/env python3
"""
Stage the govbot clone's full bill text for the parallel post shards.

The Bluesky post workflow splits into a single ``prepare`` job (which clones the
national govbot corpus) and five ``post`` shards (which do NOT clone, to avoid
five redundant ~20-minute clones). But full bill text lives *inside* the clone:
each bill dir carries a ``metadata.json`` plus govbot's pre-extracted
``files/<bill_id>_text_extracted.txt``. With the clone gone from the shards,
``bill_text.resolve_metadata_path`` finds nothing and every post falls back to
the short abstract — the regression this script fixes.

Re-shipping the whole clone is a non-starter (>14 GB, dominated by ``.git`` and
PDFs). Instead, ``prepare`` runs this once to copy the *small, useful subset* —
``metadata.json`` and the pre-extracted text file — for exactly the bills a shard
could ever call ``_get_full_text`` on: those that match at least one topic and
fall inside the freshness window. That set is a few thousand tiny text files,
not the nation. The workflow tars the staging tree into one artifact (a single
file dodges upload-artifact@v4's rejection of ``:`` in paths, which govbot dir
names like ``country:us`` are full of); each shard untars it and points
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

import shutil
import sys
from pathlib import Path

from post_to_bluesky import MAX_ACTION_AGE_DAYS, load_normalized_bills
from topic import Topic, list_topics
import bill_text

from datetime import datetime, timezone


def _fresh(action_date: str, cutoff) -> bool:
    """Mirror post_to_bluesky.main's freshness gate: keep only bills whose action
    is within MAX_ACTION_AGE_DAYS of today (undated → dropped)."""
    try:
        d = datetime.strptime(action_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    return (cutoff - d).days <= MAX_ACTION_AGE_DAYS


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
    print(f"  {len(wanted)} topic-matched, in-window bill(s) to stage full text for.")

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
