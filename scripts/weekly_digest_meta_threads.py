#!/usr/bin/env python3
"""
Threads (Meta) weekly digest thread: the Threads counterpart to
weekly_digest_bluesky.py and weekly_digest_x.py.

A single root post + up to DIGEST_MAX_HIGHLIGHTS reply posts highlighting the
most significant bill activity for the active topic over the past 7 days,
chained into one Threads thread via the API's reply_to_id. Unlike the X digest
— which has to pack every link into a final post because X charges t.co budget
per link and renders no card — Threads supports a per-post link_attachment, so
each highlight reply carries its own bill link as a preview. That makes this the
simplest of the three digests: root, then one self-contained reply per bill.

Reuse, not duplication:
  * Significance scoring, lookback windowing, highlight selection and the
    landscape (quiet-week) fallback all come from weekly_digest_bluesky.py — the same
    logic that drives the Bluesky and X digests.
  * Threads composition (500-char budgeting, the two-step publish, reply
    chaining) comes from post_to_meta_threads.py.

The active topic is selected via the BOT_TOPIC env var (see scripts/topic.py).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from post_to_bluesky import (
    _FILENAME_UNSAFE_RE,
    _slug,
    ensure_english_fields,
    load_bills,
    save_full_text,
    shorten_title,
    summarize,
)
from post_to_meta_threads import (
    DRY_RUN,
    JSONL_PATH,
    MAX_THREADS,
    MIN_SUMMARY_CHARS,
    THREADS_ACCESS_TOKEN,
    THREADS_USER_ID,
    TOPIC,
    compose_threads_post,
    publish_post,
    threads_summary_budget,
)
from weekly_digest_bluesky import (
    DIGEST_LANDSCAPE_CARDS,
    DIGEST_LOOKBACK_DAYS,
    DIGEST_MAX_HIGHLIGHTS,
    DIGEST_PER_STATE_CAP,
    LOOKBACK_FALLBACK_WINDOWS,
    _format_jurisdictions_line,
    _format_short,
    _landscape_unique_bills,
    _select_landscape_bills,
    candidates_in_window,
    collect_topic_bills,
    select_highlights,
)

ROOT = Path(__file__).resolve().parent.parent

# Seconds to pause between thread posts. Threads is far more forgiving than X,
# but each post is a two-call create+publish, so a small gap keeps the API
# happy and the thread well-ordered.
THREADS_THREAD_SLEEP = int(os.environ.get("THREADS_THREAD_SLEEP", "5"))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_root(today: datetime, window_days: int) -> str:
    end = today
    start = today - timedelta(days=window_days - 1)
    range_str = f"{_format_short(start)}–{_format_short(end)}, {end.year}"
    if window_days <= DIGEST_LOOKBACK_DAYS:
        framing = "This week's top bill activity across the states. 🧵"
    else:
        framing = (
            "Quieter past 7 days, so we widened the lens — recent bill "
            "activity across the states. 🧵"
        )
    text = f"{TOPIC.thread_title}\n{range_str}\n\n{framing}"
    if len(text) > MAX_THREADS:
        # Drop the range line first, then fall back to just the title.
        text = f"{TOPIC.thread_title}\n\n{framing}"
    if len(text) > MAX_THREADS:
        text = TOPIC.thread_title
    return text


def compose_landscape_root(today: datetime, state_counts: Counter) -> str:
    juris_line = _format_jurisdictions_line(state_counts)
    text = (
        f"{TOPIC.thread_title}\n"
        f"Week of {_format_short(today)}, {today.year}\n\n"
        "Quiet stretch — no notable floor or executive action to flag from the "
        f"past month. Still tracking {TOPIC.topic_phrase} bills in "
        f"{juris_line}. A landscape check-in 🧵"
    )
    if len(text) > MAX_THREADS:
        text = (
            f"{TOPIC.thread_title}\n"
            f"Week of {_format_short(today)}, {today.year}\n\n"
            "Quiet stretch — no notable floor action from the past month, but "
            f"we're still tracking {TOPIC.topic_phrase} bills across the "
            "states. A landscape check-in 🧵"
        )
    if len(text) > MAX_THREADS:
        text = TOPIC.thread_title
    return text


def _landscape_closing_reply() -> str:
    return (
        "🔔 Many statehouses are between sessions or on recess this time of "
        "year. When bills start moving again, they'll show up in our daily "
        "posts and next week's digest. See you then."
    )


def build_highlight_replies(highlights: list[dict]) -> list[tuple[str, str]]:
    """Compose one reply per highlight as (post_text, bill_url). The URL ships
    as that reply's link_attachment, so — unlike the X digest — every link rides
    along with its own bill instead of being packed into a final post."""
    replies: list[tuple[str, str]] = []
    for b in highlights:
        ensure_english_fields(b)
        headline = shorten_title(b)
        budget = threads_summary_budget(b, headline)
        summary = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
        text, url = compose_threads_post(b, summary, headline=headline)
        replies.append((text, url))
        print(f"  prepared reply: {b['state']} {b['identifier']} "
              f"({b['action_date']}, score={b.get('_score', 0)}, "
              f"link={'yes' if url else 'no'})")
    return replies


# ---------------------------------------------------------------------------
# Raw-artifact persistence (Threads weekly-digest folder)
# ---------------------------------------------------------------------------

def _save_digest_raw_records(bills: list[dict]) -> None:
    """Dump the verbatim bills.jsonl record (and extracted full text) for every
    bill featured in the digest to topics/<name>/<threads_subdir>/weekly_digest/,
    so the digest thread leaves the same self-contained trail as the daily feed.
    Saved up front so the artifact lands even if a reply post fails partway."""
    out_dir = TOPIC.threads_weekly_digest_bills_raw_dir()
    for b in bills:
        raw = b.get("_raw")
        if not raw:
            continue
        try:
            state = b.get("state") or "XX"
            ident_raw = (b.get("identifier") or "unknown").strip()
            ident = _FILENAME_UNSAFE_RE.sub("_", ident_raw).strip("_")[:24] or "unknown"
            date = b.get("action_date") or "no-date"
            action_slug = _slug(b.get("action_desc") or "no-action", max_len=40) or "no-action"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{state}-{ident}-{date}-{action_slug}.json"
            out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
            save_full_text(b, out_dir=out_dir.parent / "bills_full_text")
        except OSError as e:
            print(f"  ! digest raw-record save failed for "
                  f"{b.get('state')} {b.get('identifier')}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------

def post_digest_thread(root_text: str, replies: list[tuple[str, str]]) -> None:
    """Post the root, then chain each (text, url) reply under the previous post
    via reply_to_id. A failed reply is logged and skipped; subsequent replies
    chain off the last post that DID land so one hiccup doesn't orphan the
    thread (mirrors weekly_digest_x.post_thread)."""
    print(f"\n--- ROOT ({len(root_text)} chars) ---\n{root_text}\n---")
    if DRY_RUN:
        for i, (text, url) in enumerate(replies, 1):
            print(f"\n--- REPLY {i} ({len(text)} chars) ---\n{text}")
            if url:
                print(f"  ↳ link_attachment: {url}")
            print("---")
        return

    root_id = publish_post(root_text)
    if not root_id:
        print(" ! root post failed; aborting thread.", file=sys.stderr)
        return
    print(f"  posted root (media id {root_id})")
    parent_id = root_id

    for i, (text, url) in enumerate(replies, 1):
        time.sleep(THREADS_THREAD_SLEEP)
        print(f"\n--- REPLY {i} ({len(text)} chars) ---\n{text}")
        if url:
            print(f"  ↳ link_attachment: {url}")
        print("---")
        media_id = publish_post(text, link_url=url, reply_to_id=parent_id)
        if media_id:
            parent_id = media_id
            print(f"  posted reply {i} (media id {media_id})")
        else:
            print(f"  ! reply {i} post failed; continuing.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    missing = [
        n for n, v in (
            ("THREADS_ACCESS_TOKEN", THREADS_ACCESS_TOKEN),
            ("THREADS_USER_ID", THREADS_USER_ID),
        ) if not v
    ]
    if missing and not DRY_RUN:
        print(f"ERROR: missing Threads credentials: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"=== Threads weekly digest running for topic: {TOPIC.name} ===")
    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    all_bills = collect_topic_bills(records)
    if not all_bills:
        print(f"No {TOPIC.topic_phrase} bills found at all. Nothing to digest.")
        return 0

    # Primary 7-day window first, widening if it's empty so a quiet week
    # doesn't kill the digest.
    candidates: list[dict] = []
    chosen_window = LOOKBACK_FALLBACK_WINDOWS[0]
    for window in LOOKBACK_FALLBACK_WINDOWS:
        candidates = candidates_in_window(all_bills, today, window)
        print(f"Lookback {window}d: {len(candidates)} {TOPIC.topic_phrase} bill update(s).")
        if candidates:
            chosen_window = window
            break

    if candidates:
        unique_bills = {(b["state"], b["identifier"]) for b in candidates}
        state_counts = Counter(s or "?" for s, _ in unique_bills)
        print(f"  unique bills: {len(unique_bills)} (from {len(candidates)} action entries)")
        print(f"  by state: {', '.join(f'{s}={n}' for s, n in state_counts.most_common(15))}")

        highlights = select_highlights(candidates)
        print(f"\nSelected {len(highlights)} highlight(s) (cap={DIGEST_MAX_HIGHLIGHTS}, "
              f"per-state-cap={DIGEST_PER_STATE_CAP}, window={chosen_window}d):")
        for b in highlights:
            print(f"  [{b['_score']:>3}] {b['state']} {b['identifier']} "
                  f"({b['action_date']}): {b['action_desc'][:70]}")

        _save_digest_raw_records(highlights)
        replies = build_highlight_replies(highlights)
        root_text = compose_root(today, chosen_window)
        post_digest_thread(root_text, replies)
        print(f"\nDone. Posted Threads thread: 1 root + {len(replies)} highlight(s) "
              f"(window={chosen_window}d).")
        return 0

    # No floor activity in any window — ship a landscape thread so the weekly
    # slot still produces something informative.
    unique_bills = _landscape_unique_bills(all_bills)
    state_counts = Counter((b["state"] or "?") for b in unique_bills)
    distinct_states = len([s for s in state_counts if s])
    print(f"No recent floor activity. Posting landscape thread "
          f"({len(unique_bills)} bills across {distinct_states} jurisdiction(s)).")

    recent_bills = _select_landscape_bills(unique_bills, n=DIGEST_LANDSCAPE_CARDS)
    print(f"Selected {len(recent_bills)} landscape card(s):")
    for b in recent_bills:
        print(f"  {b['state']} {b['identifier']} ({b['action_date']}): "
              f"{b['action_desc'][:70]}")

    _save_digest_raw_records(recent_bills)
    bill_replies = build_highlight_replies(recent_bills)
    root_text = compose_landscape_root(today, state_counts)
    # Closing note sits after the bill cards (it has no link).
    replies = bill_replies + [(_landscape_closing_reply(), "")]
    post_digest_thread(root_text, replies)
    print(f"\nDone. Posted Threads landscape thread with {len(replies)} reply post(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
