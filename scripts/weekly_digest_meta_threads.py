#!/usr/bin/env python3
"""
Threads (Meta) weekly digest thread: the Threads counterpart to
weekly_digest_bluesky.py and weekly_digest_x.py.

Unlike the Bluesky/X digests — one account per topic, so each digest covers a
single topic — the Threads account is a SINGLE account that now spans every
topic (the daily Threads poster already interleaves all topics, see
post_to_meta_threads.py). So this digest is all-topics too: a single root post
plus up to DIGEST_MAX_HIGHLIGHTS reply posts highlighting the most significant
bill activity over the past 7 days drawn from a WIDE VARIETY of topics, and each
reply is tagged with the topic it belongs to (emoji + topic name) so the mixed
thread is legible at a glance.

How the highlights are chosen: every topics/<name>/ config is loaded, its
matching bills are scored with the shared significance scorer, and the digest
round-robins across topics — taking each topic's strongest bill in turn — so the
thread spreads across as many topics as possible before doubling up on any one.
A global per-state cap keeps one busy statehouse from crowding the thread.

Reuse, not duplication:
  * Significance scoring, lookback windowing, per-topic highlight ranking and
    the landscape (quiet-week) fallback all come from weekly_digest_bluesky.py —
    the same logic that drives the Bluesky and X digests.
  * Threads composition (500-char budgeting, the two-step publish, reply
    chaining, the per-post topic label) comes from post_to_meta_threads.py.

Because the daily poster's modules are written around a single active TOPIC
(selected via BOT_TOPIC), this script re-points that active topic per bill while
composing — see _activate_topic — so each reply's summary steering, emoji, and
topic label come from the bill's OWN topic rather than one global one. BOT_TOPIC
still has to name a valid topic so the modules import cleanly, but which one no
longer matters to the output.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import digest_multitopic as dm
import post_to_bluesky as pb
import post_to_meta_threads as pmt
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
    compose_threads_post,
    publish_post,
    threads_summary_budget,
)
from topic import Topic
from weekly_digest_bluesky import (
    DIGEST_LOOKBACK_DAYS,
    DIGEST_PER_STATE_CAP,
    _format_jurisdictions_line,
    _format_short,
)

ROOT = Path(__file__).resolve().parent.parent

# The Threads account spans every topic, so this digest pulls its highlights
# from a wide variety of topics rather than one. Up to this many highlight
# replies go under the root post.
DIGEST_MAX_HIGHLIGHTS = int(os.environ.get("DIGEST_MAX_HIGHLIGHTS", "11"))

# Title/brand for the all-topics root post. Not tied to any single topic's
# thread_title (those name one topic, e.g. "LGBTQ Bills Weekly Digest").
DIGEST_TITLE = os.environ.get(
    "THREADS_DIGEST_TITLE", "🏛️ Statehouse Weekly Digest")

# Seconds to pause between thread posts. Threads is far more forgiving than X,
# but each post is a two-call create+publish, so a small gap keeps the API
# happy and the thread well-ordered.
THREADS_THREAD_SLEEP = int(os.environ.get("THREADS_THREAD_SLEEP", "5"))


# ---------------------------------------------------------------------------
# Per-topic activation
#
# post_to_bluesky and post_to_meta_threads are written around a single module
# global TOPIC (the daily posters run one process per topic). This digest spans
# all topics in one process, so before composing each reply we re-point that
# global to the bill's own topic. Then summarize()/shorten_title() steer their
# copy on the right topic, and the Threads composer stamps the right emoji +
# topic label. Functions look their module globals up at call time, so this
# takes effect for the imported compose/summarize helpers too.
# ---------------------------------------------------------------------------

def _activate_topic(topic: Topic) -> None:
    pb.TOPIC = topic
    pmt.TOPIC = topic


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_root(today: datetime, window_days: int) -> str:
    end = today
    start = today - timedelta(days=window_days - 1)
    range_str = f"{_format_short(start)}–{_format_short(end)}, {end.year}"
    if window_days <= DIGEST_LOOKBACK_DAYS:
        framing = (
            "This week's top bill activity from statehouses across the country "
            "— spanning a wide range of topics. 🧵"
        )
    else:
        framing = (
            "Quieter past 7 days, so we widened the lens — recent bill activity "
            "across the states, spanning a wide range of topics. 🧵"
        )
    text = f"{DIGEST_TITLE}\n{range_str}\n\n{framing}"
    if len(text) > MAX_THREADS:
        # Drop the range line first, then fall back to just the title.
        text = f"{DIGEST_TITLE}\n\n{framing}"
    if len(text) > MAX_THREADS:
        text = DIGEST_TITLE
    return text


def compose_landscape_root(today: datetime, state_counts: Counter) -> str:
    juris_line = _format_jurisdictions_line(state_counts)
    text = (
        f"{DIGEST_TITLE}\n"
        f"Week of {_format_short(today)}, {today.year}\n\n"
        "Quiet stretch — little notable floor action across the states this "
        "past month. Still tracking bills on a wide range of topics in "
        f"{juris_line}. A landscape check-in 🧵"
    )
    if len(text) > MAX_THREADS:
        text = (
            f"{DIGEST_TITLE}\n"
            f"Week of {_format_short(today)}, {today.year}\n\n"
            "Quiet stretch — little notable floor action from the past month, "
            "but we're still tracking bills on a wide range of topics across "
            "the states. A landscape check-in 🧵"
        )
    if len(text) > MAX_THREADS:
        text = DIGEST_TITLE
    return text


def _landscape_closing_reply() -> str:
    return (
        "🔔 Many statehouses are between sessions or on recess this time of "
        "year. When bills start moving again, they'll show up in our daily "
        "posts and next week's digest. See you then."
    )


def build_highlight_replies(highlights: list[dict]) -> list[tuple[str, str]]:
    """Compose one reply per highlight as (post_text, bill_url). Each reply
    carries its topic label (emoji + topic name) via include_topic, so the mixed
    all-topics thread declares which topic every bill belongs to. The bill URL
    ships as that reply's link_attachment — unlike the X digest, every link
    rides along with its own bill instead of being packed into a final post.

    Each bill's summary steering, emoji, and topic label come from its OWN topic
    (_activate_topic re-points the modules' active TOPIC before composing)."""
    replies: list[tuple[str, str]] = []
    for b in highlights:
        _activate_topic(b["_topic"])
        ensure_english_fields(b)
        headline = shorten_title(b)
        budget = threads_summary_budget(b, headline, include_topic=True)
        summary = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
        text, url = compose_threads_post(
            b, summary, headline=headline, include_topic=True)
        replies.append((text, url))
        print(f"  prepared reply: [{b['_topic_name']}] {b['state']} "
              f"{b['identifier']} ({b['action_date']}, "
              f"score={b.get('_score', 0)}, link={'yes' if url else 'no'})")
    return replies


# ---------------------------------------------------------------------------
# Raw-artifact persistence (Threads weekly-digest folder, per topic)
# ---------------------------------------------------------------------------

def _save_digest_raw_records(bills: list[dict]) -> None:
    """Dump the verbatim bills.jsonl record (and extracted full text) for every
    bill featured in the digest to its OWN topic's
    topics/<name>/meta-threads/weekly_digest/ folder, so the digest thread
    leaves the same self-contained trail as the daily feed. Saved up front so
    the artifact lands even if a reply post fails partway."""
    for b in bills:
        raw = b.get("_raw")
        topic = b.get("_topic")
        if not raw or topic is None:
            continue
        out_dir = topic.threads_weekly_digest_bills_raw_dir()
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

    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    # Extract every bill once, then build each topic's match list from the shared
    # dicts (a bill can match several topics; the round-robin dedups by bill).
    extracted = dm.extract_all(records)
    matched_by_topic = dm.build_matched_by_topic(
        extracted,
        on_skip=lambda name, e: print(f"  ! skipping topic {name!r}: {e}", file=sys.stderr),
    )
    if not matched_by_topic:
        print("No topics found under topics/. Nothing to digest.")
        return 0
    print(f"=== Threads weekly digest (all topics): "
          f"{', '.join(matched_by_topic)} ===")
    for name, (_, matched) in matched_by_topic.items():
        print(f"  {name}: {len(matched)} matching bill action(s)")

    if not any(m for _, m in matched_by_topic.values()):
        print("No topic bills found at all. Nothing to digest.")
        return 0

    # Primary 7-day window first, widening if EVERY topic is empty so a quiet
    # week doesn't kill the digest.
    chosen_window, per_topic = dm.choose_active_window(
        matched_by_topic, today, DIGEST_PER_STATE_CAP)
    print(f"Chosen lookback: {chosen_window}d ({len(per_topic)} topic(s) active).")

    if per_topic:
        highlights = dm.merge_across_topics(
            per_topic, cap=DIGEST_MAX_HIGHLIGHTS, per_state_cap=DIGEST_PER_STATE_CAP)
        topics_covered = sorted({b["_topic_name"] for b in highlights})
        print(f"\nSelected {len(highlights)} highlight(s) across "
              f"{len(topics_covered)} topic(s) (cap={DIGEST_MAX_HIGHLIGHTS}, "
              f"per-state-cap={DIGEST_PER_STATE_CAP}, window={chosen_window}d): "
              f"{', '.join(topics_covered)}")
        for b in highlights:
            print(f"  [{b['_score']:>3}] [{b['_topic_name']}] {b['state']} "
                  f"{b['identifier']} ({b['action_date']}): {b['action_desc'][:60]}")

        _save_digest_raw_records(highlights)
        replies = build_highlight_replies(highlights)
        root_text = compose_root(today, chosen_window)
        post_digest_thread(root_text, replies)
        print(f"\nDone. Posted Threads thread: 1 root + {len(replies)} highlight(s) "
              f"across {len(topics_covered)} topic(s) (window={chosen_window}d).")
        return 0

    # No floor activity in any window for any topic — ship a landscape thread so
    # the weekly slot still produces something informative.
    recent_bills = dm.landscape_picks(
        matched_by_topic, cap=DIGEST_MAX_HIGHLIGHTS, per_state_cap=DIGEST_PER_STATE_CAP)
    state_counts = Counter((b["state"] or "?") for b in recent_bills)
    distinct_states = len([s for s in state_counts if s])
    topics_covered = sorted({b["_topic_name"] for b in recent_bills})
    print(f"No recent floor activity. Posting landscape thread "
          f"({len(recent_bills)} bills across {distinct_states} jurisdiction(s), "
          f"{len(topics_covered)} topic(s)).")
    for b in recent_bills:
        print(f"  [{b['_topic_name']}] {b['state']} {b['identifier']} "
              f"({b['action_date']}): {b['action_desc'][:60]}")

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
