#!/usr/bin/env python3
"""
X/Twitter weekly digest thread: the X counterpart to weekly_digest.py.

A single root tweet + up to DIGEST_MAX_HIGHLIGHTS reply tweets highlighting
the most significant bill activity for the active topic over the past 7 days,
chained into one X thread. Because X charges ~23 chars of t.co budget per
link and renders no rich link card, the per-highlight tweets carry NO link —
instead every bill's URL is collected into a single FINAL post at the end of
the thread, and the root tweet tells readers to look there. That mirrors how
the daily X poster (post_to_x.py) keeps links out of the main tweet.

Reuse, not duplication:
  * Significance scoring, lookback windowing, highlight selection and the
    landscape (quiet-week) fallback all come from weekly_digest.py — the same
    logic that drives the Bluesky digest.
  * X composition (weighted 280-char accounting, summary budgeting, the
    tweepy client) comes from post_to_x.py.

The only X-digest-specific logic here is the thread root/links-post copy, the
link-packing into the final post(s), and the sequential reply chaining.

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
    link_for,
    load_bills,
    save_full_text,
    shorten_title,
    summarize,
)
from post_to_x import (
    DRY_RUN,
    JSONL_PATH,
    MAX_TWEET,
    MIN_SUMMARY_CHARS,
    TOPIC,
    X_ACCESS_TOKEN,
    X_ACCESS_TOKEN_SECRET,
    X_API_KEY,
    X_API_SECRET,
    build_client,
    compose_x_post,
    x_summary_budget,
    x_weighted_len,
)
from weekly_digest import (
    DIGEST_LANDSCAPE_CARDS,
    DIGEST_LOOKBACK_DAYS,
    DIGEST_MAX_HIGHLIGHTS,
    DIGEST_PER_STATE_CAP,
    LOOKBACK_FALLBACK_WINDOWS,
    _format_short,
    _format_jurisdictions_line,
    _landscape_unique_bills,
    _select_landscape_bills,
    candidates_in_window,
    collect_topic_bills,
    select_highlights,
)

ROOT = Path(__file__).resolve().parent.parent

# Seconds to pause between thread posts. X returns sporadic 403s on rapid
# successive writes (see post_to_x.py's 27s gap between unrelated tweets);
# replies inside one thread are more forgiving, but we still pace them.
X_THREAD_SLEEP = int(os.environ.get("X_THREAD_SLEEP", "8"))

# X renders every link as a 23-char t.co shortlink regardless of the real URL
# length. Use that for budgeting the links post so a handful of long
# statehouse URLs still pack into a single tweet (our goal: "all links in the
# last post"), instead of x_weighted_len's raw count splitting them needlessly.
TCO_WEIGHT = 23


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_root(today: datetime, window_days: int, has_links: bool) -> str:
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
    links_line = "\n🔗 All bill links are in the last post." if has_links else ""
    text = f"{TOPIC.thread_title}\n{range_str}\n\n{framing}{links_line}"
    if x_weighted_len(text) > MAX_TWEET:
        # Drop the range line first, then the framing, before truncating.
        text = f"{TOPIC.thread_title}\n\n{framing}{links_line}"
    if x_weighted_len(text) > MAX_TWEET:
        text = f"{TOPIC.thread_title}{links_line}"
    return text


def compose_landscape_root(today: datetime, unique_bills: list[dict],
                           state_counts: Counter, has_links: bool) -> str:
    juris_line = _format_jurisdictions_line(state_counts)
    links_line = "\n🔗 All bill links are in the last post." if has_links else ""
    text = (
        f"{TOPIC.thread_title}\n"
        f"Week of {_format_short(today)}, {today.year}\n\n"
        "Quiet stretch — no notable floor or executive action to flag from the "
        f"past month. Still tracking {TOPIC.topic_phrase} bills in "
        f"{juris_line}. A landscape check-in 🧵"
        f"{links_line}"
    )
    if x_weighted_len(text) > MAX_TWEET:
        text = (
            f"{TOPIC.thread_title}\n"
            f"Week of {_format_short(today)}, {today.year}\n\n"
            "Quiet stretch — no notable floor action from the past month, but "
            f"we're still tracking {TOPIC.topic_phrase} bills across the "
            f"states. A landscape check-in 🧵{links_line}"
        )
    if x_weighted_len(text) > MAX_TWEET:
        text = f"{TOPIC.thread_title}{links_line}"
    return text


def _landscape_closing_reply() -> str:
    return (
        "🔔 Many statehouses are between sessions or on recess this time of "
        "year. When bills start moving again, they'll show up in our daily "
        "posts and next week's digest. See you then."
    )


def build_highlight_replies(highlights: list[dict]) -> tuple[list[str], list[tuple[str, str]]]:
    """Compose one tweet per highlight (no per-bill link) and, alongside,
    collect every (label, url) pair for the final links post. Each highlight
    reply is budgeted/trimmed by compose_x_post with include_link_notice=False
    so the reclaimed link-notice space goes to a fuller summary."""
    replies: list[str] = []
    link_items: list[tuple[str, str]] = []
    for b in highlights:
        ensure_english_fields(b)
        headline = shorten_title(b)
        budget = x_summary_budget(b, headline, include_link_notice=False)
        summary = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
        text, url = compose_x_post(b, summary, headline=headline,
                                   include_link_notice=False)
        replies.append(text)
        if url:
            label = f"{b['state'] or '?'} {b['identifier']}"
            link_items.append((label, url))
        print(f"  prepared reply: {b['state']} {b['identifier']} "
              f"({b['action_date']}, score={b.get('_score', 0)}, "
              f"link={'yes' if url else 'no'})")
    return replies, link_items


def build_link_posts(link_items: list[tuple[str, str]]) -> list[str]:
    """Pack '<STATE ID>: <url>' lines into as few tweets as possible, counting
    each URL as 23 weighted chars (t.co). Returns [] when there are no links.
    Normally a single post; only overflows to continuation posts when there
    are more links than fit in 280 chars."""
    if not link_items:
        return []
    header = "🔗 Bill links:"
    cont_header = "🔗 Bill links (cont.):"
    posts: list[str] = []
    cur_header = header
    cur_lines: list[str] = []
    cur_weight = x_weighted_len(cur_header)

    def flush() -> None:
        posts.append(cur_header + "".join(cur_lines))

    for label, url in link_items:
        prefix = f"\n{label}: "
        line_weight = x_weighted_len(prefix) + TCO_WEIGHT
        if cur_lines and cur_weight + line_weight > MAX_TWEET:
            flush()
            cur_header = cont_header
            cur_lines = []
            cur_weight = x_weighted_len(cur_header)
        cur_lines.append(f"{prefix}{url}")
        cur_weight += line_weight
    flush()
    return posts


# ---------------------------------------------------------------------------
# Raw-artifact persistence (X weekly-digest folder)
# ---------------------------------------------------------------------------

def _save_digest_raw_records(bills: list[dict]) -> None:
    """Dump the verbatim bills.jsonl record (and extracted full text) for every
    bill featured in the digest to topics/<name>/<x_subdir>/weekly_digest/, so
    the Friday thread leaves the same self-contained trail as the daily feed.
    Saved up front so the artifact lands even if a reply post fails partway."""
    out_dir = TOPIC.x_weekly_digest_bills_raw_dir()
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

def post_thread(client, root_text: str, reply_texts: list[str]) -> None:
    """Post the root tweet, then chain each reply to the previous tweet so the
    whole digest reads as one X thread. A failed reply is logged and skipped;
    subsequent replies chain off the last tweet that DID land so a single
    hiccup doesn't orphan the rest of the thread."""
    print(f"\n--- ROOT ({x_weighted_len(root_text)} weighted chars) ---\n{root_text}\n---")
    if client is None:
        for i, text in enumerate(reply_texts, 1):
            print(f"\n--- REPLY {i} ({x_weighted_len(text)} weighted chars) ---\n{text}\n---")
        return

    resp = client.create_tweet(text=root_text)
    parent_id = resp.data["id"]
    print(f"  posted root: https://x.com/i/web/status/{parent_id}")
    time.sleep(X_THREAD_SLEEP)

    for i, text in enumerate(reply_texts, 1):
        print(f"\n--- REPLY {i} ({x_weighted_len(text)} weighted chars) ---\n{text}\n---")
        try:
            resp = client.create_tweet(text=text, in_reply_to_tweet_id=parent_id)
            parent_id = resp.data["id"]
            print(f"  posted reply {i}: https://x.com/i/web/status/{parent_id}")
            time.sleep(X_THREAD_SLEEP)
        except Exception as e:
            print(f"  ! reply {i} post failed: {e}", file=sys.stderr)
            if hasattr(e, "response") and e.response is not None:
                print(f"   Response body: {e.response.text}", file=sys.stderr)
            continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    missing = [
        n for n, v in (
            ("X_API_KEY", X_API_KEY),
            ("X_API_SECRET", X_API_SECRET),
            ("X_ACCESS_TOKEN", X_ACCESS_TOKEN),
            ("X_ACCESS_TOKEN_SECRET", X_ACCESS_TOKEN_SECRET),
        ) if not v
    ]
    if missing and not DRY_RUN:
        print(f"ERROR: missing X credentials: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"=== X weekly digest running for topic: {TOPIC.name} ===")
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

    client = None if DRY_RUN else build_client()

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
        replies, link_items = build_highlight_replies(highlights)
        link_posts = build_link_posts(link_items)
        root_text = compose_root(today, chosen_window, has_links=bool(link_posts))
        post_thread(client, root_text, replies + link_posts)
        print(f"\nDone. Posted X thread: 1 root + {len(replies)} highlight(s) "
              f"+ {len(link_posts)} links post(s) (window={chosen_window}d).")
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
    bill_replies, link_items = build_highlight_replies(recent_bills)
    link_posts = build_link_posts(link_items)
    root_text = compose_landscape_root(today, unique_bills, state_counts,
                                       has_links=bool(link_posts))
    # Closing note sits after the bill cards but BEFORE the links post, so the
    # links stay genuinely last (matching the root's promise).
    replies = bill_replies + [_landscape_closing_reply()] + link_posts
    post_thread(client, root_text, replies)
    print(f"\nDone. Posted X landscape thread with {len(replies)} reply post(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
