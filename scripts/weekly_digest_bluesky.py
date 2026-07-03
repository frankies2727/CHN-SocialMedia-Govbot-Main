#!/usr/bin/env python3
"""
Weekly digest thread: a single root post + up to 6 reply posts highlighting
the most significant bill activity for the active topic over the past
7 days.

Reuses helpers from post_to_bluesky.py — no duplication of the bill loader,
filter, link builder, summarizer, or Bluesky client. The only digest-specific
logic here is the significance scorer, the per-state cap, and the thread
chaining via record.reply.{root,parent}.

The active topic is selected via the BOT_TOPIC env var (see
scripts/topic.py). The digest's filter, copy, and Bluesky credentials
are all derived from that topic's config.yml.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from post_to_bluesky import (
    BlueskyClient,
    TOPIC,
    FETCH_OG_IMAGE,
    JSONL_PATH,
    MAX_POST,
    BSKY_HANDLE,
    BSKY_PASSWORD,
    STATE_FULL_NAME,
    _format_date,
    compose_post,
    ensure_english_fields,
    extract_fields,
    fetch_og_image,
    load_bills,
    prepare_image_for_bluesky,
    save_raw_record,
    shorten_title,
    summarize,
    summary_budget,
    MIN_SUMMARY_CHARS,
)

ROOT = Path(__file__).resolve().parent.parent

DIGEST_LOOKBACK_DAYS = int(os.environ.get("DIGEST_LOOKBACK_DAYS", "7"))
DIGEST_MAX_HIGHLIGHTS = int(os.environ.get("DIGEST_MAX_HIGHLIGHTS", "6"))
DIGEST_PER_STATE_CAP = int(os.environ.get("DIGEST_PER_STATE_CAP", "2"))
DIGEST_LANDSCAPE_CARDS = int(os.environ.get("DIGEST_LANDSCAPE_CARDS", "3"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"

# When a bill has no usable headline rewrite, its raw legalese title is the
# post head. Cap the head length used for summary budgeting so that title
# can't starve the plain-English summary out of the card; compose_post then
# trims the over-long title to fit (see prefer_summary). 60 chars keeps the
# summary above MIN_SUMMARY_CHARS even alongside a long action line, while
# still leaving enough of the title to identify the bill.
DIGEST_HEAD_CAP = int(os.environ.get("DIGEST_HEAD_CAP", "60"))

# When the primary 7-day window is empty we widen progressively so a quiet
# legislative week doesn't mean a silent feed. Once nothing turns up in the
# widest window either, we fall back to a landscape thread (see
# build_landscape_replies) so the bot always ships something informative.
LOOKBACK_FALLBACK_WINDOWS = [DIGEST_LOOKBACK_DAYS, 14, 30]


# ---------------------------------------------------------------------------
# Significance scoring
# ---------------------------------------------------------------------------

# Each tuple: (compiled regex, score). Highest matching tier wins.
# Govbot's action descriptions vary in capitalization and connective words
# ("Signed By The Governor" vs "signed by governor"), so the patterns below
# are deliberately tolerant of articles and case.
_SCORE_TIERS: list[tuple[re.Pattern, int]] = [
    # Became law / on governor's desk / now in force
    (re.compile(
        r"\b(?:signed|approved|delivered|sent|presented|transmitted)"
        r"\s+(?:(?:by|to)\s+)?(?:the\s+)?governor\b"
        r"|\b(?:became law|chaptered|public act|enrolled and signed|act took effect)\b"
        r"|\beffective(?:\s+immediately|\s+on\b|\s+\d|\s*$|\s*\.)",
        re.IGNORECASE), 100),
    # Passed a chamber
    (re.compile(
        r"\bpassed(?:\s+(?:by|the))?\s+(?:senate|house|assembly|chamber|both chambers)\b"
        r"|\bthird reading\s*[-,]?\s*passed\b"
        r"|\bpassed\s*[-,]?\s*third reading\b"
        r"|\bread third and passed\b"
        r"|\bresolution adopted\b|\bconference report adopted\b",
        re.IGNORECASE), 70),
    # Standalone "Passed" (often a chamber pass) -- one tier lower since terser
    (re.compile(r"^\s*passed\.?\s*$", re.IGNORECASE), 70),
    # News-worthy negative outcomes
    (re.compile(
        r"\bvetoed\b|\bveto override\b|\bdied in committee\b"
        r"|\bwithdrawn from further consideration\b|\bindefinitely postponed\b"
        r"|\blaid on the table\b",
        re.IGNORECASE), 60),
    # Committee progress
    (re.compile(
        r"\b(?:engrossed|ordered to third reading|placed on third reading"
        r"|reported favorably|reported out of committee|do pass"
        r"|reported with amendment)\b"
        r"|\bcommittee report,?\s+approving\b",
        re.IGNORECASE), 40),
    # Filed / introduced
    (re.compile(
        r"\b(?:introduced|first reading|filed|prefiled|pre-filed|read first time)\b",
        re.IGNORECASE), 20),
]


def score_action(action_desc: str) -> int:
    desc = action_desc or ""
    best = 5  # default for procedural / unmatched actions
    for pat, score in _SCORE_TIERS:
        if pat.search(desc):
            best = max(best, score)
    return best


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def in_lookback_window(action_date: str, today: datetime, days: int = DIGEST_LOOKBACK_DAYS) -> bool:
    if not action_date:
        return False
    try:
        d = datetime.strptime(action_date, "%Y-%m-%d")
    except ValueError:
        return False
    cutoff = today - timedelta(days=days)
    return d >= cutoff and d <= today


def collect_topic_bills(records: list[dict]) -> list[dict]:
    """Extract every active-topic bill log entry from the raw govbot records."""
    out: list[dict] = []
    for r in records:
        b = extract_fields(r)
        if b and TOPIC.matches(b):
            # Stash the source record so save_raw_record() can dump the
            # verbatim bills.jsonl line for any bill featured in the digest.
            b["_raw"] = r
            out.append(b)
    return out


def candidates_in_window(bills: list[dict], today: datetime, days: int) -> list[dict]:
    return [b for b in bills if in_lookback_window(b["action_date"], today, days)]


def select_highlights(candidates: list[dict],
                      max_highlights: int | None = DIGEST_MAX_HIGHLIGHTS,
                      per_state_cap: int = DIGEST_PER_STATE_CAP) -> list[dict]:
    """
    Pick the top ``max_highlights`` bills, capped at ``per_state_cap`` per
    state, sorted by (score desc, action_date desc).
    Collapses multiple actions for the same bill to the highest-scoring one.

    ``max_highlights`` and ``per_state_cap`` default to the module globals so
    the single-topic Bluesky/X digests call this unchanged. Pass
    ``max_highlights=None`` to get the whole ranked, per-state-capped list
    (used by the all-topics Threads digest, which then round-robins across
    topics itself instead of truncating each topic here).
    """
    # Collapse to one entry per bill (state|identifier), keep highest score.
    best_by_bill: dict[tuple[str, str], dict] = {}
    for b in candidates:
        key = (b["state"], b["identifier"])
        b["_score"] = score_action(b["action_desc"])
        existing = best_by_bill.get(key)
        if existing is None or b["_score"] > existing["_score"] or (
            b["_score"] == existing["_score"] and b["action_date"] > existing["action_date"]
        ):
            best_by_bill[key] = b
    bills = list(best_by_bill.values())

    bills.sort(key=lambda b: (b["_score"], b["action_date"]), reverse=True)

    picked: list[dict] = []
    per_state: Counter[str] = Counter()
    for b in bills:
        state = b["state"] or "?"
        if per_state[state] >= per_state_cap:
            continue
        picked.append(b)
        per_state[state] += 1
        if max_highlights is not None and len(picked) >= max_highlights:
            break
    return picked


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _format_short(d: datetime) -> str:
    abbrev = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
              7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    return f"{abbrev[d.month]} {d.day}"


def compose_root(today: datetime, total_updates: int, distinct_states: int,
                 window_days: int = DIGEST_LOOKBACK_DAYS) -> str:
    end = today
    start = today - timedelta(days=window_days - 1)
    range_str = f"{_format_short(start)}–{_format_short(end)}, {end.year}"
    if window_days <= DIGEST_LOOKBACK_DAYS:
        framing = (
            "Catching up on this week's bill activity across the states. "
            "Top highlights 🧵"
        )
    else:
        framing = (
            "Quieter past 7 days, so we widened the lens — here's the recent "
            "bill activity across the states. Top highlights 🧵"
        )
    text = (
        f"{TOPIC.thread_title}\n"
        f"{range_str}\n\n"
        f"{framing}"
    )
    if len(text) > MAX_POST:
        text = text[:MAX_POST - 1] + "…"
    return text


# ---------------------------------------------------------------------------
# Landscape (empty-week) thread
# ---------------------------------------------------------------------------

def _parse_iso(d: str) -> datetime:
    try:
        return datetime.strptime(d or "", "%Y-%m-%d")
    except ValueError:
        return datetime.min


def _landscape_unique_bills(all_bills: list[dict]) -> list[dict]:
    """Collapse all_bills to one entry per (state, identifier), keeping the
    most-recent action so downstream counts and recency picks aren't inflated
    by repeated log entries for the same bill."""
    latest_per_bill: dict[tuple[str, str], dict] = {}
    for b in all_bills:
        key = (b["state"], b["identifier"])
        prev = latest_per_bill.get(key)
        if prev is None or _parse_iso(b["action_date"]) > _parse_iso(prev["action_date"]):
            latest_per_bill[key] = b
    return list(latest_per_bill.values())


def _format_jurisdictions_line(state_counts: Counter) -> str:
    """One-line summary of who's being tracked, sorted by count desc.
    Lists jurisdictions without counts to keep the line clean:
        'NJ, WV, CO, MA, MN, TN, WA'
    """
    parts: list[str] = []
    for s, _ in state_counts.most_common():
        if not s:
            s = "?"
        parts.append(s)
    return ", ".join(parts)


def compose_landscape_root(today: datetime, unique_bills: list[dict],
                           state_counts: Counter) -> str:
    total_bills = len(unique_bills)
    distinct_states = len([s for s in state_counts if s])
    juris_line = _format_jurisdictions_line(state_counts)
    text = (
        f"{TOPIC.thread_title}\n"
        f"Week of {_format_short(today)}, {today.year}\n\n"
        "Quiet stretch — no notable floor or executive action to flag from the "
        f"past month. Still tracking {TOPIC.topic_phrase} bills in "
        f"{juris_line}. A landscape check-in 🧵"
    )
    # If the jurisdictions line pushes us over the cap, fall back to the
    # un-enriched copy rather than truncating mid-state-list.
    if len(text) > MAX_POST:
        text = (
            f"{TOPIC.thread_title}\n"
            f"Week of {_format_short(today)}, {today.year}\n\n"
            "Quiet stretch — no notable floor or executive action to flag from "
            f"the past month. But we're still tracking {TOPIC.topic_phrase} "
            "bills across the states. A landscape check-in 🧵"
        )
    if len(text) > MAX_POST:
        text = text[:MAX_POST - 1] + "…"
    return text


def _select_landscape_bills(unique_bills: list[dict], n: int) -> list[dict]:
    """Pick the N most-recent unique bills, preferring breadth: at most one
    per state on the first pass, allowing seconds only if we run out of
    distinct states. When nothing's moving on the floor, variety across
    jurisdictions is more informative than a deep dive into one statehouse."""
    by_recency = sorted(
        unique_bills,
        key=lambda b: _parse_iso(b["action_date"]),
        reverse=True,
    )
    picked: list[dict] = []
    seen_states: set[str] = set()
    leftovers: list[dict] = []
    for b in by_recency:
        state = b["state"] or "?"
        if state in seen_states:
            leftovers.append(b)
            continue
        picked.append(b)
        seen_states.add(state)
        if len(picked) >= n:
            return picked
    # Not enough distinct states — fill remaining slots from leftovers.
    for b in leftovers:
        picked.append(b)
        if len(picked) >= n:
            break
    return picked


def _landscape_closing_reply() -> str:
    return (
        "🔔 Many statehouses are between sessions or on recess this time of "
        "year. When bills start moving again, they'll show up in our daily "
        "posts and next week's digest. See you then."
    )


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------

def post_thread(client: BlueskyClient | None, root_text: str,
                replies: list[tuple[str, str, str, str, dict | None]]) -> None:
    """
    Post a root + chain of replies. Each reply tuple is
    (text, link_url, embed_title, embed_desc, thumb_blob_or_None).
    Used by both the highlights thread and the landscape fallback.
    """
    print(f"\n--- ROOT ({len(root_text)} chars) ---\n{root_text}\n---")

    if client is None:
        root_ref = {"uri": "[dry-run-root-uri]", "cid": "[dry-run-root-cid]"}
    else:
        result = client.post(root_text, link_url="", embed_title="", embed_desc="")
        root_ref = {"uri": result["uri"], "cid": result["cid"]}
        print(f"  posted root: {result['uri']}")
        time.sleep(2)

    parent_ref = root_ref
    for text, link, ec_title, ec_desc, thumb_blob in replies:
        print(f"\n--- REPLY ({len(text)} chars) ---\n{text}\n---")
        if link:
            print(f"    link: {link}")
        if client is None:
            continue
        try:
            reply = {"root": root_ref, "parent": parent_ref}
            result = client.post(text, link, ec_title, ec_desc,
                                 thumb_blob=thumb_blob, reply=reply)
            parent_ref = {"uri": result["uri"], "cid": result["cid"]}
            time.sleep(2)
        except Exception as e:
            print(f"  ! reply post failed: {e}", file=sys.stderr)
            continue


def _save_digest_raw_records(bills: list[dict]) -> None:
    """Dump the verbatim bills.jsonl record for every bill featured in the
    digest to topics/<name>/weekly_digest/bills_raw/. Saved up front so the
    artifact lands even if an individual reply post fails partway through
    the thread."""
    out_dir = TOPIC.weekly_digest_bills_raw_dir()
    for b in bills:
        try:
            save_raw_record(b, out_dir=out_dir)
        except OSError as e:
            print(f"  ! weekly digest raw-record save failed for "
                  f"{b.get('state')} {b.get('identifier')}: {e}", file=sys.stderr)


def _build_highlight_replies(client: BlueskyClient | None,
                             highlights: list[dict]) -> list[tuple[str, str, str, str, dict | None]]:
    replies: list[tuple[str, str, str, str, dict | None]] = []
    for b in highlights:
        ensure_english_fields(b)
        # Rewrite the bill's legalese title into a plain-English headline first
        # (same as the daily poster and the X digest), then budget the summary
        # against that head. When no headline is available, cap the budgeting
        # head so a long raw title can't starve the summary; prefer_summary
        # then keeps the layman summary and trims the title instead.
        headline = shorten_title(b)
        budget = summary_budget(b, headline, head_cap=DIGEST_HEAD_CAP)
        summary = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
        text, link, ec_title, ec_desc = compose_post(
            b, summary, headline=headline, prefer_summary=True)

        thumb_blob = None
        if link and FETCH_OG_IMAGE:
            print(f"  IMG: fetching og:image for {link}")
            fetched = fetch_og_image(link)
            if fetched:
                img_bytes_raw, mime_raw = fetched
                prepared = prepare_image_for_bluesky(img_bytes_raw, mime_raw)
                if prepared and client:
                    img_bytes, img_mime = prepared
                    thumb_blob = client.upload_blob(img_bytes, img_mime)

        print(f"  prepared reply: {b['state']} {b['identifier']} "
              f"({b['action_date']}, score={b.get('_score', 0)})")
        replies.append((text, link, ec_title, ec_desc, thumb_blob))
    return replies


def main() -> int:
    if not DRY_RUN and (not BSKY_HANDLE or not BSKY_PASSWORD):
        print("ERROR: BLUESKY_HANDLE and BLUESKY_APP_PASSWORD must be set.", file=sys.stderr)
        return 1

    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    all_bills = collect_topic_bills(records)
    if not all_bills:
        # Truly empty corpus — landscape stats would say "0 bills, 0 states",
        # which isn't informative. Skip rather than post nonsense.
        print(f"No {TOPIC.topic_phrase} bills found at all. Nothing to digest.")
        return 0

    # Try the primary 7-day window first, then widen if it's empty so a
    # quiet legislative week doesn't kill the digest.
    candidates: list[dict] = []
    chosen_window = LOOKBACK_FALLBACK_WINDOWS[0]
    for window in LOOKBACK_FALLBACK_WINDOWS:
        candidates = candidates_in_window(all_bills, today, window)
        print(f"Lookback {window}d: {len(candidates)} {TOPIC.topic_phrase} bill update(s).")
        if candidates:
            chosen_window = window
            break

    client = None if DRY_RUN else BlueskyClient(BSKY_HANDLE, BSKY_PASSWORD)

    if candidates:
        # Collapse repeated action-log entries to one per bill so the root copy
        # reflects how many distinct bills moved, not how many log rows were
        # written. Otherwise three actions on each of two bills reads as
        # "6 bill updates" while the thread only shows 2 replies.
        unique_bills = {(b["state"], b["identifier"]) for b in candidates}
        unique_count = len(unique_bills)
        distinct_states = len({s or "?" for s, _ in unique_bills})
        state_counts = Counter(s or "?" for s, _ in unique_bills)
        print(f"  unique bills: {unique_count} (from {len(candidates)} action entries)")
        print(f"  by state: {', '.join(f'{s}={n}' for s,n in state_counts.most_common(15))}")

        highlights = select_highlights(candidates)
        print(f"\nSelected {len(highlights)} highlight(s) (cap={DIGEST_MAX_HIGHLIGHTS}, "
              f"per-state-cap={DIGEST_PER_STATE_CAP}, window={chosen_window}d):")
        for b in highlights:
            print(f"  [{b['_score']:>3}] {b['state']} {b['identifier']} "
                  f"({b['action_date']}): {b['action_desc'][:70]}")

        root_text = compose_root(today, unique_count, distinct_states, chosen_window)
        _save_digest_raw_records(highlights)
        replies = _build_highlight_replies(client, highlights)
        post_thread(client, root_text, replies)
        print(f"\nDone. Posted thread with {len(highlights)} highlight(s) (window={chosen_window}d).")
        return 0

    # No floor activity in any window — ship a landscape thread so the
    # weekly slot still produces something informative. Show real bill cards
    # (title, summary, action line, link) for the most-recent unique bills
    # rather than a bare list of IDs.
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

    root_text = compose_landscape_root(today, unique_bills, state_counts)
    _save_digest_raw_records(recent_bills)
    bill_replies = _build_highlight_replies(client, recent_bills)
    closing = _landscape_closing_reply()
    replies = bill_replies + [(closing, "", "", "", None)]
    post_thread(client, root_text, replies)
    print(f"\nDone. Posted landscape thread with {len(replies)} reply post(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
