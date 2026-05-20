#!/usr/bin/env python3
"""
X/Twitter version of the poster. Mirrors post_to_bluesky.py's pipeline
(state detection, abstract/subjects extraction, freshness gate, same-day
dedup, weighted state selection, Ollama summary + headline) and posts to
X via tweepy. Uses a separate dedup file (bills_used_x.json) so X dedup
is independent of Bluesky's.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tweepy

from category import load_active_category
from post_to_bluesky import (
    _format_date,
    _normalize,
    _smart_truncate,
    best_display_text,
    extract_fields,
    format_action_line,
    link_for,
    load_bills,
    shorten_title,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = ROOT / "bills.jsonl"

CATEGORY = load_active_category()
STATE_FILE = ROOT / "categories" / CATEGORY.name / "bills_used_x.json"

POST_LIMIT = int(os.environ.get("POST_LIMIT", "2"))
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "150"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"

# X budgets every URL at 23 t.co chars regardless of actual length.
MAX_TWEET = 280
X_URL_LEN = 23

X_API_KEY = os.environ.get("X_API_KEY", "")
X_API_SECRET = os.environ.get("X_API_SECRET", "")
X_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.environ.get("X_ACCESS_TOKEN_SECRET", "")

print("Checking X credentials...")
print(f"  X_API_KEY present:             {bool(X_API_KEY) and len(X_API_KEY) > 10}")
print(f"  X_API_SECRET present:          {bool(X_API_SECRET) and len(X_API_SECRET) > 10}")
print(f"  X_ACCESS_TOKEN present:        {bool(X_ACCESS_TOKEN) and len(X_ACCESS_TOKEN) > 10}")
print(f"  X_ACCESS_TOKEN_SECRET present: {bool(X_ACCESS_TOKEN_SECRET) and len(X_ACCESS_TOKEN_SECRET) > 10}")


# ---------------------------------------------------------------------------
# State persistence (X-specific file)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"posted": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_x_post(b: dict, summary: str, headline: str = "") -> tuple[str, str]:
    emoji = CATEGORY.emoji_for(b)
    url = link_for(b)
    url_block = f"\n\n{url}" if url else ""

    state_label = b["state"] or "?"
    display = best_display_text(b, headline=headline).strip()
    summary = (summary or "").strip()

    summary_block = (
        f"\n\n{summary}"
        if summary and _normalize(summary) != _normalize(display)
        else ""
    )
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block = f"\n\n{action_line}" if action_line else ""

    prefix = f"{emoji} {state_label} {b['identifier']} — "
    prefix_len = len(prefix)
    head = f"{prefix}{display}"

    def assemble(h: str, s: str, a: str, u: str) -> str:
        return h + s + a + u

    text = assemble(head, summary_block, action_block, url_block)

    # The URL costs exactly X_URL_LEN in X's budget; the rest counts normally.
    url_cost = X_URL_LEN + 2 if url else 0  # +2 for the "\n\n" separator
    non_url_len = lambda t: len(t) - len(url_block) if url else len(t)

    # Trim order matches Bluesky: summary → display in head → action desc.
    if non_url_len(text) + url_cost > MAX_TWEET and summary_block:
        overflow = non_url_len(text) + url_cost - MAX_TWEET
        new_len = max(0, len(summary) - overflow - 1)
        if new_len > 20:
            summary = _smart_truncate(summary, new_len + 1)
            summary_block = f"\n\n{summary}"
        else:
            summary_block = ""
        text = assemble(head, summary_block, action_block, url_block)

    if non_url_len(text) + url_cost > MAX_TWEET:
        avail = MAX_TWEET - url_cost - len(summary_block) - len(action_block) - prefix_len - 1
        if avail > 0:
            display_trimmed = _smart_truncate(display, avail + 1)
        else:
            display_trimmed = ""
        head = f"{prefix}{display_trimmed}".rstrip(" —")
        text = assemble(head, summary_block, action_block, url_block)

    if non_url_len(text) + url_cost > MAX_TWEET and action_block and action_line:
        nice_date = _format_date(b["action_date"])
        if nice_date:
            date_prefix = f"{nice_date}: "
            if action_line.startswith(date_prefix):
                desc_part = action_line[len(date_prefix):].rstrip(".!?")
                overflow = non_url_len(text) + url_cost - MAX_TWEET
                new_len = max(0, len(desc_part) - overflow - 1)
                if new_len > 8:
                    action_line = date_prefix + _smart_truncate(desc_part, new_len + 1)
                    action_block = f"\n\n{action_line}"
                else:
                    action_line = ""
                    action_block = ""
            else:
                action_block = f"\n\n{action_line}"
        text = assemble(head, summary_block, action_block, url_block)

    return text, url


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------

def build_client() -> tweepy.Client:
    return tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_TOKEN_SECRET,
        wait_on_rate_limit=True,
    )


def post_tweet(client: tweepy.Client | None, text: str) -> bool:
    if DRY_RUN or client is None:
        print(f"  [DRY RUN] would tweet ({len(text)} chars):\n{text}\n")
        return True
    try:
        resp = client.create_tweet(text=text)
        tweet_id = resp.data["id"]
        print(f"  posted: https://x.com/i/web/status/{tweet_id}")
        return True
    except Exception as e:
        print(f"  ! tweet failed: {e}", file=sys.stderr)
        return False


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

    print(f"=== X GovBot running for category: {CATEGORY.name} ===")
    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    state = load_state()
    seen = set(state.get("posted", []))

    candidates: list[dict] = []
    for r in records:
        b = extract_fields(r)
        if not b:
            continue
        if not CATEGORY.matches(b):
            continue
        if b["dedup_key"] in seen:
            continue
        candidates.append(b)

    cutoff = datetime.now(timezone.utc).date()

    def _fresh(b: dict) -> bool:
        try:
            d = datetime.strptime(b["action_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return False
        return (cutoff - d).days <= MAX_ACTION_AGE_DAYS

    before = len(candidates)
    candidates = [b for b in candidates if _fresh(b)]
    dropped = before - len(candidates)
    if dropped:
        print(f"  dropped {dropped} stale update(s) older than {MAX_ACTION_AGE_DAYS} days.")

    # Same-day dedup: collapse multiple log entries for same bill on same day.
    unique_by_day: dict[str, dict] = {}
    for b in candidates:
        existing = unique_by_day.get(b["same_day_key"])
        if existing is None or len(b["action_desc"]) > len(existing["action_desc"]):
            unique_by_day[b["same_day_key"]] = b
    candidates = list(unique_by_day.values())

    print(f"Found {len(candidates)} new {CATEGORY.topic_phrase} bill update(s).")
    if not candidates:
        return 0

    state_counts = Counter(b["state"] or "?" for b in candidates)
    top = state_counts.most_common(15)
    print(f"  by state: {', '.join(f'{s}={n}' for s, n in top)}")

    def recency(b: dict) -> datetime:
        try:
            return datetime.strptime(b["action_date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            return datetime.min

    def has_desc(b: dict) -> bool:
        return bool((b["action_desc"] or "").strip())

    by_state: dict[str, dict] = {}
    for b in candidates:
        st = b["state"] or "?"
        cur = by_state.get(st)
        if cur is None or (has_desc(b), recency(b)) > (has_desc(cur), recency(cur)):
            by_state[st] = b
    reps = list(by_state.values())

    descriptive = [b for b in reps if has_desc(b)]
    stubs = [b for b in reps if not has_desc(b)]

    last_posted: dict[str, str] = state.get("state_last_posted", {})
    now = datetime.now(timezone.utc)

    def state_weight(b: dict) -> float:
        ts = last_posted.get(b["state"] or "?")
        if not ts:
            days = 180
        else:
            try:
                days = (now - datetime.fromisoformat(ts)).days
            except ValueError:
                days = 180
        return min(max(days, 0), 180) + 1

    def weighted_draw(pool: list[dict], k: int) -> list[dict]:
        pool = list(pool)
        picked: list[dict] = []
        while pool and len(picked) < k:
            weights = [state_weight(b) for b in pool]
            idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            picked.append(pool.pop(idx))
        return picked

    to_post = weighted_draw(descriptive, POST_LIMIT)
    if len(to_post) < POST_LIMIT:
        to_post.extend(weighted_draw(stubs, POST_LIMIT - len(to_post)))

    distinct_states = len({b["state"] or "?" for b in to_post})
    print(f"Pool: {len(descriptive)} state(s) with descriptive bills, {len(stubs)} stub-only.")
    print(f"Will post up to {POST_LIMIT}: posting {len(to_post)} from {distinct_states} state(s).")

    client = None if DRY_RUN else build_client()

    posted = 0
    for b in to_post:
        summary_text = summarize(b)
        headline = shorten_title(b)
        text, _url = compose_x_post(b, summary_text, headline=headline)

        print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
        print(text)
        print("---")

        if post_tweet(client, text):
            seen.add(b["dedup_key"])
            last_posted[b["state"] or "?"] = now.isoformat()
            posted += 1

    state["posted"] = sorted(seen)
    state["state_last_posted"] = last_posted
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"\nDone. Posted {posted} update(s). State saved to {STATE_FILE.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
