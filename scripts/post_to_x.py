#!/usr/bin/env python3
"""
X/Twitter version of the poster. Mirrors post_to_bluesky.py's pipeline
(state detection, abstract/subjects extraction, freshness gate, same-day
dedup, weighted state selection, Ollama summary + headline) and posts to
X via tweepy. All X state lives under topics/<name>/x/ (bills_used.json
plus a bills_raw/ artifact folder) so X dedup is independent of Bluesky's.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tweepy

from topic import load_active_topic
from post_to_bluesky import (
    _FILENAME_UNSAFE_RE,
    _format_date,
    _normalize,
    _slug,
    _smart_truncate,
    _strip_act_name_echo,
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

TOPIC = load_active_topic()
STATE_FILE = TOPIC.x_state_file_path()

POST_LIMIT = int(os.environ.get("POST_LIMIT", "2"))
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "150"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"

# Force-mode: when both FORCE_STATE and FORCE_BILL_ID are set, skip the random
# weighted draw and the topic-keyword/freshness gates and tweet exactly that
# one bill to the active topic's X account. Driven by the
# post_x_specific_bill workflow. FORCE_REPOST=1 bypasses the dedup gate so an
# already-tweeted bill can be re-posted.
FORCE_STATE = (os.environ.get("FORCE_STATE") or "").strip().lower()
FORCE_BILL_ID = (os.environ.get("FORCE_BILL_ID") or "").strip()
FORCE_REPOST = os.environ.get("FORCE_REPOST") == "1"

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


def save_raw_record(b: dict) -> None:
    """Write the verbatim bills.jsonl record for a posted bill to
    topics/<name>/x/bills_raw/<STATE>-<id>-<date>-<action_slug>.json so
    every X-posted action has a self-contained raw artifact alongside the
    dedup key in x/bills_used.json. Mirrors post_to_bluesky.save_raw_record."""
    raw = b.get("_raw")
    if not raw:
        return
    state = (b.get("state") or "XX")
    ident_raw = (b.get("identifier") or "unknown").strip()
    ident = _FILENAME_UNSAFE_RE.sub("_", ident_raw).strip("_")[:24] or "unknown"
    date = b.get("action_date") or "no-date"
    action_slug = _slug(b.get("action_desc") or "no-action", max_len=40) or "no-action"
    fname = f"{state}-{ident}-{date}-{action_slug}.json"
    out_dir = TOPIC.x_bills_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_x_post(b: dict, summary: str, headline: str = "") -> tuple[str, str]:
    emoji = TOPIC.emoji_for(b)
    url = link_for(b)
    url_block = f"\n\n{url}" if url else ""

    state_label = b["state"] or "?"
    display = best_display_text(b, headline=headline).strip()
    summary = (summary or "").strip()
    # Drop a leading act name from the summary when it just echoes the headline.
    summary = _strip_act_name_echo(summary, display)

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
        print(f"  [DRY RUN] skipping tweet ({len(text)} chars)")
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

def _norm_ident(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


def _post_forced_bill(records: list[dict], client: tweepy.Client | None) -> int:
    target_ident = _norm_ident(FORCE_BILL_ID)
    state_matches: list[dict] = []
    bill_matches: list[dict] = []
    for r in records:
        b = extract_fields(r)
        if not b:
            continue
        if (b["state"] or "").lower() != FORCE_STATE:
            continue
        state_matches.append(b)
        if _norm_ident(b["identifier"]) == target_ident:
            b["_raw"] = r
            bill_matches.append(b)

    if not bill_matches:
        seen_idents = sorted({b["identifier"] for b in state_matches})
        print(
            f"ERROR: no bill matching state={FORCE_STATE!r} "
            f"identifier={FORCE_BILL_ID!r} in {JSONL_PATH.name}.",
            file=sys.stderr,
        )
        if seen_idents:
            preview = ", ".join(seen_idents[:20])
            more = "" if len(seen_idents) <= 20 else f" (+{len(seen_idents) - 20} more)"
            print(f"  identifiers seen for {FORCE_STATE}: {preview}{more}",
                  file=sys.stderr)
        else:
            print(f"  no records at all for state {FORCE_STATE} in bills.jsonl.",
                  file=sys.stderr)
        return 2

    def _recency(b: dict) -> datetime:
        try:
            return datetime.strptime(b["action_date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            return datetime.min

    def _has_desc(b: dict) -> bool:
        return bool((b["action_desc"] or "").strip())

    bill_matches.sort(key=lambda b: (_has_desc(b), _recency(b)), reverse=True)
    b = bill_matches[0]

    state = load_state()
    seen = set(state.get("posted", []))
    if not FORCE_REPOST and b["dedup_key"] in seen:
        print(
            f"Bill {b['state']} {b['identifier']} action "
            f"{b['action_date']!r} is already in {STATE_FILE.name}. "
            f"Pass force_repost=true to re-post."
        )
        return 0

    if not TOPIC.matches(b):
        print(
            f"  NOTE: bill does not match topic '{TOPIC.name}' keywords — "
            f"tweeting anyway because force mode was requested."
        )

    print(f"Force-tweeting 1 bill to topic '{TOPIC.name}':")
    print(f"  {b['state']} {b['identifier']} ({b['action_date']})  "
          f"dedup_key={b['dedup_key']}")

    summary_text = summarize(b)
    headline = shorten_title(b)
    text, _url = compose_x_post(b, summary_text, headline=headline)

    print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
    print(text)
    print("---")

    if not post_tweet(client, text):
        return 1

    if DRY_RUN:
        print(f"\nDone. Dry run — no state written to "
              f"{STATE_FILE.relative_to(ROOT)}.")
        return 0

    seen.add(b["dedup_key"])
    last_posted = state.get("state_last_posted", {})
    last_posted[b["state"] or "?"] = datetime.now(timezone.utc).isoformat()
    try:
        save_raw_record(b)
    except OSError as e:
        print(f"  ! raw-record save failed: {e}", file=sys.stderr)

    state["posted"] = sorted(seen)
    state["state_last_posted"] = last_posted
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"\nDone. State saved to {STATE_FILE.relative_to(ROOT)}.")
    return 0


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

    print(f"=== X GovBot running for topic: {TOPIC.name} ===")
    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    if FORCE_STATE and FORCE_BILL_ID:
        client = None if DRY_RUN else build_client()
        return _post_forced_bill(records, client)

    state = load_state()
    seen = set(state.get("posted", []))

    candidates: list[dict] = []
    # Map same_day_key -> every dedup_key we saw for it, so when we post one
    # action we can burn its same-day siblings too. Without this, a bill with
    # N floor amendments on one day produces N distinct dedup_keys that leak
    # through one per run, letting a single bill monopolize its state slot
    # for N consecutive runs.
    same_day_siblings: dict[str, set[str]] = {}
    for r in records:
        b = extract_fields(r)
        if not b:
            continue
        if not TOPIC.matches(b):
            continue
        same_day_siblings.setdefault(b["same_day_key"], set()).add(b["dedup_key"])
        if b["dedup_key"] in seen:
            continue
        # Stash the source record so save_raw_record() can dump the verbatim
        # JSONL artifact once the bill is posted.
        b["_raw"] = r
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

    print(f"Found {len(candidates)} new {TOPIC.topic_phrase} bill update(s).")
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
            posted += 1
            if not DRY_RUN:
                seen.add(b["dedup_key"])
                seen.update(same_day_siblings.get(b["same_day_key"], ()))
                last_posted[b["state"] or "?"] = now.isoformat()
                try:
                    save_raw_record(b)
                except Exception as e:
                    print(f"  ! raw-record save failed: {e}", file=sys.stderr)

    if DRY_RUN:
        print(f"\nDone. Dry run — composed {posted} update(s), no state "
              f"written to {STATE_FILE.relative_to(ROOT)}.")
        return 0

    state["posted"] = sorted(seen)
    state["state_last_posted"] = last_posted
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"\nDone. Posted {posted} update(s). State saved to {STATE_FILE.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
