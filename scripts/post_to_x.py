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
import time

from topic import load_active_topic
from post_to_bluesky import (
    _FILENAME_UNSAFE_RE,
    _format_date,
    _normalize,
    _slug,
    _smart_truncate,
    _strip_act_name_echo,
    _strip_headline_echo,
    best_display_text,
    display_identifier,
    ensure_english_fields,
    extract_fields,
    format_action_line,
    format_no_match_error,
    link_for,
    load_bills,
    save_full_text,
    shorten_title,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = ROOT / "bills.jsonl"

TOPIC = load_active_topic()
STATE_FILE = TOPIC.x_state_file_path()

POST_LIMIT = int(os.environ.get("POST_LIMIT", "3"))
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "62"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"

# Persistence knobs, independent of DRY_RUN. Default both ON so existing
# schedules keep their dedup guarantees and raw-artifact trail. The
# post_x_specific_bill workflow exposes them as checkboxes so an operator
# can post a one-off without polluting the state file, or do a dry-run that
# still records the bill (e.g. mark a bill as "handled" without tweeting).
SAVE_STATE = os.environ.get("SAVE_STATE", "1") == "1"
SAVE_RAW = os.environ.get("SAVE_RAW", "1") == "1"

# Force-mode: when both FORCE_STATE and FORCE_BILL_ID are set, skip the random
# weighted draw and the topic-keyword/freshness gates and tweet exactly that
# one bill to the active topic's X account. Driven by the
# post_x_specific_bill workflow. FORCE_REPOST=1 bypasses the dedup gate so an
# already-tweeted bill can be re-posted.
FORCE_STATE = (os.environ.get("FORCE_STATE") or "").strip().lower()
FORCE_BILL_ID = (os.environ.get("FORCE_BILL_ID") or "").strip()
FORCE_REPOST = os.environ.get("FORCE_REPOST") == "1"

# X doubled the per-post weighted-character cap from 280 to 560, so posts can
# now carry a fuller summary/action block before any trimming kicks in. Kept as
# an env override (default 560) so a future limit change is a config tweak, not
# a code edit. Every cap check below reads MAX_TWEET, so bumping it here widens
# the budget everywhere (compose_x_post, x_summary_budget, and the weekly digest
# which imports this constant).
MAX_TWEET = int(os.environ.get("MAX_TWEET", "560"))

# Posted at the end of the bill tweet so readers know to look for the bill
# URL in the reply. Plain ASCII so Python len() matches X's weighted count.
REPLY_NOTICE = "Link to bill in reply."
REPLY_NOTICE_BLOCK = f"\n\n{REPLY_NOTICE}"

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

# X enforces its post cap (now 560, see MAX_TWEET) with twitter-text's
# *weighted* character count, not a raw code-point count. Most code points weigh
# 1, but anything outside a few Latin / General-Punctuation ranges weighs 2 —
# that includes essentially all emoji, the ₿ symbol (U+20BF), the … ellipsis
# (U+2026), and CJK text. Python's len() weighs every code point as 1, so a post
# that len()-measures right at the cap can be 1-2 over by X's count and bounce
# (the 403 / over-limit errors we see). x_weighted_len mirrors X's algorithm so
# every cap check below matches what X actually enforces.
_X_LIGHT_RANGES = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))


def x_weighted_len(text: str) -> int:
    total = 0
    for ch in text:
        cp = ord(ch)
        # Variation selectors (e.g. U+FE0F in "⚖️" / "✈️") have no width of
        # their own — they're folded into the preceding emoji, which X already
        # counts as a single weight-2 glyph. Count them as 0 so a base+selector
        # emoji totals 2 (matching X) rather than 4.
        if 0xFE00 <= cp <= 0xFE0F:
            continue
        if any(lo <= cp <= hi for lo, hi in _X_LIGHT_RANGES):
            total += 1
        else:
            total += 2
    return total


def _weighted_truncate(text: str, max_weight: int) -> str:
    """Like _smart_truncate, but the result's X *weighted* length is
    guaranteed <= max_weight (not just its code-point len). Needed because the
    model's summary can itself contain weight-2 code points (…, emoji, CJK)
    that len()-based truncation would undercount and let slip over the cap."""
    text = (text or "").strip()
    if max_weight <= 0:
        return ""
    if x_weighted_len(text) <= max_weight:
        return text
    # Weighted length >= code-point len, so a fitting prefix has at most
    # max_weight code points. Start from that guess and shrink until the
    # (word-boundary, possibly …-suffixed) cut fits X's weighted count.
    n = min(len(text), max_weight)
    while n > 0:
        cut = _smart_truncate(text, n)
        if x_weighted_len(cut) <= max_weight:
            return cut
        n -= 1
    return ""


def x_summary_budget(b: dict, headline: str, include_link_notice: bool = True) -> int:
    """Character budget available for the summary block inside an X post,
    given the head (emoji + state + id + display), the action line, and
    the ``Link to bill in reply.`` notice that will sit alongside it.
    Returned so the caller can ask the LLM for a summary that fits
    cleanly instead of relying on compose_x_post's post-hoc trim — when
    the trim fires it just lops off the tail of the model's sentence at
    a word boundary, which usually drops the most concrete clause.

    Set ``include_link_notice=False`` when the bill's URL won't be posted
    as a same-tweet reply (e.g. the weekly digest collects every link into
    a single final post), so the summary reclaims that ~22 chars."""
    emoji = TOPIC.emoji_for(b)
    state_label = b["state"] or "?"
    ident_disp = display_identifier(b["state"], b["identifier"])
    display = best_display_text(b, headline=headline).strip()
    prefix = f"{emoji} {state_label} {ident_disp} — "
    head_len = x_weighted_len(prefix) + x_weighted_len(display)
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block_len = x_weighted_len(f"\n\n{action_line}") if action_line else 0
    url = link_for(b)
    notice_block_len = x_weighted_len(REPLY_NOTICE_BLOCK) if (url and include_link_notice) else 0
    # The summary itself is preceded by "\n\n" (2 chars). Anything below
    # MIN_SUMMARY_CHARS isn't worth asking the model for — drop the block.
    summary_sep_len = 2
    return MAX_TWEET - head_len - action_block_len - notice_block_len - summary_sep_len


# Below this floor the summary block is too short to add useful detail
# beyond the headline — return "" from the caller so the LLM round-trip
# is skipped and the post composes without a summary block at all.
MIN_SUMMARY_CHARS = 60


def compose_x_post(b: dict, summary: str, headline: str = "",
                   include_link_notice: bool = True) -> tuple[str, str]:
    """Return (main_tweet_text, bill_url). The bill URL is NOT in the main
    text — it's intended to be posted as a reply so the main tweet doesn't
    spend ~25 chars of t.co budget on a link. The main text gets a
    ``Link to bill in reply.`` notice when a URL is available.

    Set ``include_link_notice=False`` to drop that notice — used by the
    weekly digest, which threads every bill's link into a single final
    post rather than a per-bill reply, so the per-bill notice would be
    misleading. The URL is still returned so the caller can collect it."""
    emoji = TOPIC.emoji_for(b)
    url = link_for(b)
    notice_block = REPLY_NOTICE_BLOCK if (url and include_link_notice) else ""

    state_label = b["state"] or "?"
    ident_disp = display_identifier(b["state"], b["identifier"])
    display = best_display_text(b, headline=headline).strip()
    summary = (summary or "").strip()
    # Drop a leading act name from the summary when it just echoes the headline.
    summary = _strip_act_name_echo(summary, display)
    # Drop a whole leading sentence that just paraphrases the headline, matching
    # Bluesky's compose_post so X doesn't ship a redundant restatement.
    summary = _strip_headline_echo(summary, display)

    summary_block = (
        f"\n\n{summary}"
        if summary and _normalize(summary) != _normalize(display)
        else ""
    )
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block = f"\n\n{action_line}" if action_line else ""

    prefix = f"{emoji} {state_label} {ident_disp} — "
    head = f"{prefix}{display}"

    def assemble(h: str, s: str, a: str, n: str) -> str:
        return h + s + a + n

    text = assemble(head, summary_block, action_block, notice_block)

    # Trim order matches Bluesky: summary → display in head → action desc.
    # Every cap check uses X's weighted character count (x_weighted_len) and
    # every cut goes through _weighted_truncate, so weight-2 code points
    # (emoji, …, ₿, CJK) can't push the post past X's real cap the way
    # raw len() math silently let them.
    if x_weighted_len(text) > MAX_TWEET and summary_block:
        fixed = x_weighted_len(assemble(head, "\n\n", action_block, notice_block))
        summary = _weighted_truncate(summary, MAX_TWEET - fixed)
        summary_block = f"\n\n{summary}" if len(summary) > 20 else ""
        text = assemble(head, summary_block, action_block, notice_block)

    if x_weighted_len(text) > MAX_TWEET:
        fixed = x_weighted_len(assemble(prefix, summary_block, action_block, notice_block))
        display_trimmed = _weighted_truncate(display, MAX_TWEET - fixed)
        head = f"{prefix}{display_trimmed}".rstrip(" —")
        text = assemble(head, summary_block, action_block, notice_block)

    if x_weighted_len(text) > MAX_TWEET and action_block and action_line:
        nice_date = _format_date(b["action_date"])
        if nice_date:
            date_prefix = f"{nice_date}: "
            if action_line.startswith(date_prefix):
                desc_part = action_line[len(date_prefix):].rstrip(".!?")
                fixed = x_weighted_len(
                    assemble(head, summary_block, f"\n\n{date_prefix}", notice_block)
                )
                new_desc = _weighted_truncate(desc_part, MAX_TWEET - fixed)
                if len(new_desc) > 8:
                    action_line = date_prefix + new_desc
                    action_block = f"\n\n{action_line}"
                else:
                    action_line = ""
                    action_block = ""
            else:
                action_block = f"\n\n{action_line}"
        text = assemble(head, summary_block, action_block, notice_block)

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


def post_tweet(client: tweepy.Client | None, text: str, reply_url: str = "") -> bool:
    """Post the main tweet, then (if ``reply_url`` is given) post a reply
    containing just that URL. Returns True iff the main tweet was posted —
    a failed reply is logged but does not flip the result, so the bill is
    still recorded as posted and the URL ends up only missing from the
    thread (better than re-posting the whole bill on the next run)."""
    if DRY_RUN or client is None:
        print(f"  [DRY RUN] skipping tweet ({x_weighted_len(text)} weighted chars)")
        if reply_url:
            print(f"  [DRY RUN] skipping link reply: {reply_url}")
        return True
    try:
        resp = client.create_tweet(text=text)
        tweet_id = resp.data["id"]
        print(f"  posted: https://x.com/i/web/status/{tweet_id}")
    except Exception as e:
        print(f" ! tweet failed: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response body: {e.response.text}", file=sys.stderr)
        return False
    if reply_url:
        try:
            reply = client.create_tweet(text=reply_url, in_reply_to_tweet_id=tweet_id)
            reply_id = reply.data["id"]
            print(f"  link reply: https://x.com/i/web/status/{reply_id}")
        except Exception as e:
            print(f"  ! link reply failed: {e}", file=sys.stderr)
    return True


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
        format_no_match_error(
            state=FORCE_STATE,
            target_ident=FORCE_BILL_ID,
            state_matches=state_matches,
            source_filename=JSONL_PATH.name,
            raw_records=records,
        )
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
    if not FORCE_REPOST and (b["dedup_key"] in seen or b["same_day_key"] in seen):
        print(
            f"Bill {b['state']} {b['identifier']} action "
            f"{b['action_date']!r} (or another action for this bill on the "
            f"same day) is already in {STATE_FILE.name}. "
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

    ensure_english_fields(b)
    headline = shorten_title(b)
    budget = x_summary_budget(b, headline)
    summary_text = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
    text, url = compose_x_post(b, summary_text, headline=headline)

    print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
    print(text)
    if url:
        print(f"  ↳ reply: {url}")
    print("---")

    if not post_tweet(client, text, reply_url=url):
        return 1

    if SAVE_RAW:
        try:
            save_raw_record(b)
            save_full_text(b, out_dir=TOPIC.x_bills_full_text_dir())
        except OSError as e:
            print(f"  ! raw-record save failed: {e}", file=sys.stderr)
    else:
        print("  SAVE_RAW=0 — skipping bills_raw artifact.")

    if SAVE_STATE:
        seen.add(b["dedup_key"])
        # Also remember the bill+day so no other action for this bill on this
        # same day can be posted again later.
        seen.add(b["same_day_key"])
        last_posted = state.get("state_last_posted", {})
        last_posted[b["state"] or "?"] = datetime.now(timezone.utc).isoformat()
        state["posted"] = sorted(seen)
        state["state_last_posted"] = last_posted
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"\nDone. State saved to {STATE_FILE.relative_to(ROOT)}.")
    else:
        print(f"\nDone. SAVE_STATE=0 — {STATE_FILE.relative_to(ROOT)} "
              f"left unchanged.")
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
        # Skip if we've already posted this exact action (dedup_key) OR any
        # other action for this same bill on this same day (same_day_key).
        # The same_day_key guard stops a second post when another log entry
        # for the same bill+day arrives on a later run.
        if b["dedup_key"] in seen or b["same_day_key"] in seen:
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

    # Balanced draw across topic.keyword_groups: when the topic defines named
    # sub-buckets (e.g. ai_data_centers splits "ai_data_centers" vs "crypto")
    # and POST_LIMIT can cover at least one per bucket, guarantee one slot per
    # non-empty bucket before the unrestricted weighted draw fills the rest.
    # If a bucket has no candidates this run, its slot falls through to the
    # general pool rather than being skipped, so the run still posts
    # POST_LIMIT bills whenever there are enough candidates total.
    groups = TOPIC.keyword_groups
    to_post: list[dict] = []
    if groups and POST_LIMIT >= len(groups):
        by_group: dict[str, list[dict]] = {g: [] for g in groups}
        for b in descriptive:
            g = TOPIC.primary_group_for(b)
            if g in by_group:
                by_group[g].append(b)
        bucket_summary = ", ".join(f"{g}={len(by_group[g])}" for g in groups)
        print(f"  keyword_groups buckets (descriptive): {bucket_summary}")
        picked_ids: set[str] = set()
        for g in groups:
            bucket = [b for b in by_group[g] if b["dedup_key"] not in picked_ids]
            if not bucket:
                print(f"  bucket {g!r} empty — slot will fall through to general pool.")
                continue
            for b in weighted_draw(bucket, 1):
                to_post.append(b)
                picked_ids.add(b["dedup_key"])
        if len(to_post) < POST_LIMIT:
            rest = [b for b in descriptive if b["dedup_key"] not in picked_ids]
            for b in weighted_draw(rest, POST_LIMIT - len(to_post)):
                to_post.append(b)
                picked_ids.add(b["dedup_key"])
    else:
        to_post = weighted_draw(descriptive, POST_LIMIT)

    if len(to_post) < POST_LIMIT:
        picked_ids = {b["dedup_key"] for b in to_post}
        stub_pool = [b for b in stubs if b["dedup_key"] not in picked_ids]
        to_post.extend(weighted_draw(stub_pool, POST_LIMIT - len(to_post)))

    distinct_states = len({b["state"] or "?" for b in to_post})
    print(f"Pool: {len(descriptive)} state(s) with descriptive bills, {len(stubs)} stub-only.")
    print(f"Will post up to {POST_LIMIT}: posting {len(to_post)} from {distinct_states} state(s).")

    client = None if DRY_RUN else build_client()

    posted = 0
    for b in to_post:
        ensure_english_fields(b)
        headline = shorten_title(b)
        budget = x_summary_budget(b, headline)
        summary_text = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
        text, url = compose_x_post(b, summary_text, headline=headline)

        print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
        print(text)
        if url:
            print(f"  ↳ reply: {url}")
        print("---")

        if post_tweet(client, text, reply_url=url):
            posted += 1
            if SAVE_STATE:
                seen.add(b["dedup_key"])
                # Remember the bill+day itself, plus every sibling action for
                # that bill+day we already know about, so this bill can't be
                # posted again today — even if a new same-day action shows up
                # next run.
                seen.add(b["same_day_key"])
                seen.update(same_day_siblings.get(b["same_day_key"], ()))
                last_posted[b["state"] or "?"] = now.isoformat()
            if SAVE_RAW:
                try:
                    save_raw_record(b)
                    save_full_text(b, out_dir=TOPIC.x_bills_full_text_dir())
                except Exception as e:
                    print(f"  ! raw-record save failed: {e}", file=sys.stderr)

            # ←←← THIS IS THE FIX FOR THE 403 "You are not permitted" error
            time.sleep(27)   # X is very picky about rapid successive posts

    if not SAVE_RAW:
        print("  SAVE_RAW=0 — bills_raw artifacts not written.")

    if SAVE_STATE:
        state["posted"] = sorted(seen)
        state["state_last_posted"] = last_posted
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"\nDone. Posted {posted} update(s). State saved to "
              f"{STATE_FILE.relative_to(ROOT)}.")
    else:
        print(f"\nDone. Posted {posted} update(s). SAVE_STATE=0 — "
              f"{STATE_FILE.relative_to(ROOT)} left unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
