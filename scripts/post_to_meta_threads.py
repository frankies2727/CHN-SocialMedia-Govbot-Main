#!/usr/bin/env python3
"""
Threads (Meta) version of the poster. Mirrors post_to_bluesky.py's pipeline
(state detection, abstract/subjects extraction, freshness gate, same-day
dedup, weighted state selection, Ollama summary + headline) and publishes to
Threads via its Graph API. All Threads state lives under
topics/<name>/meta-threads/ (bills_used.json plus a bills_raw/ artifact folder)
so Threads dedup is independent of Bluesky's and X's.

Account model: a single Threads account (e.g. chn.govbot) dedicated to one
topic. The topic is selected via BOT_TOPIC like the other posters, and the
account's credentials come from two env vars / repo secrets:

    THREADS_ACCESS_TOKEN   long-lived token (see scripts/refresh_meta_threads_token.py)
    THREADS_USER_ID        the numeric Threads user id

Publishing is a two-step Graph API call: first create a media container
(media_type=TEXT, plus an optional link_attachment for the bill URL), then
publish that container. The bill link goes in link_attachment rather than the
post body, so it renders as a link preview and never spends any of the 500
character budget.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from topic import load_active_topic
from account_ledger import AccountLedger
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
    load_normalized_bills,
    save_full_text,
    shorten_title,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = ROOT / "bills.jsonl"

TOPIC = load_active_topic()
STATE_FILE = TOPIC.threads_state_file_path()

POST_LIMIT = int(os.environ.get("POST_LIMIT", "2"))
# The Threads account now spans every topic, so the daily poster loops over all
# topics. RUN_POST_LIMIT caps how many posts go to the single account per run
# across ALL topics combined (POST_LIMIT still caps each topic's own turn); once
# the account hits this ceiling, later topics in the loop exit early. The next
# run starts the count fresh.
RUN_POST_LIMIT = int(os.environ.get("RUN_POST_LIMIT", "3"))
# Account-level (cross-topic) ledger lives under account_state/<platform>/.
PLATFORM = "meta-threads"
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "62"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"

# Persistence knobs, independent of DRY_RUN. Default both ON so the daily
# schedule keeps its dedup guarantees and raw-artifact trail. Mirrors the
# post_to_x.py knobs.
SAVE_STATE = os.environ.get("SAVE_STATE", "1") == "1"
SAVE_RAW = os.environ.get("SAVE_RAW", "1") == "1"

# Force-mode: when both FORCE_STATE and FORCE_BILL_ID are set, skip the random
# weighted draw and the topic-keyword/freshness gates and post exactly that one
# bill to the Threads account. FORCE_REPOST=1 bypasses the dedup gate so an
# already-posted bill can be re-posted.
FORCE_STATE = (os.environ.get("FORCE_STATE") or "").strip().lower()
FORCE_BILL_ID = (os.environ.get("FORCE_BILL_ID") or "").strip()
FORCE_REPOST = os.environ.get("FORCE_REPOST") == "1"

# Threads' post body limit is 500 characters; keep some slack so a stray
# grapheme can't bounce the publish. The bill URL is sent as link_attachment,
# not in the body, so it costs nothing against this budget.
MAX_THREADS = 490
# Below this floor the summary block adds too little beyond the headline to be
# worth an LLM round-trip — compose without a summary block instead.
MIN_SUMMARY_CHARS = 80

THREADS_API = "https://graph.threads.net/v1.0"
THREADS_TIMEOUT = int(os.environ.get("THREADS_TIMEOUT", "30"))
# Threads occasionally needs a beat between container creation and publish.
THREADS_PUBLISH_RETRIES = 3

THREADS_ACCESS_TOKEN = os.environ.get("THREADS_ACCESS_TOKEN", "")
THREADS_USER_ID = os.environ.get("THREADS_USER_ID", "")

print("Checking Threads credentials...")
print(f"  THREADS_ACCESS_TOKEN present: {bool(THREADS_ACCESS_TOKEN) and len(THREADS_ACCESS_TOKEN) > 20}")
print(f"  THREADS_USER_ID present:      {bool(THREADS_USER_ID)}")


# ---------------------------------------------------------------------------
# State persistence (Threads-specific file)
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
    topics/<name>/threads/bills_raw/<STATE>-<id>-<date>-<action_slug>.json so
    every Threads-posted action has a self-contained raw artifact alongside the
    dedup key in threads/bills_used.json. Mirrors post_to_x.save_raw_record."""
    raw = b.get("_raw")
    if not raw:
        return
    state = (b.get("state") or "XX")
    ident_raw = (b.get("identifier") or "unknown").strip()
    ident = _FILENAME_UNSAFE_RE.sub("_", ident_raw).strip("_")[:24] or "unknown"
    date = b.get("action_date") or "no-date"
    action_slug = _slug(b.get("action_desc") or "no-action", max_len=40) or "no-action"
    fname = f"{state}-{ident}-{date}-{action_slug}.json"
    out_dir = TOPIC.threads_bills_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def topic_header(b: dict) -> str:
    """Topic label line shown atop each daily Threads post. The single Threads
    account spans every topic, so the daily feed interleaves bills from Labor,
    Healthcare, Transportation, etc. Tagging each post with its topic's display
    name (led by the bill's emoji) tells the mixed feed apart at a glance."""
    return f"{TOPIC.emoji_for(b)} {TOPIC.display_name}"


def threads_summary_budget(b: dict, headline: str, include_topic: bool = False) -> int:
    """Character budget available for the summary block inside a Threads post,
    given the head (emoji + state + id + display, plus an optional topic line)
    and the action line. The bill URL is NOT counted — it ships as
    link_attachment, outside the body. Returned so the caller can ask the LLM
    for a summary that fits cleanly instead of relying on compose_threads_post's
    post-hoc trim."""
    emoji = TOPIC.emoji_for(b)
    state_label = b["state"] or "?"
    ident_disp = display_identifier(b["state"], b["identifier"])
    display = best_display_text(b, headline=headline).strip()
    if include_topic:
        # Topic line + "\n" + clean bill line (the emoji rides the topic line).
        head_len = len(topic_header(b)) + 1 + len(f"{state_label} {ident_disp} — ") + len(display)
    else:
        head_len = len(f"{emoji} {state_label} {ident_disp} — ") + len(display)
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block_len = len(f"\n\n{action_line}") if action_line else 0
    summary_sep_len = 2  # the "\n\n" before the summary block
    return MAX_THREADS - head_len - action_block_len - summary_sep_len


def compose_threads_post(b: dict, summary: str, headline: str = "",
                         include_topic: bool = False) -> tuple[str, str]:
    """Return (post_text, bill_url). The URL is NOT in post_text — the caller
    passes it as link_attachment so it renders as a preview and costs nothing
    against the 500-char body budget. Trim order mirrors Bluesky/X:
    summary -> display in head -> action line.

    When include_topic is set, a topic label line is prepended so the post
    declares which topic the bill belongs to — used by the daily poster, whose
    single Threads account spans every topic. The weekly digest leaves it off
    because its root post already names the topic for the whole thread."""
    emoji = TOPIC.emoji_for(b)
    url = link_for(b)

    state_label = b["state"] or "?"
    ident_disp = display_identifier(b["state"], b["identifier"])
    display = best_display_text(b, headline=headline).strip()
    summary = (summary or "").strip()
    # Drop a leading act name / whole leading sentence that just echoes the
    # headline, matching compose_post / compose_x_post.
    summary = _strip_act_name_echo(summary, display)
    summary = _strip_headline_echo(summary, display)

    summary_block = (
        f"\n\n{summary}"
        if summary and _normalize(summary) != _normalize(display)
        else ""
    )
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block = f"\n\n{action_line}" if action_line else ""

    # head_lead is everything before the (trimmable) display title. With a topic
    # line the emoji leads that line and the bill line stays clean; otherwise the
    # emoji leads the bill line as before.
    if include_topic:
        head_lead = f"{topic_header(b)}\n{state_label} {ident_disp} — "
    else:
        head_lead = f"{emoji} {state_label} {ident_disp} — "
    head = f"{head_lead}{display}"

    def assemble(h: str, s: str, a: str) -> str:
        return h + s + a

    text = assemble(head, summary_block, action_block)

    if len(text) > MAX_THREADS and summary_block:
        fixed = len(assemble(head, "\n\n", action_block))
        summary = _smart_truncate(summary, MAX_THREADS - fixed)
        summary_block = f"\n\n{summary}" if len(summary) > 20 else ""
        text = assemble(head, summary_block, action_block)

    if len(text) > MAX_THREADS:
        fixed = len(assemble(head_lead, summary_block, action_block))
        display_trimmed = _smart_truncate(display, MAX_THREADS - fixed)
        head = f"{head_lead}{display_trimmed}".rstrip(" —")
        text = assemble(head, summary_block, action_block)

    if len(text) > MAX_THREADS and action_block:
        fixed = len(assemble(head, summary_block, "\n\n"))
        action_trimmed = _smart_truncate(action_line, MAX_THREADS - fixed)
        action_block = f"\n\n{action_trimmed}" if len(action_trimmed) > 8 else ""
        text = assemble(head, summary_block, action_block)

    return text, url


# ---------------------------------------------------------------------------
# Posting (Threads two-step container -> publish)
# ---------------------------------------------------------------------------

def _create_container(text: str, link_url: str = "", reply_to_id: str = "") -> str | None:
    params = {
        "media_type": "TEXT",
        "text": text,
        "access_token": THREADS_ACCESS_TOKEN,
    }
    if link_url:
        params["link_attachment"] = link_url
    # reply_to_id chains this post under an existing one — used by the weekly
    # digest to thread its highlight replies under the root post.
    if reply_to_id:
        params["reply_to_id"] = reply_to_id
    try:
        resp = requests.post(
            f"{THREADS_API}/{THREADS_USER_ID}/threads",
            data=params,
            timeout=THREADS_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("id")
    except Exception as e:
        print(f" ! container creation failed: {e}", file=sys.stderr)
        if getattr(e, "response", None) is not None:
            print(f"   Response body: {e.response.text}", file=sys.stderr)
        return None


def _publish_container(creation_id: str) -> str | None:
    for attempt in range(1, THREADS_PUBLISH_RETRIES + 1):
        try:
            resp = requests.post(
                f"{THREADS_API}/{THREADS_USER_ID}/threads_publish",
                data={"creation_id": creation_id, "access_token": THREADS_ACCESS_TOKEN},
                timeout=THREADS_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("id")
        except Exception as e:
            if attempt < THREADS_PUBLISH_RETRIES:
                # Container may still be processing — back off and retry.
                time.sleep(5 * attempt)
                continue
            print(f" ! publish failed: {e}", file=sys.stderr)
            if getattr(e, "response", None) is not None:
                print(f"   Response body: {e.response.text}", file=sys.stderr)
            return None
    return None


def publish_post(text: str, link_url: str = "", reply_to_id: str = "") -> str | None:
    """Create a Threads container then publish it, returning the published
    post's media id (or None on failure). Shared by the daily poster and the
    weekly digest; reply_to_id chains the post under an existing one."""
    creation_id = _create_container(text, link_url, reply_to_id)
    if not creation_id:
        return None
    return _publish_container(creation_id)


def post_thread(text: str, link_url: str = "") -> bool:
    """Create a Threads container then publish it. Returns True iff the post
    was published. The bill URL (link_url) is sent as link_attachment."""
    if DRY_RUN:
        print(f"  [DRY RUN] skipping Threads post ({len(text)} chars)")
        if link_url:
            print(f"  [DRY RUN] link_attachment: {link_url}")
        return True
    media_id = publish_post(text, link_url)
    if not media_id:
        return False
    print(f"  posted to Threads (media id {media_id})")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _norm_ident(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


def _post_forced_bill(records: list[dict]) -> int:
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
    ledger = AccountLedger(PLATFORM)
    seen = set(state.get("posted", []))
    if not FORCE_REPOST and (b["dedup_key"] in (seen | ledger.seen)
                             or b["same_day_key"] in (seen | ledger.seen)):
        print(
            f"Bill {b['state']} {b['identifier']} action "
            f"{b['action_date']!r} (or another action for this bill on the "
            f"same day) is already posted to the {PLATFORM} account. "
            f"Pass force_repost=true to re-post."
        )
        return 0

    if not TOPIC.matches(b):
        print(
            f"  NOTE: bill does not match topic '{TOPIC.name}' keywords — "
            f"posting anyway because force mode was requested."
        )

    print(f"Force-posting 1 bill to Threads topic '{TOPIC.name}':")
    print(f"  {b['state']} {b['identifier']} ({b['action_date']})  "
          f"dedup_key={b['dedup_key']}")

    ensure_english_fields(b)
    headline = shorten_title(b)
    budget = threads_summary_budget(b, headline, include_topic=True)
    summary_text = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
    text, url = compose_threads_post(b, summary_text, headline=headline, include_topic=True)

    print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
    print(text)
    if url:
        print(f"  ↳ link_attachment: {url}")
    print("---")

    if not post_thread(text, link_url=url):
        return 1

    if SAVE_RAW:
        try:
            save_raw_record(b)
            save_full_text(b, out_dir=TOPIC.threads_bills_full_text_dir())
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
        ledger.record({b["dedup_key"], b["same_day_key"]})
        ledger.save()
        print(f"\nDone. State saved to {STATE_FILE.relative_to(ROOT)}.")
    else:
        print(f"\nDone. SAVE_STATE=0 — {STATE_FILE.relative_to(ROOT)} "
              f"left unchanged.")
    return 0


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

    print(f"=== Threads GovBot running for topic: {TOPIC.name} ===")

    if FORCE_STATE and FORCE_BILL_ID:
        records = load_bills(JSONL_PATH)
        if not records:
            return 0
        return _post_forced_bill(records)

    # Normalized bills (extract_fields applied once, with _raw attached). When the
    # workflow prebuilds BILLS_NORMALIZED, every topic in the loop loads the same
    # array instead of re-parsing bills.jsonl + re-extracting per topic.
    bills = load_normalized_bills()
    if not bills:
        return 0

    state = load_state()
    # `seen` is this topic's own dedup set (persisted back to its bills_used.json).
    # `seen_all` additionally excludes anything already posted to the account under
    # ANY topic (cross-topic guard) — used only for filtering, never saved.
    ledger = AccountLedger(PLATFORM)
    seen = set(state.get("posted", []))
    seen_all = seen | ledger.seen

    # Per-run cap across all topics on the single account.
    remaining = ledger.remaining_this_run(RUN_POST_LIMIT)
    effective_limit = min(POST_LIMIT, remaining) if SAVE_STATE else POST_LIMIT
    if SAVE_STATE and effective_limit <= 0:
        print(f"Run post limit reached ({ledger.posted_this_run()}/{RUN_POST_LIMIT} "
              f"posted this run) — skipping topic '{TOPIC.name}'.")
        return 0

    candidates: list[dict] = []
    # Map same_day_key -> every dedup_key we saw for it, so when we post one
    # action we can burn its same-day siblings too (mirrors post_to_x.py).
    same_day_siblings: dict[str, set[str]] = {}
    for b in bills:
        if not TOPIC.matches(b):
            continue
        same_day_siblings.setdefault(b["same_day_key"], set()).add(b["dedup_key"])
        # Skip if we've already posted this exact action (dedup_key) OR any
        # other action for this same bill on this same day (same_day_key).
        # The same_day_key guard stops a second post when another log entry
        # for the same bill+day arrives on a later run.
        if b["dedup_key"] in seen_all or b["same_day_key"] in seen_all:
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

    # Balanced draw across topic.keyword_groups (mirrors post_to_x.py): when the
    # topic defines named sub-buckets and POST_LIMIT can cover at least one per
    # bucket, guarantee one slot per non-empty bucket before the unrestricted
    # weighted draw fills the rest.
    groups = TOPIC.keyword_groups
    to_post: list[dict] = []
    if groups and effective_limit >= len(groups):
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
        if len(to_post) < effective_limit:
            rest = [b for b in descriptive if b["dedup_key"] not in picked_ids]
            for b in weighted_draw(rest, effective_limit - len(to_post)):
                to_post.append(b)
                picked_ids.add(b["dedup_key"])
    else:
        to_post = weighted_draw(descriptive, effective_limit)

    if len(to_post) < effective_limit:
        picked_ids = {b["dedup_key"] for b in to_post}
        stub_pool = [b for b in stubs if b["dedup_key"] not in picked_ids]
        to_post.extend(weighted_draw(stub_pool, effective_limit - len(to_post)))

    distinct_states = len({b["state"] or "?" for b in to_post})
    print(f"Pool: {len(descriptive)} state(s) with descriptive bills, {len(stubs)} stub-only.")
    print(f"Account has {remaining}/{RUN_POST_LIMIT} post(s) left this run; "
          f"will post up to {effective_limit}: posting {len(to_post)} from "
          f"{distinct_states} state(s).")

    posted = 0
    for b in to_post:
        ensure_english_fields(b)
        headline = shorten_title(b)
        budget = threads_summary_budget(b, headline, include_topic=True)
        summary_text = summarize(b, max_chars=budget) if budget >= MIN_SUMMARY_CHARS else ""
        text, url = compose_threads_post(b, summary_text, headline=headline, include_topic=True)

        print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
        print(text)
        if url:
            print(f"  ↳ link_attachment: {url}")
        print("---")

        if post_thread(text, link_url=url):
            posted += 1
            if SAVE_STATE:
                siblings = same_day_siblings.get(b["same_day_key"], ())
                seen.add(b["dedup_key"])
                # Remember the bill+day itself, plus every sibling action for
                # that bill+day we already know about, so this bill can't be
                # posted again today — even if a new same-day action shows up
                # next run.
                seen.add(b["same_day_key"])
                seen.update(siblings)
                last_posted[b["state"] or "?"] = now.isoformat()
                # Mirror onto the account-wide ledger and persist immediately so
                # the next topic in the loop sees the updated dedup set + cap.
                ledger.record({b["dedup_key"], b["same_day_key"], *siblings})
                ledger.save()
            if SAVE_RAW:
                try:
                    save_raw_record(b)
                    save_full_text(b, out_dir=TOPIC.threads_bills_full_text_dir())
                except Exception as e:
                    print(f"  ! raw-record save failed: {e}", file=sys.stderr)
            # Gentle spacing between publishes; Threads is far more forgiving
            # than X but back-to-back calls are still worth avoiding.
            time.sleep(5)

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
