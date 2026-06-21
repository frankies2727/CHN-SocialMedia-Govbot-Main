#!/usr/bin/env python3
"""
Instagram version of the poster. Mirrors post_to_meta_threads.py's pipeline
(bill load, topic filter, freshness gate, same-day dedup, weighted state draw,
Ollama summary + headline) and publishes to an Instagram Business/Creator
account via the Instagram Graph API (graph.instagram.com). All Instagram state
lives under topics/<name>/instagram/ (bills_used.json plus bills_raw/ and the
rendered cards/) so Instagram dedup is independent of Bluesky/X/Threads.

Why this poster is different from the text platforms
----------------------------------------------------
Instagram has no text-only post type — every feed post needs an image, and the
Graph API fetches that image from a *public URL* (you can't upload bytes like
Bluesky's blob). So this poster adds two stages the others don't have:

  1. Render each bill to a 1080x1350 PNG card (scripts/render_bill_card.py).
  2. Commit + push those cards so they're reachable at a raw.githubusercontent
     URL, then poll the URL until it's live (the CDN can lag a few minutes)
     before asking Instagram to fetch it.

Publishing then mirrors Threads' two-step Graph call: create a media container
(image_url + caption), then publish that container. Because Instagram captions
can't carry a clickable link, the bill URL ships as plain text in the caption
and the card footer reads "Link to the bill in the description".

IMPORTANT: the repository must be public for raw.githubusercontent URLs to be
fetchable by Instagram's servers.

Account model: a single Instagram account dedicated to one topic (the LGBTQ+
launch account). The topic is selected via BOT_TOPIC; credentials come from two
env vars / repo secrets:

    INSTAGRAM_ACCESS_TOKEN   long-lived token (see refresh_instagram_token.py)
    INSTAGRAM_USER_ID        the Instagram Business account id
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from topic import load_active_topic
from render_bill_card import render_card
from post_to_bluesky import (
    _FILENAME_UNSAFE_RE,
    _normalize,
    _slug,
    _smart_truncate,
    _strip_act_name_echo,
    _strip_headline_echo,
    STATE_FULL_NAME,
    best_display_text,
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
STATE_FILE = TOPIC.instagram_state_file_path()

POST_LIMIT = int(os.environ.get("POST_LIMIT", "2"))
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "62"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"

SAVE_STATE = os.environ.get("SAVE_STATE", "1") == "1"
SAVE_RAW = os.environ.get("SAVE_RAW", "1") == "1"

FORCE_STATE = (os.environ.get("FORCE_STATE") or "").strip().lower()
FORCE_BILL_ID = (os.environ.get("FORCE_BILL_ID") or "").strip()
FORCE_REPOST = os.environ.get("FORCE_REPOST") == "1"

# Instagram captions allow 2,200 characters. The card carries the headline +
# summary visually; the caption repeats them for accessibility/feed context and
# adds the (non-clickable) bill link and hashtags.
MAX_CAPTION = 2150
MIN_SUMMARY_CHARS = 80

IG_API = "https://graph.instagram.com/v21.0"
IG_TIMEOUT = int(os.environ.get("INSTAGRAM_TIMEOUT", "60"))
IG_PUBLISH_RETRIES = 4

INSTAGRAM_ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_USER_ID = os.environ.get("INSTAGRAM_USER_ID", "")

# Where the pushed cards can be fetched from. owner/repo + branch are resolved
# from the Actions environment, falling back to the git remote for local runs.
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
RAW_HOST = "https://raw.githubusercontent.com"
# How long to wait for a freshly pushed card to become fetchable on the CDN.
RAW_POLL_TIMEOUT = int(os.environ.get("RAW_POLL_TIMEOUT", "300"))

print("Checking Instagram credentials...")
print(f"  INSTAGRAM_ACCESS_TOKEN present: {bool(INSTAGRAM_ACCESS_TOKEN) and len(INSTAGRAM_ACCESS_TOKEN) > 20}")
print(f"  INSTAGRAM_USER_ID present:      {bool(INSTAGRAM_USER_ID)}")


# ---------------------------------------------------------------------------
# State persistence (Instagram-specific files)
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


def _artifact_basename(b: dict) -> str:
    state = (b.get("state") or "XX")
    ident_raw = (b.get("identifier") or "unknown").strip()
    ident = _FILENAME_UNSAFE_RE.sub("_", ident_raw).strip("_")[:24] or "unknown"
    date = b.get("action_date") or "no-date"
    action_slug = _slug(b.get("action_desc") or "no-action", max_len=40) or "no-action"
    return f"{state}-{ident}-{date}-{action_slug}"


def save_raw_record(b: dict) -> None:
    """Write the verbatim bills.jsonl record for a posted bill to
    topics/<name>/instagram/bills_raw/. Mirrors the other posters."""
    raw = b.get("_raw")
    if not raw:
        return
    out_dir = TOPIC.instagram_bills_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{_artifact_basename(b)}.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    )


# ---------------------------------------------------------------------------
# Caption composition
# ---------------------------------------------------------------------------

def _hashtags(b: dict) -> str:
    """A small, idiomatic hashtag block: topic, state, and a couple of generic
    civic tags. Instagram allows up to 30; we keep it modest."""
    tags = []
    topic_tag = re.sub(r"[^A-Za-z0-9]", "", TOPIC.display_name)
    if topic_tag:
        tags.append(f"#{topic_tag}")
    state_name = STATE_FULL_NAME.get((b.get("state") or "").upper(), "")
    state_tag = re.sub(r"[^A-Za-z0-9]", "", state_name)
    if state_tag:
        tags.append(f"#{state_tag}")
    tags += ["#Legislation", "#StateLegislature", "#CivicTech"]
    # De-dup while preserving order.
    seen: set[str] = set()
    uniq = [t for t in tags if not (t.lower() in seen or seen.add(t.lower()))]
    return " ".join(uniq)


def compose_instagram_caption(b: dict, summary: str, headline: str = "") -> tuple[str, str]:
    """Return (caption, bill_url). The card already shows the headline, summary
    and action; the caption restates them for feed/accessibility context and
    appends the (non-clickable) bill link plus hashtags. Trim order: summary,
    then the head display."""
    emoji = TOPIC.emoji_for(b)
    url = link_for(b)

    state_label = b["state"] or "?"
    display = best_display_text(b, headline=headline).strip()
    summary = (summary or "").strip()
    summary = _strip_act_name_echo(summary, display)
    summary = _strip_headline_echo(summary, display)

    head = f"{emoji} {state_label} {b['identifier']} — {display}"
    summary_block = (
        f"\n\n{summary}"
        if summary and _normalize(summary) != _normalize(display)
        else ""
    )
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block = f"\n\n{action_line}" if action_line else ""
    link_block = f"\n\n🔗 Read the full bill: {url}" if url else ""
    tags = _hashtags(b)
    tag_block = f"\n\n{tags}" if tags else ""

    def assemble(h, s):
        return h + s + action_block + link_block + tag_block

    caption = assemble(head, summary_block)
    if len(caption) > MAX_CAPTION and summary_block:
        fixed = len(assemble(head, "\n\n"))
        summary = _smart_truncate(summary, MAX_CAPTION - fixed)
        summary_block = f"\n\n{summary}" if len(summary) > 20 else ""
        caption = assemble(head, summary_block)
    if len(caption) > MAX_CAPTION:
        fixed = len(assemble(f"{emoji} {state_label} {b['identifier']} — ", summary_block))
        display_trimmed = _smart_truncate(display, MAX_CAPTION - fixed)
        head = f"{emoji} {state_label} {b['identifier']} — {display_trimmed}".rstrip(" —")
        caption = assemble(head, summary_block)
    return caption, url


# ---------------------------------------------------------------------------
# Card rendering + public hosting
# ---------------------------------------------------------------------------

def render_bill_card(b: dict, summary: str, headline: str) -> Path:
    out_dir = TOPIC.instagram_cards_dir()
    out_path = out_dir / f"{_artifact_basename(b)}.png"
    return render_card(
        b,
        headline=headline,
        summary=summary,
        emoji=TOPIC.emoji_for(b),
        accent=TOPIC.card_accent,
        out_path=out_path,
    )


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    )


def _repo_slug() -> str:
    if GITHUB_REPOSITORY:
        return GITHUB_REPOSITORY
    res = _git("config", "--get", "remote.origin.url")
    url = (res.stdout or "").strip()
    m = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else ""


def publish_cards(paths: list[Path]) -> str | None:
    """Commit + push the rendered cards so Instagram can fetch them by URL.
    Returns the pushed commit SHA, or None on failure / nothing to push."""
    rels = [str(p.relative_to(ROOT)) for p in paths]
    add = _git("add", "--", *rels)
    if add.returncode != 0:
        print(f" ! git add failed: {add.stderr}", file=sys.stderr)
        return None
    diff = _git("diff", "--cached", "--quiet")
    if diff.returncode == 0:
        # Nothing staged (cards already committed in a prior run); use HEAD.
        return _git("rev-parse", "HEAD").stdout.strip() or None
    commit = _git("commit", "-m", f"chore: Instagram cards for {TOPIC.name} [skip ci]")
    if commit.returncode != 0:
        print(f" ! git commit failed: {commit.stderr}", file=sys.stderr)
        return None
    for attempt in range(1, 5):
        push = _git("push")
        if push.returncode == 0:
            break
        print(f" ! git push failed (attempt {attempt}): {push.stderr}", file=sys.stderr)
        time.sleep(2 ** attempt)
    else:
        return None
    return _git("rev-parse", "HEAD").stdout.strip() or None


def raw_url_for(path: Path, sha: str, slug: str) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return f"{RAW_HOST}/{slug}/{sha}/{rel}"


def wait_for_url(url: str) -> bool:
    """Poll the raw URL until it serves 200 (CDN propagation), or time out."""
    deadline = time.time() + RAW_POLL_TIMEOUT
    delay = 3
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=IG_TIMEOUT)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 30)
    print(f" ! card URL never became reachable: {url}", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Instagram publish (two-step container -> publish)
# ---------------------------------------------------------------------------

def _create_container(image_url: str, caption: str) -> str | None:
    try:
        resp = requests.post(
            f"{IG_API}/{INSTAGRAM_USER_ID}/media",
            data={
                "image_url": image_url,
                "caption": caption,
                "access_token": INSTAGRAM_ACCESS_TOKEN,
            },
            timeout=IG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("id")
    except Exception as e:
        print(f" ! container creation failed: {e}", file=sys.stderr)
        if getattr(e, "response", None) is not None:
            print(f"   Response body: {e.response.text}", file=sys.stderr)
        return None


def _publish_container(creation_id: str) -> str | None:
    for attempt in range(1, IG_PUBLISH_RETRIES + 1):
        try:
            resp = requests.post(
                f"{IG_API}/{INSTAGRAM_USER_ID}/media_publish",
                data={"creation_id": creation_id, "access_token": INSTAGRAM_ACCESS_TOKEN},
                timeout=IG_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("id")
        except Exception as e:
            if attempt < IG_PUBLISH_RETRIES:
                # Container may still be processing the fetched image — back off.
                time.sleep(5 * attempt)
                continue
            print(f" ! publish failed: {e}", file=sys.stderr)
            if getattr(e, "response", None) is not None:
                print(f"   Response body: {e.response.text}", file=sys.stderr)
            return None
    return None


def post_to_instagram(image_url: str, caption: str) -> bool:
    if DRY_RUN:
        print(f"  [DRY RUN] skipping Instagram post ({len(caption)} chars)")
        print(f"  [DRY RUN] image_url: {image_url}")
        return True
    creation_id = _create_container(image_url, caption)
    if not creation_id:
        return False
    media_id = _publish_container(creation_id)
    if not media_id:
        return False
    print(f"  posted to Instagram (media id {media_id})")
    return True


# ---------------------------------------------------------------------------
# Prepare / publish helpers shared by the daily and force paths
# ---------------------------------------------------------------------------

def _prepare(b: dict) -> dict:
    """Run the shared pipeline for one bill and render its card. Returns a dict
    with the caption, bill url and rendered card path."""
    ensure_english_fields(b)
    headline = shorten_title(b)
    summary_text = summarize(b)
    caption, url = compose_instagram_caption(b, summary_text, headline=headline)
    card_path = render_bill_card(b, summary_text, headline)
    return {"bill": b, "caption": caption, "url": url, "card_path": card_path}


def _publish_prepared(items: list[dict]) -> str | None:
    """Push every prepared card in one commit and return the commit SHA so the
    caller can build raw URLs. In DRY_RUN we skip git entirely."""
    if DRY_RUN:
        return "DRYRUN"
    sha = publish_cards([it["card_path"] for it in items])
    if not sha:
        print(" ! could not publish card images; aborting.", file=sys.stderr)
    return sha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _norm_ident(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


def _post_items(items: list[dict], sha: str, slug: str, state: dict, seen: set,
                same_day_siblings: dict[str, set[str]] | None, now: datetime) -> int:
    posted = 0
    last_posted = state.get("state_last_posted", {})
    for it in items:
        b = it["bill"]
        image_url = raw_url_for(it["card_path"], sha, slug) if not DRY_RUN else "DRY_RUN"
        print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
        print(it["caption"])
        print(f"  ↳ card: {it['card_path'].relative_to(ROOT)}")
        if not DRY_RUN:
            print(f"  ↳ image_url: {image_url}")
            if not wait_for_url(image_url):
                continue
        if not post_to_instagram(image_url, it["caption"]):
            continue
        posted += 1
        if SAVE_STATE:
            seen.add(b["dedup_key"])
            if same_day_siblings is not None:
                seen.update(same_day_siblings.get(b["same_day_key"], ()))
            last_posted[b["state"] or "?"] = now.isoformat()
        if SAVE_RAW:
            try:
                save_raw_record(b)
                save_full_text(b, out_dir=TOPIC.instagram_bills_full_text_dir())
            except Exception as e:
                print(f"  ! raw-record save failed: {e}", file=sys.stderr)
        time.sleep(5)
    state["state_last_posted"] = last_posted
    return posted


def _persist(state: dict, seen: set, posted: int) -> None:
    if SAVE_STATE:
        state["posted"] = sorted(seen)
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"\nDone. Posted {posted} update(s). State saved to "
              f"{STATE_FILE.relative_to(ROOT)}.")
    else:
        print(f"\nDone. Posted {posted} update(s). SAVE_STATE=0 — "
              f"{STATE_FILE.relative_to(ROOT)} left unchanged.")


def _post_forced_bill(records: list[dict]) -> int:
    target_ident = _norm_ident(FORCE_BILL_ID)
    state_matches: list[dict] = []
    bill_matches: list[dict] = []
    for r in records:
        b = extract_fields(r)
        if not b or (b["state"] or "").lower() != FORCE_STATE:
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

    bill_matches.sort(key=lambda b: (bool((b["action_desc"] or "").strip()), _recency(b)),
                      reverse=True)
    b = bill_matches[0]

    state = load_state()
    seen = set(state.get("posted", []))
    if not FORCE_REPOST and b["dedup_key"] in seen:
        print(f"Bill {b['state']} {b['identifier']} action {b['action_date']!r} is "
              f"already in {STATE_FILE.name}. Pass force_repost=true to re-post.")
        return 0

    if not TOPIC.matches(b):
        print(f"  NOTE: bill does not match topic '{TOPIC.name}' keywords — "
              f"posting anyway because force mode was requested.")

    print(f"Force-posting 1 bill to Instagram topic '{TOPIC.name}':")
    item = _prepare(b)
    sha = _publish_prepared([item])
    if not sha:
        return 1
    slug = _repo_slug()
    posted = _post_items([item], sha, slug, state, seen, None,
                         datetime.now(timezone.utc))
    _persist(state, seen, posted)
    return 0 if posted else 1


def main() -> int:
    missing = [
        n for n, v in (
            ("INSTAGRAM_ACCESS_TOKEN", INSTAGRAM_ACCESS_TOKEN),
            ("INSTAGRAM_USER_ID", INSTAGRAM_USER_ID),
        ) if not v
    ]
    if missing and not DRY_RUN:
        print(f"ERROR: missing Instagram credentials: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"=== Instagram GovBot running for topic: {TOPIC.name} ===")
    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    if FORCE_STATE and FORCE_BILL_ID:
        return _post_forced_bill(records)

    state = load_state()
    seen = set(state.get("posted", []))

    candidates: list[dict] = []
    same_day_siblings: dict[str, set[str]] = {}
    for r in records:
        b = extract_fields(r)
        if not b or not TOPIC.matches(b):
            continue
        same_day_siblings.setdefault(b["same_day_key"], set()).add(b["dedup_key"])
        if b["dedup_key"] in seen:
            continue
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
    print(f"  by state: {', '.join(f'{s}={n}' for s, n in state_counts.most_common(15))}")

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
        picked_ids = {b["dedup_key"] for b in to_post}
        stub_pool = [b for b in stubs if b["dedup_key"] not in picked_ids]
        to_post.extend(weighted_draw(stub_pool, POST_LIMIT - len(to_post)))

    distinct_states = len({b["state"] or "?" for b in to_post})
    print(f"Pool: {len(descriptive)} state(s) with descriptive bills, {len(stubs)} stub-only.")
    print(f"Will post up to {POST_LIMIT}: posting {len(to_post)} from {distinct_states} state(s).")

    items = [_prepare(b) for b in to_post]
    sha = _publish_prepared(items)
    if not sha:
        return 1
    slug = _repo_slug()
    posted = _post_items(items, sha, slug, state, seen, same_day_siblings, now)

    if not SAVE_RAW:
        print("  SAVE_RAW=0 — bills_raw artifacts not written.")
    _persist(state, seen, posted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
