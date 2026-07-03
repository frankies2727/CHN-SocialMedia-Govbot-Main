#!/usr/bin/env python3
"""
Instagram weekly digest: the Instagram counterpart to the Threads/Bluesky/X
weekly digests.

Instagram has no thread/reply concept, so the digest can't be a chained thread
like the text platforms. Instead it publishes a SINGLE weekly CAROUSEL post — a
swipeable set of bill cards, one per topic — drawn from a WIDE VARIETY of topics
over the past 7 days. Each card already shows the topic it belongs to (the card
renderer stamps the topic label, emoji, and accent color), so the mixed carousel
is legible slide by slide. The caption carries the digest framing plus every
featured bill's (non-clickable) link.

Like the daily Instagram poster, this works around Instagram's image-only,
fetch-by-public-URL model: render each pick to a PNG card, commit + push the
cards so they're reachable at a raw.githubusercontent URL, wait for the CDN,
then build the carousel via the Graph API (a child container per image, one
parent CAROUSEL container, then publish).

Selection (all-topics round-robin for breadth) is shared with the Threads digest
via digest_multitopic.py. Composition, card rendering, and the carousel publish
come from post_to_instagram.py / render_bill_card.py. Because those modules are
written around a single active TOPIC, this script re-points the pipeline's active
topic per bill (see _activate_topic) so each card's copy comes from its own
topic; card rendering and the caption take the topic explicitly, so no other
global is touched.

Carousel size: Instagram caps a carousel at 10 images, so at most CAROUSEL_MAX
cards become slides even though up to DIGEST_MAX_HIGHLIGHTS bills are selected;
any extra selected bills still appear (with their links) in the caption.

BOT_TOPIC still has to name a valid topic so the modules import cleanly, but the
digest iterates over all topics — which one is set no longer affects the output.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import digest_multitopic as dm
import post_to_bluesky as pb
from post_to_bluesky import (
    best_display_text,
    display_identifier,
    ensure_english_fields,
    link_for,
    load_bills,
    save_full_text,
    shorten_title,
    summarize,
    STATE_FULL_NAME,
)
from post_to_instagram import (
    CARD_MODE,
    DRY_RUN,
    IG_API,
    IG_PUBLISH_RETRIES,
    IG_TIMEOUT,
    INSTAGRAM_ACCESS_TOKEN,
    INSTAGRAM_USER_ID,
    JSONL_PATH,
    MAX_CAPTION,
    MIN_SUMMARY_CHARS,
    _artifact_basename,
    _git,
    _repo_slug,
    raw_url_for,
    wait_for_url,
)
from render_bill_card import render_card
from topic import Topic
from weekly_digest_bluesky import (
    DIGEST_LOOKBACK_DAYS,
    DIGEST_PER_STATE_CAP,
    _format_jurisdictions_line,
    _format_short,
)

ROOT = Path(__file__).resolve().parent.parent

# The Instagram account spans every topic, so this digest pulls its highlights
# from a wide variety of topics. Up to this many bills are selected; the caption
# lists them all, while the carousel shows the first CAROUSEL_MAX as slides.
DIGEST_MAX_HIGHLIGHTS = int(os.environ.get("DIGEST_MAX_HIGHLIGHTS", "11"))
# Instagram's hard limit on images in a single carousel.
CAROUSEL_MAX = int(os.environ.get("CAROUSEL_MAX", "10"))

# Title/brand for the all-topics caption. Not tied to any single topic's
# thread_title (those name one topic, e.g. "LGBTQ Bills Weekly Digest").
DIGEST_TITLE = os.environ.get(
    "INSTAGRAM_DIGEST_TITLE", "🏛️ Statehouse Weekly Digest")


# ---------------------------------------------------------------------------
# Per-topic activation
#
# post_to_bluesky is written around a single module global TOPIC (the daily
# posters run one process per topic). This digest spans all topics in one
# process, so before composing each card's copy we re-point that global to the
# bill's own topic — then summarize()/shorten_title() steer on the right topic.
# Card rendering and the caption take the topic explicitly, so nothing else
# needs the global.
# ---------------------------------------------------------------------------

def _activate_topic(topic: Topic) -> None:
    pb.TOPIC = topic


# ---------------------------------------------------------------------------
# Card + caption composition
# ---------------------------------------------------------------------------

def _prepare_item(b: dict) -> dict:
    """Compose one pick's copy under its OWN topic and render its card. Returns
    {bill, topic, topic_name, headline, display, url, card_path}."""
    topic = b["_topic"]
    _activate_topic(topic)
    ensure_english_fields(b)
    headline = shorten_title(b)
    summary = summarize(b)
    display = best_display_text(b, headline=headline).strip()
    url = link_for(b)

    out_dir = topic.instagram_weekly_digest_cards_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = render_card(
        b,
        headline=headline,
        summary=summary,
        emoji=topic.emoji_for(b),
        topic_label=topic.display_name,
        accent=topic.card_accent,
        spectrum=topic.card_spectrum,
        mode=CARD_MODE,
        out_path=out_dir / f"{_artifact_basename(b)}.png",
    )
    return {
        "bill": b, "topic": topic, "topic_name": b["_topic_name"],
        "headline": headline, "display": display, "url": url,
        "card_path": card_path,
    }


def _digest_hashtags() -> str:
    """A small, generic civic hashtag block for the all-topics digest. (The
    daily poster tags the single topic + state; the digest spans many topics, so
    it keeps to broad civic tags rather than a wall of per-topic tags.)"""
    return "#Legislation #StateLegislature #Statehouse #CivicTech"


def _entry_line(n: int, it: dict) -> str:
    topic = it["topic"]
    b = it["bill"]
    emoji = topic.emoji_for(b)
    ident_disp = display_identifier(b["state"], b["identifier"])
    state_label = b["state"] or "?"
    return (f"{n}. {emoji} {topic.display_name} — "
            f"{state_label} {ident_disp}: {it['display']}")


def compose_caption(items: list[dict], header: str) -> str:
    """Build the carousel caption: the header block, then a numbered entry per
    featured bill (topic · state · id · headline, then its link), then a generic
    hashtag block. Budgeted to Instagram's caption cap — links drop before entry
    lines, and any bills that don't fit are noted as a remainder — so a busy week
    never blows the limit."""
    hashtags = _digest_hashtags()
    reserve = len(f"\n\n{hashtags}")
    caption = header
    included = 0
    for i, it in enumerate(items, 1):
        line = _entry_line(i, it)
        link_line = f"🔗 {it['url']}" if it["url"] else ""
        full = f"\n\n{line}" + (f"\n{link_line}" if link_line else "")
        if len(caption) + len(full) + reserve <= MAX_CAPTION:
            caption += full
            included += 1
            continue
        # Link didn't fit — try the entry line alone.
        just_line = f"\n\n{line}"
        if len(caption) + len(just_line) + reserve <= MAX_CAPTION:
            caption += just_line
            included += 1
            continue
        break

    remainder = len(items) - included
    if remainder > 0:
        note = f"\n\n… and {remainder} more in our daily feed."
        if len(caption) + len(note) + reserve <= MAX_CAPTION:
            caption += note

    caption += f"\n\n{hashtags}"
    return caption


def compose_header(today: datetime, window_days: int, landscape: bool,
                   state_counts: Counter | None = None) -> str:
    if landscape:
        juris = _format_jurisdictions_line(state_counts) if state_counts else ""
        juris_line = f" in {juris}" if juris else ""
        return (
            f"{DIGEST_TITLE}\n"
            f"Week of {_format_short(today)}, {today.year}\n\n"
            "Quiet stretch — little notable floor action across the states this "
            f"past month. Still tracking bills on a wide range of topics{juris_line}. "
            "A landscape check-in. Swipe through 👉"
        )
    end = today
    start = today - timedelta(days=window_days - 1)
    range_str = f"{_format_short(start)}–{_format_short(end)}, {end.year}"
    if window_days <= DIGEST_LOOKBACK_DAYS:
        framing = ("This week's top bill activity from statehouses across the "
                   "country — a wide range of topics. Swipe through 👉")
    else:
        framing = ("Quieter past 7 days, so we widened the lens — recent bill "
                   "activity across the states and topics. Swipe through 👉")
    return f"{DIGEST_TITLE}\n{range_str}\n\n{framing}"


# ---------------------------------------------------------------------------
# Raw-artifact persistence (per topic)
# ---------------------------------------------------------------------------

def _save_digest_raw_records(items: list[dict]) -> None:
    """Dump the verbatim bills.jsonl record (and extracted full text) for every
    featured bill to its OWN topic's instagram/weekly_digest/ folder, so the
    digest leaves the same self-contained trail as the daily feed."""
    for it in items:
        b = it["bill"]
        raw = b.get("_raw")
        if not raw:
            continue
        topic = it["topic"]
        out_dir = topic.instagram_weekly_digest_bills_raw_dir()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{_artifact_basename(b)}.json").write_text(
                json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
            save_full_text(b, out_dir=out_dir.parent / "bills_full_text")
        except OSError as e:
            print(f"  ! digest raw-record save failed for "
                  f"{b.get('state')} {b.get('identifier')}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Card hosting (commit + push so Instagram can fetch by URL)
# ---------------------------------------------------------------------------

def publish_digest_cards(paths: list[Path]) -> str | None:
    """Commit + push the rendered carousel cards so Instagram can fetch them by
    raw.githubusercontent URL. Returns the pushed commit SHA (or HEAD when the
    cards were already committed), or None on failure. Mirrors
    post_to_instagram.publish_cards but with a digest-specific commit message."""
    rels = [str(p.relative_to(ROOT)) for p in paths]
    add = _git("add", "--", *rels)
    if add.returncode != 0:
        print(f" ! git add failed: {add.stderr}", file=sys.stderr)
        return None
    if _git("diff", "--cached", "--quiet").returncode == 0:
        return _git("rev-parse", "HEAD").stdout.strip() or None
    commit = _git("commit", "-m", "chore: Instagram weekly-digest cards [skip ci]")
    if commit.returncode != 0:
        print(f" ! git commit failed: {commit.stderr}", file=sys.stderr)
        return None
    for attempt in range(1, 5):
        if _git("push").returncode == 0:
            break
        print(f" ! git push failed (attempt {attempt})", file=sys.stderr)
        time.sleep(2 ** attempt)
    else:
        return None
    return _git("rev-parse", "HEAD").stdout.strip() or None


# ---------------------------------------------------------------------------
# Carousel publish (child containers -> parent CAROUSEL -> publish)
# ---------------------------------------------------------------------------

def _create_carousel_item(image_url: str) -> str | None:
    try:
        resp = requests.post(
            f"{IG_API}/{INSTAGRAM_USER_ID}/media",
            data={
                "image_url": image_url,
                "is_carousel_item": "true",
                "access_token": INSTAGRAM_ACCESS_TOKEN,
            },
            timeout=IG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("id")
    except Exception as e:
        print(f" ! carousel item creation failed: {e}", file=sys.stderr)
        if getattr(e, "response", None) is not None:
            print(f"   Response body: {e.response.text}", file=sys.stderr)
        return None


def _create_carousel_container(children: list[str], caption: str) -> str | None:
    try:
        resp = requests.post(
            f"{IG_API}/{INSTAGRAM_USER_ID}/media",
            data={
                "media_type": "CAROUSEL",
                "children": ",".join(children),
                "caption": caption,
                "access_token": INSTAGRAM_ACCESS_TOKEN,
            },
            timeout=IG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("id")
    except Exception as e:
        print(f" ! carousel container creation failed: {e}", file=sys.stderr)
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
                time.sleep(5 * attempt)
                continue
            print(f" ! publish failed: {e}", file=sys.stderr)
            if getattr(e, "response", None) is not None:
                print(f"   Response body: {e.response.text}", file=sys.stderr)
            return None
    return None


def post_carousel(slide_items: list[dict], caption: str, sha: str, slug: str) -> bool:
    """Create a child container per slide card, assemble them into one CAROUSEL
    container with the caption, then publish. Returns True iff published."""
    children: list[str] = []
    for it in slide_items:
        image_url = raw_url_for(it["card_path"], sha, slug)
        print(f"  slide: [{it['topic_name']}] {it['bill']['state']} "
              f"{it['bill']['identifier']} -> {image_url}")
        if not wait_for_url(image_url):
            print("  ! card URL not reachable; skipping this slide.", file=sys.stderr)
            continue
        child = _create_carousel_item(image_url)
        if child:
            children.append(child)
        else:
            print("  ! child container failed; skipping this slide.", file=sys.stderr)

    if len(children) < 2:
        print(f" ! only {len(children)} slide(s) available; a carousel needs at "
              f"least 2. Aborting.", file=sys.stderr)
        return False

    parent = _create_carousel_container(children, caption)
    if not parent:
        return False
    media_id = _publish_container(parent)
    if not media_id:
        return False
    print(f"  posted carousel to Instagram (media id {media_id}, "
          f"{len(children)} slides)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _select(today: datetime) -> tuple[list[dict], int, bool, Counter | None]:
    """Return (picks, window_days, is_landscape, state_counts_for_landscape).
    picks are the cross-topic round-robin selection (highlights, or the
    quiet-week landscape fallback)."""
    records = load_bills(JSONL_PATH)
    if not records:
        return [], DIGEST_LOOKBACK_DAYS, False, None

    extracted = dm.extract_all(records)
    matched_by_topic = dm.build_matched_by_topic(
        extracted,
        on_skip=lambda name, e: print(f"  ! skipping topic {name!r}: {e}", file=sys.stderr),
    )
    if not matched_by_topic:
        print("No topics found under topics/. Nothing to digest.")
        return [], DIGEST_LOOKBACK_DAYS, False, None
    print(f"=== Instagram weekly digest (all topics): "
          f"{', '.join(matched_by_topic)} ===")
    for name, (_, matched) in matched_by_topic.items():
        print(f"  {name}: {len(matched)} matching bill action(s)")

    if not any(m for _, m in matched_by_topic.values()):
        print("No topic bills found at all. Nothing to digest.")
        return [], DIGEST_LOOKBACK_DAYS, False, None

    chosen_window, per_topic = dm.choose_active_window(
        matched_by_topic, today, DIGEST_PER_STATE_CAP)
    print(f"Chosen lookback: {chosen_window}d ({len(per_topic)} topic(s) active).")

    if per_topic:
        highlights = dm.merge_across_topics(
            per_topic, cap=DIGEST_MAX_HIGHLIGHTS, per_state_cap=DIGEST_PER_STATE_CAP)
        return highlights, chosen_window, False, None

    recent = dm.landscape_picks(
        matched_by_topic, cap=DIGEST_MAX_HIGHLIGHTS, per_state_cap=DIGEST_PER_STATE_CAP)
    state_counts = Counter((b["state"] or "?") for b in recent)
    return recent, chosen_window, True, state_counts


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

    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    picks, window_days, landscape, state_counts = _select(today)
    if not picks:
        return 0

    topics_covered = sorted({b["_topic_name"] for b in picks})
    print(f"\nSelected {len(picks)} bill(s) across {len(topics_covered)} topic(s) "
          f"(cap={DIGEST_MAX_HIGHLIGHTS}, carousel-max={CAROUSEL_MAX}, "
          f"{'landscape' if landscape else f'window={window_days}d'}): "
          f"{', '.join(topics_covered)}")
    for b in picks:
        print(f"  [{b.get('_score', 0):>3}] [{b['_topic_name']}] {b['state']} "
              f"{b['identifier']} ({b['action_date']}): {b['action_desc'][:60]}")

    # Render every featured bill's card up front so its raw artifact + card land
    # even if publishing fails partway.
    items = [_prepare_item(b) for b in picks]
    _save_digest_raw_records(items)

    # The caption lists every featured bill; the carousel shows the first
    # CAROUSEL_MAX as slides (Instagram's hard limit).
    slide_items = items[:CAROUSEL_MAX]
    header = compose_header(today, window_days, landscape, state_counts)
    caption = compose_caption(items, header)

    print(f"\n--- CAPTION ({len(caption)} chars) ---\n{caption}\n---")
    print(f"Carousel: {len(slide_items)} slide(s) from {len(items)} featured bill(s).")
    for it in items:
        print(f"  card: {it['card_path'].relative_to(ROOT)}")

    if DRY_RUN:
        print("\n[DRY RUN] skipping card push + carousel publish.")
        return 0

    sha = publish_digest_cards([it["card_path"] for it in slide_items])
    if not sha:
        print(" ! could not publish card images; aborting.", file=sys.stderr)
        return 1
    slug = _repo_slug()
    if not post_carousel(slide_items, caption, sha, slug):
        return 1
    print(f"\nDone. Posted Instagram weekly-digest carousel "
          f"({len(slide_items)} slides across {len(topics_covered)} topic(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
