#!/usr/bin/env python3
"""
Instagram weekly digest: the Instagram counterpart to the Threads/Bluesky/X
weekly digests.

Instagram has no thread/reply concept, so the digest can't be a chained thread
like the text platforms. Instead it publishes a SINGLE weekly CAROUSEL post: an
intro/cover slide (title, the kind of bills included, and the date range)
followed by a swipeable set of bill cards, one per topic, drawn from a WIDE
VARIETY of topics over the past 7 days. Each bill card already shows the topic it
belongs to (the card renderer stamps the topic label, emoji, and accent color),
so the mixed carousel is legible slide by slide. The caption carries the digest
framing plus every featured bill's (non-clickable) link.

Like the daily Instagram poster, this works around Instagram's image-only,
fetch-by-public-URL model: render each pick to a card (as JPEG — the carousel
media builder is stricter than single-image posts and rejects PNG), commit +
push the cards so they're reachable at a raw.githubusercontent URL, wait for the
CDN, then build the carousel via the Graph API (a child container per image,
each polled to FINISHED, one parent CAROUSEL container polled to FINISHED, then
publish).

Selection (all-topics round-robin for breadth) is shared with the Threads digest
via digest_multitopic.py. Composition, card rendering, and the carousel publish
come from post_to_instagram.py / render_bill_card.py. Because those modules are
written around a single active TOPIC, this script re-points the pipeline's active
topic per bill (see _activate_topic) so each card's copy comes from its own
topic; card rendering and the caption take the topic explicitly, so no other
global is touched.

Carousel size: Instagram caps a carousel at 10 images. The cover takes one, so
the digest selects at most CAROUSEL_MAX - 1 bill cards (see bill_slots()) —
every selected bill is both a slide and a caption entry, so the caption never
lists a bill that isn't shown.

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
from PIL import Image

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
    IG_INTERNAL_ERROR_SUBCODES,
    IG_PUBLISH_RETRIES,
    IG_PUBLISHED_UNCONFIRMED,
    IG_TIMEOUT,
    INSTAGRAM_ACCESS_TOKEN,
    INSTAGRAM_USER_ID,
    JSONL_PATH,
    MAX_CAPTION,
    MIN_SUMMARY_CHARS,
    _artifact_basename,
    _git,
    _internal_error_subcode,
    _repo_slug,
    raw_url_for,
    wait_for_url,
)
from render_bill_card import render_card, render_cover_card
from topic import Topic
from weekly_digest_bluesky import (
    DIGEST_LOOKBACK_DAYS,
    DIGEST_PER_STATE_CAP,
    _format_jurisdictions_line,
    _format_short,
)

ROOT = Path(__file__).resolve().parent.parent

# Instagram's hard limit on images in a single carousel.
CAROUSEL_MAX = int(os.environ.get("CAROUSEL_MAX", "10"))
# The Instagram account spans every topic, so this digest pulls its highlights
# from a wide variety of topics. The carousel is the intro/cover slide plus one
# bill card per topic, so the number of bill cards is capped at CAROUSEL_MAX - 1
# (the cover takes one slot). DIGEST_MAX_HIGHLIGHTS can lower that further but
# never raise it past what the carousel can hold — the caption lists exactly the
# bills that appear as cards, so the two never disagree (see bill_slots()).
DIGEST_MAX_HIGHLIGHTS = int(os.environ.get("DIGEST_MAX_HIGHLIGHTS", "9"))


def bill_slots() -> int:
    """How many bill cards the carousel can show: CAROUSEL_MAX minus the cover
    slide, clamped by DIGEST_MAX_HIGHLIGHTS. This is the single source of truth
    for both selection and the caption, so every featured bill is also a slide."""
    return max(1, min(DIGEST_MAX_HIGHLIGHTS, CAROUSEL_MAX - 1))

# Title/brand for the all-topics caption. Not tied to any single topic's
# thread_title (those name one topic, e.g. "LGBTQ Bills Weekly Digest").
DIGEST_TITLE = os.environ.get(
    "INSTAGRAM_DIGEST_TITLE", "🏛️ Statehouse Weekly Digest")

# The carousel's intro/cover slide is topic-neutral (the digest spans every
# topic), so it uses a govbot accent rather than any one topic's color and lives
# in its own account-level folder rather than under a single topic.
DIGEST_COVER_DIR = ROOT / "account_state" / "instagram" / "weekly_digest"
COVER_TITLE = os.environ.get("INSTAGRAM_DIGEST_COVER_TITLE", "Weekly Digest")
COVER_SUBTITLE = os.environ.get(
    "INSTAGRAM_DIGEST_COVER_SUBTITLE",
    "Top state legislation from a wide range of topics")


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

def _to_jpeg(png_path: Path) -> Path:
    """Convert a rendered PNG card to a JPEG sibling and drop the PNG. The daily
    poster publishes single PNG images fine, but Instagram's CAROUSEL media
    builder is stricter and rejects PNG children with a generic internal error
    (subcode 2207085), so every carousel slide ships as JPEG. Cards are already
    RGB, so this is a straight re-encode."""
    jpg_path = png_path.with_suffix(".jpg")
    with Image.open(png_path) as im:
        im.convert("RGB").save(jpg_path, "JPEG", quality=90, optimize=True)
    png_path.unlink(missing_ok=True)
    return jpg_path


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
    card_path = _to_jpeg(render_card(
        b,
        headline=headline,
        summary=summary,
        emoji=topic.emoji_for(b),
        topic_label=topic.display_name,
        accent=topic.card_accent,
        spectrum=topic.card_spectrum,
        mode=CARD_MODE,
        out_path=out_dir / f"{_artifact_basename(b)}.png",
    ))
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


def _date_range_label(today: datetime, window_days: int, landscape: bool) -> str:
    if landscape:
        return f"Week of {_format_short(today)}, {today.year}"
    end = today
    start = today - timedelta(days=window_days - 1)
    return f"{_format_short(start)}–{_format_short(end)}, {end.year}"


def render_cover(items: list[dict], today: datetime, window_days: int,
                 landscape: bool) -> dict:
    """Render the carousel's intro slide (title, the kind of bills, date range)
    and return it as a slide pseudo-item so it can lead the carousel. The
    'INCLUDING' line names the topics actually covered this week, in selection
    (significance) order."""
    topics = list(dict.fromkeys(it["topic"].display_name for it in items))
    including = " · ".join(topics)
    n_states = len({(it["bill"]["state"] or "?") for it in items})
    coverage = (f"{len(topics)} topic{'s' if len(topics) != 1 else ''} · "
                f"{n_states} state{'s' if n_states != 1 else ''}")
    DIGEST_COVER_DIR.mkdir(parents=True, exist_ok=True)
    path = _to_jpeg(render_cover_card(
        title=COVER_TITLE,
        subtitle=COVER_SUBTITLE,
        including=including,
        date_label=_date_range_label(today, window_days, landscape),
        coverage_label=coverage,
        mode=CARD_MODE,
        out_path=DIGEST_COVER_DIR / f"cover-{CARD_MODE}.png",
    ))
    return {"card_path": path, "topic_name": "cover",
            "bill": {"state": "", "identifier": "(cover)"}}


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
            resp = getattr(e, "response", None)
            subcode = _internal_error_subcode(resp)
            if subcode in IG_INTERNAL_ERROR_SUBCODES:
                # Known Instagram false negative: media_publish returns a generic
                # internal error (subcode 2207085) even though the carousel was
                # in fact published. Retrying can't republish the same container,
                # so stop and report it as published-but-unconfirmed rather than
                # failing the whole run for a post that almost certainly went live.
                print(f" ! publish returned internal-error subcode {subcode}; "
                      f"Instagram likely published the carousel anyway — treating "
                      f"it as posted.", file=sys.stderr)
                if resp is not None:
                    print(f"   Response body: {resp.text}", file=sys.stderr)
                return IG_PUBLISHED_UNCONFIRMED
            if attempt < IG_PUBLISH_RETRIES:
                time.sleep(5 * attempt)
                continue
            print(f" ! publish failed: {e}", file=sys.stderr)
            if resp is not None:
                print(f"   Response body: {resp.text}", file=sys.stderr)
            return None
    return None


def _container_status(container_id: str) -> str | None:
    """The Graph API status_code of a media container: FINISHED, IN_PROGRESS,
    ERROR, EXPIRED, or PUBLISHED. None if the check itself failed."""
    try:
        resp = requests.get(
            f"{IG_API}/{container_id}",
            params={"fields": "status_code", "access_token": INSTAGRAM_ACCESS_TOKEN},
            timeout=IG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("status_code")
    except Exception as e:
        print(f"  ! container status check failed: {e}", file=sys.stderr)
        return None


def _wait_container_finished(container_id: str, timeout: int = 180) -> bool:
    """Poll a container until it reports FINISHED (ready to publish). Carousel
    containers are assembled asynchronously from their children, and publishing
    one that's still IN_PROGRESS returns a generic 'internal error' (subcode
    2207085) — so gate the publish on FINISHED rather than firing immediately."""
    deadline = time.time() + timeout
    delay = 3
    while time.time() < deadline:
        status = _container_status(container_id)
        if status == "FINISHED":
            return True
        if status in ("ERROR", "EXPIRED"):
            print(f"  ! container {container_id} assembly {status}; aborting.",
                  file=sys.stderr)
            return False
        print(f"    container {container_id} status: {status or '(unknown)'} — waiting…")
        time.sleep(delay)
        delay = min(delay * 2, 20)
    print(f"  ! container {container_id} never reached FINISHED before timeout.",
          file=sys.stderr)
    return False


def post_carousel(slide_items: list[dict], caption: str, sha: str, slug: str) -> bool:
    """Create a child container per slide card, wait for EACH child to finish
    processing, assemble them into one CAROUSEL container with the caption, wait
    for that to finish too, then publish. Returns True iff published.

    Both waits matter: a child whose image Instagram couldn't fetch/process
    otherwise surfaces only as a generic internal error (subcode 2207085) at
    publish time, and publishing a still-assembling parent triggers the same."""
    children: list[str] = []
    for it in slide_items:
        image_url = raw_url_for(it["card_path"], sha, slug)
        print(f"  slide: [{it['topic_name']}] {it['bill']['state']} "
              f"{it['bill']['identifier']} -> {image_url}")
        if not wait_for_url(image_url):
            print("  ! card URL not reachable; skipping this slide.", file=sys.stderr)
            continue
        child = _create_carousel_item(image_url)
        if not child:
            print("  ! child container failed; skipping this slide.", file=sys.stderr)
            continue
        # Each child must finish processing before it can go in the carousel.
        if not _wait_container_finished(child, timeout=120):
            print("  ! child never finished processing; skipping this slide.",
                  file=sys.stderr)
            continue
        children.append(child)

    if len(children) < 2:
        print(f" ! only {len(children)} slide(s) ready; a carousel needs at "
              f"least 2. Aborting.", file=sys.stderr)
        return False

    parent = _create_carousel_container(children, caption)
    if not parent:
        return False
    print(f"  carousel container {parent} created from {len(children)} children; "
          f"waiting for assembly…")
    if not _wait_container_finished(parent):
        return False
    print(f"  carousel container {parent} FINISHED; publishing…")
    media_id = _publish_container(parent)
    if not media_id:
        return False
    if media_id == IG_PUBLISHED_UNCONFIRMED:
        print(f"  carousel treated as posted (unconfirmed publish, "
              f"{len(children)} slides).")
        return True
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

    # Cap at how many bill cards the carousel can actually show (CAROUSEL_MAX
    # minus the cover slide), so every selected bill becomes a slide AND a
    # caption entry — the two never disagree.
    cap = bill_slots()
    if per_topic:
        highlights = dm.merge_across_topics(
            per_topic, cap=cap, per_state_cap=DIGEST_PER_STATE_CAP)
        return highlights, chosen_window, False, None

    recent = dm.landscape_picks(
        matched_by_topic, cap=cap, per_state_cap=DIGEST_PER_STATE_CAP)
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
          f"(bill-slots={bill_slots()}, carousel-max={CAROUSEL_MAX}, "
          f"{'landscape' if landscape else f'window={window_days}d'}): "
          f"{', '.join(topics_covered)}")
    for b in picks:
        print(f"  [{b.get('_score', 0):>3}] [{b['_topic_name']}] {b['state']} "
              f"{b['identifier']} ({b['action_date']}): {b['action_desc'][:60]}")

    # Render every featured bill's card up front so its raw artifact + card land
    # even if publishing fails partway.
    items = [_prepare_item(b) for b in picks]
    _save_digest_raw_records(items)

    # Slide 1 is the intro/cover (Weekly Digest, the kind of bills, date range);
    # the bill cards follow. Selection is already capped at bill_slots() so every
    # featured bill fits as a slide, and the caption lists exactly those same
    # bills — cards and caption never disagree. The [:CAROUSEL_MAX] slice is a
    # belt-and-suspenders guard against ever exceeding the hard limit.
    cover_item = render_cover(items, today, window_days, landscape)
    slide_items = ([cover_item] + items)[:CAROUSEL_MAX]
    header = compose_header(today, window_days, landscape, state_counts)
    caption = compose_caption(items, header)

    print(f"\n--- CAPTION ({len(caption)} chars) ---\n{caption}\n---")
    print(f"Carousel: {len(slide_items)} slide(s) "
          f"(1 cover + {len(slide_items) - 1} bill card(s)) "
          f"from {len(items)} featured bill(s).")
    for it in slide_items:
        print(f"  slide: {it['card_path'].relative_to(ROOT)}")

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
