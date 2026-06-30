#!/usr/bin/env python3
"""
State Extravaganza: a manual, on-demand digest thread that spotlights recent
bill activity from a HAND-PICKED set of states (rather than the whole country),
posted to a single platform.

Where the weekly digest is topic-first ("everything moving on <topic> across all
states this week"), the State Extravaganza is state-first: pick one or more
states, pick how far back to look (never more than 62 days), and ship a thread
of the most significant bills those statehouses produced. The first post is a
formal "🏛️ State Extravaganza!! 🧵" header; each reply is one bill card, chained
into a single thread exactly like the weekly digest.

Configurable knobs (all via env vars; the manual workflow wires them to the
"Run workflow" form):

  * PLATFORM                 — bluesky | x | threads | instagram (which feed)
                               Instagram has no text thread, so its "thread" is a
                               single carousel post: a cover slide + one rendered
                               bill card per highlight (max 10 slides total).
  * EXTRAVAGANZA_STATES      — space/comma separated state codes, e.g. "CA NY TX"
                               (empty = every state, i.e. a national extravaganza)
  * BOT_TOPIC                — topic folder; selects the bill filter, the copy
                               focus, and (for Bluesky/Threads) the account
                               credentials. X is a single account, so the
                               workflow pins BOT_TOPIC to the X bot's topic.
  * NUM_POSTS                — number of bill posts in the thread (max highlights)
  * EXTRAVAGANZA_LOOKBACK_DAYS — recency window in days; HARD-CAPPED at 62.
  * EXTRAVAGANZA_PER_STATE_CAP — max bills per state (defaults to NUM_POSTS, i.e.
                               effectively uncapped so a single-state run can
                               fill the whole thread).
  * DRY_RUN                  — "1" composes + prints the thread without posting.

Reuse, not duplication: the significance scorer, the bill loader/filter, every
platform's post composition, and the thread-chaining all come from the existing
weekly-digest + daily-poster modules. The only logic unique to this file is the
state filter, the 62-day cap, and the extravaganza root copy.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timezone

from post_to_bluesky import (
    JSONL_PATH,
    STATE_FULL_NAME,
    TOPIC,
    US_STATES,
    load_bills,
)
from weekly_digest_bluesky import (
    _format_short,
    candidates_in_window,
    collect_topic_bills,
    score_action,
)

# The whole point of this digest is a recent-activity spotlight, so the lookback
# window is never allowed past 62 days no matter what the workflow form passes.
MAX_LOOKBACK_DAYS = 62

PLATFORM = os.environ.get("PLATFORM", "bluesky").strip().lower()
NUM_POSTS = max(1, int(os.environ.get("NUM_POSTS", "6")))
# Per-state cap defaults to NUM_POSTS — i.e. no real cap — so a one-state
# extravaganza can fill every slot from that state. Set EXTRAVAGANZA_PER_STATE_CAP
# to a smaller number to keep a multi-state run broad.
PER_STATE_CAP = max(1, int(os.environ.get("EXTRAVAGANZA_PER_STATE_CAP", str(NUM_POSTS))))
DRY_RUN = os.environ.get("DRY_RUN") == "1"


def _resolve_lookback() -> int:
    raw = os.environ.get("EXTRAVAGANZA_LOOKBACK_DAYS", str(MAX_LOOKBACK_DAYS))
    try:
        days = int(raw)
    except ValueError:
        days = MAX_LOOKBACK_DAYS
    if days < 1:
        days = 1
    if days > MAX_LOOKBACK_DAYS:
        print(f"  lookback {days}d exceeds the {MAX_LOOKBACK_DAYS}-day cap; "
              f"clamping to {MAX_LOOKBACK_DAYS}.")
        days = MAX_LOOKBACK_DAYS
    return days


def parse_states() -> list[str]:
    """Parse EXTRAVAGANZA_STATES into a list of valid, de-duplicated, uppercase
    state codes (order preserved). Empty/unset means "all states"."""
    raw = os.environ.get("EXTRAVAGANZA_STATES", "")
    seen: set[str] = set()
    out: list[str] = []
    for tok in raw.replace(",", " ").split():
        code = tok.strip().upper()
        if not code or code in seen:
            continue
        if code not in US_STATES:
            print(f"  ! ignoring unknown state code: {tok!r}", file=sys.stderr)
            continue
        seen.add(code)
        out.append(code)
    return out


def filter_by_states(bills: list[dict], states: list[str]) -> list[dict]:
    if not states:
        return bills
    want = set(states)
    return [b for b in bills if (b.get("state") or "").upper() in want]


def states_label(states: list[str]) -> str:
    """Human-readable scope line for the root post. Spells out full state names
    for a handful of states, collapses to a count for a big list."""
    if not states:
        return "all states"
    names = [STATE_FULL_NAME.get(c, c) for c in states]
    if len(names) == 1:
        return names[0]
    if len(names) <= 4:
        return ", ".join(names[:-1]) + " & " + names[-1]
    return f"{len(names)} states"


def select_extravaganza(candidates: list[dict], max_posts: int,
                        per_state_cap: int) -> list[dict]:
    """Collapse each bill to its highest-scoring action, then take the top
    `max_posts` by (significance score, recency), capped at `per_state_cap`
    bills per state. Mirrors weekly_digest_bluesky.select_highlights but reads
    the extravaganza's own NUM_POSTS / per-state knobs instead of the digest
    globals."""
    best_by_bill: dict[tuple[str, str], dict] = {}
    for b in candidates:
        key = (b["state"], b["identifier"])
        b["_score"] = score_action(b["action_desc"])
        existing = best_by_bill.get(key)
        if existing is None or b["_score"] > existing["_score"] or (
            b["_score"] == existing["_score"]
            and b["action_date"] > existing["action_date"]
        ):
            best_by_bill[key] = b

    bills = sorted(best_by_bill.values(),
                   key=lambda b: (b["_score"], b["action_date"]), reverse=True)

    picked: list[dict] = []
    per_state: Counter[str] = Counter()
    for b in bills:
        state = b["state"] or "?"
        if per_state[state] >= per_state_cap:
            continue
        picked.append(b)
        per_state[state] += 1
        if len(picked) >= max_posts:
            break
    return picked


def compose_root(scope: str, topic_label: str, today: datetime,
                 window_days: int, max_len: int, len_fn, *,
                 has_links: bool = False) -> str:
    """Build the formal extravaganza header post, trimming progressively to fit
    the platform's character budget (len_fn measures it — len for Bluesky/
    Threads, x_weighted_len for X)."""
    range_str = f"past {window_days} days"
    title = "🏛️ State Extravaganza!! 🧵"
    framing = (f"Spotlighting {topic_label} bill activity from {scope} "
               f"over the {range_str}.")
    links_line = "\n🔗 All bill links are in the last post." if has_links else ""
    text = f"{title}\n{scope}\n\n{framing}{links_line}"
    if len_fn(text) > max_len:
        text = f"{title}\n\n{framing}{links_line}"
    if len_fn(text) > max_len:
        text = f"{title}\n{scope}{links_line}"
    if len_fn(text) > max_len:
        text = f"{title}{links_line}"
    return text


# ---------------------------------------------------------------------------
# Per-platform handlers
# ---------------------------------------------------------------------------

def run_bluesky(candidates: list[dict], scope: str, today: datetime,
                window: int) -> int:
    from post_to_bluesky import BSKY_HANDLE, BSKY_PASSWORD, BlueskyClient, MAX_POST
    from weekly_digest_bluesky import (
        _build_highlight_replies,
        _save_digest_raw_records,
        post_thread,
    )

    if not DRY_RUN and (not BSKY_HANDLE or not BSKY_PASSWORD):
        print("ERROR: BLUESKY_HANDLE and BLUESKY_APP_PASSWORD must be set "
              "for this topic.", file=sys.stderr)
        return 1

    highlights = select_extravaganza(candidates, NUM_POSTS, PER_STATE_CAP)
    _log_highlights(highlights, window)

    client = None if DRY_RUN else BlueskyClient(BSKY_HANDLE, BSKY_PASSWORD)
    _save_digest_raw_records(highlights)
    replies = _build_highlight_replies(client, highlights)
    root_text = compose_root(scope, TOPIC.display_name, today, window,
                             MAX_POST, len)
    post_thread(client, root_text, replies)
    print(f"\nDone. Posted Bluesky State Extravaganza: 1 root + "
          f"{len(replies)} bill post(s).")
    return 0


def run_x(candidates: list[dict], scope: str, today: datetime,
          window: int) -> int:
    from post_to_x import (
        MAX_TWEET,
        X_ACCESS_TOKEN,
        X_ACCESS_TOKEN_SECRET,
        X_API_KEY,
        X_API_SECRET,
        build_client,
        x_weighted_len,
    )
    from weekly_digest_x import (
        _save_digest_raw_records,
        build_highlight_replies,
        build_link_posts,
        post_thread,
    )

    missing = [n for n, v in (
        ("X_API_KEY", X_API_KEY),
        ("X_API_SECRET", X_API_SECRET),
        ("X_ACCESS_TOKEN", X_ACCESS_TOKEN),
        ("X_ACCESS_TOKEN_SECRET", X_ACCESS_TOKEN_SECRET),
    ) if not v]
    if missing and not DRY_RUN:
        print(f"ERROR: missing X credentials: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    highlights = select_extravaganza(candidates, NUM_POSTS, PER_STATE_CAP)
    _log_highlights(highlights, window)

    client = None if DRY_RUN else build_client()
    _save_digest_raw_records(highlights)
    replies, link_items = build_highlight_replies(highlights)
    link_posts = build_link_posts(link_items)
    root_text = compose_root(scope, TOPIC.display_name, today, window,
                             MAX_TWEET, x_weighted_len,
                             has_links=bool(link_posts))
    post_thread(client, root_text, replies + link_posts)
    print(f"\nDone. Posted X State Extravaganza: 1 root + {len(replies)} "
          f"bill post(s) + {len(link_posts)} links post(s).")
    return 0


def run_threads(candidates: list[dict], scope: str, today: datetime,
                window: int) -> int:
    from post_to_meta_threads import (
        MAX_THREADS,
        THREADS_ACCESS_TOKEN,
        THREADS_USER_ID,
    )
    from weekly_digest_meta_threads import (
        _save_digest_raw_records,
        build_highlight_replies,
        post_digest_thread,
    )

    missing = [n for n, v in (
        ("THREADS_ACCESS_TOKEN", THREADS_ACCESS_TOKEN),
        ("THREADS_USER_ID", THREADS_USER_ID),
    ) if not v]
    if missing and not DRY_RUN:
        print(f"ERROR: missing Threads credentials: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    highlights = select_extravaganza(candidates, NUM_POSTS, PER_STATE_CAP)
    _log_highlights(highlights, window)

    _save_digest_raw_records(highlights)
    replies = build_highlight_replies(highlights)
    root_text = compose_root(scope, TOPIC.display_name, today, window,
                             MAX_THREADS, len)
    post_digest_thread(root_text, replies)
    print(f"\nDone. Posted Threads State Extravaganza: 1 root + "
          f"{len(replies)} bill post(s).")
    return 0


# ---------------------------------------------------------------------------
# Instagram (carousel: cover slide + one bill card per highlight)
# ---------------------------------------------------------------------------

# Instagram carousels allow at most 10 slides; the cover takes one of them.
IG_MAX_SLIDES = 10


def _render_cover_slide(rbc, scope: str, today: datetime, window: int,
                        mode: str, out_path):
    """Render the formal "State Extravaganza" cover card (slide 1 of the
    carousel), reusing the bill-card renderer's canvas geometry, fonts, accent
    treatment, topic tag, and GOVBOT wordmark so it sits flush with the bill
    slides that follow."""
    from PIL import Image, ImageDraw

    theme = rbc.THEMES.get(mode, rbc.THEMES["light"])
    accent = tuple(TOPIC.card_accent)
    colors = rbc._accent_colors(accent, TOPIC.card_spectrum)
    INNER0, INNER1, INNER_W = rbc.INNER0, rbc.INNER1, rbc.INNER_W

    img = Image.new("RGB", (rbc.CARD, rbc.CARD), theme.bg)
    draw = ImageDraw.Draw(img)

    # Bottom band: a left label + the GOVBOT wordmark above a hairline divider.
    wm_font = rbc._mono(40, semibold=True)
    wm_w = rbc._wordmark_width(draw, wm_font, TOPIC.card_spectrum)
    wm_h = rbc._wordmark_height(wm_font)
    band_top = INNER1 - wm_h
    divider_y = band_top - 30
    hairline = tuple(round(theme.bg[k] + (theme.ink[k] - theme.bg[k]) * 0.16)
                     for k in range(3))
    draw.line([(INNER0, divider_y), (INNER1, divider_y)], fill=hairline, width=2)
    rbc._draw_wordmark(img, draw, round(INNER1 - wm_w), band_top,
                       colors, TOPIC.card_spectrum, theme)
    foot_font = rbc._mono(22, semibold=True)
    foot_y = band_top + (wm_h - rbc._line_h(foot_font, 1.0)) // 2
    rbc._draw_tracked(draw, INNER0, foot_y, "STATE EXTRAVAGANZA",
                      foot_font, theme.muted, tracking=2)

    # Top: topic tag chip.
    y = INNER0
    _, tag_h = rbc._draw_topic_tag(img, draw, INNER0, y,
                                   TOPIC.display_name.upper(),
                                   TOPIC.default_emoji, accent, theme)
    y += tag_h + 60

    # Title (big serif, auto-fit) with an accent underline beneath the last line.
    title = "State Extravaganza"
    title_lines = []
    title_font = rbc._serif(120, weight=700)
    for size in (120, 104, 92, 80, 70):
        title_font = rbc._serif(size, weight=700)
        title_lines = rbc._wrap(draw, title, title_font, INNER_W)
        if len(title_lines) <= 3:
            break
    title_lines = rbc._truncate(draw, title_lines, title_font, 3, INNER_W)
    title_lh = rbc._line_h(title_font, 1.0)
    for i, ln in enumerate(title_lines):
        draw.text((INNER0, y), ln, font=title_font, fill=theme.ink)
        if i == len(title_lines) - 1 and ln:
            lw = round(draw.textlength(ln, font=title_font))
            asc, _ = title_font.getmetrics()
            img.paste(rbc._h_gradient(lw, 8, colors), (INNER0, y + asc + 12))
        y += title_lh

    # Scope (states) line, then a quiet window/topic framing line.
    y += 40
    sub_font = rbc._serif(40, weight=600)
    for ln in rbc._truncate(draw, rbc._wrap(draw, scope, sub_font, INNER_W),
                            sub_font, 2, INNER_W):
        draw.text((INNER0, y), ln, font=sub_font, fill=theme.ink)
        y += rbc._line_h(sub_font, 1.2)

    y += 26
    framing = f"{TOPIC.display_name} bills · past {window} days"
    fr_font = rbc._mono(26)
    for ln in rbc._truncate(draw, rbc._wrap(draw, framing, fr_font, INNER_W),
                            fr_font, 3, INNER_W):
        draw.text((INNER0, y), ln, font=fr_font, fill=theme.muted)
        y += rbc._line_h(fr_font, 1.4)

    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def _compose_ig_caption(scope: str, window: int, prepared: list[dict],
                        max_caption: int) -> str:
    """One shared caption for the whole carousel: the formal header, the scope,
    a framing line, then a numbered index of each bill slide (with its plain
    bill link, since Instagram captions can't carry clickable links), trimmed to
    fit the caption budget."""
    from post_to_bluesky import display_identifier, link_for

    header = "🏛️ State Extravaganza!! 🧵"
    framing = (f"Spotlighting {TOPIC.display_name} bill activity from {scope} "
               f"over the past {window} days. Swipe through the highlights 👉")
    lines = [header, scope, "", framing, ""]

    for i, p in enumerate(prepared, 1):
        b = p["bill"]
        emoji = TOPIC.emoji_for(b)
        ident = display_identifier(b["state"], b["identifier"])
        head = (p["headline"] or b.get("title") or "").strip()
        url = link_for(b)
        block = f"{i}. {emoji} {b['state'] or '?'} {ident} — {head}"
        if url:
            block += f"\n   🔗 {url}"
        if len("\n".join(lines + [block])) > max_caption - 80:
            lines.append("…")
            break
        lines.append(block)

    topic_tag = "".join(c for c in TOPIC.display_name if c.isalnum())
    tags = " ".join(t for t in (f"#{topic_tag}" if topic_tag else "",
                                "#StateLegislature", "#Legislation",
                                "#CivicTech") if t)
    if len("\n".join(lines)) + len(tags) + 2 <= max_caption:
        lines += ["", tags]
    return "\n".join(lines)


def run_instagram(candidates: list[dict], scope: str, today: datetime,
                  window: int) -> int:
    import time

    import requests
    import render_bill_card as rbc
    from post_to_bluesky import (
        _slug,
        ensure_english_fields,
        save_full_text,
        shorten_title,
        summarize,
    )
    from post_to_instagram import (
        CARD_MODE,
        IG_API,
        IG_PUBLISH_RETRIES,
        IG_TIMEOUT,
        INSTAGRAM_ACCESS_TOKEN,
        INSTAGRAM_USER_ID,
        MAX_CAPTION,
        publish_cards,
        raw_url_for,
        render_bill_card,
        save_raw_record,
        wait_for_url,
        _repo_slug,
    )

    if not DRY_RUN and (not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_USER_ID):
        print("ERROR: INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_USER_ID must be set.",
              file=sys.stderr)
        return 1

    # Cover slide takes one of the 10 carousel slots.
    max_bills = min(NUM_POSTS, IG_MAX_SLIDES - 1)
    if NUM_POSTS > max_bills:
        print(f"  note: Instagram carousels cap at {IG_MAX_SLIDES} slides; "
              f"using a cover + {max_bills} bill slide(s).")
    highlights = select_extravaganza(candidates, max_bills, PER_STATE_CAP)
    _log_highlights(highlights, window)

    prepared: list[dict] = []
    for b in highlights:
        ensure_english_fields(b)
        headline = shorten_title(b)
        summary = summarize(b)
        card_path = render_bill_card(b, summary, headline)
        prepared.append({"bill": b, "headline": headline, "card": card_path})
        print(f"  prepared slide: {b['state']} {b['identifier']} "
              f"({b['action_date']}, score={b.get('_score', 0)})")

    cover_name = f"state-extravaganza-{_slug(scope, max_len=40) or 'all'}-{today.date()}.png"
    cover_path = _render_cover_slide(
        rbc, scope, today, window, CARD_MODE,
        TOPIC.instagram_cards_dir() / cover_name)
    caption = _compose_ig_caption(scope, window, prepared, MAX_CAPTION)
    card_paths = [cover_path] + [p["card"] for p in prepared]

    if DRY_RUN:
        print(f"\n--- COVER SLIDE: {cover_path.name} ({CARD_MODE} theme) ---")
        print(f"\n--- CAROUSEL CAPTION ({len(caption)} chars) ---\n{caption}\n---")
        for p in prepared:
            print(f"  slide: {p['bill']['state']} {p['bill']['identifier']} "
                  f"-> {p['card'].name}")
        print(f"\nDone (dry-run). Would post Instagram carousel: cover + "
              f"{len(prepared)} bill slide(s).")
        return 0

    # Push every card so Instagram can fetch it by raw.githubusercontent URL,
    # then build a child container per slide and bundle them into a carousel.
    sha = publish_cards(card_paths)
    if not sha:
        print(" ! could not publish card images; aborting.", file=sys.stderr)
        return 1
    slug = _repo_slug()

    children: list[str] = []
    for path in card_paths:
        url = raw_url_for(path, sha, slug)
        if not wait_for_url(url):
            if path == cover_path:
                print(" ! cover slide URL never became reachable; aborting.",
                      file=sys.stderr)
                return 1
            print(f" ! slide URL never reachable, skipping: {url}",
                  file=sys.stderr)
            continue
        try:
            resp = requests.post(
                f"{IG_API}/{INSTAGRAM_USER_ID}/media",
                data={"image_url": url, "is_carousel_item": "true",
                      "access_token": INSTAGRAM_ACCESS_TOKEN},
                timeout=IG_TIMEOUT,
            )
            resp.raise_for_status()
            child = resp.json().get("id")
            if child:
                children.append(child)
        except Exception as e:
            print(f" ! carousel child container failed: {e}", file=sys.stderr)
            if getattr(e, "response", None) is not None:
                print(f"   Response body: {e.response.text}", file=sys.stderr)

    if len(children) < 2:
        print(" ! fewer than 2 slides survived; need at least a cover + 1 bill "
              "for a carousel. Aborting.", file=sys.stderr)
        return 1

    try:
        resp = requests.post(
            f"{IG_API}/{INSTAGRAM_USER_ID}/media",
            data={"media_type": "CAROUSEL", "children": ",".join(children),
                  "caption": caption, "access_token": INSTAGRAM_ACCESS_TOKEN},
            timeout=IG_TIMEOUT,
        )
        resp.raise_for_status()
        parent_id = resp.json().get("id")
    except Exception as e:
        print(f" ! carousel container creation failed: {e}", file=sys.stderr)
        if getattr(e, "response", None) is not None:
            print(f"   Response body: {e.response.text}", file=sys.stderr)
        return 1
    if not parent_id:
        return 1

    media_id = None
    for attempt in range(1, IG_PUBLISH_RETRIES + 1):
        try:
            resp = requests.post(
                f"{IG_API}/{INSTAGRAM_USER_ID}/media_publish",
                data={"creation_id": parent_id,
                      "access_token": INSTAGRAM_ACCESS_TOKEN},
                timeout=IG_TIMEOUT,
            )
            resp.raise_for_status()
            media_id = resp.json().get("id")
            break
        except Exception as e:
            if attempt < IG_PUBLISH_RETRIES:
                time.sleep(5 * attempt)  # carousel may still be assembling
                continue
            print(f" ! carousel publish failed: {e}", file=sys.stderr)
            if getattr(e, "response", None) is not None:
                print(f"   Response body: {e.response.text}", file=sys.stderr)
            return 1
    if not media_id:
        return 1
    print(f"  posted Instagram carousel (media id {media_id})")

    for p in prepared:
        try:
            save_raw_record(p["bill"])
            save_full_text(p["bill"], out_dir=TOPIC.instagram_bills_full_text_dir())
        except Exception as e:
            print(f"  ! raw-record save failed for {p['bill'].get('state')} "
                  f"{p['bill'].get('identifier')}: {e}", file=sys.stderr)

    print(f"\nDone. Posted Instagram State Extravaganza carousel: cover + "
          f"{len(prepared)} bill slide(s).")
    return 0


_HANDLERS = {
    "bluesky": run_bluesky,
    "x": run_x,
    "threads": run_threads,
    "instagram": run_instagram,
}


def _log_highlights(highlights: list[dict], window: int) -> None:
    print(f"\nSelected {len(highlights)} bill(s) (max={NUM_POSTS}, "
          f"per-state-cap={PER_STATE_CAP}, window={window}d):")
    for b in highlights:
        print(f"  [{b['_score']:>3}] {b['state']} {b['identifier']} "
              f"({b['action_date']}): {b['action_desc'][:70]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if PLATFORM not in _HANDLERS:
        print(f"ERROR: PLATFORM must be one of {', '.join(_HANDLERS)} "
              f"(got {PLATFORM!r}).", file=sys.stderr)
        return 1

    states = parse_states()
    window = _resolve_lookback()
    scope = states_label(states)
    print(f"=== State Extravaganza: platform={PLATFORM}, topic={TOPIC.name}, "
          f"states={scope}, window={window}d, posts={NUM_POSTS} ===")

    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    all_bills = collect_topic_bills(records)
    if not all_bills:
        print(f"No {TOPIC.topic_phrase} bills found at all. Nothing to post.")
        return 0

    in_states = filter_by_states(all_bills, states)
    if not in_states:
        print(f"No {TOPIC.topic_phrase} bills found for {scope}. Nothing to post.")
        return 0

    candidates = candidates_in_window(in_states, today, window)
    print(f"Lookback {window}d: {len(candidates)} {TOPIC.topic_phrase} "
          f"bill update(s) from {scope}.")
    if not candidates:
        # Diagnose WHY there's nothing in the window: the topic does match
        # bills for these states across all dates, they're just older than the
        # cutoff. Surface the total and the most-recent action date so a zero
        # result distinguishes "out of session / stale" from "no such bills".
        scope_unique = {(b["state"], b["identifier"]) for b in in_states}
        latest = max((b["action_date"] for b in in_states if b["action_date"]),
                     default="")
        print(f"  diagnostic: {len(scope_unique)} {TOPIC.topic_phrase} bill(s) "
              f"exist for {scope} across all dates"
              + (f"; most recent action {latest} "
                 f"({(today - datetime.strptime(latest, '%Y-%m-%d')).days} days "
                 f"ago, past the {window}-day window)." if latest else "."))
        print(f"No {TOPIC.topic_phrase} activity for {scope} in the last "
              f"{window} days. Nothing to post.")
        return 0

    unique_bills = {(b["state"], b["identifier"]) for b in candidates}
    state_counts = Counter(s or "?" for s, _ in unique_bills)
    print(f"  unique bills: {len(unique_bills)} (from {len(candidates)} "
          f"action entries)")
    print(f"  by state: "
          f"{', '.join(f'{s}={n}' for s, n in state_counts.most_common(15))}")

    return _HANDLERS[PLATFORM](candidates, scope, today, window)


if __name__ == "__main__":
    sys.exit(main())
