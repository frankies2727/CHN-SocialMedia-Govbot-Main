#!/usr/bin/env python3
"""
Instagram bill-card renderer.

Instagram's Graph API has no text-only post type — every feed post needs an
image. This module turns the same (state, bill id, headline, summary, action)
data the other posters compose into a branded 1080x1350 (4:5) PNG card so the
Instagram poster has something to publish. The text is produced by the shared
pipeline in post_to_bluesky.py exactly as for Bluesky/X/Threads; this file only
concerns itself with laying it out as an image.

The card is dark-mode with a per-topic accent color (passed in by the poster
from the topic's config). The action + date line is folded into the main body
copy, and the footer points readers to the accompanying caption ("Link to the
bill in the description") since the Instagram post ships as a card image plus a
text caption and Instagram captions can't carry a clickable link.

Public entry point:

    render_card(bill, headline=..., summary=..., emoji=..., accent=..., out_path=...) -> Path

Run directly to emit a sample card from a real bill record for visual review:

    python scripts/render_bill_card.py [out.png]

Emoji are drawn from Noto Color Emoji when present (rendered to a bitmap tile
and resized), and silently omitted if that font or glyph is unavailable, so the
card never shows a "tofu" box.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Canvas geometry (Instagram 4:5 portrait, the tallest allowed feed ratio) -
CARD_W = 1080
CARD_H = 1350
MARGIN = 80

ACCENT_STRIPE_H = 16    # thin accent rule across the very top
HEADER_TOP = 96         # y where the emoji + state/bill-id header begins
HEADER_BOTTOM = 250     # body is vertically centered below this
EMOJI_PX = 92           # rendered emoji size in the header row

# --- Dark-mode palette ------------------------------------------------------
BG = (18, 18, 23)               # near-black card body
ACCENT = (37, 99, 235)          # default govbot blue; overridden per topic
HEADER_TEXT = (255, 255, 255)
HEADLINE_COLOR = (243, 244, 246)   # near-white
SUMMARY_COLOR = (191, 199, 212)    # light slate
FOOTER_COLOR = (138, 143, 152)     # muted gray
DIVIDER = (44, 46, 54)

# --- Fonts ------------------------------------------------------------------
# Liberation Sans is a metric-compatible Helvetica/Arial clone present on the
# GitHub Actions ubuntu runners; DejaVu is the universal fallback.
_FONT_CANDIDATES = {
    "bold": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "regular": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}
_EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


def _font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[weight]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Last resort: PIL's built-in bitmap font (won't scale, but never crashes).
    return ImageFont.load_default()


def _lighten(rgb: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    """Blend rgb toward white by fraction f, so a saturated accent stays legible
    as text on the dark background."""
    return tuple(round(c + (255 - c) * f) for c in rgb)


# State postal code -> full name. Kept local so the renderer can run standalone
# (e.g. the sample below) without importing the heavy posting module; the
# Instagram poster passes the already-resolved name through anyway.
STATE_FULL_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}

MONTHS = ["", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def _format_date(yyyy_mm_dd: str) -> str:
    try:
        y, m, d = (int(x) for x in (yyyy_mm_dd or "").split("-"))
        return f"{MONTHS[m]} {d}, {y}"
    except (ValueError, IndexError):
        return ""


# ---------------------------------------------------------------------------
# Emoji
# ---------------------------------------------------------------------------

def _render_emoji(emoji: str, target_px: int) -> Image.Image | None:
    """Render a single emoji to an RGBA tile of side target_px, or None if the
    color-emoji font / glyph isn't available. NotoColorEmoji only ships bitmap
    strikes at size 109, so we render at 109 with embedded_color and resize."""
    if not emoji or not Path(_EMOJI_FONT).exists():
        return None
    try:
        font = ImageFont.truetype(_EMOJI_FONT, 109)
        tile = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        d.text((0, 0), emoji, font=font, embedded_color=True)
        bbox = tile.getbbox()
        if not bbox:
            return None
        glyph = tile.crop(bbox)
        scale = target_px / max(glyph.width, glyph.height)
        new_size = (max(1, round(glyph.width * scale)),
                    max(1, round(glyph.height * scale)))
        return glyph.resize(new_size, Image.Resampling.LANCZOS)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------

def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
          max_w: int) -> list[str]:
    """Greedy word-wrap to max_w pixels."""
    lines: list[str] = []
    for paragraph in (text or "").split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            trial = f"{cur} {w}"
            if draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def _block_height(lines, font, line_gap, max_lines=None) -> int:
    """Pixel height of a wrapped block as _draw_block would render it, so the
    body can be measured up-front and vertically centered."""
    asc, desc = font.getmetrics()
    n = len(lines) if max_lines is None else min(len(lines), max_lines)
    return (asc + desc + line_gap) * n


def _draw_block(draw, lines, font, x, y, color, line_gap, max_lines=None,
                ellipsis_after=None):
    """Draw wrapped lines top-down; returns the y below the block. If max_lines
    is set, truncate and append an ellipsis to the last kept line."""
    asc, desc = font.getmetrics()
    line_h = asc + desc + line_gap
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > (ellipsis_after or 10**9):
            last = last[:-1].rstrip()
        lines[-1] = (last + "…") if last else "…"
    for ln in lines:
        draw.text((x, y), ln, font=font, fill=color)
        y += line_h
    return y


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

def render_card(
    bill: dict,
    *,
    headline: str = "",
    summary: str = "",
    emoji: str = "",
    accent: tuple[int, int, int] = ACCENT,
    brand: str = "govbot",
    out_path: str | Path = "card.png",
) -> Path:
    """Render a bill into a 1080x1350 dark-mode PNG card and return the output
    Path.

    bill is the dict shape produced by post_to_bluesky.extract_fields (uses
    keys: state, identifier, action_desc, action_date, title). headline and
    summary are the already-composed strings from the shared pipeline; emoji is
    the topic emoji and accent the topic's card color (TOPIC.card_accent)."""
    img = Image.new("RGB", (CARD_W, CARD_H), BG)
    draw = ImageDraw.Draw(img)

    accent = tuple(accent)
    accent_text = _lighten(accent, 0.30)   # legible accent for text on dark BG

    state = (bill.get("state") or "").upper()
    state_name = STATE_FULL_NAME.get(state, state or "Legislature")
    identifier = bill.get("identifier") or ""
    display = (headline or bill.get("title") or "").strip()
    summary = (summary or "").strip()

    inner_w = CARD_W - 2 * MARGIN

    # --- Top accent stripe + header (header sits on the dark body, not on a
    # colored band) ---------------------------------------------------------
    draw.rectangle([0, 0, CARD_W, ACCENT_STRIPE_H], fill=accent)

    state_font = _font("bold", 52)
    id_font = _font("regular", 38)
    hx = MARGIN
    emoji_img = _render_emoji(emoji, EMOJI_PX)
    # Center the emoji against the two stacked text lines (state + id).
    text_block_h = 60 + id_font.getmetrics()[0]
    if emoji_img is not None:
        ey = HEADER_TOP + (text_block_h - emoji_img.height) // 2
        img.paste(emoji_img, (hx, ey), emoji_img)
        hx += emoji_img.width + 28
    draw.text((hx, HEADER_TOP), state_name, font=state_font, fill=HEADLINE_COLOR)
    draw.text((hx, HEADER_TOP + 60), identifier, font=id_font, fill=accent_text)

    # --- Footer (measured first so the body can center between it and the
    # header) ---------------------------------------------------------------
    foot_y = CARD_H - MARGIN - 70

    # --- Body: measure every block, then vertically center the whole stack
    # between the header and the footer ------------------------------------
    headline_font = _font("bold", 60)
    summary_font = _font("regular", 42)
    action_font = _font("bold", 40)
    GAP_HEAD_SUM = 36
    GAP_SUM_DATE = 78   # extra breathing room so the date sits a little lower

    head_lines = _wrap(draw, display, headline_font, inner_w)
    head_h = _block_height(head_lines, headline_font, 12, max_lines=4)

    has_summary = bool(summary and summary.strip().lower() != display.strip().lower())
    sum_lines = _wrap(draw, summary, summary_font, inner_w) if has_summary else []
    sum_h = _block_height(sum_lines, summary_font, 14, max_lines=11) if has_summary else 0

    nice_date = _format_date(bill.get("action_date", ""))
    action = (bill.get("action_desc") or "").strip().rstrip(".")
    action_line = " · ".join(p for p in (nice_date, action) if p)
    act_lines = _wrap(draw, action_line, action_font, inner_w) if action_line else []
    act_h = _block_height(act_lines, action_font, 10, max_lines=3) if action_line else 0

    total_h = head_h
    if has_summary:
        total_h += GAP_HEAD_SUM + sum_h
    if action_line:
        total_h += GAP_SUM_DATE + act_h

    region_top = HEADER_BOTTOM
    region_bottom = foot_y - 40
    y = region_top + max(0, (region_bottom - region_top - total_h) // 2)

    y = _draw_block(draw, head_lines, headline_font, MARGIN, y, HEADLINE_COLOR,
                    line_gap=12, max_lines=4, ellipsis_after=inner_w)
    if has_summary:
        y += GAP_HEAD_SUM
        y = _draw_block(draw, sum_lines, summary_font, MARGIN, y, SUMMARY_COLOR,
                        line_gap=14, max_lines=11, ellipsis_after=inner_w)
    # Action + date line, folded into the main copy (not the footer), styled in
    # the accent so the legislative status reads as part of the message.
    if action_line:
        y += GAP_SUM_DATE
        _draw_block(draw, act_lines, action_font, MARGIN, y, accent_text,
                    line_gap=10, max_lines=3, ellipsis_after=inner_w)

    # --- Footer -------------------------------------------------------------
    draw.line([(MARGIN, foot_y), (CARD_W - MARGIN, foot_y)], fill=DIVIDER, width=2)

    label_font = _font("regular", 34)
    brand_font = _font("bold", 36)
    draw.text((MARGIN, foot_y + 28), "Link to the bill in the description",
              font=label_font, fill=FOOTER_COLOR)
    brand_w = draw.textlength(brand, font=brand_font)
    draw.text((CARD_W - MARGIN - brand_w, foot_y + 26), brand, font=brand_font,
              fill=accent_text)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
# Sample (visual review)
# ---------------------------------------------------------------------------

# LGBTQ+ is the launch topic, so the sample mirrors that account's emoji/accent.
_SAMPLE_BILL = {
    "state": "MN",
    "identifier": "HF1234",
    "title": "Conversion therapy prohibition for minors.",
    "action_desc": "Signed by the Governor.",
    "action_date": "2026-06-15",
}
_SAMPLE_HEADLINE = "Banning conversion therapy for minors statewide"
_SAMPLE_SUMMARY = (
    "Bars licensed mental health providers from practicing conversion therapy "
    "on patients under 18 and makes any violation grounds for professional "
    "disciplinary action."
)
_LGBTQ_ACCENT = (192, 38, 211)   # fuchsia; per-topic card_accent for lgbtq


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "instagram-card-sample.png"
    path = render_card(
        _SAMPLE_BILL,
        headline=_SAMPLE_HEADLINE,
        summary=_SAMPLE_SUMMARY,
        emoji="🏳️‍🌈",
        accent=_LGBTQ_ACCENT,
        out_path=out,
    )
    print(f"Wrote sample card to {path.resolve()}")
