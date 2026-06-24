#!/usr/bin/env python3
"""
Instagram bill-card renderer — "Daylight" template.

Instagram's Graph API has no text-only post type — every feed post needs an
image. This module turns the same (state, bill id, headline, summary, status,
date) data the other posters compose into a branded 1080x1080 (1:1) PNG card so
the Instagram poster has something to publish. The text is produced by the
shared pipeline in post_to_bluesky.py exactly as for Bluesky/X/Threads; this
file only concerns itself with laying it out as an image.

The layout is the "GovBot Post" design (Direction A — Daylight): a light cream
card sitting inside a colored frame, with the GOVBOT wordmark, a state/bill-id
eyebrow, a serif (Newsreader) headline with a highlighter underline on its last
line, a monospace (IBM Plex Mono) summary, side-by-side STATUS and DATE cards,
and a footer pointing readers to the caption ("Full bill linked in description")
since Instagram captions can't carry a clickable link.

Color treatment is driven by `spectrum`:
  * spectrum=True  -> the LGBTQ+ pride rainbow is used for the frame, the
    wordmark dots, the headline highlight, and the STATUS/DATE accent bars.
  * spectrum=False -> the topic's single accent color (passed by the poster
    from the topic config) is used as a two-stop gradient in those same places.

Public entry point:

    render_card(bill, headline=..., summary=..., accent=..., spectrum=...,
                out_path=...) -> Path

Run directly to emit a sample card from a real bill record for visual review:

    python scripts/render_bill_card.py [out.png]

Fonts (IBM Plex Mono, Newsreader) are vendored under assets/fonts/ so the card
renders identically on the GitHub Actions runner; if they're missing the code
falls back to DejaVu so it never crashes.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Canvas geometry (Instagram 1:1 square, matching the design template) -----
CARD = 1080
FRAME = 22          # colored border thickness around the cream card
PAD = 74            # inner padding inside the cream card
INNER0 = FRAME + PAD            # content box left/top (= 96)
INNER1 = CARD - FRAME - PAD     # content box right/bottom (= 984)
INNER_W = INNER1 - INNER0       # = 888

# --- Palette ----------------------------------------------------------------
# The card ships in two themes — "light" (the template's daylight cream) and
# "dark" — sharing the same layout; the poster picks one per run. Each theme
# bundles the five tones the layout needs: card body, headline/value ink, the
# muted summary/footer tone, the tile background, and the tile label tone.
LABEL_SIZE = 16
DEFAULT_ACCENT = (37, 99, 235)  # govbot blue fallback


class Theme:
    def __init__(self, bg, ink, muted, tile_bg, tile_label):
        self.bg = bg                  # card body fill
        self.ink = ink                # wordmark, headline, tile values
        self.muted = muted            # summary + footer
        self.tile_bg = tile_bg        # STATUS/DATE tile fill
        self.tile_label = tile_label  # STATUS/DATE labels


THEMES = {
    "light": Theme(
        bg=(250, 247, 240),     # #faf7f0 cream
        ink=(27, 26, 23),       # #1b1a17
        muted=(87, 84, 76),     # #57544c
        tile_bg=(241, 237, 226),  # #f1ede2
        tile_label=(74, 71, 64),
    ),
    "dark": Theme(
        bg=(24, 23, 20),        # warm near-black
        ink=(255, 255, 255),    # white
        muted=(255, 255, 255),  # white (all dark-mode text white for legibility)
        tile_bg=(38, 36, 32),   # slightly elevated panel
        tile_label=(255, 255, 255),
    ),
}

# Pride spectrum used when spectrum=True (the LGBTQ+ launch palette).
PRIDE = [
    (228, 3, 3),     # #E40303 red
    (255, 140, 0),   # #FF8C00 orange
    (255, 212, 0),   # #FFD400 yellow
    (30, 158, 62),   # #1E9E3E green
    (31, 111, 235),  # #1F6FEB blue
    (123, 44, 191),  # #7B2CBF violet
]

# --- Fonts ------------------------------------------------------------------
_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_MONO_REGULAR = _FONT_DIR / "IBMPlexMono-Regular.ttf"
_MONO_SEMIBOLD = _FONT_DIR / "IBMPlexMono-SemiBold.ttf"
_SERIF_VF = _FONT_DIR / "Newsreader.ttf"
# Color-emoji font for the topic emoji (eyebrow) and the 🔗 in the footer.
# Vendored so the runner renders glyphs identically; falls back to the system
# copy, and the card silently omits the emoji if neither is present.
_EMOJI_CANDIDATES = [
    _FONT_DIR / "NotoColorEmoji.ttf",
    Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
]

# DejaVu is the universal fallback present on the runners if the vendored fonts
# are ever unavailable, so the card degrades instead of crashing.
_MONO_FALLBACK = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_SERIF_FALLBACK = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]


@lru_cache(maxsize=None)
def _mono(size: int, semibold: bool = False) -> ImageFont.FreeTypeFont:
    path = _MONO_SEMIBOLD if semibold else _MONO_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    for fb in _MONO_FALLBACK:
        if Path(fb).exists():
            return ImageFont.truetype(fb, size)
    return ImageFont.load_default()


@lru_cache(maxsize=None)
def _serif(size: int, weight: int = 700) -> ImageFont.FreeTypeFont:
    """Newsreader instance at the given weight. The vendored file is a variable
    font (Weight 200-800, Optical Size 6-72); we pin the weight and let the
    optical size track the point size so large display text stays crisp."""
    if _SERIF_VF.exists():
        font = ImageFont.truetype(str(_SERIF_VF), size)
        try:
            # Axis order from get_variation_axes(): [Weight, Optical Size].
            font.set_variation_by_axes([weight, max(6, min(size, 72))])
        except Exception:
            pass
        return font
    for fb in _SERIF_FALLBACK:
        if Path(fb).exists():
            return ImageFont.truetype(fb, size)
    return ImageFont.load_default()


@lru_cache(maxsize=None)
def _emoji_font_path() -> str | None:
    for p in _EMOJI_CANDIDATES:
        if Path(p).exists():
            return str(p)
    return None


def _render_emoji(emoji: str, target_px: int) -> Image.Image | None:
    """Render a single emoji to an RGBA tile target_px tall, or None if the
    color-emoji font / glyph isn't available. Noto Color Emoji only ships bitmap
    strikes at size 109, so we render at 109 with embedded_color and resize."""
    path = _emoji_font_path()
    if not emoji or not path:
        return None
    try:
        font = ImageFont.truetype(path, 109)
        tile = Image.new("RGBA", (200, 160), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        d.text((0, 0), emoji, font=font, embedded_color=True)
        bbox = tile.getbbox()
        if not bbox:
            return None
        glyph = tile.crop(bbox)
        scale = target_px / glyph.height
        new_size = (max(1, round(glyph.width * scale)), max(1, target_px))
        return glyph.resize(new_size, Image.Resampling.LANCZOS)
    except Exception:
        return None


def _lighten(rgb: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    """Blend rgb toward white by fraction f."""
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
# Gradients
# ---------------------------------------------------------------------------

def _interp(colors: list[tuple[int, int, int]], t: float) -> tuple[int, int, int]:
    """Color at position t in [0, 1] along an evenly-spaced multi-stop ramp."""
    if t <= 0:
        return colors[0]
    if t >= 1:
        return colors[-1]
    span = len(colors) - 1
    pos = t * span
    i = int(pos)
    frac = pos - i
    a, b = colors[i], colors[i + 1]
    return tuple(round(a[k] + (b[k] - a[k]) * frac) for k in range(3))


def _h_gradient(w: int, h: int, colors: list[tuple[int, int, int]]) -> Image.Image:
    """A w x h image whose color ramps horizontally (left -> right) across the
    evenly-spaced color stops."""
    w = max(1, w)
    row = [_interp(colors, x / (w - 1) if w > 1 else 0.0) for x in range(w)]
    strip = Image.new("RGB", (w, 1))
    strip.putdata(row)
    return strip.resize((w, max(1, h)))


def _v_gradient(w: int, h: int, colors: list[tuple[int, int, int]]) -> Image.Image:
    """A w x h image whose color ramps vertically (top -> bottom)."""
    h = max(1, h)
    col = [_interp(colors, y / (h - 1) if h > 1 else 0.0) for y in range(h)]
    strip = Image.new("RGB", (1, h))
    strip.putdata(col)
    return strip.resize((max(1, w), h))


def _diagonal_gradient(size: int, colors: list[tuple[int, int, int]],
                       angle_deg: float = 30.0) -> Image.Image:
    """A size x size image with the color ramp running on a diagonal, so a thin
    frame shows the first color at the top-left and the last at the bottom-right
    (mirrors the template's 120deg frame gradient)."""
    diag = int(size * 1.5) + 2
    g = _h_gradient(diag, diag, colors)
    g = g.rotate(angle_deg, resample=Image.BICUBIC, expand=False)
    off = (diag - size) // 2
    return g.crop((off, off, off + size, off + size))


def _accent_colors(accent: tuple[int, int, int], spectrum: bool
                   ) -> list[tuple[int, int, int]]:
    """The color stops used for frame/accents: the pride rainbow when spectrum,
    otherwise a two-stop ramp built from the topic's accent."""
    return list(PRIDE) if spectrum else [accent, _lighten(accent, 0.45)]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
          max_w: int) -> list[str]:
    """Greedy word-wrap to max_w pixels."""
    lines: list[str] = []
    for paragraph in (text or "").split("\n"):
        words = paragraph.split()
        if not words:
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


def _truncate(draw, lines: list[str], font, max_lines: int, max_w: int
              ) -> list[str]:
    """Cap a wrapped block to max_lines, adding an ellipsis to the last kept
    line if anything was dropped."""
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    last = kept[-1]
    while last and draw.textlength(last + "…", font=font) > max_w:
        last = last[:-1].rstrip()
    kept[-1] = (last + "…") if last else "…"
    return kept


def _line_h(font: ImageFont.FreeTypeFont, factor: float) -> int:
    asc, desc = font.getmetrics()
    return round((asc + desc) * factor)


def _draw_tracked(draw, x: float, y: float, text: str,
                  font: ImageFont.FreeTypeFont, fill, tracking: float) -> float:
    """Draw text with extra letter-spacing (PIL has no native tracking).
    Returns the x just past the last glyph."""
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking
    return x - tracking if text else x


def _tracked_width(draw, text: str, font: ImageFont.FreeTypeFont,
                   tracking: float) -> float:
    if not text:
        return 0.0
    return sum(draw.textlength(ch, font=font) for ch in text) + tracking * (len(text) - 1)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _rainbow_dot(d: int, colors: list[tuple[int, int, int]], spectrum: bool
                 ) -> Image.Image:
    """A circular dot for the wordmark O's: hard horizontal pride stripes when
    spectrum, otherwise a solid accent disc."""
    d = max(2, d)
    if spectrum:
        base = Image.new("RGB", (d, d))
        bd = ImageDraw.Draw(base)
        n = len(colors)
        for i, c in enumerate(colors):
            y0 = round(d * i / n)
            y1 = round(d * (i + 1) / n)
            bd.rectangle([0, y0, d, y1], fill=c)
    else:
        base = Image.new("RGB", (d, d), colors[0])
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, d - 1, d - 1], fill=255)
    out = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    out.paste(base, (0, 0), mask)
    return out


def _wordmark_height(font: ImageFont.FreeTypeFont) -> int:
    asc, desc = font.getmetrics()
    return asc + desc


def _draw_wordmark(img, draw, x: int, y: int, colors, spectrum: bool,
                   theme: Theme) -> None:
    """Render the GOVBOT wordmark with the two O's replaced by rainbow dots."""
    font = _mono(40, semibold=True)
    tracking = 8
    asc, _ = font.getmetrics()
    dot_d = round(40 * 0.8)
    cy = y + asc * 0.55          # vertical center of the cap height
    segments = [("G", False), ("O", True), ("VB", False), ("O", True), ("T", False)]
    cx = float(x)
    for text, is_dot in segments:
        if is_dot:
            dot = _rainbow_dot(dot_d, colors, spectrum)
            img.paste(dot, (round(cx), round(cy - dot_d / 2)), dot)
            cx += dot_d + tracking
        else:
            cx = _draw_tracked(draw, cx, y, text, font, theme.ink, tracking) + tracking


def _draw_status_tile(img, draw, x: int, y: int, w: int, label: str, value: str,
                      colors, spectrum: bool, theme: Theme) -> int:
    """Draw a rounded STATUS/DATE tile and return its height."""
    pad_x, pad_y = 28, 24
    bar_h = 8
    radius = 6
    label_font = _mono(LABEL_SIZE, semibold=True)
    value_font = _serif(38, weight=600)

    text_w = w - 2 * pad_x
    value_lines = _truncate(draw, _wrap(draw, value, value_font, text_w),
                            value_font, max_lines=2, max_w=text_w)
    val_lh = _line_h(value_font, 1.05)
    label_lh = _line_h(label_font, 1.2)

    tile_h = bar_h + pad_y + label_lh + 6 + val_lh * len(value_lines) + pad_y

    # Tile body (rounded) then the gradient accent bar across the top.
    draw.rounded_rectangle([x, y, x + w, y + tile_h], radius=radius, fill=theme.tile_bg)
    bar = _h_gradient(w, bar_h, colors)
    bar_mask = Image.new("L", (w, bar_h), 0)
    ImageDraw.Draw(bar_mask).rounded_rectangle([0, 0, w, bar_h + radius],
                                               radius=radius, fill=255)
    img.paste(bar, (x, y), bar_mask)

    ty = y + bar_h + pad_y
    _draw_tracked(draw, x + pad_x, ty, label, label_font, theme.tile_label, tracking=3)
    ty += label_lh + 6
    for ln in value_lines:
        draw.text((x + pad_x, ty), ln, font=value_font, fill=theme.ink)
        ty += val_lh
    return tile_h


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

def render_card(
    bill: dict,
    *,
    headline: str = "",
    summary: str = "",
    emoji: str = "",   # topic emoji, shown to the left of the state in the eyebrow
    accent: tuple[int, int, int] = DEFAULT_ACCENT,
    spectrum: bool = False,
    mode: str = "light",
    brand: str = "govbot",
    out_path: str | Path = "card.png",
) -> Path:
    """Render a bill into a 1080x1080 PNG card and return the output Path.

    The layout is the GovBot Post design; mode picks the "light" (daylight
    cream) or "dark" theme — both share the layout and the colored frame.

    bill is the dict shape produced by post_to_bluesky.extract_fields (uses
    keys: state, identifier, action_desc, action_date, title). headline and
    summary are the already-composed strings from the shared pipeline; accent is
    the topic's card color (TOPIC.card_accent) and spectrum selects the pride
    rainbow (TOPIC.card_spectrum) over that single accent."""
    accent = tuple(accent)
    colors = _accent_colors(accent, spectrum)
    theme = THEMES.get(mode, THEMES["light"])

    img = Image.new("RGB", (CARD, CARD), theme.bg)
    # Colored frame: paste the diagonal gradient full-bleed, then lay the card
    # body on top inset by FRAME so only the border shows.
    img.paste(_diagonal_gradient(CARD, colors, angle_deg=30.0), (0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([FRAME, FRAME, CARD - FRAME, CARD - FRAME],
                           radius=2, fill=theme.bg)

    state = (bill.get("state") or "").upper()
    state_name = STATE_FULL_NAME.get(state, state or "Legislature")
    identifier = (bill.get("identifier") or "").strip()
    display = (headline or bill.get("title") or "").strip()
    summary = (summary or "").strip()

    # ---- measure the four stacked blocks (header / hero / status / footer) so
    # they can be distributed top-to-bottom like the template's space-between --
    wordmark_font = _mono(40, semibold=True)

    eyebrow = " · ".join(p for p in (state_name, identifier) if p).upper()
    eyebrow_font = _mono(30, semibold=True)
    eyebrow_h = _line_h(eyebrow_font, 1.1)

    # Auto-fit the headline: keep it as large as the template's 90px when the
    # copy is short, but step the size down so longer headlines stay within ~3
    # lines instead of swelling the hero block and colliding with the tiles.
    headline_font = _serif(90, weight=700)
    head_lines = _wrap(draw, display, headline_font, INNER_W)
    for size in (90, 80, 72, 64, 58):
        headline_font = _serif(size, weight=700)
        head_lines = _wrap(draw, display, headline_font, INNER_W)
        if len(head_lines) <= 3:
            break
    head_lines = _truncate(draw, head_lines, headline_font, max_lines=4, max_w=INNER_W)
    head_lh = _line_h(headline_font, 0.98)
    head_h = head_lh * len(head_lines)

    summary_font = _mono(25)
    has_summary = bool(summary and summary.lower() != display.lower())
    sum_lines = (_truncate(draw, _wrap(draw, summary, summary_font, INNER_W),
                           summary_font, max_lines=4, max_w=INNER_W) if has_summary else [])
    sum_lh = _line_h(summary_font, 1.45)
    sum_h = sum_lh * len(sum_lines)

    GAP_EYE_HEAD = 18
    GAP_HEAD_SUM = 34
    hero_h = eyebrow_h + GAP_EYE_HEAD + head_h + (GAP_HEAD_SUM + sum_h if has_summary else 0)

    # Status / date tiles
    status_val = (bill.get("action_desc") or "").strip().rstrip(".")
    if status_val:
        status_val = status_val[0].upper() + status_val[1:]
    date_val = _format_date(bill.get("action_date", ""))
    show_tiles = bool(status_val or date_val)
    tile_gap = 24
    tile_w = (INNER_W - tile_gap) // 2
    # Measure tile height up front (height depends on the wrapped value).
    status_h = 0
    if show_tiles:
        vf = _serif(38, weight=600)
        tw = tile_w - 2 * 28
        sl = len(_truncate(draw, _wrap(draw, status_val or "—", vf, tw), vf, 2, tw))
        dl = len(_truncate(draw, _wrap(draw, date_val or "—", vf, tw), vf, 2, tw))
        status_h = 8 + 24 + _line_h(_mono(LABEL_SIZE, semibold=True), 1.2) + 6 + _line_h(vf, 1.05) * max(sl, dl) + 24

    footer_font = _mono(18)
    # The footer row now carries the GOVBOT wordmark on the left (moved down from
    # the old header) next to the caption pointer, so the row is as tall as the
    # wordmark.
    footer_h = max(_line_h(footer_font, 1.2), _wordmark_height(wordmark_font))

    blocks = [hero_h]
    if show_tiles:
        blocks.append(status_h)
    blocks.append(footer_h)
    used = sum(blocks)
    gaps = len(blocks) - 1
    gap = max(28, (INNER_W - used) // gaps) if gaps else 0

    # ---- draw ---------------------------------------------------------------
    y = INNER0

    # Hero: eyebrow + headline (with highlighter underline on last line) + summary.
    # The topic emoji (when available) leads the eyebrow, to the left of the state.
    ex = INNER0
    emoji_img = _render_emoji(emoji, round(eyebrow_font.size * 1.15))
    if emoji_img is not None:
        asc, _ = eyebrow_font.getmetrics()
        img.paste(emoji_img, (ex, round(y + asc * 0.5 - emoji_img.height / 2)), emoji_img)
        ex += emoji_img.width + 18
    _draw_tracked(draw, ex, y, eyebrow, eyebrow_font, theme.ink, tracking=3)
    y += eyebrow_h + GAP_EYE_HEAD

    for i, ln in enumerate(head_lines):
        is_last = i == len(head_lines) - 1
        if is_last and ln:
            # Highlighter bar behind the last line's text, like the template.
            lw = draw.textlength(ln, font=headline_font)
            asc, _ = headline_font.getmetrics()
            hl_h = round(head_lh * 0.24)
            hl_y = y + round(asc * 0.74)
            hl = _h_gradient(round(lw), hl_h, colors)
            img.paste(hl, (INNER0, hl_y))
        draw.text((INNER0, y), ln, font=headline_font, fill=theme.ink)
        y += head_lh

    if has_summary:
        y += GAP_HEAD_SUM
        for ln in sum_lines:
            draw.text((INNER0, y), ln, font=summary_font, fill=theme.muted)
            y += sum_lh

    # Status / date tiles, anchored above the footer but never allowed to ride
    # up into the summary (the max() guards the long-copy edge case).
    if show_tiles:
        ty = max(INNER0 + INNER_W - footer_h - gap - status_h, y + 24)
        _draw_status_tile(img, draw, INNER0, ty, tile_w, "STATUS",
                          status_val or "—", colors, spectrum, theme)
        _draw_status_tile(img, draw, INNER0 + tile_w + tile_gap, ty, tile_w,
                          "DATE", date_val or "—", colors, spectrum, theme)

    # Footer: the GOVBOT wordmark (left, moved down from the old header) and the
    # caption pointer (right), pinned to the bottom of the content box. The
    # wordmark sets the row height; the link is vertically centered against it.
    fy = INNER0 + INNER_W - footer_h
    _draw_wordmark(img, draw, INNER0, fy, colors, spectrum, theme)
    # Right side: 🔗 + "Link to the bill in the description", right-aligned and
    # vertically centered within the footer row.
    link_text = "Link to the bill in the description"
    link_w = draw.textlength(link_text, font=footer_font)
    link_emoji = _render_emoji("🔗", round(footer_font.size * 1.2))
    emoji_w = (link_emoji.width + 8) if link_emoji is not None else 0
    sx = INNER1 - link_w - emoji_w
    link_lh = _line_h(footer_font, 1.2)
    link_y = fy + (footer_h - link_lh) // 2
    if link_emoji is not None:
        asc, _ = footer_font.getmetrics()
        img.paste(link_emoji, (round(sx), round(link_y + asc * 0.5 - link_emoji.height / 2)),
                  link_emoji)
    draw.text((sx + emoji_w, link_y), link_text, font=footer_font, fill=theme.muted)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
# Sample (visual review)
# ---------------------------------------------------------------------------

# LGBTQ+ is the launch topic, so the sample mirrors that account's accent and
# uses the pride spectrum.
_SAMPLE_BILL = {
    "state": "DE",
    "identifier": "HCR 145",
    "title": "June is officially Pride Month",
    "action_desc": "Introduced",
    "action_date": "2026-06-01",
}
_SAMPLE_HEADLINE = "June is officially Pride Month"
_SAMPLE_SUMMARY = (
    "Delaware will officially recognize June 2026 as LGBTQ+ Pride Month."
)
_LGBTQ_ACCENT = (192, 38, 211)   # #C026D3 fuchsia; lgbtq card_accent


if __name__ == "__main__":
    # Optional args: [out_path] [mode]; with no mode, emit both light and dark.
    out = sys.argv[1] if len(sys.argv) > 1 else "instagram-card-sample.png"
    modes = [sys.argv[2]] if len(sys.argv) > 2 else ["light", "dark"]
    for m in modes:
        target = out if len(modes) == 1 else f"{Path(out).with_suffix('')}-{m}.png"
        path = render_card(
            _SAMPLE_BILL,
            headline=_SAMPLE_HEADLINE,
            summary=_SAMPLE_SUMMARY,
            emoji="🏳️‍🌈",
            accent=_LGBTQ_ACCENT,
            spectrum=True,
            mode=m,
            out_path=target,
        )
        print(f"Wrote {m} sample card to {path.resolve()}")
