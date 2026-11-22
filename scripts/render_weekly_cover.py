#!/usr/bin/env python3
"""
The Instagram weekly digest's cover, designed as a magazine "issue" so the weekly
post reads differently in the feed from the daily v2 bill cards (which are
full-bleed topic colour). The bill slides that follow it are v2 cards.

Layout (1080x1350, the same 4:5 as the v2 cards, so the carousel is one size):
  * Warm paper background, near-black ink, Govbot green for the date line.
  * Top strip: "NO. <week of the year>" on the right, then a thick + thin gold
    rule (the Govbot site's gold), like a newspaper masthead.
  * Masthead: "Legislation Bills / on the Move" in big bold type, the date range
    under it.
  * "IN THIS ISSUE": one row per bill slide, in slide order, a table of contents:
    the topic's emoji in a circle of its colour, the topic name, and the state and
    bill number.
  * Footer: a gold rule, "Swipe for each bill" with a drawn arrow, and the GOVBOT
    wordmark.
A quiet week (nothing to feature) keeps the masthead and says so in big type.

Public entry point:
    render_weekly_cover(entries, date_label=..., issue_no=..., out_path=...,
                        quiet=False) -> Path
where each entry is {emoji, topic, accent, state, identifier}.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_bill_card as R  # noqa: E402  (fonts, emoji, wrap, state names)

W, H = 1080, 1350
PAD = 80
PAPER = (247, 242, 232)
INK = (27, 26, 23)
GREEN = (11, 125, 69)          # Govbot brand green (light theme), 5.2:1 on PAPER
RULE = (27, 26, 23)
FAINT = (214, 207, 192)        # row separators

# Masthead title (two lines) and the gold of the masthead/footer rules: the
# Govbot site's gold, as on its "Govbot" wordmark and gold accents.
TITLE = ("Legislation Bills", "on the Move")
GOLD = (229, 185, 78)          # #E5B94E


def _save(img: Image.Image, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(out_path, "JPEG", quality=92, subsampling=0, optimize=True)
    else:
        img.save(out_path, "PNG")
    return out_path


def _masthead(d: ImageDraw.ImageDraw, date_label: str, issue_no: int | None) -> int:
    """Top strip, rules, title and date line. Returns the y below them."""
    small = R._font(26, "bold")
    if issue_no:
        no = f"NO. {issue_no}"
        R._draw_tracked(d, W - PAD - R._tracked_width(d, no, small, 6), PAD, no, small, INK, 6)
    d.rectangle([PAD, 134, W - PAD, 140], fill=GOLD)
    d.rectangle([PAD, 148, W - PAD, 150], fill=GOLD)

    lines = TITLE
    # The largest size (up to 132px) at which the longer line fits the width.
    size = 132
    while size > 72 and max(d.textlength(t, font=R._font(size, "bold")) for t in lines) > W - 2 * PAD + 6:
        size -= 4
    tf = R._font(size, "bold")
    y = 170 + (132 - size)          # a smaller title sits lower, so the date line stays put
    for line in lines:
        d.text((PAD - 6, y), line, font=tf, fill=INK)
        y += size
    y += 48
    df = R._font(40, "medium")
    d.text((PAD, y), (date_label or "").upper(), font=df, fill=GREEN)
    y += 64
    d.rectangle([PAD, y, W - PAD, y + 2], fill=RULE)
    return y + 2


def _footer(d: ImageDraw.ImageDraw, left: str) -> None:
    d.rectangle([PAD, H - 150, W - PAD, H - 144], fill=GOLD)
    if left:
        lf = R._font(32, "medium")
        d.text((PAD, H - 108), left, font=lf, fill=INK)
        # Roboto has no arrow glyph, so draw one: a shaft and a filled head.
        ax = PAD + d.textlength(left, font=lf) + 18
        ay = H - 108 + 21
        d.line([(ax, ay), (ax + 34, ay)], fill=INK, width=4)
        d.polygon([(ax + 46, ay), (ax + 30, ay - 10), (ax + 30, ay + 10)], fill=INK)
    wf = R._font(34, "bold")
    R._draw_tracked(d, W - PAD - R._tracked_width(d, "GOVBOT", wf, 7), H - 106,
                    "GOVBOT", wf, INK, 7)


def _row(img, d, y: int, h: int, e: dict) -> None:
    """One contents row: emoji disc, topic, and the state + bill at the right."""
    cy = y + h // 2
    r = min(34, h // 2 - 6)
    accent = tuple(e.get("accent") or GREEN)
    d.ellipse([PAD, cy - r, PAD + 2 * r, cy + r], fill=accent)
    emo = R._render_emoji(e.get("emoji") or "", round(r * 1.25))
    if emo:
        img.paste(emo, (round(PAD + r - emo.width / 2), round(cy - emo.height / 2)), emo)

    state = (e.get("state") or "").upper()
    meta = f"{R.STATE_FULL_NAME.get(state, state)} · {e.get('identifier', '')}"
    mf = R._font(28, "regular")
    mw = d.textlength(meta, font=mf)
    mx = W - PAD - mw
    tx = PAD + 2 * r + 26
    tf = R._font(34, "bold")
    topic = R._truncate(d, [e.get("topic") or ""], tf, 1, max(80, mx - tx - 24))[0]
    d.text((tx, cy - 22), topic, font=tf, fill=INK)
    d.text((mx, cy - 17), meta, font=mf, fill=INK)


def render_weekly_cover(entries: list[dict], *, date_label: str,
                        issue_no: int | None = None,
                        out_path: str | Path = "weekly-cover.jpg",
                        quiet: bool = False) -> Path:
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    y = _masthead(d, date_label, issue_no)

    if quiet or not entries:
        y += 90
        big = R._font(76, "bold")
        for ln in R._wrap(d, "No statehouse activity this week.", big, W - 2 * PAD):
            d.text((PAD, y), ln, font=big, fill=INK)
            y += R._line_h(big, 1.08)
        y += 30
        d.text((PAD, y), "We'll be back next week.", font=R._font(42, "regular"), fill=INK)
        _footer(d, "")
        return _save(img, Path(out_path))

    y += 34
    R._draw_tracked(d, PAD, y, "IN THIS ISSUE", R._font(26, "bold"), INK, 5)
    y += 56
    rows = entries[:9]
    bottom = H - 150 - 30
    h = min(96, (bottom - y) // len(rows))
    for i, e in enumerate(rows):
        _row(img, d, y, h, e)
        y += h
        if i < len(rows) - 1:
            d.line([(PAD, y), (W - PAD, y)], fill=FAINT, width=2)
    _footer(d, "Swipe for each bill")
    return _save(img, Path(out_path))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    sample = [
        {"emoji": "\U0001F916", "topic": "AI, Data Centers & Crypto", "accent": (79, 70, 229), "state": "MI", "identifier": "SB 1050"},
        {"emoji": "\U0001F3E0", "topic": "Housing", "accent": (234, 88, 12), "state": "DE", "identifier": "SB 116"},
        {"emoji": "\U0001FA7A", "topic": "Healthcare", "accent": (13, 148, 136), "state": "PR", "identifier": "PC 1450"},
        {"emoji": "\U0001F393", "topic": "Education", "accent": (37, 99, 235), "state": "DC", "identifier": "B26-0123"},
        {"emoji": "\U0001F5F3", "topic": "Elections & Voting Rights", "accent": (190, 18, 60), "state": "US", "identifier": "HR 9368"},
        {"emoji": "\U0001F33F", "topic": "Environment & Climate", "accent": (22, 163, 74), "state": "CA", "identifier": "AB 2559"},
        {"emoji": "⚖️", "topic": "Criminal Justice", "accent": (71, 85, 105), "state": "DE", "identifier": "SB 300"},
        {"emoji": "\U0001F477", "topic": "Labor & Workers' Rights", "accent": (202, 138, 4), "state": "CA", "identifier": "SB 171"},
        {"emoji": "\U0001F3F3️‍\U0001F308", "topic": "LGBTQ+", "accent": (147, 51, 234), "state": "CA", "identifier": "SB 998"},
    ]
    print(render_weekly_cover(sample, date_label="Nov 15–Nov 21, 2026", issue_no=47,
                              out_path=out / "weekly-cover-sample.jpg"))
    print(render_weekly_cover(sample[:3], date_label="Nov 15–Nov 21, 2026", issue_no=47,
                              out_path=out / "weekly-cover-3.jpg"))
    print(render_weekly_cover([], date_label="Nov 15–Nov 21, 2026", issue_no=47,
                              out_path=out / "weekly-cover-quiet.jpg", quiet=True))
