#!/usr/bin/env python3
"""
Daily single-topic Instagram bill card, design v2 (visual-first).

Drop this file into scripts/ next to render_bill_card.py. It reuses that module's
vendored fonts, emoji renderer, wrapping helpers and STATE_FULL_NAME so the look
stays consistent. It does NOT replace render_bill_card.py; the old layout stays
available until the new one is verified end to end.

Layout (1080x1350, Instagram 4:5), approved by the maintainer:
  * Full-bleed topic color background (the topic's card_accent, darkened only as
    much as needed so white text has >= 4.5:1 contrast).
  * Top-left, stacked: a solid white TOPIC pill with BLACK text, and an outlined
    white "STATE . BILL" pill with white bold text (both large, high contrast).
  * Top-right: a big translucent circle holding the topic emoji (hero image).
  * Left column only (about 540px wide): the 2-sentence plain-language
    description, bold white, auto-shrinking so it clears the timeline.
    No separate headline and no subtext.
  * Bottom: 5-stage progress timeline (Introduced, Committee, Passed chamber,
    Governor, Law). The current stage is highlighted and the ACTION DATE sits
    directly under that stage.
  * Footer: GOVBOT wordmark only.

Public entry point:
    render_card_v2(bill, text=..., emoji=..., topic_label=..., accent=...,
                   out_path=...) -> Path
`bill` has the same shape render_bill_card.render_card already receives
(state, identifier, action_desc, action_date).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_bill_card as R  # noqa: E402  (fonts, emoji, wrap, state names)

W, H = 1080, 1350
PAD = 80
COL_W = 540            # text column width; keeps text clear of the hero image
TEXT_BOTTOM = 1040     # text must end above the timeline
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
STAGES = ["Introduced", "Committee", "Passed chamber", "Governor", "Law"]


# --- color helpers -----------------------------------------------------------
def _lum(c):
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (f(x) for x in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _deepen(accent):
    """Darken the topic accent until white text has >= 4.5:1 contrast."""
    c = tuple(accent)
    while _contrast(WHITE, c) < 4.5:
        c = tuple(round(v * 0.93) for v in c)
    return c


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


# --- status -> timeline stage --------------------------------------------------
# The bill's latest action text varies a lot by state ("Comp. became Pub. Ch.
# 1110", "WITHDRAWN APP P.L.2025, c.343.", "Assigned to s/c Civil Justice
# Subcommittee", "Aprobado por el Senado en Votación Final"). Each rule below is
# (stage, name, pattern) and the FIRST match wins, so the order matters:
#   * law and governor rules come first, and vetoes before approvals ("Vetoed …
#     Returned from the Governor without approval");
#   * failed votes and committee "do pass" reports come before the generic
#     "passed" rule;
#   * every pattern is word-bounded ("assigned" must not read as "signed").
# Tuned against ~1,300 distinct actions from samples/ and the topics' bills_raw
# records (scripts/test_render_bill_card_v2.py keeps the tricky ones).
LAW, GOVERNOR, PASSED, COMMITTEE, INTRODUCED = 4, 3, 2, 1, 0

_STAGE_RULES = [
    # Federal bills: the President signs them into law.
    (LAW, "signed by the President", r"^us\|.*(\bsigned by (the )?president\b|\bbecame public law\b)"),
    (GOVERNOR, "sent to the President", r"^us\|.*\bpresented to (the )?president\b"),
    # Massachusetts' "Enacted" is the final legislative vote, before the governor.
    (GOVERNOR, "enacted, sent to governor (MA)", r"^ma\|.*\benacted\b"),
    # A veto that was overridden, or a line-item veto of a signed bill, is law.
    (LAW, "veto override", r"\boverrid\w*|\bline[- ]item veto"),
    # Vetoed (or a veto message / sustained veto): it reached the governor.
    (GOVERNOR, "veto", r"\bveto"),
    (LAW, "became law without signature",
     r"without (the )?(governor'?s )?signature|without the signature of the governor"),
    (LAW, "signed by governor",
     r"\b(signed|approved) by (the )?gov(ernor|\.|\b)|\bgovernor (signed|approved)\b|"
     r"\bsigned into law\b"),
    (LAW, "enacted / chaptered",
     r"\bbec(ame|omes) (law|pub|act)|\benacted\b|\bchaptered\b|"
     r"\bchap(ter)?\.? ?(no\.? ?)?\d+|\bp\. ?l\.|\bpublic (law|act)\b|"
     r"\bpub\. ?ch\b|\|act (no\.? ?)?\d+|\bact no\.? ?\d+|\|ley\b"),
    (LAW, "effective", r"\beffective\b|\beff\. ?\d"),
    (LAW, "filed with secretary of state",
     r"(filed with|delivered to) (the )?secretary of state"),
    (GOVERNOR, "sent to governor",
     r"\bgovern(or|ador|adora)\b|\bgubernatorial\b|\bratified\b|\bgobernador"),
    # Failed floor votes, committee recommendations and votes on amendments are
    # not passage of the bill.
    (COMMITTEE, "failed vote",
     r"failed to pass|refused passage|failed passage|\bdefeated by (the )?(senate|house)"),
    (COMMITTEE, "committee recommendation",
     r"\bdo (not )?pass\b|passage recommended|\bto pass\b|ought to (pass|be adopted)|"
     r"\bbe adopted\b|recommend|inexpedient to legislate|\bfavorabl|"
     r"informe|\bcomisi[oó]n\b|passed to rules|passed (on )?(2nd|second) (reading|consideration)|"
     r"immediate passage"),
    (COMMITTEE, "amendment vote",
     r"\|(amendment|amend|s?sa \d|ss#?\d* for|ha \d|hcs|sca)\b|floor amendment"),
    (PASSED, "died after first chamber", r"\bdied in (returning )?messages\b"),
    (PASSED, "passed",
     r"\bpass(ed|es|age)\b|\bagreed to\b|\bconcurr?|\bin conference\b|"
     r"conference committee|committee of conference|\benroll|\badopted\b|final reading|"
     r"\bsigned (by|in) (the )?(h\.|s\.|house|senate|president|speaker|officers)|"
     r"\b(speaker|president) signed\b|signature requested|ready for sig|"
     r"\bfirmado por\b|\baprobad[oa]\b|aprobaci[oó]n final|votaci[oó]n final|"
     r"received (from|in) (the )?(house|senate|assembly)|message[^.]* (received|from) (the )?(house|senate)|"
     r"returned from (the )?(senate|house|assembly)|\b[hs] bills with [hs] amend|"
     r"\b(house|senate) agreed\b|legis(lative)? resolve|permanently filed|immediate effect|"
     r"(sent|transmitted|delivered) to (the )?(house|senate|assembly)|"
     r"enviado (a|al|a la) (c[aá]mara|senado)|\bsecond house\b"),
    # California: "To Com. on RLS. for assignment" is the desk step right after
    # introduction, before any policy committee.
    (INTRODUCED, "introduced", r"to com\. on rls\.? for assignment"),
    (COMMITTEE, "committee",
     r"\brefer|\bre-refer|\breferid|\bremitid|\bcommit|\bcomm?\b|\bcom\. on (?!rls\b)|"
     r"subcommittee|\bs/c\b|\breport\w*|\bhearing\b|vista p[uú]blica|"
     r"work session|executive session|reuni[oó]n ejecutiva|\bassigned to\b|"
     r"\bdied (in|on)\b|held in committee|under submission|inactive file|"
     r"\bsecond (read|consideration)|\b(2nd|3rd|third) read|\bread (second|third|a third|a second)|"
     r"\b[hs] (second|third) read|\|\d?r?s$|\bcalendar|\bcal\b|orders of the day|"
     r"\btable\b|\bdischarged\b|interim study|\bengross|\bcs by\b|"
     r"\btitle suff\b|\bnoba\b|\bcow\b"),
    (INTRODUCED, "introduced",
     r"introduc|pre-?filed|first read|read (a )?first time|\bread once\b|"
     r"primera lectura|filed with the clerk|held at desk|electronically reproduced|"
     r"print number|from printer"),
]
_STAGE_RES = [(stage, name, re.compile(pat)) for stage, name, pat in _STAGE_RULES]


def classify_stage(status: str, *, other_chamber: bool = False,
                   state: str = "") -> tuple[int, str]:
    """Map the bill's latest action text to (stage index, rule name).

    The rule name is "" when nothing matched and the stage fell back to
    Introduced; the poster logs how often that happens. `state` is the bill's
    two-letter code, for the few wordings that mean different things in
    different places ("Signed by President", Massachusetts' "Enacted").
    `other_chamber` is True when the action happened in the chamber the bill did
    NOT start in: the bill has passed its first chamber even if the action only
    says "Referred to Labor & Industry", so it is at least "Passed chamber"."""
    # Rules see "<state>|<action>", so a pattern can anchor on either.
    s = f"{(state or '').lower()}|" + " ".join((status or "").lower().split())
    for stage, name, rx in _STAGE_RES:
        if stage < PASSED and other_chamber:
            return PASSED, "action in second chamber"
        if rx.search(s):
            return stage, name
    if other_chamber:
        return PASSED, "action in second chamber"
    return INTRODUCED, ""


def stage_index(status: str, *, other_chamber: bool = False, state: str = "") -> int:
    """Best-effort map from the bill's latest action text to a timeline stage.
    Unknown wording falls back to 0 (Introduced). See classify_stage."""
    return classify_stage(status, other_chamber=other_chamber, state=state)[0]


def bill_stage(bill: dict) -> tuple[int, str]:
    """classify_stage for a poster bill dict (state, action_desc, _raw)."""
    return classify_stage(bill.get("action_desc", ""),
                          other_chamber=in_second_chamber(bill),
                          state=bill.get("state") or "")


def _chamber(org: str) -> str:
    m = re.search(r'"classification":\s*"(\w+)"', org or "")
    return m.group(1) if m else ""


def in_second_chamber(bill: dict) -> bool:
    """True when the record says the action happened in the chamber opposite the
    one the bill started in (Open States classifications "upper" / "lower").
    Executive or whole-legislature actions, and records without the fields, are
    False (California aside, below). Reads the verbatim feed record the poster
    attaches as bill["_raw"]."""
    raw = bill.get("_raw") or {}
    start = _chamber(((raw.get("bill") or {}).get("from_organization")) or "")
    acted = _chamber(((((raw.get("log") or {}).get("action")) or {})
                      .get("jurisdiction_id")) or "")
    if start and acted:
        return {start, acted} == {"upper", "lower"}
    # California's actions name the chamber ("In Senate. Read first time.") and
    # its bill numbers name where the bill started (AB/ACR… Assembly, SB/SCR…
    # Senate), which covers its records that lack the classification.
    if (bill.get("state") or "").upper() == "CA":
        ident = (bill.get("identifier") or "").strip().upper()
        action = (bill.get("action_desc") or "").strip().lower()
        return ((ident.startswith("A") and action.startswith("in senate"))
                or (ident.startswith("S") and action.startswith("in assembly")))
    return False


# --- renderer ------------------------------------------------------------------
def render_card_v2(bill: dict, *, text: str, emoji: str, topic_label: str,
                   accent: tuple[int, int, int],
                   out_path: str | Path = "card.png") -> Path:
    bg = _deepen(accent)
    light = _mix(bg, WHITE, 0.14)     # circle behind the emoji
    soft = _mix(bg, WHITE, 0.45)      # inactive timeline line/dots
    dim = _mix(bg, WHITE, 0.90)       # inactive stage labels
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # hero: big circle + emoji, top right (may bleed slightly off the edge)
    cx, cy, rad = 875, 330, 265
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=light)
    emo = R._render_emoji(emoji, 315)
    if emo:
        img.paste(emo, (round(cx - emo.width / 2), round(cy - emo.height / 2)), emo)

    # topic pill (white fill, BLACK text)
    pf = R._font(32, "bold")
    label = (topic_label or "").upper()
    tw = R._tracked_width(d, label, pf, 3)
    pw, ph = round(tw + 60), 74
    d.rounded_rectangle([PAD, PAD, PAD + pw, PAD + ph], radius=ph // 2, fill=WHITE)
    R._draw_tracked(d, PAD + 30, PAD + 17, label, pf, BLACK, 3)

    # state . bill pill (outlined, white text), stacked under the topic pill
    state = (bill.get("state") or "").upper()
    state_name = R.STATE_FULL_NAME.get(state, state)
    meta = f"{state_name} · {bill.get('identifier', '')}".upper()
    mf = R._font(30, "bold")
    mw = R._tracked_width(d, meta, mf, 3)
    my = PAD + ph + 16
    d.rounded_rectangle([PAD, my, PAD + mw + 60, my + ph], radius=ph // 2,
                        outline=WHITE, width=4)
    R._draw_tracked(d, PAD + 30, my + 18, meta, mf, WHITE, 3)

    # description: left column, largest size that clears the timeline
    y = my + ph + 56
    f, lh, lines = None, 0, []
    for size in (52, 48, 44, 40, 36):
        f = R._font(size, "bold")
        lh = R._line_h(f, 1.04)
        lines = R._wrap(d, text or "", f, COL_W)
        if y + len(lines) * lh <= TEXT_BOTTOM:
            break
    lines = R._truncate(d, lines, f, max(1, (TEXT_BOTTOM - y) // lh), COL_W)
    for ln in lines:
        d.text((PAD, y), ln, font=f, fill=WHITE)
        y += lh

    # timeline with the action date under the current stage
    cur = bill_stage(bill)[0]
    ty = 1135
    x0, x1 = PAD + 40, W - PAD - 40
    step = (x1 - x0) / (len(STAGES) - 1)
    d.line([(x0, ty), (x1, ty)], fill=soft, width=8)
    d.line([(x0, ty), (x0 + step * cur, ty)], fill=WHITE, width=8)
    lab = R._font(22, "bold")
    for i, name in enumerate(STAGES):
        x = x0 + step * i
        if i < cur:
            d.ellipse([x - 14, ty - 14, x + 14, ty + 14], fill=WHITE)
        elif i == cur:
            d.ellipse([x - 26, ty - 26, x + 26, ty + 26], fill=WHITE)
            d.ellipse([x - 12, ty - 12, x + 12, ty + 12], fill=bg)
        else:
            d.ellipse([x - 14, ty - 14, x + 14, ty + 14], fill=bg,
                      outline=soft, width=5)
        t = name.upper()
        tw_ = d.textlength(t, font=lab)
        ax = min(max(x - tw_ / 2, PAD - 20), W - PAD - tw_ + 20)
        d.text((ax, ty + 42), t, font=lab, fill=WHITE if i <= cur else dim)
        if i == cur:
            ds = R._format_date(bill.get("action_date", ""))
            if ds:
                df = R._font(26, "medium")
                dw = d.textlength(ds, font=df)
                dx = min(max(x - dw / 2, PAD - 20), W - PAD - dw + 20)
                d.text((dx, ty + 78), ds, font=df, fill=WHITE)

    # footer wordmark
    wf = R._font(34, "bold")
    ww = R._tracked_width(d, "GOVBOT", wf, 7)
    R._draw_tracked(d, W - PAD - ww, H - 72, "GOVBOT", wf, WHITE, 7)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Instagram's content-publishing API documents JPEG as the only supported
    # image format, so the poster asks for a .jpg; PNG stays available for local
    # previews.
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(out_path, "JPEG", quality=92, subsampling=0, optimize=True)
    else:
        img.save(out_path, "PNG")
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "card-v2-sample.png"
    p = render_card_v2(
        {"state": "PA", "identifier": "SB 965",
         "action_desc": "Reported as committed", "action_date": "2026-10-19"},
        text=("Pennsylvania would start a pilot program letting pregnant women "
              "with high blood pressure or blood sugar be monitored from home. "
              "Doctors would get readings from connected devices, starting in 2026."),
        emoji="\U0001FA7A", topic_label="Healthcare", accent=(13, 148, 136),
        out_path=out)
    print("wrote", p)
