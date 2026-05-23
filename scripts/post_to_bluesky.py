#!/usr/bin/env python3
"""
Filter govbot's bills.jsonl for the active topic (transportation by
default), dedupe against the per-topic state file, summarize with a
local LLM (Gemma served by Ollama), and post to Bluesky with rich
link-card embeds.

The topic is selected via the BOT_TOPIC env var and read from
topics/<name>/config.yml. See scripts/topic.py.

Bill links go to each state's official legislature page when we have a
deep-link builder for that state, otherwise to the state legislature
homepage as a fallback.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

from topic import Topic, load_active_topic

ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = ROOT / "bills.jsonl"

TOPIC: Topic = load_active_topic()
STATE_FILE = TOPIC.state_file_path()

POST_LIMIT = int(os.environ.get("POST_LIMIT", "4"))  # how many bluesky posts per run
# Drop bill actions older than this many days so the feed never posts
# year-old news as if it were fresh. Slow topics still have thousands
# of candidates inside this window. Override via env for tuning.
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "150"))
DRY_RUN = os.environ.get("DRY_RUN") == "1"
# og:image fetching is paused by default. Set FETCH_OG_IMAGE=1 to re-enable
# thumbnail scraping from bill-page URLs. When off, posts still get an external
# link card — just without the image.
FETCH_OG_IMAGE = os.environ.get("FETCH_OG_IMAGE", "0") == "1"

BSKY_HANDLE = TOPIC.bluesky_handle()
BSKY_PASSWORD = TOPIC.bluesky_password()

BLUESKY_API = "https://bsky.social/xrpc"

# Local LLM via Ollama. Defaults assume `ollama serve` is running on the
# same host (e.g. installed in the GitHub Actions step before this script runs)
# and the model has been pulled with `ollama pull <LLM_MODEL>`.
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:11434/api/chat")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "180"))

IMG_MAX_DOWNLOAD = 5 * 1024 * 1024
IMG_TARGET_SIZE  = 900 * 1024
IMG_FETCH_TIMEOUT = 10
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC","PR","GU","VI","AS","MP",
}

STATE_FULL_NAME = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri",
    "MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey",
    "NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio",
    "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
    "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont",
    "VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
    "DC":"Washington D.C.","PR":"Puerto Rico",
}

MAX_POST = 290
LINK_PREFIX = "🔗 "

# Titles at or below this length are used as-is in the post head; longer ones
# get rewritten by the local model into a short plain-English headline.
HEADLINE_THRESHOLD = 90
HEADLINE_MAX_LEN = 70


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_bills(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} does not exist. Did `govbot logs` run?", file=sys.stderr)
        return []
    bills = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                bills.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(bills)} records from {path.name}")
    return bills


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

_STATE_TAG_PATTERN = re.compile(r"\bstate:([a-z]{2})\b", re.IGNORECASE)


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def detect_state(record: dict) -> str:
    for s in _walk_strings(record):
        m = _STATE_TAG_PATTERN.search(s)
        if m:
            code = m.group(1).upper()
            if code in US_STATES:
                return code
    return ""


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _looks_like_code_title(title: str) -> bool:
    t = title.strip()
    if not t:
        return True
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return len(t) < 35 and upper_ratio > 0.7


# A leading bill-designation prefix, e.g. "SS#2/SCS/SB 1012 - " or "HB 5 - " —
# alphanumeric codes optionally joined by slashes, the bill number, a dash.
# Used to strip the redundant prefix before summarizing.
_BILL_NUMBER_PREFIX_RE = re.compile(
    r"^\s*[A-Z0-9#]+(?:/[A-Z0-9#]+)*\s+\d+\s*[-–—]\s+"
)
# The multi-part committee/substitute form (at least one slash, e.g.
# "SS#2/SCS/SB 1012 - ") is a strong signal of a Missouri-style record that
# dumps the whole abstract into the title — distinct from a plain "HB 5 - ".
_SUBSTITUTE_PREFIX_RE = re.compile(
    r"^\s*[A-Z0-9#]+(?:/[A-Z0-9#]+)+\s+\d+\s*[-–—]\s+"
)


def _is_blob_title(title: str) -> bool:
    """True when the `title` field is actually a wall of legalese (the whole
    abstract) rather than a real short headline. Some states — Missouri among
    them — dump the entire multi-thousand-character abstract into the title."""
    t = (title or "").strip()
    if not t:
        return False
    if len(t) > 300:
        return True
    if "\r\n" in t or t.count("\n") >= 2:
        return True
    return bool(_SUBSTITUTE_PREFIX_RE.match(t))


def extract_fields(record: dict) -> dict | None:
    bill = record.get("bill") or {}
    log = record.get("log") or {}

    identifier = bill.get("identifier") or record.get("id") or ""
    title = bill.get("title") or ""
    if not identifier or not title:
        return None

    state = detect_state(record)
    session = bill.get("legislative_session") or ""

    abstract = ""
    for a in (bill.get("abstracts") or []):
        text = a.get("abstract", "") if isinstance(a, dict) else (a if isinstance(a, str) else "")
        if text:
            abstract = text
            break

    subjects = bill.get("subject") or []
    subjects_text = " ".join(str(s) for s in subjects) if isinstance(subjects, list) else str(subjects or "")

    action = log.get("action") or {}
    action_desc = action.get("description") or ""
    action_date_raw = action.get("date") or ""
    action_date = action_date_raw[:10] if action_date_raw else ""

    # Fall back to the record-level timestamp ("YYYYMMDDTHHMMSSZ") when the
    # log's action.date is missing. Without a date, format_action_line returns
    # nothing, so the post collapses to "<emoji> <state> <id> — <title>" and
    # multiple date-less records for the same bill all look like the same
    # post.
    if not action_date:
        ts = record.get("timestamp") or ""
        m = re.match(r"^(\d{4})(\d{2})(\d{2})", ts)
        if m:
            action_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # If we still have no date AND no action description, there's nothing
    # actionable to say beyond the bill's static title — skip rather than
    # emit a bare post that's indistinguishable from other date-less updates
    # of the same bill.
    if not action_date and not action_desc:
        return None

    dedup_key = f"{state}|{identifier}|{action_date}|{action_desc[:40]}"
    same_day_key = f"{state}|{identifier}|{action_date}"

    return {
        "state": state,
        "session": session,
        "identifier": identifier,
        "title": title,
        "abstract": abstract,
        "subjects": subjects_text,
        "action_desc": action_desc,
        "action_date": action_date,
        "dedup_key": dedup_key,
        "same_day_key": same_day_key,
    }


_BOILERPLATE_TITLE_RE = re.compile(
    r"^\s*(an act\s+)?(relating to|concerning|regarding|to amend|"
    r"to provide for|to authorize|to require)\b",
    re.IGNORECASE,
)


def best_display_text(b: dict, headline: str = "") -> str:
    title = (b["title"] or "").strip()
    abstract = (b["abstract"] or "").strip()
    if _looks_like_code_title(title) and abstract:
        return abstract
    # Blob titles are walls of legalese — never show them raw in the post
    # head. Use the model headline, falling back to the first clean sentence.
    if _is_blob_title(title):
        return headline or _first_sentence(abstract or title)
    # OR/TX-style boilerplate ("Relating to transportation; prescribing…")
    if abstract and _BOILERPLATE_TITLE_RE.match(title) and len(abstract) < 220:
        return abstract
    # Long, semicolon-laden multi-clause titles — prefer a shorter abstract.
    if abstract and len(title) > 120 and ";" in title and len(abstract) < len(title):
        return abstract
    # A model-rewritten headline replaces a long legalese title outright,
    # since the trim cascade in compose_post would otherwise have to chop the
    # title mid-clause and lose the action line in the process.
    if headline and len(title) > HEADLINE_THRESHOLD:
        return headline
    return title


# ---------------------------------------------------------------------------
# Action + date formatting
# ---------------------------------------------------------------------------

def _format_date(yyyy_mm_dd: str) -> str:
    try:
        d = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d")
    except ValueError:
        return ""
    abbrev = {1:"Jan.", 2:"Feb.", 3:"March", 4:"April", 5:"May", 6:"June",
              7:"July", 8:"Aug.", 9:"Sept.", 10:"Oct.", 11:"Nov.", 12:"Dec."}
    return f"{abbrev[d.month]} {d.day}, {d.year}"


# Some sources (e.g. Rhode Island) prefix action descriptions with their own
# MM/DD/YYYY date, which would otherwise duplicate the formatted date we
# prepend in format_action_line.
_LEADING_DATE_RE = re.compile(
    r"^\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\s*[:\-–—]?\s+"
)


def _strip_leading_date(s: str) -> str:
    return _LEADING_DATE_RE.sub("", s or "", count=1)


# Some sources (e.g. California) append a parenthetical date to the action
# description, e.g. "Do pass. (Ayes 14. Noes 0.) (May 14)." That repeats the
# formatted date we already prepend, so strip it when it matches the action
# date (a parenthetical for a *different* date is kept — it carries info).
_MONTH_PREFIXES = {1:"jan", 2:"feb", 3:"mar", 4:"apr", 5:"may", 6:"jun",
                   7:"jul", 8:"aug", 9:"sep", 10:"oct", 11:"nov", 12:"dec"}
_TRAILING_PAREN_RE = re.compile(r"\(\s*([^()]*?)\s*\)\s*\.?\s*$")


def _strip_trailing_date(s: str, date_yyyy_mm_dd: str) -> str:
    s = s or ""
    try:
        d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d")
    except ValueError:
        return s
    m = _TRAILING_PAREN_RE.search(s)
    if not m:
        return s
    inner = m.group(1).strip().lower().rstrip(".")
    num = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(?:[/-]\d{2,4})?", inner)
    if num and int(num.group(1)) == d.month and int(num.group(2)) == d.day:
        return s[:m.start()].rstrip()
    name = re.fullmatch(r"([a-z]{3,9})\.?\s+(\d{1,2})(?:,?\s*\d{2,4})?", inner)
    if name and name.group(1).startswith(_MONTH_PREFIXES[d.month]) \
            and int(name.group(2)) == d.day:
        return s[:m.start()].rstrip()
    return s


def _smart_case(s: str) -> str:
    s = s.strip().rstrip(".")
    if not s:
        return s
    letters = [c for c in s if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        small = {"a","an","and","of","or","the","to","by","in","on","for","with","at"}
        words = s.lower().split()
        out = []
        for i, w in enumerate(words):
            out.append(w.capitalize() if (i == 0 or w not in small) else w)
        return " ".join(out)
    return s[0].upper() + s[1:] if s[0].isalpha() else s


def format_action_line(action_desc: str, date_yyyy_mm_dd: str) -> str:
    desc = _smart_case(_strip_trailing_date(_strip_leading_date(action_desc), date_yyyy_mm_dd))
    nice_date = _format_date(date_yyyy_mm_dd)
    if desc and nice_date:
        desc_with_period = desc if desc.endswith((".", "!", "?", ".)")) else desc + "."
        return f"{nice_date}: {desc_with_period}"
    return ""


# ---------------------------------------------------------------------------
# OG image fetching
# ---------------------------------------------------------------------------

_OG_IMAGE_PATTERNS = [
    re.compile(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', re.IGNORECASE),
    re.compile(r'<meta\s+name=["\']twitter:image["\']\s+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']twitter:image["\']', re.IGNORECASE),
]


def _extract_og_image_url(html: str, base_url: str) -> str:
    head_only = html[:40000]
    for pat in _OG_IMAGE_PATTERNS:
        m = pat.search(head_only)
        if m:
            url = m.group(1).strip().replace("&amp;", "&")
            return urljoin(base_url, url)
    return ""


def _requests_get_lenient(url, **kwargs):
    try:
        return requests.get(url, **kwargs)
    except requests.exceptions.SSLError:
        print(f"  IMG: SSL verify failed, retrying without verification...")
        kwargs2 = dict(kwargs)
        kwargs2["verify"] = False
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        return requests.get(url, **kwargs2)


def fetch_og_image(page_url: str) -> tuple[bytes, str] | None:
    try:
        page_host = urlparse(page_url).netloc.lower()
        if not page_host:
            return None

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }

        r = _requests_get_lenient(page_url, headers=headers, timeout=IMG_FETCH_TIMEOUT, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        if "html" not in ctype:
            return None

        html_bytes = b""
        for chunk in r.iter_content(chunk_size=8192):
            html_bytes += chunk
            if len(html_bytes) > 500_000:
                break
        try:
            html = html_bytes.decode("utf-8", errors="replace")
        except Exception:
            return None

        img_url = _extract_og_image_url(html, page_url)
        if not img_url:
            return None

        img_host = urlparse(img_url).netloc.lower()
        if img_host and img_host != page_host:
            if img_host.lstrip("www.") != page_host.lstrip("www."):
                print(f"  IMG: ✗ og:image is off-site ({img_host}), skipping")
                return None

        ir = _requests_get_lenient(img_url, headers=headers, timeout=IMG_FETCH_TIMEOUT, stream=True)
        ir.raise_for_status()

        img_bytes = b""
        for chunk in ir.iter_content(chunk_size=16384):
            img_bytes += chunk
            if len(img_bytes) > IMG_MAX_DOWNLOAD:
                print(f"  IMG: ✗ og:image too large (>{IMG_MAX_DOWNLOAD//1024} KB), skipping")
                return None

        mime = ir.headers.get("content-type", "").split(";")[0].strip().lower() or "image/jpeg"
        if not mime.startswith("image/") or "svg" in mime:
            return None

        return (img_bytes, mime)
    except Exception as e:
        print(f"  IMG: ✗ fetch failed: {e}")
        return None


def prepare_image_for_bluesky(img_bytes: bytes, mime: str) -> tuple[bytes, str] | None:
    try:
        from PIL import Image
    except ImportError:
        return (img_bytes, mime) if len(img_bytes) <= IMG_TARGET_SIZE else None

    try:
        im = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        print(f"  IMG: ✗ Pillow could not open the image: {e}")
        return None

    if len(img_bytes) <= IMG_TARGET_SIZE and mime in ("image/jpeg", "image/png", "image/webp"):
        return (img_bytes, mime)

    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGB")

    max_side = 1600
    if max(im.size) > max_side:
        ratio = max_side / max(im.size)
        new_size = (int(im.size[0] * ratio), int(im.size[1] * ratio))
        im = im.resize(new_size, Image.Resampling.LANCZOS)

    for quality in (85, 75, 65, 55, 45):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= IMG_TARGET_SIZE:
            return (data, "image/jpeg")

    return None


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def _clean_summary(text: str) -> str:
    text = (text or "").strip()
    # Small models sometimes wrap output in quotes or markdown code fences.
    if text.startswith("```"):
        text = text.strip("`").strip()
    text = text.strip().strip('"').strip("'").strip()
    # Take only the first sentence/line if the model rambles.
    for sep in ("\n\n", "\n"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    return text


def _is_allcaps_line(line: str) -> bool:
    """A line that is mostly uppercase letters — a section header
    ('INSPECTIONS OF LONG-TERM CARE FACILITIES') or a trailing drafter name
    ('SCOTT SVAGERA'), not prose."""
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.8


def _clean_for_llm(text: str) -> str:
    """Normalize a raw bill abstract/title into prose the model can summarize:
    drop the bill-number prefix, ALL-CAPS section headers, and the trailing
    drafter name, and collapse the source's \\r\\n line breaks."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _BILL_NUMBER_PREFIX_RE.sub("", text.strip(), count=1)
    kept = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # ALL-CAPS lines that don't end in sentence punctuation are section
        # headers or a trailing drafter name — strip them so the model sees
        # continuous prose instead of echoing a header.
        if _is_allcaps_line(line) and not line.endswith((".", "!", "?")):
            continue
        kept.append(line)
    collapsed = " ".join(" ".join(kept).split())
    if collapsed:
        return collapsed
    # Everything looked like a header (rare) — fall back to the raw text so
    # the caller still has something to work with.
    return " ".join(text.split())


def _collect_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a structured bill abstract on its ALL-CAPS section headers.
    Returns (intro_prose, [(section_title, section_body), ...]), preserving
    the order. Returns ("", []) when there are no headers."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _BILL_NUMBER_PREFIX_RE.sub("", text.strip(), count=1)

    intro_lines: list[str] = []
    sections: list[tuple[str, list[str]]] = []

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if _is_allcaps_line(line) and not line.endswith((".", "!", "?")):
            sections.append((line, []))
        elif sections:
            sections[-1][1].append(line)
        else:
            intro_lines.append(line)

    intro = " ".join(" ".join(intro_lines).split())
    out = [(t, " ".join(" ".join(body).split())) for t, body in sections]
    return intro, out


def _omnibus_digest(text: str) -> str:
    """Compact table-of-contents digest for omnibus bills (3+ titled sections).
    Without this, `_clean_for_llm` strips the ALL-CAPS section headers and the
    2000-char window we send to the model is dominated by the first section,
    so the headline ends up naming the whole bill after that one sub-section
    (e.g. an 8-section Missouri real-estate omnibus turning into "Independence
    Nuisance Property Sale Act"). Returns "" when the abstract isn't an
    omnibus."""
    intro, sections = _collect_sections(text)
    # Sections with no body are usually a trailing drafter name caught by the
    # ALL-CAPS rule (e.g. "SCOTT SVAGERA"), not a real topic.
    sections = [(t, b) for t, b in sections if b]
    if len(sections) < 3:
        return ""
    titles = [_smart_case(t) for t, _ in sections]
    head = intro or "This act modifies multiple provisions."
    return head + " Sections covered: " + "; ".join(titles) + "."


_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy compares."""
    return " ".join(_NORM_RE.sub(" ", (s or "").lower()).split())


_LEAD_FILLER_RE = re.compile(
    r"^(aims to|is intended to|seeks to|would|will|shall|is designed to|"
    r"is meant to|attempts to|works to)\s+",
    re.IGNORECASE,
)


_LEAD_ARTICLE_RE = re.compile(r"^(the|an|a)\s+", re.IGNORECASE)


def _strip_title_prefix(summary: str, title: str) -> str:
    """If the summary opens by restating the title, drop that restatement."""
    if not summary or not title:
        return summary
    # Allow the summary to introduce the title with a leading article
    # ("The Artificial Intelligence Bill of Rights aims to...") even when the
    # title itself has no article.
    body = _LEAD_ARTICLE_RE.sub("", summary, count=1)
    skipped = len(summary) - len(body)
    # Strip a leading article from the title too, so a summary and a title
    # that both open with "The" still match (the summary side is already
    # article-free in `body`).
    n_title = _normalize(_LEAD_ARTICLE_RE.sub("", title, count=1))
    if not n_title or len(n_title) < 6:
        return summary
    n_body = _normalize(body)
    if not n_body.startswith(n_title):
        return summary
    # Walk forward through the un-stripped summary until the normalized prefix
    # first covers the title; that's where the restatement ends. (Start from
    # `skipped`, not `skipped + len(title)`: the article-stripped n_title can
    # be shorter than the original title, so the boundary may come earlier.)
    for i in range(skipped, len(summary) + 1):
        if _normalize(summary[skipped:i]).startswith(n_title):
            rest = summary[i:].lstrip(" -—:,.;")
            rest = _LEAD_FILLER_RE.sub("", rest)
            if not rest:
                return summary
            return rest[:1].upper() + rest[1:]
    return summary


_ACT_TERMINATORS = {"act", "bill", "law", "resolution"}
_NAME_CONNECTORS = {"of", "and", "for", "the", "to", "a", "an", "&"}


def _strip_act_name_echo(summary: str, headline: str) -> str:
    """When the summary opens by naming the bill's own act ("The AI
    Non-Sentience and Responsibility Act establishes…") and that name echoes
    the headline shown directly above it, drop the naming clause so the post
    doesn't say the same thing twice. Returns the summary unchanged when there
    is no such echo (e.g. the leading name doesn't overlap the headline)."""
    if not summary or not headline:
        return summary

    body = _LEAD_ARTICLE_RE.sub("", summary, count=1)
    tokens = body.split()
    if len(tokens) < 3:
        return summary

    # Walk the leading run of capitalized words / connectors up to an act
    # terminator ("Act", "Bill", …). Anything else means there is no act name.
    name_words: list[str] = []
    end_idx = -1
    for i, tok in enumerate(tokens):
        bare = tok.strip(",.;:—-").lower()
        if bare in _ACT_TERMINATORS:
            end_idx = i
            break
        if tok[:1].isupper() or bare in _NAME_CONNECTORS:
            if tok[:1].isupper() and bare not in _NAME_CONNECTORS:
                name_words.append(bare)
            continue
        return summary  # a lowercase non-connector word — not an act name
    if end_idx < 1 or not name_words:
        return summary

    # Only strip when the act name genuinely echoes the headline: require two
    # shared (normalized) words so an unrelated act keeps its name.
    head_tokens = set(_normalize(headline).split())
    name_tokens = _normalize(" ".join(name_words)).split()
    shared = sum(1 for t in name_tokens if t in head_tokens)
    if shared < 2:
        return summary

    # Char offset of the text after the terminator token.
    pos = 0
    for tok in tokens[: end_idx + 1]:
        pos = body.index(tok, pos) + len(tok)
    rest = body[pos:].lstrip(" -—:,.;")
    rest = _LEAD_FILLER_RE.sub("", rest)
    if not rest:
        return summary
    return rest[:1].upper() + rest[1:]


def _smart_truncate(text: str, max_len: int) -> str:
    """Truncate to <= max_len, ending at a sentence or word boundary."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    floor = max(1, int(max_len * 0.6))
    for end in (".", "!", "?"):
        idx = cut.rfind(end)
        if idx >= floor:
            return cut[: idx + 1]
    idx = cut.rfind(" ")
    if idx >= floor:
        return cut[:idx].rstrip(",;:- ") + "…"
    return cut.rstrip(",;:- ") + "…"


def _first_sentence(text: str) -> str:
    """First sentence of a cleaned abstract — the non-LLM fallback summary.
    Returns "" when there's no usable prose, so the caller can drop the
    summary block entirely rather than post raw legalese."""
    cleaned = _clean_for_llm(text)
    if not cleaned:
        return ""
    m = re.search(r"[.!?](?:\s|$)", cleaned)
    sentence = cleaned[: m.end()].strip() if m else cleaned
    return _smart_truncate(sentence, 180)


def summarize(b: dict) -> str:
    abstract = (b["abstract"] or "").strip()
    title = b["title"].strip()
    blob = _is_blob_title(title)

    # When the only content is a short real title (common for Iowa, Indiana,
    # etc., which don't ship abstracts in OpenStates data), there's nothing
    # the model can add without restating the title — and asking a small
    # model to do so anyway invites hallucination (e.g. inventing an
    # unrelated state's statutes). Skip summarization and let the title
    # stand alone. Blob titles are the opposite case: the "title" is itself
    # the full abstract, so there's plenty of substance to summarize even
    # when the title and abstract fields are identical.
    if not abstract:
        return ""
    if not blob and abstract.lower() == title.lower():
        return ""

    # Multi-section omnibus bills get a table-of-contents digest so the model
    # sees every topic, not just whatever fits in the 2000-char window.
    clean_abstract = _omnibus_digest(abstract) or _clean_for_llm(abstract)
    if not clean_abstract:
        return ""

    # For blob bills the title is the same wall of legalese as the abstract;
    # feeding it as a "Title:" line just confuses the model, so send only the
    # cleaned description.
    if blob:
        user_prompt = (
            f"Description: {clean_abstract[:2000]}\n\n"
            "Write the one-sentence neutral summary now."
        )
    else:
        user_prompt = (
            f"Title: {title}\n"
            f"Description: {clean_abstract[:2000]}\n\n"
            "Write the one-sentence neutral summary now."
        )

    try:
        r = requests.post(
            LLM_API_URL,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": TOPIC.summary_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.3},
            },
            timeout=LLM_TIMEOUT,
        )
        if not r.ok:
            print(f"  ! LLM {r.status_code}: {r.text[:300]}", file=sys.stderr)
            r.raise_for_status()
        data = r.json()
        # Ollama /api/chat returns {"message": {"content": "..."}, ...}
        # Ollama /api/generate returns {"response": "...", ...}
        text = (data.get("message") or {}).get("content") or data.get("response") or ""
        return _strip_title_prefix(_clean_summary(text), b["title"])
    except Exception as e:
        print(f"  ! summarization failed, using fallback: {e}", file=sys.stderr)
        # A clean first sentence beats raw legalese; "" drops the block.
        return _strip_title_prefix(_first_sentence(abstract), b["title"])


def shorten_title(b: dict) -> str:
    """Ask the local model to rewrite a long legalese title as a short
    plain-English headline. Returns "" when the original title is already
    short enough, when there's no abstract to ground the rewrite, or when
    the model output is unusable. The caller falls back to smart-truncating
    the original title in any of those cases."""
    title = (b["title"] or "").strip()
    abstract = (b["abstract"] or "").strip()
    blob = _is_blob_title(title)
    if len(title) <= HEADLINE_THRESHOLD:
        return ""
    # Without an abstract a normal long title is the only signal — letting a
    # small model paraphrase a title-only record invites hallucinated
    # specifics. Blob titles carry the full abstract inline, so they are
    # always safe (and necessary) to rewrite into a real headline.
    if not blob and (not abstract or _normalize(abstract) == _normalize(title)):
        return ""

    # Blob bills: the title is the same legalese as the abstract. Ground the
    # rewrite on the cleaned body rather than echoing the raw title back.
    # For omnibus bills (3+ titled sections) use a table-of-contents digest
    # so the headline reflects the full bill, not just the first section.
    body = _omnibus_digest(abstract or title) or _clean_for_llm(abstract or title)
    if not body:
        return ""

    system_prompt = TOPIC.headline_system_prompt()
    if blob:
        user_prompt = (
            f"Description: {body[:2000]}\n\n"
            f"Write the headline now (under {HEADLINE_MAX_LEN} characters)."
        )
    else:
        user_prompt = (
            f"Title: {title}\n"
            f"Description: {body[:2000]}\n\n"
            f"Write the headline now (under {HEADLINE_MAX_LEN} characters)."
        )

    try:
        r = requests.post(
            LLM_API_URL,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"num_predict": 60, "temperature": 0.3},
            },
            timeout=LLM_TIMEOUT,
        )
        if not r.ok:
            print(f"  ! LLM headline {r.status_code}: {r.text[:300]}", file=sys.stderr)
            return ""
        data = r.json()
        text = (data.get("message") or {}).get("content") or data.get("response") or ""
    except Exception as e:
        print(f"  ! headline rewrite failed: {e}", file=sys.stderr)
        return ""

    headline = _clean_summary(text).rstrip(".!?,; ")
    if not headline:
        return ""
    # If the model echoed the title (small models often do), bail.
    if _normalize(headline).startswith(_normalize(title)[:60]):
        return ""
    if len(headline) > HEADLINE_MAX_LEN:
        return ""
    return headline


# ---------------------------------------------------------------------------
# Bluesky
# ---------------------------------------------------------------------------

class BlueskyClient:
    def __init__(self, handle: str, password: str):
        self.session = requests.Session()
        r = self.session.post(
            f"{BLUESKY_API}/com.atproto.server.createSession",
            json={"identifier": handle, "password": password},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        self.did = d["did"]
        self.session.headers["Authorization"] = f"Bearer {d['accessJwt']}"

    def upload_blob(self, data: bytes, mime: str) -> dict | None:
        try:
            r = self.session.post(
                f"{BLUESKY_API}/com.atproto.repo.uploadBlob",
                data=data,
                headers={"Content-Type": mime},
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("blob")
        except Exception as e:
            print(f"  - blob upload failed: {e}", file=sys.stderr)
            return None

    def post(self, text: str, link_url: str, embed_title: str, embed_desc: str,
             thumb_blob: dict | None = None,
             reply: dict | None = None) -> dict:
        record = {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if reply:
            # reply = {"root": {"uri":..., "cid":...}, "parent": {"uri":..., "cid":...}}
            record["reply"] = reply
        if link_url:
            external = {"uri": link_url, "title": embed_title[:300], "description": embed_desc[:1000]}
            if thumb_blob:
                external["thumb"] = thumb_blob
            record["embed"] = {"$type": "app.bsky.embed.external", "external": external}
            if link_url in text:
                tb = text.encode("utf-8")
                ub = link_url.encode("utf-8")
                start = tb.find(ub)
                if start >= 0:
                    record["facets"] = [{
                        "index": {"byteStart": start, "byteEnd": start + len(ub)},
                        "features": [{"$type": "app.bsky.richtext.facet#link", "uri": link_url}],
                    }]
        r = self.session.post(
            f"{BLUESKY_API}/com.atproto.repo.createRecord",
            json={"repo": self.did, "collection": "app.bsky.feed.post", "record": record},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Per-state bill URL builders
#
# Each builder takes (session, identifier) -- e.g. ("2025-2026", "HB 4798") --
# and returns a URL that links directly to the bill on the state's official
# legislature website, or None if it can't construct a reliable URL for the
# given inputs. When a builder returns None (or no builder is registered for
# a state) link_for() falls back to STATE_LEGISLATURE_URLS, which lists the
# best entry-point page for every state + DC + PR.
#
# Govbot's `legislative_session` field varies wildly by state. Some examples:
#   IL '104th'       MA '194th'        OH '136'        IN '2026'
#   FL '2026'        MI '2025-2026'    NY '2025'       WI '2025'
#   MO '2025R'       MN '2025s1'       GA '2025_26'    CT '2025'
# The helpers below extract the bits we need.
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(20\d{2}|19\d{2})")


def _first_year(session: str) -> str:
    """Extract the first 4-digit year from a session string (e.g. '2025-2026' -> '2025')."""
    m = _YEAR_RE.search(session or "")
    return m.group(1) if m else ""


def _split_ident(ident: str) -> tuple[str, str]:
    """'HB 1032' -> ('HB', '1032'); 'SCR 1' -> ('SCR', '1'); strips leading zeros from number."""
    m = re.match(r"\s*([A-Za-z]+)\s*0*(\d+)", ident or "")
    if not m:
        return ("", "")
    return (m.group(1).upper(), m.group(2))


def _leading_int(s: str) -> str:
    """'104th' -> '104'; '194' -> '194'; '' -> ''."""
    m = re.match(r"(\d+)", s or "")
    return m.group(1) if m else ""


# ---------- per-state builders --------------------------------------------
# Patterns marked "verified" follow the documented public URL format; patterns
# marked "best-effort" are the most reasonable guess from the state's URL
# scheme and may need adjustment if the state changes its site.

def _b_ca(session, ident):  # verified — leginfo.legislature.ca.gov billNavClient
    # CA's per-bill URL keys off a bill_id of the form
    # <year1><year2>0<TYPE><NUM> for the 2-year session — e.g. SB 1072 in the
    # 2025-2026 session is 202520260SB1072. The trailing 0 before the type is
    # a constant. Sessions always start in an odd calendar year, so if govbot
    # hands us the even year we step back to the session's start year.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    y1 = int(year)
    if y1 % 2 == 0:
        y1 -= 1
    return ("https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml"
            f"?bill_id={y1}{y1 + 1}0{typ}{num}")


def _b_fl(session, ident):  # verified — flsenate.gov serves both chambers
    # Florida special sessions append a letter to the year (Special Session A,
    # B, C, …). The canonical URL is /Session/Bill/<year><letter>/<number>
    # — the letter goes on the year, NOT the bill number. Govbot/OpenStates
    # may carry the letter on the session string ("2026D") or as a trailing
    # letter on the identifier ("SB 2D" / "SB 2-D"); accept either.
    year = _first_year(session)
    if not year:
        return None
    suffix = ""
    m = re.search(r"\d{4}\s*([A-Za-z])\b", session or "")
    if m:
        suffix = m.group(1).upper()
    m = re.match(r"\s*([A-Za-z]+)\s*0*(\d+)\s*-?\s*([A-Za-z]?)\s*$", ident or "")
    if not m:
        return None
    num = m.group(2)
    if not suffix and m.group(3):
        suffix = m.group(3).upper()
    return f"https://flsenate.gov/Session/Bill/{year}{suffix}/{num}"


def _b_in(session, ident):  # verified — iga.in.gov clean URL
    year = _first_year(session)
    typ, num = _split_ident(ident)
    return f"https://iga.in.gov/{year}/bills/{typ.lower()}{num}" if (year and typ and num) else None


def _b_ia(session, ident):  # verified — legis.iowa.gov BillBook
    # Iowa General Assemblies are 2-year terms convening in odd calendar
    # years. GA N spans years (1843 + 2*N) and the next year, so
    # GA = (year - 1843) // 2 works for either year of the biennium:
    # 91st GA = 2025-2026, 90th GA = 2023-2024.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    ga = (int(year) - 1843) // 2
    return ("https://www.legis.iowa.gov/legislation/BillBook"
            f"?ga={ga}&ba={typ}{num}")


def _b_mi(session, ident):  # verified — needs 4-digit zero-padded number
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if year and typ and num:
        return f"https://www.legislature.mi.gov/Bills/Bill?ObjectName={year}-{typ}-{num.zfill(4)}"
    return None


def _b_ny(session, ident):  # verified — nysenate.gov shows both chambers
    year = _first_year(session)
    typ, num = _split_ident(ident)
    return f"https://www.nysenate.gov/legislation/bills/{year}/{typ}{num}" if (year and typ and num) else None


def _b_ma(session, ident):  # verified — uses General Court number (194 = 2025-2026)
    gc = _leading_int(session)
    typ, num = _split_ident(ident)
    return f"https://malegislature.gov/Bills/{gc}/{typ}{num}" if (gc and typ and num) else None


def _b_oh(session, ident):  # verified — uses GA number, identifier lowercase
    ga = _leading_int(session)
    typ, num = _split_ident(ident)
    return f"https://www.legislature.ohio.gov/legislation/{ga}/{typ.lower()}{num}" if (ga and typ and num) else None


# Wisconsin special sessions get a per-session URL slug of the form
# <2-letter month><last digit of year> (May 2026 -> "my6", January 2018 ->
# "jr8"). The month can't be derived from OpenStates' "YYYYSn" session id, so
# known special sessions are listed here, mapping that id to the biennium's
# odd start year and the slug. Add an entry when a new special session occurs.
_WI_SPECIAL_SESSIONS = {
    "2026S1": ("2025", "my6"),  # May 2026 Special Session
}


def _b_wi(session, ident):  # verified — docs.legis.wisconsin.gov
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    s = (session or "").strip().upper()
    # A bare biennium year ("2025") is a regular session; a trailing letter
    # ("2026S1") marks a special session, which lives under a separate slug.
    if re.search(r"\d{4}\s*[A-Z]", s):
        special = _WI_SPECIAL_SESSIONS.get(s)
        if not special:
            return None  # unknown special session — fall back to homepage
        year, code = special
        return f"https://docs.legis.wisconsin.gov/{year}/related/proposals/{code}_{typ.lower()}{num}"
    year = _first_year(s)
    if not year:
        return None
    y = int(year)
    if y % 2 == 0:
        y -= 1  # bienniums are named by their odd start year
    return f"https://docs.legis.wisconsin.gov/{y}/related/proposals/{typ.lower()}{num}"


def _b_nc(session, ident):  # verified — ncleg.gov BillLookUp
    year = _first_year(session)
    typ, num = _split_ident(ident)
    return f"https://www.ncleg.gov/BillLookUp/{year}/{typ}{num}" if (year and typ and num) else None


def _b_nj(session, ident):  # verified — bill-search needs biennium start year
    # Govbot/OpenStates encode NJ's session as the legislature number
    # (e.g. "221" = 221st legislature, 2024-2025) but njleg.state.nj.us
    # URLs use the calendar start year of the biennium. NJ legislature N
    # convenes in calendar year 1582 + 2*N (218th=2018, 221st=2024, 222nd=2026).
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    year = _first_year(session)
    if not year:
        m = re.match(r"\s*(\d{3})\b", session or "")
        if m:
            year = str(1582 + 2 * int(m.group(1)))
    if not year:
        return None
    return f"https://www.njleg.state.nj.us/bill-search/{year}/{typ}{num}"


def _b_ct(session, ident):  # verified — search by year + bill number
    year = _first_year(session)
    _, num = _split_ident(ident)
    if year and num:
        return ("https://www.cga.ct.gov/asp/cgabillstatus/cgabillstatus.asp"
                f"?selBillType=Bill&which_year={year}&bill_num={num}")
    return None


def _b_mo(session, ident):  # best-effort -- senate.mo.gov search; LegiScan fallback
    # Missouri rebuilt both chambers' trackers around opaque internal numeric
    # bill IDs (BillInformation?billid=NNN), so there's no per-bill URL we can
    # compute from the bill number. The senate bill-tracking search resolves a
    # senate bill number to a single-result page for the current session; it
    # doesn't index House bills, so those fall back to LegiScan, which has
    # stable per-bill pages for every MO bill.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    if typ.startswith("S"):
        return ("https://www.senate.mo.gov/Billtracking/bills/billSearch"
                f"?Term={typ}{num}&Submit=Submit&handler=BillSearch")
    year = _first_year(session)
    if year:
        return f"https://legiscan.com/MO/bill/{typ}{num}/{year}"
    return f"https://legiscan.com/MO/bill/{typ}{num}"


def _b_mn(session, ident):  # verified — revisor.mn.gov bills bill.php
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    chamber = "House" if typ.startswith("H") else "Senate"
    # MN's `ssn` param: 0 = regular, 1 = first special, 2 = second special, …
    # Govbot encodes specials as e.g. "2025s1". Without `ssn` the page errors
    # with "Session year and type are required".
    m = re.search(r"s(\d+)", session or "", re.IGNORECASE)
    ssn = m.group(1) if m else "0"
    return (f"https://www.revisor.mn.gov/bills/bill.php"
            f"?b={chamber}&f={typ}{num}&ssn={ssn}&y={year}")


def _b_nm(session, ident):  # best-effort — nmlegis.gov Legislation form
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    chamber = "H" if typ.startswith("H") else "S"
    leg_type = "B"
    if "JR" in typ: leg_type = "JR"
    elif "JM" in typ: leg_type = "JM"
    elif "M" in typ and not typ.startswith("M"): leg_type = "M"
    return (f"https://www.nmlegis.gov/Legislation/Legislation"
            f"?Chamber={chamber}&LegType={leg_type}&LegNo={num}&year={year[-2:]}")


def _b_hi(session, ident):  # best-effort — capitol.hawaii.gov
    year = _first_year(session)
    typ, num = _split_ident(ident)
    return f"https://www.capitol.hawaii.gov/sessions/session{year}/bills/{typ}{num}_.HTM" if (year and typ and num) else None


def _b_ks(session, ident):  # verified — kslegislature.gov biennium URL
    # KS canonical bill URL is /b{YYYY}_{YY}/bills/{type}{num}/ on the .gov
    # domain. The legacy /li/b{biennium}/measures/{type}_{num}/ path that was
    # used here previously now 301s into a 404: the redirect target keeps the
    # `_` between bill type and number, but the new path requires no
    # underscore (e.g. /b2025_26/bills/sb113/, not .../sb_113/).
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    y = int(year)
    if y % 2 == 0:
        y -= 1  # bienniums start in odd years
    next_yy = str(y + 1)[-2:]
    return f"https://www.kslegislature.gov/b{y}_{next_yy}/bills/{typ.lower()}{num}/"


def _b_pa(session, ident):  # verified — legis.state.pa.us cfdocs billInfo form
    # PA identifiers are HB/SB/HR/SR + number. Chamber is the first letter
    # of the prefix, the rest is the bill type (B for bills, R for resolutions).
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    body = typ[0]
    if body not in ("H", "S"):
        return None
    btype = typ[1:] or "B"
    return ("https://www.legis.state.pa.us/cfdocs/billInfo/billInfo.cfm"
            f"?sYear={year}&sInd=0&body={body}&type={btype}&bn={num}")


def _b_ak(session, ident):  # verified — akleg.gov basis/Bill/Detail
    # Alaska's URL uses the calendar year of the session. OpenStates often
    # encodes Alaska sessions as the legislature number ("34" = 2025-2026);
    # the Nth Alaska Legislature convenes in calendar year 1957 + 2*N.
    year = _first_year(session)
    if not year:
        m = re.match(r"\s*(\d{1,2})\b", session or "")
        if m:
            year = str(1957 + 2 * int(m.group(1)))
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    return f"https://www.akleg.gov/basis/Bill/Detail/{year}?Root={typ}{num}"


def _b_or(session, ident):  # verified — olis.oregonlegislature.gov Measures/Overview
    # OLIS session URL component is YYYY{R|S}N — e.g. 2025R1 (regular session)
    # or 2025S1 (1st special session). Fall back to R1 if unspecified.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    m = re.search(r"([RSrs]\d+)", session or "")
    sub = m.group(1).upper() if m else "R1"
    return f"https://olis.oregonlegislature.gov/liz/{year}{sub}/Measures/Overview/{typ}{num}"


def _b_co(session, ident):  # verified — leg.colorado.gov /bills/<typ><yy>[<sess>]-<num>
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    # CO conventions: HB numbers are 4 digits (e.g. HB25-1001); SB and joint /
    # concurrent / simple resolutions are 3 digits (SB25-001, SJR25-006).
    width = 4 if typ == "HB" else 3
    # Extraordinary sessions carry a letter past the year ("2025B" = the 2025
    # special session); CO bakes that letter into the bill slug (sb25b-004),
    # so SB25B-004 and the regular-session SB25-004 are distinct bills. The
    # regular session ("2025A") carries no letter.
    sess = ""
    m = re.search(r"20\d{2}\s*([B-Z])", (session or "").upper())
    if m:
        sess = m.group(1).lower()
    return f"https://leg.colorado.gov/bills/{typ.lower()}{year[-2:]}{sess}-{num.zfill(width)}"


def _b_wa(session, ident):  # verified — app.leg.wa.gov billsummary
    # WA bienniums start in odd years (2025-2026 biennium → Year=2025 in
    # the URL). If govbot hands us an even-year session string we still
    # want the start year, so drop one when needed.
    typ, num = _split_ident(ident)
    year = _first_year(session)
    if not (typ and num and year):
        return None
    y = int(year)
    if y % 2 == 0:
        y -= 1
    return (f"https://app.leg.wa.gov/billsummary"
            f"?BillNumber={num}&Year={y}&Initiative=false")


def _b_tn(session, ident):  # verified — wapp.capitol.tn.gov BillInfo form
    # Tennessee URLs key off the General Assembly number (e.g. 114th GA
    # spans 2025-2026). Govbot/OpenStates may carry the GA directly as a
    # 3-digit session string ("114", "114S1") or as a calendar year; handle
    # both. GA N spans years (2025 + 2*(N-114)) and the next year.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    ga = ""
    # Match 3 leading digits not followed by another digit, so we accept
    # both "114" and "114S1" but don't misread a year like "2025" as GA 202.
    m = re.match(r"\s*(\d{3})(?!\d)", session or "")
    if m:
        ga = m.group(1)
    else:
        year = _first_year(session)
        if year:
            ga = str(114 + (int(year) - 2025) // 2)
    if not ga:
        return None
    return ("https://wapp.capitol.tn.gov/apps/BillInfo/Default.aspx"
            f"?BillNumber={typ}{num.zfill(4)}&GA={ga}")


def _b_wv(session, ident):  # verified — wvlegislature.gov Bill_Status form
    # Regular sessions use sessiontype=RS; specials look like "2026 1X" / "1X" /
    # "FS" in govbot's session string. We pass through whatever code follows
    # the year if present, otherwise default to RS.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    sessiontype = "RS"
    m = re.search(r"(\d+X|FS|ES|\d+S)\b", session or "", re.IGNORECASE)
    if m:
        sessiontype = m.group(1).upper()
    btype = "res" if any(t in typ for t in ("CR", "JR", "R")) and typ != "HB" and typ != "SB" else "bill"
    return ("https://www.wvlegislature.gov/Bill_Status/Bills_history.cfm"
            f"?input={num}&year={year}&sessiontype={sessiontype}&btype={btype}")


def _b_ms(session, ident):  # verified — billstatus.ls.state.ms.us history page
    # Mississippi bill action-history pages live at:
    #   https://billstatus.ls.state.ms.us/<seg>/pdf/history/<TYPE>/<TYPE><NUM4>.xml
    # NUM4 is zero-padded to 4 digits. <seg> is the calendar year for regular
    # sessions ("2026") or year + extraordinary-session code ("20251E" = 2025
    # 1st Extraordinary). The .xml file renders as a styled HTML page in
    # browsers, with the bill's full action log.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    seg = year
    cleaned = (session or "").strip().upper()
    # OpenStates / govbot encode regular sessions as just the year and
    # specials with an alphanumeric suffix ("20251E"). Pass through any
    # year-prefixed compact identifier; otherwise fall back to bare year.
    if re.fullmatch(r"\d{4}[A-Z\d]+", cleaned):
        seg = cleaned
    return (f"https://billstatus.ls.state.ms.us/{seg}/pdf/history/"
            f"{typ}/{typ}{num.zfill(4)}.xml")


def _b_nd(session, ident):  # verified — ndlegis.gov assembly bill-overview page
    # ND organizes bills by Legislative Assembly number; the Nth Assembly
    # convenes in calendar year 1887 + 2N (1st LA = 1889, 69th LA = 2025).
    # The URL is /assembly/<N>-<YYYY>/{regular|special}/bill-overview/bo<num>.html
    # and bill numbers are unique across chambers (HB: 1000-1999,
    # SB: 2000-2999), so the same path serves both. Resolutions use other
    # number ranges and aren't covered here — they fall back to legis.nd.gov.
    typ, num = _split_ident(ident)
    if typ not in ("HB", "SB") or not num:
        return None
    raw = session or ""
    year = _first_year(raw)
    if not year:
        # OpenStates / govbot sometimes encode ND sessions as just the
        # legislative assembly number ("69", "69th", "69X1") rather than a
        # calendar year. Decode it to the biennium start year
        # (LA N -> 1887 + 2N).
        m = re.match(r"\s*(\d{1,3})", raw)
        if m:
            year = str(1887 + 2 * int(m.group(1)))
    if not year:
        return None
    y = int(year)
    if y % 2 == 0:
        y -= 1  # bienniums start in odd years
    assembly = (y - 1887) // 2
    # Special sessions live under /special/ rather than /regular/. Govbot
    # marks them inconsistently — sometimes by the words "special" or
    # "extraordinary", sometimes by a trailing "X"/"S" code on the assembly
    # or year ("69X1", "69s1", "2025S1").
    is_special = bool(re.search(r"(?i)special|extra|\d[xs]", raw))
    sub = "special" if is_special else "regular"
    return (f"https://www.ndlegis.gov/assembly/{assembly}-{y}/{sub}/"
            f"bill-overview/bo{num}.html")


def _b_de(session, ident):  # best-effort -- LegiScan fallback
    # Delaware's official site (legis.delaware.gov) keys per-bill pages off
    # opaque internal LegislationIds that aren't exposed in OpenStates data,
    # and the AllLegislation browser does its filtering client-side with no
    # query-string entry point we can construct. LegiScan has stable per-bill
    # pages for every DE bill, so use it as the canonical deep link rather
    # than dropping readers on the homepage.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    year = _first_year(session)
    if year:
        return f"https://legiscan.com/DE/bill/{typ}{num}/{year}"
    return f"https://legiscan.com/DE/bill/{typ}{num}"


def _b_me(session, ident):  # best-effort -- LegiScan fallback
    # Maine bills are Legislative Documents (LD). The official tracker at
    # legislature.maine.gov uses a hash-fragment SPA URL that isn't a stable
    # server-side route, and the LawMakerWeb summary pages key off paper
    # numbers (HP/SP) we don't have in OpenStates data. LegiScan resolves
    # LD numbers cleanly, so use it as the deep link.
    typ, num = _split_ident(ident)
    if typ != "LD" or not num:
        return None
    year = _first_year(session)
    if year:
        return f"https://legiscan.com/ME/bill/{typ}{num}/{year}"
    return f"https://legiscan.com/ME/bill/{typ}{num}"


def _b_al(session, ident):  # best-effort — alison.legislature.state.al.us PDF
    # Alabama redesigned its Alison site in 2025 around opaque internal bill
    # IDs that OpenStates no longer captures (instrumentUrl was dropped from
    # the GraphQL API on 2025-01-20). The next-best stable per-bill URL is
    # the introduced-text PDF, served from a predictable path:
    #   /files/pdf/SearchableInstruments/<SESSION>/<TYPE><NUM>-int.pdf
    # SESSION = year + session-type code: "2026RS" (Regular), "2026FS" (First
    # Special), "2025SS1" (1st Special), etc. OpenStates encodes these in
    # lowercase ("2026rs"); upper-case for the URL.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    code = "RS"
    cleaned = (session or "").strip().upper()
    m = re.fullmatch(r"\d{4}([A-Z]+\d?)", cleaned)
    if m:
        code = m.group(1)
    return ("https://alison.legislature.state.al.us/files/pdf/SearchableInstruments/"
            f"{year}{code}/{typ}{num}-int.pdf")


def _b_ne(session, ident):  # verified — nebraskalegislature.gov FloorDocs PDF
    # Nebraska's bill viewer (cv/view_bill.php) keys off opaque DocumentIDs
    # that aren't in OpenStates data, and the search-by-number form is
    # POST-only with a CSRF token, so neither is linkable. The introduced-
    # bill PDF is the only stable, computable URL on nebraskalegislature.gov:
    #   /FloorDocs/<LegN>/PDF/Intro/<TYP><NUM>.pdf
    # OpenStates encodes NE sessions as the legislature number (e.g. "109"
    # for the 109th = 2025-2026 biennium); accept that or a calendar year.
    # Nth NE legislature convenes in calendar year 2*N + 1807 (109 -> 2025).
    typ, num = _split_ident(ident)
    if typ not in ("LB", "LR") or not num:
        return None
    leg = ""
    year = _first_year(session)
    if year:
        y = int(year)
        if y % 2 == 0:
            y -= 1
        leg = str((y - 1807) // 2)
    else:
        m = re.match(r"\s*(\d{2,3})(?:st|nd|rd|th)?\b", session or "")
        if m:
            leg = m.group(1)
    if not leg:
        return None
    return f"https://nebraskalegislature.gov/FloorDocs/{leg}/PDF/Intro/{typ}{num}.pdf"


def _b_nh(session, ident):  # verified — gc.nh.gov results.aspx renders bill inline
    # NH's billinfo.aspx pages key off opaque internal IDs we can't compute,
    # but results.aspx with adv=2 + txtbillno + txtsessionyear renders the
    # bill row inline (number, title, status) and links to the internal-ID
    # page. Falls back to LegiScan if we don't have a year — keeps readers
    # off the bare gencourt homepage.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    year = _first_year(session)
    if year:
        return ("https://gc.nh.gov/bill_status/results.aspx"
                f"?adv=2&txtbillno={typ}{num}&txtsessionyear={year}")
    return f"https://legiscan.com/NH/bill/{typ}{num}"


def _b_ri(session, ident):  # verified — webserver.rilegislature.gov BillText
    # RI's per-bill landing page is the bill text view, served at:
    #   https://webserver.rilegislature.gov/BillText{YY}/{Chamber}Text{YY}/{TYP}{NUM}.htm
    # YY = 2-digit calendar year, Chamber = "House"/"Senate", TYP = "H"/"S".
    # Page shows title, sponsors, intro date, committee referral, and full
    # bill text. There's no public per-bill status URL with an action log;
    # this is the canonical detail page on rilegislature.gov.
    #
    # Only HB/SB are handled — the URL scheme for HR/SR/HJR/SJR resolutions
    # on RI's site is undocumented; let those fall back to the homepage
    # rather than 404 readers into a broken link.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    if typ not in ("HB", "SB"):
        return None
    chamber = "House" if typ == "HB" else "Senate"
    body = typ[0]
    yy = year[-2:]
    return (f"https://webserver.rilegislature.gov/BillText{yy}/"
            f"{chamber}Text{yy}/{body}{num}.htm")


def _b_sc(session, ident):  # verified — scstatehouse.gov sess<GA>_<Y1>-<Y2> path
    # SC General Assemblies are 2-year terms convening in odd calendar years.
    # GA N spans years (1773 + 2*N) and the next year, so
    # GA = (year - 1773) // 2 works for either year of the biennium:
    # 126th GA = 2025-2026, 125th GA = 2023-2024. OpenStates encodes SC
    # sessions as either the calendar-year range ("2025-2026") or the GA
    # number ("126"); accept either. Bill numbers are unique across chambers
    # (House: 3000-4999, Senate: 1-2999) and the same /bills/<num>.htm path
    # serves both. Resolutions use other ranges/paths and fall back to the
    # homepage.
    typ, num = _split_ident(ident)
    if typ not in ("H", "S", "HB", "SB") or not num:
        return None
    year = _first_year(session)
    if year:
        y = int(year)
        if y % 2 == 0:
            y -= 1  # bienniums start in odd years
        ga = (y - 1773) // 2
    else:
        m = re.match(r"\s*(\d{2,3})(?:st|nd|rd|th)?\b", session or "")
        if not m:
            return None
        ga = int(m.group(1))
        y = 1773 + 2 * ga
    return f"https://www.scstatehouse.gov/sess{ga}_{y}-{y + 1}/bills/{num}.htm"


def _b_md(session, ident):  # verified — mgaleg.maryland.gov Legislation/Details
    # Maryland bill detail pages live at:
    #   https://mgaleg.maryland.gov/mgawebsite/Legislation/Details/<type><num4>?ys=<SESSION>
    # NUM4 is zero-padded to 4 digits; the type is lowercase. SESSION is
    # uppercased year + session code: "2025RS" (Regular), "2025S1" (1st
    # Special), etc. OpenStates encodes sessions in lowercase ("2025rs");
    # upper-case for the URL and default to <year>RS if no explicit code.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    sess = (session or "").strip().upper()
    if not re.fullmatch(r"\d{4}[A-Z]+\d?", sess):
        sess = f"{year}RS"
    return ("https://mgaleg.maryland.gov/mgawebsite/Legislation/Details/"
            f"{typ.lower()}{num.zfill(4)}?ys={sess}")


def _b_id(session, ident):  # verified — legislature.idaho.gov sessioninfo
    # Idaho bills, resolutions, concurrent resolutions, and joint memorials
    # all live at:
    #   https://legislature.idaho.gov/sessioninfo/<year>/legislation/<TYPE><NUMN>/
    # The type is upper-case. Bills (H, S) zero-pad the number to 4 digits;
    # resolutions and memorials (HR, SR, HCR, SCR, HJR, SJR, HJM, SJM, HP)
    # zero-pad to 3.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    width = 4 if typ in ("H", "S") else 3
    return ("https://legislature.idaho.gov/sessioninfo/"
            f"{year}/legislation/{typ}{num.zfill(width)}/")


def _b_ga(session, ident):  # verified — legis.ga.gov legacy display path 302s to bill page
    # Georgia's modern bill page is keyed off opaque numeric IDs
    # (legis.ga.gov/legislation/<id>) not exposed in OpenStates data, but the
    # legacy display path is still a stable entry point that redirects to the
    # per-bill page:
    #   https://www.legis.ga.gov/Legislation/en-US/display/<biennium>/<TYP>/<NUM>
    # <biennium> = start + end year concatenated ("20252026"). GA General
    # Assemblies convene in odd calendar years, so if the session string
    # carries only the even year (govbot encodes GA as e.g. '2025_26'), roll
    # back to the biennium start year.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    year = _first_year(session)
    if not year:
        return None
    y = int(year)
    if y % 2 == 0:
        y -= 1  # bienniums start in odd years
    biennium = f"{y}{y + 1}"
    return ("https://www.legis.ga.gov/Legislation/en-US/display/"
            f"{biennium}/{typ}/{num}")


def _b_wy(session, ident):  # verified — wyoleg.gov Legislation/<year>/<TYP><NUM4>
    # Wyoming sessions are annual (General Session in odd years, Budget
    # Session in even years); each year is its own session. The official
    # per-bill page lives at:
    #   https://www.wyoleg.gov/Legislation/<year>/<TYP><NUM4>
    # NUM4 is zero-padded to 4 digits. Wyoming uses "SF" (Senate File), not
    # "SB"; HB, HJ, SJ, HR, SR all follow the same path.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    return f"https://www.wyoleg.gov/Legislation/{year}/{typ}{num.zfill(4)}"


def _b_ar(session, ident):  # verified — arkleg.state.ar.us Bills/Detail?id=...&ddBienniumSession=...
    # Arkansas bill detail URLs are
    #   /Bills/Detail?id=<TYPE><NUM>&ddBienniumSession=<B>%2F<YEAR><CODE>
    # <B> is the odd start year of the biennium (2025-2026 -> 2025).
    # <CODE> is R (regular, odd year), F (fiscal, even year), or EX<n>
    # (extraordinary session #n). OpenStates/govbot encode AR sessions as
    # the year for regulars ("2025"), year+F for fiscals ("2024F",
    # "2026F"), and year+S<n> or year+ES<n> for specials ("2023S1",
    # "2023ES1"). The slash between biennium and session is URL-encoded.
    typ, num = _split_ident(ident)
    year = _first_year(session)
    if not (typ and num and year):
        return None
    y = int(year)
    biennium = y if y % 2 == 1 else y - 1  # bienniums start in odd years
    cleaned = (session or "").strip().upper()
    # Match EX1 / ES1 / S1 (extraordinary sessions) before falling back to
    # F / R / parity default. The trailing \d+ keeps plain "2025" from
    # being misread as a special session.
    m = re.search(r"(?:EX|ES|S)(\d+)", cleaned)
    if m:
        code = f"EX{m.group(1)}"
    elif "F" in cleaned:
        code = "F"
    elif "R" in cleaned:
        code = "R"
    else:
        code = "R" if y % 2 == 1 else "F"
    return ("https://www.arkleg.state.ar.us/Bills/Detail"
            f"?id={typ}{num}&ddBienniumSession={biennium}%2F{year}{code}")


def _b_vt(session, ident):  # verified — legislature.vermont.gov bill status page
    # Vermont organizes bills by biennium and addresses each biennium in URLs
    # by its second (even) calendar year — the 2025-2026 biennium is "2026".
    # Bill identifiers use a dotted form on the site: S.44, H.123.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    years = _YEAR_RE.findall(session or "")
    if not years:
        return None
    y = int(years[-1])
    if y % 2:  # odd -> first year of the biennium; the URL uses the even year
        y += 1
    return f"https://legislature.vermont.gov/bill/status/{y}/{typ}.{num}"


def _b_il(session, ident):  # best-effort -- LegiScan fallback
    # Illinois' redesigned ilga.gov keys per-bill pages off opaque internal
    # LegIDs that aren't exposed in OpenStates data, and govbot carries the
    # session only as a General Assembly ordinal ("104th"), not a year.
    # LegiScan has stable per-bill pages, so use it as the deep link rather
    # than dropping readers on the ilga.gov homepage. The Nth General
    # Assembly opens the biennium starting 2025 + 2*(N - 104) (104th -> 2025).
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    ga = _leading_int(session)
    if ga:
        year = 2025 + 2 * (int(ga) - 104)
        return f"https://legiscan.com/IL/bill/{typ}{num}/{year}"
    return f"https://legiscan.com/IL/bill/{typ}{num}"


def _b_ky(session, ident):  # verified — apps.legislature.ky.gov record page
    # KY session strings are "<year><code>", e.g. "2025RS" (Regular Session)
    # or "2025SS" (Special Session). The record URL keys off the 2-digit year
    # plus the lowercased code: 2025RS -> 25rs, identifier lowercase.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    m = re.search(r"[A-Za-z]+", session or "")
    code = (m.group(0) if m else "RS").lower()
    return f"https://apps.legislature.ky.gov/record/{year[-2:]}{code}/{typ.lower()}{num}.html"


def _b_ok(session, ident):  # verified — oklegislature.gov BillInfo
    # OK's Session param is a 4-digit code: 2-digit year + 2-digit session
    # number, "00" for the regular session (2025 -> 2500). OpenStates marks
    # extraordinary sessions with a trailing letter ("2017A" -> 1701).
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    m = re.search(r"\d{4}\s*([A-Za-z])", session or "")
    sess_no = f"{ord(m.group(1).upper()) - 64:02d}" if m else "00"
    return ("https://www.oklegislature.gov/BillInfo.aspx"
            f"?Bill={typ}{num}&Session={year[-2:]}{sess_no}")


STATE_BILL_URL_BUILDERS = {
    "CA": _b_ca,
    "FL": _b_fl, "IN": _b_in, "IA": _b_ia, "MI": _b_mi, "NY": _b_ny,
    "MA": _b_ma, "OH": _b_oh, "WI": _b_wi, "NC": _b_nc, "NJ": _b_nj,
    "CT": _b_ct, "MO": _b_mo, "MN": _b_mn, "NM": _b_nm, "HI": _b_hi,
    "KS": _b_ks, "WV": _b_wv, "PA": _b_pa, "AK": _b_ak, "OR": _b_or,
    "CO": _b_co, "WA": _b_wa, "TN": _b_tn, "RI": _b_ri, "MS": _b_ms,
    "AL": _b_al, "ND": _b_nd, "NH": _b_nh, "DE": _b_de, "ME": _b_me,
    "NE": _b_ne, "SC": _b_sc, "MD": _b_md, "ID": _b_id, "GA": _b_ga,
    "WY": _b_wy, "AR": _b_ar, "VT": _b_vt, "IL": _b_il,
    "KY": _b_ky, "OK": _b_ok,
}


# Generic state-legislature entry pages used when no deep-link is available.
# These are stable canonical URLs that get the reader to the right site even
# when we can't compute the per-bill URL.
STATE_LEGISLATURE_URLS = {
    "AL": "https://alison.legislature.state.al.us/",
    "AK": "https://www.akleg.gov/",
    "AZ": "https://www.azleg.gov/",
    "AR": "https://www.arkleg.state.ar.us/",
    "CA": "https://leginfo.legislature.ca.gov/",
    "CO": "https://leg.colorado.gov/",
    "CT": "https://www.cga.ct.gov/",
    "DE": "https://legis.delaware.gov/",
    "FL": "https://www.flsenate.gov/",
    "GA": "https://www.legis.ga.gov/",
    "HI": "https://www.capitol.hawaii.gov/",
    "ID": "https://legislature.idaho.gov/",
    "IL": "https://www.ilga.gov/",
    "IN": "https://iga.in.gov/",
    "IA": "https://www.legis.iowa.gov/",
    "KS": "https://www.kslegislature.org/",
    "KY": "https://legislature.ky.gov/",
    "LA": "https://www.legis.la.gov/",
    "ME": "https://legislature.maine.gov/",
    "MD": "https://mgaleg.maryland.gov/",
    "MA": "https://malegislature.gov/",
    "MI": "https://www.legislature.mi.gov/",
    "MN": "https://www.leg.mn.gov/",
    "MS": "https://www.legislature.ms.gov/",
    "MO": "https://www.senate.mo.gov/",
    "MT": "https://leg.mt.gov/",
    "NE": "https://nebraskalegislature.gov/",
    "NV": "https://www.leg.state.nv.us/",
    "NH": "https://www.gencourt.state.nh.us/",
    "NJ": "https://www.njleg.state.nj.us/",
    "NM": "https://www.nmlegis.gov/",
    "NY": "https://www.nysenate.gov/",
    "NC": "https://www.ncleg.gov/",
    "ND": "https://www.legis.nd.gov/",
    "OH": "https://www.legislature.ohio.gov/",
    "OK": "https://www.oklegislature.gov/",
    "OR": "https://olis.oregonlegislature.gov/",
    "PA": "https://www.legis.state.pa.us/",
    "RI": "https://www.rilegislature.gov/",
    "SC": "https://www.scstatehouse.gov/",
    "SD": "https://sdlegislature.gov/",
    "TN": "https://www.capitol.tn.gov/legislation/",
    "TX": "https://capitol.texas.gov/",
    "UT": "https://le.utah.gov/",
    "VT": "https://legislature.vermont.gov/",
    "VA": "https://lis.virginia.gov/",
    "WA": "https://leg.wa.gov/",
    "WV": "https://www.wvlegislature.gov/",
    "WI": "https://docs.legis.wisconsin.gov/",
    "WY": "https://www.wyoleg.gov/",
    "DC": "https://lims.dccouncil.gov/",
    "PR": "https://www.oslpr.org/",
}


def link_for(b: dict) -> str:
    """
    Build the best available URL for a bill. Tries the per-state deep-link
    builder first, then falls back to the state's legislature homepage.
    Returns "" only if the state code is unknown.
    """
    state = (b.get("state") or "").upper()
    session = b.get("session", "")
    identifier = b.get("identifier", "")
    if not state:
        return ""

    builder = STATE_BILL_URL_BUILDERS.get(state)
    if builder:
        try:
            url = builder(session, identifier)
        except Exception:
            url = None
        if url:
            return url

    return STATE_LEGISLATURE_URLS.get(state, "")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_post(b: dict, summary: str, headline: str = "") -> tuple[str, str, str, str]:
    emoji = TOPIC.emoji_for(b)
    link = link_for(b)
    link_block = f"\n\n{LINK_PREFIX}{link}" if link else ""

    state_label = b["state"] or "?"
    display = best_display_text(b, headline=headline).strip()
    summary = (summary or "").strip()
    # Drop a leading act name from the summary when it just echoes the headline
    # ("AI Non-Sentience Act…" appearing in both lines).
    summary = _strip_act_name_echo(summary, display)

    summary_block = (
        f"\n\n{summary}"
        if summary and _normalize(summary) != _normalize(display)
        else ""
    )
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block = f"\n\n{action_line}" if action_line else ""

    prefix_len = len(emoji) + len(f" {state_label} {b['identifier']} — ")
    head = f"{emoji} {state_label} {b['identifier']} — {display}"

    def assemble(h, s, a, l):
        return h + s + a + l

    text = assemble(head, summary_block, action_block, link_block)

    # Trim order: summary → title in head → action description. Date+action
    # is the news; it's preserved over a long title or a long body summary.
    if len(text) > MAX_POST and summary_block:
        overflow = len(text) - MAX_POST
        new_len = max(0, len(summary) - overflow - 1)
        if new_len > 20:
            summary = _smart_truncate(summary, new_len + 1)
            summary_block = f"\n\n{summary}"
        else:
            summary_block = ""
        text = assemble(head, summary_block, action_block, link_block)

    if len(text) > MAX_POST:
        avail = MAX_POST - len(link_block) - len(summary_block) - len(action_block) \
                - prefix_len - 1
        if avail > 0:
            display_trimmed = _smart_truncate(display, avail + 1)
        else:
            display_trimmed = ""
        head = f"{emoji} {state_label} {b['identifier']} — {display_trimmed}".rstrip(" —")
        text = assemble(head, summary_block, action_block, link_block)

    # Only reached when the action description itself is so long it can't fit
    # even with display fully trimmed. Falls back to the old date+desc trim.
    if len(text) > MAX_POST and action_block and action_line:
        nice_date = _format_date(b["action_date"])
        if nice_date:
            date_prefix = f"{nice_date}: "
            if action_line.startswith(date_prefix):
                desc_part = action_line[len(date_prefix):].rstrip(".!?")
                overflow = len(text) - MAX_POST
                new_len = max(0, len(desc_part) - overflow - 1)
                if new_len > 8:
                    action_line = date_prefix + _smart_truncate(desc_part, new_len + 1)
                    action_block = f"\n\n{action_line}"
                else:
                    action_line = ""
                    action_block = ""
            else:
                action_block = f"\n\n{action_line}"
        text = assemble(head, summary_block, action_block, link_block)

    state_name = STATE_FULL_NAME.get(b["state"], b["state"] or "Bill")
    embed_title = f"{state_name} {b['identifier']}"[:300]
    embed_desc = (summary or _clean_for_llm(b["abstract"]) or display)[:280]
    return text, link, embed_title, embed_desc


# ---------------------------------------------------------------------------
# State persistence
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


_FILENAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, max_len: int = 40) -> str:
    """Lowercase, collapse non-alphanumerics into underscores, cap length."""
    s = _FILENAME_UNSAFE_RE.sub("_", (text or "").strip().lower()).strip("_")
    return s[:max_len].rstrip("_")


def save_raw_record(b: dict, out_dir: Path | None = None) -> None:
    """Write the verbatim bills.jsonl record for a posted bill to
    topics/<name>/bills_raw/<STATE>-<id>-<date>-<action_slug>.json so
    every posted action has a self-contained raw artifact alongside the
    dedup key in bills_used.json. Pass ``out_dir`` to redirect the file
    elsewhere (e.g. the weekly digest's own raw-record folder). One file
    per posted action, kept forever — pruning is a manual repo-hygiene
    decision."""
    raw = b.get("_raw")
    if not raw:
        return
    state = (b.get("state") or "XX")
    # Identifier keeps original case (HB2763, SR 008 → HB2763, SR_008) so
    # the filename matches how the bill is shown in the post.
    ident_raw = (b.get("identifier") or "unknown").strip()
    ident = _FILENAME_UNSAFE_RE.sub("_", ident_raw).strip("_")[:24] or "unknown"
    date = b.get("action_date") or "no-date"
    action_slug = _slug(b.get("action_desc") or "no-action", max_len=40) or "no-action"
    fname = f"{state}-{ident}-{date}-{action_slug}.json"
    if out_dir is None:
        out_dir = TOPIC.bills_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not DRY_RUN and (not BSKY_HANDLE or not BSKY_PASSWORD):
        print(f"ERROR: {TOPIC.bluesky_handle_env()} and "
              f"{TOPIC.bluesky_password_env()} must be set.", file=sys.stderr)
        return 1

    records = load_bills(JSONL_PATH)
    if not records:
        return 0

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
        if b["dedup_key"] in seen:
            continue
        # Stash the source record so save_raw_record() can dump the verbatim
        # bills.jsonl line for any bill we end up posting.
        b["_raw"] = r
        candidates.append(b)

    # Freshness gate: a state's newest *unposted* match can genuinely be a
    # year-old action (part-time legislatures, niche topics). Posting that as
    # news is misleading, so drop anything past the age cap. Mirrors
    # weekly_digest.in_lookback_window.
    cutoff = datetime.now(timezone.utc).date()

    def _fresh(b: dict) -> bool:
        try:
            d = datetime.strptime(b["action_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return False  # undated candidate -> can't confirm freshness, drop
        return (cutoff - d).days <= MAX_ACTION_AGE_DAYS

    before = len(candidates)
    candidates = [b for b in candidates if _fresh(b)]
    dropped = before - len(candidates)
    if dropped:
        print(f"  dropped {dropped} stale update(s) older than "
              f"{MAX_ACTION_AGE_DAYS} days.")

    # Same-day dedup (collapse multiple log entries for same bill on same day).
    unique_by_day: dict[str, dict] = {}
    for b in candidates:
        existing = unique_by_day.get(b["same_day_key"])
        if existing is None or len(b["action_desc"]) > len(existing["action_desc"]):
            unique_by_day[b["same_day_key"]] = b
    candidates = list(unique_by_day.values())

    print(f"Found {len(candidates)} new {TOPIC.topic_phrase} bill update(s).")
    if not candidates:
        return 0

    # Print a state-distribution summary so we can see coverage.
    from collections import Counter
    state_counts = Counter(b["state"] or "?" for b in candidates)
    top = state_counts.most_common(15)
    print(f"  by state: {', '.join(f'{s}={n}' for s,n in top)}")

    # Selection: keep each state's single most-recent bill, then run a
    # weighted random draw across those per-state representatives. Recency
    # only decides which bill represents a state — it does NOT decide which
    # states win the run. The draw is weighted toward states we haven't
    # posted recently (tracked in state["state_last_posted"]), so coverage
    # rotates across all states over time instead of the freshest states
    # monopolizing every run.
    def recency(b: dict) -> datetime:
        try:
            return datetime.strptime(b["action_date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            return datetime.min

    def has_desc(b: dict) -> bool:
        # Stub records (real action_date but empty action_desc) produce a
        # post with no body action line, so the reader sees only the title
        # with no indication of what just happened. Keep them out of the
        # draw unless there aren't enough descriptive bills to fill the run.
        return bool((b["action_desc"] or "").strip())

    # One representative per state: prefer a descriptive bill over a stub,
    # then the most recent. (b["state"] is "" for unknown — bucket as "?".)
    by_state: dict[str, dict] = {}
    for b in candidates:
        st = b["state"] or "?"
        cur = by_state.get(st)
        if cur is None or (has_desc(b), recency(b)) > (has_desc(cur), recency(cur)):
            by_state[st] = b
    reps = list(by_state.values())

    descriptive = [b for b in reps if has_desc(b)]
    stubs = [b for b in reps if not has_desc(b)]

    # Weight each state by how long since we last posted it: never-posted
    # states get the max weight, recently-posted states get the least. The
    # 180-day cap keeps one ancient state from dwarfing every other.
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
        to_post.extend(weighted_draw(stubs, POST_LIMIT - len(to_post)))

    distinct_states = len({b["state"] or "?" for b in to_post})
    print(f"Pool: {len(descriptive)} state(s) with descriptive bills, "
          f"{len(stubs)} stub-only.")
    print(f"Will post up to {POST_LIMIT}: posting {len(to_post)} from {distinct_states} state(s).")

    client = None if DRY_RUN else BlueskyClient(BSKY_HANDLE, BSKY_PASSWORD)

    for b in to_post:
        summary = summarize(b)
        headline = shorten_title(b)
        text, link, ec_title, ec_desc = compose_post(b, summary, headline=headline)

        thumb_blob = None
        if link and FETCH_OG_IMAGE:
            print(f"  IMG: fetching og:image for {link}")
            fetched = fetch_og_image(link)
            if fetched:
                img_bytes_raw, mime_raw = fetched
                print(f"  IMG: downloaded {len(img_bytes_raw)//1024} KB ({mime_raw})")
                prepared = prepare_image_for_bluesky(img_bytes_raw, mime_raw)
                if prepared:
                    img_bytes, img_mime = prepared
                    if client:
                        thumb_blob = client.upload_blob(img_bytes, img_mime)
                        if thumb_blob:
                            print(f"  IMG: ✓ attached ({len(img_bytes)//1024} KB, {img_mime})")
                        else:
                            print(f"  IMG: ✗ blob upload failed")
                    else:
                        print(f"  IMG: [dry-run] would attach ({len(img_bytes)//1024} KB)")
                else:
                    print(f"  IMG: ✗ couldn't fit under size cap")
            else:
                print(f"  IMG: ✗ no usable og:image found")

        print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
        print(f"    same_day_key: {b['same_day_key']}")
        print(text)
        print("---")

        if client:
            try:
                client.post(text, link, ec_title, ec_desc, thumb_blob=thumb_blob)
                time.sleep(2)
            except requests.HTTPError as e:
                print(f"  ! post failed: {e.response.status_code} {e.response.text}", file=sys.stderr)
                continue

        seen.add(b["dedup_key"])
        seen.update(same_day_siblings.get(b["same_day_key"], ()))
        last_posted[b["state"] or "?"] = now.isoformat()
        try:
            save_raw_record(b)
        except OSError as e:
            print(f"  ! raw-record save failed: {e}", file=sys.stderr)

    state["posted"] = sorted(seen)
    state["state_last_posted"] = last_posted
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"\nDone. State saved to {STATE_FILE.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
