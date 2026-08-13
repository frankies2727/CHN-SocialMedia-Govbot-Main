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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

import bill_text
from topic import Topic, load_active_topic

ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = ROOT / "bills.jsonl"

TOPIC: Topic = load_active_topic()
STATE_FILE = TOPIC.state_file_path()

POST_LIMIT = int(os.environ.get("POST_LIMIT", "4"))  # how many bluesky posts per run
# Drop bill actions older than this many days so the feed never posts
# year-old news as if it were fresh. Slow topics still have thousands
# of candidates inside this window. Override via env for tuning.
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "32"))
# Character budget the model is told to fill for the plain-English summary
# (see topic.post_copy_system_prompt). 240 is the old single-post length; the
# daily posters raise it so the 2-post thread — and Instagram's caption — can
# carry a fuller rundown. Read at _post_copy() time, so a caller (or another
# platform's script, via `pb.POST_COPY_MAX_CHARS = …`) sets it before
# summarizing. The weekly digests leave it at the default.
POST_COPY_MAX_CHARS = int(os.environ.get("POST_COPY_MAX_CHARS", "240"))
# Summary length the daily THREAD posters (Bluesky/X/Threads) request: enough to
# lead post 1 and spill a continuation into the post-2 self-reply. Instagram sets
# its own, larger target (it has a 2,200-char caption, not a per-post cap).
DAILY_SUMMARY_CHARS = int(os.environ.get("DAILY_SUMMARY_CHARS", "450"))
# Continuation cues for the 2-post daily thread so readers know a reply follows:
# post 1 ends with CONT_SUFFIX, post 2 opens with CONT_PREFIX. Added only when a
# post 2 actually exists. Shared by the Bluesky/X/Threads thread composers.
# Post 1 trails off with "..." on the sentence line and a second "..." on its own
# line; post 2 picks the thread back up with a leading "... ...".
CONT_SUFFIX = "...\n..."
CONT_PREFIX = "... ..."
DRY_RUN = os.environ.get("DRY_RUN") == "1"
# og:image fetching is paused by default. Set FETCH_OG_IMAGE=1 to re-enable
# thumbnail scraping from bill-page URLs. When off, posts still get an external
# link card — just without the image.
FETCH_OG_IMAGE = os.environ.get("FETCH_OG_IMAGE", "0") == "1"

# Persistence knobs, independent of DRY_RUN. Default both ON so the daily
# scheduler keeps its dedup guarantees and raw-artifact trail. The
# post_bluesky_specific_bill workflow exposes them as checkboxes so an
# operator can post a one-off without polluting the state file, or do a
# dry-run that still records the bill (e.g. mark a bill as "handled" without
# publishing).
SAVE_STATE = os.environ.get("SAVE_STATE", "1") == "1"
SAVE_RAW = os.environ.get("SAVE_RAW", "1") == "1"

# Force-mode: when both FORCE_STATE and FORCE_BILL_ID are set, skip the random
# weighted draw and the topic-keyword/freshness gates and post exactly that one
# bill to the active topic's Bluesky account. Driven by the
# post_bluesky_specific_bill workflow. FORCE_REPOST=1 bypasses the dedup gate
# so an already-posted bill can be re-posted.
FORCE_STATE = (os.environ.get("FORCE_STATE") or "").strip().lower()
FORCE_BILL_ID = (os.environ.get("FORCE_BILL_ID") or "").strip()
FORCE_REPOST = os.environ.get("FORCE_REPOST") == "1"

BSKY_HANDLE = TOPIC.bluesky_handle()
BSKY_PASSWORD = TOPIC.bluesky_password()

BLUESKY_API = "https://bsky.social/xrpc"

# Local LLM via Ollama. Defaults assume `ollama serve` is running on the
# same host (e.g. installed in the GitHub Actions step before this script runs)
# and the model has been pulled with `ollama pull <LLM_MODEL>`.
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:11434/api/chat")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "420"))
# On a timeout/transport error the post-copy call is retried (see _post_copy).
# The retries use this shorter ceiling so a genuinely wedged model fails fast and
# the run falls back to deterministic copy instead of burning the job's minute
# budget — a warm model answers in seconds, so a short retry window is plenty.
LLM_RETRY_TIMEOUT = int(os.environ.get("LLM_RETRY_TIMEOUT", "180"))
# How long Ollama keeps the model resident between requests. Without this it
# defaults to 5 minutes, so a topic that spends several minutes downloading
# PDFs between summaries can force a cold model reload mid-run. Pin it for the
# whole run (set to "-1" to keep loaded indefinitely, "0" to unload eagerly).
LLM_KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "30m")
# LLM relevance gate: after keyword matching selects a bill, ask the local model
# to confirm it's genuinely about the topic (catches omnibus/budget bills that
# match on a single incidental subject tag). On by default; set RELEVANCE_GATE=0
# to disable (e.g. if the LLM is unavailable and you want keyword-only behavior).
RELEVANCE_GATE = (os.environ.get("RELEVANCE_GATE", "1").strip() != "0")
# The gate reads a wide slice of the bill so its judgment is well grounded, which
# on the loaded free runner can make the CPU model's call run long. Give the gate
# its OWN timeout — longer than the copy call's LLM_TIMEOUT — so a slow call is
# allowed to finish and return a real verdict instead of hitting the ceiling and
# falling open (which lets the bill through unjudged). Overridable per run. Kept
# above LLM_TIMEOUT (the copy call's ceiling) so the invariant holds after that
# ceiling was raised.
RELEVANCE_GATE_TIMEOUT = int(os.environ.get("RELEVANCE_GATE_TIMEOUT", "480"))

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
    # Federal (U.S. Congress). govbot tags federal bills "state:usa"; detect_state
    # normalizes that to the "US" code used throughout this module.
    "US",
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
    "US":"U.S. Congress",
}

MAX_POST = 290
LINK_PREFIX = "🔗 "
# Short, clickable anchor text shown in place of the long bill URL. The URL
# itself stays the facet's link target (and the external link card), so tapping
# it still opens the full bill document — it just no longer eats ~80 characters
# of the post.
LINK_ANCHOR = "Read the full bill"

# Titles at or below this length (characters) are used as-is in the post head;
# longer ones get rewritten by the local model into a short plain-English
# headline. Set as low as possible so virtually every real title is rephrased
# into punchy layman's terms (and the freed head space goes to a fuller
# summary) — shorten_title still bails to the raw title when there's no
# abstract AND no full bill text to ground the rewrite, so title-only records
# can't be hallucinated into something new.
HEADLINE_THRESHOLD = 2
# Longest headline shown in the post head. The daily feed now posts a 2-post
# thread, so the summary continues into the self-reply and no longer competes
# with the headline for post-1 space — a full, complete headline should never be
# chopped to "…". Kept a little above the prompt's ~80-char target so a slightly
# long but complete model headline fits whole instead of being truncated
# mid-phrase ("… List of Most Dangerous…").
HEADLINE_MAX_LEN = 90


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


# Optional prebuilt, pre-normalized bills file. The Bluesky workflow runs many
# topics per shard, each as a separate process; without this each one would
# re-parse bills.jsonl and re-run extract_fields() (a topic-independent step)
# over every record. Building the normalized list once per shard and pointing
# the topic processes at it via BILLS_NORMALIZED turns N per-line JSON parses +
# N extract passes into a single array load.
NORMALIZED_PATH = os.environ.get("BILLS_NORMALIZED", "").strip()


def build_normalized(records: list[dict]) -> list[dict]:
    """Apply extract_fields() to every raw record once, attaching the source
    record as ``_raw`` (needed by save_raw_record/save_full_text for posted
    bills). Topic-independent: the result is reused across topics. Records that
    extract_fields() rejects (missing title/date, etc.) are dropped here, the
    same as the inline loop in main() used to do."""
    out: list[dict] = []
    for r in records:
        b = extract_fields(r)
        if not b:
            continue
        b["_raw"] = r
        out.append(b)
    return out


def load_normalized_bills() -> list[dict]:
    """Return the per-run normalized bill list. When BILLS_NORMALIZED points at
    a prebuilt file (written once per shard via ``--build-normalized``), load it
    with a single json.loads of the array — skipping the per-line parse and the
    extract_fields() pass that would otherwise repeat for every topic. Falls
    back to parsing bills.jsonl directly when no prebuilt file is configured, so
    local runs and the other platforms are unaffected."""
    if NORMALIZED_PATH:
        p = Path(NORMALIZED_PATH)
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                print(f"Loaded {len(data)} normalized records from {p.name}")
                return data
            except (OSError, ValueError) as e:
                print(f"  ! normalized cache unreadable ({e}); "
                      f"rebuilding from {JSONL_PATH.name}", file=sys.stderr)
    return build_normalized(load_bills(JSONL_PATH))


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

# govbot tags every jurisdiction with a "state:<code>" marker in its source
# paths — two-letter codes for the 50 states/territories/DC, and the special
# three-letter "state:usa" for federal (U.S. Congress) bills. Match both widths
# so federal bills aren't silently dropped (a 2-letter-only pattern skips
# "usa"), then normalize "usa" -> "US" in detect_state.
_STATE_TAG_PATTERN = re.compile(r"\bstate:([a-z]{2,3})\b", re.IGNORECASE)


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
            # Federal bills carry "state:usa"; everywhere else in this module a
            # federal bill is the 2-letter "US".
            if code == "USA":
                return "US"
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


# Some California records arrive with the legislature site's scraped boilerplate
# fused onto the real title: the clean subject word, then the digest header with
# NO separating space — "Employment.LEGISLATIVE COUNSEL'S DIGESTSB 1444, as
# amended, Committee on Labor…", "Peace officers.LEGISLATIVE COUNSEL'S DIGESTSB
# 938, as amended…". Shown raw, that whole wall of legalese becomes the post
# headline. Cut it back to the clean leading subject at the "LEGISLATIVE
# COUNSEL'S DIGEST" marker, and strip any leading "Bill Text - <id>" / trailing
# "skip to content …" web chrome, so the headline path sees a short subject it
# can build real copy around instead of the raw digest.
_COUNSEL_DIGEST_RE = re.compile(
    r"\s*LEGISLATIVE\s+COUNSEL['’]?S?\s+DIGEST.*$", re.IGNORECASE | re.DOTALL
)
_TITLE_CHROME_PREFIX_RE = re.compile(r"^\s*Bill\s+Text\s*[-–—]\s*\S+\s+", re.IGNORECASE)
_TITLE_CHROME_TAIL_RE = re.compile(r"\s*skip to content.*$", re.IGNORECASE | re.DOTALL)


def _sanitize_title(title: str) -> str:
    """Strip scraped web/digest boilerplate a few states fuse onto the real
    title, cutting it back to the clean leading subject. Never returns empty —
    falls back to the original title when the cut would leave nothing."""
    t = title or ""
    t = _TITLE_CHROME_PREFIX_RE.sub("", t)
    t = _TITLE_CHROME_TAIL_RE.sub("", t)
    t = _COUNSEL_DIGEST_RE.sub("", t)
    t = t.strip()
    return t if t else (title or "")


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


# A Pennsylvania/federal-style statute title: "An Act amending the act of July
# 31, 1968 (P.L.805, No.247), known as …, providing for …". It reads as pure
# legalese and, at 250-ish characters, slips under the 300-char blob threshold,
# so it must be caught separately and delegalesed before it can be shown raw.
_LEGALESE_ACT_TITLE_RE = re.compile(
    r"^\s*an\s+act\b.*\b(?:amending|providing for|relating to|relative to|"
    r"to amend|authorizing|to authorize|concerning|establishing|prohibiting|"
    r"requiring|creating)\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_legalese_act_title(title: str) -> bool:
    """True when the title is a bare statute enacting clause ("An Act amending
    …, providing for …") rather than a plain-English headline — legalese that
    should be delegalesed before display."""
    t = (title or "").strip()
    if len(t) < 40:
        return False
    return bool(_LEGALESE_ACT_TITLE_RE.match(t))


# A quoted short-title the bill gives itself — '"Affordable Power Purchase
# Agreements Extension Act"' — is already a clean, plain-ish name. When a raw
# title leads with one (New Jersey's '"<name> Act"; concerns …' form), it makes a
# far better headline than the legalese tail that follows it.
_QUOTED_ACT_NAME_RE = re.compile(r'["“”]([^"“”]{6,90}?(?:\bAct\b|\bLaw\b))["“”]')


def _quoted_act_name(title: str) -> str:
    """The bill's own quoted short-title ("… Act"/"… Law") if the title carries
    one, else "". Used as a clean headline fallback for records whose title is a
    quoted act name followed by a legalese description."""
    m = _QUOTED_ACT_NAME_RE.search(title or "")
    if not m:
        return ""
    return " ".join(m.group(1).split()).strip()


# Enacting-clause connectors that mark a title as a bare legalese subject line
# rather than a plain-English headline ("relative to …", "concerning …").
_CONNECTOR_TITLE_RE = re.compile(r"^\s*(?:relative to|relating to|concerning)\b",
                                 re.IGNORECASE)


def _title_is_legalese(title: str) -> bool:
    """True when a title should be delegalesed before it can stand as the post
    headline: a bare "An Act …" enacting clause, a '"<name> Act"; concerns …'
    quoted-name-plus-description (New Jersey), or a long "relative to …" /
    "concerning …" subject line (New Hampshire)."""
    t = (title or "").strip()
    if _is_legalese_act_title(t):
        return True
    if _quoted_act_name(t) and ";" in t:
        return True
    return len(t) > 55 and bool(_CONNECTOR_TITLE_RE.match(t))


def _is_vague_subject_title(title: str, subjects: str = "") -> bool:
    """True when the `title` is just a generic subject label, not a description
    of the bill. California titles omnibus/budget bills this way — "State
    government.", "Public safety.", "Courts." — with the same word echoed in the
    `subject` field. Such a title tells a reader nothing, so it must never stand
    as the headline; the copy has to be built from the bill's provisions instead.
    Conservative: only a short, few-word title that matches the subject list
    qualifies, so real short titles ("Data Center Tax Credit Act") don't."""
    t = (title or "").strip().rstrip(".").strip()
    if not t or len(t) > 40 or len(t.split()) > 4:
        return False
    key = re.sub(r"[^a-z0-9]", "", t.lower())
    if not key:
        return False
    subs = re.sub(r"[^a-z0-9]", "", (subjects or "").lower())
    return key in subs


def extract_fields(record: dict) -> dict | None:
    bill = record.get("bill") or {}
    log = record.get("log") or {}

    identifier = bill.get("identifier") or record.get("id") or ""
    title = bill.get("title") or ""
    if not identifier or not title:
        return None
    # Strip scraped web/digest boilerplate (California's "…LEGISLATIVE COUNSEL'S
    # DIGEST…" wall, leginfo nav chrome) before anything downstream — display,
    # headline generation, and the LLM prompt — ever sees the title.
    title = _sanitize_title(title)

    state = detect_state(record)
    session = bill.get("legislative_session") or ""

    # Path to the bill's on-disk metadata.json, used to fetch the full bill
    # text (PDF) for richer summaries. May be "" — summarize() handles that.
    sources = record.get("sources") or {}
    sources_bill = sources.get("bill") or ""

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

    # The feed sometimes carries an authoritative per-bill URL in the action's
    # sources. For most states the per-state builder reconstructs a cleaner link
    # from (session, identifier), but a few states can't be reconstructed (see
    # _TRUSTED_SOURCE_URL_RE), so keep the first source URL around for link_for.
    source_url = ""
    for s in (action.get("sources") or []):
        if isinstance(s, dict) and s.get("url"):
            source_url = s["url"]
            break

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
        "sources_bill": sources_bill,
        "source_url": source_url,
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
    # No headline, but a bare legalese title should never show raw: an "An Act
    # amending …, providing for <purpose>" enacting clause, a '"<name> Act";
    # concerns …' quoted-name title (New Jersey), or a long "relative to …"
    # subject line (New Hampshire). Delegalese it to the bill's own quoted name
    # or stated purpose; keep the raw title only if that fails.
    if _title_is_legalese(title):
        cleaned = _delegalese_headline(title)
        if cleaned:
            return cleaned
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


# Some sources append the action date to the description, which repeats the
# formatted date we already prepend in format_action_line. Two shapes occur:
#   * parenthetical — e.g. California's "Do pass. (Ayes 14. Noes 0.) (May 14)."
#   * bare (no parentheses) — e.g. North Carolina's "Signed by Gov. 7/7/2026",
#     North Dakota's "Signed by Governor 01/23", Vermont's "... Governor May 21".
# Strip the trailing date in either shape, but only when it names the same day
# as the action date — a trailing date for a *different* day is kept, since it
# carries information the prepended date doesn't.
_MONTH_PREFIXES = {1:"jan", 2:"feb", 3:"mar", 4:"apr", 5:"may", 6:"jun",
                   7:"jul", 8:"aug", 9:"sep", 10:"oct", 11:"nov", 12:"dec"}
_TRAILING_PAREN_RE = re.compile(r"\(\s*([^()]*?)\s*\)\s*\.?\s*$")
# A bare date token anchored to the end of the string: numeric ("7/7/2026",
# "01/23", "7-7-26") or a month name ("May 21", "July 7, 2026").
_TRAILING_BARE_DATE_RE = re.compile(
    r"(?P<date>"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2}(?:,?\s*\d{2,4})?"
    r")"
    r"\s*\.?\s*$"
)


def _date_token_matches(token: str, d: "datetime") -> bool:
    """True if ``token`` (e.g. "7/7/2026" or "May 21") names the same month and
    day as ``d``. Year, if present, is ignored — the day is what identifies the
    action, and abbreviated/2-digit years are common."""
    inner = token.strip().lower().rstrip(".")
    num = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(?:[/-]\d{2,4})?", inner)
    if num:
        return int(num.group(1)) == d.month and int(num.group(2)) == d.day
    name = re.fullmatch(r"([a-z]{3,9})\.?\s+(\d{1,2})(?:,?\s*\d{2,4})?", inner)
    if name:
        return name.group(1).startswith(_MONTH_PREFIXES[d.month]) \
            and int(name.group(2)) == d.day
    return False


def _strip_trailing_date(s: str, date_yyyy_mm_dd: str) -> str:
    s = s or ""
    try:
        d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d")
    except ValueError:
        return s
    # Parenthetical form first — the date sits inside its own "(...)" and may be
    # preceded by other parentheticals (vote tallies, etc.) that must be kept.
    m = _TRAILING_PAREN_RE.search(s)
    if m and _date_token_matches(m.group(1), d):
        return s[:m.start()].rstrip()
    # Bare form — the date trails the text directly. Cut at the date token and
    # drop the separators between it and the action text, but keep a legitimate
    # abbreviation period (e.g. "Gov.") for _smart_case to normalize.
    m = _TRAILING_BARE_DATE_RE.search(s)
    if m and _date_token_matches(m.group("date"), d):
        return s[:m.start("date")].rstrip(" ,:;-–—")
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


# Procedural action descriptions — especially federal committee referrals —
# carry a long boilerplate tail that adds no news and swallows the whole post,
# e.g. "Referred to the Committee on Education and Workforce, and in addition to
# the Committee on Foreign Affairs, for a period to be subsequently determined by
# the Speaker, in each case for consideration of such provisions as fall within
# the jurisdiction of the committee concerned." Cut everything from the first
# boilerplate marker so only the substantive lead ("Referred to the Committee on
# Education and Workforce") survives.
_ACTION_BOILERPLATE_RE = re.compile(
    r"\s*[,;]?\s*(?:"
    r"and in addition to\b"
    r"|for a period to be subsequently determined\b"
    r"|in each case for consideration\b"
    r"|for consideration of such provisions as fall within\b"
    r").*$",
    re.IGNORECASE | re.DOTALL,
)


def _shorten_action_desc(desc: str, max_len: int = 180) -> str:
    """Trim procedural boilerplate and cap runaway action descriptions at a
    clean clause boundary so the date+action line never eats the whole post."""
    desc = _ACTION_BOILERPLATE_RE.sub("", desc or "").strip().rstrip(",;")
    if len(desc) <= max_len:
        return desc
    # Too long even after de-boilerplating — cut at the last sentence, then
    # clause, then word boundary that fits, and mark the truncation.
    window = desc[: max_len + 1]
    for sep in (". ", "; ", ", "):
        idx = window.rfind(sep)
        if idx >= max_len // 2:
            return desc[:idx].rstrip(",;") + "…"
    idx = window.rfind(" ")
    cut = desc[: idx if idx >= max_len // 2 else max_len].rstrip(",;")
    return cut + "…"


def format_action_line(action_desc: str, date_yyyy_mm_dd: str) -> str:
    trimmed = _shorten_action_desc(
        _strip_trailing_date(_strip_leading_date(action_desc), date_yyyy_mm_dd)
    )
    desc = _smart_case(trimmed)
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
    # The model is asked for a JSON object; when its closing brace/quotes leak
    # into the extracted string value the blurb ends with JSON punctuation
    # (observed tail on NY S7189: '…takes effect immediately."}\''). A '}' never
    # legitimately appears in a plain-English blurb, so cut from the first one,
    # then peel any leftover stray quotes/braces/backticks off the end.
    brace = text.find("}")
    if brace != -1:
        text = text[:brace].rstrip(" \t`{}\"'‘’“”")
    text = text.strip()
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
    # Scrub Minnesota redline markup before the line split (a deleted span can
    # straddle lines) so the deterministic fallbacks never emit "new text begin
    # …" gibberish from an already-committed artifact.
    text = _strip_mn_redline(text)
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


# Characters trimmed from the FRONT of the body that remains after a leading
# title / act-name restatement is removed. Beyond whitespace and connective
# punctuation (dashes, colon, comma, period, semicolon) this MUST include quote
# marks — straight and curly, single and double — because the model routinely
# wraps the bill's name in quotes ("The 'Foo Act' shifts control…"). The match
# ends right after the name but before its closing quote, so without stripping
# quotes here that quote is left dangling at the very start of the post
# (observed: "' shifts control of US international education programs…").
_POST_TITLE_STRIP_CHARS = (
    " -:,.;"        # space, hyphen, colon, comma, period, semicolon
    "—–"  # em dash, en dash
    "\"'`"          # straight double quote, straight single quote, backtick
    "‘’"  # curly single quotes  ‘ ’
    "“”"  # curly double quotes  “ ”
)


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
            rest = summary[i:].lstrip(_POST_TITLE_STRIP_CHARS)
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
    rest = body[pos:].lstrip(_POST_TITLE_STRIP_CHARS)
    rest = _LEAD_FILLER_RE.sub("", rest)
    if not rest:
        return summary
    return rest[:1].upper() + rest[1:]


# Function words ignored when comparing a summary sentence against the headline:
# they carry no topical signal, so leaving them in would dilute the overlap
# score and let near-duplicate sentences slip through.
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "as",
    "at", "with", "from", "that", "this", "these", "those", "is", "are", "be",
    "will", "shall", "would", "each", "every", "any", "all", "into", "its",
    "their", "his", "her", "it", "which", "who", "whom", "annually", "yearly",
}


def _content_words(s: str) -> list[str]:
    """Topical words of a string: normalized, stopwords and 1–2 char tokens
    dropped. Used to score how much a summary sentence overlaps the headline."""
    return [w for w in _normalize(s).split() if len(w) > 2 and w not in _STOPWORDS]


# Common abbreviations that end in a period but do NOT end a sentence, so the
# splitter doesn't break "Dr. Mun Choi Day" into two sentences.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "st", "sen", "rep", "gov", "no", "vs", "etc",
    "inc", "co", "jr", "sr", "prof", "dept", "fig", "approx",
}

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+[\"')\]]*\s+")


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations ('Dr.') and lone
    initials intact. Best-effort — only used to decide whether a leading
    sentence merely restates the headline."""
    text = (text or "").strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        candidate = text[start:m.start()].strip()
        last = re.split(r"\s+", candidate)[-1].strip("\"')(.,;:") if candidate else ""
        # Don't split after an abbreviation or a single-letter initial.
        if last.lower() in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
            continue
        parts.append(text[start:m.end()].strip())
        start = m.end()
    if start < len(text):
        parts.append(text[start:].strip())
    return [p for p in parts if p]


def _strip_headline_echo(summary: str, headline: str) -> str:
    """Drop a leading summary sentence that merely restates the headline in
    different words — the paraphrase case that `_strip_title_prefix` (exact
    prefix only) misses, e.g. headline "Designates March first as Dr. Mun Choi
    Day in Missouri" + summary opening "Missouri will designate March first
    annually as Dr. Mun Choi Day." Only fires when a substantive later sentence
    remains, so the post never collapses to an empty body. Returns the summary
    unchanged when the first sentence carries new information."""
    if not summary or not headline:
        return summary
    sentences = _split_sentences(summary)
    if len(sentences) < 2:
        return summary

    head_words = set(_content_words(headline))
    if len(head_words) < 3:
        return summary
    first_words = set(_content_words(sentences[0]))
    if not first_words:
        return summary

    # Two-sided test: the first sentence is mostly headline words (adds little
    # new), AND it covers nearly all of the headline (it really is the same
    # statement). Both thresholds guard against stripping a sentence that
    # shares some vocabulary but contributes a real new fact.
    in_headline = sum(1 for w in first_words if w in head_words) / len(first_words)
    covered = sum(1 for w in head_words if w in first_words) / len(head_words)
    if in_headline < 0.7 or covered < 0.7:
        return summary

    rest = " ".join(sentences[1:]).strip()
    rest = _LEAD_ARTICLE_RE.sub("", rest, count=1)
    if len(rest) < 15:
        return summary
    return rest[:1].upper() + rest[1:]


# Connector / function words that read as dangling when a truncated summary
# ends on them — dropped (with any trailing punctuation) before the ellipsis so
# a cut never ships "…citing…" or "…disclose stocks and…".
_DANGLING_TAIL_RE = re.compile(
    r"[\s,;:]+(?:and|or|but|nor|the|a|an|of|to|for|with|by|on|in|at|from|as|"
    r"that|which|who|whose|including|citing|such|plus|per|than|into|onto|upon|"
    r"over|under|about|via|while|when|where|whether|because|so)$",
    re.IGNORECASE,
)


def _drop_dangling_tail(s: str) -> str:
    """Strip trailing dangling connectors/punctuation from a truncated string
    so the ellipsis attaches to a content word, not "…, citing" or "… and"."""
    s = s.rstrip(",;:- ")
    prev = None
    while prev != s:
        prev = s
        s = _DANGLING_TAIL_RE.sub("", s).rstrip(",;:- ")
    return s


def _smart_truncate(text: str, max_len: int) -> str:
    """Truncate to <= max_len, ending at a sentence or word boundary.

    A period is only a sentence end when it isn't glued to an alphanumeric
    character on its right — a decimal ("8.5"), an abbreviation ("No."), or a
    URL. Without that guard a cut lands mid-figure and ships a dangling number:
    "…and allocates 8." from an original "…and allocates 8.5 percent…"."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    floor = max(1, int(max_len * 0.6))
    best = -1
    for m in re.finditer(r"[.!?]", cut):
        idx = m.start()
        if idx < floor:
            continue
        if cut[idx] == ".":
            nxt = text[idx + 1] if idx + 1 < len(text) else ""
            # A genuine sentence-ending period is followed by whitespace or the
            # end of the text; one glued to a digit/letter is a decimal point,
            # an abbreviation, or a URL — not a boundary.
            if nxt.isalnum():
                continue
        best = idx
    if best >= 0:
        return cut[: best + 1]
    idx = cut.rfind(" ")
    if idx >= floor:
        return _drop_dangling_tail(cut[:idx]) + "…"
    return _drop_dangling_tail(cut) + "…"


# California-style statute-digest filler that adds no meaning to a social blurb:
# cross-reference tags ("as defined"), deadline clauses ("on or before August
# 15, 2026,"), and hedges ("subject to certain requirements"). Stripped from a
# fallback summary so it reads like plain English, not a code cross-reference.
_DIGEST_FILLER_RE = re.compile(
    r",?\s*\b(?:as defined|as specified|as provided|as described|"
    r"among other things|subject to certain requirements|"
    r"in this regard|as prescribed)\b",
    re.IGNORECASE,
)
_DEADLINE_CLAUSE_RE = re.compile(
    r",\s*on or before\s+[A-Za-z]+\s+\d{1,2},\s*\d{4}\s*,", re.IGNORECASE
)


def _operative_rephrase(sentence: str) -> str:
    """Rephrase a "This bill would <verb> …" digest sentence into a clean blurb
    opening — "Would <verb> …" — and strip statute-digest filler. Leaves a
    non-operative sentence unchanged apart from the filler cleanup."""
    s = " ".join((sentence or "").split())
    s = re.sub(r"^\s*(?:this bill|the bill)\s*,?\s*(?=would\b)", "", s, flags=re.IGNORECASE)
    s = _DEADLINE_CLAUSE_RE.sub(" ", s)
    s = _DIGEST_FILLER_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).replace(" ,", ",").replace(",,", ",").strip()
    return s[:1].upper() + s[1:] if s else s


# A leading statute enumerator — a subsection/paragraph marker "(a)", "(u)",
# "(iii)", "(1)", or "1." — that opens a quoted statute fragment. Pure noise at
# the head of a plain-English blurb, so stripped from the non-LLM fallbacks
# (observed: a post opening "(u) In educating voters, the State Board shall…").
_LEADING_ENUMERATOR_RE = re.compile(r"^(?:\(\s*[A-Za-z0-9]{1,4}\s*\)\s*|\d{1,3}\.\s+)+")

# Statutory-citation / section markers. A fallback blurb dense with these reads
# as legalese, not English ("Parental leave fund account (AS 23.10.705). Sec.
# 19. AS 23.15.630(b) … are repealed."), so a body-derived fallback carrying
# several of them is dropped in favor of a clean bare headline.
_CITATION_MARKER_RE = re.compile(
    r"\bAS\s+\d|\bSec\.|\bSection\s+\d|§|\bP\.?\s*L\.|\bNo\.\s*\d|"
    r"\b\d+\.\d+(?:\.\d+)+|\(\s*[A-Za-z0-9]{1,3}\s*\)",
    re.IGNORECASE,
)

# Inline per-line numbering that some sources (Alaska's HTML full text) weave
# through the prose as standalone 1–2 digit tokens, rather than at line starts
# where bill_text.clean_bill_text can strip them:
#   "… to read: 18 (87) parental leave fund account … 19 * Sec. 19. …"
_INLINE_LINENO_RE = re.compile(r"(?<=\s)\d{1,2}(?=\s)")


def _is_citation_heavy(text: str, threshold: int = 2) -> bool:
    """True when a candidate blurb is dominated by statute citations / section
    markers — i.e. it reads as raw legalese rather than plain English."""
    return len(_CITATION_MARKER_RE.findall(text or "")) >= threshold


# A candidate blurb pulled from a bill's data tables (budget allowances, rate
# schedules) is unreadable noise: pdftotext concatenates the columns into tokens
# like "20194,184,333minusFCPBAminusSCPBA" (observed on NH HB1738's RGGI budget
# table). Detect a blurb that is really a mangled table row so the deterministic
# fallbacks drop it rather than posting the gibberish.
_TABLE_GIBBERISH_RE = re.compile(r"\d\s*minus|minus[A-Za-z]|[A-Za-z]{0,5}\d{4,}[A-Za-z]{2,}")


def _is_table_gibberish(text: str) -> bool:
    """True when a candidate blurb is really a mangled data-table row (fused
    digit/letter tokens, "minus"-glued figures) rather than readable prose."""
    if not text:
        return False
    if _TABLE_GIBBERISH_RE.search(text):
        return True
    # A single long token mixing several digits and letters ("20194,184,333min…")
    # is a collapsed table cell, never a real word.
    for tok in text.split():
        if len(tok) >= 12:
            digits = sum(c.isdigit() for c in tok)
            letters = sum(c.isalpha() for c in tok)
            if digits >= 4 and letters >= 4:
                return True
    return False


def _strip_inline_line_numbers(text: str) -> str:
    """Remove inline per-line numbering (see _INLINE_LINENO_RE). Only fires when
    the standalone 1–2 digit tokens are dense AND largely sequential — the
    fingerprint of line numbering — so a bill that merely cites a few small
    numbers is left untouched. Statutory refs like "AS 23.10.705" are safe: the
    digits there abut a period, not whitespace, so they never match."""
    if not text:
        return text
    matches = _INLINE_LINENO_RE.findall(text)
    if len(matches) < 12:
        return text
    nums = [int(t) for t in matches]
    # Adjacent tokens that step up by one (or reset to the top of a new page)
    # are line numbers; a high fraction of them means the doc is line-numbered.
    seq = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1 or (a >= 8 and b <= 2))
    if seq < 0.5 * (len(nums) - 1):
        return text
    return re.sub(r"\s{2,}", " ", _INLINE_LINENO_RE.sub(" ", text)).strip()


def _excerpt_summary(excerpt: str) -> str:
    """Turn a topic-match excerpt (one or more abstract sentences naming the
    provisions that pulled a bill into its feed) into a fallback blurb. Prefers
    the operative "This bill would …" sentence over a leading "Existing law …"
    description so the blurb reads as what the bill *does*, not the prior state
    of the law. Falls back to the first sentence when no operative one is found."""
    if not excerpt:
        return ""
    # Split on sentence punctuation AND the "…" that matching_excerpt inserts
    # between (or after truncating) provisions, so a leading truncated
    # "Existing law …" clause doesn't hide the operative sentence behind it.
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", excerpt) if s.strip()]
    for s in sentences:
        if re.match(r"\s*(this bill|the bill)\b", s, re.IGNORECASE):
            return _first_sentence(_operative_rephrase(s))
    return _first_sentence(excerpt)


# The lead-in half of a statute-digest provision pair — "Existing law <does X>"
# — describes the STATUS QUO, not the bill. A blurb must never open with it.
_EXISTING_LAW_RE = re.compile(r"^\s*(?:existing law|current law|under existing law)\b",
                              re.IGNORECASE)


def _first_sentence(text: str) -> str:
    """First substantive sentence of a cleaned abstract — the non-LLM fallback
    summary. Skips a leading "Existing law …" status-quo sentence to reach the
    operative "This bill would …" one when the text pairs them (California
    digests), rephrasing it to plain English. Returns "" when there's no usable
    prose, so the caller can drop the summary block rather than post filler."""
    cleaned = _clean_for_llm(_strip_inline_line_numbers(text))
    # Drop a leading statute enumerator ("(u) ", "(1) ") so the blurb opens on
    # prose rather than a subsection marker.
    cleaned = _LEADING_ENUMERATOR_RE.sub("", cleaned).strip()
    if not cleaned:
        return ""
    sentences = re.findall(r"[^.!?]*[.!?]", cleaned) or [cleaned]
    first = sentences[0].strip()
    # When the lead sentence is a status-quo "Existing law …" description, prefer
    # the first operative "This bill would …" sentence that follows it.
    if _EXISTING_LAW_RE.match(first):
        for s in sentences[1:]:
            if re.match(r"\s*(?:this bill|the bill)\b", s, re.IGNORECASE):
                return _smart_truncate(_operative_rephrase(s), 200)
    # A first sentence that's just statute citations reads as legalese; drop it
    # so the caller ships a clean headline instead of a raw fragment. Same for a
    # mangled data-table row.
    if _is_citation_heavy(first) or _is_table_gibberish(first):
        return ""
    return _smart_truncate(first, 200)


# ---------------------------------------------------------------------------
# Language detection + translation
#
# Puerto Rico is the only US jurisdiction in the govbot feed whose action
# descriptions arrive in Spanish (titles and abstracts are sometimes
# pre-translated upstream, action_desc almost never is). Posts must always
# be in English regardless of source, so any field that still looks Spanish
# at compose time is run through the same local Ollama model used for
# summaries. Detection is a cheap heuristic — Spanish-exclusive characters
# (ñ, ¿, ¡) or a small Spanish stopword/legislative-phrase list — so we
# don't spend an LLM round-trip on text that's already English.
# ---------------------------------------------------------------------------

# Spanish-exclusive punctuation — opens questions/exclamations; effectively
# never appears in English, so it triggers on its own.
_SPANISH_PUNCT_RE = re.compile(r"[¿¡]")
# Accented vowels + ñ. Individually WEAK: English legislative text routinely
# carries them inside proper nouns ("Representative Montaño", "San José"), so an
# accent counts only alongside a Spanish function word, never on its own — the
# old detector's lone-accent trigger sent whole English bills through a costly
# (and text-mangling) translation round-trip.
_SPANISH_ACCENT_RE = re.compile(r"[ñÑáéíóúÁÉÍÓÚ]")
# Multiword Spanish legislative phrases — unambiguous, so each triggers alone.
_SPANISH_STRONG_RE = re.compile(
    r"\b(?:proyecto\s+del\s+senado|proyecto\s+de\s+la\s+cámara|"
    r"asamblea\s+legislativa|primera\s+lectura|segunda\s+lectura|"
    r"tercera\s+lectura|referido\s+a\s+la\s+comisión|"
    r"cámara\s+de\s+representantes)\b",
    re.IGNORECASE,
)
# Common Spanish function words / legislative nouns. Individually these can
# coincide with names or codes, so two are required — or one plus an accent.
_SPANISH_WEAK_RE = re.compile(
    r"\b(?:de\s+la|del|en\s+el|en\s+la|por\s+el|por\s+la|para\s+el|para\s+la|"
    r"se\s+ha|se\s+hace|aprobad[oa]|senado|cámara|representantes|comisión|"
    r"enmendar)\b",
    re.IGNORECASE,
)


def _looks_spanish(text: str) -> bool:
    """Cheap heuristic: does this legislative text still read as Spanish?

    Puerto Rico is the only feed jurisdiction that ships Spanish, so this only
    needs to separate genuine Spanish from English that merely contains an
    accented proper noun ("Representative Montaño of Boston"). A lone accent or
    ñ therefore never triggers on its own — it must sit alongside a Spanish
    function word. Spanish-exclusive punctuation (¿¡) and unambiguous multiword
    legislative phrases trigger by themselves."""
    if not text:
        return False
    if _SPANISH_PUNCT_RE.search(text) or _SPANISH_STRONG_RE.search(text):
        return True
    weak = len(_SPANISH_WEAK_RE.findall(text))
    if weak >= 2:
        return True
    return weak >= 1 and bool(_SPANISH_ACCENT_RE.search(text))


def _translate_to_english(text: str, num_predict: int = 400) -> str:
    """Best-effort translate Spanish legislative text to English via the
    configured Ollama model. Returns the original text on any failure so
    the post still goes out — better to ship a partially Spanish line than
    drop the post entirely. ``num_predict`` caps the output length: the 400-token
    default suits the short metadata fields; the full-body translator raises it so
    a long chunk isn't truncated mid-sentence."""
    if not text or not text.strip():
        return text
    try:
        r = requests.post(
            LLM_API_URL,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": (
                        "You translate Spanish legislative text from Puerto "
                        "Rico to clear, neutral English. Preserve bill "
                        "numbers, dates, chamber names, and proper nouns. If "
                        "the input is already English, return it verbatim. "
                        "Output ONLY the English translation — no preamble, "
                        "no commentary, no surrounding quotes, no notes."
                    )},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "keep_alive": LLM_KEEP_ALIVE,
                "options": {"num_predict": num_predict, "temperature": 0.1},
            },
            timeout=LLM_TIMEOUT,
        )
        if not r.ok:
            print(f"  ! translation {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return text
        data = r.json()
        out = (data.get("message") or {}).get("content") or data.get("response") or ""
        out = (out or "").strip().strip('"').strip("'").strip()
        return out or text
    except Exception as e:
        print(f"  ! translation failed: {e}", file=sys.stderr)
        return text


def ensure_english_fields(b: dict) -> dict:
    """Translate any Spanish-looking title / abstract / action_desc / subjects
    in `b` to English in place. Mutates and returns `b` so callers can chain.
    Runs the detector first so already-English fields don't trigger LLM
    calls. Used for Puerto Rico bills whose action_desc almost always
    arrives in Spanish from OpenStates."""
    for field_name in ("title", "abstract", "action_desc", "subjects"):
        val = b.get(field_name) or ""
        if _looks_spanish(val):
            translated = _translate_to_english(val)
            if translated and translated != val:
                print(f"  translated {field_name}: {val[:80]!r} -> {translated[:80]!r}")
            b[field_name] = translated
    return b


# Full bill bodies (Puerto Rico ships them in Spanish) are far longer than the
# short metadata fields ensure_english_fields() handles, so a single LLM call
# with the summary-sized output budget would truncate the translation. Translate
# the body in bounded chunks instead — each with its own generous output budget —
# and cache the English result. Only the leading window is translated, and the
# summarizer reads that same window (char_cap in _post_copy), so whatever we
# translate is exactly what the model can see — the cap sizes both. It stays
# bounded so a very long PDF doesn't run up dozens of chunk translations; any
# tail past the cap is kept verbatim so the persisted bills_full_text artifact
# stays complete.
FULLTEXT_TRANSLATE_MAX_CHARS = 18000
_FULLTEXT_TRANSLATE_CHUNK_CHARS = 1800


def _chunk_for_translation(text: str, size: int) -> list[str]:
    """Split ``text`` into <= ``size``-char chunks on paragraph, then sentence,
    then hard boundaries, so each translation call sees a coherent span rather
    than a mid-word cut."""
    chunks: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            # Flush what we have, then hard-split the oversized paragraph on
            # sentence ends, falling back to fixed-width slices.
            if buf:
                chunks.append(buf)
                buf = ""
            for piece in re.split(r"(?<=[.;:])\s+", para):
                while len(piece) > size:
                    chunks.append(piece[:size])
                    piece = piece[size:]
                if not piece:
                    continue
                if len(buf) + len(piece) + 1 > size:
                    if buf:
                        chunks.append(buf)
                    buf = piece
                else:
                    buf = f"{buf} {piece}".strip()
            continue
        if len(buf) + len(para) + 2 > size:
            if buf:
                chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}".strip() if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def _translate_full_text_to_english(text: str) -> str:
    """Translate a (Spanish) bill body to English, chunked so no single call
    truncates. Caps the translated span at FULLTEXT_TRANSLATE_MAX_CHARS — well
    past what any downstream consumer reads — and appends any remaining tail
    verbatim so the persisted artifact stays complete. Fails open per chunk: a
    chunk that doesn't translate keeps its original text so the post still
    ships."""
    if not text or not text.strip():
        return text
    head, tail = (
        text[:FULLTEXT_TRANSLATE_MAX_CHARS],
        text[FULLTEXT_TRANSLATE_MAX_CHARS:],
    )
    out_parts: list[str] = []
    for chunk in _chunk_for_translation(head, _FULLTEXT_TRANSLATE_CHUNK_CHARS):
        translated = _translate_to_english(chunk, num_predict=900)
        out_parts.append(translated or chunk)
    translated_head = "\n\n".join(p for p in out_parts if p)
    return translated_head + (("\n\n" + tail) if tail.strip() else "")


def _get_full_text(b: dict) -> str:
    """Full bill body text, extracted from the bill's PDF via bill_text and
    cached on the record so shorten_title() and summarize() share a single
    fetch (shorten_title runs first). Returns "" when extraction isn't
    possible (no PDF link, pdftotext missing, network error, etc.); the empty
    result is cached too so a failed fetch isn't retried by the next caller.

    Runs only for bills that have already passed the topic filter and the post
    draw, so it never fetches thousands of PDFs."""
    if "full_text" in b:
        return b["full_text"]
    full_text = ""
    sources_bill = b.get("sources_bill") or ""
    if sources_bill:
        try:
            full_text, reason = bill_text.extract_bill_text_verbose(sources_bill)
            full_text = _strip_extraction_header(full_text or "")
        except Exception as e:
            print(f"  TEXT: ✗ extraction error, using abstract: {e}", file=sys.stderr)
            full_text, reason = "", "error"
        if full_text:
            print(f"  TEXT: ✓ FULL PDF TEXT USED ({len(full_text)} chars) "
                  f"for {b.get('state','??')} {b.get('identifier','?')}")
            # Translate a Spanish body (Puerto Rico) to English ONCE, here where
            # it's fetched, so every consumer — summarize(), shorten_title(), the
            # relevance gate, the deterministic fallback copy, and the persisted
            # bills_full_text artifact — reads English from a single pass.
            # ensure_english_fields() only covers the short metadata fields; the
            # body has to be handled here. Detect first so an English PDF never
            # triggers the (much costlier) chunked LLM round-trips.
            if _looks_spanish(full_text):
                translated = _translate_full_text_to_english(full_text)
                if translated and translated != full_text:
                    print(f"  TEXT: translated bill body to English "
                          f"({len(full_text)} -> {len(translated)} chars) "
                          f"for {b.get('state','??')} {b.get('identifier','?')}")
                    full_text = translated
        else:
            print(f"  TEXT: ✗ full PDF text NOT used ({reason}) "
                  f"for {b.get('state','??')} {b.get('identifier','?')} — using abstract")
    else:
        print(f"  TEXT: ✗ full PDF text NOT used (no-sources-path) "
              f"for {b.get('state','??')} {b.get('identifier','?')} — using abstract")
    # Cache on the record (saved to topics/<name>/bills_full_text/ for future
    # RAG/digest use when non-empty) so summarize() reuses this fetch.
    b["full_text"] = full_text
    return full_text


# A floor-amendment sheet ("AMEND House Bill No. 444 by inserting…") is the
# document a state surfaces for an amendment action — and sometimes even for a
# later action like passage. It is NOT the bill's substance: its body is just
# the one- or two-line change. Summarizing it produces a fragment about that
# lone change ("…takes effect July 1, 2027") that reads as disconnected from a
# headline grounded on the bill's abstract. Detect it so summarize() and
# shorten_title() fall back to the abstract, which describes the actual
# legislation; the post's action line still reports the amendment event.
# A document HEADER that identifies the whole document as an amendment sheet —
# "HOUSE AMENDMENT NO. 2 TO HOUSE BILL NO. 364", "SENATE AMENDMENT NO. 1 TO
# SENATE SUBSTITUTE NO. 1 FOR SENATE BILL NO. 16". This is unambiguous: a real
# bill body never opens this way, so it marks an amendment sheet regardless of
# length (some insert a whole block of replacement statute text and run long).
_AMENDMENT_SHEET_HEADER_RE = re.compile(
    r"\b(?:HOUSE|SENATE)\s+AMENDMENT\s+NO\.|"
    r"\bAMENDMENT\s+NO\.\s*\d+\s+TO\b.{0,80}\bBILL\s+NO\.",
    re.IGNORECASE | re.DOTALL,
)

# Weaker amendment language ("AMEND … Bill No.", "This amendment") that can also
# appear inside a genuine bill, so it only marks an amendment sheet alongside a
# short overall length.
_AMENDMENT_DOC_RE = re.compile(
    r"\bAMEND\b.{0,80}\bBill\s+No\.|"
    r"\bThis\s+amendment\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_amendment_doc(text: str) -> bool:
    """True when extracted "full text" is actually a floor-amendment sheet
    rather than the bill body, so summarize()/shorten_title()/the relevance gate
    fall back to the abstract instead of turning line-by-line diffs ("delete
    'projects' on line 22 …") into the post."""
    if not text:
        return False
    head = text[:1500]
    # A document whose header names it an amendment sheet IS one, at any length.
    if _AMENDMENT_SHEET_HEADER_RE.search(head):
        return True
    # Weaker markers only count when the document is also short — real bill
    # bodies run long, so a substantive bill that merely contains amendment
    # language is never dropped.
    return bool(_AMENDMENT_DOC_RE.search(head)) and len(text) < 2500


# A bill that repeals a statute section and re-enacts it "in lieu thereof"
# (Missouri's standard amendatory form, and the generic repeal-and-reenact
# pattern many states use) carries the ENTIRE existing section forward in the
# bill text, with only the genuinely new language marked in bold-face — a
# distinction pdftotext throws away. So the extracted full text is dominated by
# unchanged law (residency rules, organ-donor donations, fraud penalties, …),
# and a naive summary describes that carried-over boilerplate instead of the
# one provision the bill actually adds. Detect this so summarize() and
# shorten_title() can anchor on the bill's stated purpose rather than the wall
# of unchanged statute. Checked only over the document's opening, where the
# enacting clause lives, so a passing "repeal" deep in the body can't trigger
# it.
_AMENDATORY_RE = re.compile(
    r"\benact(?:ed)?\s+in\s+lieu\s+thereof\b|"
    r"\bamend(?:ed|ing)?\s+and\s+re-?enact(?:ed|ing)?\b|"
    r"\b(?:repeal|repealing|repealed)\b[^.]{0,160}?\b(?:re-?enact|reenact|reenacting|enacting)\b|"
    r"\bto\s+amend\b[^.]{0,140}?\bby\s+(?:repealing|adding|amending)\b|"
    # New Jersey's amend-by-carrying-forward form: the enacting clause says it is
    # "amending various parts of the statutory law" and each affected section is
    # then reprinted whole under "N.J.S.… is amended to read as follows:".
    r"\bamending\s+(?:various|multiple|certain|parts?\b)[^.]{0,60}?\bstatutory\s+law\b|"
    r"\bis\s+amended\s+to\s+read\s+as\s+follows\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_amendatory_reenactment(text: str) -> bool:
    """True when the bill text is an amendatory repeal-and-reenact: it carries
    an existing statute section forward (mostly unchanged) and changes only a
    small part. Only the opening of the document is inspected."""
    if not text:
        return False
    return bool(_AMENDATORY_RE.search(text[:1200]))


# The bill's own one-line statement of purpose, taken from the
# "AN ACT … relating to <purpose>" enacting clause. For amendatory bills this
# is the most reliable signal of what the bill actually changes, since the body
# text is mostly unchanged law. Captures up to the end of the clause and is
# searched only near the top, where the clause always appears.
_ACT_PURPOSE_RE = re.compile(
    r"\brelating to\s+(.+?)\s*(?:\.\s|\.$|;|\bbe it enacted\b|\bto read as follows\b)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_act_purpose(text: str) -> str:
    """Pull '<purpose>' from a bill's 'AN ACT … relating to <purpose>' enacting
    clause. Returns a cleaned phrase like 'sex designation on driver's
    licenses', or "" when the clause is absent or implausibly long (a malformed
    capture swallowing the body)."""
    if not text:
        return ""
    m = _ACT_PURPOSE_RE.search(text[:1500])
    if not m:
        return ""
    purpose = " ".join(m.group(1).split()).strip(" ,;:.")
    if not purpose or len(purpose) > 160:
        return ""
    return purpose


# ---------------------------------------------------------------------------
# Full-text preparation and deterministic (non-LLM) fallbacks
#
# A bill that amends an existing statute opens by reciting that statute's ENTIRE
# long title — a multi-hundred-character parenthetical of unrelated boilerplate
# ("An act to empower cities of the second class A … providing for mediation;
# providing for transferable development rights; …") — before it ever states
# what THIS bill does. Fed that verbatim, a small model spends its whole context
# window on the recited boilerplate and never reaches the operative section, so
# it returns empty or off-topic copy and the post falls back to the raw legalese
# title. These helpers (a) strip the recitation so the operative text leads, and
# (b) derive a clean plain-English headline/summary straight from the bill when
# the model still comes back empty, so a post is never shipped as raw legalese.
# ---------------------------------------------------------------------------

# The recited long title of an amended act sits inside `entitled "…"` (or
# `known as "…"`). It's the OTHER act's description, never this bill's, so drop
# it wholesale. Bounded so a stray quote can't swallow the operative body.
_RECITED_LONGTITLE_RE = re.compile(
    r'\bentitled\s*"[^"]{0,4000}?"', re.IGNORECASE | re.DOTALL
)
# The bill's own subject in an "An Act … providing for <purpose>" / "relating to
# <purpose>" enacting clause. Captured from the LAST such connector so a recited
# act's many "providing for …" clauses don't win over the bill's own tail.
_PROVIDING_FOR_RE = re.compile(
    r"\b(?:providing for|relating to|relative to|concerning|concerns|"
    r"prohibiting|authorizing|requiring|establishing|creating)\b\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)


# pdftotext/legislative artifacts that survive into already-committed text or a
# stale cache (bill_text.clean_bill_text strips these on fresh extraction, but
# older .txt files and cache entries predate that): the "<--" amendment arrows,
# the print-run / session / sponsor front-matter lines, and Minnesota's redline
# markup ("new text begin/end", "deleted text begin/end"). The bills_full_text/*
# artifacts committed before the bill_text fix still carry the MN markers, so
# scrub them here too — for the LLM window and the deterministic fallbacks alike
# — the same reason the arrow/front-matter regexes are duplicated in this module.
_ARTIFACT_ARROW_RE = re.compile(r"<-{1,}")
_ARTIFACT_FRONTMATTER_RE = re.compile(
    r"(?im)^\s*(?:(?:PRIOR\s+)?PRINTER'?S\s+NO\.?.*|Session\s+of.*|"
    r"INTRODUCED\s+BY\b.*)$"
)
# Minnesota redline: drop deleted spans whole (content included), strip the
# inserted-text markers but keep the wrapped text. Delete spans first. See
# bill_text._MN_DELETED_RE / _MN_NEWTEXT_MARK_RE for the same patterns.
_MN_DELETED_RE = re.compile(
    r"deleted\s+text\s+begin\b.*?deleted\s+text\s+end\b", re.IGNORECASE | re.DOTALL
)
_MN_NEWTEXT_MARK_RE = re.compile(r"new\s+text\s+(?:begin|end)\b", re.IGNORECASE)


def _strip_mn_redline(text: str) -> str:
    """Remove Minnesota redline markup from extracted bill text (deleted spans
    dropped whole, inserted-text markers stripped keeping their text)."""
    if not text:
        return text
    text = _MN_DELETED_RE.sub(" ", text)
    return _MN_NEWTEXT_MARK_RE.sub(" ", text)
# The govbot dataset prepends a metadata header to each extracted bill document:
#     Title: <...>
#     Official Title: <...>
#     Source: versions - <label>
#     Media Type: <text/html|application/pdf>
#     Visual Markup Detected: true -- this document likely contains redline …
#     <blank>
#     ==========...==========      (a rule of many '=')
#     <blank>
#     <the actual bill text>
# One block per document version, so a multi-version bill carries several. Left
# in, this header is pure noise to the model — and, worse, when the local LLM
# returns an empty blurb the deterministic fallback (_summary_from_body /
# _first_sentence) emits the header itself as the post body (observed: "Versions
# - Introduced Version Media Type. Text/html === A3497 assembly, No.", and the
# newer "… Current Version Media Type. Application/pdf Visual Markup Detected.").
# The header always opens with "Title:" and runs to the "====" rule, but the set
# of keys between them varies (govbot added the "Visual Markup Detected:" note),
# so after the first known key we consume ANY further "Key: value" lines up to
# the rule rather than an exact key whitelist — otherwise one unlisted key (the
# Visual-Markup note) left the whole header, and its "Media Type:"/"Visual Markup
# Detected:" text, leaking into the post. Strip every such block so no consumer —
# model, fallback, relevance gate, persisted artifact — ever sees it.
_EXTRACTION_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:Title|Official Title|Source|Media Type|Visual Markup Detected):[^\n]*\n"
    r"(?:[ \t]*[A-Za-z][A-Za-z '-]{0,40}:[^\n]*\n)*"
    r"[ \t]*\n?[ \t]*={20,}[ \t]*\n+"
)


def _strip_extraction_header(text: str) -> str:
    """Remove the govbot dataset's ``Title:/Source:/Media Type: … ====`` document
    header(s) so only the real bill text remains (see _EXTRACTION_HEADER_RE)."""
    if not text or "Media Type:" not in text:
        return text
    return _EXTRACTION_HEADER_RE.sub("", text).lstrip()


# A bill that amends existing law by "repealing and re-enacting" a statute
# section carries the ENTIRE section forward, mostly unchanged, and buries its
# one real change deep in — or after — pages of recited statute. But legislatures
# write a plain-English explainer for their own members that says exactly what
# the bill does, and it sits OUTSIDE the operative text: New Jersey's "STATEMENT"
# (at the very end), a "SYNOPSIS" line (near the top), or California's
# "LEGISLATIVE COUNSEL'S DIGEST". Fed only the first N characters of the bill,
# the model never reaches a trailing STATEMENT, so it summarizes the recited
# boilerplate (school-board bidding rules) instead of the actual change
# (extending clean-energy contracts) — the exact NJ S4162 failure. These pull
# that explainer out so it can LEAD the window the model reads.
_BILL_STATEMENT_RE = re.compile(
    r"\bSTATEMENT\b\s+(This\s+(?:bill|act|amendment|resolution|supplement)\b.{40,3500})",
    re.IGNORECASE | re.DOTALL,
)
_BILL_SYNOPSIS_RE = re.compile(
    r"\bSYNOPSIS\b\s+(.{15,700}?)\s*(?:\bCURRENT\s+VERSION\s+OF\s+TEXT\b|\bAn\s+Act\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_BILL_DIGEST_RE = re.compile(
    r"\bLEGISLATIVE\s+COUNSEL['’]?S?\s+DIGEST\b\s*(.{40,3000}?)"
    r"(?:\bDigest\s+Key\b|\bBill\s+Text\b|\bWHEREAS\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_bill_explainer(text: str) -> str:
    """The bill's own plain-English explainer written for legislators — New
    Jersey's trailing "STATEMENT", a "SYNOPSIS" line, or California's
    "LEGISLATIVE COUNSEL'S DIGEST" — collapsed to a single clean paragraph, or ""
    when the bill carries none. This is the most reliable statement of what a
    recitation-heavy amendatory bill actually changes."""
    if not text:
        return ""
    for rx, cap in ((_BILL_STATEMENT_RE, 2200), (_BILL_SYNOPSIS_RE, 700),
                    (_BILL_DIGEST_RE, 2200)):
        m = rx.search(text)
        if not m:
            continue
        explainer = " ".join(m.group(1).split()).strip()
        # Illinois synopses open "AS INTRODUCED: <statute citations> Amends the …
        # Act. <plain description>". Drop that citation preamble so the plain
        # description leads instead of a wall of "35 ILCS 405/2 from Ch. 120 …".
        explainer = re.sub(r"^AS\s+INTRODUCED:\s*", "", explainer, flags=re.IGNORECASE)
        verb = re.search(
            r"\b(?:Amends|Creates|Provides|Repeals|Adds|Establishes|Requires|"
            r"Authorizes|Prohibits|Permits|Appropriates|Reenacts|Designates)\b",
            explainer,
        )
        if verb and verb.start() > 0 and _is_citation_heavy(explainer[:verb.start()]):
            explainer = explainer[verb.start():]
        # Reject an explainer that is still a citation list or a mangled table —
        # better no explainer than prepending legalese as the plain summary.
        if len(explainer) < 25 or _is_citation_heavy(explainer) or _is_table_gibberish(explainer):
            continue
        return explainer[:cap]
    return ""


def _prepare_full_text_for_llm(text: str) -> str:
    """Reduce extracted bill text to the part that describes what THIS bill does.

    Drops the recited long title of any amended act (pure boilerplate) so the
    operative section — the new requirement, program, ban, or moratorium — leads
    the window the model actually reads, and scrubs the pdftotext arrow/front-
    matter artifacts in case the text came from an older extraction or cache.
    When the bill carries its own plain-English explainer (a trailing STATEMENT,
    a SYNOPSIS, or a digest — often far past the window the model reads), that is
    lifted to the FRONT so the copy is grounded on what the bill actually does,
    not on the recited statute it merely carries forward. Best-effort; returns
    the input lightly cleaned when neither a recitation nor an explainer is
    present."""
    if not text:
        return ""
    text = _strip_mn_redline(text)
    explainer = _extract_bill_explainer(text)
    cleaned = _ARTIFACT_ARROW_RE.sub(" ", text)
    cleaned = _ARTIFACT_FRONTMATTER_RE.sub("", cleaned)
    # Some sources (Alaska's HTML full text) weave per-line numbers through the
    # prose as standalone digit tokens; strip them so neither the model nor the
    # deterministic fallback reads "… to read: 18 (87) … 19 * Sec. 19 …".
    cleaned = _strip_inline_line_numbers(cleaned)
    cleaned = _RECITED_LONGTITLE_RE.sub("known by its short title", cleaned)
    if explainer:
        cleaned = (
            "Plain-language summary of what this bill does "
            f"(written by the bill's own drafters): {explainer}\n\n"
            "Full bill text follows.\n\n" + cleaned
        )
    return cleaned


def _bill_purpose(*texts: str) -> str:
    """The bill's own subject phrase, pulled from its 'An Act … providing for
    <purpose>' / 'relating to <purpose>' enacting clause. Tries each source in
    order (title first, then full text) and returns the first plausible phrase,
    or "" when none is found. Used both to anchor the model and as the headline
    fallback when the model returns nothing."""
    for text in texts:
        if not text:
            continue
        # Drop the recited long title first so its own "providing for …" clauses
        # can't be mistaken for the bill's purpose.
        stripped = _RECITED_LONGTITLE_RE.sub(" ", text)
        # Take the LAST connector match: the bill's purpose is the tail of the
        # enacting clause, after any structural "in <article>," qualifiers.
        m = None
        for m in _PROVIDING_FOR_RE.finditer(stripped[:1500]):
            pass
        if not m:
            continue
        # Stop at the first sentence end / enacting boilerplate, or at a repeated
        # connector ("… and relative to …", "… and concerning …") so a title that
        # strings two subjects together keeps just the first, primary one instead
        # of trailing an awkward "and relative to …" into the headline.
        purpose = re.split(
            r"(?:\.\s|\.$|;|\bbe it enacted\b|\bthe general assembly\b|"
            r"\bto read as follows\b|\bis amended\b|"
            r"\band\s+relat(?:ive|ing)\s+to\b|\band\s+concerning\b)",
            m.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        purpose = " ".join(purpose.split()).strip(" ,;:.")
        # Drop a leading filler word that reads oddly at the start of a headline.
        purpose = re.sub(r"^(?:an?|the|optional)\s+", "", purpose, flags=re.IGNORECASE)
        if 8 <= len(purpose) <= 200:
            return purpose
    return ""


def _delegalese_headline(title: str, full_text: str = "") -> str:
    """A clean, plain-English headline derived deterministically from the bill —
    the safety net for when the model returns no usable headline. Prefers the
    bill's own 'providing for <purpose>' subject over the raw legalese title, so
    the post never leads with 'An Act amending the act of July 31, 1968
    (P.L.805, No.247), known as …'. Returns "" when nothing clean can be found."""
    # The bill's own quoted short-title ("… Act"/"… Law"), when present, is the
    # cleanest possible headline — use it before falling to the purpose clause.
    name = _quoted_act_name(title)
    if name:
        headline = _smart_case(name)
        if len(headline) > HEADLINE_MAX_LEN:
            headline = _smart_truncate(headline, HEADLINE_MAX_LEN)
        return headline.rstrip(".!?,; ")
    purpose = _bill_purpose(title, full_text)
    if not purpose:
        return ""
    headline = _smart_case(purpose)
    if len(headline) > HEADLINE_MAX_LEN:
        headline = _smart_truncate(headline, HEADLINE_MAX_LEN)
    return headline.rstrip(".!?,; ")


# Acronyms kept uppercase when a fallback summary lowercases PA's shouting
# inserted-text markers. Small, general-audience set; everything else is treated
# as an amendment marker and lowercased.
_KNOWN_ACRONYMS = {
    "US", "USA", "AI", "EPA", "FBI", "CIA", "DNA", "NASA", "IRS", "GDP",
    "LLC", "PA", "ID", "TV", "CEO", "DUI", "DMV", "ICE", "LGBTQ",
}


def _summary_from_body(prepared_body: str, max_chars: int = 240) -> str:
    """A plain-language blurb pulled straight from the operative bill text — the
    non-LLM fallback so a post with full text says something concrete rather than
    shipping a bare headline. Conservative: returns "" rather than a citation
    fragment when it can't isolate a clean operative sentence."""
    if not prepared_body:
        return ""
    body = prepared_body
    # Jump to the NEWLY ADDED text: "… to read as follows:" / "… to read:" marks
    # where the new section begins, past the "Section 1 … is amended by adding a
    # section to read:" scaffolding. Fall back to the enacting clause.
    for marker in (r"to read as follows:", r"to read:", r"hereby enacts as follows:"):
        mm = list(re.finditer(marker, body, re.IGNORECASE))
        if mm:
            body = body[mm[-1].end():]
            break
    # Drop leading "Section <id>. <Header>.--" scaffolding and (a)/(1) enumerators
    # so the blurb opens on prose, not a section header.
    body = re.sub(r"^\s*(?:Section\s+[\w.]+\.?\s*)+", "", body.strip(), flags=re.IGNORECASE)
    body = re.sub(r"^[^.]{0,200}?\.--", "", body.strip())
    cleaned = _clean_for_llm(body)
    # Drop leading "(a) " / "(1) " enumerators and stray single ALL-CAPS
    # amendment-marker tokens ("RESOLUTION", "ON") that open PA's tracked text.
    cleaned = re.sub(r"^(?:\(\s*[a-z0-9]{1,3}\s*\)\s*|[A-Z]{2,}\s+)+", "", cleaned).strip()
    if not cleaned:
        return ""
    # Take the first genuinely substantive sentence(s): long enough to be prose,
    # not a bare statute citation ("The act of July 31, 1968 (P.L.805, No.247)").
    citation_re = re.compile(
        r"^(?:the act of|the general assembly|section\b|this act|p\.?\s*l\.)",
        re.IGNORECASE,
    )
    picked: list[str] = []
    # End a clause at sentence punctuation OR a ":" that introduces an
    # enumerated list ("… as follows:"), so the blurb stops at a clean boundary
    # rather than running the whole (a)/(1)/(2) list together.
    for sent in re.findall(r"[^.!?:]*[.!?:]", cleaned):
        s = " ".join(sent.split()).strip()
        # Drop a leading statute enumerator ("(a) ", "(1) ") so a picked clause
        # opens on prose, not a subsection marker.
        s = _LEADING_ENUMERATOR_RE.sub("", s).strip()
        # Lowercase interior ALL-CAPS words — PA prints inserted text in caps,
        # which pdftotext leaves shouting mid-sentence. Preserve genuine acronyms.
        s = re.sub(
            r"\b[A-Z]{2,}\b",
            lambda w: w.group(0) if w.group(0) in _KNOWN_ACRONYMS else w.group(0).lower(),
            s,
        )
        core = s.rstrip(".!?:")
        if len(core) < 30:
            continue
        if citation_re.match(s):
            continue
        # A mangled data-table row (budget/rate schedule) is unreadable noise.
        if _is_table_gibberish(core):
            continue
        letters = [c for c in core if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.5:
            continue
        # Trim a trailing ":" and normalize the opening capital.
        core = core.rstrip(" :;,")
        if core:
            picked.append(core[:1].upper() + core[1:])
        if len(picked) >= 2 or len(" ".join(picked)) >= max_chars:
            break
    if not picked:
        return ""
    out = ". ".join(picked)
    if not out.endswith((".", "!", "?")):
        out += "."
    # A blurb dominated by statute citations / section markers reads as legalese,
    # not plain English — drop it so the post ships a clean headline rather than a
    # citation fragment ("Parental leave fund account (AS 23.10.705). Sec. 19. …").
    # Same for a mangled data-table row (NH HB1738's RGGI budget figures).
    if _is_citation_heavy(out) or _is_table_gibberish(out):
        return ""
    return _smart_truncate(out, max_chars)


def _parse_copy_json(text: str) -> tuple[str, str]:
    """Pull ("headline", "summary") out of the model's JSON reply, tolerating
    the small-model habits of wrapping the object in ```json fences or trailing
    a stray sentence after it. Returns ("", "") when no JSON object is found."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^\s*json\s*", "", text, flags=re.IGNORECASE)
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return "", ""
    return (
        str(obj.get("headline") or "").strip(),
        str(obj.get("summary") or "").strip(),
    )


def _post_copy(b: dict) -> dict:
    """Generate the post's headline AND its plain-English blurb in a SINGLE
    local-LLM call, returned as {"headline": str, "summary": str}.

    The old design ran two independent calls — shorten_title() for the head and
    summarize() for the body — each reading the same bill text, so they kept
    converging on the same sentence ("Establishes regulations regarding AI in
    mental health care" / "The legislature is establishing oversight of AI use
    in mental healthcare…"). Generating both at once lets us instruct the model
    that the summary must ADD detail the headline doesn't already carry, killing
    the repetition by construction (and halving the LLM calls per bill).

    Reads the richest groundable source — the full bill text (a data dump of the
    actual legislation), then an omnibus table-of-contents digest, then the
    abstract or blob title — exactly as the two old functions did, and inherits
    their guards: amendment sheets are dropped, amendatory re-enactments are
    steered onto the bill's stated purpose, and a bill with nothing but a bare
    title is left un-rewritten (no source to ground a rewrite → return both ""
    so the raw title stands and no blurb is invented).

    Cached on the record so the single call happens once even though both
    shorten_title() and summarize() read from it (whichever runs first pays for
    the call; the other reuses the cache). Either field may be "".
    """
    cached = b.get("_post_copy")
    if cached is not None:
        return cached
    result = {"headline": "", "summary": ""}

    title = (b["title"] or "").strip()
    abstract = (b["abstract"] or "").strip()
    blob = _is_blob_title(title)

    # Prefer the real bill body text (extracted from the bill's PDF via
    # bill_text); fall back to the abstract when extraction isn't possible.
    full_text = _get_full_text(b)
    # A floor-amendment sheet isn't the bill body — its one- or two-line change
    # would mislead both the headline and the blurb. Drop it; the action line
    # still reports the amendment.
    if full_text and _is_amendment_doc(full_text):
        full_text = ""

    # A blob title IS the bill description (PR does this for nearly every bill,
    # Missouri sometimes) — use it as the source when no separate abstract ships.
    if not abstract and blob:
        abstract = title

    abstract_usable = bool(abstract) and _normalize(abstract) != _normalize(title)

    # No groundable source beyond a bare title: a small model would have to
    # invent specifics, so skip the call and leave both fields empty. The caller
    # keeps the raw title and drops the blurb (mirrors the old early-outs in
    # shorten_title() and summarize()).
    if not full_text and not blob and not abstract_usable:
        b["_post_copy"] = result
        return result

    # An amendatory repeal-and-reenact bill carries an existing statute section
    # forward mostly unchanged; anchor on the bill's stated "relating to …"
    # purpose so neither field describes the carried-over boilerplate.
    amendatory = bool(full_text) and _is_amendatory_reenactment(full_text)
    act_purpose = _extract_act_purpose(full_text) if amendatory else ""

    # Source text fed to the model: real bill body first (wide window — wider
    # still for amendatory bills whose one new provision is often the final
    # subsection), then the omnibus digest, then the cleaned abstract/title.
    # For full text, first strip the recited long title of any amended act so the
    # operative section — what THIS bill does — leads the window the model reads,
    # instead of a wall of the amended statute's own boilerplate.
    if full_text:
        body = _clean_for_llm(_prepare_full_text_for_llm(full_text))
        # Read up to the full translated window (FULLTEXT_TRANSLATE_MAX_CHARS):
        # whatever we translate to English, the summarizer should be able to see,
        # so the operative provision is never cut off before the model reads it —
        # including an amendatory bill's one new provision at the very end.
        char_cap = FULLTEXT_TRANSLATE_MAX_CHARS
    else:
        source = abstract if abstract_usable else title
        body = _omnibus_digest(source) or _clean_for_llm(source)
        char_cap = 2000
    if not body:
        b["_post_copy"] = result
        return result

    # The bill's own stated subject ("An Act … providing for <purpose>"), used to
    # anchor the model on what the bill does — decisive for amend-by-recitation
    # bills whose operative text is short next to the statute they cite — and as
    # the deterministic headline fallback if the model returns nothing usable.
    bill_purpose = _bill_purpose(title, full_text)
    purpose_note = ""
    if bill_purpose and not amendatory:
        purpose_note = (
            f"The bill's own stated purpose is \"{bill_purpose}\" — center the "
            f"headline and summary on that, not on any statute it merely amends "
            f"or cites.\n\n"
        )

    amendatory_note = ""
    if amendatory:
        amendatory_note = (
            "NOTE: This bill repeals an existing statute section and re-enacts it, "
            "so nearly all of the text below is existing law carried forward "
            "UNCHANGED. Base BOTH the headline and the summary only on what the bill "
            "newly adds or changes, never the carried-over provisions"
        )
        amendatory_note += (
            f" — the bill's stated purpose is \"{act_purpose}\"." if act_purpose else "."
        )
        amendatory_note += "\n\n"

    # The bill's own state, so the prompt can anchor the copy to it and reject
    # the incidental-other-state misattribution (e.g. a California stablecoin
    # bill that recognizes New York crypto licenses being framed "…in New
    # York"). Falls back to the bare code, then "" when the state is unknown.
    state_name = STATE_FULL_NAME.get(b.get("state") or "", b.get("state") or "")
    # Federal bills change national law, not one state's — the "home state"
    # anti-cross-attribution rule below is phrased possessively ("changes
    # <state>'s own law"), which reads wrong for Congress. Flag federal so the
    # system prompt uses a federal-specific variant instead.
    federal = (b.get("state") or "") == "US"
    if federal:
        home_state_line = "This is a federal (U.S. Congress) bill.\n"
    else:
        home_state_line = f"This is a {state_name} bill.\n" if state_name else ""

    # When this bill landed in the feed because of one buried provision (an
    # omnibus bill matching a narrow topic on a single line item), the full
    # text the model reads is dominated by other, unrelated provisions, so the
    # headline/summary drift off-topic for the feed they ran in. Anchor the
    # copy on the exact provision that earned the topic placement so the post
    # justifies why it belongs in this feed. Empty when the title itself
    # carried the topic signal (the whole bill is already on-topic).
    try:
        topic_excerpt = TOPIC.matching_excerpt(b)
    except Exception:
        topic_excerpt = ""
    topic_anchor_note = ""
    if topic_excerpt:
        topic_anchor_note = (
            f"FEED FOCUS: This post runs in a feed about {TOPIC.prompt_topic}. "
            f"The bill was selected for this feed because of this specific "
            f"provision:\n\"{topic_excerpt}\"\n"
            f"Base BOTH the headline and the summary on THAT provision — what it "
            f"does and who it affects. The bill may also cover unrelated "
            f"subjects; do not let them displace this one.\n\n"
        )

    # A generic subject title ("State government.") describes nothing — tell the
    # model to ignore it and build the copy from the provisions below, and don't
    # feed it back as an authoritative "Title:" line.
    vague_title = _is_vague_subject_title(title, b.get("subjects", ""))
    vague_note = ""
    if vague_title:
        vague_note = (
            f"NOTE: The bill's listed title (\"{title}\") is a generic subject "
            f"label that does NOT describe the bill's contents. Ignore it and base "
            f"the headline and summary on the specific provision(s) and text below.\n\n"
        )

    # For blob bills the title is the same wall of legalese as the body, so don't
    # send it as a separate "Title:" line. A vague subject title is likewise
    # useless as an anchor, so omit it too.
    if blob or vague_title:
        user_prompt = f"{amendatory_note}{vague_note}{purpose_note}{topic_anchor_note}{home_state_line}Bill text:\n{body[:char_cap]}"
    else:
        user_prompt = f"{amendatory_note}{purpose_note}{topic_anchor_note}{home_state_line}Title: {title}\nBill text:\n{body[:char_cap]}"

    system_prompt = TOPIC.post_copy_system_prompt(
        max_chars=POST_COPY_MAX_CHARS,
        amendatory=amendatory, home_state=state_name, federal=federal
    )

    def _call_llm(extra_system: str = "", timeout: int = LLM_TIMEOUT) -> tuple[str, str]:
        """One post-copy round-trip → (headline, summary). Raises on transport
        or HTTP error so the caller can fall back."""
        messages = [{"role": "system", "content": system_prompt}]
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        messages.append({"role": "user", "content": user_prompt})
        r = requests.post(
            LLM_API_URL,
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
                "format": "json",
                "keep_alive": LLM_KEEP_ALIVE,
                "options": {"num_predict": 320, "temperature": 0.4},
            },
            timeout=timeout,
        )
        if not r.ok:
            print(f"  ! LLM post-copy {r.status_code}: {r.text[:300]}", file=sys.stderr)
            r.raise_for_status()
        data = r.json()
        # Ollama /api/chat returns {"message": {"content": "..."}, ...}
        # Ollama /api/generate returns {"response": "...", ...}
        text = (data.get("message") or {}).get("content") or data.get("response") or ""
        return _parse_copy_json(text)

    # Up to three attempts total (one initial + two retries) so a transient
    # timeout — e.g. the local model briefly swap-bound on an oversized omnibus
    # bill — doesn't sink the post straight to the raw-legalese fallback. The
    # first attempt gets the full LLM_TIMEOUT (room for a cold or large call); the
    # retries use the shorter LLM_RETRY_TIMEOUT, since a model warm from the first
    # attempt answers in seconds and a still-wedged one should fail fast rather
    # than burn the job's minute budget. The empty/garbled-JSON nudge (a model
    # that DID respond, just unusably) still runs once within each attempt.
    raw_headline, raw_summary = "", ""
    last_exc = None
    for attempt in range(3):
        call_timeout = LLM_TIMEOUT if attempt == 0 else LLM_RETRY_TIMEOUT
        try:
            raw_headline, raw_summary = _call_llm(timeout=call_timeout)
            if not raw_headline and not raw_summary:
                raw_headline, raw_summary = _call_llm(
                    "Return ONLY the JSON object {\"headline\": \"...\", "
                    "\"summary\": \"...\"} — both fields non-empty, plain English.",
                    timeout=call_timeout,
                )
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            if attempt < 2:
                print(f"  ! post-copy attempt {attempt + 1} timed out/failed ({e}); "
                      f"retrying with a {LLM_RETRY_TIMEOUT}s timeout...", file=sys.stderr)
    if last_exc is not None:
        print(f"  ! post-copy generation failed after 3 attempts, using fallback: "
              f"{last_exc}", file=sys.stderr)
        # LLM unreachable: fall back to deterministic, non-legalese copy so the
        # post is still informative. Headline from the bill's stated purpose;
        # blurb from the on-topic provision, then the operative body, then the
        # abstract's first sentence.
        result["headline"] = _delegalese_headline(title, full_text)
        result["summary"] = _strip_title_prefix(
            _excerpt_summary(topic_excerpt)
            or _summary_from_body(body)
            or _first_sentence(abstract or full_text),
            b["title"],
        )
        b["_post_copy"] = result
        return result

    # Headline cleanup mirrors the old shorten_title() tail: strip a trailing
    # period, drop it when the model echoed the raw legalese title, and trim
    # (rather than discard) an over-long one — a slightly long plain-English
    # headline still beats falling back to the raw statute title.
    headline = _clean_summary(raw_headline).rstrip(".!?,; ")
    if headline and _normalize(headline).startswith(_normalize(title)[:60]):
        headline = ""
    if len(headline) > HEADLINE_MAX_LEN:
        headline = _smart_truncate(headline, HEADLINE_MAX_LEN).rstrip(".!?,; ")
    # Did the model give a DISTINCT headline (vs. one we'll derive from the title
    # below)? best_display_text shows that headline in the post head whenever one
    # exists, so with a real model headline the raw title never appears in the
    # head — meaning a title restatement in the body is the ONLY place the bill's
    # name shows, and must be kept. Only when the head itself is title-derived is
    # that restatement a genuine duplicate worth folding away.
    headline_from_model = bool(headline)
    # Still no usable headline (model returned nothing, or echoed the title):
    # derive a clean one from the bill's stated purpose so the post never leads
    # with raw legalese. Only overrides a blank — a good model headline wins.
    if not headline:
        headline = _delegalese_headline(title, full_text)
    result["headline"] = headline

    # Summary cleanup mirrors the old summarize() tail. Belt-and-suspenders: if
    # the model ignored the no-repeat rule and the blurb just restates the
    # headline, drop it (compose_post also guards, but this keeps the cached
    # value clean for every platform that reads it).
    summary = _clean_summary(raw_summary)
    # Only fold a leading title restatement out of the body when the head is
    # title-derived (no distinct model headline). With a real model headline the
    # title shows nowhere else, so keep it here; compose_post's display-aware
    # _strip_act_name_echo / _strip_headline_echo still drop it if it actually
    # duplicates the headline. (Stripping it unconditionally erased the bill's
    # name from posts like HR 9603, whose head showed a distinct headline.)
    if not headline_from_model:
        summary = _strip_title_prefix(summary, b["title"])
    if summary and headline and _normalize(summary) == _normalize(headline):
        summary = ""
    # When the local model returns an empty (or self-repeating, just-dropped)
    # blurb, the post would ship with informative space left unused. Fall back to
    # the on-topic provision (body-matched omnibus bills), then the operative
    # bill text — so a bill with full text always says something concrete rather
    # than shipping a bare headline.
    if not summary:
        summary = _excerpt_summary(topic_excerpt) or _summary_from_body(body)
        if not headline_from_model:
            summary = _strip_title_prefix(summary, b["title"])
        if summary and headline and _normalize(summary) == _normalize(headline):
            summary = ""
    result["summary"] = summary

    b["_post_copy"] = result
    return result


def is_on_topic(b: dict, topic: "Topic | None" = None) -> bool:
    """LLM relevance gate. Keyword matching (TOPIC.matches) is a cheap net that
    can let an omnibus/budget bill through on a single incidental subject tag
    (e.g. a whole state budget matching the AI/crypto feed because it lists
    "CRYPTOCURRENCY & NFTS" among hundreds of subjects). This asks the local
    model to read the actual bill text and confirm the bill is genuinely about
    this feed's topic.

    ``topic`` defaults to the process's active TOPIC (the single-topic posters
    and digests); the all-topics Threads/Instagram digest passes each bill's own
    claiming topic so a bill is judged against the feed it will run in.

    Returns True (post it) unless the model is confident the bill is off-topic.
    Fails OPEN: no groundable text, the gate disabled, or any extraction / LLM /
    parse error all return True, so the keyword match still stands and a gate
    hiccup never silently drops a whole run's posts. Cached on the record."""
    if not RELEVANCE_GATE:
        return True
    t = topic if topic is not None else TOPIC
    cached = b.get("_on_topic")
    if cached is not None:
        return cached

    full_text = _get_full_text(b)
    # An amendment sheet isn't the bill body — don't judge relevance off it.
    if full_text and _is_amendment_doc(full_text):
        full_text = ""
    source = full_text or (b.get("abstract") or "").strip()
    # Nothing substantive to read: the keyword match already vouched for the
    # bill and there's no text to contradict it, so keep it (don't gate).
    if len(source) < 200:
        b["_on_topic"] = True
        return True

    system_prompt = t.relevance_gate_system_prompt()
    user_prompt = (
        f"Topic: {t.prompt_topic}\n"
        f"Bill title: {(b.get('title') or '').strip()}\n"
        f"Bill text:\n{source[:3500]}"
    )
    result = True
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
                "format": "json",
                "keep_alive": LLM_KEEP_ALIVE,
                "options": {"num_predict": 80, "temperature": 0.2},
            },
            timeout=RELEVANCE_GATE_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content") or data.get("response") or ""
        verdict = json.loads(text).get("on_topic", True)
        if isinstance(verdict, str):
            verdict = verdict.strip().lower() not in ("false", "no", "0", "")
        # Skip only on an explicit, parseable "off-topic" verdict.
        result = bool(verdict)
    except Exception as e:
        print(f"  relevance gate error ({e}); keeping bill", file=sys.stderr)
        result = True

    b["_on_topic"] = result
    return result


def summarize(b: dict, max_chars: int = 240) -> str:
    """Plain-English blurb for the post body. Generated together with the
    headline in a single local-LLM call (see _post_copy) so the two never
    restate each other. max_chars bounds the returned length; compose_post still
    trims further to fit the whole post."""
    summary = _post_copy(b)["summary"]
    if summary and len(summary) > max_chars:
        summary = _smart_truncate(summary, max_chars)
    return summary


def shorten_title(b: dict) -> str:
    """Plain-English headline for the post head, generated together with the
    blurb in a single local-LLM call (see _post_copy). Returns "" when the title
    is already short enough to use as-is, or when there was no groundable source
    to rewrite from — the caller then keeps the raw title."""
    title = (b["title"] or "").strip()
    if len(title) <= HEADLINE_THRESHOLD:
        return ""
    return _post_copy(b)["headline"]


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
            # The visible text shows a short anchor (LINK_ANCHOR), not the raw
            # URL, so point the facet at the anchor span and link it to the real
            # URL — keeps the post short while still tapping through to the bill.
            # Offsets are byte indices into the UTF-8 text. Fall back to the raw
            # URL if for some reason it appears inline instead.
            tb = text.encode("utf-8")
            anchor = (LINK_PREFIX + LINK_ANCHOR).encode("utf-8")
            marker = LINK_PREFIX.encode("utf-8")
            pos = tb.rfind(anchor)
            if pos >= 0:
                record["facets"] = [{
                    "index": {"byteStart": pos + len(marker), "byteEnd": pos + len(anchor)},
                    "features": [{"$type": "app.bsky.richtext.facet#link", "uri": link_url}],
                }]
            elif link_url in text:
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
    """'HB 1032' -> ('HB', '1032'); 'SCR 1' -> ('SCR', '1'); strips leading zeros.

    Some sources prefix the identifier with committee-substitute or substitute
    markers ('SCS SB 836', 'HCS HB 4798', 'SS#2/SCS/SB 1012'); take the LAST
    letters+digits pair so those prefixes don't break URL building.
    """
    matches = list(re.finditer(r"([A-Za-z]+)\s*0*(\d+)", ident or ""))
    if not matches:
        return ("", "")
    m = matches[-1]
    return (m.group(1).upper(), m.group(2))


def _leading_int(s: str) -> str:
    """'104th' -> '104'; '194' -> '194'; '' -> ''."""
    m = re.match(r"(\d+)", s or "")
    return m.group(1) if m else ""


_RI_BILL_IDENT_RE = re.compile(r"^([HS])B\s*(\d.*)$")


def display_identifier(state: str, identifier: str) -> str:
    """Human-facing bill number for posts/cards.

    Rhode Island officially numbers its bills with a single-letter chamber
    prefix — House bills are ``H####`` and Senate bills are ``S####`` — but the
    OpenStates feed records them as ``HB####``/``SB####``. Collapse the doubled
    letter for RI display only; resolutions (HR/SR/HJR/SJR) keep their codes.

    The canonical ``identifier`` is left untouched everywhere it matters for
    machine matching — dedup keys, the URL builders, on-disk source paths, and
    no-match diagnostics — so this only affects what a reader sees.
    """
    ident = (identifier or "").strip()
    if (state or "").upper() != "RI":
        return ident
    m = _RI_BILL_IDENT_RE.match(ident)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return ident


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


def _b_la(session, ident):  # verified — legis.la.gov BillInfo keys off a session CODE
    # Louisiana's per-bill page is /legis/BillInfo.aspx?s=<code>&b=<TYPE><NUM>,
    # where <code> is NOT the plain year: the 2026 Regular Session is "26RS" and
    # the 2025 First Extraordinary Session is "251ES" (2-digit year + session
    # number + "ES"). OpenStates/govbot hand us a plain year ("2026") for regular
    # sessions and a year+S<n> marker for special ones ("2025S1"), so map both.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    yy = year[-2:]
    m = re.search(r"\d{4}\s*(?:ES|S)\s*(\d+)", session or "", re.IGNORECASE)
    code = f"{yy}{m.group(1)}ES" if m else f"{yy}RS"
    return f"https://www.legis.la.gov/legis/BillInfo.aspx?s={code}&b={typ}{num}"


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


def _b_mo(session, ident):  # best-effort -- LegiScan per-bill page
    # Missouri rebuilt both chambers' trackers around opaque internal numeric
    # bill IDs (BillInformation?billid=NNN), so there's no per-bill URL we can
    # compute from the bill number. The senate bill-tracking search used to
    # resolve a senate bill number to a single-result page, but that
    # `billSearch?...&handler=BillSearch` endpoint is a Razor Pages AJAX handler
    # — wrong for a shareable browser link — and now returns HTTP 503, while it
    # also carries no session year so the link rots after sine die. So route
    # both chambers through LegiScan, which has stable, session-scoped per-bill
    # pages for every MO bill.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
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
    # Resolutions (HCR, SCR, HR, SR, HJR, SJR) live under /resolutions/ on the
    # same domain — /bills/hcr5027/ 404s while /resolutions/hcr5027/ resolves.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    y = int(year)
    if y % 2 == 0:
        y -= 1  # bienniums start in odd years
    next_yy = str(y + 1)[-2:]
    path = "resolutions" if typ.upper() in {"HR", "SR", "HCR", "SCR", "HJR", "SJR"} else "bills"
    return f"https://www.kslegislature.gov/b{y}_{next_yy}/{path}/{typ.lower()}{num}/"


def _b_pr(session, ident):  # official oslpr.org homepage fallback
    # Puerto Rico's official tracker (sutra.oslpr.org) keys every measure by an
    # internal record ID we can't derive from the bill number, and govbot now
    # carries that official sutra.oslpr.org URL in the feed (see
    # _TRUSTED_SOURCE_URL_RE, which link_for prefers). We deliberately do NOT
    # build an openstates.org link here — that's a third-party aggregator, not
    # the official site — so when the feed has no sutra URL we fall through to
    # the official oslpr.org homepage instead.
    return None


def _b_pa(session, ident):  # verified — palegis.us (PA's current site)
    # Pennsylvania retired legis.state.pa.us and moved to www.palegis.us, whose
    # per-bill URL is a clean /legislation/bills/<year>/<typ><num> path (e.g.
    # HB 2499 of 2025 -> .../legislation/bills/2025/hb2499). govbot carries this
    # official URL in the feed too (see _TRUSTED_SOURCE_URL_RE); this builder is
    # the fallback when a record has no source URL. PA identifiers are
    # HB/SB/HR/SR + number; the type goes lowercase in the path.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    if typ[0] not in ("H", "S"):
        return None
    return f"https://www.palegis.us/legislation/bills/{year}/{typ.lower()}{num}"


def _b_ak(session, ident):  # verified — akleg.gov keys the path on the LEGISLATURE number
    # Alaska's bill URL is /basis/Bill/Detail/<leg#>?Root=<TYPE><NUM>, where
    # <leg#> is the Alaska Legislature NUMBER (34 = 2025-2026), NOT the calendar
    # year — the /Detail/2025 form does not load. OpenStates/govbot carry the
    # session as that number already ("34"); if a 4-digit calendar year shows up
    # instead, map it back: the Nth legislature convenes in 1957 + 2*N, so
    # N = (year - 1957) // 2.
    typ, num = _split_ident(ident)
    if not (typ and num):
        return None
    leg = _leading_int(session)
    if leg and len(leg) == 4:
        leg = str((int(leg) - 1957) // 2)
    if not leg:
        return None
    return f"https://www.akleg.gov/basis/Bill/Detail/{leg}?Root={typ}{num}"


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


def _b_nd(session, ident, action_date=""):  # verified — ndlegis.gov assembly bill-overview page
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
    # or year ("69X1", "69s1", "2025S1"). Also: ND's regular session of
    # assembly N meets only Jan–April of the biennium's odd start year
    # (year y), so any action dated in a later year must be a special-session
    # action — use that as a fallback signal when the session string itself
    # carries no explicit special marker (govbot often just emits "69").
    is_special = bool(re.search(r"(?i)special|extra|\d[xs]", raw))
    if not is_special and action_date:
        am = re.match(r"^(\d{4})", action_date)
        if am and int(am.group(1)) != y:
            is_special = True
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
    # RI numbers ALL House measures (bills, resolutions, joint resolutions) in
    # a single H#### sequence served from HouseText, and all Senate measures
    # as S#### from SenateText — the page keys only on the chamber letter and
    # number, not the measure type. So HR/SR/HJR/SJR resolve the same way as
    # HB/SB: e.g. HR 8616 -> .../HouseText26/H8616.htm. Handle any H*/S* type.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    body = typ[0]
    if body not in ("H", "S"):
        return None
    chamber = "House" if body == "H" else "Senate"
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


def _b_ut(session, ident):  # verified — le.utah.gov ~YYYY/bills/static path
    # Utah bill pages live at:
    #   https://le.utah.gov/~<YYYY>/bills/static/<TYPE><NUM>.html
    # NUM is zero-padded: HB/SB to 4 digits (HB0011), resolutions
    # (HJR, SJR, HCR, SCR, HR, SR) to 3 digits (HJR001). The path covers the
    # General Session for the calendar year. Special / extraordinary sessions
    # use different paths that aren't reliably derivable from OpenStates'
    # session encoding, so let those fall back to the legislature homepage.
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    if re.search(r"\d{4}\s*[A-Za-z]", session or ""):
        return None  # special / extraordinary session — drop to homepage
    width = 4 if typ in ("HB", "SB") else 3
    return f"https://le.utah.gov/~{year}/bills/static/{typ}{num.zfill(width)}.html"


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


def _ordinal(n: int) -> str:
    """1 -> '1st', 119 -> '119th', 121 -> '121st', 122 -> '122nd', 123 -> '123rd'."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# congress.gov path slug for each federal bill/resolution type. Keys are the
# uppercased identifier prefix as it arrives from govbot ("HR 1234", "S 1379",
# "HJRES 5", "SCONRES 12"). "HB"/"SB" are accepted too in case a source
# normalizes the chambers that way.
_CONGRESS_TYPE_SLUGS = {
    "HR": "house-bill",
    "HB": "house-bill",
    "S": "senate-bill",
    "SB": "senate-bill",
    "HRES": "house-resolution",
    "SRES": "senate-resolution",
    "HJRES": "house-joint-resolution",
    "SJRES": "senate-joint-resolution",
    "HCONRES": "house-concurrent-resolution",
    "SCONRES": "senate-concurrent-resolution",
}


def _b_us(session, ident):  # verified — congress.gov canonical bill URL
    # Federal bills. govbot's legislative_session for Congress is the plain
    # congress number ("119"); congress.gov URLs spell it as an ordinal
    # ("119th-congress") and slug the chamber/type ("HR 1234" -> house-bill/1234).
    congress = _leading_int(session)
    typ, num = _split_ident(ident)
    if not (congress and typ and num):
        return None
    slug = _CONGRESS_TYPE_SLUGS.get(typ)
    if not slug:
        return None
    return (f"https://www.congress.gov/bill/{_ordinal(int(congress))}-congress/"
            f"{slug}/{num}")


STATE_BILL_URL_BUILDERS = {
    "US": _b_us,
    "CA": _b_ca,
    "FL": _b_fl, "IN": _b_in, "IA": _b_ia, "LA": _b_la, "MI": _b_mi, "NY": _b_ny,
    "MA": _b_ma, "OH": _b_oh, "WI": _b_wi, "NC": _b_nc, "NJ": _b_nj,
    "CT": _b_ct, "MO": _b_mo, "MN": _b_mn, "NM": _b_nm, "HI": _b_hi,
    "KS": _b_ks, "WV": _b_wv, "PA": _b_pa, "PR": _b_pr, "AK": _b_ak, "OR": _b_or,
    "CO": _b_co, "WA": _b_wa, "TN": _b_tn, "RI": _b_ri, "MS": _b_ms,
    "AL": _b_al, "ND": _b_nd, "NH": _b_nh, "DE": _b_de, "ME": _b_me,
    "NE": _b_ne, "SC": _b_sc, "MD": _b_md, "ID": _b_id, "GA": _b_ga,
    "WY": _b_wy, "AR": _b_ar, "VT": _b_vt, "IL": _b_il,
    "KY": _b_ky, "OK": _b_ok, "UT": _b_ut,
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
    "GU": "https://www.guamlegislature.gov/",
    "VI": "https://www.legvi.org/",
    "US": "https://www.congress.gov/",
}

# Source URLs we must never surface as the "read the full bill" link even as a
# last resort — OpenStates API/data endpoints and machine formats, not a page a
# reader can use.
_UNUSABLE_SOURCE_URL_RE = re.compile(
    r"(?i)(?:^https?://(?:\w+\.)*openstates\.org|/api/|\.json($|\?)|ocd-bill/)"
)


# For a few states the feed's raw per-bill URL is a better deep link than
# anything we can reconstruct from (session, identifier). Michigan is the clear
# case: its bill page keys off the calendar YEAR the bill was introduced, which
# for a 2-year session ("2025-2026") can be either year and isn't derivable from
# the bill number — but the feed carries the correct ObjectName URL.
#
# The other entries are states whose official bill pages key off an opaque
# internal id (Illinois' LegId, Delaware's LegislationId, New Hampshire's id,
# Missouri's billid, Maine's LawMakerWeb ID) that OpenStates never exposed — so
# those states fell back to a LegiScan deep link (see _b_il/_b_de/_b_nh/_b_mo/
# _b_me). govbot now carries the official state URL in each action's sources, so
# trust it and only drop to the LegiScan builder when the feed has no official
# link.
#
# Only trust source URLs that match the state's known public bill-page pattern,
# so API, WSDL, bulk-data, or bare-homepage source URLs never leak into a post.
_TRUSTED_SOURCE_URL_RE = {
    "MI": re.compile(
        r"^https?://(?:www\.)?legislature\.mi\.gov/Bills/Bill\?ObjectName=\S+",
        re.IGNORECASE,
    ),
    # Illinois — official ilga.gov bill-status page (carries the LegId we can't derive).
    "IL": re.compile(
        r"^https?://(?:www\.)?ilga\.gov/Legislation/BillStatus\b",
        re.IGNORECASE,
    ),
    # Delaware — legis.delaware.gov BillDetail (keyed on LegislationId).
    "DE": re.compile(
        r"^https?://(?:www\.)?legis\.delaware\.gov/BillDetail\b",
        re.IGNORECASE,
    ),
    # New Hampshire — General Court per-bill page (gc.nh.gov, formerly
    # gencourt.state.nh.us), the billinfo.aspx page keyed on an internal id.
    # Match ONLY billinfo.aspx: the feed sometimes carries the bare
    # advanced.aspx search *form* (no bill number) as the source URL, which is
    # useless to a reader. Rejecting it here lets link_for fall through to
    # _b_nh, which builds a working results.aspx?...&txtbillno=... deep link.
    "NH": re.compile(
        r"^https?://(?:www\.)?(?:gc\.nh\.gov|gencourt\.state\.nh\.us)/bill_status/billinfo\.aspx",
        re.IGNORECASE,
    ),
    # Missouri — House BillContent.aspx and Senate BillTracking BillInformation,
    # both keyed on a year + internal billid.
    "MO": re.compile(
        r"^https?://(?:www\.)?(?:house|senate)\.mo\.gov/\S*bill",
        re.IGNORECASE,
    ),
    # Maine — official LawMakerWeb bill summary (keyed on an internal ID), e.g.
    # legislature.maine.gov/LawMakerWeb/summary.asp?ID=280100956.
    "ME": re.compile(
        r"^https?://(?:www\.)?legislature\.maine\.gov/LawMakerWeb/",
        re.IGNORECASE,
    ),
    # Puerto Rico — official SUTRA tracker (keyed on an internal medida ID we
    # can't derive; _b_pr only knows the oslpr.org homepage).
    "PR": re.compile(
        r"^https?://(?:www\.)?sutra\.oslpr\.org/",
        re.IGNORECASE,
    ),
    # Pennsylvania — current palegis.us bill page (the derivable form the
    # rebuilt _b_pa also targets; trust the exact feed URL, incl. resolutions).
    "PA": re.compile(
        r"^https?://(?:www\.)?palegis\.us/legislation/",
        re.IGNORECASE,
    ),
}


def link_for(b: dict) -> str:
    """
    Build the best available URL for a bill. Prefers a trusted per-bill URL
    carried in the feed for states we can't reconstruct (see
    _TRUSTED_SOURCE_URL_RE), then the per-state deep-link builder, then the
    state's legislature homepage. Returns "" only if the state code is unknown.
    """
    state = (b.get("state") or "").upper()
    session = b.get("session", "")
    identifier = b.get("identifier", "")
    if not state:
        return ""

    trusted = _TRUSTED_SOURCE_URL_RE.get(state)
    if trusted:
        src = (b.get("source_url") or "").strip()
        if src and trusted.match(src):
            return src

    builder = STATE_BILL_URL_BUILDERS.get(state)
    if builder:
        try:
            # ND uses the action date to disambiguate /regular/ vs /special/
            # session paths — see _b_nd.
            if state == "ND":
                url = builder(session, identifier, b.get("action_date", ""))
            else:
                url = builder(session, identifier)
        except Exception:
            url = None
        if url:
            return url

    homepage = STATE_LEGISLATURE_URLS.get(state, "")
    if homepage:
        return homepage

    # No per-state builder and no legislature homepage — this is a territory
    # (e.g. MP, AS) we can't deep-link or even land on a home page. A plausible
    # per-bill/document URL carried in the feed beats shipping no link at all.
    src = (b.get("source_url") or "").strip()
    if src.startswith(("http://", "https://")) and not _UNUSABLE_SOURCE_URL_RE.search(src):
        return src
    return ""


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

# Below this floor the summary block is too short to add useful detail beyond
# the headline — callers skip the LLM round-trip and compose without a summary.
MIN_SUMMARY_CHARS = 60


def summary_budget(b: dict, headline: str, head_cap: int | None = None) -> int:
    """Character budget available for the summary block in a Bluesky post,
    given the head (emoji + state + id + display), the action line, and the
    bill link that share the post. Returned so the caller can ask the LLM for
    a summary that fits cleanly instead of relying on compose_post's post-hoc
    trim — when that trim fires it lops the tail off the model's sentence at a
    word boundary, which usually drops the most concrete clause (the "…include
    stock and…" failure mode). Mirrors x_summary_budget in post_to_x.py.

    head_cap bounds the display length used for budgeting only. The weekly
    digest passes it so a long raw legalese title (when no headline rewrite is
    available) doesn't starve the summary out of the post — the caller pairs
    this with compose_post(prefer_summary=True), which trims that long title to
    fit rather than dropping the summary."""
    emoji = TOPIC.emoji_for(b)
    state_label = b["state"] or "?"
    ident_disp = display_identifier(b["state"], b["identifier"])
    display = best_display_text(b, headline=headline).strip()
    if head_cap is not None and len(display) > head_cap:
        display = display[:head_cap]
    prefix = f"{emoji} {state_label} {ident_disp} — "
    head_len = len(prefix) + len(display)
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block_len = len(f"\n\n{action_line}") if action_line else 0
    link = link_for(b)
    link_block_len = len(f"\n\n{LINK_PREFIX}{LINK_ANCHOR}") if link else 0
    # The summary itself is preceded by "\n\n" (2 chars).
    summary_sep_len = 2
    return MAX_POST - head_len - action_block_len - link_block_len - summary_sep_len


def compose_post(b: dict, summary: str, headline: str = "",
                 prefer_summary: bool = False) -> tuple[str, str, str, str]:
    emoji = TOPIC.emoji_for(b)
    link = link_for(b)
    link_block = f"\n\n{LINK_PREFIX}{LINK_ANCHOR}" if link else ""

    state_label = b["state"] or "?"
    ident_disp = display_identifier(b["state"], b["identifier"])
    display = best_display_text(b, headline=headline).strip()
    summary = (summary or "").strip()
    # Drop a leading act name from the summary when it just echoes the headline
    # ("AI Non-Sentience Act…" appearing in both lines).
    summary = _strip_act_name_echo(summary, display)
    # Drop a whole leading sentence that paraphrases the headline ("Missouri
    # will designate March first as Dr. Mun Choi Day" under a headline that
    # already says exactly that), keeping the substantive follow-on sentence.
    summary = _strip_headline_echo(summary, display)

    summary_block = (
        f"\n\n{summary}"
        if summary and _normalize(summary) != _normalize(display)
        else ""
    )
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block = f"\n\n{action_line}" if action_line else ""

    prefix_len = len(emoji) + len(f" {state_label} {ident_disp} — ")
    head = f"{emoji} {state_label} {ident_disp} — {display}"

    def assemble(h, s, a, l):
        return h + s + a + l

    text = assemble(head, summary_block, action_block, link_block)

    # Weekly digest: the plain-English summary is the whole point of the card,
    # so when a long raw legalese title (no headline rewrite available) would
    # overflow, trim that title first and keep the summary intact. Residual
    # overflow then falls through to the normal cascade below.
    if prefer_summary and len(text) > MAX_POST and summary_block:
        avail = MAX_POST - len(link_block) - len(summary_block) - len(action_block) \
                - prefix_len - 1
        display = _smart_truncate(display, avail + 1) if avail > 0 else ""
        head = f"{emoji} {state_label} {ident_disp} — {display}".rstrip(" —")
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
        head = f"{emoji} {state_label} {ident_disp} — {display_trimmed}".rstrip(" —")
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
    embed_title = f"{state_name} {ident_disp}"[:300]
    embed_desc = (summary or _clean_for_llm(b["abstract"]) or display)[:280]
    return text, link, embed_title, embed_desc


def _split_summary(summary: str, first_budget: int) -> tuple[str, str]:
    """Split a summary into (head, tail) at a sentence boundary so `head` fits in
    `first_budget` chars. `tail` is the remaining sentences (verbatim), "" when
    the whole summary fits. If even the first sentence exceeds the budget it
    becomes `head` on its own (the caller trims as a last resort) so a bill never
    silently loses its lead sentence."""
    summary = (summary or "").strip()
    if len(summary) <= first_budget:
        return summary, ""
    sents = _split_sentences(summary)
    if len(sents) <= 1:
        return summary, ""
    head_parts: list[str] = []
    i = 0
    while i < len(sents):
        candidate = " ".join(head_parts + [sents[i]]).strip()
        if head_parts and len(candidate) > first_budget:
            break
        head_parts.append(sents[i])
        i += 1
    return " ".join(head_parts).strip(), " ".join(sents[i:]).strip()


def compose_thread(b: dict, summary: str, headline: str = "") -> tuple[str, str, str, str, str]:
    """Compose the daily 2-post thread for a bill:

      • post 1 — head (emoji + state + id + display) plus as much of the summary
        as fits MAX_POST;
      • post 2 (self-reply) — the summary's continuation (if any), then the
        action line, then the bill link.

    Returns (post1_text, post2_text, link, embed_title, embed_desc). post2_text
    is "" only when there is no continuation, no action line, AND no link — the
    caller then posts post 1 alone. The link embed rides on post 2."""
    emoji = TOPIC.emoji_for(b)
    link = link_for(b)
    state_label = b["state"] or "?"
    ident_disp = display_identifier(b["state"], b["identifier"])
    display = best_display_text(b, headline=headline).strip()

    summary = (summary or "").strip()
    summary = _strip_act_name_echo(summary, display)
    summary = _strip_headline_echo(summary, display)
    if summary and _normalize(summary) == _normalize(display):
        summary = ""

    prefix = f"{emoji} {state_label} {ident_disp} — "
    head = f"{prefix}{display}"
    # Last-resort: if the head alone overflows, trim the display.
    if len(head) > MAX_POST:
        display = _smart_truncate(display, max(0, MAX_POST - len(prefix)))
        head = f"{prefix}{display}".rstrip(" —")

    # Reserve room for the continuation cues appended below (post 1 ends with
    # CONT_SUFFIX, post 2 opens with CONT_PREFIX) so they never push past MAX_POST.
    suffix_cost = len(f" {CONT_SUFFIX}")
    prefix_cost = len(f"{CONT_PREFIX} ")

    # --- Post 1: head + leading summary that fits MAX_POST ---
    p1_budget = MAX_POST - len(head) - 2 - suffix_cost  # 2 = "\n\n"
    s_head, s_tail = _split_summary(summary, max(0, p1_budget)) if summary else ("", "")
    if s_head and len(head) + 2 + len(s_head) > MAX_POST - suffix_cost:
        # A single lead sentence longer than the post: trim it, and push the
        # trimmed-off remainder into the continuation so nothing is lost.
        keep = _smart_truncate(s_head, max(0, MAX_POST - len(head) - 2 - suffix_cost))
        s_tail = (s_head[len(keep):].strip() + (" " + s_tail if s_tail else "")).strip()
        s_head = keep
    post1 = head + (f"\n\n{s_head}" if s_head else "")

    # --- Post 2: continuation + action line + link ---
    action_line = format_action_line(b["action_desc"], b["action_date"])
    link_block = f"\n\n{LINK_PREFIX}{LINK_ANCHOR}" if link else ""
    action_block = f"\n\n{action_line}" if action_line else ""
    cont_budget = MAX_POST - len(action_block) - len(link_block) - 2 - prefix_cost
    if s_tail and len(s_tail) > cont_budget:
        s_tail = _smart_truncate(s_tail, max(0, cont_budget))
    post2 = f"{s_tail}{action_block}{link_block}" if s_tail else f"{action_line}{link_block}"
    post2 = post2.strip()

    # Continuation cues — only when there is a post 2 to point readers to.
    if post2:
        post1 = f"{post1} {CONT_SUFFIX}"
        post2 = f"{CONT_PREFIX} {post2}"

    state_name = STATE_FULL_NAME.get(b["state"], b["state"] or "Bill")
    embed_title = f"{state_name} {ident_disp}"[:300]
    embed_desc = (summary or _clean_for_llm(b["abstract"]) or display)[:280]
    return post1, post2, link, embed_title, embed_desc


def publish_thread(client: "BlueskyClient | None", b: dict, headline: str) -> bool:
    """Compose and post a bill's 2-post daily thread: post 1 (headline + summary
    lead), then a self-reply carrying the summary continuation + action line +
    bill link (with the link embed card). Uses the DAILY_SUMMARY_CHARS-length
    summary so there's genuinely more to say across the two posts.

    Returns True when post 1 lands (or in dry-run) so the caller marks the bill
    seen; False if post 1 fails so the caller skips it for a later retry. A
    failed continuation is logged but still returns True — post 1 is already
    live, and re-running would duplicate it."""
    summary = summarize(b, max_chars=DAILY_SUMMARY_CHARS)
    post1, post2, link, ec_title, ec_desc = compose_thread(b, summary, headline=headline)

    thumb_blob = None
    if client is not None and link and FETCH_OG_IMAGE:
        fetched = fetch_og_image(link)
        if fetched:
            prepared = prepare_image_for_bluesky(*fetched)
            if prepared:
                thumb_blob = client.upload_blob(*prepared)
                print(f"  IMG: {'✓ attached' if thumb_blob else '✗ upload failed'}")

    print(f"\n--- {b['state'] or '?'} {b['identifier']} ({b['action_date']}) ---")
    print(post1)
    if post2:
        print("  ↳ reply ↴")
        print(post2)
    if link:
        print(f"    link: {link}")
    print("---")

    if client is None:
        return True
    try:
        root = client.post(post1, "", "", "")
        time.sleep(2)
    except requests.HTTPError as e:
        print(f"  ! post failed: {e.response.status_code} {e.response.text}", file=sys.stderr)
        return False
    except requests.RequestException as e:
        print(f"  ! post failed (network): {e}", file=sys.stderr)
        return False
    print(f"  posted: {root.get('uri', '')}")

    if post2:
        ref = {"uri": root["uri"], "cid": root["cid"]}
        try:
            rep = client.post(post2, link, ec_title, ec_desc, thumb_blob=thumb_blob,
                              reply={"root": ref, "parent": ref})
            print(f"  ↳ reply posted: {rep.get('uri', '')}")
        except requests.RequestException as e:
            # The thread already has post 1; don't fail the bill over the reply.
            print(f"  ! reply failed (post 1 is live): {e}", file=sys.stderr)
    return True


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


def _artifact_stem(b: dict) -> str:
    """Shared <STATE>-<id>-<date>-<action_slug> stem so a bill's raw record and
    its full-text file share a filename (just different extensions)."""
    state = (b.get("state") or "XX")
    # Identifier keeps original case (HB2763, SR 008 → HB2763, SR_008) so
    # the filename matches how the bill is shown in the post.
    ident_raw = (b.get("identifier") or "unknown").strip()
    ident = _FILENAME_UNSAFE_RE.sub("_", ident_raw).strip("_")[:24] or "unknown"
    date = b.get("action_date") or "no-date"
    action_slug = _slug(b.get("action_desc") or "no-action", max_len=40) or "no-action"
    return f"{state}-{ident}-{date}-{action_slug}"


def _stash_posted(b: dict, text: str | None = None, link: str | None = None,
                  post_url: str | None = None) -> None:
    """Attach the exact composed post text, the bill link, and the URL of the
    published post to the record so save_raw_record persists them inside
    bills_raw/<...>.json. This is what lets the GitHub Pages dashboard show each
    post exactly as posted — the model-written headline, the "Read the full
    bill" link, and a link to the post itself — instead of reconstructing an
    approximation from the raw bill fields.

    Purely additive and never raises: capturing display copy must never be able
    to interfere with posting."""
    try:
        raw = b.get("_raw")
        if isinstance(raw, dict):
            if text:
                raw["posted_text"] = text
            if link:
                raw["posted_link"] = link
            if post_url:
                raw["posted_url"] = post_url
    except Exception:
        pass


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
    fname = f"{_artifact_stem(b)}.json"
    if out_dir is None:
        out_dir = TOPIC.bills_raw_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    out_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")


def save_full_text(b: dict, out_dir: Path | None = None) -> None:
    """Write the extracted full bill text (when available) to
    topics/<name>/bills_full_text/<STATE>-<id>-<date>-<action_slug>.txt so the
    real legislative body of every posted bill is visible as a plain-text file,
    not buried inside JSON. No-op when no full text was extracted (e.g. the
    bill had no PDF link or pdftotext was unavailable)."""
    full_text = b.get("full_text")
    if not full_text:
        return
    fname = f"{_artifact_stem(b)}.txt"
    if out_dir is None:
        out_dir = TOPIC.bills_full_text_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text(full_text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _norm_ident(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


_IDENT_PREFIX_RE = re.compile(r"^([A-Z]+)")
_IDENT_SPLIT_RE = re.compile(r"^([A-Z]+)(\d*)")


def _ident_prefix(ident: str) -> str:
    """Leading alpha chars of a bill identifier (e.g. 'SB' from 'SB3392').
    Returns '' when the ident has no leading letters."""
    m = _IDENT_PREFIX_RE.match(_norm_ident(ident))
    return m.group(1) if m else ""


def _ident_sort_key(ident: str) -> tuple:
    """Natural-sort key so SB1 < SB2 < SB10 < SB100 (rather than the
    string sort that interleaves them as SB1, SB10, SB100, SB2)."""
    norm = _norm_ident(ident)
    m = _IDENT_SPLIT_RE.match(norm)
    if not m:
        return ("", 0, norm)
    prefix = m.group(1)
    num = int(m.group(2)) if m.group(2) else 0
    return (prefix, num, norm)


def format_no_match_error(
    state: str,
    target_ident: str,
    state_matches: list[dict],
    source_filename: str,
    raw_records: list[dict] | None = None,
) -> None:
    """Diagnostic for force-mode when no record matches the requested bill.

    Default sorting of the seen-identifier preview was alphabetical, which
    buried the requested bill behind thousands of amendment records (IL's
    AM1030xxx etc.). This surfaces:
      - totals and a per-chamber-prefix breakdown,
      - the numeric range of bills sharing the typed prefix and a
        +/- 50 numeric neighborhood window so an operator can see whether
        the requested number is above the clone's max or sitting in a gap,
      - and (if raw_records is provided) a check for the typed identifier
        in the raw JSONL before extract_fields() filters records lacking a
        title or action data, so a bill that exists in govbot's clone but
        has no qualifying log entries doesn't read as "missing"."""
    print(
        f"ERROR: no bill matching state={state!r} "
        f"identifier={target_ident!r} in {source_filename}.",
        file=sys.stderr,
    )

    norm_target = _norm_ident(target_ident)
    target_prefix = _ident_prefix(target_ident)
    target_key = _ident_sort_key(target_ident)
    target_num = target_key[1]

    if raw_records is not None:
        raw_idents: set[str] = set()
        state_upper = state.upper()
        for r in raw_records:
            if detect_state(r) != state_upper:
                continue
            bill = r.get("bill") or {}
            raw_id = bill.get("identifier") or r.get("id") or ""
            if raw_id:
                raw_idents.add(raw_id)
        target_in_raw = next(
            (i for i in raw_idents if _norm_ident(i) == norm_target),
            None,
        )
        if target_in_raw:
            print(
                f"  NOTE: {target_in_raw!r} IS present in {source_filename}, "
                f"but extract_fields() dropped every record for it. Likely "
                f"empty bill.title or no log entries with action.description "
                f"/ action.date yet.",
                file=sys.stderr,
            )

    if not state_matches:
        print(f"  no records at all for state {state} in {source_filename}.",
              file=sys.stderr)
        return

    all_idents = {b["identifier"] for b in state_matches}
    print(f"  {state.upper()} has {len(all_idents)} distinct identifiers "
          f"across {len(state_matches)} records.", file=sys.stderr)

    prefix_counts = Counter(_ident_prefix(i) for i in all_idents)
    breakdown = ", ".join(
        f"{p or '(none)'}={n}"
        for p, n in prefix_counts.most_common()
    )
    print(f"  prefix breakdown: {breakdown}", file=sys.stderr)

    if not target_prefix:
        sorted_idents = sorted(all_idents, key=_ident_sort_key)
        preview = ", ".join(sorted_idents[:20])
        more = ("" if len(sorted_idents) <= 20
                else f" (+{len(sorted_idents) - 20} more)")
        print(f"  identifiers seen (first 20): {preview}{more}",
              file=sys.stderr)
        return

    same_prefix = [i for i in all_idents if _ident_prefix(i) == target_prefix]
    if not same_prefix:
        print(f"  no identifiers starting with {target_prefix!r} in "
              f"{state.upper()}.", file=sys.stderr)
        return

    same_prefix_sorted = sorted(same_prefix, key=_ident_sort_key)
    nums = [n for n in (_ident_sort_key(i)[1] for i in same_prefix) if n > 0]
    if nums:
        lo, hi = min(nums), max(nums)
        print(f"  {target_prefix} range: {target_prefix}{lo} – "
              f"{target_prefix}{hi} ({len(same_prefix)} distinct)",
              file=sys.stderr)
        if target_num > 0 and target_num > hi:
            print(f"  ! {target_ident} is above the clone's highest "
                  f"{target_prefix} ({target_prefix}{hi}) — likely a "
                  f"different session/GA than what govbot cloned.",
                  file=sys.stderr)

    if target_num > 0:
        window = sorted(
            (i for i in same_prefix
             if abs(_ident_sort_key(i)[1] - target_num) <= 50),
            key=_ident_sort_key,
        )
        if window:
            print(f"  {target_prefix} numbers within ±50 of "
                  f"{target_ident}: {', '.join(window)}", file=sys.stderr)
        else:
            print(f"  no {target_prefix} bills within ±50 of "
                  f"{target_ident} — sits in a gap.", file=sys.stderr)

    preview = ", ".join(same_prefix_sorted[:10])
    more = ("" if len(same_prefix_sorted) <= 10
            else f" (+{len(same_prefix_sorted) - 10} more)")
    print(f"  {target_prefix} identifiers (first 10): {preview}{more}",
          file=sys.stderr)


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
    seen = set(state.get("posted", []))
    if not FORCE_REPOST and (b["dedup_key"] in seen or b["same_day_key"] in seen):
        print(
            f"Bill {b['state']} {b['identifier']} action "
            f"{b['action_date']!r} (or another action for this bill on the "
            f"same day) is already in {STATE_FILE.name}. "
            f"Pass force_repost=true to re-post."
        )
        return 0

    if not TOPIC.matches(b):
        print(
            f"  NOTE: bill does not match topic '{TOPIC.name}' keywords — "
            f"posting anyway because force mode was requested."
        )

    print(f"Force-posting 1 bill to topic '{TOPIC.name}':")
    print(f"  {b['state']} {b['identifier']} ({b['action_date']})  "
          f"dedup_key={b['dedup_key']}")

    client = None if DRY_RUN else BlueskyClient(BSKY_HANDLE, BSKY_PASSWORD)

    ensure_english_fields(b)
    # Headline first so the summary's character budget can reserve the exact
    # head length, then ask the model for a summary that fits the leftover
    # space instead of writing 240 chars that compose_post truncates mid-clause.
    # Headline + fuller summary, posted as a 2-post thread (post 1 = headline +
    # summary lead; self-reply = summary continuation + action line + bill link).
    headline = shorten_title(b)
    if not publish_thread(client, b, headline):
        return 1

    if SAVE_RAW:
        try:
            save_raw_record(b)
            save_full_text(b)
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
        print(f"\nDone. State saved to {STATE_FILE.relative_to(ROOT)}.")
    else:
        print(f"\nDone. SAVE_STATE=0 — {STATE_FILE.relative_to(ROOT)} "
              f"left unchanged.")
    return 0


def main() -> int:
    if not DRY_RUN and (not BSKY_HANDLE or not BSKY_PASSWORD):
        print(f"ERROR: {TOPIC.bluesky_handle_env()} and "
              f"{TOPIC.bluesky_password_env()} must be set.", file=sys.stderr)
        return 1

    # The forced-bill path wants the raw, unnormalized records (it does its own
    # lookup by state/id), so load them directly and skip the normalized cache.
    if FORCE_STATE and FORCE_BILL_ID:
        return _post_forced_bill(load_bills(JSONL_PATH))

    # extract_fields() has already been applied (once per shard when a prebuilt
    # cache is used); each record here is a normalized dict carrying its source
    # record under "_raw".
    normalized = load_normalized_bills()
    if not normalized:
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
    for b in normalized:
        if not TOPIC.matches(b):
            continue
        same_day_siblings.setdefault(b["same_day_key"], set()).add(b["dedup_key"])
        # Skip if we've already posted this exact action (dedup_key) OR any
        # other action for this same bill on this same day (same_day_key).
        # The same_day_key guard is what stops a second post when another log
        # entry for the same bill+day arrives on a later run.
        if b["dedup_key"] in seen or b["same_day_key"] in seen:
            continue
        candidates.append(b)

    # Freshness gate: a state's newest *unposted* match can genuinely be a
    # year-old action (part-time legislatures, niche topics). Posting that as
    # news is misleading, so drop anything past the age cap. Mirrors
    # weekly_digest_bluesky.in_lookback_window.
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

    # Backfill reserve: the rest of the candidate pool in weighted order, so a
    # bill the relevance gate skips is replaced by the next best candidate rather
    # than shrinking the run. to_post (the state-spread picks) stays first; the
    # reserve only fills slots the gate frees up.
    picked_ids = {b["dedup_key"] for b in to_post}
    reserve = weighted_draw([b for b in descriptive if b["dedup_key"] not in picked_ids], 10**9)
    reserve += weighted_draw([b for b in stubs if b["dedup_key"] not in picked_ids], 10**9)
    ordered = to_post + reserve

    distinct_states = len({b["state"] or "?" for b in to_post})
    print(f"Pool: {len(descriptive)} state(s) with descriptive bills, "
          f"{len(stubs)} stub-only.")
    print(f"Will post up to {POST_LIMIT}: {len(to_post)} primary picks + "
          f"{len(reserve)} in reserve for gate-skips (from {distinct_states} state(s)).")

    client = None if DRY_RUN else BlueskyClient(BSKY_HANDLE, BSKY_PASSWORD)

    posted = 0
    for b in ordered:
        if posted >= POST_LIMIT:
            break
        ensure_english_fields(b)
        # Relevance gate: keyword matching picked this bill, but confirm with the
        # local model that it's genuinely on-topic before posting (drops omnibus
        # budgets that matched on one incidental subject tag). Fails open. A skip
        # pulls the next reserve candidate so the run still reaches POST_LIMIT.
        if not is_on_topic(b):
            print(f"  ⤫ relevance gate: off-topic for '{TOPIC.name}', "
                  f"skipping {b['state'] or '?'} {b['identifier']}")
            continue
        # Headline + fuller summary, posted as a 2-post thread (post 1 = headline
        # + summary lead; self-reply = summary continuation + action line + bill
        # link). publish_thread handles network/HTTP errors and returns False so
        # a failed bill is skipped (not marked seen) for a later retry.
        print(f"    same_day_key: {b['same_day_key']}")
        headline = shorten_title(b)
        if not publish_thread(client, b, headline):
            continue

        # Posted (or previewed in dry-run) — count it toward POST_LIMIT so
        # gate-skipped bills are backfilled rather than shrinking the run.
        posted += 1

        if SAVE_STATE:
            seen.add(b["dedup_key"])
            # Remember the bill+day itself, plus every sibling action for that
            # bill+day we already know about, so this bill can't be posted
            # again today — even if a new same-day action shows up next run.
            seen.add(b["same_day_key"])
            seen.update(same_day_siblings.get(b["same_day_key"], ()))
            last_posted[b["state"] or "?"] = now.isoformat()
        if SAVE_RAW:
            try:
                save_raw_record(b)
                save_full_text(b)
            except OSError as e:
                print(f"  ! raw-record save failed: {e}", file=sys.stderr)


    if not SAVE_RAW:
        print("  SAVE_RAW=0 — bills_raw artifacts not written.")

    if SAVE_STATE:
        state["posted"] = sorted(seen)
        state["state_last_posted"] = last_posted
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"\nDone. State saved to {STATE_FILE.relative_to(ROOT)}.")
    else:
        print(f"\nDone. SAVE_STATE=0 — {STATE_FILE.relative_to(ROOT)} "
              f"left unchanged.")
    return 0


if __name__ == "__main__":
    # `--build-normalized [out]` parses bills.jsonl once and writes the
    # pre-normalized list (extract_fields applied, source kept as _raw) so the
    # per-topic processes that follow can skip that work. Topic-independent, but
    # importing this module still requires BOT_TOPIC — the workflow sets any
    # valid topic for this step.
    if len(sys.argv) >= 2 and sys.argv[1] == "--build-normalized":
        out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else (ROOT / "bills_normalized.json")
        normalized = build_normalized(load_bills(JSONL_PATH))
        out_path.write_text(json.dumps(normalized), encoding="utf-8")
        print(f"Wrote {len(normalized)} normalized records to {out_path}")
        sys.exit(0)
    # Daily Bluesky poster: ask the model for a fuller summary so the 2-post
    # thread (post 1 lead + self-reply continuation) has more to say. Set here,
    # not at module import, so the weekly digest — which imports this module and
    # summarizes to its own tight per-card budget — keeps the default length.
    POST_COPY_MAX_CHARS = DAILY_SUMMARY_CHARS
    sys.exit(main())
