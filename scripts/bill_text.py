#!/usr/bin/env python3
"""
Full bill-text extraction for the govbot-bluesky pipeline.

The records govbot dumps into ``bills.jsonl`` only carry *metadata* (title,
abstract, sponsors, action descriptions). This module bridges that gap to the
bill's actual legislative body, given a record's ``sources.bill`` path:

1. **Preferred — govbot's pre-extracted text.** Current govbot runs its own
   text extraction and commits the result into each bill's sibling ``files/``
   directory (``{bill_id}_text_extracted.txt``), so a ``govbot clone`` already
   ships clean bill text. When that file is present we read it straight off
   disk — no network, no ``pdftotext``, and none of the PDF-download/extract
   failures that used to leave a bill without a grounded summary.
2. **Fallback — download + extract.** For older clone layouts that lack the
   ``files/`` directory, it locates the document link inside ``metadata.json``
   (``versions[].links[]`` with ``media_type: "application/pdf"``), downloads
   it, and extracts clean plain text via ``pdftotext`` (poppler).

This is the downstream Python implementation of the idea behind upstream issue
chihacknight/govbot#31 ("Extract full bill text from PDFs for RAG"). It runs
only for the handful of bills that survive the topic-keyword filter and the
post draw, so it never downloads thousands of PDFs.

Every function degrades gracefully: any failure (missing file, no PDF link,
``pdftotext`` not installed, network error, empty output) returns ``None`` and
the caller falls back to the existing abstract-only behavior.

Standalone use (verification aid only, not a documented feature):

    python scripts/bill_text.py <sources.bill relative path>
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# Cache extracted text keyed on the resolved document URL, so two
# states/sessions pointing at the same document share a hit and re-runs reuse
# prior extractions. Lives outside the per-topic tree (text is topic-agnostic).
BILL_TEXT_CACHE_DIR = ROOT / ".bill_text_cache"

PDF_MAX_DOWNLOAD = 25 * 1024 * 1024   # hard cap on downloaded document size
TEXT_MAX_CHARS = 200_000              # safety cap on extracted/cleaned text
PDF_FETCH_TIMEOUT = 30               # seconds
PDFTOTEXT_TIMEOUT = 60               # seconds

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Ordered candidate base directories for resolving a ``sources.bill`` path like
# "il-legislation/country:us/state:il/sessions/104th/bills/SB1696/metadata.json".
# The current govbot CLI clones into ``<cwd>/govbot_data/repos`` (the govbot-bills
# action runs from the repo checkout, so that resolves to ``ROOT/govbot_data/
# repos/**/bills/*/metadata.json``); older versions used ``~/.govbot/repos``.
# Both are tried, newest layout first. Override with the GOVBOT_DATA_ROOT env
# var. This is the single most fragile assumption in the module, so it is
# deliberately forgiving.
_HOME = Path.home()


def _candidate_bases() -> list[Path]:
    bases: list[Path] = []
    env_root = os.environ.get("GOVBOT_DATA_ROOT", "").strip()
    if env_root:
        bases.append(Path(env_root).expanduser())
    bases += [
        # Current govbot: ./govbot_data/repos relative to where `govbot clone`
        # ran (the repo checkout in CI, i.e. ROOT).
        ROOT / "govbot_data" / "repos",
        ROOT / "govbot_data",
        Path("govbot_data") / "repos",
        Path.cwd() / "govbot_data" / "repos",
        # Older govbot layout.
        _HOME / ".govbot" / "repos",
        _HOME / ".govbot",
        ROOT,
        Path("."),
        ROOT / ".govbot" / "repos",
        ROOT / ".govbot",
        ROOT / "data",
        ROOT / "legislation",
    ]
    return bases


# Cached base dir discovered at runtime (the first that resolved a real path),
# so we don't re-probe every candidate for every bill.
_discovered_base: Path | None = None

# In-process memo so the same document is never re-read (even from disk) twice
# within one run. Maps resolved URL -> (extracted text or None, reason).
_run_memo: dict[str, tuple[str | None, str]] = {}


# ---------------------------------------------------------------------------
# Prerequisite check ("doctor")
# ---------------------------------------------------------------------------

def pdftotext_available() -> bool:
    """True when the ``pdftotext`` binary (poppler-utils) is on PATH."""
    return shutil.which("pdftotext") is not None


# ---------------------------------------------------------------------------
# Path + link resolution
# ---------------------------------------------------------------------------

def resolve_metadata_path(sources_bill: str, root: Path = ROOT) -> Path | None:
    """Resolve a record's ``sources.bill`` relative path to an on-disk
    ``metadata.json``. Tries each candidate base directory and returns the
    first that exists, plus a fallback that drops the leading path segment
    (covers layouts where the ``<state>-legislation/`` prefix isn't on disk).
    The base that first resolves is cached for subsequent calls. Returns
    ``None`` if nothing resolves."""
    global _discovered_base
    if not sources_bill:
        return None

    rel = sources_bill.lstrip("/")

    # Caller-supplied root (used by tests) takes priority, then the cached
    # discovered base, then the standard candidate list.
    bases: list[Path] = []
    if root != ROOT:
        bases.append(root)
    if _discovered_base is not None:
        bases.append(_discovered_base)
    for b in _candidate_bases():
        if b not in bases:
            bases.append(b)

    parts = Path(rel).parts
    trimmed = Path(*parts[1:]) if len(parts) > 1 else None

    candidates: list[tuple[Path, Path]] = []  # (base, full path)
    for base in bases:
        candidates.append((base, base / rel))
    candidates.append((Path(sources_bill), Path(sources_bill)))  # as-is
    if trimmed is not None:
        for base in bases:
            candidates.append((base, base / trimmed))

    for base, c in candidates:
        try:
            if c.is_file():
                if root == ROOT:
                    _discovered_base = base
                return c
        except OSError:
            continue
    return None


def find_extracted_text_file(metadata_path: Path) -> Path | None:
    """Locate govbot's pre-extracted full text sitting next to ``metadata.json``.

    The current govbot architecture runs its own text extraction and commits the
    result into each bill's sibling ``files/`` directory, so a ``govbot clone``
    already ships clean bill text — no PDF download or ``pdftotext`` needed. The
    main bill body is ``{bill_id}_text_extracted.txt``; amendment versions are
    ``{bill_id}_{amendment_id}_extracted.txt`` (both end in ``_extracted.txt``).
    metadata.json carries no pointer to these, so we locate them by convention.

    Returns the best (main-body-first) extracted-text Path, or ``None`` when the
    ``files/`` dir or an extracted file is absent (older clone layout → caller
    falls back to the PDF-download path)."""
    files_dir = metadata_path.parent / "files"
    if not files_dir.is_dir():
        return None
    try:
        candidates = sorted(p for p in files_dir.glob("*_extracted.txt") if p.is_file())
    except OSError:
        return None
    if not candidates:
        return None
    # Prefer the canonical whole-bill text over amendment fragments: the exact
    # "{bill_id}_text_extracted.txt", then any "*_text_extracted.txt", then the
    # first file alphabetically as a last resort.
    bill_id = metadata_path.parent.name
    exact = files_dir / f"{bill_id}_text_extracted.txt"
    if exact.is_file():
        return exact
    for c in candidates:
        if c.name.endswith("_text_extracted.txt"):
            return c
    return candidates[0]


def find_document_link(metadata_path: Path) -> tuple[str, str] | None:
    """Parse a bill ``metadata.json`` and return ``(url, media_type)`` for the
    best document link: a PDF if present, otherwise the first HTML link.
    Returns ``None`` on any parse error or when no usable link exists."""
    import json
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None

    pdf_link: tuple[str, str] | None = None
    html_link: tuple[str, str] | None = None

    # OCD data carries document links under versions[] (bill text) and
    # sometimes documents[] (fiscal notes, analyses). Prefer versions, but
    # fall back to documents so HTML/PDF bodies stored there still resolve.
    containers = (meta.get("versions") or []) + (meta.get("documents") or [])
    for container in containers:
        if not isinstance(container, dict):
            continue
        for link in (container.get("links") or []):
            if not isinstance(link, dict):
                continue
            url = (link.get("url") or "").strip()
            if not url:
                continue
            mtype = (link.get("media_type") or "").strip().lower()
            if mtype == "application/pdf" or url.lower().endswith(".pdf"):
                if pdf_link is None:
                    pdf_link = (url, "application/pdf")
            elif (mtype in ("text/html", "application/xhtml+xml")
                  or url.lower().endswith((".htm", ".html"))) and html_link is None:
                html_link = (url, "text/html")

    return pdf_link or html_link


def has_local_full_text(sources_bill: str, root: Path = ROOT) -> bool:
    """Cheap, network-free probe: True when a bill's full text is readily
    available without a download — a govbot pre-extracted
    ``files/<id>_text_extracted.txt`` sits beside its metadata, or the
    ``metadata.json`` carries a document link. Only filesystem/JSON reads, so it
    is safe to call for every candidate during bill selection (unlike
    ``extract_bill_text_verbose``, which may download a PDF and run pdftotext).

    Best-effort: returns ``False`` on an empty ``sources_bill`` or any resolution
    failure. The pre-extracted file is the strong signal (it becomes reason
    ``ok-extracted-file`` with no network); a bare document link is a weaker
    signal (extraction could still fail), but both mean full text is *likely* to
    be used, which is what the selection priority wants."""
    if not sources_bill:
        return False
    meta = resolve_metadata_path(sources_bill, root=root)
    if meta is None:
        return False
    ext = find_extracted_text_file(meta)
    if ext is not None:
        try:
            # A non-trivially sized pre-extracted file will clean to real text.
            if ext.stat().st_size > 100:
                return True
        except OSError:
            pass
    return find_document_link(meta) is not None


# ---------------------------------------------------------------------------
# Download + extraction
# ---------------------------------------------------------------------------

def _requests_get_lenient(url: str, **kwargs):
    """GET that retries once without TLS verification on SSL errors. Mirrors
    the helper in post_to_bluesky.py; duplicated here to keep this module
    self-contained and free of import cycles."""
    try:
        return requests.get(url, **kwargs)
    except requests.exceptions.SSLError:
        kwargs2 = dict(kwargs)
        kwargs2["verify"] = False
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        return requests.get(url, **kwargs2)


def _download(url: str) -> Path | None:
    """Download ``url`` to a temp file, aborting past PDF_MAX_DOWNLOAD.
    Returns the temp file path or ``None`` on failure. Supports ``file://``
    URLs (and bare local paths) so extraction can be tested offline."""
    # Local file shortcut (file:// or an existing local path).
    if url.startswith("file://"):
        local = Path(url[len("file://"):])
        return local if local.is_file() else None
    if "://" not in url:
        local = Path(url)
        return local if local.is_file() else None

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = _requests_get_lenient(url, headers=headers, timeout=PDF_FETCH_TIMEOUT, stream=True)
        r.raise_for_status()
        fd, tmp_name = tempfile.mkstemp(suffix=".doc")
        size = 0
        with os.fdopen(fd, "wb") as fh:
            for chunk in r.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                size += len(chunk)
                if size > PDF_MAX_DOWNLOAD:
                    fh.close()
                    os.unlink(tmp_name)
                    print(f"  TEXT: ✗ document too large (>{PDF_MAX_DOWNLOAD//1024//1024} MB), skipping")
                    return None
                fh.write(chunk)
        return Path(tmp_name)
    except Exception as e:
        print(f"  TEXT: ✗ download failed: {e}")
        return None


def _extract_pdf(pdf_path: Path) -> str | None:
    """Run ``pdftotext`` (default mode — no -layout, which mangles prose) and
    return stdout, or ``None`` on failure/empty output."""
    if not pdftotext_available():
        print("  TEXT: ✗ pdftotext not installed (install poppler-utils)")
        return None
    try:
        proc = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=PDFTOTEXT_TIMEOUT,
        )
    except (subprocess.SubprocessError, OSError) as e:
        print(f"  TEXT: ✗ pdftotext failed: {e}")
        return None
    if proc.returncode != 0:
        print(f"  TEXT: ✗ pdftotext exit {proc.returncode}: {proc.stderr[:200]}")
        return None
    out = (proc.stdout or "").strip()
    return out or None


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)


def _extract_html(html_path: Path) -> str | None:
    """Minimal tag-strip fallback for HTML-only document links."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    html = _HTML_SCRIPT_STYLE_RE.sub(" ", html)
    text = _HTML_TAG_RE.sub(" ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">"))
    text = text.strip()
    return text or None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Line-stripping heuristics adapted in spirit from the OpenStates project
# (openstates/openstates-scrapers), MIT License — legislative bills number
# every line and carry running page headers/footers that add noise for an LLM.
_LEADING_LINE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s+")
_PAGE_MARKER_RE = re.compile(r"^\s*(page\s+\d+\s+of\s+\d+|[-–]?\s*\d{1,4}\s*[-–]?)\s*$",
                             re.IGNORECASE)
_ALLCAPS_BANNER_RE = re.compile(r"^[A-Z0-9 .,'\"&/()-]{4,}$")

# pdftotext pulls in the arrow markers some states (Pennsylvania most notably)
# print in the margin to flag inserted/amended text — they surface as "<--" or a
# "<-" glued to the next word ("<-moratorium"). They're pure layout noise that
# fragments the prose an LLM reads, so strip the arrow token wherever it appears.
_AMEND_ARROW_RE = re.compile(r"<-{1,}")

# Minnesota prints its bills as a redline: inserted statute text is wrapped in
# "new text begin … new text end" and deleted statute text in "deleted text
# begin … deleted text end". Extraction pulls these marker words inline, so a
# naive summary reads "new text end new text begin Final billing for … new text
# end" (observed on MN SF 4171). Clean them so only readable prose remains:
#   * DELETED spans are removed entirely, content and all — the bill is taking
#     that language OUT, so it must not appear in the summary. Non-greedy +
#     DOTALL so one span never swallows to a later "deleted text end", and so a
#     span that straddles line breaks is still matched.
#   * NEW-text markers are stripped but the wrapped text is kept — that is the
#     new law the bill adds. Delete spans first (a single line can carry a
#     deleted span immediately followed by a new one).
_MN_DELETED_RE = re.compile(
    r"deleted\s+text\s+begin\b.*?deleted\s+text\s+end\b", re.IGNORECASE | re.DOTALL
)
_MN_NEWTEXT_MARK_RE = re.compile(r"new\s+text\s+(?:begin|end)\b", re.IGNORECASE)
# Front-matter noise lines that carry no policy content: the print run header,
# the session line, and the sponsor/committee introduction block. Stripping them
# keeps the extracted body starting at the bill's actual subject.
_FRONT_MATTER_RE = re.compile(
    r"^\s*(?:PRIOR\s+)?PRINTER'?S\s+NO\.?.*$"
    r"|^\s*Session\s+of\s*$"
    r"|^\s*Session\s+of\s+\d.*$"
    r"|^\s*INTRODUCED\s+BY\b.*$"
    # Maryland / general first-reading masthead lines.
    r"|^\s*Introduced\s+and\s+read\s+first\s+time\b.*$"
    r"|^\s*Assigned\s+to:\s*.*$"
    # Delaware-style cover sheet: "SPONSOR: Sen. Mantzavinos / DELAWARE STATE
    # SENATE / 153rd GENERAL ASSEMBLY / SENATE AMENDMENT NO. 1 / TO / SENATE
    # SUBSTITUTE NO. 1 / FOR / SENATE BILL NO. 16". Each line is short and
    # ALL-CAPS but none repeats, so the running-banner rule never sees them.
    r"|^\s*SPONSORS?:\s*.*$"
    r"|^\s*\d+(?:st|nd|rd|th)\s+GENERAL\s+ASSEMBLY\s*$"
    r"|^\s*(?:HOUSE|SENATE)\s+(?:BILL|AMENDMENT|SUBSTITUTE|RESOLUTION)"
    r"\s+NO\.\s*\d+\s*$"
    # "DELAWARE STATE SENATE" and the bare "TO" / "FOR" connectors that sit
    # between the cover sheet's stacked bill-number lines.
    r"|^\s*[A-Z][A-Z ]{2,30}\s(?:STATE\s+)?(?:SENATE|HOUSE\s+OF\s+REPRESENTATIVES)\s*$"
    r"|^\s*(?:TO|FOR)\s*$",
    re.IGNORECASE,
)

# A bill's masthead often opens with a sponsor block that runs across several
# lines before the bill's real text — Maryland's "By: Senators Hershey, Bailey,
# … and West / Introduced and read first time: … / Assigned to: Rules / A BILL
# ENTITLED / AN ACT concerning …". Left in, the summary becomes the sponsor list
# (observed on MD SB2102: "Senators Hershey, … Introduced and read first time.").
# Drop the whole span from a leading "By: Senators/Delegates/Representatives" up
# to where the bill actually begins ("A BILL ENTITLED" / "AN ACT"). Bounded to a
# short gap and anchored to that lookahead so it can only remove the masthead,
# never body text (if the bill-start anchor isn't close after "By:", nothing is
# stripped).
_SPONSOR_FRONTMATTER_RE = re.compile(
    r"\bBy:\s+(?:Senators?|Delegates?|Representatives?)\b.{0,400}?"
    r"(?=\bA\s+BILL\s+ENTITLED\b|\bAN\s+ACT\b)",
    re.IGNORECASE | re.DOTALL,
)


# Some legislature sites serve a bill as an HTML page whose scraped text is
# prefixed with the site's navigation chrome — a wall of menu items, search
# boxes, and "share this" links — before the actual bill body. Fed that verbatim,
# a small model summarizes the MENU instead of the bill. Two real examples:
#
#   California (leginfo): "Bill Text - SB-1444 Employment. skip to content home
#     accessibility FAQ feedback sitemap login x Quick Search: … SHARE THIS:
#     SB1444:v98#DOCUMENTBill Start … CALIFORNIA LEGISLATURE …"  → produced the
#     CA HR-127 / SB-1444 / SB-938 failures.
#   Minnesota (revisor/leg): "SF 4171 Introduction … Skip to main content Skip to
#     footer Minnesota Legislature … Menu House … Committees … A bill for an act
#     relating to … BE IT ENACTED …"  → 5 KB of menu before the bill.
#
# This isn't one site's quirk, so the fix is general: when a chrome signature
# sits at the top, cut forward to where the bill body demonstrably begins. CA
# embeds an explicit "#DOCUMENT" start marker; every other layout is cut to the
# earliest universal enacting anchor ("A bill for an act", "Be it enacted", a
# "<Chamber> Bill/Resolution No." caption, "CALIFORNIA LEGISLATURE", …). Guarded
# by the chrome signature and an anchor match, so a bill with no chrome — or one
# whose body start can't be located — is returned untouched rather than cut.
_WEB_CHROME_SIG_RE = re.compile(
    # "menu Home …" is Alaska's (akleg.gov) nav bar, which carries no
    # skip-to-content link: "Alaska State Legislature The Alaska State
    # Legislature menu Home Senate Current Members Past Members …".
    r"skip to (?:main )?content|skip to footer|Quick Search\s*:"
    r"|\bmenu\s+Home\b",
    re.IGNORECASE,
)
_CA_DOCUMENT_MARKER_RE = re.compile(r"#DOCUMENT\s*(?:Bill\s*Start\s*)?", re.IGNORECASE)
# Universal "the bill body starts here" anchors, kept in the output (cut BEFORE
# them). Specific legislative phrases that a navigation menu never contains.
_BILL_BODY_ANCHOR_RE = re.compile(
    r"\bA\s+bill\s+for\s+an\s+act\b"
    r"|\bBe\s+it\s+enacted\b"
    r"|\bThe\s+people\s+of\s+the\s+State\s+of\b"
    r"|\bCALIFORNIA\s+LEGISLATURE\b"
    r"|\b(?:House|Senate|Assembly)\s+(?:Bill|Resolution|Joint\s+Resolution|"
    r"Concurrent\s+Resolution)\s+No\."
    # Alaska's bill-detail heading, which follows its nav bar and opens the
    # bill's own title: "Enrolled HB 10: Relating to the Board of Regents …".
    r"|\b(?:Enrolled|Engrossed|Introduced)\s+[A-Z]{2,4}\s?\d+:\s",
    re.IGNORECASE,
)


def _strip_web_chrome(text: str) -> str:
    """Drop a leading block of legislature-website navigation chrome (menus,
    search boxes, "share this" links) that a naive HTML scrape leaves in front of
    the bill body. Only fires when a chrome signature sits at the very top AND a
    content boundary is found, so a real bill is never truncated. Returns the
    input unchanged otherwise."""
    if not text:
        return text
    if not _WEB_CHROME_SIG_RE.search(text[:1200]):
        return text
    # California embeds an explicit document-start marker; prefer it.
    m = _CA_DOCUMENT_MARKER_RE.search(text)
    if m:
        return text[m.end():].lstrip()
    # Otherwise cut to the earliest enacting-clause anchor (kept in the output).
    m = _BILL_BODY_ANCHOR_RE.search(text)
    if m and m.start() > 0:
        return text[m.start():].lstrip()
    return text


# ---------------------------------------------------------------------------
# Document front matter (masthead / publisher metadata)
# ---------------------------------------------------------------------------
# Distinct from website chrome above: this is junk inside the *document* itself.
# Nearly every legislature prints a masthead before the bill — chamber name,
# bill number, sponsor list, printer's numbers, amendment stamps — and some
# publishers prepend their own metadata block. Fed that verbatim, a small model
# summarizes the masthead; worse, when the model call fails the deterministic
# excerpt fallback quotes it straight into the post. Three real examples from
# one weekly-digest run, each posted verbatim:
#
#   US HR 3633  "HR 3633 EH: … U.S. House of Representatives / text/xml / EN /
#               Pursuant to Title 17 Section 105 of the United States Code, this
#               file is not subject to copyright protection …"
#   IL SB2981   "Full Text of SB2981 Illinois General Assembly ILGA.GOV …
#               Introduced 1/27/2026, by Sen. Graciela Guzmán SYNOPSIS AS
#               INTRODUCED:"
#   CA AB2575   "Amended IN Senate June 18, 2026 Amended IN Senate June 11, 2026
#               … CALIFORNIA LEGISLATURE— … Assembly Bill No. 2575Introduced by
#               Assembly Member Ortega February 20, 2026"
#
# Rather than a rule per legislature, cut forward to where the bill's own
# purpose demonstrably begins — the enacting clause or long title, phrases a
# masthead never contains. Two guards keep this from ever eating real text:
# the anchor must appear within _FRONTMATTER_MAX_SCAN of the top, and the span
# being dropped must actually look like a masthead (see _looks_like_masthead),
# so a bill whose body merely opens with one of these phrases is left alone.
# Anchors are written with a (?<![A-Za-z]) lookbehind rather than \b, and
# without a trailing \b, because PDF and XML extraction routinely glues the
# masthead straight onto the anchor with no space: "…January 21, 2026An act to
# add Chapter 13.6…" (CA SB903) and "…jurisdiction of the committee concernedA
# BILLTo amend the Internal Revenue Code…" (US HR 10102). A \b on either side
# fails on both — "6A" and "LLTo" carry no word boundary — which is exactly how
# those two bills kept their mastheads while their siblings were cleaned.
# Each entry is (pattern, keep_anchor): keep when the anchor reads as the start
# of the sentence ("AN ACT To provide…", "A bill for an act relating to…"),
# drop when it is a bare section label ("SYNOPSIS AS INTRODUCED:").
_FRONTMATTER_ANCHORS = (
    (re.compile(r"(?<![A-Za-z])AN\s+ACT", re.IGNORECASE), True),
    (re.compile(r"(?<![A-Za-z])A\s+BILL\s+ENTITLED", re.IGNORECASE), True),
    (re.compile(r"(?<![A-Za-z])A\s+BILL(?=\s*(?:TO|FOR)\b)", re.IGNORECASE), True),
    (re.compile(r"(?<![A-Za-z])BE\s+IT\s+ENACTED", re.IGNORECASE), True),
    # Section labels: the useful prose is what follows, so drop the label.
    (re.compile(r"(?<![A-Za-z])SYNOPSIS(?:\s+AS\s+INTRODUCED)?:\s*", re.IGNORECASE), False),
    # A federal bill that has not passed a chamber carries no "AN ACT"/"A BILL";
    # its long title is a "To <verb> …" clause right after the bill number
    # ("119TH CONGRESS / 2D SESSION / H. R. 9629 / To require an assessment…").
    # Drop everything through the number so the text opens on that clause.
    # The lookahead is case-SENSITIVE: a long title always opens with a capital
    # "To …". Under IGNORECASE it also matched the lowercase "to" inside "To
    # amend the Internal Revenue Code of 1986 to establish …", cutting the
    # sentence in half and opening the post mid-clause. The referral line is
    # bounded too, so a glued masthead can't run away with the match.
    (re.compile(
        r"(?:H\.\s?R\.|S\.|H\.\s?J\.\s?RES\.|S\.\s?J\.\s?RES\.)\s*\d+\s*"
        r"(?:IN THE (?:HOUSE OF REPRESENTATIVES|SENATE)[^\n]{0,120})?\s*(?=(?-i:To)\s+[a-z])",
        re.IGNORECASE), False),
)

# Extraction glues a masthead's last word straight onto the enacting phrase
# ("…jurisdiction of the committee concernedA BILLTo amend…"), which defeats the
# (?<![A-Za-z]) guard on every anchor above. Break the seam first so the anchors
# see a real boundary. Runs after _US_BILL_GLUE_RE has split "A BILLTo".
_GLUED_ANCHOR_RE = re.compile(r"(?<=[a-z])(?=(?:A\s+BILL|AN\s+ACT))")

# Enacting-clause boilerplate. Every bill has one and none of them say anything
# about the bill, but when the model call fails the excerpt fallback quotes
# whatever leads the text — which is how a post opened "The People of the State
# of New York, represented in Senate and Assembly, do enact as follows".
# Bounded to the top of the document and to a short span so it can only match
# the preamble itself, never a sentence of the bill.
_ENACTING_CLAUSE_RE = re.compile(
    r"^\s*(?:The\s+People\s+of\s+the\s+State\s+of|Be\s+it\s+enacted\s+by)"
    r"[^.]{0,200}?do\s+enact\s+as\s+follows\s*:\s*",
    re.IGNORECASE,
)

# The federal XML dump glues the enacting phrase to the long title, leaving
# "A BILLTo amend …". Split it so the sentence reads normally once the masthead
# above it is cut away.
# No lookbehind and no trailing \b: the whole point is that both sides are glued
# ("concernedA BILLTo amend"), so demanding a boundary on either side is exactly
# what stops the rule from firing on the case it exists for.
_US_BILL_GLUE_RE = re.compile(r"(A\s+BILL)(?=To\s+[a-z])")

_FRONTMATTER_MAX_SCAN = 4000

# Michigan-style bills open with an amend-citation and nothing else:
#
#   A bill to amend 1956 PA 218, entitled "The insurance code of 1956," by
#   amending section 3701 (MCL 500.3701), as amended by 2016 PA 276.
#   the people of the state of michigan enact: Sec. 3701. As used in this chapter…
#
# The clause names the statute being edited but never says what the edit does,
# so a post led by it reads "A bill to amend 1956 PA 218 … by amending section
# 3701 (MCL 500.3701), as amended by 2016 PA 276" and tells a reader nothing
# (MI HB 4207). Skip it and its enacting clause so both the model and the
# deterministic fallback start on the bill's operative text.
#
# Only when the clause carries no purpose language of its own — Minnesota's
# "A bill for an act relating to public safety; providing for local correctional
# officers…" is the same shape but genuinely describes the bill, so it stays.
# Quoted spans are removed before that test, because the quoted part is the
# amended act's OLD title ("An act to provide for the granting of military
# leaves…"), which describes the statute rather than this bill.
_AMEND_PREAMBLE_RE = re.compile(
    r"\A\s*A\s+bill\s+to\s+amend\b(?P<pre>.{0,700}?)"
    r"(?:the\s+people\s+of\s+the\s+state\s+of\s+[A-Za-z ]{2,20}?\s+enact\s*:"
    r"|be\s+it\s+enacted[^:]{0,160}:)\s*",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_SPAN_RE = re.compile(r"[\"“][^\"”]{0,300}[\"”]")
_PURPOSE_LANGUAGE_RE = re.compile(
    r"\brelating to\b|\bproviding for\b|\bto provide\b|\bto require\b"
    r"|\bto prohibit\b|\bto establish\b|\bto create\b|\bto authorize\b"
    r"|\bto regulate\b|\bto repeal\b",
    re.IGNORECASE,
)


def _strip_amend_preamble(text: str) -> str:
    """Drop a content-free "A bill to amend <act>, by amending section N …"
    preamble and the enacting clause after it. Returns the text unchanged when
    the preamble describes the bill's purpose, or when dropping it would leave
    nothing behind."""
    m = _AMEND_PREAMBLE_RE.match(text)
    if not m:
        return text
    if _PURPOSE_LANGUAGE_RE.search(_QUOTED_SPAN_RE.sub(" ", m.group("pre"))):
        return text
    rest = text[m.end():].lstrip()
    return rest or text

# Illinois opens its synopsis with a run of statute citations before any prose:
# "New Act30 ILCS 105/5.1038 new    Creates the Climate Change Superfund Act."
# They say nothing to a reader, and the sentence splitter downstream turns the
# run into a stray "1038 new" leading the post. Drop the citations, keep the
# prose. Anchored to the very start and required to begin with a citation, so
# it can only ever trim this header.
_IL_CITATION_RUN_RE = re.compile(
    # Lookahead allows a bare leading "ILCS ..." too: extraction sometimes eats
    # the chapter number, leaving "ILCS 2610/12.8 new50 ILCS 205/25..." with no
    # digits in front, which the digits-required form missed (IL HB1036).
    r"^(?=\s*(?:New\s+Act|\d*\s*ILCS))"
    r"(?:\s*(?:New\s+Act|\d*\s*ILCS\s+[\d./A-Za-z-]+|new|rep\.|"
    r"from\s+Ch\.\s*[\d.,\s-]*(?:par\.\s*[\d.,\s-]*)?))+\s*",
    re.IGNORECASE,
)


# Phrases that occur only in a masthead or a publisher's metadata block, never
# in the text of a bill. Their presence settles the question on its own — needed
# because some front matter is itself wordy prose (the federal copyright notice
# and the "Ms. Salinas introduced the following bill; which was referred to the
# Committee on…" referral line together read like sentences, so the lowercase
# ratio below cleared the bar and left US HR 10102's masthead in place).
_DEFINITE_FRONTMATTER_RE = re.compile(
    r"Pursuant to Title 17[,\s]+Section 105"
    r"|introduced the following bill;\s*which was referred to"
    r"|^text/xml\s*$"
    r"|Full Text of\s+\S+\s+Illinois General Assembly"
    r"|CALIFORNIA LEGISLATURE"
    r"|Amended IN\s+(?:Senate|Assembly)",
    re.IGNORECASE | re.MULTILINE,
)


def _looks_like_masthead(span: str) -> bool:
    """True when ``span`` is masthead/metadata rather than the bill's own prose.

    A masthead is dominated by capitalised chamber names, sponsor surnames, bill
    numbers, dates and stamps; real bill text reads as sentences and so carries a
    high proportion of lowercase words. Requiring this before cutting means a
    bill that simply opens with an enacting clause keeps everything above it —
    unless the span carries one of the give-away phrases above, which no bill
    body contains."""
    if _DEFINITE_FRONTMATTER_RE.search(span):
        return True
    words = span.split()
    if not words:
        return False
    lower = sum(1 for w in words if w[:1].islower())
    return (lower / len(words)) < 0.55


def _strip_bill_frontmatter(text: str) -> str:
    """Cut a leading masthead/publisher-metadata block, keeping the bill's own
    purpose clause onward. Returns the input unchanged when no anchor is found
    near the top or the span above it doesn't look like a masthead."""
    if not text:
        return text
    cuts = []
    for rx, keep_anchor in _FRONTMATTER_ANCHORS:
        m = rx.search(text, 0, _FRONTMATTER_MAX_SCAN)
        if m:
            cuts.append(m.start() if keep_anchor else m.end())
    if not cuts:
        return text
    best = min(cuts)
    # A cut of 0 means the text already opens on the bill's purpose — there is no
    # masthead left to remove. Returning here is what makes this idempotent, and
    # idempotence is load-bearing: cached text is re-cleaned on every read, and
    # bills whose body quotes a further "AN ACT" (Pennsylvania's "AN ACT /
    # Amending the act of March 4, 1971 …, entitled "An act relating to …"") would
    # otherwise be cut again at that inner phrase on each pass, walking the text
    # forward and eating the bill a slice at a time.
    if best == 0:
        return text
    if not _looks_like_masthead(text[:best]):
        return text
    return text[best:].lstrip()


# California glues the "LEGISLATIVE COUNSEL'S DIGEST" section label and the
# bill-citation line after it straight onto the previous sentence, so a summary
# opened "…relating to health care services.LEGISLATIVE COUNSEL'S DIGESTAB 2575,
# as amended, Ortega." Drop the label and that citation; the digest prose that
# follows is the plain-English explainer worth keeping.
_CA_COUNSEL_DIGEST_RE = re.compile(
    r"LEGISLATIVE\s+COUNSEL'?S\s+DIGEST\s*"
    r"(?:[A-Z]{1,3}\s?\d+,\s*as\s+(?:amended|introduced|added)[^.]*\.)?\s*",
    re.IGNORECASE,
)


def clean_bill_text(text: str) -> str:
    """Strip per-line numbering, page headers/footers, and repeated ALL-CAPS
    banner lines from extracted bill text, then collapse runs of blank lines.
    Best-effort: returns the (possibly lightly cleaned) text, never raises."""
    if not text:
        return ""

    text = _strip_web_chrome(text)
    text = _CA_COUNSEL_DIGEST_RE.sub(" ", text)
    text = _US_BILL_GLUE_RE.sub(r"\1\n", text)
    text = _GLUED_ANCHOR_RE.sub("\n", text)
    text = _strip_bill_frontmatter(text)
    text = _strip_amend_preamble(text)
    text = _ENACTING_CLAUSE_RE.sub("", text, count=1)
    text = _IL_CITATION_RUN_RE.sub("", text, count=1)
    text = text.replace("\f", "\n")
    text = _AMEND_ARROW_RE.sub(" ", text)
    # Minnesota redline markup — remove deleted spans whole, keep inserted text
    # (markers stripped). Runs on the full text before the line split because a
    # single marker pair routinely straddles several lines.
    text = _MN_DELETED_RE.sub(" ", text)
    text = _MN_NEWTEXT_MARK_RE.sub(" ", text)
    # Sponsor/first-reading masthead (spans lines) — drop it before the line split
    # so the summary starts at the bill's actual purpose, not the sponsor list.
    text = _SPONSOR_FRONTMATTER_RE.sub(" ", text)
    raw_lines = text.split("\n")

    # Count repeated short ALL-CAPS lines — these are running banners/headers
    # (e.g. the bill's short title stamped at the top of every page).
    banner_counts: dict[str, int] = {}
    for ln in raw_lines:
        s = ln.strip()
        if s and len(s) <= 60 and _ALLCAPS_BANNER_RE.match(s):
            banner_counts[s] = banner_counts.get(s, 0) + 1
    repeated_banners = {s for s, n in banner_counts.items() if n >= 3}

    cleaned: list[str] = []
    for ln in raw_lines:
        ln = _LEADING_LINE_NUMBER_RE.sub("", ln)
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        if _PAGE_MARKER_RE.match(s):
            continue
        if _FRONT_MATTER_RE.match(s):
            continue
        if s in repeated_banners:
            continue
        cleaned.append(s)

    # Collapse runs of blank lines into a single paragraph break.
    out_lines: list[str] = []
    blank = False
    for ln in cleaned:
        if not ln:
            if not blank and out_lines:
                out_lines.append("")
            blank = True
        else:
            out_lines.append(ln)
            blank = False

    result = "\n".join(out_lines).strip()
    if len(result) > TEXT_MAX_CHARS:
        result = result[:TEXT_MAX_CHARS]
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_bill_text(sources_bill: str, root: Path = ROOT) -> str | None:
    """Resolve, download, and extract clean full text for a bill given its
    ``sources.bill`` path. Returns ``None`` at any failure point so callers can
    fall back to abstract-only behavior. See ``extract_bill_text_verbose`` for
    the reason string used in logging."""
    text, _reason = extract_bill_text_verbose(sources_bill, root=root)
    return text


@lru_cache(maxsize=1)
def _cleaner_fingerprint() -> str:
    """Short hash of this module's source, used as the cache subdirectory.

    The on-disk cache stores text that has ALREADY been cleaned, keyed on the
    document URL alone — so it is content-addressed for the *document* but not
    for the *cleaning code*. CI restores it from the newest previous run
    (restore-keys: bill-text-cache-), so an entry written before a cleaning fix
    kept serving its old dirty text to every later run, indefinitely: bills
    cached with their site's navigation menu still in front of them went on
    posting that menu long after the chrome stripper shipped, which is why
    earlier post-quality fixes appeared to have no effect. Keying the cache on
    the cleaning code's own hash retires exactly those entries whenever the
    cleaning changes, with no version constant for anyone to remember to bump.
    Falls back to a fixed name if the source can't be read."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


def extract_bill_text_verbose(sources_bill: str, root: Path = ROOT) -> tuple[str | None, str]:
    """Like ``extract_bill_text`` but also returns a short reason string so the
    caller can log why full text was or wasn't used. Reasons:
    ``ok`` | ``ok-extracted-file`` | ``no-sources-path`` |
    ``metadata-not-found`` | ``no-document-link`` | ``download-failed`` |
    ``pdftotext-missing`` | ``extract-failed`` | ``empty-after-clean``. Uses an
    in-process memo plus an on-disk content-addressed cache keyed on the
    document URL."""
    if not sources_bill:
        return None, "no-sources-path"

    metadata_path = resolve_metadata_path(sources_bill, root=root)
    if metadata_path is None:
        return None, "metadata-not-found"

    # Prefer govbot's pre-extracted text committed beside the bill: it's already
    # on disk from the clone, so it skips the PDF download + pdftotext entirely
    # (no more download/extract failures leaving a bill summary-less) and needs
    # no poppler on the runner. Fall through to the PDF path only when the new
    # files/ layout isn't present (older clone) or the extracted file is empty.
    extracted_path = find_extracted_text_file(metadata_path)
    if extracted_path is not None:
        memo_key = f"file://{extracted_path}"
        if memo_key in _run_memo:
            return _run_memo[memo_key]
        try:
            raw_extracted = extracted_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw_extracted = ""
        text_extracted = clean_bill_text(raw_extracted)
        if text_extracted:
            result = (text_extracted, "ok-extracted-file")
            _run_memo[memo_key] = result
            return result
        # Empty extracted file → fall back to the download path below.

    link = find_document_link(metadata_path)
    if link is None:
        return None, "no-document-link"
    url, media_type = link

    if url in _run_memo:
        return _run_memo[url]

    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_file = BILL_TEXT_CACHE_DIR / _cleaner_fingerprint() / f"{cache_key}.txt"
    if cache_file.is_file():
        try:
            cached = cache_file.read_text(encoding="utf-8")
            result = (cached or None, "ok" if cached else "empty-after-clean")
            _run_memo[url] = result
            return result
        except OSError:
            pass

    if media_type == "application/pdf" and not pdftotext_available():
        result = (None, "pdftotext-missing")
        _run_memo[url] = result
        return result

    doc_path = _download(url)
    if doc_path is None:
        result = (None, "download-failed")
        _run_memo[url] = result
        return result

    try:
        if media_type == "application/pdf":
            raw_text = _extract_pdf(doc_path)
        else:
            raw_text = _extract_html(doc_path)
    finally:
        # Only delete files we created in the temp dir, not local fixtures.
        try:
            if str(doc_path).startswith(tempfile.gettempdir()):
                doc_path.unlink(missing_ok=True)
        except OSError:
            pass

    if not raw_text:
        result = (None, "extract-failed")
        _run_memo[url] = result
        return result

    text = clean_bill_text(raw_text)
    if not text:
        result = (None, "empty-after-clean")
        _run_memo[url] = result
        return result

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
    except OSError:
        pass

    result = (text, "ok")
    _run_memo[url] = result
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python scripts/bill_text.py <sources.bill metadata.json path>",
              file=sys.stderr)
        return 2
    sources_bill = argv[0]
    print(f"pdftotext available: {pdftotext_available()}")
    meta = resolve_metadata_path(sources_bill)
    print(f"resolved metadata: {meta}")
    if meta is None:
        return 1
    print(f"pre-extracted text file: {find_extracted_text_file(meta)}")
    link = find_document_link(meta)
    print(f"document link: {link}")
    text, reason = extract_bill_text_verbose(sources_bill)
    print(f"extraction reason: {reason}")
    if not text:
        print("no text extracted")
        return 1
    print(f"\n----- extracted text ({len(text)} chars) -----\n")
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
