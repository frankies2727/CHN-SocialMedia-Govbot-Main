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
# Front-matter noise lines that carry no policy content: the print run header,
# the session line, and the sponsor/committee introduction block. Stripping them
# keeps the extracted body starting at the bill's actual subject.
_FRONT_MATTER_RE = re.compile(
    r"^\s*(?:PRIOR\s+)?PRINTER'?S\s+NO\.?.*$"
    r"|^\s*Session\s+of\s*$"
    r"|^\s*Session\s+of\s+\d.*$"
    r"|^\s*INTRODUCED\s+BY\b.*$",
    re.IGNORECASE,
)


# California's legislature site (leginfo.legislature.ca.gov) serves each bill as
# an HTML page whose scraped text is prefixed with the site's navigation chrome:
#   "Bill Text - SB-1444 Employment. skip to content home accessibility FAQ
#    feedback sitemap login x Quick Search: Bill Number Bill Keyword Home Bill
#    Information California Law Publications … My Subscriptions My Favorites …
#    SHARE THIS: SB1444:v98#DOCUMENTBill Start … CALIFORNIA LEGISLATURE …"
# Fed that verbatim, a small model summarizes the MENU ("skip to content home
# accessibility faq feedback sitemap login …") instead of the bill — the exact
# failure seen on CA HR-127 and CA SB-1444. The real document always begins at
# the "#DOCUMENT" marker the page embeds; everything before it is chrome. Cut it
# (plus the "Bill Start" label that immediately follows) so the extracted text
# opens on the actual bill. Guarded by a chrome signature so it only fires on a
# genuinely chrome-prefixed page, never on a bill that merely cites a website.
_WEB_CHROME_SIG_RE = re.compile(r"skip to content|Quick Search:", re.IGNORECASE)
_CA_DOCUMENT_MARKER_RE = re.compile(r"#DOCUMENT\s*(?:Bill\s*Start\s*)?", re.IGNORECASE)


def _strip_web_chrome(text: str) -> str:
    """Drop a leading block of legislature-website navigation chrome (menus,
    search boxes, "share this" links) that a naive HTML scrape leaves in front of
    the bill body. Currently targets California's leginfo layout. Only fires when
    a chrome signature sits at the very top AND a content anchor is found, so a
    real bill is never truncated. Returns the input unchanged otherwise."""
    if not text:
        return text
    if not _WEB_CHROME_SIG_RE.search(text[:800]):
        return text
    m = _CA_DOCUMENT_MARKER_RE.search(text)
    if m:
        return text[m.end():].lstrip()
    idx = text.find("CALIFORNIA LEGISLATURE")
    if idx > 0:
        return text[idx:].lstrip()
    return text


def clean_bill_text(text: str) -> str:
    """Strip per-line numbering, page headers/footers, and repeated ALL-CAPS
    banner lines from extracted bill text, then collapse runs of blank lines.
    Best-effort: returns the (possibly lightly cleaned) text, never raises."""
    if not text:
        return ""

    text = _strip_web_chrome(text)
    text = text.replace("\f", "\n")
    text = _AMEND_ARROW_RE.sub(" ", text)
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
    cache_file = BILL_TEXT_CACHE_DIR / f"{cache_key}.txt"
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
        BILL_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
