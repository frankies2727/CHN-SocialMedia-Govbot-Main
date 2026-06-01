#!/usr/bin/env python3
"""
Full bill-text extraction for the govbot-bluesky pipeline.

The records govbot dumps into ``bills.jsonl`` only carry *metadata* (title,
abstract, sponsors, action descriptions). The actual legislative body of each
bill lives as a PDF (occasionally HTML) referenced inside the bill's on-disk
``metadata.json`` under ``versions[].links[]`` with
``media_type: "application/pdf"``. This module bridges that gap: given a
record's ``sources.bill`` path, it locates the metadata, finds the document
link, downloads it, and extracts clean plain text via ``pdftotext`` (poppler).

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
# govbot clones into ``~/.govbot/repos`` (per the govbot docs:
# ``~/.govbot/repos/**/bills/*/metadata.json``), so that is the primary base.
# Several fallbacks follow so a different layout still resolves. Override with
# the GOVBOT_DATA_ROOT env var. This is the single most fragile assumption in
# the module, so it is deliberately forgiving.
_HOME = Path.home()


def _candidate_bases() -> list[Path]:
    bases: list[Path] = []
    env_root = os.environ.get("GOVBOT_DATA_ROOT", "").strip()
    if env_root:
        bases.append(Path(env_root).expanduser())
    bases += [
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


def clean_bill_text(text: str) -> str:
    """Strip per-line numbering, page headers/footers, and repeated ALL-CAPS
    banner lines from extracted bill text, then collapse runs of blank lines.
    Best-effort: returns the (possibly lightly cleaned) text, never raises."""
    if not text:
        return ""

    text = text.replace("\f", "\n")
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
    ``ok`` | ``no-sources-path`` | ``metadata-not-found`` | ``no-document-link``
    | ``download-failed`` | ``pdftotext-missing`` | ``extract-failed`` |
    ``empty-after-clean``. Uses an in-process memo plus an on-disk
    content-addressed cache keyed on the document URL."""
    if not sources_bill:
        return None, "no-sources-path"

    metadata_path = resolve_metadata_path(sources_bill, root=root)
    if metadata_path is None:
        return None, "metadata-not-found"

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
