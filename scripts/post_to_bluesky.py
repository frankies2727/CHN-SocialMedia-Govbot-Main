#!/usr/bin/env python3
"""
Filter govbot's bills.jsonl for the active category (transportation by
default), dedupe against the per-category state file, summarize with a
local Qwen model (served by Ollama), and post to Bluesky with rich
link-card embeds.

The category is selected via the BOT_CATEGORY env var and read from
categories/<name>/config.yml. See scripts/category.py.

Bill links go to each state's official legislature page when we have a
deep-link builder for that state, otherwise to the state legislature
homepage as a fallback.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

from category import Category, load_active_category

ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = ROOT / "bills.jsonl"

CATEGORY: Category = load_active_category()
STATE_FILE = CATEGORY.state_file_path()

POST_LIMIT = int(os.environ.get("POST_LIMIT", "4"))  # how many bluesky posts per run
DRY_RUN = os.environ.get("DRY_RUN") == "1"
# og:image fetching is paused by default. Set FETCH_OG_IMAGE=1 to re-enable
# thumbnail scraping from bill-page URLs. When off, posts still get an external
# link card — just without the image.
FETCH_OG_IMAGE = os.environ.get("FETCH_OG_IMAGE", "0") == "1"

BSKY_HANDLE = CATEGORY.bluesky_handle()
BSKY_PASSWORD = CATEGORY.bluesky_password()

BLUESKY_API = "https://bsky.social/xrpc"

# Local Qwen via Ollama. Defaults assume `ollama serve` is running on the
# same host (e.g. installed in the GitHub Actions step before this script runs)
# and the Qwen model has been pulled with `ollama pull <QWEN_MODEL>`.
QWEN_API_URL = os.environ.get("QWEN_API_URL", "http://localhost:11434/api/chat")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen2.5:1.5b")
QWEN_TIMEOUT = int(os.environ.get("QWEN_TIMEOUT", "180"))

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


def best_display_text(b: dict) -> str:
    if _looks_like_code_title(b["title"]) and b["abstract"]:
        return b["abstract"]
    return b["title"]


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
    desc = _smart_case(action_desc)
    nice_date = _format_date(date_yyyy_mm_dd)
    if desc and nice_date:
        desc_with_period = desc if desc.endswith((".", "!", "?")) else desc + "."
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


def summarize(b: dict) -> str:
    abstract = (b["abstract"] or "").strip()
    title = b["title"].strip()
    fallback = abstract[:180] if (abstract and abstract.lower() != title.lower()) else ""

    body_for_prompt = abstract if (abstract and abstract.lower() != title.lower()) else title

    user_prompt = (
        f"Title: {title}\n"
        f"Description: {body_for_prompt[:2000]}\n\n"
        "Write the one-sentence neutral summary now."
    )

    try:
        r = requests.post(
            QWEN_API_URL,
            json={
                "model": QWEN_MODEL,
                "messages": [
                    {"role": "system", "content": CATEGORY.summary_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.3},
            },
            timeout=QWEN_TIMEOUT,
        )
        if not r.ok:
            print(f"  ! Qwen {r.status_code}: {r.text[:300]}", file=sys.stderr)
            r.raise_for_status()
        data = r.json()
        # Ollama /api/chat returns {"message": {"content": "..."}, ...}
        # Ollama /api/generate returns {"response": "...", ...}
        text = (data.get("message") or {}).get("content") or data.get("response") or ""
        return _clean_summary(text)
    except Exception as e:
        print(f"  ! summarization failed, using fallback: {e}", file=sys.stderr)
        return fallback


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


def _b_wi(session, ident):  # verified — docs.legis.wisconsin.gov
    year = _first_year(session)
    typ, num = _split_ident(ident)
    return f"https://docs.legis.wisconsin.gov/{year}/related/proposals/{typ.lower()}{num}" if (year and typ and num) else None


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


def _b_mo(session, ident):  # best-effort — chamber-specific deep link by year+session-code
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    m = re.search(r"20\d{2}\s*([A-Za-z]\d*)", session or "")
    code = (m.group(1) if m else "R").upper()
    if typ.startswith("H"):
        return f"https://house.mo.gov/Bill.aspx?bill={typ}{num}&year={year}&code={code}"
    return f"https://www.senate.mo.gov/{year[-2:]}info/BTS_Web/Bill.aspx?SessionType={code}&BillID={typ}{num}"


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


def _b_ks(session, ident):  # best-effort — kslegislature.org biennium URL
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    next_year = str(int(year) + 1)
    return f"https://www.kslegislature.org/li/b{year}_{next_year[-2:]}/measures/{typ.lower()}_{num}/"


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


def _b_co(session, ident):  # verified — leg.colorado.gov /bills/<typ><yy>-<num>
    year = _first_year(session)
    typ, num = _split_ident(ident)
    if not (year and typ and num):
        return None
    # CO conventions: HB numbers are 4 digits (e.g. HB25-1001); SB and joint /
    # concurrent / simple resolutions are 3 digits (SB25-001, SJR25-006).
    width = 4 if typ == "HB" else 3
    return f"https://leg.colorado.gov/bills/{typ.lower()}{year[-2:]}-{num.zfill(width)}"


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
            f"?BillNumber={typ}{num}&GA={ga}")


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


STATE_BILL_URL_BUILDERS = {
    "FL": _b_fl, "IN": _b_in, "IA": _b_ia, "MI": _b_mi, "NY": _b_ny,
    "MA": _b_ma, "OH": _b_oh, "WI": _b_wi, "NC": _b_nc, "NJ": _b_nj,
    "CT": _b_ct, "MO": _b_mo, "MN": _b_mn, "NM": _b_nm, "HI": _b_hi,
    "KS": _b_ks, "WV": _b_wv, "PA": _b_pa, "AK": _b_ak, "OR": _b_or,
    "CO": _b_co, "WA": _b_wa, "TN": _b_tn, "RI": _b_ri, "MS": _b_ms,
    "AL": _b_al,
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

def compose_post(b: dict, summary: str) -> tuple[str, str, str, str]:
    emoji = CATEGORY.emoji_for(b)
    link = link_for(b)
    link_block = f"\n\n{LINK_PREFIX}{link}" if link else ""

    state_label = b["state"] or "?"
    display = best_display_text(b).strip()
    summary = (summary or "").strip()

    summary_block = f"\n\n{summary}" if (summary and summary.lower() != display.lower()) else ""
    action_line = format_action_line(b["action_desc"], b["action_date"])
    action_block = f"\n\n{action_line}" if action_line else ""

    head = f"{emoji} {state_label} {b['identifier']} — {display}"

    def assemble(h, s, a, l):
        return h + s + a + l

    text = assemble(head, summary_block, action_block, link_block)

    if len(text) > MAX_POST and summary_block:
        overflow = len(text) - MAX_POST
        new_len = max(0, len(summary) - overflow - 1)
        if new_len > 20:
            summary = summary[:new_len].rstrip() + "…"
            summary_block = f"\n\n{summary}"
        else:
            summary_block = ""
        text = assemble(head, summary_block, action_block, link_block)

    if len(text) > MAX_POST and action_block and action_line:
        nice_date = _format_date(b["action_date"])
        if nice_date:
            date_prefix = f"{nice_date}: "
            if action_line.startswith(date_prefix):
                desc_part = action_line[len(date_prefix):].rstrip(".!?")
                overflow = len(text) - MAX_POST
                new_len = max(0, len(desc_part) - overflow - 1)
                if new_len > 8:
                    action_line = date_prefix + desc_part[:new_len].rstrip() + "…"
                    action_block = f"\n\n{action_line}"
                else:
                    # Not enough room for a meaningful description — drop
                    # the action block entirely rather than emit a bare
                    # date, which conveys no information on its own.
                    action_line = ""
                    action_block = ""
            else:
                action_block = f"\n\n{action_line}"
        text = assemble(head, summary_block, action_block, link_block)

    if len(text) > MAX_POST:
        avail = MAX_POST - len(link_block) - len(summary_block) - len(action_block) \
                - len(emoji) - len(f" {state_label} {b['identifier']} — ") - 1
        display_trimmed = display[:max(0, avail)].rstrip() + "…"
        head = f"{emoji} {state_label} {b['identifier']} — {display_trimmed}"
        text = assemble(head, summary_block, action_block, link_block)

    state_name = STATE_FULL_NAME.get(b["state"], b["state"] or "Bill")
    embed_title = f"{state_name} {b['identifier']}"[:300]
    embed_desc = (b["abstract"] or summary or display)[:280]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not DRY_RUN and (not BSKY_HANDLE or not BSKY_PASSWORD):
        print(f"ERROR: {CATEGORY.bluesky_handle_env()} and "
              f"{CATEGORY.bluesky_password_env()} must be set.", file=sys.stderr)
        return 1

    records = load_bills(JSONL_PATH)
    if not records:
        return 0

    state = load_state()
    seen = set(state.get("posted", []))

    candidates: list[dict] = []
    for r in records:
        b = extract_fields(r)
        if not b:
            continue
        if not CATEGORY.matches(b):
            continue
        if b["dedup_key"] in seen:
            continue
        candidates.append(b)

    # Same-day dedup (collapse multiple log entries for same bill on same day).
    unique_by_day: dict[str, dict] = {}
    for b in candidates:
        existing = unique_by_day.get(b["same_day_key"])
        if existing is None or len(b["action_desc"]) > len(existing["action_desc"]):
            unique_by_day[b["same_day_key"]] = b
    candidates = list(unique_by_day.values())

    print(f"Found {len(candidates)} new {CATEGORY.topic_phrase} bill update(s).")
    if not candidates:
        return 0

    # Print a state-distribution summary so we can see coverage.
    from collections import Counter
    state_counts = Counter(b["state"] or "?" for b in candidates)
    top = state_counts.most_common(15)
    print(f"  by state: {', '.join(f'{s}={n}' for s,n in top)}")

    def sort_key(b: dict):
        try:
            return datetime.strptime(b["action_date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            return datetime.min

    # Round-robin pick across states so one state with a big batch of updates
    # (e.g. all bills "withdrawn" on the same day a session ends) can't
    # monopolize the run. Each pass picks at most one bill per state, ordered
    # by which state has the freshest pending bill. If there aren't enough
    # distinct states to fill POST_LIMIT, later passes pick a second per state.
    by_state: dict[str, list[dict]] = {}
    for b in candidates:
        by_state.setdefault(b["state"] or "?", []).append(b)
    for bills in by_state.values():
        bills.sort(key=sort_key, reverse=True)
    state_order = sorted(by_state.keys(), key=lambda s: sort_key(by_state[s][0]), reverse=True)

    to_post: list[dict] = []
    while len(to_post) < POST_LIMIT and any(by_state[s] for s in state_order):
        for s in state_order:
            if not by_state[s]:
                continue
            to_post.append(by_state[s].pop(0))
            if len(to_post) >= POST_LIMIT:
                break

    distinct_states = len({b["state"] or "?" for b in to_post})
    print(f"Will post up to {POST_LIMIT}: posting {len(to_post)} from {distinct_states} state(s).")

    client = None if DRY_RUN else BlueskyClient(BSKY_HANDLE, BSKY_PASSWORD)

    for b in to_post:
        summary = summarize(b)
        text, link, ec_title, ec_desc = compose_post(b, summary)

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

    state["posted"] = sorted(seen)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"\nDone. State saved to {STATE_FILE.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
