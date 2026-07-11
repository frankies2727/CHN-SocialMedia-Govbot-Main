#!/usr/bin/env python3
"""Build the data file that powers the GitHub Pages dashboard.

Reads every platform's posted records out of the repo and writes a single
``docs/data.json`` that ``docs/index.html`` renders as a dark-mode, fully
filterable dashboard. Run it locally with ``python scripts/build_dashboard.py``
or let the GitHub Actions workflow run it automatically on every push.

Platforms are DISCOVERED, not hard-coded: any directory under a topic that
contains a ``bills_used.json`` is treated as a platform (so folders like
``x-including-Crypto`` are picked up automatically, and new platforms such as
``bluesky`` appear the moment they post — no code change needed).

No third-party packages are required.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOPICS_DIR = REPO / "topics"
DOCS_DIR = REPO / "docs"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Public Bluesky account handle for each topic. These are read live (no auth)
# from Bluesky's public API by the dashboard, so the feed shows Bluesky posts
# exactly as posted. Handles are public and stable.
BLUESKY_HANDLES = {
    "ai_data_centers":         "govbotaidatacenter.bsky.social",
    "criminal_justice":        "govbotcrimejustice.bsky.social",
    "education":               "govboteducation.bsky.social",
    "elections_voting_rights": "govbotelections.bsky.social",
    "environment_climate":     "govbotclimate.bsky.social",
    "healthcare":              "govbothealthcare.bsky.social",
    "housing":                 "govbothousing.bsky.social",
    "immigration":             "govbotimmigration.bsky.social",
    "labor":                   "govbotlaborrights.bsky.social",
    "lgbtq":                   "govbotlgbtq.bsky.social",
    "reproductive_rights":     "govbotreproductive.bsky.social",
    "taxation":                "govbottaxation.bsky.social",
    "transportation":          "govbottransport.bsky.social",
}

# Nice display name + a stable color for known platform folders (matched
# case-insensitively). Unknown folders get a prettified name and a fallback
# color, so nothing is ever dropped just because it isn't listed here.
PLATFORM_INFO = {
    "instagram":          ("Instagram",        "#3987e5"),
    "meta-threads":       ("Meta Threads",     "#d95926"),
    "threads":            ("Threads",          "#d95926"),
    "bluesky":            ("Bluesky",          "#199e70"),
    "x":                  ("X",                "#9085e9"),
    "x-including-crypto": ("X (incl. Crypto)", "#9085e9"),
    "twitter":            ("X",                "#9085e9"),
    "facebook":           ("Facebook",         "#256abf"),
    "mastodon":           ("Mastodon",         "#6d5fe0"),
    "tiktok":             ("TikTok",           "#d55181"),
    "linkedin":           ("LinkedIn",         "#2f7bbf"),
}
# Order platforms sensibly; anything unknown falls to the end, alphabetically.
PLATFORM_ORDER = ["instagram", "meta-threads", "threads", "bluesky",
                  "x", "x-including-crypto", "twitter"]
FALLBACK_COLORS = ["#c98500", "#e66767", "#d55181", "#199e70", "#9085e9", "#3987e5"]


def prettify(folder: str) -> str:
    return folder.replace("-", " ").replace("_", " ").strip().title()


def platform_display(folder: str, fallback_idx: int) -> tuple[str, str]:
    info = PLATFORM_INFO.get(folder.lower())
    if info:
        return info
    return prettify(folder), FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]


def load_topic_meta(topic_dir: Path) -> dict:
    """Return {name, emoji, color} for a topic from its config.yml."""
    cfg = topic_dir / "config.yml"
    meta = {"name": topic_dir.name.replace("_", " ").title(), "emoji": "📄", "color": "#3987e5"}
    if not cfg.exists():
        return meta
    text = cfg.read_text(encoding="utf-8", errors="ignore")

    def grab(key: str):
        m = re.search(rf'^{key}:\s*"?(.*?)"?\s*$', text, re.MULTILINE)
        return m.group(1).strip() if m else None

    if (v := grab("display_name")):
        meta["name"] = v
    if (v := grab("default_emoji")):
        meta["emoji"] = v
    if (v := grab("card_accent")):
        meta["color"] = v
    return meta


def parse_record(raw: str) -> dict | None:
    parts = (raw.split("|", 3) + ["", "", "", ""])[:4]
    state, bill, date, action = (p.strip() for p in parts)
    if not state or not bill or not DATE_RE.match(date):
        return None
    return {"state": state, "bill": bill, "date": date, "action": action}


STATE_RE = re.compile(r"state:([a-z]{2})")


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    # Legislative titles are often SHOUTED in all caps — make them readable.
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        text = text.title()
    return text


def feed_item(path: Path, folder: str, topic_key: str) -> dict | None:
    """Turn one bills_raw/*.json record into a display-ready feed card."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    bill = d.get("bill", {})
    action = d.get("log", {}).get("action", {})
    src = d.get("sources", {}).get("bill", "")
    m = STATE_RE.search(src)
    state = (m.group(1).upper() if m else path.name[:2].upper())
    identifier = bill.get("identifier") or d.get("id") or ""
    date = (action.get("date") or d.get("timestamp") or "")[:10]
    if not DATE_RE.match(date):
        # timestamps like 20260618T000000Z -> 2026-06-18
        ts = (d.get("timestamp") or "")
        if len(ts) >= 8 and ts[:8].isdigit():
            date = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
    if not identifier or not DATE_RE.match(date):
        return None
    title = clean(bill.get("title", ""))
    abstract = ""
    for a in bill.get("abstracts", []) or []:
        cand = clean(a.get("abstract", ""))
        if len(cand) >= 60:           # skip one-word subject tags like "Consumer Protection"
            abstract = cand
            break
    return {
        "platform": folder,
        "topic": topic_key,
        "state": state,
        "bill": identifier,
        "title": title,
        "summary": abstract,
        "action": clean(action.get("description", "")),
        "date": date,
        # Present only for posts made after the "save exactly as posted" change:
        # the verbatim post text, the bill link, and the URL of the live post.
        "posted": (d.get("posted_text") or "").strip(),
        "link": (d.get("posted_link") or "").strip(),
        "post_url": (d.get("posted_url") or "").strip(),
    }


def git_commit_dates() -> dict:
    """Map each bills_raw/*.json path -> the date its file was committed
    (YYYY-MM-DD) — i.e. roughly when the bill was actually posted. The bot
    commits each post's artifact right after posting, so this orders the feed by
    real post day rather than by the bill's (sometimes far-future) action date.

    Returns {} on any failure; callers fall back to the action date. Works best
    with full git history (the Pages workflow checks out with fetch-depth: 0)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "--format=%cs", "--name-only",
             "--", ":(glob)topics/**/bills_raw/*.json"],
            capture_output=True, text=True, timeout=180, check=True,
        ).stdout
    except Exception:
        return {}
    dates: dict = {}
    cur = ""
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if DATE_RE.match(line):
            cur = line
        elif cur and line.endswith(".json"):
            # git log is newest-first; overwriting leaves the oldest (adding)
            # commit date, which is when the post first landed.
            dates[line] = cur
    return dates


def discover_platforms() -> list[str]:
    """Every distinct platform folder name that holds a bills_used.json."""
    found: set[str] = set()
    for used in TOPICS_DIR.glob("*/*/bills_used.json"):
        found.add(used.parent.name)
    ordered = [p for p in PLATFORM_ORDER if p in found]
    ordered += sorted(f for f in found if f not in PLATFORM_ORDER)
    return ordered


def main() -> None:
    platform_folders = discover_platforms()
    platforms = []
    fallback_i = 0
    for folder in platform_folders:
        name, color = platform_display(folder, fallback_i)
        if folder.lower() not in PLATFORM_INFO:
            fallback_i += 1
        platforms.append({"key": folder, "name": name, "color": color})

    commit_dates = git_commit_dates()

    topics = []
    records: list[dict] = []
    feed: list[dict] = []
    bluesky_accounts = []
    for topic_dir in sorted(p for p in TOPICS_DIR.iterdir() if p.is_dir()):
        meta = load_topic_meta(topic_dir)
        topics.append({"key": topic_dir.name, **meta})
        handle = BLUESKY_HANDLES.get(topic_dir.name)
        if handle:
            bluesky_accounts.append({"handle": handle, "topic": topic_dir.name, **meta})
        for folder in platform_folders:
            used = topic_dir / folder / "bills_used.json"
            if used.exists():
                try:
                    posted = json.loads(used.read_text(encoding="utf-8")).get("posted", [])
                except (json.JSONDecodeError, OSError):
                    posted = []
                for raw in posted:
                    rec = parse_record(raw)
                    if rec:
                        rec.update(topic=topic_dir.name, platform=folder)
                        records.append(rec)
            # Richer per-post content for the feed lives in bills_raw/*.json.
            raw_dir = topic_dir / folder / "bills_raw"
            if raw_dir.is_dir():
                for raw_file in raw_dir.glob("*.json"):
                    item = feed_item(raw_file, folder, topic_dir.name)
                    if item:
                        rel = str(raw_file.relative_to(REPO))
                        # Post day = commit date of the artifact, else action date.
                        item["posted_date"] = commit_dates.get(rel) or item["date"]
                        feed.append(item)

    # Only keep topics that actually have posts; keep platform order as discovered.
    posted_topics = {r["topic"] for r in records}
    topics = [t for t in topics if t["key"] in posted_topics]
    posted_platforms = {r["platform"] for r in records}
    platforms = [p for p in platforms if p["key"] in posted_platforms]

    records.sort(key=lambda r: r["date"], reverse=True)

    # De-duplicate feed items (same bill+date+action can be saved under both the
    # per-topic aggregate and the platform folder). Order by real post day
    # (commit date), newest first, then by action date, and cap it.
    seen = set()
    feed.sort(key=lambda f: (f.get("posted_date") or f["date"], f["date"]), reverse=True)
    deduped = []
    for f in feed:
        k = (f["platform"], f["state"], f["bill"], f["date"], f["action"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(f)
    feed = deduped[:500]

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "platforms": platforms,
        "topics": topics,
        "records": records,
        "feed": feed,
        "bluesky": bluesky_accounts,
    }

    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "data.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)} — {len(records)} posts across "
          f"{len(platforms)} platform(s): "
          f"{', '.join(p['name'] for p in platforms)}; {len(topics)} topics; "
          f"{len(feed)} feed cards.")


if __name__ == "__main__":
    main()
