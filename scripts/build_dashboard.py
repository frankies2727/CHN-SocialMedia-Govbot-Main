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
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOPICS_DIR = REPO / "topics"
DOCS_DIR = REPO / "docs"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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

    topics = []
    records: list[dict] = []
    for topic_dir in sorted(p for p in TOPICS_DIR.iterdir() if p.is_dir()):
        meta = load_topic_meta(topic_dir)
        topics.append({"key": topic_dir.name, **meta})
        for folder in platform_folders:
            used = topic_dir / folder / "bills_used.json"
            if not used.exists():
                continue
            try:
                posted = json.loads(used.read_text(encoding="utf-8")).get("posted", [])
            except (json.JSONDecodeError, OSError):
                continue
            for raw in posted:
                rec = parse_record(raw)
                if rec:
                    rec.update(topic=topic_dir.name, platform=folder)
                    records.append(rec)

    # Only keep topics that actually have posts; keep platform order as discovered.
    posted_topics = {r["topic"] for r in records}
    topics = [t for t in topics if t["key"] in posted_topics]
    posted_platforms = {r["platform"] for r in records}
    platforms = [p for p in platforms if p["key"] in posted_platforms]

    records.sort(key=lambda r: r["date"], reverse=True)

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "platforms": platforms,
        "topics": topics,
        "records": records,
    }

    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "data.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)} — {len(records)} posts across "
          f"{len(platforms)} platform(s): "
          f"{', '.join(p['name'] for p in platforms)}; {len(topics)} topics.")


if __name__ == "__main__":
    main()
