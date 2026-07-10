#!/usr/bin/env python3
"""Build the data file that powers the GitHub Pages dashboard.

Reads every platform's posted records out of the repo and rolls them up into a
single ``docs/data.json`` that ``docs/index.html`` renders as a dark-mode
dashboard. Run it locally with ``python scripts/build_dashboard.py`` or let the
GitHub Actions workflow run it automatically on every push.

No third-party packages are required: PyYAML is used if present, otherwise the
three fields we need are pulled from each ``config.yml`` with a small regex.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOPICS_DIR = REPO / "topics"
DOCS_DIR = REPO / "docs"

# Platforms we surface, in display order. Add a new one here (key -> label) and
# the dashboard picks it up automatically once its bills_used.json files exist.
PLATFORMS = {
    "instagram": "Instagram",
    "meta-threads": "Meta Threads",
    "bluesky": "Bluesky",
    "x": "X",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_topic_meta(topic_dir: Path) -> dict:
    """Return {display_name, emoji, color} for a topic from its config.yml."""
    cfg = topic_dir / "config.yml"
    name = topic_dir.name
    meta = {
        "name": name.replace("_", " ").title(),
        "emoji": "📄",
        "color": "#3987e5",
    }
    if not cfg.exists():
        return meta
    text = cfg.read_text(encoding="utf-8", errors="ignore")

    def grab(key: str) -> str | None:
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
    """Parse a 'STATE|BILL|DATE|action' string into a dict (or None if bad)."""
    parts = (raw.split("|", 3) + ["", "", "", ""])[:4]
    state, bill, date, action = (p.strip() for p in parts)
    if not state or not bill or not DATE_RE.match(date):
        return None
    return {"state": state, "bill": bill, "date": date, "action": action}


def main() -> None:
    records: list[dict] = []  # one row per posted item, tagged with topic+platform

    for topic_dir in sorted(p for p in TOPICS_DIR.iterdir() if p.is_dir()):
        meta = load_topic_meta(topic_dir)
        for pkey, pname in PLATFORMS.items():
            used = topic_dir / pkey / "bills_used.json"
            if not used.exists():
                continue
            try:
                posted = json.loads(used.read_text(encoding="utf-8")).get("posted", [])
            except (json.JSONDecodeError, OSError):
                continue
            for raw in posted:
                rec = parse_record(raw)
                if not rec:
                    continue
                rec.update(
                    topic=topic_dir.name,
                    topic_name=meta["name"],
                    emoji=meta["emoji"],
                    color=meta["color"],
                    platform=pkey,
                    platform_name=pname,
                )
                records.append(rec)

    # ---- roll-ups -------------------------------------------------------
    def bump(d: dict, key, amt=1):
        d[key] = d.get(key, 0) + amt

    by_platform: dict[str, int] = {}
    by_state: dict[str, int] = {}
    by_month: dict[str, int] = {}
    topics: dict[str, dict] = {}

    for r in records:
        bump(by_platform, r["platform"])
        bump(by_state, r["state"])
        bump(by_month, r["date"][:7])
        t = topics.setdefault(
            r["topic"],
            {
                "key": r["topic"],
                "name": r["topic_name"],
                "emoji": r["emoji"],
                "color": r["color"],
                "count": 0,
                "platforms": {},
            },
        )
        t["count"] += 1
        bump(t["platforms"], r["platform"])

    dates = sorted(r["date"] for r in records)
    bills = {(r["state"], r["bill"]) for r in records}

    recent = sorted(records, key=lambda r: r["date"], reverse=True)[:80]

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "totals": {
            "posts": len(records),
            "platforms": len([p for p in by_platform if by_platform[p]]),
            "topics": len([t for t in topics.values() if t["count"]]),
            "states": len(by_state),
            "bills": len(bills),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
        },
        "platforms": [
            {"key": k, "name": PLATFORMS[k], "count": by_platform[k]}
            for k in PLATFORMS
            if by_platform.get(k)
        ],
        "topics": sorted(topics.values(), key=lambda t: t["count"], reverse=True),
        "states": sorted(
            ({"code": k, "count": v} for k, v in by_state.items()),
            key=lambda s: s["count"],
            reverse=True,
        ),
        "timeline": [
            {"month": m, "count": by_month[m]} for m in sorted(by_month)
        ],
        "recent": recent,
    }

    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "data.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)} — {len(records)} posts across "
          f"{data['totals']['platforms']} platform(s), "
          f"{data['totals']['topics']} topics, {data['totals']['states']} states.")


if __name__ == "__main__":
    main()
