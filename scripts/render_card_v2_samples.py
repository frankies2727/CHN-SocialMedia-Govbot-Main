#!/usr/bin/env python3
"""Render review samples of the v2 Instagram card, offline, into one folder:

  * one card per topic in topics/, from a real bill this repo already posted
    (its state, number, latest action and date), cycling the timeline stages;
  * edge cases: the longest state name (District of Columbia), U.S. Congress,
    a territory with no full name on file, very short and very long text, and
    every timeline stage.

There is no language model here, so the description is the bill's own title or
abstract trimmed to two sentences (the topic cards) or a hand-written line (the
edge cases): these check the layout, not the wording. Nothing is posted and
nothing is written under topics/.

    python3 scripts/render_card_v2_samples.py OUT_DIR
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("BOT_TOPIC", "healthcare")  # post_to_bluesky loads a topic on import

from PIL import Image  # noqa: E402

import post_to_bluesky as pb  # noqa: E402
from render_bill_card_v2 import STAGES, bill_stage, render_card_v2  # noqa: E402
from topic import Topic, list_topics  # noqa: E402

EDGE = [
    ("DC", "B26-0123", "Introduced", "2026-10-02", "Housing",
     "Would cap yearly rent increases for most apartments at the rate of inflation "
     "plus two percent. Landlords who charge more could be fined."),
    ("NJ", "A4794", "Referred to Assembly Housing Committee", "2026-09-14", "Housing",
     "Would cap rent hikes."),
    ("CA", "AB 1655", "Read third time. Passed. Ordered to the Senate. (Ayes 58. Noes 16.)",
     "2026-06-01", "Immigration",
     "Would let counties keep giving cash aid and food help to families while their "
     "immigration paperwork is pending, instead of cutting them off after six months. "
     "Counties would have to tell families about the change in writing within thirty "
     "days, in their own language, and report every year how many families kept their "
     "benefits because of it."),
    ("MA", "H 5417", "Enacted and laid before the Governor", "2026-11-12", "Taxation",
     "Would let Lynnfield add a local tax of up to 0.75% on restaurant meals. The town "
     "would vote on it first."),
    ("NY", "S 10680", "SIGNED CHAP.312", "2026-11-03", "AI, Data Centers & Crypto",
     "Requires state agencies to tell people when an AI system helped decide their "
     "benefits. People can ask for a human to review the decision."),
    ("MP", "HB 24-61", "House First Reading", "2026-10-20", "Education",
     "Would require schools to post their safety plans online."),
    ("US", "HR 10879", "Referred to the House Committee on Education and Workforce.",
     "2026-10-30", "Labor & Workers' Rights",
     "Would ban noncompete agreements for workers who earn under the income limits it sets."),
]


def _topic_by_label() -> dict[str, Topic]:
    return {t.display_name: t for t in (Topic.load(n) for n in list_topics())}


def _text_from_record(b: dict) -> str:
    text = " ".join(pb.best_display_text(b).split())
    sentences = pb._split_sentences(text)[:2]
    return " ".join(sentences) or text


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[Path, str]] = []
    for i, name in enumerate(list_topics()):
        topic = Topic.load(name)
        want = i % len(STAGES)
        pick = None
        for f in sorted(glob.glob(str(ROOT / f"topics/{name}/*/bills_raw/*.json"))):
            raw = json.loads(Path(f).read_text())
            b = pb.extract_fields(raw)
            if not b:
                continue
            b["_raw"] = raw
            pick = pick or b
            if bill_stage(b)[0] == want:
                pick = b
                break
        if not pick:
            print(f"  {name}: no posted bill on file, skipped")
            continue
        path = out / f"topic-{name}.jpg"
        render_card_v2(pick, text=_text_from_record(pick), emoji=topic.default_emoji,
                       topic_label=topic.display_name, accent=topic.card_accent,
                       out_path=path)
        stage, rule = bill_stage(pick)
        rendered.append((path, f"{name}: {pick['state']} {pick['identifier']} — "
                               f"{STAGES[stage]} ({rule or 'fallback'}) from "
                               f"{pick['action_desc'][:60]!r}"))

    topics = _topic_by_label()
    for i, (state, ident, action, date, label, text) in enumerate(EDGE):
        t = topics[label]
        bill = {"state": state, "identifier": ident, "action_desc": action,
                "action_date": date}
        path = out / f"edge-{i}-{state}.jpg"
        render_card_v2(bill, text=text, emoji=t.default_emoji, topic_label=label,
                       accent=t.card_accent, out_path=path)
        rendered.append((path, f"edge {state} {ident}: {STAGES[bill_stage(bill)[0]]}, "
                               f"{len(text.split())} words"))

    # One contact sheet to skim them all.
    w, h, cols = 360, 450, 5
    rows = -(-len(rendered) // cols)
    sheet = Image.new("RGB", (w * cols, h * rows), "white")
    for i, (path, _) in enumerate(rendered):
        sheet.paste(Image.open(path).resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(out / "contact-sheet.jpg", quality=85)
    for path, note in rendered:
        print(f"{path.name}: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "card-v2-samples")))
