#!/usr/bin/env python3
"""Offline checks for the v2 Instagram card: the latest-action -> timeline stage
mapping on real action text from several states, and the rendered file's size
and format. Run: python3 scripts/test_render_bill_card_v2.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_bill_card_v2 import (  # noqa: E402
    STAGES, bill_stage, classify_stage, render_card_v2,
)

I, C, P, G, L = range(5)

# (state, action text as the feed records it, expected stage). Every one is a
# real action from samples/ or a topic's bills_raw; most were misread before.
CASES = [
    ("TN", "Assigned to s/c Civil Justice Subcommittee", C),       # not "signed"
    ("TN", "Comp. became Pub. Ch. 1110", L),
    ("NJ", "WITHDRAWN APP P.L.2025, c.343.", L),
    ("NJ", "Approved P.L., 2026. c.96.", L),
    ("GA", "Act 415", L),
    ("PR", "Ley 44-2026", L),
    ("NY", "SIGNED CHAP.192", L),
    ("MI", "filed with Secretary of State 07/23/2026 11:48 AM", L),
    ("RI", "07/05/2025 Effective without Governor's signature", L),
    ("DE", "Signed by Governor with Line Item Veto", L),
    ("NC", "Veto Overridden", L),
    ("US", "Signed by President.", L),
    ("IL", "Governor Approved", L),
    ("HI", "Vetoed on 07/15/2026 - Returned from the Governor without approval (Gov. Msg. No. 1369).", G),
    ("MA", "Enacted and laid before the Governor", G),
    ("FL", "Signed by Officers and presented to Governor", G),
    ("NC", "Ratified", G),
    ("PR", "Enviado a la Gobernadora", G),
    ("CA", "From committee: Do pass as amended and re-refer to Com. on L. GOV. (Ayes 11. Noes 0.) (June 16)", C),
    ("PA", "Signed in Senate", P),
    ("UT", "Senate/ signed by President/ returned to House", P),
    ("WA", "Speaker signed.", P),
    ("AL", "Signature Requested", P),
    ("PR", "Aprobado por el Senado en Votación Final", P),
    ("PR", "Texto de Aprobación Final enviado al Senado", P),
    ("CA", "Read third time. Passed. Ordered to the Senate. (Ayes 58. Noes 16.)", P),
    ("US", "Submitted in the Senate, considered, and agreed to without amendment by Unanimous Consent.", P),
    ("AL", "Third Reading in Second House", P),
    ("FL", "Died in returning Messages", P),
    ("PA", "Reported as committed", C),
    ("MO", "HCS Reported Do Pass (H) - AYES: 5 NOES: 3 PRESENT: 2", C),
    ("KS", "Committee Report recommending bill be passed as amended by House Committee on Federal and State Affairs", C),
    ("MA", "Committee reported ought to be adopted", C),
    ("WA", "Passed to Rules Committee for second reading.", C),
    ("TN", "Passed on Second Consideration, refer to Senate Commerce and Labor Committee", C),
    ("WI", "Second reading, failed to pass, yeas 10 nays 20", C),
    ("MO", "SA 1 to SS for SCS S offered & adopted (Coleman)--(4822S07.23S)", C),
    ("NJ", "Assembly Floor Amendment Passed (Quijano) (Voice)", C),
    ("FL", "Died in Commerce and Tourism", C),
    ("US", "Read twice and referred to the Committee on Finance.", C),
    ("PR", "Referido a Comisión(es)", C),
    ("CA", "Introduced. Read first time. To Com. on RLS. for assignment. To print.", I),
    ("PR", "Aparece en Primera Lectura del Senado", I),
    ("ME", "READ ONCE.", I),
]


def raw(start: str, acted: str) -> dict:
    return {"bill": {"from_organization": f'~{{"classification": "{start}"}}'},
            "log": {"action": {"jurisdiction_id": f'~{{"classification": "{acted}"}}'}}}


def main() -> int:
    failures = 0
    for state, text, want in CASES:
        got, rule = classify_stage(text, state=state)
        if got != want:
            failures += 1
            print(f"FAIL {state} {text!r}: {STAGES[got]} ({rule}), want {STAGES[want]}")

    # A referral in the other chamber means the bill passed its first one.
    b = {"state": "PA", "action_desc": "Referred to Labor & Industry",
         "_raw": raw("lower", "upper")}
    if bill_stage(b) != (P, "action in second chamber"):
        failures += 1
        print(f"FAIL second-chamber referral: {bill_stage(b)}")
    # ...but a governor action is still the governor's, and the same chamber
    # is just a committee referral.
    b["action_desc"] = "Approved by the Governor"
    if bill_stage(b)[0] != L:
        failures += 1
        print(f"FAIL second-chamber law: {bill_stage(b)}")
    b.update(action_desc="Referred to Labor & Industry", _raw=raw("lower", "lower"))
    if bill_stage(b)[0] != C:
        failures += 1
        print(f"FAIL same-chamber referral: {bill_stage(b)}")
    # California without the classification: the bill number says where it began.
    ca = {"state": "CA", "identifier": "AB 2460",
          "action_desc": "In Senate. Read first time. To Com. on RLS. for assignment."}
    if bill_stage(ca)[0] != P or bill_stage({**ca, "identifier": "SB 24"})[0] != I:
        failures += 1
        print(f"FAIL California chamber from bill number: {bill_stage(ca)}")
    # Unknown wording falls back to Introduced with no rule name (it's logged).
    if classify_stage("Added as Co-Sponsor Sen. Sue Rezin", state="IL") != (I, ""):
        failures += 1
        print("FAIL co-sponsor fallback")

    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        bill = {"state": "DC", "identifier": "B26-0123",
                "action_desc": "Introduced", "action_date": "2026-10-02"}
        out = render_card_v2(bill, text="Would cap yearly rent increases.",
                             emoji="\U0001F3E0", topic_label="Housing",
                             accent=(234, 88, 12), out_path=Path(tmp) / "c.jpg")
        im = Image.open(out)
        if (im.format, im.size, im.mode) != ("JPEG", (1080, 1350), "RGB"):
            failures += 1
            print(f"FAIL card file: {im.format} {im.size} {im.mode}")

    total = len(CASES) + 6
    print(f"{total - failures}/{total} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
