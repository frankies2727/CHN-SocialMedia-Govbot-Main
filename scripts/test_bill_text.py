#!/usr/bin/env python3
"""Offline checks for bill_text: the newest version wins, and a pre-extracted
file for an earlier, different bill (gutted and refilled) is recognised as
stale. Run: python3 scripts/test_bill_text.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bill_text  # noqa: E402

# California's SB 259 as govbot records it: a navigation page without a date,
# the introduced text, nine amendments, the August 2026 gut-and-amend, the
# enrolled version.
SB259 = {
    "identifier": "SB 259",
    "title": "Elections: vote by mail ballots.",
    "versions": [
        {"note": "SB259", "links": [{"url": "http://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202520260SB259", "media_type": "text/html"}]},
        {"date": "2025-02-03", "note": "02/03/25 - Introduced", "links": [{"url": "https://leginfo.legislature.ca.gov/faces/billPdf.xhtml?bill_id=202520260SB259&version=20250SB25999INT", "media_type": "application/pdf"}]},
        {"date": "2025-09-09", "note": "09/09/25 - Amended Assembly", "links": [{"url": "https://leginfo.legislature.ca.gov/faces/billPdf.xhtml?bill_id=202520260SB259&version=20250SB25991AMD", "media_type": "application/pdf"}]},
        {"date": "2026-08-20", "note": "08/20/26 - Amended Assembly", "links": [{"url": "https://leginfo.legislature.ca.gov/faces/billPdf.xhtml?bill_id=202520260SB259&version=20250SB25990AMD", "media_type": "application/pdf"}]},
        {"date": "2026-08-30", "note": "08/30/26 - Enrolled", "links": [{"url": "https://leginfo.legislature.ca.gov/faces/billPdf.xhtml?bill_id=202520260SB259&version=20250SB25989ENR", "media_type": "application/pdf"}]},
    ],
}
STALE_HEADER = ("Title: Bill Text - SB-259 Fair Online Pricing Act.\nOfficial Title: \n"
                "Source: versions - SB259\nMedia Type: text/html\n\n" + "=" * 80 + "\n\n"
                "Bill Text - SB-259 Fair Online Pricing Act. This bill, the Fair Online Pricing Act, would prohibit a price offered to a consumer through the consumer's online device from being generated based on certain input data. " * 3)
FRESH_HEADER = ("Title: Bill Text - SB-259 Elections: vote by mail ballots.\nOfficial Title: \n"
                "Source: versions - SB259\nMedia Type: text/html\n\n" + "=" * 80 + "\n\n"
                "Existing law makes it a misdemeanor for any person having charge of a completed vote by mail ballot to interfere with its return. " * 3)


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        sys.exit(1)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bill = Path(tmp) / "bills" / "SB259"
        (bill / "files").mkdir(parents=True)
        (bill / "metadata.json").write_text(json.dumps(SB259), encoding="utf-8")

        # The newest version's PDF, not the undated nav page nor the introduced text.
        link = bill_text.find_document_link(bill / "metadata.json")
        check("newest version wins", link == (SB259["versions"][-1]["links"][0]["url"], "application/pdf"))

        # A file from before the gut-and-amend is stale; one matching the title is not.
        check("old title is stale", bill_text.extracted_text_is_stale(STALE_HEADER, SB259["title"]))
        check("current title is not stale", not bill_text.extracted_text_is_stale(FRESH_HEADER, SB259["title"]))
        check("no header is never stale", not bill_text.extracted_text_is_stale("just some text", SB259["title"]))
        check("an ordinary amendment keeps the file",
              not bill_text.extracted_text_is_stale("Title: Bill Text - SB-259 Elections: vote by mail ballots, signature verification.\n\nx", SB259["title"]))

        # End to end, offline: a stale file is skipped (the download path then runs;
        # with no network the result is abstract-only, never the stale text).
        (bill / "files" / "billNavClient_SB259_extracted.txt").write_text(STALE_HEADER, encoding="utf-8")
        bill_text._run_memo.clear()
        bill_text._download = lambda url: None  # no network in the test
        text, reason = bill_text.extract_bill_text_verbose("x-legislation/bills/SB259/metadata.json", root=Path(tmp))
        check("stale file is not used", text is None and reason == "download-failed")

        # A matching file is used as before.
        (bill / "files" / "billNavClient_SB259_extracted.txt").write_text(FRESH_HEADER, encoding="utf-8")
        bill_text._run_memo.clear()
        text, reason = bill_text.extract_bill_text_verbose("x-legislation/bills/SB259/metadata.json", root=Path(tmp))
        check("fresh file is used", reason == "ok-extracted-file" and text is not None and "vote by mail" in text)
    print("all good")


if __name__ == "__main__":
    main()
