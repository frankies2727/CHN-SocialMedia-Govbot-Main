#!/usr/bin/env python3
"""Offline checks that a revoked Meta token (Graph error 190) stops every
Threads/Instagram script at once with exit code 3, without retrying and
without printing the token, and that the Threads digest reports a failed root
post instead of claiming it posted. Every HTTP call is faked; nothing is sent.
Run: python3 scripts/test_meta_errors.py"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from unittest import mock

import requests

TOKEN = "FAKE-TOKEN-must-never-be-printed-0123456789"
os.environ.update({
    "BOT_TOPIC": "labor",
    "DRY_RUN": "0",
    "THREADS_ACCESS_TOKEN": TOKEN,
    "THREADS_USER_ID": "111",
    "INSTAGRAM_ACCESS_TOKEN": TOKEN,
    "INSTAGRAM_USER_ID": "222",
})
sys.path.insert(0, str(Path(__file__).resolve().parent))

import meta_errors  # noqa: E402
import post_to_instagram as pi  # noqa: E402
import post_to_meta_threads as pmt  # noqa: E402
import refresh_meta_threads_token as rmt  # noqa: E402
import weekly_digest_meta_threads as wdmt  # noqa: E402
print("--- imports done; checks follow ---")

REVOKED = {"error": {"message": "Error validating access token: The session has "
                     "been invalidated because the user changed their password.",
                     "type": "OAuthException", "code": 190, "error_subcode": 0}}
OTHER = {"error": {"message": "Unsupported request", "code": 100}}


class FakeResponse:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.text = str(body)
        # requests puts the full URL (query string and token included) in the
        # HTTPError text; mimic that so a leak would show up in the output.
        self.url = f"https://graph.example/refresh?access_token={TOKEN}"

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error for url: {self.url}",
                                     response=self)


failures: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        failures.append(name)


def run_expecting_exit(fn, *args):
    """Call fn; return (exit code or None, captured stderr+stdout)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stderr(out), contextlib.redirect_stdout(out):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


print("meta_errors")
check("190 is recognised", meta_errors.token_error_message(FakeResponse(400, REVOKED)) is not None)
check("other Graph errors are ignored", meta_errors.token_error_message(FakeResponse(400, OTHER)) is None)
check("non-JSON body is ignored", meta_errors.token_error_message(FakeResponse(502, ValueError("html"))) is None)
check("no response is ignored", meta_errors.token_error_message(None) is None)
code, out = run_expecting_exit(meta_errors.stop_if_token_rejected, FakeResponse(400, OTHER), "X")
check("non-190 does not exit", code is None and out == "")

posts: list[str] = []


def fake_post(status, body):
    def _post(url, *a, **k):
        posts.append(url)
        return FakeResponse(status, body)
    return _post


print("Threads poster")
posts.clear()
with mock.patch.object(pmt.requests, "post", fake_post(400, REVOKED)), \
        mock.patch.object(pmt.time, "sleep", lambda s: None):
    code, out = run_expecting_exit(pmt.publish_post, "hello")
check("revoked token exits 3", code == meta_errors.TOKEN_REJECTED_EXIT)
check("stops after the first request", len(posts) == 1)
check("message names the secret", "THREADS_ACCESS_TOKEN" in out)
check("token never printed", TOKEN not in out)

posts.clear()
with mock.patch.object(pmt.requests, "post", fake_post(400, OTHER)):
    code, out = run_expecting_exit(pmt.publish_post, "hello")
check("other errors still just fail the post", code is None and len(posts) == 1)

# A 190 on publish (container already created) must not be retried.
calls = iter([FakeResponse(200, {"id": "c1"})] + [FakeResponse(400, REVOKED)] * 5)
posts.clear()
with mock.patch.object(pmt.requests, "post", lambda url, **k: posts.append(url) or next(calls)), \
        mock.patch.object(pmt.time, "sleep", lambda s: None):
    code, _ = run_expecting_exit(pmt.publish_post, "hello")
check("190 on publish exits without retrying", code == 3 and len(posts) == 2)

print("Threads digest")
with mock.patch.object(wdmt, "publish_post", lambda *a, **k: None):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        result = wdmt.post_digest_thread("root", [("reply", "")])
check("failed root post returns False", result is False)
with mock.patch.object(wdmt, "publish_post", lambda *a, **k: "m1"), \
        mock.patch.object(wdmt.time, "sleep", lambda s: None):
    with contextlib.redirect_stdout(io.StringIO()):
        result = wdmt.post_digest_thread("root", [("reply", "")])
check("posted root returns True", result is True)

print("Instagram poster")
posts.clear()
with mock.patch.object(pi.requests, "post", fake_post(400, REVOKED)):
    code, out = run_expecting_exit(pi._create_container, "https://x/card.jpg", "cap", "alt")
check("container: revoked token exits 3", code == 3 and len(posts) == 1)
check("container: token never printed", TOKEN not in out)
posts.clear()
with mock.patch.object(pi.requests, "post", fake_post(400, REVOKED)), \
        mock.patch.object(pi.time, "sleep", lambda s: None):
    code, _ = run_expecting_exit(pi._publish_container, "c1")
check("publish: revoked token exits 3 without retrying", code == 3 and len(posts) == 1)

print("Threads token refresh")
with mock.patch.object(rmt.requests, "get", lambda *a, **k: FakeResponse(400, REVOKED)):
    code, out = run_expecting_exit(rmt.main)
check("revoked token exits 3", code == 3)
check("token never printed", TOKEN not in out)
with mock.patch.object(rmt.requests, "get", lambda *a, **k: FakeResponse(500, OTHER)):
    code, out = run_expecting_exit(lambda: sys.exit(rmt.main()))
check("other failure exits 1 without the URL", code == 1 and TOKEN not in out)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("All checks passed.")
