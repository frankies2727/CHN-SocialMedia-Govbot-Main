#!/usr/bin/env python3
"""
Refresh the long-lived Instagram access token so it never lapses.

This uses the Instagram API with Instagram Login (graph.instagram.com), whose
long-lived tokens behave exactly like Threads': valid for 60 days, refreshable
any time after they're 24 hours old, which rolls the 60-day window forward.
Running this on a schedule (see .github/workflows/instagram-refresh-token.yml)
keeps the token alive as long as the workflow runs at least once every ~50 days.

Persisting the refreshed token back into the INSTAGRAM_ACCESS_TOKEN repo secret
requires updating a GitHub Actions secret, which the default GITHUB_TOKEN cannot
do. To enable write-back, provide a Personal Access Token with `secrets: write`
permission on this repo via the INSTAGRAM_REFRESH_PAT env var. Without it the
script still calls the refresh endpoint (which extends validity server-side) and
reports the new expiry, but does NOT print the token — a fresh access token must
never end up in plaintext CI logs.

Env vars:
    INSTAGRAM_ACCESS_TOKEN    (required) current long-lived token
    INSTAGRAM_REFRESH_PAT     (optional) PAT with secrets:write for write-back
    GITHUB_REPOSITORY         (provided by Actions) "owner/repo" for write-back
"""

from __future__ import annotations

import base64
import os
import sys

import requests

REFRESH_URL = "https://graph.instagram.com/refresh_access_token"
TIMEOUT = 30
SECRET_NAME = "INSTAGRAM_ACCESS_TOKEN"


def refresh(token: str) -> tuple[str, int]:
    """Call the Instagram refresh endpoint. Returns (new_token, expires_in_seconds)."""
    resp = requests.get(
        REFRESH_URL,
        params={"grant_type": "ig_refresh_token", "access_token": token},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    new_token = data.get("access_token", "")
    expires_in = int(data.get("expires_in", 0))
    if not new_token:
        raise RuntimeError(f"refresh response had no access_token: {data}")
    return new_token, expires_in


def update_repo_secret(repo: str, pat: str, name: str, value: str) -> None:
    """Encrypt `value` with the repo's Actions public key and PUT it as the
    named secret. Requires PyNaCl (libsodium) — imported lazily so the refresh
    path still works when write-back isn't configured."""
    try:
        from nacl import encoding, public
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "PyNaCl is required to write back the secret. "
            "Add `PyNaCl` to requirements.txt or install it."
        ) from e

    api = f"https://api.github.com/repos/{repo}/actions/secrets"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    key_resp = requests.get(f"{api}/public-key", headers=headers, timeout=TIMEOUT)
    key_resp.raise_for_status()
    key_data = key_resp.json()
    public_key = public.PublicKey(key_data["key"].encode(), encoding.Base64Encoder())
    sealed = public.SealedBox(public_key).encrypt(value.encode())
    encrypted_value = base64.b64encode(sealed).decode()

    put_resp = requests.put(
        f"{api}/{name}",
        headers=headers,
        json={"encrypted_value": encrypted_value, "key_id": key_data["key_id"]},
        timeout=TIMEOUT,
    )
    put_resp.raise_for_status()


def main() -> int:
    token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
    if not token:
        print("ERROR: INSTAGRAM_ACCESS_TOKEN is not set.", file=sys.stderr)
        return 1

    try:
        new_token, expires_in = refresh(token)
    except Exception as e:
        print(f"ERROR: token refresh failed: {e}", file=sys.stderr)
        if getattr(e, "response", None) is not None:
            print(f"  Response body: {e.response.text}", file=sys.stderr)
        return 1

    days = expires_in // 86400
    print(f"Refreshed Instagram token; new validity ~{days} days ({expires_in}s).")

    pat = os.environ.get("INSTAGRAM_REFRESH_PAT", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not pat:
        print(
            "No INSTAGRAM_REFRESH_PAT set — skipping secret write-back. The refresh "
            "call above already extended the token's window; as long as this "
            "workflow runs at least every ~50 days the token stays valid. For "
            "guaranteed persistence, add a PAT with secrets:write as "
            "INSTAGRAM_REFRESH_PAT."
        )
        return 0
    if not repo:
        print("ERROR: GITHUB_REPOSITORY not set; cannot write back secret.", file=sys.stderr)
        return 1

    try:
        update_repo_secret(repo, pat, SECRET_NAME, new_token)
    except Exception as e:
        print(f"ERROR: writing {SECRET_NAME} secret failed: {e}", file=sys.stderr)
        return 1
    print(f"Wrote refreshed token back to the {SECRET_NAME} repo secret.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
