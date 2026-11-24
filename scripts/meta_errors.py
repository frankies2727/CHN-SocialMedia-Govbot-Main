"""Stop a run as soon as Meta rejects the access token itself.

Meta's Graph APIs (Threads and Instagram) answer a revoked or expired token with
error code 190 ("Error validating access token ..."). Retrying, moving on to the
next topic or refreshing can't fix that: someone has to log in again and save a
new token in the repo secret. Without this check the posters logged the error,
carried on through every topic, and the runs ended green or "cancelled" (see
govbot-meta-threads-post #96 and govbot-meta-threads-weekly-digest #26).

stop_if_token_rejected() exits with TOKEN_REJECTED_EXIT, which the workflows
turn into a red run with a clear message. It prints only Meta's error message,
never the request URL (which can carry the token).
"""

from __future__ import annotations

import sys

TOKEN_REJECTED_EXIT = 3
TOKEN_ERROR_CODE = 190


def token_error_message(resp) -> str | None:
    """Meta's message if `resp` is a Graph error 190, else None."""
    if resp is None:
        return None
    try:
        err = resp.json().get("error") or {}
    except (ValueError, AttributeError):
        return None
    if not isinstance(err, dict) or err.get("code") != TOKEN_ERROR_CODE:
        return None
    return " ".join(str(err.get("message") or "no message").split())


def stop_if_token_rejected(resp, secret_name: str) -> None:
    """Exit the run if `resp` says the access token was rejected (error 190)."""
    message = token_error_message(resp)
    if message is None:
        return
    # ::error:: puts the message on the run's summary page in GitHub Actions.
    print(f"::error title=Meta rejected {secret_name}::{message} "
          f"The token is revoked or expired and refreshing can't fix it. "
          f"Generate a new token and save it in the {secret_name} secret "
          f"(README, Troubleshooting: 'Meta rejected …').", file=sys.stderr)
    sys.exit(TOKEN_REJECTED_EXIT)
