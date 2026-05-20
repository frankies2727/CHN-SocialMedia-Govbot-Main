#!/usr/bin/env python3
"""
X/Twitter version of the poster.
Reuses ALL your existing logic from category.py, Ollama, dedup, etc.
Just posts to X instead of Bluesky.
"""

import os
import sys
from pathlib import Path

import tweepy
from category import Category, load_active_category

# ------------------------------------------------------------------
# Load category config (exactly like your Bluesky script)
# ------------------------------------------------------------------
CATEGORY: Category = load_active_category()
ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = ROOT / "bills.jsonl"

POST_LIMIT = int(os.environ.get("POST_LIMIT", "3"))   # Start conservative for X costs
DRY_RUN = os.environ.get("DRY_RUN") == "1"
MAX_ACTION_AGE_DAYS = int(os.environ.get("MAX_ACTION_AGE_DAYS", "150"))

# X credentials from GitHub Secrets
X_API_KEY = os.environ.get("X_API_KEY")
X_API_SECRET = os.environ.get("X_API_SECRET")
X_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.environ.get("X_ACCESS_TOKEN_SECRET")

print("🔍 Checking X credentials...")
print(f"X_API_KEY present: {bool(X_API_KEY) and len(X_API_KEY) > 10}")
print(f"X_API_SECRET present: {bool(X_API_SECRET) and len(X_API_SECRET) > 10}")
print(f"X_ACCESS_TOKEN present: {bool(X_ACCESS_TOKEN) and len(X_ACCESS_TOKEN) > 10}")
print(f"X_ACCESS_TOKEN_SECRET present: {bool(X_ACCESS_TOKEN_SECRET) and len(X_ACCESS_TOKEN_SECRET) > 10}")

if not all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET]):
    print("❌ ERROR: Missing X API credentials")
    sys.exit(1)

print("✅ All X credentials loaded successfully!")


# Initialize Tweepy client (X's official way)
client = tweepy.Client(
    consumer_key=X_API_KEY,
    consumer_secret=X_API_SECRET,
    access_token=X_ACCESS_TOKEN,
    access_token_secret=X_ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

# ------------------------------------------------------------------
# Reuse your existing functions (we import what we need)
# ------------------------------------------------------------------
# Copy-paste only the parts we need from post_to_bluesky.py to keep it clean

def load_bills(path: Path):
    import json
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return []
    bills = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    bills.append(json.loads(line))
                except:
                    continue
    print(f"Loaded {len(bills)} bills")
    return bills

def extract_fields(record: dict):
    # Minimal version - you can copy more from your original if needed
    bill = record.get("bill") or {}
    log = record.get("log") or {}
    identifier = bill.get("identifier") or ""
    title = bill.get("title") or ""
    if not identifier or not title:
        return None
    state = ""  # your detect_state function if you want full power
    action = log.get("action") or {}
    action_desc = action.get("description") or ""
    action_date = (action.get("date") or "")[:10]
    dedup_key = f"{state}|{identifier}|{action_date}|{action_desc[:40]}"
    return {
        "state": state,
        "identifier": identifier,
        "title": title,
        "action_desc": action_desc,
        "action_date": action_date,
        "dedup_key": dedup_key,
    }

# ------------------------------------------------------------------
# X Posting Function
# ------------------------------------------------------------------
def post_to_x(text: str, reply_to_id: str = None):
    if DRY_RUN:
        print(f"🔹 [DRY RUN] Would post to X:\n{text[:500]}...\n")
        return "DRY_RUN_ID"

    try:
        response = client.create_tweet(
            text=text,
            in_reply_to_tweet_id=reply_to_id
        )
        tweet_id = response.data['id']
        print(f"✅ Posted to X! https://x.com/i/web/status/{tweet_id}")
        return tweet_id
    except Exception as e:
        print(f"❌ Failed to post: {e}")
        return None

# ------------------------------------------------------------------
# Main Logic (simplified starter version)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== X GovBot running for category: {CATEGORY.name} ===")
    
    bills = load_bills(JSONL_PATH)
    posted_count = 0
    state_file = CATEGORY.state_file_path()  # reuses your bills_used.json

    # Load already posted keys
    import json
    if state_file.exists():
        with state_file.open() as f:
            used = json.load(f).get("posted", [])
    else:
        used = []

    for record in bills:
        b = extract_fields(record)
        if not b:
            continue
        if b["dedup_key"] in used:
            continue
        if not CATEGORY.matches(b):   # your keyword filter
            continue

        # Build nice post (you can make this fancier later)
        emoji = CATEGORY.emoji_for(b) if hasattr(CATEGORY, 'emoji_for') else "📜"
        title = b["title"][:180] + "..." if len(b["title"]) > 180 else b["title"]
        action = b.get("action_desc", "")[:120]

        post_text = f"{emoji} {b['identifier']} — {title}\n\n{action}\n\n#StateBills #DataForGood"

        post_id = post_to_x(post_text)
        if post_id and not DRY_RUN:
            used.append(b["dedup_key"])
            posted_count += 1

        if posted_count >= POST_LIMIT:
            break

    # Save updated dedup state
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w") as f:
        json.dump({"posted": used}, f, indent=2)

    print(f"Finished. Posted {posted_count} updates.")
