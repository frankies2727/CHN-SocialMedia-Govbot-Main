# govbot-bluesky

A multi-category Bluesky bot platform that posts new state-legislative bill activity with AI summaries, powered by [chihacknight/govbot](https://github.com/chihacknight/govbot). Runs on GitHub Actions for free.

Each **category** (transportation, immigration, taxation, AI/data centers, housing, education, criminal justice/policing, LGBTQ, healthcare, labor/workers' rights, reproductive rights, elections/voting rights, …) is its own Bluesky account with its own keyword list, emoji map, summary focus, and dedup state. All categories share the same workflow run, so adding a new bot doesn't multiply CI minutes.

## How it works

On a cron (every 6 hours by default), one workflow:

1. Installs and runs `govbot`, which clones state legislation repos and dumps `bills.jsonl`. **This is the slow step (~8 min)** and now runs once for all categories.
2. Installs [Ollama](https://ollama.com/) on the runner and pulls a small **Gemma** model (`gemma3:4b`) — summarization runs entirely on the runner, no third-party API key required.
3. Loops over `categories/*/`: for each one, `scripts/post_to_bluesky.py` filters `bills.jsonl` against the category's keywords, asks the local model for a one-sentence neutral summary, and posts to that category's Bluesky account with a clickable link.
4. Commits each `categories/<name>/bills_used.json` back to the repo so the next run knows what's already been posted.

A second workflow runs **every Friday at ~4 pm ET** (`weekly-digest.yml`) and posts a threaded weekly digest per category: a root post summarizing the week's activity plus up to 6 reply posts highlighting the most significant updates (signed into law, passed, vetoed, etc.). Bills are scored by action significance and capped at 2 per state to keep the digest broad. Configure via env vars in the workflow: `DIGEST_LOOKBACK_DAYS`, `DIGEST_MAX_HIGHLIGHTS`, `DIGEST_PER_STATE_CAP`.

## Setup

### 1. Use this repo as a template
Click **Use this template** on GitHub (or fork). Clone locally.

### 2. Add a `govbot.yml`
Run `govbot` locally once with no config — it launches a wizard that creates `govbot.yml` for you (pick states and tags). Commit the result.

If you'd rather skip the wizard, see the [govbot docs](https://chihacknight.github.io/govbot/).

### 3. Add repository secrets per category
In **Settings → Secrets and variables → Actions**, add **two secrets per category**:

| Secret | Value |
| --- | --- |
| `BLUESKY_HANDLE_<NAME>` | The category's handle, e.g. `chn-transportation.bsky.social` |
| `BLUESKY_APP_PASSWORD_<NAME>` | An app password from Bluesky **Settings → App Passwords** (not your main password!) |

`<NAME>` is the upper-case category folder name. So for `categories/transportation/`, the secrets are `BLUESKY_HANDLE_TRANSPORTATION` and `BLUESKY_APP_PASSWORD_TRANSPORTATION`. For `categories/ai_data_centers/`: `BLUESKY_HANDLE_AI_DATA_CENTERS` and `BLUESKY_APP_PASSWORD_AI_DATA_CENTERS`.

Summarization uses a local Gemma model via Ollama on the GitHub Actions runner, so no third-party LLM API key is needed.

### 4. Enable Actions
On the Actions tab, enable workflows. The first run can be triggered manually via **Run workflow** on `govbot-bluesky-post`.

## Adding a category

The whole point of the `categories/` layout is that adding a new bot is a drop-in. The shared workflow already loops every folder under `categories/`, so once these three steps are done the new bot goes live on the next cron tick — no Python or workflow edits required.

1. **Create the folder** `categories/<name>/` and add a `config.yml` (copy `categories/transportation/config.yml` as a starting point). Fill in the category's `keywords`, `emojis`, `prompt_topic`, and digest copy.
2. **Add Bluesky secrets** in repo settings: `BLUESKY_HANDLE_<NAME>` and `BLUESKY_APP_PASSWORD_<NAME>` (upper-case folder name, underscores preserved).
3. **Commit** the new folder. The next scheduled run picks it up.

To dry-run before committing:

```bash
BOT_CATEGORY=<name> DRY_RUN=1 python scripts/post_to_bluesky.py
```

## Configuration knobs

Edit `.github/workflows/post.yml`:

- `cron:` — change the schedule. `0 */6 * * *` is every 6 hours.
- `POST_LIMIT` — max posts per run **per category** (default 4). Prevents flooding if many bills land at once.

Edit `scripts/post_to_bluesky.py` (or override via env vars in the workflow):

- `LLM_MODEL` — default is `gemma3:4b` (a good quality/speed balance on a 2-core CI runner). Drop to `gemma3:1b` for faster runs or bump to `gemma3:12b` for richer summaries — pull time and per-summary latency will go up accordingly.
- `LLM_API_URL` — defaults to `http://localhost:11434/api/chat` (Ollama). Point at any Ollama-compatible endpoint to use a different host.
- `LLM_TIMEOUT` — per-request timeout in seconds (default 180).
- `MAX_POST` — post length cap. Bluesky's actual limit is 300 graphemes; we keep some slack.

## Local testing

```bash
# 1. Install Ollama (https://ollama.com/) and pull the model
ollama pull gemma3:4b
# Make sure `ollama serve` is running (the desktop app starts it automatically;
# on Linux the install script enables a systemd service).

# 2. Dry-run a specific category
pip install -r requirements.txt
BOT_CATEGORY=transportation DRY_RUN=1 python scripts/post_to_bluesky.py
```

Dry run prints composed posts without hitting Bluesky. State still updates so you can iterate without re-summarizing.

If you don't have Ollama running locally, summaries fall back to the first clean sentence of the abstract (or are omitted) — the rest of the pipeline still works.

## Layout

```
.github/workflows/
  post.yml                                     # cron + pipeline (loops every category)
  weekly-digest.yml                            # Friday digest (loops every category)
scripts/
  post_to_bluesky.py                           # shared bot (parameterized by BOT_CATEGORY)
  weekly_digest.py                             # shared digest
  category.py                                  # config loader
categories/
  transportation/
    config.yml                                 # keywords, emojis, prompt focus
    bills_used.json                            # per-category dedup state (committed)
  immigration/
    config.yml
    bills_used.json
  taxation/
    config.yml
    bills_used.json
  ai_data_centers/
    config.yml
    bills_used.json
  housing/
    config.yml
    bills_used.json
  education/
    config.yml
    bills_used.json
  criminal_justice/
    config.yml
    bills_used.json
  lgbtq/
    config.yml
    bills_used.json
  healthcare/
    config.yml
    bills_used.json
  labor/
    config.yml
    bills_used.json
  reproductive_rights/
    config.yml
    bills_used.json
  elections_voting_rights/
    config.yml
    bills_used.json
```

## Notes & gotchas

- **First run is loud.** Without a populated `categories/<name>/bills_used.json`, every matching item is "new". Each category folder ships with an empty `{"posted": []}` file; the `POST_LIMIT` cap protects you, but consider seeding the file with current GUIDs (see below) before enabling the cron.
- **Idempotency** is via RSS `<guid>`. If govbot's RSS doesn't include guids, the bot falls back to the link, then to a `feed_name:title` synthetic id.
- **Permissions.** The workflow needs `contents: write` to commit state back. This is set in the workflow file already, but org-level settings can override it — check **Settings → Actions → General → Workflow permissions** if commits aren't landing.

### Seeding state to skip the backlog

After running `govbot logs > bills.jsonl` once, seed a category's `bills_used.json` with the dedup keys for everything currently in the feed. The bot will then treat those as "already posted" and only flag genuinely new updates from then on:

```bash
BOT_CATEGORY=transportation python -c "
import os, json, sys
sys.path.insert(0, 'scripts')
from post_to_bluesky import CATEGORY, JSONL_PATH, load_bills, extract_fields
keys = []
for r in load_bills(JSONL_PATH):
    b = extract_fields(r)
    if b and CATEGORY.matches(b):
        keys.append(b['dedup_key'])
out = CATEGORY.state_file_path()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'posted': sorted(set(keys))}, indent=2))
print(f'Seeded {len(set(keys))} dedup keys into {out}.')
"
git add categories/transportation/bills_used.json && git commit -m "seed transportation backlog" && git push
```

Repeat with `BOT_CATEGORY=<name>` for each category before enabling its workflow.

## License

MIT — do whatever.
