# 🏛️ govbot-social

**A free, multi-topic, multi-platform bot network that posts new U.S. legislative activity — federal (U.S. Congress) and all 50 states — each bill summarized in plain English by a local AI model — to Bluesky, X/Twitter, Threads, and Instagram.**

### Follow the bots

<p align="left">
  <a href="https://bsky.app/profile/govbottaxation.bsky.social"><img alt="Bluesky" src="https://img.shields.io/badge/Bluesky-0285FF?style=for-the-badge&logo=bluesky&logoColor=white"></a>
  <a href="https://x.com/Govbot27"><img alt="X" src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white"></a>
  <a href="https://www.instagram.com/legislationtracker.govbot/?hl=en"><img alt="Instagram" src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"></a>
  <a href="https://www.threads.com/@legislationtracker.govbot?hl=en"><img alt="Threads" src="https://img.shields.io/badge/Threads-000000?style=for-the-badge&logo=threads&logoColor=white"></a>
</p>

Powered by [chihacknight/govbot](https://github.com/chihacknight/govbot) for the raw legislative data and [Ollama](https://ollama.com/) + [Gemma](https://ai.google.dev/gemma) for on-runner summarization. Everything runs on scheduled **GitHub Actions** — no servers, no paid LLM API, no hosting bill.

<p align="left">
  <img alt="Runs on GitHub Actions" src="https://img.shields.io/badge/runs%20on-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Summaries: local LLM" src="https://img.shields.io/badge/summaries-local%20Gemma%20via%20Ollama-000000">
  <img alt="No API key required" src="https://img.shields.io/badge/LLM%20API%20key-not%20required-success">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Table of contents

- [What it does](#what-it-does)
- [Why it's different](#why-its-different)
- [The topics](#the-topics)
- [Architecture](#architecture)
- [How a post is made](#how-a-post-is-made)
- [What the model actually reads](#what-the-model-actually-reads)
- [Quick start](#quick-start)
- [Adding a new topic](#adding-a-new-topic)
- [The `config.yml` reference](#the-configyml-reference)
- [Configuration knobs](#configuration-knobs)
- [Local development](#local-development)
- [Workflows](#workflows)
- [Repository layout](#repository-layout)
- [State, dedup & seeding the backlog](#state-dedup--seeding-the-backlog)
- [Troubleshooting & gotchas](#troubleshooting--gotchas)
- [Contributing](#contributing)
- [Credits & license](#credits--license)

## What it does

Every day, a GitHub Actions workflow:

1. **Fetches** fresh bill activity from the U.S. Congress and 50+ states and territories via `govbot`.
2. **Filters** it down per topic using a curated keyword model (with context and negative keywords to keep the feeds clean).
3. **Reads the actual bill** — it downloads each candidate bill's PDF and extracts the full statutory text so summaries are grounded in the real legislation, not just the title.
4. **Summarizes** each bill into one neutral, jargon-free sentence using a local Gemma model — *no third-party API, no key, no per-call cost*.
5. **Posts** it to the matching topic's Bluesky and/or X account with a rich link card pointing at the official legislature page.

A separate **weekly digest** workflow threads together the week's most significant actions per topic (bills signed into law, passed, vetoed, etc.).

Each **topic** is its own social account with its own keyword list, emoji map, summary focus, and independent dedup state — but every topic shares one workflow run, so adding a bot doesn't multiply your CI minutes.

## Why it's different

- 🆓 **Genuinely free to run.** No hosted server, no LLM API key. Summarization happens on the Actions runner with a local model. The whole thing fits in GitHub's free tier.
- 🧠 **Grounded summaries.** Most bill bots parrot the title. This one pulls the bill's PDF, extracts the full text with `pdftotext`, and asks the model to translate the *substance* into plain layman's terms — spelling out acronyms, swapping legalese ("appropriates" → "sets aside money for") for everyday words.
- 🧩 **Drop-in topics.** Adding a new subject area is three steps: create a folder, write a `config.yml`, add two secrets. The shared workflow auto-discovers it on the next run. No Python or YAML pipeline edits.
- 🐦 **Four platforms, one pipeline.** The same filtering/summarization engine drives Bluesky, X, Threads, and Instagram, with fully independent dedup state per platform.
- 🎯 **Quality filtering.** Title hits, multi-keyword body matches, context keywords, negative keywords, and per-bucket draws keep omnibus-budget noise and off-topic referenda out of the feeds.
- 📚 **Auditable.** Every posted bill's raw record and extracted full text is committed back to the repo, so there's a permanent trail of exactly what was posted and why.

## The topics

Thirteen topics ship out of the box, each its own account:

| Topic folder | Account focus |
| --- | --- |
| `ai_data_centers` | AI, Data Centers & Crypto |
| `criminal_justice` | Criminal Justice & Policing |
| `education` | Education |
| `elections_voting_rights` | Elections & Voting Rights |
| `environment_climate` | Environment & Climate |
| `healthcare` | Healthcare |
| `housing` | Housing |
| `immigration` | Immigration |
| `labor` | Labor & Workers' Rights |
| `lgbtq` | LGBTQ |
| `reproductive_rights` | Reproductive Rights |
| `taxation` | Taxation |
| `transportation` | Transportation |

Each folder is self-contained — its keywords, emoji rules, prompt focus, digest copy, and dedup state all live under `topics/<name>/`.

## Architecture

Every scheduled workflow runs as **two jobs**. The `govbot` clone across 50+ states
is the slow, disk-heavy half (~40–60 min) and is identical for every topic, so it
happens **once** in `prepare`, which ships a compact corpus as an artifact. The
`post` job downloads that — it never clones — so one runner never holds the giant
clone *and* the language model at the same time, and each half gets its own
120-minute budget instead of the two sharing one.

```mermaid
%%{init: {'theme':'base','themeVariables':{
  'background':'#0d1117','primaryColor':'#161b22','primaryTextColor':'#e6edf3',
  'primaryBorderColor':'#8b949e','lineColor':'#8b949e','secondaryColor':'#1f2937',
  'tertiaryColor':'#161b22','fontSize':'15px','clusterBkg':'#0d1117',
  'clusterBorder':'#30363d','edgeLabelBackground':'#0d1117','textColor':'#e6edf3'}}}%%
flowchart TD
    subgraph PREP["🧱 job 1 · prepare — 120 min cap, runs once"]
        A["govbot clones 50+ states"] --> B["dump bills.jsonl"]
        B --> C["stage full-text bundle<br/>(metadata + extracted text)"]
        C --> D[["upload artifact<br/>bills-corpus"]]
    end

    D ==> E

    subgraph POST["🚀 job 2 · post / digest — 120 min cap, 95 min script budget"]
        E["download corpus<br/>(no re-clone)"] --> F["install Ollama<br/>+ gemma3:4b (cached)"]
        F --> G["keyword filter<br/>+ freshness + dedup"]
        G --> H["clean bill text<br/>(strip mastheads, chrome, citations)"]
        H --> I["LLM relevance gate<br/>fails open"]
        I --> J["select the on-topic window<br/>≤12k chars, ≤17k total prompt"]
        J --> K["LLM headline + summary<br/>one JSON call, salvaged if malformed"]
        K --> L["compose + publish"]
        L --> M["commit dedup state<br/>+ raw artifacts"]
    end

    L --> N1["🦋 Bluesky<br/>per-topic accounts"]
    L --> N2["✖️ X"]
    L --> N3["🧵 Threads"]
    L --> N4["📸 Instagram<br/>rendered cards"]

    classDef prep fill:#132a3a,stroke:#3b82f6,stroke-width:1px,color:#e6edf3
    classDef post fill:#14281d,stroke:#3fb950,stroke-width:1px,color:#e6edf3
    classDef llm  fill:#2d2136,stroke:#a371f7,stroke-width:1px,color:#e6edf3
    classDef out  fill:#3a2a13,stroke:#d29922,stroke-width:1px,color:#e6edf3
    class A,B,C,D prep
    class E,F,G,H,L,M post
    class I,J,K llm
    class N1,N2,N3,N4 out
```

Three design choices make it cheap and scalable:

- **One fetch, many topics.** The clone and the Ollama install happen once per job,
  then every topic reuses the same `bills.jsonl`. Bluesky's daily poster and weekly
  digest fan out across **5 parallel shards** (`index % 5`) that auto-rebalance as
  topics are added.
- **Secrets by convention.** The workflow exposes `toJSON(secrets)` as a single
  `ALL_SECRETS` env var, and each script looks up `BLUESKY_HANDLE_<TOPIC>` at
  runtime. Adding a topic never requires touching the workflow file.
- **The run bounds itself.** Each script gets a wall-clock budget
  (`RUN_DEADLINE_MINUTES`, 95) under the job cap, and every model call is clamped
  to the time remaining. A slow run finishes early with fewer posts instead of
  being cancelled by Actions mid-step — a cancel skips the commit step, which is
  how live posts once lost their dedup record and got published twice.

## How a post is made

For a single topic, `scripts/post_to_bluesky.py` (and its sibling `post_to_x.py`):

1. **Loads the topic** from `topics/<name>/config.yml` via `scripts/topic.py` (selected by the `BOT_TOPIC` env var).
2. **Filters** `bills.jsonl`:
   - A keyword in the **title** is a strong signal — one hit matches.
   - The noisier **abstract/subjects** body needs **two distinct** keyword hits (so a mental-health budget's lone "transportation" line item won't pull the bill into the transportation feed).
   - **`context_keywords`** (e.g. "human trafficking") only count when a core keyword co-occurs.
   - **`negative_keywords`** disqualify a title outright.
3. **Applies a freshness gate** — bill *actions* older than `MAX_ACTION_AGE_DAYS` are dropped so the feed never posts year-old news as fresh.
4. **Dedupes** against `topics/<name>/bluesky/bills_used.json` (keyed by RSS `<guid>`, falling back to link, then a synthetic `feed_name:title` id).
5. **Draws** up to `POST_LIMIT` bills using a weighted random selection that spreads coverage across states (and across `keyword_groups` buckets where configured).
6. **Extracts the full bill text** from the bill's PDF (`scripts/bill_text.py` → `pdftotext`), degrading gracefully to abstract-only if the PDF or `pdftotext` is unavailable.
7. **Confirms topic relevance with the local model** (the LLM relevance gate) — keyword matching is a cheap net that can let an omnibus/budget bill through on a single incidental subject tag (e.g. a whole state budget matching the AI/crypto feed because it lists "CRYPTOCURRENCY & NFTS" among hundreds of subjects). The model reads the actual bill text and drops it if it isn't genuinely about the topic. Fails **open** (any LLM/parse error keeps the bill), bypassed for force-posts, and toggleable with `RELEVANCE_GATE=0`.
   Before that call the gate applies a free check the model cannot be trusted with: **if neither the bill's title nor its own text mentions anything from the feed's subject, it matched on subject tags alone and is dropped.** Those tags describe a bill's final form and a state omnibus carries dozens of unrelated ones — NC HB 377 ("2026 Court Changes.", 63 tags) reached the criminal-justice feed on `BAIL` + `PUBLIC DEFENDERS` + `INDIGENT DEFENSE`, but the document on disk is an estates bill end to end, so the post it produced was about electronic wills. Handing that to the model does not help: it sees the same title, and a criminal-justice feed does cover courts. The title guard matters — across the 654 archived bills with real full text, 27 have no keyword in the document but 19 of those have one in the title and are genuine, so only the 8 metadata-only matches are dropped.
8. **Summarizes** via the local model — one neutral, plain-English sentence under ~160 characters, plus a short noun-phrase headline — picks a topical emoji, and composes a post that fits Bluesky's 300-grapheme limit.
10. **Posts** with a rich external link card to the official legislature page (with a state-homepage fallback when no deep link is known).
11. **Commits** the updated dedup state and raw artifacts back to the repo — written **after every post**, not once at the end, so a run that dies partway can never republish a bill it already posted.

The **weekly digest** (`scripts/weekly_digest_bluesky.py`) instead scores the week's actions by significance (signed → passed → vetoed → …), caps to `DIGEST_PER_STATE_CAP` bills per state to keep coverage broad, and posts a root summary plus up to `DIGEST_MAX_HIGHLIGHTS` threaded replies.

## What the model actually reads

The single biggest lever on post quality is **how much fits in one prompt**, and it
is easy to get wrong because most of the prompt is not the bill.

| part of the prompt | size | notes |
| --- | --- | --- |
| System prompt | **~9,300 chars** (~2,300 tokens) | Built per topic by `Topic.post_copy_system_prompt()` in `scripts/topic.py`. |
| Per-bill notes | ~300–2,000 chars | Bill status, stated purpose, the provision that earned the topic match, home-state anchor. |
| Bill text | whatever is left | Capped by `POST_COPY_MAX_SOURCE_CHARS` (12,000) **and** by what remains of the total. |
| **Total ceiling** | **17,000 chars** | `POST_COPY_MAX_PROMPT_CHARS`. |

That ~9,300-character system prompt is more than half the budget before a single
word of legislation. It is long because nearly every rule in it is scar tissue
from a real bad post — don't-manufacture-stakes, stay-non-partisan,
describe-the-effect-not-the-edit, name-who-is-actually-affected each exist because
something specific went out wrong. Roughly, by share:

| section | share |
| --- | --- |
| Shared rules (jargon, acronyms, don't invent facts, formatting) | 22% |
| Stay strictly non-partisan | 14% |
| `headline` spec | 13% |
| Never manufacture stakes | 12% |
| `summary` spec | 10% |
| Everything else (role, voice, status-quo, effect-not-edit, who's-affected) | 29% |

**Why the total is what matters, not the bill length.** `gemma3:4b` does not error
when a prompt is too big — it returns empty or unparseable JSON, and the post
silently falls back to deterministic copy (a raw title, a definitions paragraph, a
string of statute citations). Measured across one digest run:

| total prompt | outcome |
| --- | --- |
| 14,266 – 17,036 chars | model wrote the headline and summary ✅ |
| 21,265 – 21,772 chars | nothing usable → fallback copy ❌ |

A 17,834-character bill worked; a bill capped to the same 12,000-character window
failed. The difference was the **total**. So the bill text is budgeted as
*total − system − notes*, floored at `MIN_BILL_TEXT_CHARS` (1,500) so the model is
never handed instructions with no evidence.

**For bills longer than the budget**, the window is *selected*, not truncated:
`_prepare_full_text_for_llm` lifts the bill's own plain-language explainer to the
front when it has one, and `_relevant_window` then spends the remaining budget on
the passages that actually mention the feed's subject — scored with the topic's own
keyword regex, kept in document order, joined with an ellipsis. On a 107,739-char
Michigan bill this raised on-topic mentions inside the window from 25 to 62 and cut
its glossary lines from 18 to 11.

> **Raising these.** `POST_COPY_MAX_PROMPT_CHARS` is tuned to `gemma3:4b`. A larger
> model is what makes a larger number readable — raising the ceiling alone
> reproduces the fallback posts. `gemma3:12b` was tried and reverted: at 8.1 GB on a
> 16 GB runner it thrashed swap and could not finish a single summary in 900 s.

## Quick start

### 1. Use this repo as a template

Click **Use this template** on GitHub (or fork), then clone locally.

### 2. Generate a `govbot.yml`

Run `govbot` locally once with no config — it launches a wizard that writes `govbot.yml` (pick states and tags). Commit the result. To skip the wizard, see the [govbot docs](https://chihacknight.github.io/govbot/).

### 3. Create your social accounts and add secrets

In **Settings → Secrets and variables → Actions**, add credentials for each topic/platform you want live.

**Bluesky** — two secrets per topic:

| Secret | Value |
| --- | --- |
| `BLUESKY_HANDLE_<NAME>` | The topic's handle, e.g. `chn-transportation.bsky.social` |
| `BLUESKY_APP_PASSWORD_<NAME>` | An **app password** from Bluesky → *Settings → App Passwords* (never your main password) |

`<NAME>` is the **upper-case topic folder name**. So `topics/transportation/` → `BLUESKY_HANDLE_TRANSPORTATION` + `BLUESKY_APP_PASSWORD_TRANSPORTATION`; `topics/ai_data_centers/` → `BLUESKY_HANDLE_AI_DATA_CENTERS` + `BLUESKY_APP_PASSWORD_AI_DATA_CENTERS`.

**X/Twitter** — four developer-app secrets (shared by the X workflow):

| Secret | Value |
| --- | --- |
| `X_API_KEY` / `X_API_SECRET` | Your X app's consumer key & secret |
| `X_ACCESS_TOKEN` / `X_ACCESS_TOKEN_SECRET` | The posting account's access token & secret |

**Threads (Meta)** — a single dedicated account (e.g. `chn.govbot`), two secrets:

| Secret | Value |
| --- | --- |
| `THREADS_ACCESS_TOKEN` | A **long-lived** Threads access token (see below) |
| `THREADS_USER_ID` | The account's numeric Threads user id |
| `THREADS_REFRESH_PAT` | *(optional)* A PAT with `secrets: write` so the weekly refresh workflow can persist the rolled-forward token |

> **Getting the Threads token:** create a Meta app with the *"Access the Threads API"* use case, add the `threads_basic` + `threads_content_publish` permissions, add your Threads account under **App roles → Roles → Threads Testers** (and accept the invite inside Threads → *Settings → Website permissions*). Then generate a short-lived token in the **Graph API Explorer** and exchange it for a 60-day long-lived token via `GET https://graph.threads.net/access_token?grant_type=th_exchange_token&client_secret=…&access_token=…`. The `meta-threads-refresh-token` workflow keeps it from lapsing — see [Threads token refresh](#threads-token-refresh).

**Instagram (Meta)** — a single dedicated Instagram **Business/Creator** account, two secrets:

| Secret | Value |
| --- | --- |
| `INSTAGRAM_ACCESS_TOKEN` | A **long-lived** Instagram access token (see below) |
| `INSTAGRAM_USER_ID` | The Instagram **Business account** id |
| `INSTAGRAM_REFRESH_PAT` | *(optional)* A PAT with `secrets: write` so the weekly refresh workflow can persist the rolled-forward token |

> **Instagram is image-first.** Unlike the text platforms, Instagram's Graph API has no text-only post type and fetches the post image from a *public URL*. So the Instagram poster renders each bill into a 1080×1350 card (`scripts/render_bill_card.py`, dark mode, per-topic `card_accent`), commits + pushes it, waits for it to go live on `raw.githubusercontent.com`, then publishes via the two-step container→publish Graph call. The bill link can't be clickable in an Instagram caption, so it ships as plain text and the card footer reads *"Link to the bill in the description."* **This requires the repository to be public** so Instagram's servers can fetch the card.

> **Getting the Instagram token:** use the *Instagram API with Instagram Login* (`graph.instagram.com`). Create a Meta app, add the `instagram_business_basic` + `instagram_business_content_publish` permissions, connect your Instagram Business/Creator account, then exchange for a 60-day long-lived token. The `instagram-refresh-token` workflow keeps it from lapsing (same 60-day roll-forward model as Threads).

> Summarization runs entirely on the runner via a local Gemma model — **no OpenAI/Anthropic/other LLM API key is ever needed.**

### 4. Enable Actions

On the **Actions** tab, enable workflows. Trigger the first run manually via **Run workflow** on *govbot-bluesky-post*. (Read [Seeding the backlog](#state-dedup--seeding-the-backlog) first — the very first run treats *everything* as new.)

## Adding a new topic

The `topics/` layout exists so that adding a bot is a drop-in. The shared workflows already loop every folder under `topics/`, so once these steps are done the new bot goes live on the next run — **no Python or workflow edits required.**

1. **Create the folder** `topics/<name>/` and add a `config.yml` (copy `topics/transportation/config.yml` as a starting point). Fill in `keywords`, `emojis`, `prompt_topic`, and the `digest` copy.
2. **Add the secrets** in repo settings: `BLUESKY_HANDLE_<NAME>` / `BLUESKY_APP_PASSWORD_<NAME>` (and the X secrets if you run the X bot for it).
3. **Sync the dropdowns** so the manual workflows list your new topic:
   ```bash
   python scripts/sync_topic_choices.py
   ```
   This rewrites the managed `choice` option blocks in the `workflow_dispatch` YAMLs (they can't be populated dynamically at runtime).
4. **Dry-run** to preview before committing:
   ```bash
   BOT_TOPIC=<name> DRY_RUN=1 python scripts/post_to_bluesky.py
   ```
5. **Commit** the new folder. The next scheduled run picks it up.

## The `config.yml` reference

```yaml
name: transportation                # MUST match the folder name
display_name: "Transportation"      # Human label used in digest titles
prompt_topic: "transportation"      # Steers the LLM's summary focus
default_emoji: "🚗"                  # Fallback when no emoji rule matches

keywords:                           # Core match terms (word-boundary, case-insensitive)
  - transit
  - light rail
  - bike lane
  # ...

context_keywords:                   # (optional) only count when a core keyword co-occurs
  - safety

negative_keywords:                  # (optional) disqualify a title outright
  - referendum on liquor

emojis:                             # First matching rule wins; else default_emoji
  - emoji: "🚆"
    match: ["rail", "amtrak", "metra"]
  - emoji: "✈️"
    match: ["airport", "aviation"]

keyword_groups:                     # (optional) named buckets to balance the draw
  ai_data_centers: ["artificial intelligence", "data center"]
  crypto: ["cryptocurrency", "digital asset"]

x_subdir: x                         # (optional) X state subfolder; default "x"

digest:
  thread_title: "🗳️ Transportation Bills Weekly Digest"
  topic_phrase: "transportation"
```

| Field | Required | Purpose |
| --- | --- | --- |
| `name` | ✅ | Must equal the folder name (validated on load). |
| `keywords` | ✅ | Core match terms. |
| `display_name` / `prompt_topic` / `default_emoji` | — | Default to sensible values derived from `name`. |
| `context_keywords` | — | Broad terms that only match alongside a core keyword. |
| `negative_keywords` | — | Veto terms — a title hit drops the bill. |
| `emojis` | — | Ordered emoji rules (substring match over title+abstract+subjects). |
| `keyword_groups` | — | Named buckets so a multi-theme account (e.g. AI **and** crypto) posts at least one of each per run. First matching bucket wins. |
| `x_subdir` | — | Where X state lives under the topic (default `x`); override to rebrand a feed. |
| `digest` | — | `thread_title` and `topic_phrase` for the weekly thread. |

## Configuration knobs

All knobs are env vars (set defaults in the workflow `env:` blocks or override per-run):

| Variable | Default | What it controls |
| --- | --- | --- |
| `BOT_TOPIC` | — (required) | Which `topics/<name>/` folder this run posts for. |
| `POST_LIMIT` | `4` (X workflow sets `4`; Bluesky sets `2`) | Max posts per run **per topic** — flood protection. |
| `MAX_ACTION_AGE_DAYS` | `32` | Drop bill actions older than this so old news never posts as fresh. |
| `DRY_RUN` | `0` | `1` composes posts and prints them without publishing. State still updates so you can iterate. |
| `LLM_MODEL` | `gemma3:4b` | Ollama model. **`gemma3:12b` does not work on a free runner** — 8.1 GB on a 16 GB box, it thrashes swap and cannot finish a summary in 900 s. A ~4.7 GB middle model (`qwen2.5:7b`, `llama3.1:8b`) is the realistic upgrade; trial it on a manual dispatch first. |
| `LLM_API_URL` | `http://localhost:11434/api/chat` | Ollama endpoint — point at any Ollama-compatible host. |
| `LLM_TIMEOUT` | `1620` (27 min) | Per-call ceiling for headline+summary. Generous on purpose so a merely *slow* call finishes instead of dropping to fallback copy; `RUN_DEADLINE_MINUTES` is what keeps it safe. |
| `LLM_RETRY_TIMEOUT` | `1620` | Ceiling for the two retries. |
| `RELEVANCE_GATE_TIMEOUT` | `1620` | Ceiling for the on-topic check. |
| `RELEVANCE_GATE` | `1` | `0` disables the LLM on-topic check (keyword matching only). |
| `RUN_DEADLINE_MINUTES` | `0` (off); workflows set `95` | Wall-clock budget for the whole script. Every model call is clamped to the time left, so the run finishes early with fewer posts rather than being **cancelled** by Actions — a cancel skips the commit step and loses the run's state. |
| `POST_COPY_MAX_PROMPT_CHARS` | `17000` | Ceiling on the **entire** prompt (system + notes + bill text). The real quality lever — see *What the model actually reads*. |
| `POST_COPY_MAX_SOURCE_CHARS` | `12000` | Upper bound on the bill-text portion alone. |
| `GOVBOT_TZ` | `America/Chicago` | Timezone the digests reckon their date range in. Actions runs on UTC, so without this an evening run labels itself with tomorrow's date. |
| `POST_DEADLINE_MINUTES` | `40`; X workflow sets `90` | X poster's own run budget (predates `RUN_DEADLINE_MINUTES`; both clamp with `min()`). |
| `RESERVE_LIMIT` | `8` | How many extra candidates the X poster may fall through to when the relevance gate skips a pick. |
| `MAX_TWEET_FAILURES` / `TWEET_ATTEMPTS` | `3` / `3` | Give up after N failed posts; retry transport blips N times. |
| `FETCH_OG_IMAGE` | `0` | `1` re-enables scraping `og:image` thumbnails for link cards. |
| `SAVE_STATE` / `SAVE_RAW` | `1` / `1` | Toggle writing dedup state / raw artifacts (independent of `DRY_RUN`). |
| `FORCE_STATE` / `FORCE_BILL_ID` / `FORCE_REPOST` | — | Force-post one specific bill, bypassing the keyword/freshness/dedup gates (used by the *specific-bill* workflows). |
| `DIGEST_LOOKBACK_DAYS` | `7` | Weekly digest window. |
| `DIGEST_MAX_HIGHLIGHTS` | `6` | Max reply posts in a digest thread. |
| `DIGEST_PER_STATE_CAP` | `2` | Cap bills per state in a digest to keep it broad. |
| `EXTRAVAGANZA_MODE` | `state` | *(Extravaganza)* `state` = STATE-first framing (`🏛️ State Extravaganza!!`, header leads with the state scope); `topic` = TOPIC-first framing (`🎯 Topic Extravaganza!!`, header leads with the topic, nationwide by default). Same selection pipeline and knobs either way — only the header/cover copy changes. |
| `PLATFORM` | `bluesky` | *(Extravaganza)* Which feed to post to: `bluesky`, `x`, `threads`, or `instagram` (a carousel post). |
| `EXTRAVAGANZA_STATES` | — | *(Extravaganza)* Space/comma-separated 2-letter state codes to pull bills from; blank = all states. |
| `NUM_POSTS` | `6` | *(Extravaganza)* Number of bill posts in the thread. |
| `EXTRAVAGANZA_LOOKBACK_DAYS` | `62` | *(Extravaganza)* Recency window in days, **hard-capped at 62**. |
| `EXTRAVAGANZA_PER_STATE_CAP` | `NUM_POSTS` | *(Extravaganza)* Max bills per state; defaults to no real cap so a single-state run can fill the thread. |

The cron schedules live in the `on.schedule` block at the top of each workflow file (the daily posters default to early-morning UTC; the digests run weekly). Adjust them there.

## Local development

```bash
# 1. Install Ollama (https://ollama.com/) and pull the model
ollama pull gemma3:4b
#    Make sure `ollama serve` is running — the desktop app starts it
#    automatically; the Linux install script enables a systemd service.

# 2. (Optional) install poppler so full-text PDF extraction works
#    macOS:  brew install poppler      Debian/Ubuntu: apt-get install poppler-utils

# 3. Install Python deps and dry-run a topic
pip install -r requirements.txt
BOT_TOPIC=transportation DRY_RUN=1 python scripts/post_to_bluesky.py
```

A dry run prints the composed posts without hitting Bluesky/X. If Ollama isn't running, summaries fall back to the first clean sentence of the abstract (or are omitted) — the rest of the pipeline still works. If `pdftotext` is missing, full-text extraction is skipped and the model summarizes the abstract.

## Workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `post_to_bluesky.yml` | Mon/Wed + manual | Fetch → filter → summarize → post **all topics** to Bluesky. `prepare` + `post` **×5 shards**. |
| `weekly-digest-bluesky.yml` | **Fri** + manual | Threaded weekly digest per topic on Bluesky. `prepare` + `digest` **×5 shards**. |
| `post_to_x.yml` | Tue/Thu/Sun + manual | Same pipeline, posting to an X account. `prepare` + `post`. |
| `weekly-digest-x.yml` | **Fri** + manual | Weekly digest thread on X. `prepare` + `digest`. |
| `post_to_meta_threads.yml` | Tue/Thu/Sun + manual | Same pipeline, posting to a Meta Threads account (dedicated to the `lgbtq` topic; 3 posts/run). |
| `weekly-digest-meta-threads.yml` | **Sat** + manual | Weekly digest thread on Threads (root + a self-contained reply per highlight). |
| `meta-threads-refresh-token.yml` | Weekly + manual | Rolls the 60-day Threads token forward so it never lapses. |
| `post_to_instagram.yml` | Mon/Wed + manual | Renders each bill to a card image, pushes it, then posts to a Meta Instagram Business account (dedicated to the `lgbtq` topic; 2 posts/run). |
| `instagram-refresh-token.yml` | Weekly + manual | Rolls the 60-day Instagram token forward so it never lapses. |
| `post_bluesky_specific_bill.yml` | Manual | Force-post one specific `state` + `bill_id` to a chosen topic's Bluesky account (with dry-run / repost toggles). |
| `post_x_specific_bill.yml` | Manual | Same one-off force-post, for X. |
| `collect-samples.yml` | Manual | Save a batch of full bill records into `samples/` (optionally compose/post them too). Useful for prompt-tuning and tests. |
| `posts-extravaganza.yml` | Manual | On-demand **Extravaganza** thread with a `mode` knob: `state` posts a `🏛️ State Extravaganza!! 🧵` thread whose header leads with the picked state(s); `topic` posts a `🎯 Topic Extravaganza!! 🧵` thread whose header leads with the topic (nationwide by default). Both share the same knobs — pick the platform, the state(s), the topic(s) (space/comma list or `all`; ignored for X), the number of posts, and a lookback window (capped at 62 days) — and post **one thread per selected topic** (on Bluesky each goes to that topic's own account; on Threads/Instagram several threads to the one shared account), one threaded reply per bill like the weekly digest. On Instagram (no text threads) each is a **carousel**: a cover slide + one rendered bill card per highlight (max 10 slides). |

**When things run** (UTC cron; the digests reckon their date range in
`GOVBOT_TZ`, default `America/Chicago`, so an evening run is not labelled as
tomorrow):

| day | what posts |
| --- | --- |
| Mon / Wed | Bluesky + Instagram daily |
| Tue / Thu / **Sun** | X + Threads daily |
| **Fri** | X digest, then Bluesky digest — text/news feeds while the week is live |
| **Sat** | Threads digest, then Instagram digest — weekend-morning visual feeds |

Note the lead time: cron starts the *workflow*, not the post. `prepare` clones
first, so a post lands roughly **45–70 minutes after** the cron fires. Adjust a
schedule by that lead, not by the time you want to see the post.

Every job is capped at **120 minutes**, with the script holding a smaller
wall-clock budget under it. All scheduled workflows start with a **free-disk-space** step — `govbot` cloning 50+ states plus the Ollama model would otherwise overflow the runner's ~14 GB and crash with *"No space left on device."* Ollama's binary and model are cached between runs to skip the ~600 MB + ~3.3 GB downloads.

## Repository layout

```
.github/workflows/
  post_to_bluesky.yml          # daily Bluesky pipeline (all topics, sharded)
  post_to_x.yml                # daily X pipeline
  weekly-digest-bluesky.yml    # Friday Bluesky digest threads
  weekly-digest-x.yml          # X digest threads
  post_bluesky_specific_bill.yml  # manual one-off force-post (Bluesky)
  post_x_specific_bill.yml        # manual one-off force-post (X)
  collect-samples.yml          # save sample bill records to samples/
scripts/
  topic.py                     # Topic config loader + matching/emoji logic
  post_to_bluesky.py           # shared Bluesky bot (parameterized by BOT_TOPIC)
  post_to_x.py                 # shared X bot (reuses the Bluesky engine)
  post_to_meta_threads.py      # shared Threads bot (reuses the Bluesky engine)
  post_to_instagram.py         # shared Instagram bot (renders + posts card images)
  render_bill_card.py          # Pillow renderer for the Instagram bill cards
  weekly_digest_bluesky.py     # Bluesky weekly digest builder
  weekly_digest_x.py           # X weekly digest builder
  weekly_digest_meta_threads.py # Threads weekly digest builder
  bill_text.py                 # full bill-text extraction from PDFs (pdftotext)
  refresh_meta_threads_token.py # roll the Threads long-lived token forward
  refresh_instagram_token.py   # roll the Instagram long-lived token forward
  sync_topic_choices.py        # keep workflow choice dropdowns in sync with topics/
samples/                       # saved bill records for prompt-tuning / tests
topics/
  <name>/
    config.yml                 # keywords, emojis, prompt focus, digest copy
    bluesky/ (or bluesky_subdir)      # Bluesky account state:
      bills_used.json          #   per-topic Bluesky dedup state (committed)
      bills_raw/               #   raw JSON of each posted bill (audit trail)
      bills_full_text/         #   extracted full text of each posted bill
      weekly_digest/           #   digest highlight artifacts
    x/  (or x_subdir)          # mirror of the above for the X account
    meta-threads/ (or threads_subdir) # mirror of the above for the Threads account
    instagram/ (or instagram_subdir)  # mirror for Instagram, plus cards/ (rendered PNGs)
requirements.txt               # requests, Pillow, PyYAML, tweepy
```

## State, dedup & seeding the backlog

- **Idempotency** is per-platform and per-topic. Bluesky dedup lives in `topics/<name>/<bluesky_subdir>/bills_used.json` (default `bluesky/`); X dedup under `topics/<name>/<x_subdir>/bills_used.json`; Threads dedup under `topics/<name>/<threads_subdir>/bills_used.json`; Instagram dedup under `topics/<name>/<instagram_subdir>/bills_used.json`. Keys are the RSS `<guid>` (falling back to link, then `feed_name:title`).
- **First run is loud.** With an empty state file, *every* matching bill is "new." Each topic ships with `{"posted": []}`, and `POST_LIMIT` caps the blast radius — but you'll likely want to seed the backlog first.
- **Permissions.** Posting workflows need `contents: write` to commit state back. This is set in the workflows, but org-level settings can override it — check **Settings → Actions → General → Workflow permissions** if commits aren't landing.

### Seed a topic to skip the backlog

After running `govbot logs > bills.jsonl` once, mark everything currently in the feed as "already posted" so the bot only flags genuinely new activity from then on:

```bash
BOT_TOPIC=transportation python -c "
import json, sys
sys.path.insert(0, 'scripts')
from post_to_bluesky import TOPIC, JSONL_PATH, load_bills, extract_fields
keys = []
for r in load_bills(JSONL_PATH):
    b = extract_fields(r)
    if b and TOPIC.matches(b):
        keys.append(b['dedup_key'])
out = TOPIC.state_file_path()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'posted': sorted(set(keys))}, indent=2))
print(f'Seeded {len(set(keys))} dedup keys into {out}.')
"
git add topics/transportation/bluesky/bills_used.json
git commit -m "seed transportation backlog" && git push
```

Repeat with `BOT_TOPIC=<name>` for each topic before enabling its workflow.

### Threads token refresh

Threads access tokens differ from Bluesky app passwords: a long-lived token is valid for **60 days**, but can be *refreshed* (which rolls the 60-day window forward) any time after it's 24 hours old. The `meta-threads-refresh-token.yml` workflow does this on a weekly cron via `scripts/refresh_meta_threads_token.py`, so the token never lapses as long as the bot keeps running.

Persisting a refreshed token means updating the `THREADS_ACCESS_TOKEN` repo secret, which the default `GITHUB_TOKEN` can't do. To enable automatic write-back, add a **`THREADS_REFRESH_PAT`** secret — a Personal Access Token with `secrets: write` on this repo. The refresh script encrypts the new token (via PyNaCl) and writes it back through the GitHub API. Without the PAT, the workflow still refreshes and reports the new expiry but won't persist the token (and never prints it to the logs).

## Troubleshooting & gotchas

- **A state was skipped in the logs.** `govbot` panics on states it doesn't support; the workflow wraps each clone in `|| echo skipped` and prints a *supported vs skipped* summary at the end. That's expected, not an error.
- **Summaries look thin / generic.** Make sure `poppler-utils` is installed so full-text extraction runs — without it the model only sees the abstract. Bumping `LLM_MODEL` to `gemma3:12b` also helps (at a latency cost).
- **Nothing posted but no error.** Check `POST_LIMIT`, the freshness window (`MAX_ACTION_AGE_DAYS`), and whether the bills were already in `bills_used.json`.
- **New topic missing from the manual workflow dropdown.** Run `python scripts/sync_topic_choices.py` and commit the updated YAMLs.
- **Runner crashed with "No space left on device."** The free-disk-space step must run before the govbot clone; don't remove it.
- **Threads: `"requires the threads_basic permission … or your user must be in the list of Threads testers."`** The account isn't enrolled as a tester. Add it under **App roles → Roles → Threads Testers**, accept the invite in Threads (*Settings → Website permissions*), then regenerate the token.
- **A post is a raw legal title, a definitions paragraph, or a string of statute citations.** That is the deterministic fallback, which means the model returned nothing usable. Almost always the prompt was too big — check the bill's size and see *What the model actually reads*. `gemma3:4b` fails silently here; it does not error.
- **A fix landed but posts didn't change.** `.bill_text_cache/` stores *already-cleaned* text keyed by document URL, and CI restores it from the newest prior run — so before the cache was keyed on the cleaning code's own hash, an entry written pre-fix served its stale text forever. That key now lives in the cache path; changing `scripts/bill_text.py` retires the entries it would have changed.
- **A job was cancelled and posts got published twice.** A cancel skips the commit step, so the dedup record for posts that already went out is lost. Both guards are in place now — state is written after *every* post, and the commit step runs on `always()` — but if you add a workflow, keep both.
- **The digest's date range is a day ahead.** `GOVBOT_TZ` reckons the digest day; UTC would call a 7 pm Chicago run "tomorrow".
- **A weekly digest was cancelled at the job cap.** Check `RUN_DEADLINE_MINUTES` is set below `timeout-minutes` for that job. Without it a single stuck bill can want 108+ minutes against a 120-minute cap.
- **Threads posts stopped after ~2 months.** The long-lived token expired. Make sure `meta-threads-refresh-token.yml` is enabled (and ideally set `THREADS_REFRESH_PAT`), or re-run the manual token exchange.

## Contributing

Issues and PRs welcome — new topics, better keyword models, additional state deep-link builders, and prompt improvements are all great contributions. To propose a new topic, open a PR adding `topics/<name>/config.yml` and run `sync_topic_choices.py`. Use the `samples/` records and `DRY_RUN=1` to validate filtering and summaries before publishing.

## Credits & license

- Legislative data: [chihacknight/govbot](https://github.com/chihacknight/govbot) (Chi Hack Night).
- Summarization: [Gemma](https://ai.google.dev/gemma) served locally by [Ollama](https://ollama.com/).
- Full-text extraction inspired by upstream [govbot#31](https://github.com/chihacknight/govbot/issues/31).

Released under the **MIT License** — do whatever, just keep people informed.
