# Social Media Posts Dashboard

A dark-mode dashboard that shows every bill update this bot has posted, across
all social platforms — how many posts, which topics, which states, and when.

**Live page:** once GitHub Pages is turned on (see below), it lives at
`https://<your-username>.github.io/<your-repo>/`

## Layout

The page reads top to bottom as: a summary row of numbers, then one feed per
platform:

1. **Summary numbers** — total posts, platforms, topics, states, unique bills.
2. **Bluesky feeds** (live).
3. **Meta Threads posts**, then **X posts**, then **Instagram posts** (from saved
   bill data).

## Filtering

The summary numbers and all the post sections update live as you filter by:

- **Topic** — pick one of the subject areas.
- **State** — narrow to a single legislature.
- **Date range** — set a start and/or end date.
- **⬇ CSV** — download the filtered posts as a spreadsheet.

Platforms are detected automatically from the repo, so any platform the bot
starts posting to shows up as its own section — no edits needed.

## Bluesky feeds (live, exactly as posted)

Pick an account on the left and its posts load **live from Bluesky's public API**
— the real headline, summary, action, the "Read the full bill" link, and an
"Open on Bluesky" link — exactly as they appear on Bluesky. There's a search box
to filter by state or bill code. This needs no repo data and always shows the
latest posts, so it only works on the published page (it makes a live internet
request).

## Meta Threads · X · Instagram posts

These platforms don't offer a public feed API, so each has its own section built
from the bill records the bot saves in that platform's `bills_raw/` folder (bill,
a plain-language summary, and the dated action), newest first and filtered by the
same controls. A post appears here once that detail has been committed to the repo.

## What's in here

| File | What it does |
|------|--------------|
| `index.html` | The dashboard itself. Dark mode, no external dependencies. |
| `data.json` | The numbers the dashboard shows. Rebuilt automatically. |
| `README.md` | This file. |

The data comes from `scripts/build_dashboard.py`, which reads the post history in
`account_state/` and `topics/*/…/bills_used.json` and rolls it up.

## Turning the page on (one-time, ~2 minutes)

You only do this once. No coding required.

1. Go to your repository on GitHub.
2. Click the **Settings** tab (top of the page).
3. In the left sidebar, click **Pages**.
4. Under **Build and deployment → Source**, choose **GitHub Actions**.
5. That's it. Open the **Actions** tab and wait for the **pages-dashboard** run to
   finish (green check). Your dashboard is now live at the link shown on the Pages
   settings screen.

## How it stays up to date

The `pages-dashboard` workflow rebuilds and republishes the dashboard:

- every time the bot posts new content to `main`,
- once a day as a safety net, and
- any time you click **Run workflow** on the Actions tab.

You never have to touch it after the one-time setup.

## Updating it by hand (optional)

If you ever want to refresh the numbers locally:

```bash
python scripts/build_dashboard.py   # rewrites docs/data.json
```

Then commit the change. On GitHub, the workflow does this for you.
