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

Each platform section has **its own** filter bar (so filtering one section leaves
the others alone):

- **Topic** — pick one of the subject areas (hidden when the section only has one).
- **State** — narrow to a single legislature.
- **Date** — set a start and/or end post day.
- **⬇ CSV** — download that section's filtered posts as a spreadsheet.
- **Reset** — clear that section's filters.

The Bluesky section has its own controls too: the account list on the left and a
state/bill-code search box. Platforms are detected automatically from the repo, so
any platform the bot starts posting to shows up as its own section — no edits needed.

## Bluesky feeds (live, exactly as posted)

Pick an account on the left and its posts load **live from Bluesky's public API**
— the real headline, summary, action, the "Read the full bill" link, and an
"Open on Bluesky" link — exactly as they appear on Bluesky. There's a search box
to filter by state or bill code. This needs no repo data and always shows the
latest posts, so it only works on the published page (it makes a live internet
request).

## Meta Threads · X · Instagram posts

These platforms don't offer a public feed API, so each has its own colour-accented,
scrollable section built from the bill records the bot saves in that platform's
`bills_raw/` folder. Posts are ordered by the **day they were actually posted**
(from each file's commit date), newest first, and filtered by the same controls.

The posting scripts now save each post **exactly as it was published** — the real
headline, the "Read the full bill" link, and a link to the live post — so posts
made from now on show the same rich card as Bluesky, including an **Open on
X / Threads / Instagram** link. Posts made *before* this change fall back to the
bill title + summary + action (no post link), since that exact text wasn't saved
at the time.

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
