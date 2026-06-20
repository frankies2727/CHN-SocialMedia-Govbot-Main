#!/usr/bin/env python3
"""
Topic configuration loader.

Each Bluesky bot is one topic (transportation, immigration, taxation, …).
A topic is described by a YAML file at topics/<name>/config.yml; all
of its per-bot state lives in the same folder. The loader exposes a single
Topic object that the post + digest scripts use to filter bills, pick
emojis, and look up Bluesky credentials.

Adding a new topic is a drop-in operation: create the folder, add the
config.yml, add BLUESKY_HANDLE_<NAME> + BLUESKY_APP_PASSWORD_<NAME> repo
secrets. The shared workflow loops over topics/ and picks it up on
the next cron tick — no Python or workflow edits needed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TOPICS_DIR = ROOT / "topics"


# ---------------------------------------------------------------------------
# Proper-noun collision guard
#
# Single common-word keywords (e.g. "gay", "green", "justice") routinely double
# as people's surnames. Honorary resolutions ("Grayson Gay, commended") would
# then match a topic on the surname alone — and because the summarizer is
# steered by the topic, it can invent a policy angle the bill never had. These
# helpers detect the case where a title's only keyword signal is a surname so
# the matcher can ignore it. They are deliberately conservative: anything
# ambiguous (Title-Cased titles, lower-case occurrences, multi-word keywords)
# is treated as a genuine match rather than risk dropping a real bill.
# ---------------------------------------------------------------------------

# Capitalized words that lead a phrase without signaling a personal name, so a
# keyword right after one of these isn't a surname (e.g. "The Gay community…").
_NAME_LEAD_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "or", "no",
}


def _is_title_cased(title: str) -> bool:
    """Heuristic: is the title written in Title Case (every content word
    capitalized)? When it is, capitalization carries no proper-noun signal, so
    a capitalized keyword can't be told apart from a surname — callers then skip
    the name guard and treat the keyword as a genuine match."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", title)
    content = [w for w in words if len(w) >= 4]
    if len(content) < 2:
        return False
    caps = sum(1 for w in content if w[0].isupper())
    return caps / len(content) >= 0.8


def _occurrence_is_personal_name(title: str, m: "re.Match") -> bool:
    """True when this keyword occurrence reads as part of a personal name —
    a Capitalized word (e.g. "Gay") immediately preceded by another capitalized
    word that looks like a given name or honorific (e.g. "Grayson", "Senator")
    rather than a sentence-leading function word."""
    word = m.group(0)
    if not word[:1].isupper():
        return False  # lower-case "gay" is the common word, not a surname
    prefix = title[: m.start()]
    prev = re.search(r"([A-Za-z][A-Za-z.'\-]*)\W*$", prefix)
    if not prev:
        return False  # nothing before it (title start) — just normal capitalization
    token = prev.group(1)
    if not token[:1].isupper():
        return False
    if token.lower().strip(".") in _NAME_LEAD_STOPWORDS:
        return False
    return True


def _is_proper_name_only_match(title_raw: str, title_hits: set[str]) -> bool:
    """True when the title's keyword hits are ALL single common words appearing
    solely as part of a personal name, so the title carries no real topical
    signal. Multi-word keyword hits, Title-Cased titles, or any occurrence that
    isn't a surname make this return False (treat as a genuine match)."""
    if not title_hits:
        return False
    if any(" " in k or "-" in k for k in title_hits):
        return False
    if _is_title_cased(title_raw):
        return False
    for kw in title_hits:
        occurrences = list(re.finditer(r"\b" + re.escape(kw) + r"\b", title_raw, re.IGNORECASE))
        if not occurrences:
            return False
        if not all(_occurrence_is_personal_name(title_raw, m) for m in occurrences):
            return False
    return True


@dataclass
class Topic:
    name: str
    display_name: str
    prompt_topic: str
    default_emoji: str
    keywords: list[str]
    emojis: list[dict]
    thread_title: str
    topic_phrase: str
    _keyword_re: re.Pattern = field(repr=False)
    context_keywords: list[str] = field(default_factory=list)
    _context_re: re.Pattern | None = field(repr=False, default=None)
    negative_keywords: list[str] = field(default_factory=list)
    _negative_re: re.Pattern | None = field(repr=False, default=None)
    # X/Twitter state subfolder. Defaults to "x" so every existing topic keeps
    # its current path; override in config.yml when a topic's X feed has been
    # rebranded (e.g. ai_data_centers uses "x-including-Crypto" to flag that
    # the X bot also tweets crypto-related bills).
    x_subdir: str = "x"
    # Threads/Meta state subfolder. Defaults to "meta-threads" so the platform
    # is unambiguous when browsing the repo; override in config.yml only if a
    # topic's Threads feed is ever rebranded (mirrors x_subdir).
    threads_subdir: str = "meta-threads"
    # Optional named keyword buckets used by the X poster to balance the daily
    # draw across sub-topics (e.g. ai_data_centers splits its keywords into
    # an "ai_data_centers" bucket and a "crypto" bucket so each X run posts at
    # least one of each when POST_LIMIT permits). Order matters: a bill that
    # matches multiple buckets is classified into the first matching bucket,
    # so put the bucket you want to protect from starvation first.
    keyword_groups: dict[str, list[str]] = field(default_factory=dict)
    _keyword_group_res: dict[str, re.Pattern] = field(repr=False, default_factory=dict)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, name: str) -> "Topic":
        path = TOPICS_DIR / name / "config.yml"
        if not path.exists():
            raise FileNotFoundError(
                f"Topic config not found: {path}. "
                f"Expected a folder at topics/{name}/ with config.yml."
            )
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg_name = data.get("name") or name
        if cfg_name != name:
            raise ValueError(
                f"Topic folder name ({name!r}) does not match "
                f"config.yml name ({cfg_name!r})."
            )

        keywords = list(data.get("keywords") or [])
        if not keywords:
            raise ValueError(f"Topic {name!r}: keywords list is empty.")

        display_name = data.get("display_name") or name.replace("_", " ").title()
        prompt_topic = data.get("prompt_topic") or display_name.lower()
        default_emoji = data.get("default_emoji") or "📜"
        emojis = list(data.get("emojis") or [])

        digest = data.get("digest") or {}
        thread_title = digest.get("thread_title") or f"🗳️ {display_name} Bills Weekly Digest"
        topic_phrase = digest.get("topic_phrase") or prompt_topic

        keyword_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
            re.IGNORECASE,
        )

        context_keywords = list(data.get("context_keywords") or [])
        context_re = None
        if context_keywords:
            context_re = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in context_keywords) + r")\b",
                re.IGNORECASE,
            )

        negative_keywords = list(data.get("negative_keywords") or [])
        negative_re = None
        if negative_keywords:
            negative_re = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in negative_keywords) + r")\b",
                re.IGNORECASE,
            )

        x_subdir = (data.get("x_subdir") or "x").strip() or "x"
        threads_subdir = (data.get("threads_subdir") or "meta-threads").strip() or "meta-threads"

        raw_groups = data.get("keyword_groups") or {}
        keyword_groups: dict[str, list[str]] = {}
        keyword_group_res: dict[str, re.Pattern] = {}
        for group_name, group_kws in raw_groups.items():
            kws = [k for k in (group_kws or []) if k]
            if not kws:
                continue
            keyword_groups[str(group_name)] = list(kws)
            keyword_group_res[str(group_name)] = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b",
                re.IGNORECASE,
            )

        return cls(
            name=name,
            display_name=display_name,
            prompt_topic=prompt_topic,
            default_emoji=default_emoji,
            keywords=keywords,
            emojis=emojis,
            thread_title=thread_title,
            topic_phrase=topic_phrase,
            _keyword_re=keyword_re,
            context_keywords=context_keywords,
            _context_re=context_re,
            negative_keywords=negative_keywords,
            _negative_re=negative_re,
            x_subdir=x_subdir,
            threads_subdir=threads_subdir,
            keyword_groups=keyword_groups,
            _keyword_group_res=keyword_group_res,
        )

    # ------------------------------------------------------------------
    # Bill matching / emoji selection
    # ------------------------------------------------------------------

    def matches(self, b: dict) -> bool:
        # Title is the strongest signal — a single keyword hit there is enough.
        # Abstract/subjects are noisier (omnibus appropriations bills name every
        # department and line item, so a lone "transportation" mention from a
        # mental-health budget's capital line item shouldn't pull the bill into
        # the transportation feed). Require at least two distinct keyword hits
        # there before counting a match without title support.
        title_raw = b.get("title") or ""
        title = title_raw.lower()
        # Negative keywords disqualify a title outright — used to filter out
        # local-advisory referenda on highways, SPLOST, alcohol, etc. from the
        # elections feed even when "referendum" matches a core keyword.
        if self._negative_re is not None and self._negative_re.search(title):
            return False
        title_hits = {m.group(1).lower() for m in self._keyword_re.finditer(title)}
        # A single common-word keyword (e.g. "gay") routinely collides with a
        # person's surname in honorary resolutions ("Grayson Gay, commended"),
        # which would wrongly pull the bill into a topic feed and — worse — let
        # the topic-steered summarizer invent a policy angle the bill never had.
        # When the title's only signal is one such keyword appearing solely as
        # part of a proper name, don't treat the title as a match; fall through
        # to the body/context checks, which need real corroboration.
        if title_hits and not _is_proper_name_only_match(title_raw, title_hits):
            return True
        body = " ".join([b.get("abstract", ""), b.get("subjects", "")]).lower()
        distinct = {m.group(1).lower() for m in self._keyword_re.finditer(body)}
        if len(distinct) >= 2:
            return True
        # Context keywords (e.g. "human trafficking") are too broad to stand on
        # their own — they only count when a core topic keyword co-occurs.
        if self._context_re is not None:
            full = title + " " + body
            if self._context_re.search(full) and self._keyword_re.search(full):
                return True
        return False

    def emoji_for(self, b: dict) -> str:
        s = " ".join([b.get("title", ""), b.get("abstract", ""), b.get("subjects", "")]).lower()
        for rule in self.emojis:
            patterns = rule.get("match") or []
            emoji = rule.get("emoji") or ""
            if not emoji or not patterns:
                continue
            if any(p.lower() in s for p in patterns):
                return emoji
        return self.default_emoji

    def primary_group_for(self, b: dict) -> str | None:
        """Classify a bill into one of the named keyword_groups buckets.
        Returns the first bucket whose keyword regex matches the bill's
        title / abstract / subjects, or None if no bucket matches (or no
        buckets are configured). Iteration order follows config order, so
        a bill matching multiple buckets is assigned to the first one —
        put the bucket you want to protect from starvation first."""
        if not self._keyword_group_res:
            return None
        text = " ".join([
            b.get("title", ""),
            b.get("abstract", ""),
            b.get("subjects", ""),
        ])
        for name, pattern in self._keyword_group_res.items():
            if pattern.search(text):
                return name
        return None

    # ------------------------------------------------------------------
    # Prompts and copy
    # ------------------------------------------------------------------

    def summary_system_prompt(self, max_chars: int = 160, max_sentences: int = 1) -> str:
        sentence_rule = (
            f"Output one or two plain-text sentences totaling under {max_chars} characters"
            if max_sentences > 1
            else f"Output exactly ONE plain-text sentence under {max_chars} characters"
        )
        return (
            f"You summarize US legislative bills for a civic-engagement Bluesky bot "
            f"focused on {self.prompt_topic}. The bill's title is shown directly above "
            f"your summary, so DO NOT restate or paraphrase the title — lead with the "
            f"substantive action (who must do what, what changes, who is affected). "
            f"Never begin with or restate the bill's number, type, or chamber (do not "
            f"write 'House Resolution 6023', 'Senate Bill 12', or 'Kansas Resolution') — "
            f"that label already appears above your summary. "
            f"Write in plain layman's terms: replace legislative jargon with the everyday "
            f"word a general audience uses — 'block' or 'override' instead of 'preempt' or "
            f"'preemption', 'sets aside money for' instead of 'appropriates', 'lets' "
            f"instead of 'authorizes', 'requires' instead of 'mandates', 'bans' instead of "
            f"'prohibits'. If the measure is a resolution or memorial that only states an "
            f"opinion, say plainly what the legislature is urging, supporting, or opposing "
            f"and name the concrete areas it affects in everyday language. "
            f"Spell out agency and program acronyms in plain English (e.g. write 'the "
            f"Environmental Protection Agency', not 'EPA') unless the acronym is universally "
            f"known to a general audience (FBI, NASA, DNA). Do not introduce facts not "
            f"present in the title or description — never invent agency names, statute "
            f"citations, or states. "
            f"The description may be the bill's full statutory text containing bill-number "
            f"prefixes, section and chapter citations, line numbers, and a drafter's name — "
            f"ignore those and summarize the bill's substantive policy change. Read the whole "
            f"text first, then translate it into plain layman's terms a non-lawyer "
            f"understands: never copy legalese verbatim, and do not repeat the same word or "
            f"phrase. When two sentences are "
            f"allowed, use the second only to add a concrete, substantive detail (who is "
            f"affected, key requirement, dollar threshold, penalty, or effective date) — never filler. "
            f"{sentence_rule}, neutral and "
            f"concrete. No emoji, no hashtags, no editorializing, no surrounding quotes, "
            f"no leading phrases like 'This bill', 'The bill', or 'The Act'. Do not "
            f"include any preamble, explanation, or trailing notes."
        )


    def headline_system_prompt(self) -> str:
        return (
            f"You write short Bluesky headlines for US legislative bills focused on "
            f"{self.prompt_topic}. Rewrite the bill as a plain-English headline "
            f"under 70 characters. Strip statute verbs ('Requiring', 'Prohibiting', "
            f"'Concerning', 'Relating to', 'An act to', 'Establishing'). Use noun "
            f"phrases, not full sentences (e.g. 'Daily recess for elementary students; "
            f"Kansas fitness test'). Keep the substantive change. The description may "
            f"be a long statute — write a headline about the bill's overall purpose, "
            f"never echo a section or chapter header verbatim. Spell out unfamiliar "
            f"acronyms. Do not invent facts not present in the title or description — "
            f"never invent agency names, statute citations, or states. No emoji, no "
            f"hashtags, no surrounding quotes, no trailing period, no preamble — output "
            f"only the headline text."
        )

    # ------------------------------------------------------------------
    # Paths and credentials
    # ------------------------------------------------------------------

    def state_file_path(self) -> Path:
        return TOPICS_DIR / self.name / "bills_used.json"

    def bills_raw_dir(self) -> Path:
        return TOPICS_DIR / self.name / "bills_raw"

    # Full bill text extracted from each posted bill's PDF lives here as plain
    # .txt files (one per posted action), so the actual legislative body is
    # readable without digging through the raw JSON record.
    def bills_full_text_dir(self) -> Path:
        return TOPICS_DIR / self.name / "bills_full_text"

    # Weekly-digest highlights live in their own weekly_digest/ subfolder so
    # the raw artifacts for bills featured in the Sunday thread don't mix
    # with the daily feed's bills_raw/.
    def weekly_digest_bills_raw_dir(self) -> Path:
        return TOPICS_DIR / self.name / "weekly_digest" / "bills_raw"

    # X/Twitter state lives in its own subfolder (default "x", overridable via
    # x_subdir in config.yml) so its dedup file and raw artifacts sit beside —
    # but never collide with — Bluesky's.
    def x_state_file_path(self) -> Path:
        return TOPICS_DIR / self.name / self.x_subdir / "bills_used.json"

    def x_bills_raw_dir(self) -> Path:
        return TOPICS_DIR / self.name / self.x_subdir / "bills_raw"

    def x_bills_full_text_dir(self) -> Path:
        return TOPICS_DIR / self.name / self.x_subdir / "bills_full_text"

    # X weekly-digest highlights get their own weekly_digest/ subfolder under
    # the X state dir so the Friday thread's raw artifacts don't mix with the
    # daily X feed's bills_raw/. Mirrors weekly_digest_bills_raw_dir() on the
    # Bluesky side.
    def x_weekly_digest_bills_raw_dir(self) -> Path:
        return TOPICS_DIR / self.name / self.x_subdir / "weekly_digest" / "bills_raw"

    # Threads/Meta state lives in its own subfolder (default "meta-threads") so
    # its dedup file and raw artifacts sit beside — but never collide with —
    # Bluesky's and X's. Mirrors the x_* helpers above.
    def threads_state_file_path(self) -> Path:
        return TOPICS_DIR / self.name / self.threads_subdir / "bills_used.json"

    def threads_bills_raw_dir(self) -> Path:
        return TOPICS_DIR / self.name / self.threads_subdir / "bills_raw"

    def threads_bills_full_text_dir(self) -> Path:
        return TOPICS_DIR / self.name / self.threads_subdir / "bills_full_text"

    def threads_weekly_digest_bills_raw_dir(self) -> Path:
        return TOPICS_DIR / self.name / self.threads_subdir / "weekly_digest" / "bills_raw"

    def _secret_suffix(self) -> str:
        return self.name.upper()

    def bluesky_handle_env(self) -> str:
        return f"BLUESKY_HANDLE_{self._secret_suffix()}"

    def bluesky_password_env(self) -> str:
        return f"BLUESKY_APP_PASSWORD_{self._secret_suffix()}"

    def bluesky_handle(self) -> str:
        return _read_secret(self.bluesky_handle_env())

    def bluesky_password(self) -> str:
        return _read_secret(self.bluesky_password_env())


# ---------------------------------------------------------------------------
# Secret resolution
#
# In the shared workflow we expose toJSON(secrets) as a single ALL_SECRETS env
# var so adding a new topic never requires editing the workflow file. The
# script tries plain env vars first (so local dev with a single
# BLUESKY_HANDLE_TRANSPORTATION export still works) and falls back to the
# JSON map.
# ---------------------------------------------------------------------------

_ALL_SECRETS_CACHE: dict[str, str] | None = None


def _all_secrets() -> dict[str, str]:
    global _ALL_SECRETS_CACHE
    if _ALL_SECRETS_CACHE is not None:
        return _ALL_SECRETS_CACHE
    raw = os.environ.get("ALL_SECRETS", "")
    if not raw:
        _ALL_SECRETS_CACHE = {}
        return _ALL_SECRETS_CACHE
    try:
        parsed = json.loads(raw)
        _ALL_SECRETS_CACHE = {str(k): str(v) for k, v in parsed.items() if v is not None}
    except json.JSONDecodeError:
        _ALL_SECRETS_CACHE = {}
    return _ALL_SECRETS_CACHE


def _read_secret(env_name: str) -> str:
    direct = os.environ.get(env_name)
    if direct:
        return direct
    return _all_secrets().get(env_name, "")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_topics() -> list[str]:
    if not TOPICS_DIR.exists():
        return []
    out: list[str] = []
    for child in sorted(TOPICS_DIR.iterdir()):
        if child.is_dir() and (child / "config.yml").exists():
            out.append(child.name)
    return out


def load_active_topic() -> Topic:
    """Resolve the topic for this run from the BOT_TOPIC env var."""
    name = os.environ.get("BOT_TOPIC", "").strip()
    if not name:
        raise RuntimeError(
            "BOT_TOPIC env var is required. Set it to a folder name under "
            f"topics/ — available: {', '.join(list_topics()) or '(none)'}."
        )
    return Topic.load(name)
