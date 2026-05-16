#!/usr/bin/env python3
"""
Category configuration loader.

Each Bluesky bot is one category (transportation, immigration, taxation, …).
A category is described by a YAML file at categories/<name>/config.yml; all
of its per-bot state lives in the same folder. The loader exposes a single
Category object that the post + digest scripts use to filter bills, pick
emojis, and look up Bluesky credentials.

Adding a new category is a drop-in operation: create the folder, add the
config.yml, add BLUESKY_HANDLE_<NAME> + BLUESKY_APP_PASSWORD_<NAME> repo
secrets. The shared workflow loops over categories/ and picks it up on
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
CATEGORIES_DIR = ROOT / "categories"


@dataclass
class Category:
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

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, name: str) -> "Category":
        path = CATEGORIES_DIR / name / "config.yml"
        if not path.exists():
            raise FileNotFoundError(
                f"Category config not found: {path}. "
                f"Expected a folder at categories/{name}/ with config.yml."
            )
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg_name = data.get("name") or name
        if cfg_name != name:
            raise ValueError(
                f"Category folder name ({name!r}) does not match "
                f"config.yml name ({cfg_name!r})."
            )

        keywords = list(data.get("keywords") or [])
        if not keywords:
            raise ValueError(f"Category {name!r}: keywords list is empty.")

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
        title = (b.get("title") or "").lower()
        if self._keyword_re.search(title):
            return True
        body = " ".join([b.get("abstract", ""), b.get("subjects", "")]).lower()
        distinct = {m.group(1).lower() for m in self._keyword_re.finditer(body)}
        if len(distinct) >= 2:
            return True
        # Context keywords (e.g. "human trafficking") are too broad to stand on
        # their own — they only count when a core category keyword co-occurs.
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

    # ------------------------------------------------------------------
    # Prompts and copy
    # ------------------------------------------------------------------

    def summary_system_prompt(self) -> str:
        return (
            f"You summarize US legislative bills for a civic-engagement Bluesky bot "
            f"focused on {self.prompt_topic}. The bill's title is shown directly above "
            f"your summary, so DO NOT restate or paraphrase the title — lead with the "
            f"substantive action (who must do what, what changes, who is affected). "
            f"Spell out agency and program acronyms in plain English (e.g. write 'the "
            f"Environmental Protection Agency', not 'EPA') unless the acronym is universally "
            f"known to a general audience (FBI, NASA, DNA). Do not introduce facts not "
            f"present in the title or description — never invent agency names, statute "
            f"citations, or states. "
            f"The description may be a long statute containing bill-number prefixes, "
            f"section and chapter citations, and a drafter's name — ignore those and "
            f"summarize the bill's substantive policy change. "
            f"Output exactly ONE plain-text sentence under 160 characters, neutral and "
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
        return CATEGORIES_DIR / self.name / "bills_used.json"

    def bills_raw_dir(self) -> Path:
        return CATEGORIES_DIR / self.name / "bills_raw"

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
# var so adding a new category never requires editing the workflow file. The
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

def list_categories() -> list[str]:
    if not CATEGORIES_DIR.exists():
        return []
    out: list[str] = []
    for child in sorted(CATEGORIES_DIR.iterdir()):
        if child.is_dir() and (child / "config.yml").exists():
            out.append(child.name)
    return out


def load_active_category() -> Category:
    """Resolve the category for this run from the BOT_CATEGORY env var."""
    name = os.environ.get("BOT_CATEGORY", "").strip()
    if not name:
        raise RuntimeError(
            "BOT_CATEGORY env var is required. Set it to a folder name under "
            f"categories/ — available: {', '.join(list_categories()) or '(none)'}."
        )
    return Category.load(name)
