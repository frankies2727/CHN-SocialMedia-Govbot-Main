#!/usr/bin/env python3
"""
Shared cross-topic selection for the single-account weekly digests.

Bluesky and X run one account per topic, so their weekly digests cover a single
topic. Threads and Instagram each publish to ONE account that spans every topic,
so their weekly digests are all-topics: a single post (a Threads thread / an
Instagram carousel) that draws up to N highlights from a WIDE VARIETY of topics,
each tagged with the topic it belongs to.

This module holds the platform-agnostic half of that: load every topics/ config,
match bills, rank each topic's activity with the shared significance scorer, and
round-robin across topics for breadth. The platform digests
(weekly_digest_meta_threads.py, weekly_digest_instagram.py) own the rest —
composing and publishing in their medium, and re-pointing the pipeline's active
TOPIC per bill so each post's copy/emoji/label comes from the bill's own topic.

Every pick is stamped with the topic that claimed it:
    b["_topic"]      -> the Topic object
    b["_topic_name"] -> its folder name
and carries b["_score"] (the significance score) and b["_raw"] (the verbatim
bills.jsonl record, for the digest's raw-artifact trail).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Callable

from post_to_bluesky import extract_fields
from topic import Topic, list_topics
from weekly_digest_bluesky import (
    DIGEST_PER_STATE_CAP,
    LOOKBACK_FALLBACK_WINDOWS,
    _landscape_unique_bills,
    candidates_in_window,
    score_action,
    select_highlights,
)


def extract_all(records: list[dict]) -> list[dict]:
    """Run extract_fields once over the corpus, stashing each source record on
    its bill so the digest can dump the verbatim bills.jsonl line later. The
    resulting dicts are shared across every topic's match list (a bill can match
    more than one topic); the round-robin dedups by bill and stamps the winning
    topic, so sharing is safe."""
    out: list[dict] = []
    for r in records:
        b = extract_fields(r)
        if b:
            b["_raw"] = r
            out.append(b)
    return out


def build_matched_by_topic(
    extracted: list[dict],
    on_skip: Callable[[str, Exception], None] | None = None,
) -> dict[str, tuple[Topic, list[dict]]]:
    """Load every topics/ config and build {name: (topic, matching_bills)} from
    the shared extracted dicts. A topic whose config fails to load is skipped
    (on_skip is called with its name and the error, if provided)."""
    matched: dict[str, tuple[Topic, list[dict]]] = {}
    for name in list_topics():
        try:
            topic = Topic.load(name)
        except (FileNotFoundError, ValueError) as e:
            if on_skip is not None:
                on_skip(name, e)
            continue
        matched[name] = (topic, [b for b in extracted if topic.matches(b)])
    return matched


def rank_topics_in_window(
    matched_by_topic: dict[str, tuple[Topic, list[dict]]],
    today: datetime,
    window: int,
    per_state_cap: int = DIGEST_PER_STATE_CAP,
) -> dict[str, tuple[Topic, list[dict]]]:
    """For one lookback window, return {name: (topic, ranked_bills)} for every
    topic with at least one bill update in the window. ranked_bills is that
    topic's full significance-ranked, per-state-capped list (no per-topic
    truncation — the cross-topic round-robin picks from it)."""
    per_topic: dict[str, tuple[Topic, list[dict]]] = {}
    for name, (topic, matched) in matched_by_topic.items():
        cands = candidates_in_window(matched, today, window)
        if not cands:
            continue
        ranked = select_highlights(
            cands, max_highlights=None, per_state_cap=per_state_cap)
        per_topic[name] = (topic, ranked)
    return per_topic


def choose_active_window(
    matched_by_topic: dict[str, tuple[Topic, list[dict]]],
    today: datetime,
    per_state_cap: int = DIGEST_PER_STATE_CAP,
    windows: list[int] = LOOKBACK_FALLBACK_WINDOWS,
) -> tuple[int, dict[str, tuple[Topic, list[dict]]]]:
    """Try each lookback window in turn, returning (window, per_topic) for the
    first window in which ANY topic has activity so a quiet week widens the lens
    instead of going silent. per_topic is empty (and window is the widest tried)
    when nothing turns up anywhere."""
    per_topic: dict[str, tuple[Topic, list[dict]]] = {}
    chosen = windows[0]
    for window in windows:
        per_topic = rank_topics_in_window(matched_by_topic, today, window, per_state_cap)
        if per_topic:
            return window, per_topic
        chosen = window
    return chosen, {}


def merge_across_topics(
    per_topic: dict[str, tuple[Topic, list[dict]]],
    cap: int,
    per_state_cap: int = DIGEST_PER_STATE_CAP,
) -> list[dict]:
    """Round-robin across topics for breadth: take each topic's strongest
    remaining bill in turn until we hit ``cap``, so the digest spreads over as
    many topics as possible before doubling up on any one. Topics are visited
    strongest-first (by their top bill's score). Dedups by bill across topics —
    a bill matched by two topics is claimed by whichever reaches it first — and
    enforces a global per-state cap so one busy statehouse can't dominate. The
    claiming topic is stamped on each pick as _topic / _topic_name."""
    order = sorted(
        per_topic,
        key=lambda n: (-per_topic[n][1][0]["_score"], n),
    )
    iters = {n: iter(per_topic[n][1]) for n in order}
    picked: list[dict] = []
    seen_bills: set[tuple[str, str]] = set()
    per_state: Counter[str] = Counter()

    while len(picked) < cap:
        progressed = False
        for name in order:
            if len(picked) >= cap:
                break
            topic = per_topic[name][0]
            for b in iters[name]:
                key = (b["state"], b["identifier"])
                if key in seen_bills:
                    continue
                st = b["state"] or "?"
                if per_state[st] >= per_state_cap:
                    continue
                seen_bills.add(key)
                per_state[st] += 1
                b["_topic"] = topic
                b["_topic_name"] = name
                picked.append(b)
                progressed = True
                break
        if not progressed:
            break
    return picked


def landscape_picks(
    matched_by_topic: dict[str, tuple[Topic, list[dict]]],
    cap: int,
    per_state_cap: int = DIGEST_PER_STATE_CAP,
) -> list[dict]:
    """Quiet-week fallback: when no topic had floor activity in any window, pick
    each topic's most-recent unique bills, round-robined across topics for
    breadth and capped at ``cap`` (same merge, and its global per-state cap, as
    the highlights path)."""
    per_topic: dict[str, tuple[Topic, list[dict]]] = {}
    for name, (topic, matched) in matched_by_topic.items():
        if not matched:
            continue
        unique = _landscape_unique_bills(matched)
        for b in unique:
            b["_score"] = score_action(b["action_desc"])
        unique.sort(key=lambda b: (b["_score"], b["action_date"]), reverse=True)
        per_topic[name] = (topic, unique)
    return merge_across_topics(per_topic, cap, per_state_cap)
