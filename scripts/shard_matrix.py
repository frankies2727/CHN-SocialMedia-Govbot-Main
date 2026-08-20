#!/usr/bin/env python3
"""Emit the GitHub Actions matrix for the Bluesky shard jobs, labelled by topic.

The Bluesky daily poster and weekly digest each fan out into NUM_SHARDS parallel
jobs, and GitHub names them "post (0)" … "post (4)" — which says nothing about
what any of them is doing. This builds an `include:` matrix carrying both the
shard number and a short label naming the topics that shard will handle, so the
Actions UI reads "post 1 · Justice, Housing, Taxes".

The shard assignment MUST mirror the posting step's own loop, which walks
`topics/*/` in glob (alphabetical) order and keeps a topic when
`index % NUM_SHARDS == SHARD`. That loop stays the source of truth for what
actually runs; this script only produces the label, so a drift here mislabels a
job but never changes which topics it posts.

Usage: python scripts/shard_matrix.py [num_shards]   # prints compact JSON
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topic import Topic, list_topics  # noqa: E402


def build(num_shards: int) -> dict:
    buckets: list[list[str]] = [[] for _ in range(num_shards)]
    for i, name in enumerate(list_topics()):
        try:
            label = Topic.load(name).short_name
        except Exception:
            # A malformed config.yml must not take the whole run down — the
            # posting loop skips on its own terms, so fall back to the folder.
            label = name
        buckets[i % num_shards].append(label)
    return {
        "include": [
            {"shard": n, "label": ", ".join(labels) if labels else "no topics"}
            for n, labels in enumerate(buckets)
        ]
    }


if __name__ == "__main__":
    shards = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(json.dumps(build(shards), separators=(",", ":")))
