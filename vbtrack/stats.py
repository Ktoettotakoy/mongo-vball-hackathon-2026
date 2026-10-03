"""Per-player statistics from LLM-labelled action events.

Events live in the `events` collection (see vbtrack/label.py). Players are keyed by
`track_id` (tracker ID such as "#19"), not by jersey number.

  python -m vbtrack.stats                      # all videos, print table, refresh `player_stats`
  python -m vbtrack.stats --video annotated_10.mp4
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from pymongo.collection import Collection

from .mongodb import MongoDB

STATS_COLLECTION = os.getenv("MONGO_STATS_COLLECTION", "player_stats")


def _count(action: str, successful: bool | None = None) -> dict:
    cond: list[Any] = [{"$eq": ["$action", action]}]
    if successful is not None:
        cond.append({"$eq": ["$successful", successful]})
    return {"$sum": {"$cond": [{"$and": cond}, 1, 0]}}


def player_stats_pipeline(video: str | None = None) -> list[dict]:
    """Aggregation that turns raw events into one row per track_id."""
    match: dict[str, Any] = {"track_id": {"$ne": None}}
    if video:
        match["video"] = video
    return [
        {"$match": match},
        {"$group": {
            "_id": "$track_id",
            "player_no": {"$max": "$player_no"},   # any jersey read wins over null
            "videos": {"$addToSet": "$video"},
            "total": {"$sum": 1},
            "successful": {"$sum": {"$cond": ["$successful", 1, 0]}},
            "serves": _count("serve"),
            "serves_won": _count("serve", True),
            "spikes": _count("spike"),
            "kills": _count("spike", True),
            "blocks": _count("block"),
            "blocks_won": _count("block", True),
            "full_blocks": {"$sum": {"$cond": [{"$eq": ["$block_type", "full"]}, 1, 0]}},
            "solo_blocks": {"$sum": {"$cond": [{"$eq": ["$block_type", "solo"]}, 1, 0]}},
        }},
        {"$project": {
            "_id": 0, "track_id": "$_id", "player_no": 1, "videos": 1,
            "total": 1, "successful": 1, "serves": 1, "serves_won": 1,
            "spikes": 1, "kills": 1, "blocks": 1, "blocks_won": 1,
            "full_blocks": 1, "solo_blocks": 1,
            "success_pct": {"$round": [{"$multiply": [
                {"$divide": ["$successful", {"$max": ["$total", 1]}]}, 100]}, 1]},
        }},
        {"$sort": {"kills": -1, "total": -1, "track_id": 1}},
    ]


def player_stats(events: Collection, video: str | None = None) -> list[dict]:
    return list(events.aggregate(player_stats_pipeline(video)))


def refresh_player_stats(events: Collection) -> int:
    """Rebuild the `player_stats` collection (all videos) with $merge. Return the row count."""
    pipeline = player_stats_pipeline() + [
        {"$merge": {"into": STATS_COLLECTION, "on": "track_id",
                    "whenMatched": "replace", "whenNotMatched": "insert"}},
    ]
    target = events.database[STATS_COLLECTION]
    target.create_index("track_id", unique=True)
    target.delete_many({})   # drop players that no longer have events
    events.aggregate(pipeline)
    return target.count_documents({})


def print_table(rows: list[dict]) -> None:
    head = f"{'player':>8} {'no':>4} {'serves':>8} {'spikes(k)':>10} {'blocks(w)':>10} {'total':>6} {'succ%':>6}"
    print(head)
    print("-" * len(head))
    for r in rows:
        no = "-" if r.get("player_no") is None else str(r["player_no"])
        print(f"{r['track_id']:>8} {no:>4} {r['serves']:>8} {r['spikes']:>5}({r['kills']:>2})"
              f" {r['blocks']:>6}({r['blocks_won']:>2}) {r['total']:>6} {r['success_pct']:>6}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vbtrack.stats", description="Aggregate per-player action stats")
    ap.add_argument("--video", default=None, help="limit to one video (file name)")
    args = ap.parse_args(argv)
    db = MongoDB()
    try:
        rows = player_stats(db.collection, args.video)
        print_table(rows)
        if not args.video:
            n = refresh_player_stats(db.collection)
            print(f"\n{STATS_COLLECTION}: {n} players written")
    finally:
        db.client.close()


if __name__ == "__main__":
    main()
