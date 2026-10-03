"""MongoDB persistence.

Collections
-----------
sessions : one doc per run (video file or live stream)
frames   : one doc per processed frame: players + ball   (bulk-inserted in batches)
tracks   : one doc per player track, rebuilt from `frames` when a session ends (+ voted jersey number)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoStore:
    def __init__(self, uri: str = "mongodb://localhost:27017", db: str = "volleyball",
                 batch_size: int = 100, flush_seconds: float = 1.0,
                 client: MongoClient | None = None):
        self.client = client or MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db: Database = self.client[db]
        self.batch_size = batch_size
        self.flush_seconds = flush_seconds
        self._buf: list[dict] = []
        self._last_flush = time.monotonic()
        self.session_id: ObjectId | None = None
        self.ensure_indexes()

    # ---------- setup ----------
    def ensure_indexes(self) -> None:
        self.db.frames.create_index([("session_id", ASCENDING), ("frame", ASCENDING)], unique=True)
        self.db.frames.create_index([("session_id", ASCENDING), ("players.track_id", ASCENDING)])
        self.db.tracks.create_index([("session_id", ASCENDING), ("track_id", ASCENDING)], unique=True)
        self.db.sessions.create_index([("started_at", ASCENDING)])

    # ---------- session lifecycle ----------
    def start_session(self, source: str, fps: float, width: int, height: int,
                      live: bool, meta: dict[str, Any] | None = None) -> ObjectId:
        doc = {
            "source": source, "live": live, "fps": fps,
            "width": width, "height": height,
            "started_at": utcnow(), "status": "running",
            "frame_count": 0, **(meta or {}),
        }
        self.session_id = self.db.sessions.insert_one(doc).inserted_id
        return self.session_id

    def update_session(self, fields: dict[str, Any]) -> None:
        self.db.sessions.update_one({"_id": self.session_id}, {"$set": fields})

    def add_frame(self, doc: dict) -> None:
        doc["session_id"] = self.session_id
        self._buf.append(doc)
        if len(self._buf) >= self.batch_size or time.monotonic() - self._last_flush >= self.flush_seconds:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self.db.frames.insert_many(self._buf, ordered=False)
            self.db.sessions.update_one({"_id": self.session_id},
                                        {"$inc": {"frame_count": len(self._buf)}})
            self._buf = []
        self._last_flush = time.monotonic()

    def end_session(self, status: str = "done", track_numbers: dict[int, dict] | None = None) -> None:
        self.flush()
        self.build_tracks()
        for tid, j in (track_numbers or {}).items():
            self.db.tracks.update_one({"session_id": self.session_id, "track_id": tid},
                                      {"$set": {"number": j["number"], "jersey_votes": j["votes"]}})
        self.db.sessions.update_one({"_id": self.session_id},
                                    {"$set": {"status": status, "ended_at": utcnow()}})

    # ---------- derived data ----------
    def build_tracks(self) -> None:
        """Summarise each player track_id into the `tracks` collection."""
        pipeline = [
            {"$match": {"session_id": self.session_id}},
            {"$unwind": "$players"},
            {"$group": {
                "_id": {"session_id": "$session_id", "track_id": "$players.track_id"},
                "first_frame": {"$min": "$frame"},
                "last_frame": {"$max": "$frame"},
                "first_t": {"$min": "$t"},
                "last_t": {"$max": "$t"},
                "frames_seen": {"$sum": 1},
                "avg_conf": {"$avg": "$players.conf"},
                "avg_x": {"$avg": "$players.foot.x"},
                "avg_y": {"$avg": "$players.foot.y"},
            }},
            {"$project": {
                "_id": 0, "session_id": "$_id.session_id", "track_id": "$_id.track_id",
                "first_frame": 1, "last_frame": 1, "first_t": 1, "last_t": 1,
                "frames_seen": 1, "avg_conf": 1, "avg_foot": {"x": "$avg_x", "y": "$avg_y"},
            }},
            {"$merge": {"into": "tracks", "on": ["session_id", "track_id"],
                        "whenMatched": "replace", "whenNotMatched": "insert"}},
        ]
        self.db.frames.aggregate(pipeline)

    def close(self) -> None:
        self.client.close()
