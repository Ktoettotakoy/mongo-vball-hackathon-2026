"""Read-only HTTP API for the dashboard.

  uvicorn vbtrack.api:app --reload --port 8000

The Vite dev server proxies /api to this server (dashboard/vbtrack/vite.config.ts).
"""
from __future__ import annotations

from fastapi import FastAPI, Query

from .mongodb import MongoDB
from .stats import player_stats

app = FastAPI(title="vbtrack API")
_db = MongoDB()
events = _db.collection
runs = events.database["label_runs"]

EVENT_FIELDS = {"_id": 0, "video": 1, "frame": 1, "t": 1, "player_no": 1, "track_id": 1,
                "action": 1, "successful": 1, "block_type": 1}


@app.get("/api/health")
def health() -> dict:
    events.database.command("ping")
    return {"ok": True}


@app.get("/api/videos")
def videos() -> list[dict]:
    """Labelling runs, one per video: status and event count."""
    return list(runs.find({}, {"_id": 0, "raw_response": 0}).sort("video", 1))


@app.get("/api/events")
def list_events(video: str | None = Query(None)) -> list[dict]:
    q = {"video": video} if video else {}
    return list(events.find(q, EVENT_FIELDS).sort([("video", 1), ("frame", 1)]))


@app.get("/api/players")
def players(video: str | None = Query(None)) -> list[dict]:
    """Aggregated stats per track_id (all videos, or one video)."""
    return player_stats(events, video)
