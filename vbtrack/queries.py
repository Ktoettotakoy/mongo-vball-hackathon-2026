"""Example read-side queries against the stored data."""
from __future__ import annotations

import csv
import sys

from bson import ObjectId
from pymongo.database import Database


def print_sessions(db: Database) -> None:
    for s in db.sessions.find().sort("started_at", -1):
        print(f"{s['_id']}  {s['status']:<11} {s.get('frame_count', 0):>7} frames  "
              f"{'live' if s['live'] else 'file'}  {s['source']}  {s['started_at']:%Y-%m-%d %H:%M}")


def ball_trajectory(db: Database, session: str, out_csv: str | None = None, detected_only: bool = False):
    q = {"session_id": ObjectId(session), "ball": {"$ne": None}}
    if detected_only:
        q["ball.interpolated"] = False
    cur = db.frames.find(q, {"_id": 0, "frame": 1, "t": 1, "ball": 1}).sort("frame", 1)
    rows = [{"frame": d["frame"], "t": d["t"], "x": d["ball"]["x"], "y": d["ball"]["y"],
             "vx": d["ball"]["vx"], "vy": d["ball"]["vy"], "conf": d["ball"]["conf"],
             "interpolated": d["ball"]["interpolated"]} for d in cur]
    fh = open(out_csv, "w", newline="") if out_csv else sys.stdout
    w = csv.DictWriter(fh, fieldnames=["frame", "t", "x", "y", "vx", "vy", "conf", "interpolated"])
    w.writeheader()
    w.writerows(rows)
    if out_csv:
        fh.close()
        print(f"wrote {len(rows)} points to {out_csv}")
    return rows


def print_players(db: Database, session: str) -> None:
    for t in db.tracks.find({"session_id": ObjectId(session)}).sort("frames_seen", -1):
        print(f"track {t['track_id']:>4}  frames {t['first_frame']}-{t['last_frame']} "
              f"(seen {t['frames_seen']})  avg conf {t['avg_conf']:.2f}  "
              f"avg foot ({t['avg_foot']['x']:.0f}, {t['avg_foot']['y']:.0f})")


def player_positions(db: Database, session: str, track_id: int):
    """Foot positions over time for one player (e.g. for a court heatmap)."""
    return list(db.frames.aggregate([
        {"$match": {"session_id": ObjectId(session), "players.track_id": track_id}},
        {"$unwind": "$players"},
        {"$match": {"players.track_id": track_id}},
        {"$project": {"_id": 0, "frame": 1, "t": 1, "x": "$players.foot.x", "y": "$players.foot.y"}},
        {"$sort": {"frame": 1}},
    ]))
