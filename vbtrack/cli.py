"""Command line entry point.

  python -m vbtrack run --source match.mp4
  python -m vbtrack run --source rtsp://cam/stream --show
  python -m vbtrack run --source 0                       # webcam
  python -m vbtrack sessions
  python -m vbtrack ball --session <id> --csv ball.csv
  python -m vbtrack players --session <id>
"""
from __future__ import annotations

import argparse
import logging
import os

from .storage import MongoStore


def parse_roi(s: str | None):
    if not s:
        return None
    pts = [tuple(int(float(v)) for v in p.split(",")) for p in s.split(";") if p.strip()]
    if len(pts) < 3:
        raise argparse.ArgumentTypeError("ROI needs at least 3 points: 'x1,y1;x2,y2;x3,y3'")
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vbtrack", description="Volleyball player + ball tracking into MongoDB")
    ap.add_argument("--mongo", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", default=os.getenv("MONGO_DB", "volleyball"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process a video file or live stream")
    r.add_argument("--source", required=True, help="video path, webcam index, or rtsp/http URL")
    r.add_argument("--player-weights", default="yolo11n.pt")
    r.add_argument("--ball-weights", default=None, help="custom volleyball detector (.pt)")
    r.add_argument("--ball-class", type=int, default=0)
    r.add_argument("--imgsz", type=int, default=1280)
    r.add_argument("--ball-imgsz", type=int, default=1280)
    r.add_argument("--player-conf", type=float, default=0.35)
    r.add_argument("--ball-conf", type=float, default=0.15)
    r.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--roi", type=parse_roi, default=None, help="court polygon 'x1,y1;x2,y2;...'")
    r.add_argument("--device", default=None)
    r.add_argument("--show", action="store_true")
    r.add_argument("--save-video", default=None)
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--batch-size", type=int, default=100)

    sub.add_parser("sessions", help="list sessions")
    b = sub.add_parser("ball", help="dump ball trajectory")
    b.add_argument("--session", required=True)
    b.add_argument("--csv", default=None)
    b.add_argument("--detected-only", action="store_true")
    p = sub.add_parser("players", help="player track summary")
    p.add_argument("--session", required=True)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "run":
        from .pipeline import Config, Pipeline  # heavy import only when needed
        store = MongoStore(args.mongo, args.db, batch_size=args.batch_size)
        cfg = Config(
            source=args.source, player_weights=args.player_weights, ball_weights=args.ball_weights,
            ball_class=args.ball_class, imgsz=args.imgsz, ball_imgsz=args.ball_imgsz,
            player_conf=args.player_conf, ball_conf=args.ball_conf, tracker=args.tracker,
            stride=args.stride, roi=args.roi, device=args.device, show=args.show,
            save_video=args.save_video, max_frames=args.max_frames,
        )
        try:
            Pipeline(cfg, store).run()
        finally:
            store.close()
        return

    from . import queries
    store = MongoStore(args.mongo, args.db)
    try:
        if args.cmd == "sessions":
            queries.print_sessions(store.db)
        elif args.cmd == "ball":
            queries.ball_trajectory(store.db, args.session, args.csv, args.detected_only)
        elif args.cmd == "players":
            queries.print_players(store.db, args.session)
    finally:
        store.close()


if __name__ == "__main__":
    main()
