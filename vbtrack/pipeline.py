"""Main loop: read frames -> person + ball detection -> trackers -> MongoDB.

Default detector is the Hugging Face D-FINE volleyball model. --detector yolo keeps COCO YOLO.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

import cv2
import numpy as np
from ultralytics.engine.results import Boxes

from . import models
from .ball import BallCandidate, BallTracker
from .jersey import JerseyVoter
from .storage import MongoStore, utcnow

log = logging.getLogger("vbtrack")

COCO_PERSON = 0
COCO_SPORTS_BALL = 32


@dataclass
class Config:
    source: str
    detector: str = "dfine"
    dfine_repo: str = models.PERSON_BALL_REPO
    tile: int = 0
    player_weights: str = "yolo11n.pt"
    ball_weights: str | None = models.REF_BALL_WEIGHTS
    ball_class: int = 0
    imgsz: int = 1280
    ball_imgsz: int = 1280
    player_conf: float = 0.6
    ball_conf: float = 0.4
    ref_ball_conf: float = 0.7
    jersey: bool = True
    jersey_repo: str = models.JERSEY_REPO
    jersey_every: int = 3
    jersey_min_height: int = 70
    court: bool = True
    court_weights: str = models.COURT_WEIGHTS
    court_every: int = 5
    tracker: str = "bytetrack.yaml"
    stride: int = 1
    roi: list[tuple[int, int]] | None = None
    device: str | None = None
    show: bool = False
    save_video: str | None = None
    max_frames: int | None = None
    reconnect_attempts: int = 10
    extra_meta: dict = field(default_factory=dict)


def make_tracker(name: str, device: str | None = None):
    """Build an Ultralytics ByteTrack/BoT-SORT instance from its yaml config."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    args = IterableSimpleNamespace(**YAML.load(check_yaml(name)))
    args.device = device or "cpu"
    return TRACKER_MAP[args.tracker_type](args=args)


def is_live(source: str) -> bool:
    return source.isdigit() or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


class Pipeline:
    def __init__(self, cfg: Config, store: MongoStore):
        self.cfg = cfg
        self.store = store
        det_conf = min(cfg.player_conf, cfg.ball_conf)
        if cfg.detector == "dfine":
            self.dfine = models.PersonBallDetector(cfg.dfine_repo, cfg.device, tile=cfg.tile, threshold=det_conf)
            self.player_model = None
        else:
            self.dfine = None
            from ultralytics import YOLO
            self.player_model = YOLO(cfg.player_weights)
        self.ball_model = (
            models.YoloBallDetector(cfg.ball_weights, cfg.device, cfg.ball_imgsz, cfg.ref_ball_conf, cfg.ball_class)
            if cfg.ball_weights else None
        )
        self.player_tracker = make_tracker(cfg.tracker, cfg.device)
        self.ball = BallTracker(min_conf=cfg.ball_conf)
        self.reader = models.JerseyReader(cfg.jersey_repo, cfg.device) if cfg.jersey else None
        self.voter = JerseyVoter(min_conf=0.45, min_votes=1)
        self.court = models.CourtDetector(cfg.court_weights, cfg.device) if cfg.court and not cfg.roi else None
        self.roi = np.array(cfg.roi, dtype=np.int32) if cfg.roi else None
        self.live = is_live(cfg.source)
        self.net_x = None

    def _persons_and_balls(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.dfine:
            persons, balls = self.dfine(frame)
        else:
            classes = [COCO_PERSON] if self.ball_model else [COCO_PERSON, COCO_SPORTS_BALL]
            res = self.player_model.predict(
                frame, classes=classes, imgsz=self.cfg.imgsz,
                conf=min(self.cfg.player_conf, self.cfg.ball_conf),
                device=self.cfg.device, verbose=False,
            )[0]
            b = res.boxes.cpu().numpy()
            rows = np.hstack([b.xyxy, b.conf[:, None]]).astype(np.float32)
            cls = b.cls.astype(int)
            persons, balls = rows[cls == COCO_PERSON], rows[cls == COCO_SPORTS_BALL]
        if self.ball_model:
            extra = self.ball_model(frame)
            if len(extra):
                balls = np.vstack([balls, extra]) if len(balls) else extra
        return persons, balls

    def _detect(self, frame: np.ndarray):
        persons, balls = self._persons_and_balls(frame)
        det = np.hstack([persons, np.zeros((len(persons), 1), dtype=np.float32)])
        tracks = self.player_tracker.update(Boxes(det, frame.shape[:2]).cpu().numpy(), frame)
        players = []
        for x1, y1, x2, y2, tid, score, *_ in tracks:
            if score < self.cfg.player_conf:
                continue
            foot = (float(x1 + x2) / 2, float(y2))
            if self._off_court(foot, frame.shape[0]):
                continue
            if self._on_net(float(x1), float(x2), float(y1), float(y2)):
                continue
            players.append({
                "track_id": int(tid), "conf": round(float(score), 3),
                "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                "foot": {"x": round(foot[0], 1), "y": round(foot[1], 1)},
            })
        ball_cands = [
            BallCandidate(float(x1 + x2) / 2, float(y1 + y2) / 2, float(x2 - x1), float(y2 - y1), float(c))
            for x1, y1, x2, y2, c in balls
        ]
        return players, ball_cands, persons

    def _read_jerseys(self, frame: np.ndarray, players: list[dict]) -> None:
        readable = [p for p in players if p["bbox"][3] - p["bbox"][1] >= self.cfg.jersey_min_height]
        for p, (number, conf) in zip(readable, self.reader.read(frame, [p["bbox"] for p in readable])):
            if number is not None:
                p["jersey_read"] = {"number": number, "conf": round(conf, 3)}
                self.voter.add(p["track_id"], number, conf)
        for p in players:
            p["number"] = self.voter.number(p["track_id"])

    def _off_court(self, foot, height: int) -> bool:
        """Benches in this camera sit on the near baseline, at the bottom edge of the frame."""
        if foot[1] > height * 0.90:
            return True
        if self.roi is not None and cv2.pointPolygonTest(self.roi, foot, False) < 0:
            return True
        return False

    def _on_net(self, x1, x2, y1, y2) -> bool:
        """Referee stands are centered on the net pole. Attackers stand beside it."""
        net_x = getattr(self, "net_x", None)
        if net_x is None:
            return False
        cx = (x1 + x2) / 2
        return abs(cx - net_x) < 30 and (y2 - y1) > (x2 - x1)

    def _update_court(self, frame: np.ndarray, persons: np.ndarray) -> None:
        # Keypoints on this wide side view cluster on the net, not the court outline.
        # Keep the net x for the referee filter. Only trust a polygon that covers a real share of the frame.
        self.court.observe(frame)
        kp = self.court.keypoints()
        if kp is not None:
            xs = [float(x) for x, y, s in kp if s >= 0.5 and x == x]
            if xs:
                self.net_x = float(np.median(xs))
        if not self.court.ready:
            return
        roi = self.court.roi()
        h, w = frame.shape[:2]
        if roi:
            poly = np.array(roi, dtype=np.int32)
            if cv2.contourArea(poly) < 0.15 * w * h:
                log.info("court polygon too small (net cluster, not the floor), ignoring it")
                roi = None
        if roi:
            self.roi = np.array(roi, dtype=np.int32)
            kp = self.court.keypoints()
            self.store.update_session({"roi": roi, "court": {
                "source": self.cfg.court_weights,
                "keypoints": [
                    None if np.isnan(x) else {
                        "x": round(float(x), 1), "y": round(float(y), 1), "seen": round(float(s), 2),
                    }
                    for x, y, s in kp
                ],
            }})
            log.info("court found, ROI %s", roi)
        self.court = None

    def _draw(self, frame, players, ball):
        for p in players:
            x1, y1, x2, y2 = map(int, p["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = str(p["number"]) if p.get("number") else (
                p["jersey_read"]["number"] if p.get("jersey_read") else f"#{p['track_id']}")
            cv2.putText(frame, label, (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if ball:
            color = (0, 165, 255) if ball["interpolated"] else (0, 0, 255)
            cv2.circle(frame, (int(ball["x"]), int(ball["y"])), 8, color, 2)
        return frame

    def run(self) -> None:
        cap = open_capture(self.cfg.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1 or fps > 240:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sid = self.store.start_session(
            self.cfg.source, fps, w, h, self.live,
            meta={
                "detector": self.cfg.detector,
                "player_weights": self.cfg.dfine_repo if self.dfine else self.cfg.player_weights,
                "ball_weights": self.cfg.ball_weights,
                "jersey_weights": self.cfg.jersey_repo if self.reader else None,
                "tracker": self.cfg.tracker, "stride": self.cfg.stride, "roi": self.cfg.roi,
                **self.cfg.extra_meta,
            },
        )
        log.info("session %s started (%s, %.1f fps, %dx%d)", sid, "live" if self.live else "file", fps, w, h)

        writer = None
        if self.cfg.save_video:
            writer = cv2.VideoWriter(
                self.cfg.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps / self.cfg.stride, (w, h)
            )

        started_wall = utcnow()
        t0 = time.monotonic()
        frame_idx, processed, failures, status = -1, 0, 0, "done"
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if not self.live:
                        break
                    failures += 1
                    if failures > self.cfg.reconnect_attempts:
                        status = "stream_lost"
                        break
                    log.warning("stream read failed, reconnecting (%d/%d)", failures, self.cfg.reconnect_attempts)
                    cap.release()
                    time.sleep(min(2 * failures, 10))
                    try:
                        cap = open_capture(self.cfg.source)
                    except RuntimeError:
                        pass
                    self.ball.reset()
                    continue
                failures = 0
                frame_idx += 1
                if frame_idx % self.cfg.stride:
                    continue

                t = (time.monotonic() - t0) if self.live else frame_idx / fps
                players, cands, persons = self._detect(frame)
                if self.court and processed % self.cfg.court_every == 0:
                    self._update_court(frame, persons)
                    if self.roi is not None:
                        players = [
                            p for p in players
                            if cv2.pointPolygonTest(self.roi, (p["foot"]["x"], p["foot"]["y"]), False) >= 0
                        ]
                if self.reader and processed % self.cfg.jersey_every == 0:
                    self._read_jerseys(frame, players)
                elif self.reader:
                    for p in players:
                        p["number"] = self.voter.number(p["track_id"])
                bs = self.ball.step(cands)
                ball = None if bs is None else {
                    "x": round(bs.x, 1), "y": round(bs.y, 1),
                    "vx": round(bs.vx, 2), "vy": round(bs.vy, 2),
                    "conf": None if bs.conf is None else round(bs.conf, 3),
                    "bbox": None if bs.bbox is None else [round(v, 1) for v in bs.bbox],
                    "interpolated": bs.interpolated,
                }

                self.store.add_frame({
                    "frame": frame_idx, "t": round(t, 3),
                    "ts": started_wall + timedelta(seconds=t),
                    "players": players, "ball": ball,
                })
                processed += 1

                if writer or self.cfg.show:
                    vis = self._draw(frame, players, ball)
                    if writer:
                        writer.write(vis)
                    if self.cfg.show:
                        cv2.imshow("vbtrack", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            status = "stopped"
                            break

                if processed % 100 == 0:
                    log.info(
                        "frame %d  players=%d  ball=%s  %.1f fps",
                        frame_idx, len(players), "yes" if ball else "no", processed / (time.monotonic() - t0),
                    )
                if self.cfg.max_frames and processed >= self.cfg.max_frames:
                    break
        except KeyboardInterrupt:
            status = "stopped"
        finally:
            cap.release()
            if writer:
                writer.release()
            if self.cfg.show:
                cv2.destroyAllWindows()
            self.store.end_session(status, track_numbers=self.voter.summary() if self.reader else None)
            log.info("session %s %s: %d frames stored", sid, status, processed)
