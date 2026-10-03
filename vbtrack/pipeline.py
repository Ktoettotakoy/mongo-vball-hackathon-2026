"""Main loop: read frames -> YOLO players (+ball) -> ball tracker -> MongoDB."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

import cv2
import numpy as np
from ultralytics import YOLO

from .ball import BallCandidate, BallTracker
from .storage import MongoStore, utcnow

log = logging.getLogger("vbtrack")

COCO_PERSON = 0
COCO_SPORTS_BALL = 32


@dataclass
class Config:
    source: str
    player_weights: str = "yolo11n.pt"
    ball_weights: str | None = None          # custom volleyball model; None = use COCO "sports ball"
    ball_class: int = 0                      # class id of the ball in the custom model
    imgsz: int = 1280                        # high res helps a lot with the small ball
    ball_imgsz: int = 1280
    player_conf: float = 0.35
    ball_conf: float = 0.15
    tracker: str = "bytetrack.yaml"          # or botsort.yaml
    stride: int = 1                          # process every Nth frame
    roi: list[tuple[int, int]] | None = None # court polygon; players outside are dropped
    device: str | None = None                # "cpu", "0", "mps"...
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
    args.device = device or "cpu"   # only used by BoT-SORT's optional ReID encoder
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
        self.player_model = YOLO(cfg.player_weights)
        self.ball_model = YOLO(cfg.ball_weights) if cfg.ball_weights else None
        self.player_tracker = make_tracker(cfg.tracker, cfg.device)
        self.ball = BallTracker(min_conf=cfg.ball_conf)
        self.roi = np.array(cfg.roi, dtype=np.int32) if cfg.roi else None
        self.live = is_live(cfg.source)

    # ---------- detection ----------
    def _detect(self, frame: np.ndarray):
        # One detector pass for people (+ ball when no custom ball model). We run the
        # tracker ourselves on the person boxes only: model.track() would push the ball
        # through ByteTrack too, which silently drops low-confidence ball detections.
        classes = [COCO_PERSON] if self.ball_model else [COCO_PERSON, COCO_SPORTS_BALL]
        res = self.player_model.predict(
            frame, classes=classes, imgsz=self.cfg.imgsz,
            conf=min(self.cfg.player_conf, self.cfg.ball_conf),
            device=self.cfg.device, verbose=False,
        )[0]

        players, ball_cands = [], []
        boxes = res.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)

        person = boxes[cls == COCO_PERSON]
        tracks = self.player_tracker.update(person, frame)  # called on empty frames too, so lost tracks age out
        # rows: x1, y1, x2, y2, track_id, score, cls, det_idx
        for x1, y1, x2, y2, tid, score, *_ in tracks:
            if score < self.cfg.player_conf:
                continue
            foot = (float(x1 + x2) / 2, float(y2))
            if self.roi is not None and cv2.pointPolygonTest(self.roi, foot, False) < 0:
                continue
            players.append({
                "track_id": int(tid), "conf": round(float(score), 3),
                "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                "foot": {"x": round(foot[0], 1), "y": round(foot[1], 1)},
            })

        if not self.ball_model:
            for (x1, y1, x2, y2), c in zip(boxes.xyxy[cls == COCO_SPORTS_BALL], boxes.conf[cls == COCO_SPORTS_BALL]):
                ball_cands.append(BallCandidate(float(x1 + x2) / 2, float(y1 + y2) / 2,
                                                float(x2 - x1), float(y2 - y1), float(c)))

        if self.ball_model:
            br = self.ball_model.predict(frame, imgsz=self.cfg.ball_imgsz, conf=self.cfg.ball_conf,
                                         classes=[self.cfg.ball_class], device=self.cfg.device,
                                         verbose=False)[0]
            if br.boxes is not None:
                for box, c in zip(br.boxes.xyxy.cpu().numpy(), br.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    ball_cands.append(BallCandidate((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(c)))
        return players, ball_cands

    # ---------- drawing ----------
    def _draw(self, frame, players, ball):
        if self.roi is not None:
            cv2.polylines(frame, [self.roi], True, (255, 255, 0), 2)
        for p in players:
            x1, y1, x2, y2 = map(int, p["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"#{p['track_id']}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        if ball:
            color = (0, 165, 255) if ball["interpolated"] else (0, 0, 255)
            cv2.circle(frame, (int(ball["x"]), int(ball["y"])), 8, color, 2)
        return frame

    # ---------- main loop ----------
    def run(self) -> None:
        cap = open_capture(self.cfg.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1 or fps > 240:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sid = self.store.start_session(
            self.cfg.source, fps, w, h, self.live,
            meta={"player_weights": self.cfg.player_weights, "ball_weights": self.cfg.ball_weights,
                  "tracker": self.cfg.tracker, "stride": self.cfg.stride, "roi": self.cfg.roi,
                  **self.cfg.extra_meta},
        )
        log.info("session %s started (%s, %.1f fps, %dx%d)", sid, "live" if self.live else "file", fps, w, h)

        writer = None
        if self.cfg.save_video:
            writer = cv2.VideoWriter(self.cfg.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / self.cfg.stride, (w, h))

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
                players, cands = self._detect(frame)
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
                    log.info("frame %d  players=%d  ball=%s  %.1f fps", frame_idx, len(players),
                             "yes" if ball else "no", processed / (time.monotonic() - t0))
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
            self.store.end_session(status)
            log.info("session %s %s: %d frames stored", sid, status, processed)
