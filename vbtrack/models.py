"""Fine-tuned volleyball models from the Hugging Face Hub.

  * grdesignbuild/volleyball-person-ball-detector  D-FINE-S, classes person + ball, run on tiles
  * grdesignbuild/volleyball-jersey-number-reader  ResNet-18, reads the number on a torso crop
  * Davidsv/volley-ref-ai                          YOLO11 ball detector + YOLO11-pose court keypoints

Weights are referenced either as local paths or as `hf://<owner>/<repo>/<file>` and cached by
huggingface_hub. The tiling / merge / decode logic is ported from the repos' predict.py and read_number.py.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision

from .hub import COURT_WEIGHTS, JERSEY_REPO, PERSON_BALL_REPO, REF_BALL_WEIGHTS, resolve_weights


def torch_device(device: str | None) -> str:
    """Map an Ultralytics-style device string ("0", "cpu", "mps", None) to a torch device."""
    if device is None:
        return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return f"cuda:{device}" if device.isdigit() else device


# ---------------------------------------------------------------- person + ball (D-FINE)

def tiles_for(w: int, h: int, tile: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles plus one full-frame pass; a single full frame when the image is small."""
    if max(w, h) <= tile * 1.15:
        return [(0, 0, w, h)]
    stride = int(tile * (1 - overlap))
    xs = list(range(0, max(1, w - tile) + 1, stride))
    ys = list(range(0, max(1, h - tile) + 1, stride))
    if xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    if ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    out = [(x, y, min(w, x + tile), min(h, y + tile)) for y in ys for x in xs]
    out.append((0, 0, w, h))
    return out


def _cover(c, k) -> float:
    """Fraction of box c covered by box k."""
    ix = max(0.0, min(c[2], k[2]) - max(c[0], k[0]))
    iy = max(0.0, min(c[3], k[3]) - max(c[1], k[1]))
    return ix * iy / max(1e-6, (c[2] - c[0]) * (c[3] - c[1]))


def merge_tiles(dets: list[tuple], nms_iou: float = 0.5, contain: float = 0.85) -> list[tuple]:
    """dets: (x1, y1, x2, y2, score, label, cut). Class-wise NMS over uncut boxes, drop boxes mostly inside a
    stronger one, then admit boxes cut by a tile edge only when nothing already covers them."""
    final = []
    for label in {d[5] for d in dets}:
        d = [x for x in dets if x[5] == label]
        uncut = sorted([x for x in d if not x[6]], key=lambda x: -x[4])
        cut = sorted([x for x in d if x[6]], key=lambda x: -x[4])
        kept: list[tuple] = []
        if uncut:
            keep = torchvision.ops.nms(torch.tensor([x[:4] for x in uncut], dtype=torch.float32),
                                       torch.tensor([x[4] for x in uncut], dtype=torch.float32), nms_iou)
            for k in keep.tolist():
                c = uncut[k]
                if not any(_cover(c, q) > contain for q in kept):
                    kept.append(c)
        for c in cut:
            if not any(_cover(c, q) > 0.4 for q in kept):
                kept.append(c)
        final.extend(x[:6] for x in kept)
    return final


class PersonBallDetector:
    """D-FINE-S fine-tuned on volleyball footage. Returns (persons, balls), each an (N, 5) array x1,y1,x2,y2,score.

    tile=0 (auto) runs frames up to 2000 px wide in one pass, as the model card recommends for 1080p, and tiles
    larger (4K) frames at 864 px. Every tile is a full model pass (~0.2 s each on Apple MPS), so tiling 1080p
    finds slightly smaller balls at a large speed cost."""

    def __init__(self, repo: str = PERSON_BALL_REPO, device: str | None = None,
                 tile: int = 0, threshold: float = 0.3, batch: int = 8):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.device = torch_device(device)
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForObjectDetection.from_pretrained(repo).to(self.device).eval()
        label2id = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.person_id, self.ball_id = label2id["person"], label2id["ball"]
        self.tile, self.threshold, self.batch = tile, threshold, batch

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        tile = self.tile or (max(w, h) if max(w, h) <= 2000 else 864)
        regions = tiles_for(w, h, tile)
        dets, m = [], 3
        for i in range(0, len(regions), self.batch):
            chunk = regions[i:i + self.batch]
            crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in chunk]
            inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            sizes = torch.tensor([(y2 - y1, x2 - x1) for x1, y1, x2, y2 in chunk])
            results = self.processor.post_process_object_detection(out, threshold=self.threshold, target_sizes=sizes)
            for r, (tx1, ty1, tx2, ty2) in zip(results, chunk):
                full = (tx1, ty1, tx2, ty2) == (0, 0, w, h)
                tw, th = tx2 - tx1, ty2 - ty1
                for s, lab, (bx1, by1, bx2, by2) in zip(r["scores"].tolist(), r["labels"].tolist(),
                                                        r["boxes"].tolist()):
                    # a box touching an interior tile edge is probably a cut object
                    cut = (not full) and ((bx1 < m and tx1 > 0) or (by1 < m and ty1 > 0)
                                          or (bx2 > tw - m and tx2 < w) or (by2 > th - m and ty2 < h))
                    dets.append((bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1, s, lab, cut))
        merged = merge_tiles(dets)

        def pick(label):
            rows = [d[:5] for d in merged if d[5] == label]
            return np.array(rows, dtype=np.float32).reshape(-1, 5)
        return pick(self.person_id), pick(self.ball_id)


# ---------------------------------------------------------------- jersey numbers

def torso_crop(frame_bgr: np.ndarray, box) -> np.ndarray:
    """Chest region the reader was trained on: 10-60 % of box height, widened 10 % per side, 224x224 RGB."""
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - 0.10 * w)), int(max(0, y1 + 0.10 * h))
    cx2, cy2 = int(min(W, x2 + 0.10 * w)), int(min(H, y1 + 0.60 * h))
    crop = frame_bgr[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)]
    return cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def decode_numbers(logits: torch.Tensor, min_readable: float = 0.5) -> list[tuple[str | None, float]]:
    """23 logits: tens digit 0-9 or blank (0-10), ones digit (11-20), unreadable/readable (21-22)."""
    tens, ones, read = logits[:, :11].softmax(-1), logits[:, 11:21].softmax(-1), logits[:, 21:23].softmax(-1)
    out = []
    for i in range(logits.shape[0]):
        if read[i, 1] < min_readable:
            out.append((None, float(read[i, 0])))
            continue
        t, o = int(tens[i].argmax()), int(ones[i].argmax())
        out.append((str(o) if t == 10 else f"{t}{o}", float(read[i, 1] * tens[i].max() * ones[i].max())))
    return out


class JerseyReader:
    def __init__(self, repo: str = JERSEY_REPO, device: str | None = None):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.device = torch_device(device)
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForImageClassification.from_pretrained(repo).to(self.device).eval()

    @torch.no_grad()
    def read(self, frame_bgr: np.ndarray, boxes) -> list[tuple[str | None, float]]:
        if not len(boxes):
            return []
        crops = [torso_crop(frame_bgr, b) for b in boxes]
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        return decode_numbers(self.model(**inputs).logits.float().cpu())


# ---------------------------------------------------------------- volley-ref-ai (YOLO)

class YoloBallDetector:
    """volley-ref-ai YOLO11s ball model. Returns an (N, 5) array x1,y1,x2,y2,score."""

    def __init__(self, weights: str = REF_BALL_WEIGHTS, device: str | None = None,
                 imgsz: int = 1280, conf: float = 0.3, ball_class: int = 0):
        from ultralytics import YOLO
        self.model = YOLO(resolve_weights(weights))
        self.device, self.imgsz, self.conf, self.ball_class = device, imgsz, conf, ball_class

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, classes=[self.ball_class],
                               device=self.device, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return np.zeros((0, 5), dtype=np.float32)
        return np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)


class CourtDetector:
    """volley-ref-ai YOLO11n-pose court model: 14 court keypoints per frame.

    Keypoint detection is noisy (pose mAP@50 ~0.29), so we take a per-keypoint median over several
    frames and use the convex hull of the result, expanded a little, as the court ROI (players serve and
    dig from behind the lines)."""

    def __init__(self, weights: str = COURT_WEIGHTS, device: str | None = None,
                 kpt_conf: float = 0.5, samples: int = 5, expand: float = 0.15):
        from ultralytics import YOLO
        self.model = YOLO(resolve_weights(weights))
        self.device, self.kpt_conf, self.samples, self.expand = device, kpt_conf, samples, expand
        self._history: list[np.ndarray] = []      # each (14, 3): x, y, conf

    @property
    def ready(self) -> bool:
        return len(self._history) >= self.samples

    def observe(self, frame_bgr: np.ndarray, feet: list[tuple[float, float]] | None = None,
                min_feet: int = 4) -> bool:
        """Add one frame's keypoints. With `feet` (people's foot points in this frame), the sample is only kept
        when its court polygon contains at least `min_feet` of them: off-domain views (wide arena shots,
        broadcast close-ups) otherwise give confident but wrong keypoints that would filter out every player."""
        r = self.model.predict(frame_bgr, device=self.device, verbose=False)[0]
        if r.keypoints is None or r.keypoints.data is None or not len(r.keypoints.data):
            return False
        best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        kp = r.keypoints.data[best].cpu().numpy()     # (14, 3)
        ok = kp[:, 2] >= self.kpt_conf
        if ok.sum() < 4:
            return False
        if feet is not None:
            hull = self._expanded_hull(kp[ok, :2])
            if sum(cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in feet) < min_feet:
                return False
        self._history.append(kp)
        return True

    def _expanded_hull(self, pts: np.ndarray) -> np.ndarray:
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        c = hull.mean(axis=0)
        return (c + (hull - c) * (1 + self.expand)).astype(np.float32)

    def keypoints(self) -> np.ndarray | None:
        """(14, 3) median x, y and fraction of observations where the point was confident; NaN when never seen."""
        if not self._history:
            return None
        h = np.stack(self._history)
        ok = h[:, :, 2] >= self.kpt_conf
        med = np.full((h.shape[1], 2), np.nan)
        for i in np.flatnonzero(ok.any(axis=0)):
            med[i] = np.median(h[ok[:, i], i, :2], axis=0)
        return np.hstack([med, ok.mean(axis=0)[:, None]])

    def roi(self) -> list[tuple[int, int]] | None:
        kp = self.keypoints()
        if kp is None:
            return None
        pts = kp[~np.isnan(kp[:, 0]), :2]
        if len(pts) < 4:
            return None
        return [(int(x), int(y)) for x, y in self._expanded_hull(pts)]
