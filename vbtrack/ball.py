"""Single-ball tracker: picks one ball per frame and smooths it with a Kalman filter.

A volleyball is small, fast and motion-blurred, so the detector misses it or
fires on heads/lights now and then. This tracker:
  * gates candidates by distance to the predicted position (rejects false hits),
  * predicts through short gaps (marks those points as interpolated),
  * drops the track after `max_missing` frames and re-acquires on the best detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallCandidate:
    x: float          # centre x (px)
    y: float          # centre y (px)
    w: float
    h: float
    conf: float


@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float
    conf: float | None     # detector confidence, None when predicted
    bbox: list[float] | None
    interpolated: bool


class KalmanCV:
    """Constant-velocity Kalman filter over (x, y, vx, vy)."""

    def __init__(self, x: float, y: float, process_noise: float = 3.0, meas_noise: float = 10.0):
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([meas_noise**2, meas_noise**2, 1e4, 1e4])
        self.q = process_noise
        self.R = np.eye(2) * meas_noise**2
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def predict(self, dt: float = 1.0) -> np.ndarray:
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        # white-acceleration process noise
        G = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])
        Q = (G @ G.T) * self.q**2
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q
        return self.s

    def update(self, x: float, y: float) -> np.ndarray:
        z = np.array([x, y])
        innov = z - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ innov
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.s


class BallTracker:
    def __init__(self, gate_px: float = 150.0, max_missing: int = 10, min_conf: float = 0.15):
        self.gate_px = gate_px
        self.max_missing = max_missing
        self.min_conf = min_conf
        self.kf: KalmanCV | None = None
        self.missing = 0

    def reset(self) -> None:
        self.kf = None
        self.missing = 0

    def step(self, candidates: list[BallCandidate]) -> BallState | None:
        cands = [c for c in candidates if c.conf >= self.min_conf]

        if self.kf is None:
            if not cands:
                return None
            best = max(cands, key=lambda c: c.conf)
            self.kf = KalmanCV(best.x, best.y)
            self.missing = 0
            return self._state(best, interpolated=False)

        px, py = self.kf.predict()[:2]
        # gate grows while we're coasting, since uncertainty grows too
        gate = self.gate_px * (1 + 0.5 * self.missing)
        in_gate = [c for c in cands if np.hypot(c.x - px, c.y - py) <= gate]

        if in_gate:
            # trade off closeness to prediction vs detector confidence
            best = min(in_gate, key=lambda c: np.hypot(c.x - px, c.y - py) / gate - c.conf)
            self.kf.update(best.x, best.y)
            self.missing = 0
            return self._state(best, interpolated=False)

        self.missing += 1
        if self.missing > self.max_missing:
            self.reset()
            # try to re-acquire immediately on the strongest detection
            return self.step(candidates) if cands else None
        return self._state(None, interpolated=True)

    def _state(self, c: BallCandidate | None, interpolated: bool) -> BallState:
        x, y, vx, vy = (float(v) for v in self.kf.s)
        bbox = [c.x - c.w / 2, c.y - c.h / 2, c.x + c.w / 2, c.y + c.h / 2] if c else None
        return BallState(x=x, y=y, vx=vx, vy=vy, conf=c.conf if c else None,
                         bbox=bbox, interpolated=interpolated)
