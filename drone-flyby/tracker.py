"""Frame-global object memory.

Every object sits still on the ground, and the ground moves through the frame by an almost exact
per-frame homography (fitted on Helsinki: 0.7 px median centre error). So once an object has been
seen it can be reported on every later frame by propagating its box, whether or not the camera is
looking at it. Detections from the current view correct position and class.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES

HELSINKI_H = np.array([
    [1.00814, -0.00100, -13.63787],
    [0.00067, 1.01348, 50.72964],
    [0.0, 0.0, 1.0],
])

LEVEL_WEIGHT = {0: 0.35, 1: 0.65, 2: 0.85}
STALENESS = float(os.environ.get('DRONE_STALENESS', '0.97'))
MATCH_IOU = 0.2
MATCH_CENTRE_FRACTION = 0.6


@dataclass
class Detection:
    box: Tuple[float, float, float, float]
    class_id: int
    confidence: float
    level: int


@dataclass
class Track:
    box: np.ndarray
    class_scores: np.ndarray
    hits: int = 1
    misses: int = 0
    best_level: int = 0
    last_seen: int = 0
    last_obs_centre: Optional[np.ndarray] = None
    last_obs_frame: int = 0

    @property
    def class_id(self) -> int:
        return int(np.argmax(self.class_scores))

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


def warp_box(box: np.ndarray, H: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ H.T
    pts = pts[:, :2] / pts[:, 2:3]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def warp_point(p: np.ndarray, H: np.ndarray) -> np.ndarray:
    q = H @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def visible_fraction(box: Sequence[float], region: Sequence[float]) -> float:
    ix = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 0.0


@dataclass
class WorldModel:
    H: np.ndarray = field(default_factory=lambda: HELSINKI_H.copy())
    tracks: List[Track] = field(default_factory=list)
    frame: Optional[int] = None
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))
    drift_samples: int = 0
    use_drift: bool = False

    def step_matrix(self) -> np.ndarray:
        if not self.use_drift:
            return self.H
        T = np.array([[1, 0, self.drift[0]], [0, 1, self.drift[1]], [0, 0, 1.0]])
        return T @ self.H

    def advance(self, frame: int) -> None:
        if self.frame is None:
            self.frame = frame
            return
        steps = frame - self.frame
        if steps <= 0:
            return
        M = np.linalg.matrix_power(self.step_matrix(), steps)
        for t in self.tracks:
            t.box = warp_box(t.box, M)
        self.frame = frame
        self.tracks = [t for t in self.tracks if visible_fraction(t.box, (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)) > 0.25]

    def update(self, detections: List[Detection], region: Sequence[int], level: int) -> None:
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                score = iou(t.box, d.box)
                size = max(t.box[2] - t.box[0], t.box[3] - t.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1])
                dc = np.hypot((t.box[0] + t.box[2] - d.box[0] - d.box[2]) / 2, (t.box[1] + t.box[3] - d.box[1] - d.box[3]) / 2)
                if score >= MATCH_IOU or dc < MATCH_CENTRE_FRACTION * size:
                    pairs.append((score - dc / (size + 1e-6) * 0.1, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._merge(self.tracks[ti], detections[di])

        for ti, t in enumerate(self.tracks):
            if ti in used_t:
                continue
            if visible_fraction(t.box, region) > 0.8 and self._should_be_visible(t, level):
                t.misses += 1

        for di, d in enumerate(detections):
            if di in used_d:
                continue
            scores = np.zeros(len(OBJECT_CLASSES))
            scores[d.class_id] = d.confidence * LEVEL_WEIGHT[d.level]
            c = np.array([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
            self.tracks.append(Track(box=np.array(d.box, float), class_scores=scores, best_level=d.level,
                                     last_seen=self.frame, last_obs_centre=c, last_obs_frame=self.frame))

        self.tracks = [t for t in self.tracks if not (t.misses >= 2 and t.misses > t.hits)]
        self._dedupe()

    def _should_be_visible(self, t: Track, level: int) -> bool:
        side = min(t.box[2] - t.box[0], t.box[3] - t.box[1]) / (4 if level == 0 else 2 if level == 1 else 1)
        return side >= 10 and level >= t.best_level

    def _merge(self, t: Track, d: Detection) -> None:
        c_obs = np.array([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
        if self.use_drift and t.last_obs_centre is not None and d.level >= 1 and self.frame > t.last_obs_frame:
            n = self.frame - t.last_obs_frame
            predicted = t.last_obs_centre
            M = self.step_matrix()
            for _ in range(n):
                predicted = warp_point(predicted, M)
            residual = (c_obs - predicted) / n
            if np.all(np.abs(residual) < 25):
                self.drift_samples += 1
                rate = max(0.05, 1.0 / self.drift_samples)
                self.drift = self.drift + rate * residual
        w = LEVEL_WEIGHT[d.level] if d.level >= t.best_level else LEVEL_WEIGHT[d.level] * 0.5
        t.box = (1 - w) * t.box + w * np.array(d.box, float)
        t.class_scores[d.class_id] += d.confidence * LEVEL_WEIGHT[d.level]
        t.hits += 1
        t.misses = 0
        t.best_level = max(t.best_level, d.level)
        t.last_seen = self.frame
        t.last_obs_centre = c_obs
        t.last_obs_frame = self.frame

    def _dedupe(self) -> None:
        keep: List[Track] = []
        for t in sorted(self.tracks, key=lambda t: -t.class_scores.sum()):
            dup = next((k for k in keep if iou(k.box, t.box) > 0.5), None)
            if dup is None:
                keep.append(t)
            else:
                dup.class_scores += t.class_scores
                dup.hits += t.hits
        self.tracks = keep

    def report(self) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
        out = []
        for t in self.tracks:
            total = t.class_scores.sum()
            if total <= 0:
                continue
            purity = t.class_scores.max() / total
            evidence = 1 - np.exp(-1.5 * total)
            staleness = STALENESS ** max(0, self.frame - t.last_seen)
            confidence = float(np.clip(purity * evidence * staleness, 0.01, 1.0))
            out.append((t.class_id, tuple(t.box), confidence))
        return out
