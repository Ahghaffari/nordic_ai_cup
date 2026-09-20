"""Online estimate of the per-frame ground homography.

The camera geometry and speed are constant within a flight but differ slightly between flights, and a
few pixels per frame of error compounds over the ~30 frames an object stays in view. Consecutive views
overlap, so SIFT matches between them (lifted into source pixels) give thousands of ground
correspondences per frame; a RANSAC fit over a rolling window of those pairs is the flight's homography.
"""

import queue
import os
import threading
from collections import deque
from typing import Optional, Sequence

import cv2
import numpy as np

H_PRIOR = np.array([
    [1.0068889, -0.0014277, -12.9629119],
    [0.0000176, 1.0133931, 53.1187622],
    [0.0000001, -0.0000008, 1.0],
])

# The window fit is the per-frame cost of the whole endpoint: with 12 pairs of 1500 correspondences the RANSAC
# takes about 237 ms, against 57 ms for the detector, and frames arriving every 333 ms are skipped when we run
# over. Tunable so the trade against homography accuracy can be measured on the local scorer.
MAX_POINTS_PER_PAIR = int(os.environ.get('DRONE_MOTION_POINTS', '1500'))
WINDOW_PAIRS = int(os.environ.get('DRONE_MOTION_WINDOW', '12'))
RANSAC_ITERS = int(os.environ.get('DRONE_MOTION_ITERS', '3000'))
MIN_INLIERS = 400


class MotionEstimator:
    def __init__(self, prior: np.ndarray = H_PRIOR, n_features: int = 1000):
        self.H = prior.copy()
        self.inliers = 0
        self.pairs = 0
        self._sift = cv2.SIFT_create(n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._previous = None
        self._window = deque(maxlen=WINDOW_PAIRS)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0)

    def current(self) -> np.ndarray:
        with self._lock:
            return self.H.copy()

    def process(self, frame: int, image: np.ndarray, region: Sequence[int]) -> None:
        grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(grey, None)
        if descriptors is None or len(keypoints) < 20:
            self._previous = None
            return
        x1, y1, x2, y2 = region
        sx, sy = (x2 - x1) / grey.shape[1], (y2 - y1) / grey.shape[0]
        points = np.array([[x1 + k.pt[0] * sx, y1 + k.pt[1] * sy] for k in keypoints], np.float32)
        current = (frame, points, descriptors, max(sx, sy))
        previous, self._previous = self._previous, current
        if previous is None or frame != previous[0] + 1:
            return

        matches = [m for m, n in self._matcher.knnMatch(previous[2], descriptors, k=2) if m.distance < 0.75 * n.distance]
        if len(matches) < 30:
            return
        src = previous[1][[m.queryIdx for m in matches]]
        dst = points[[m.trainIdx for m in matches]]
        threshold = 2.0 * max(previous[3], current[3])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
        if mask is None or mask.sum() < 30:
            return
        keep = mask.ravel() > 0
        src, dst = src[keep], dst[keep]
        if len(src) > MAX_POINTS_PER_PAIR:
            idx = self._rng.choice(len(src), MAX_POINTS_PER_PAIR, replace=False)
            src, dst = src[idx], dst[idx]
        self._window.append((src, dst))
        self._refit()

    def _refit(self) -> None:
        src = np.vstack([s for s, _ in self._window])
        dst = np.vstack([d for _, d in self._window])
        if len(src) < MIN_INLIERS:
            return
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=RANSAC_ITERS)
        if H is None or mask.sum() < MIN_INLIERS:
            return
        H = H / H[2, 2]
        with self._lock:
            self.H = H
            self.inliers = int(mask.sum())
            self.pairs += 1


class _Worker:
    def __init__(self):
        self.queue: 'queue.Queue' = queue.Queue(maxsize=8)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            estimator, frame, image, region = self.queue.get()
            try:
                estimator.process(frame, image, region)
            except Exception:
                pass

    def submit(self, estimator: MotionEstimator, frame: int, image: np.ndarray, region: Sequence[int]) -> None:
        try:
            self.queue.put_nowait((estimator, frame, image, region))
        except queue.Full:
            pass


_worker: Optional[_Worker] = None
cv2.SIFT_create(500).detectAndCompute(np.random.default_rng(0).integers(0, 255, (540, 960), dtype=np.uint8), None)


def submit(estimator: MotionEstimator, frame: int, image: np.ndarray, region: Sequence[int]) -> None:
    global _worker
    if _worker is None:
        _worker = _Worker()
    _worker.submit(estimator, frame, image, region)
