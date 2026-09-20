"""Detectors that turn one transmitted view into frame-global detections (source pixels)."""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np

from dtos import DroneFlybyPredictRequestDto, OBJECT_CLASSES
from tracker import Detection, visible_fraction

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get('DRONE_MODEL', '/mnt/data/nordicai/drone-flyby/models/detector.pt')
CONF_THRESHOLD = float(os.environ.get('DRONE_CONF', '0.15'))
IMG_SIZE = int(os.environ.get('DRONE_IMGSZ', '960'))
# Scales fire at different object sizes: 1280 finds mine layers and medium hulls that 960 misses,
# 960 keeps spacecraft and jammers that 1280 loses. Listing several runs them all and merges per class.
IMG_SIZES = [int(v) for v in re.split(r'[,;:]', os.environ.get('DRONE_IMGSZ_MULTI', '')) if v] or [IMG_SIZE]
MERGE_IOU = float(os.environ.get('DRONE_MERGE_IOU', '0.6'))
DEVICE = os.environ.get('DRONE_DEVICE', '0')


def view_to_source(box: Sequence[float], region: Sequence[int]):
    rx1, ry1, rx2, ry2 = region
    sx, sy = (rx2 - rx1) / 960.0, (ry2 - ry1) / 540.0
    return (rx1 + box[0] * sx, ry1 + box[1] * sy, rx1 + box[2] * sx, ry1 + box[3] * sy)


def merge(found, iou_threshold: float):
    """Keep the most confident box of each cluster, per class, so two scales do not report one object twice."""
    kept = []
    for xyxy, cls, conf in sorted(found, key=lambda f: -f[2]):
        x1, y1, x2, y2 = xyxy
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        duplicate = False
        for kx, kc, _ in kept:
            if int(kc) != int(cls):
                continue
            ix = max(0.0, min(x2, kx[2]) - max(x1, kx[0]))
            iy = max(0.0, min(y2, kx[3]) - max(y1, kx[1]))
            overlap = ix * iy
            union = area + max(kx[2] - kx[0], 0) * max(kx[3] - kx[1], 0) - overlap
            if union > 0 and overlap / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((xyxy, cls, conf))
    return kept


class YoloDetector:
    def __init__(self, path: str = MODEL_PATH):
        from ultralytics import YOLO

        self.model = YOLO(path)
        real = os.path.realpath(path)
        digest = hashlib.md5(Path(real).read_bytes()).hexdigest()[:12]
        args = getattr(self.model, 'ckpt', {}).get('train_args', {})
        logger.warning('DETECTOR %s md5=%s base=%s run=%s epoch=%s', real, digest,
                       str(args.get('model', '?')).split('/')[-1], args.get('name', '?'),
                       getattr(self.model, 'ckpt', {}).get('epoch', -2) + 1)
        self.warm_up()

    def warm_up(self) -> None:
        blank = np.zeros((540, 960, 3), np.uint8)
        for size in IMG_SIZES:
            for _ in range(3):
                self.model.predict(blank, imgsz=size, device=DEVICE, conf=CONF_THRESHOLD, verbose=False)

    def __call__(self, image: np.ndarray, request: DroneFlybyPredictRequestDto) -> List[Detection]:
        found = []
        for size in IMG_SIZES:
            boxes = self.model.predict(image, imgsz=size, device=DEVICE, conf=CONF_THRESHOLD, verbose=False)[0].boxes
            found += list(zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()))

        region = request.view.source_region_xyxy
        level = request.view.resolution_level
        out = []
        for xyxy, cls, conf in merge(found, MERGE_IOU) if len(IMG_SIZES) > 1 else found:
            if int(cls) >= len(OBJECT_CLASSES):
                continue
            out.append(Detection(view_to_source(xyxy, region), int(cls), float(conf), level))
        return out


class OracleDetector:
    """Ground truth restricted to what the current view could plausibly show. Local testing only."""

    MIN_SIDE = {0: 7, 1: 6, 2: 5}

    def __init__(self, scene: str = 'helsinki', noise_px: float = 2.0, seed: int = 0):
        from utils import load_annotations

        self.load = lambda frame: load_annotations(frame, scene)
        self.rng = np.random.default_rng(seed)
        self.noise = noise_px

    def __call__(self, image: np.ndarray, request: DroneFlybyPredictRequestDto) -> List[Detection]:
        region = request.view.source_region_xyxy
        level = request.view.resolution_level
        factor = (4, 2, 1)[level]
        out = []
        for ann in self.load(request.frame):
            box = np.array(ann['bbox'], float)
            if visible_fraction(box, region) < 0.5:
                continue
            if min(box[2] - box[0], box[3] - box[1]) / factor < self.MIN_SIDE[level]:
                continue
            box = box + self.rng.normal(0, self.noise * factor, 4)
            out.append(Detection(tuple(box), OBJECT_CLASSES.index(ann['object_id']), 0.9, level))
        return out


def build_detector():
    if os.environ.get('DRONE_ORACLE') == '1':
        return OracleDetector()
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f'No detector weights at {MODEL_PATH}; set DRONE_MODEL or DRONE_ORACLE=1')
    return YoloDetector()
