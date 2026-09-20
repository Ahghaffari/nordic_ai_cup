"""Replay a recorded validation run through the pipeline offline with any detector/settings, then score it.

The camera route is fixed by the recording (the frame-locked policy produces the same route for every
detector), so replaying its requests reproduces an attempt without the website.
"""

import argparse
import base64
import glob
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))

RUN = '/mnt/data/nordicai/drone-flyby/recordings/validation_run11_long_v2_e1_camerafix'


def replay(model_path, run_dir=RUN, env=None):
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    import tracker, motion, detector as det_module
    importlib.reload(tracker)
    importlib.reload(det_module)
    from dtos import OBJECT_CLASSES, DroneFlybyPredictRequestDto
    from utils import clip_bbox_to_frame, source_bbox_to_global, decode_image
    detector = det_module.YoloDetector.__new__(det_module.YoloDetector)
    from ultralytics import YOLO
    detector.model = YOLO(model_path)
    world, mot = tracker.WorldModel(), motion.MotionEstimator()
    preds = {}
    for f in sorted(glob.glob(str(Path(run_dir) / '*.json'))):
        meta = json.load(open(f))
        meta['view']['image'] = base64.b64encode(open(f[:-5] + '.png', 'rb').read()).decode()
        meta.pop('response', None)
        req = DroneFlybyPredictRequestDto.model_validate(meta)
        image = decode_image(req.view.image)
        world.H = mot.current()
        world.advance(req.frame)
        world.update(detector(image, req), req.view.source_region_xyxy, req.view.resolution_level)
        out = []
        for cls, box, conf in sorted(world.report(), key=lambda r: -r[2])[:500]:
            b = clip_bbox_to_frame(source_bbox_to_global(box, 3840, 2160))
            if b is not None:
                out.append({'object_id': OBJECT_CLASSES[cls], 'bbox': list(b), 'confidence': conf})
        preds[req.frame] = out
        mot.process(req.frame, image, req.view.source_region_xyxy)
    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--env', default='')
    args = parser.parse_args()
    env = dict(kv.split('=') for kv in args.env.split(',') if kv)
    from valgt_score import score
    for m in args.models:
        path = m if m.startswith('/') else f'/mnt/data/nordicai/drone-flyby/models/{m}'
        s, aps = score(replay(path, args.run, env), per_class=True)
        print(f'{Path(path).name:28s} {args.env or "defaults":24s} local mAP50 {s:.3f} | ' + ' '.join(f'{k[:6]}={v:.2f}' for k, v in aps.items()), flush=True)
