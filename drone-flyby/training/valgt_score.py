"""COCO mAP@0.50 of recorded (or replayed) validation responses against the validation ground truth."""

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtos import OBJECT_CLASSES  # noqa: E402

OUT = Path('/mnt/data/nordicai/drone-flyby/valgt')
W, H = 3840, 2160


def load_gt():
    return {int(k): v for k, v in json.loads((OUT / 'gt.json').read_text()).items()}


def score(predictions, gt=None, per_class=False):
    """predictions: {frame: [{'object_id', 'bbox' (normalized xyxy), 'confidence'}]}"""
    from faster_coco_eval import COCO, COCOeval_faster
    gt = gt or load_gt()
    frames = sorted(gt)
    cat = {n: i + 1 for i, n in enumerate(OBJECT_CLASSES)}
    present = sorted({g['object_id'] for f in frames for g in gt[f] if not g['ignore']}, key=OBJECT_CLASSES.index)
    anns, aid = [], 1
    for f in frames:
        for g in gt[f]:
            x1, y1, x2, y2 = g['bbox']
            anns.append({'id': aid, 'image_id': f, 'category_id': cat[g['object_id']], 'bbox': [x1, y1, x2 - x1, y2 - y1],
                         'area': (x2 - x1) * (y2 - y1), 'iscrowd': int(g['ignore']), 'ignore': int(g['ignore'])})
            aid += 1
    coco_gt = COCO({'images': [{'id': f, 'width': W, 'height': H} for f in frames],
                    'categories': [{'id': cat[n], 'name': n} for n in OBJECT_CLASSES], 'annotations': anns})
    dets = []
    for f, items in predictions.items():
        if f not in gt:
            continue
        for a in items:
            x1, y1, x2, y2 = a['bbox'][0] * W, a['bbox'][1] * H, a['bbox'][2] * W, a['bbox'][3] * H
            if x2 > x1 and y2 > y1:
                dets.append({'image_id': f, 'category_id': cat[a['object_id']], 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(a['confidence'])})
    if not dets:
        return 0.0
    ev = COCOeval_faster(coco_gt, coco_gt.loadRes(dets), 'bbox')
    ev.params.imgIds = frames
    ev.params.catIds = [cat[n] for n in present]
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate(); ev.accumulate()
    prec = ev.eval['precision']
    aps = {}
    for i, n in enumerate(present):
        p = prec[0, :, i, 0, -1]; p = p[p > -1]
        aps[n] = float(np.mean(p)) if p.size else 0.0
    m = float(np.mean(list(aps.values())))
    return (m, aps) if per_class else m


def recorded_predictions(run_dir):
    preds = {}
    for f in sorted(glob.glob(str(Path(run_dir) / '*.json'))):
        x = json.load(open(f))
        preds[x['frame']] = x['response']['annotations']
    return preds


if __name__ == '__main__':
    official = {'validation_run2_m_e1_motion_score0.452': 0.452, 'validation_run3_staleness1.0_score0.423': 0.423,
                'validation_run4_staleness0.93_score0.412': 0.412, 'validation_run5_repeat0.97_det_score0.403': 0.403,
                'validation_run7_framelocked_score0.395': 0.395, 'validation_run10_long_v2_e1_score0.379_camerarace': 0.379,
                'validation_run13_long_v2_e2_score0.358': 0.358, 'validation_run14_long_v2_e3_score0.323': 0.323,
                'validation_run15_long_v1_e2_score0.403': 0.403}
    rows = []
    gt = load_gt()
    for d in sorted(glob.glob('/mnt/data/nordicai/drone-flyby/recordings/validation_run*')):
        if not Path(d).is_dir() or 'oracle' in d:
            continue
        s = score(recorded_predictions(d), gt)
        rows.append((Path(d).name, official.get(Path(d).name), s))
        print(f'{Path(d).name:52s} official {official.get(Path(d).name)!s:6s} local {s:.3f}')
    pairs = [(o, s) for _, o, s in rows if o is not None]
    o, s = np.array(pairs).T
    from scipy.stats import spearmanr, pearsonr
    print('pearson %.2f  spearman %.2f  (n=%d)' % (pearsonr(o, s)[0], spearmanr(o, s)[0], len(pairs)))
