"""Link multi-model, multi-view detections of the recorded validation flight into object tracks."""

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtos import OBJECT_CLASSES  # noqa: E402

OUT = Path('/mnt/data/nordicai/drone-flyby/valgt')
MIN_CONF = 0.08
W, H = 3840, 2160


def warp(box, M):
    x1, y1, x2, y2 = box
    p = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ np.array(M).T
    p = p[:, :2] / p[:, 2:3]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    views = json.loads((OUT / 'views_dets.json').read_text())
    Hs = {int(k): np.array(v) for k, v in json.loads((OUT / 'homographies.json').read_text()).items()}
    dets = collections.defaultdict(list)
    for v in views:
        for model, ds in v['dets'].items():
            for x1, y1, x2, y2, c, p in ds:
                if p >= MIN_CONF:
                    dets[v['frame']].append({'box': np.array([x1, y1, x2, y2]), 'cls': c, 'conf': p, 'model': model,
                                             'level': v['level'], 'png': v['png'], 'region': v['region']})
    frames = sorted({v['frame'] for v in views})
    tracks = []
    for f in frames:
        if f - 1 in Hs:
            for t in tracks:
                if t['alive']:
                    t['box'] = warp(t['box'], Hs[f - 1])
                    if t['box'][1] > H or t['box'][3] < -200:
                        t['alive'] = False
        live = [t for t in tracks if t['alive']]
        groups = collections.defaultdict(list)
        unmatched = []
        for d in dets[f]:
            best, score = None, 0.0
            for i, t in enumerate(live):
                size = max(t['box'][2] - t['box'][0], t['box'][3] - t['box'][1], d['box'][2] - d['box'][0], d['box'][3] - d['box'][1])
                dc = np.hypot(*(((t['box'][:2] + t['box'][2:]) - (d['box'][:2] + d['box'][2:])) / 2))
                s = iou(t['box'], d['box'])
                if (s > 0.25 or dc < 0.45 * size) and s + 1 - dc / size > score:
                    best, score = i, s + 1 - dc / size
            if best is None:
                unmatched.append(d)
            else:
                groups[best].append(d)
        for i, ds in groups.items():
            t = live[i]
            wts = np.array([d['conf'] for d in ds])
            obs = (np.array([d['box'] for d in ds]) * wts[:, None]).sum(0) / wts.sum()
            t['box'] = 0.3 * t['box'] + 0.7 * obs
            t['obs'].append({'frame': f, 'box': obs.tolist(), 'dets': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for d in ds]})
        for d in sorted(unmatched, key=lambda d: -d['conf']):
            merged = next((t for t in tracks if t['alive'] and t['obs'] and t['obs'][-1]['frame'] == f and iou(t['box'], d['box']) > 0.25), None)
            if merged:
                merged['obs'][-1]['dets'].append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()})
                continue
            tracks.append({'id': len(tracks), 'box': d['box'].copy(), 'alive': True,
                           'obs': [{'frame': f, 'box': d['box'].tolist(), 'dets': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}]}]})

    summary = []
    for t in tracks:
        alld = [d for o in t['obs'] for d in o['dets']]
        votes = collections.Counter()
        for d in alld:
            votes[d['cls']] += d['conf']
        models = {d['model'] for d in alld if d['conf'] >= 0.3}
        best = max(alld, key=lambda d: d['conf'])
        summary.append({'id': t['id'], 'n_frames': len(t['obs']), 'n_dets': len(alld), 'first': t['obs'][0]['frame'], 'last': t['obs'][-1]['frame'],
                        'max_conf': best['conf'], 'models_conf30': sorted(models), 'class': OBJECT_CLASSES[votes.most_common(1)[0][0]],
                        'votes': {OBJECT_CLASSES[k]: round(v, 2) for k, v in votes.most_common(3)}, 'best_det': best, 'obs': t['obs']})
    (OUT / 'tracks.json').write_text(json.dumps(summary))
    strong = [s for s in summary if s['max_conf'] >= 0.5 and len(s['models_conf30']) >= 2 and s['n_frames'] >= 3]
    weak = [s for s in summary if s not in strong and s['max_conf'] >= 0.25 and s['n_frames'] >= 2]
    print('tracks', len(summary), '| strong', len(strong), '| review', len(weak), '| noise', len(summary) - len(strong) - len(weak))
    print('strong by class', dict(collections.Counter(s['class'] for s in strong)))
    print('review by class', dict(collections.Counter(s['class'] for s in weak)))


if __name__ == '__main__':
    main()
