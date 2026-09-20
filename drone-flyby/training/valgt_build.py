"""Turn verified validation tracks into per-frame ground truth (accepted boxes + ignore regions)."""

import json
from pathlib import Path

import numpy as np

OUT = Path('/mnt/data/nordicai/drone-flyby/valgt')
W, H = 3840, 2160

ACCEPT = {
    174: 'hangar', 381: 'hangar', 281: 'helicopter', 129: 'helicopter', 371: 'helicopter',
    167: 'jet_plane', 173: 'jet_plane', 390: 'jet_plane', 0: 'large_launcher', 966: 'large_launcher',
    1136: 'large_tower', 118: 'large_tower', 328: 'large_tower', 332: 'medium_launcher', 194: 'medium_plane',
    974: 'mine_roller', 166: 'mine_roller', 273: 'mine_roller', 378: 'small_plane', 379: 'small_plane',
    384: 'small_plane', 619: 'small_tower', 202: 'small_tower', 1333: 'small_tower', 201: 'spacecraft',
    67: 'spacecraft', 87: 'tank', 819: 'tank', 42: 'tank', 465: 'tank', 1345: 'jammer',
    37: 'large_launcher', 172: 'large_launcher', 1350: 'jammer', 528: 'medium_plane', 223: 'spacecraft',
    203: 'spacecraft', 1: 'tank', 226: 'tank', 975: 'small_plane',
}
IGNORE = [266, 1086, 620, 1131, 1178, 1317, 1077, 251, 1286]


def warp(box, M):
    x1, y1, x2, y2 = box
    p = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ M.T
    p = p[:, :2] / p[:, 2:3]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    tracks = {t['id']: t for t in json.loads((OUT / 'tracks.json').read_text())}
    Hs = {int(k): np.array(v) for k, v in json.loads((OUT / 'homographies.json').read_text()).items()}
    frames = list(range(1, 250))

    def boxes_over_flight(track):
        observed = {}
        for o in track['obs']:
            dets = [d for d in o['dets'] if d['conf'] >= 0.15] or o['dets']
            per_model = {}
            for d in dets:
                per_model.setdefault(d['model'], []).append(d['box'])
            observed[o['frame']] = np.mean([np.median(np.array(b), axis=0) for b in per_model.values()], axis=0)
        out = {}
        obs_frames = sorted(observed)
        for f in frames:
            nearest = sorted(obs_frames, key=lambda g: abs(g - f))[:3]
            cands = []
            for g in nearest:
                box = observed[g].copy()
                if g < f:
                    for k in range(g, f):
                        box = warp(box, Hs[k])
                else:
                    for k in range(g - 1, f - 1, -1):
                        box = warp(box, np.linalg.inv(Hs[k]))
                cands.append(box)
            box = np.mean(cands, axis=0)
            area = (box[2] - box[0]) * (box[3] - box[1])
            clipped = np.array([max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])])
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            if (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) < 0.3 * area:
                continue
            out[f] = clipped.tolist()
        return out

    gt = {f: [] for f in frames}
    for tid, cls in ACCEPT.items():
        for f, box in boxes_over_flight(tracks[tid]).items():
            dup = next((g for g in gt[f] if g['object_id'] == cls and not g['ignore'] and iou(g['bbox'], box) > 0.3), None)
            if dup is None:
                gt[f].append({'object_id': cls, 'bbox': box, 'ignore': False, 'track': tid})
    for tid in IGNORE:
        cls = tracks[tid]['class']
        for f, box in boxes_over_flight(tracks[tid]).items():
            gt[f].append({'object_id': cls, 'bbox': box, 'ignore': True, 'track': tid})
    (OUT / 'gt.json').write_text(json.dumps({str(k): v for k, v in gt.items()}))
    per_frame = [sum(1 for g in gt[f] if not g['ignore']) for f in frames]
    print('GT objects per frame: mean %.1f, max %d, frames with none %d' % (np.mean(per_frame), max(per_frame), sum(1 for n in per_frame if n == 0)))
    print('first 30:', per_frame[:30])
    print('100-130:', per_frame[100:130])


if __name__ == '__main__':
    main()
