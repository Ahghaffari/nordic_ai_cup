"""Build a YOLO dataset of 960x540 views at all three resolution levels.

Synthetic part: object sprites pasted onto aerial backgrounds (INRIA tiles, and Helsinki frames with
the objects inpainted) at random position, scale and heading, rendered as L0/L1/L2 views exactly like
the evaluator crops them. Labels follow the ground-truth convention via the fitted footprints.
Real part: the Helsinki frames themselves, cut into L0/L1/L2 views with their true annotations.
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
from dtos import OBJECT_CLASSES  # noqa: E402
from sprite_library import footprint_box  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

DATA = Path('/mnt/data/nordicai/drone-flyby')
W4K, H4K = 3840, 2160
VIEW = (960, 540)
REGION = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
MIN_VISIBLE = 0.45
REAL_VAL_FRAMES = {2, 7, 12, 17, 22}

G = {}


def load_globals():
    G['tiles'] = [tifffile.imread(p) for p in sorted(glob.glob(str(DATA / 'datasets/backgrounds/inria/**/*.tif'), recursive=True))]
    G['tiles'] = [cv2.cvtColor(t, cv2.COLOR_RGB2BGR) for t in G['tiles']]
    helsinki = []
    for f in frame_numbers():
        img = load_frame(f)
        mask = np.zeros(img.shape[:2], np.uint8)
        for a in load_annotations(f):
            x1, y1, x2, y2 = a['bbox']
            mask[max(0, y1 - 6):y2 + 6, max(0, x1 - 6):x2 + 6] = 255
        helsinki.append(cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA))
    G['helsinki'] = helsinki
    lib = json.loads((DATA / 'sprites/library.json').read_text())
    by_class = {c: [] for c in OBJECT_CLASSES}
    for rec in lib:
        rgba = cv2.imread(str(DATA / 'sprites' / rec['file']), cv2.IMREAD_UNCHANGED)
        by_class[rec['object_id']].append((rgba, rec['footprint']))
    G['sprites'] = by_class


def grade(img, rng, strength=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.9, 1.0 + 0.5 * strength)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4) * strength) % 180
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    contrast = rng.uniform(1.0 - 0.1 * strength, 1.0 + 0.3 * strength)
    brightness = rng.uniform(-20, 15) * strength
    mean = out.mean()
    out = (out - mean) * contrast + mean + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def background(rng):
    if rng.random() < 0.75:
        tile = G['tiles'][rng.integers(len(G['tiles']))]
        cw = int(2176 * rng.uniform(0.75, 1.35))
        ch = cw * 9 // 16
        x = rng.integers(0, tile.shape[1] - cw)
        y = rng.integers(0, tile.shape[0] - ch)
        bg = cv2.resize(tile[y:y + ch, x:x + cw], (W4K, H4K), interpolation=cv2.INTER_CUBIC)
        bg = grade(bg, rng, 1.0)
    else:
        bg = G['helsinki'][rng.integers(len(G['helsinki']))].copy()
        bg = grade(bg, rng, 0.4)
    if rng.random() < 0.5:
        bg = bg[:, ::-1]
    if rng.random() < 0.5:
        bg = bg[::-1]
    return np.ascontiguousarray(bg)


def transform_sprite(rgba, fp, phi, scale, flip):
    if flip:
        rgba = rgba[:, ::-1]
        fp = dict(fp, center=[rgba.shape[1] - fp['center'][0], fp['center'][1]], theta=math.pi - fp['theta'])
    h, w = rgba.shape[:2]
    c, s = math.cos(phi) * scale, math.sin(phi) * scale
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    rot = corners @ np.array([[c, -s], [s, c]]).T
    ox, oy = -rot[:, 0].min(), -rot[:, 1].min()
    out_w, out_h = int(math.ceil(rot[:, 0].max() + ox)), int(math.ceil(rot[:, 1].max() + oy))
    M = np.array([[c, -s, ox], [s, c, oy]], np.float32)
    warped = cv2.warpAffine(np.ascontiguousarray(rgba), M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    cx, cy = M @ np.array([fp['center'][0], fp['center'][1], 1.0], np.float32)
    return warped, fp, float(cx), float(cy)


def paste(canvas, sprite, x0, y0, rng):
    h, w = sprite.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if x2 <= x1 or y2 <= y1:
        return
    part = sprite[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    alpha = cv2.GaussianBlur(part[..., 3].astype(np.float32) / 255.0, (0, 0), 0.6)[..., None]
    rgb = part[..., :3].astype(np.float32) * rng.uniform(0.85, 1.15)
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = np.clip(rgb * alpha + region * (1 - alpha), 0, 255).astype(np.uint8)


def overlaps(box, boxes, margin=8):
    return any(not (box[2] + margin < b[0] or b[2] + margin < box[0] or box[3] + margin < b[1] or b[3] + margin < box[1]) for b in boxes)


def composite(rng):
    canvas = background(rng)
    labels = []
    for _ in range(rng.integers(10, 34)):
        cls = int(rng.integers(len(OBJECT_CLASSES)))
        pool = G['sprites'][OBJECT_CLASSES[cls]]
        rgba, fp = pool[rng.integers(len(pool))]
        phi = rng.choice([0, 0.5, 1, 1.5]) * math.pi if rng.random() < 0.5 else rng.uniform(0, 2 * math.pi)
        scale = rng.uniform(0.85, 1.2)
        sprite, fp_t, scx, scy = transform_sprite(rgba, fp, phi, scale, rng.random() < 0.5)
        for _ in range(10):
            px = rng.uniform(-0.2, 1.2) * W4K if rng.random() < 0.08 else rng.uniform(40, W4K - 40)
            py = rng.uniform(-0.2, 1.2) * H4K if rng.random() < 0.08 else rng.uniform(40, H4K - 40)
            box = footprint_box(fp_t, px, py, rotation=phi, scale=scale)
            if not overlaps(box, [l[1] for l in labels]):
                break
        else:
            continue
        paste(canvas, sprite, int(round(px - scx)), int(round(py - scy)), rng)
        labels.append((cls, box))
    return canvas, labels


def render(canvas, level, cx, cy):
    rw, rh = REGION[level]
    x1, y1 = cx - rw // 2, cy - rh // 2
    crop = canvas[y1:y1 + rh, x1:x1 + rw]
    if level < 2:
        crop = cv2.resize(crop, VIEW, interpolation=cv2.INTER_AREA)
    return crop, (x1, y1, x1 + rw, y1 + rh)


def yolo_lines(labels, region):
    rx1, ry1, rx2, ry2 = region
    rw, rh = rx2 - rx1, ry2 - ry1
    lines = []
    for cls, (x1, y1, x2, y2) in labels:
        full = (x2 - x1) * (y2 - y1)
        ix1, iy1, ix2, iy2 = max(x1, rx1, 0), max(y1, ry1, 0), min(x2, rx2, W4K), min(y2, ry2, H4K)
        if ix2 <= ix1 or iy2 <= iy1 or (ix2 - ix1) * (iy2 - iy1) < MIN_VISIBLE * full:
            continue
        bw, bh = (ix2 - ix1) / rw, (iy2 - iy1) / rh
        if bw * VIEW[0] < 2 or bh * VIEW[1] < 2:
            continue
        lines.append(f'{cls} {((ix1 + ix2) / 2 - rx1) / rw:.6f} {((iy1 + iy2) / 2 - ry1) / rh:.6f} {bw:.6f} {bh:.6f}')
    return lines


def random_center(level, rng, labels, bias):
    rw, rh = REGION[level]
    if labels and rng.random() < bias:
        _, (x1, y1, x2, y2) = labels[rng.integers(len(labels))]
        cx = (x1 + x2) / 2 + rng.uniform(-0.4, 0.4) * rw
        cy = (y1 + y2) / 2 + rng.uniform(-0.4, 0.4) * rh
    else:
        cx, cy = rng.uniform(0, W4K), rng.uniform(0, H4K)
    return int(np.clip(cx, rw // 2, W4K - rw // 2)), int(np.clip(cy, rh // 2, H4K - rh // 2))


def write(out, split, name, img, lines):
    cv2.imwrite(str(out / 'images' / split / f'{name}.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    (out / 'labels' / split / f'{name}.txt').write_text('\n'.join(lines))


def synth_job(args):
    seed, out, split = args
    rng = np.random.default_rng(seed)
    canvas, labels = composite(rng)
    views = [(0, 1920, 1080)]
    views += [(1, *random_center(1, rng, labels, 0.5)) for _ in range(2)]
    views += [(2, *random_center(2, rng, labels, 0.75)) for _ in range(3)]
    for i, (level, cx, cy) in enumerate(views):
        img, region = render(canvas, level, cx, cy)
        write(out, split, f's{seed:06d}_{i}_L{level}', img, yolo_lines(labels, region))
    return len(labels)


def real_views(out, repeat):
    count = 0
    for f in frame_numbers():
        split = 'val' if f in REAL_VAL_FRAMES else 'train'
        img = load_frame(f)
        labels = [(OBJECT_CLASSES.index(a['object_id']), a['bbox']) for a in load_annotations(f)]
        views = [(0, 1920, 1080)]
        views += [(1, x, y) for x in (960, 1920, 2880) for y in (540, 1080, 1620)]
        views += [(2, 480 + 960 * i, 270 + 540 * j) for i in range(4) for j in range(4)]
        for level, cx, cy in views:
            view, region = render(img, level, cx, cy)
            lines = yolo_lines(labels, region)
            reps = 1 if split == 'val' else repeat
            for r in range(reps):
                write(out, split, f'real_f{f:03d}_L{level}_{cx}_{cy}_r{r}', view, lines)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='synth_v1')
    parser.add_argument('--train', type=int, default=3000)
    parser.add_argument('--val', type=int, default=100)
    parser.add_argument('--real-repeat', type=int, default=3)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()

    out = DATA / 'datasets' / args.name
    for sub in ('images/train', 'images/val', 'images/val_synth', 'labels/train', 'labels/val', 'labels/val_synth'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    load_globals()
    jobs = [(i, out, 'train') for i in range(args.train)] + [(1_000_000 + i, out, 'val_synth') for i in range(args.val)]
    with mp.get_context('fork').Pool(args.workers) as pool:
        placed = sum(pool.imap_unordered(synth_job, jobs, chunksize=8))
    n_real = real_views(out, args.real_repeat)

    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(OBJECT_CLASSES))
    (out / 'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
    (out / 'data_synthval.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val_synth\nnames:\n{names}\n')
    print(f'{len(jobs)} composites, {placed} objects placed, {n_real} real views -> {out}')


if __name__ == '__main__':
    main()
