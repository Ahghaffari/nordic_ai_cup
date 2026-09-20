"""Dataset v2 part: sprites pasted onto recorded validation-scene views.

Each recorded Level-1 view (960x540 covering 1920x1080 source pixels) is cleaned by inpainting every
low-confidence detection, upsampled to source resolution, used as a canvas for sprites exactly as in
make_dataset.py, and rendered back to a Level-1 view (and sometimes a Level-2 crop). Clean views
without sprites are kept as pure negatives.
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
import make_dataset as md  # noqa: E402
from dtos import OBJECT_CLASSES  # noqa: E402
from sprite_library import footprint_box  # noqa: E402

DATA = Path('/mnt/data/nordicai/drone-flyby')
CANVAS = (1920, 1080)

G = {}


def load_globals(mask_file: Path):
    lib = json.loads((DATA / 'sprites/library.json').read_text())
    by_class = {c: [] for c in OBJECT_CLASSES}
    for rec in lib:
        rgba = cv2.imread(str(DATA / 'sprites' / rec['file']), cv2.IMREAD_UNCHANGED)
        by_class[rec['object_id']].append((rgba, rec['footprint']))
    G['sprites'] = by_class
    md.G['sprites'] = by_class
    G['views'] = list(json.loads(mask_file.read_text()).values())


def clean_background(entry, rng):
    view = cv2.imread(entry['png'])
    mask = np.zeros(view.shape[:2], np.uint8)
    for x1, y1, x2, y2 in entry['boxes']:
        pad = 4
        mask[max(0, int(y1) - pad):int(y2) + pad, max(0, int(x1) - pad):int(x2) + pad] = 255
    if mask.any():
        view = cv2.inpaint(view, mask, 5, cv2.INPAINT_TELEA)
    if rng.random() < 0.5:
        view = view[:, ::-1]
    if rng.random() < 0.5:
        view = view[::-1]
    return np.ascontiguousarray(view)


def composite(view, rng):
    canvas = cv2.resize(view, CANVAS, interpolation=cv2.INTER_CUBIC)
    canvas = md.grade(canvas, rng, 0.3)
    labels = []
    for _ in range(rng.integers(3, 11)):
        cls = int(rng.integers(len(OBJECT_CLASSES)))
        pool = G['sprites'][OBJECT_CLASSES[cls]]
        rgba, fp = pool[rng.integers(len(pool))]
        phi = rng.choice([0, 0.5, 1, 1.5]) * math.pi if rng.random() < 0.5 else rng.uniform(0, 2 * math.pi)
        scale = rng.uniform(0.85, 1.2)
        sprite, fp_t, scx, scy = md.transform_sprite(rgba, fp, phi, scale, rng.random() < 0.5)
        for _ in range(10):
            px = rng.uniform(-0.05, 1.05) * CANVAS[0]
            py = rng.uniform(-0.05, 1.05) * CANVAS[1]
            box = footprint_box(fp_t, px, py, rotation=phi, scale=scale)
            if not md.overlaps(box, [l[1] for l in labels]):
                break
        else:
            continue
        md.paste(canvas, sprite, int(round(px - scx)), int(round(py - scy)), rng)
        labels.append((cls, box))
    return canvas, labels


def job(args):
    seed, out, split, negative = args
    rng = np.random.default_rng(seed)
    entry = G['views'][seed % len(G['views'])]
    view = clean_background(entry, rng)
    region = (0, 0, CANVAS[0], CANVAS[1])
    if negative:
        md.write(out, split, f'v{seed:06d}_neg_L1', view, [])
        return 0
    canvas, labels = composite(view, rng)
    l1 = cv2.resize(canvas, (960, 540), interpolation=cv2.INTER_AREA)
    md.write(out, split, f'v{seed:06d}_L1', l1, md.yolo_lines(labels, region))
    if labels and rng.random() < 0.3:
        _, (x1, y1, x2, y2) = labels[rng.integers(len(labels))]
        cx = int(np.clip((x1 + x2) / 2 + rng.uniform(-300, 300), 480, CANVAS[0] - 480))
        cy = int(np.clip((y1 + y2) / 2 + rng.uniform(-170, 170), 270, CANVAS[1] - 270))
        crop = canvas[cy - 270:cy + 270, cx - 480:cx + 480]
        md.write(out, split, f'v{seed:06d}_L2', crop, md.yolo_lines(labels, (cx - 480, cy - 270, cx + 480, cy + 270)))
    return len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='valbg_v2')
    parser.add_argument('--composites', type=int, default=3000)
    parser.add_argument('--negatives', type=int, default=400)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--masks', default=str(DATA / 'analysis/val_view_masks.json'))
    args = parser.parse_args()

    out = DATA / 'datasets' / args.name
    for sub in ('images/train', 'labels/train'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    load_globals(Path(args.masks))
    jobs = [(i, out, 'train', False) for i in range(args.composites)]
    jobs += [(500_000 + i, out, 'train', True) for i in range(args.negatives)]
    with mp.get_context('fork').Pool(args.workers) as pool:
        placed = sum(pool.imap_unordered(job, jobs, chunksize=8))
    print(f'{len(jobs)} jobs, {placed} objects placed -> {out}')


if __name__ == '__main__':
    main()
