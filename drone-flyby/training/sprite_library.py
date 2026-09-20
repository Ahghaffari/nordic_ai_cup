"""Clean SAM masks and fit, per sprite, the rotated footprint rectangle whose axis-aligned
bounds reproduce the ground-truth box convention (the simulator boxes the 3D model bounds,
not the visible pixels). Pasting code rotates this rectangle to derive labels."""

import json
import math
from pathlib import Path

import cv2
import numpy as np

SPRITES = Path('/mnt/data/nordicai/drone-flyby/sprites')
GRABCUT_CLASSES = {'small_launcher'}
CONTRAST_CLASSES = {'medium_launcher'}


def largest_central_component(mask, gt):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    best, best_score = 0, -1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        d = math.hypot(x + w / 2 - cx, y + h / 2 - cy) + 1
        score = area / d
        if score > best_score:
            best, best_score = i, score
    return (labels == best).astype(np.uint8)


def grabcut_mask(rgb, gt):
    x1, y1, x2, y2 = gt
    w, h = x2 - x1, y2 - y1
    inner = (x1 + w // 5, y1 + h // 5, max(2, w * 3 // 5), max(2, h * 3 // 5))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(rgb, mask, inner, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)


def contrast_mask(rgb, gt):
    grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.abs(grey - cv2.GaussianBlur(grey, (0, 0), 1.5))
    detail = cv2.GaussianBlur(detail, (0, 0), 1.0)
    x1, y1, x2, y2 = gt
    mask = np.zeros(rgb.shape[:2], np.uint8)
    box = detail[y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = (box > max(6.0, np.percentile(box, 75))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def fit_footprint(mask, gt):
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:
        rw, rh, ang = rh, rw, ang + 90
    theta = math.radians(ang)
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    det = c * c - s * s
    if abs(det) > 0.35:
        L = (gw * c - gh * s) / det
        W = (gh * c - gw * s) / det
    else:
        m = ((gw + gh) / (c + s) - rw - rh) / 2
        L, W = rw + m, rh + m
    L, W = max(L, 0.8 * rw, 2.0), max(W, 0.8 * rh, 2.0)
    bw, bh = L * c + W * s, L * s + W * c
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return {'center': [rcx, rcy], 'theta': theta, 'L': L, 'W': W,
            'offset': [gcx - rcx, gcy - rcy], 'scale_fix': [gw / bw, gh / bh]}


def footprint_box(fp, cx, cy, rotation=0.0, scale=1.0):
    theta = fp['theta'] + rotation
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    L, W = fp['L'] * scale, fp['W'] * scale
    bw, bh = (L * c + W * s) * fp['scale_fix'][0], (L * s + W * c) * fp['scale_fix'][1]
    ox, oy = fp['offset'][0] * scale, fp['offset'][1] * scale
    return cx + ox - bw / 2, cy + oy - bh / 2, cx + ox + bw / 2, cy + oy + bh / 2


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    records = json.loads((SPRITES / 'sprites.json').read_text())
    library = []
    for rec in records:
        rgba = cv2.imread(str(SPRITES / rec['file']), cv2.IMREAD_UNCHANGED)
        rgb, mask = rgba[..., :3].copy(), (rgba[..., 3] > 127).astype(np.uint8)
        gt = rec['gt_in_sprite']
        if rec['object_id'] in GRABCUT_CLASSES:
            mask = grabcut_mask(rgb, gt)
        if rec['object_id'] in CONTRAST_CLASSES:
            mask = contrast_mask(rgb, gt)
        mask = largest_central_component(mask, gt)
        if mask.sum() < 12:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        rgba[..., 3] = mask * 255
        cv2.imwrite(str(SPRITES / rec['file']), rgba)
        fp = fit_footprint(mask, gt)
        rec_out = dict(rec, footprint=fp, mask_fill=float(mask.sum() / ((gt[2] - gt[0]) * (gt[3] - gt[1]))))
        rec_out['reproduced_iou'] = iou(footprint_box(fp, *fp['center']), gt)
        library.append(rec_out)
    (SPRITES / 'library.json').write_text(json.dumps(library, indent=1))
    by = {}
    for r in library:
        by.setdefault(r['object_id'], []).append(r)
    for k in sorted(by):
        v = by[k]
        print(f"{k:16s} n={len(v):2d} fill={np.mean([r['mask_fill'] for r in v]):.2f} "
              f"repro_iou={np.min([r['reproduced_iou'] for r in v]):.3f} "
              f"L/W={np.median([r['footprint']['L'] for r in v]):.0f}/{np.median([r['footprint']['W'] for r in v]):.0f} "
              f"theta={np.degrees(np.median([r['footprint']['theta'] for r in v])):.0f}")


if __name__ == '__main__':
    main()
