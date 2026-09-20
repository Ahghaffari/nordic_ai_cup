"""Cut every annotated object out of the Helsinki frames as an RGBA sprite using SAM 2.1 box prompts."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402

DATA = Path('/mnt/data/nordicai/drone-flyby')
EDGE_MARGIN = 3
CONTEXT = 2.0
SAM_SIDE = 768


def segment(model, frame, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    side = int(max(w, h) * CONTEXT) + 16
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx1, ry1 = max(0, cx - side // 2), max(0, cy - side // 2)
    rx2, ry2 = min(frame.shape[1], rx1 + side), min(frame.shape[0], ry1 + side)
    region = frame[ry1:ry2, rx1:rx2]
    scale = SAM_SIDE / max(region.shape[:2])
    big = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    prompt = [(x1 - rx1) * scale, (y1 - ry1) * scale, (x2 - rx1) * scale, (y2 - ry1) * scale]
    result = model(big, bboxes=[prompt], verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None
    mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_AREA)
    full = np.zeros(frame.shape[:2], np.uint8)
    full[ry1:ry2, rx1:rx2] = mask
    inside = np.zeros_like(full)
    inside[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2] = 1
    return full * inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DATA / 'sprites'))
    parser.add_argument('--sam', default=str(DATA / 'models' / 'sam2.1_b.pt'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = SAM(args.sam)

    records = []
    for frame_no in frame_numbers():
        frame = load_frame(frame_no)
        H, W = frame.shape[:2]
        for ann in load_annotations(frame_no):
            x1, y1, x2, y2 = ann['bbox']
            if x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or x2 > W - EDGE_MARGIN or y2 > H - EDGE_MARGIN:
                continue
            mask = segment(model, frame, ann['bbox'])
            if mask is None or mask.sum() < 20:
                continue
            ys, xs = np.nonzero(mask)
            mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            pad = 6
            px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
            px2, py2 = min(W, x2 + pad), min(H, y2 + pad)
            rgba = np.dstack([frame[py1:py2, px1:px2], mask[py1:py2, px1:px2] * 255])
            name = f"{ann['object_id']}_{frame_no:03d}.png"
            cv2.imwrite(str(out / name), rgba)
            box_area = (x2 - x1) * (y2 - y1)
            inter = max(0, min(x2, mx2) - max(x1, mx1)) * max(0, min(y2, my2) - max(y1, my1))
            union = box_area + (mx2 - mx1) * (my2 - my1) - inter
            records.append({
                'file': name, 'object_id': ann['object_id'], 'frame': frame_no,
                'gt_in_sprite': [x1 - px1, y1 - py1, x2 - px1, y2 - py1],
                'mask_in_sprite': [int(mx1 - px1), int(my1 - py1), int(mx2 - px1), int(my2 - py1)],
                'center_y': (y1 + y2) / 2,
                'mask_fill': float(mask.sum() / box_area),
                'mask_gt_iou': inter / union,
            })
    (out / 'sprites.json').write_text(json.dumps(records, indent=1))
    print(f'{len(records)} sprites -> {out}')


if __name__ == '__main__':
    main()
