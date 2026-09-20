"""Train the view detector on a generated dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATA = Path('/mnt/data/nordicai/drone-flyby')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synth_v1')
    parser.add_argument('--model', default='yolo26m.pt')
    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--device', default='0,1')
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--data-yaml', default='data.yaml')
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--save-period', type=int, default=-1)
    parser.add_argument('--cache', default='ram')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    name = args.name or f'{Path(args.model).stem}_{args.dataset}'
    model = YOLO(str(DATA / 'models' / args.model))
    model.train(
        data=str(DATA / 'datasets' / args.dataset / args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=12,
        cache=False if args.cache == 'none' else args.cache,
        fraction=args.fraction,
        project=str(DATA / 'runs'),
        name=name,
        exist_ok=False,
        seed=args.seed,
        patience=args.patience,
        save_period=args.save_period,
        close_mosaic=5,
        mosaic=1.0,
        scale=0.25,
        degrees=0.0,
        translate=0.1,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        mixup=0.0,
        plots=True,
    )


if __name__ == '__main__':
    main()
