"""Keep models/detector.pt pointing at the best epoch seen so far.

Watches every run whose validation set is a selection set (data yaml name contains 'select'), reads
mAP@0.50 per epoch from results.csv, and when an epoch checkpoint beats the deployed model by a margin,
copies it into models/ and repoints detector.pt atomically. The server hot-swaps it once idle.
"""

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

DATA = Path('/mnt/data/nordicai/drone-flyby')
MODELS = DATA / 'models'
LINK = MODELS / 'detector.pt'
STATE = MODELS / 'deployed.json'
LOG = MODELS / 'DEPLOYMENTS.log'


def candidates(pattern: str):
    for run in sorted((DATA / 'runs').glob(pattern)):
        args_file, results = run / 'args.yaml', run / 'results.csv'
        if not args_file.exists() or not results.exists():
            continue
        if 'select' not in Path(yaml.safe_load(args_file.read_text())['data']).name:
            continue
        with open(results) as handle:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(handle)]
        for row in rows:
            epoch = int(float(row['epoch']))
            weights = run / 'weights' / f'epoch{epoch - 1}.pt'
            if weights.exists() and time.time() - weights.stat().st_mtime > 30:
                yield float(row['metrics/mAP50(B)']), float(row['metrics/mAP50-95(B)']), run.name, epoch, weights


def deploy(map50, map5095, run, epoch, weights):
    target = MODELS / f'{run}_e{epoch}.pt'
    shutil.copy2(weights, target)
    tmp = MODELS / '.detector.pt.tmp'
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target)
    os.replace(tmp, LINK)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    STATE.write_text(json.dumps({'path': str(target), 'run': run, 'epoch': epoch, 'map50': map50,
                                 'map50_95': map5095, 'deployed_at': now}, indent=1))
    with open(LOG, 'a') as handle:
        handle.write(f'{now}  {target.name}  select mAP50={map50:.4f} mAP50-95={map5095:.4f}\n')
    print(f'{now} deployed {target.name} mAP50={map50:.4f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='y26*')
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--margin', type=float, default=0.002)
    args = parser.parse_args()
    while True:
        current = json.loads(STATE.read_text()) if STATE.exists() else {'map50': -1.0}
        found = list(candidates(args.runs))
        if found:
            best = max(found)
            if best[0] > current['map50'] + args.margin:
                deploy(*best)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
