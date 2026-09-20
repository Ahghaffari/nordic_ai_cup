from __future__ import annotations

import argparse
import random
from pathlib import Path


def generate_unique(rng: random.Random, count: int, used: set[int]):
    out = []
    while len(out) < count:
        seed = rng.randint(0, 2**32 - 1)
        if seed in used:
            continue
        used.add(seed)
        out.append(seed)
    return out


def write_seed_file(path: Path, seeds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(s)) for s in seeds) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="seed_banks")
    p.add_argument("--master-seed", type=int, default=20260917)
    p.add_argument("--mining", type=int, default=40)
    p.add_argument("--dev", type=int, default=20)
    p.add_argument("--holdout", type=int, default=30)
    args = p.parse_args()

    if min(args.mining, args.dev, args.holdout) <= 0:
        raise ValueError("All bank sizes must be positive")

    rng = random.Random(args.master_seed)
    used: set[int] = set()
    mining = generate_unique(rng, args.mining, used)
    dev = generate_unique(rng, args.dev, used)
    holdout = generate_unique(rng, args.holdout, used)

    out = Path(args.out_dir)
    write_seed_file(out / "mining_seeds.txt", mining)
    write_seed_file(out / "dev_seeds.txt", dev)
    write_seed_file(out / "holdout_seeds.txt", holdout)

    print(f"Wrote {len(mining)} mining seeds  -> {out / 'mining_seeds.txt'}")
    print(f"Wrote {len(dev)} dev seeds         -> {out / 'dev_seeds.txt'}")
    print(f"Wrote {len(holdout)} holdout seeds -> {out / 'holdout_seeds.txt'}")
    print("Banks are disjoint. Never put DEV/HOLDOUT seeds into training hard-seed replay.")


if __name__ == "__main__":
    main()
