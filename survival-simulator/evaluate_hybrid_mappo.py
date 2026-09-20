from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import csv
import random
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame
import torch

from controller_config import (
    ControllerConfig,
    available_profiles,
    get_controller_config,
)
from hybrid_controller import SpawnCoordinator, compose_action
from mappo_policy import HybridMAPPO
from src.core import SimulationCore
from v2_encoder import OBS_DIM, encode_state


TAIL_TICKS = 2000  # official dt=0.1 -> final ~200 simulated seconds


def load_policy_and_config(
    checkpoint_path: str,
    device: torch.device,
    profile_override: str = "",
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
    if ckpt_obs_dim != OBS_DIM:
        raise RuntimeError(
            f"Checkpoint obs_dim={ckpt_obs_dim}, current OBS_DIM={OBS_DIM}."
        )

    hidden_dim = int(ckpt.get("hidden_dim", 256))
    critic_embed_dim = int(ckpt.get("critic_embed_dim", 128))

    policy = HybridMAPPO(OBS_DIM, hidden_dim, critic_embed_dim).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    if profile_override:
        controller_config = get_controller_config(profile_override)
        config_source = f"profile override: {profile_override}"
    elif ckpt.get("controller_config"):
        controller_config = ControllerConfig.from_dict(ckpt["controller_config"])
        config_source = "checkpoint controller_config"
    else:
        controller_config = get_controller_config("v4_robust7")
        config_source = "fallback profile: v4_robust7"

    return policy, controller_config, ckpt, config_source


def _snapshot(state: Dict) -> Dict[str, float]:
    agents = list(state.get("observations", []) or [])
    if not agents:
        return {
            "pop": 0.0,
            "mean_energy": 0.0,
            "min_energy": 0.0,
            "mean_age": 0.0,
            "old_fraction": 0.0,
            "young_fraction": 0.0,
            "spawnable_fraction": 0.0,
            "fruit_seen_fraction": 0.0,
            "tree_seen_fraction": 0.0,
            "predator_seen_fraction": 0.0,
        }

    energies = np.asarray([float(a["energy"]) for a in agents], dtype=np.float64)
    ages = np.asarray([float(a["age"]) for a in agents], dtype=np.float64)

    fruit_seen = []
    tree_seen = []
    predator_seen = []
    for agent in agents:
        obs = agent.get("observations", []) or []
        fruit_seen.append(any(o.get("type") == "Fruit" for o in obs))
        tree_seen.append(any(o.get("type") == "Tree" for o in obs))
        predator_seen.append(any(o.get("type") == "Predator" for o in obs))

    return {
        "pop": float(len(agents)),
        "mean_energy": float(energies.mean()),
        "min_energy": float(energies.min()),
        "mean_age": float(ages.mean()),
        "old_fraction": float(np.mean(ages >= 45.0)),
        "young_fraction": float(np.mean(ages < 30.0)),
        "spawnable_fraction": float(np.mean(energies > 100.0)),
        "fruit_seen_fraction": float(np.mean(fruit_seen)),
        "tree_seen_fraction": float(np.mean(tree_seen)),
        "predator_seen_fraction": float(np.mean(predator_seen)),
    }


def _tail_summary(history: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(history)
    if not items:
        return {}
    keys = items[0].keys()
    return {
        f"tail_{key}": float(np.mean([item[key] for item in items]))
        for key in keys
    }


def run_episode(
    policy: Optional[HybridMAPPO],
    device: torch.device,
    seed: int,
    controller_config: ControllerConfig,
    heuristic_only: bool = False,
):
    sim = SimulationCore(seed=int(seed))
    state = sim.step([])
    coordinator = SpawnCoordinator(controller_config)

    births = 0
    deaths = 0
    max_population = int(state["num_agents"])
    mode_counter = Counter()
    tail_history = deque(maxlen=TAIL_TICKS)

    while int(state["num_agents"]) > 0 and float(state["sim_time"]) < 3000.0:
        agents, obs_np, _ = encode_state(state)

        if heuristic_only:
            raw_np = np.zeros((len(agents), 3), dtype=np.float32)
        else:
            if policy is None:
                raise RuntimeError("policy is required unless --heuristic-only")
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            with torch.inference_mode():
                raw_t, _, _ = policy.act_actor(obs_t, deterministic=True)
            raw_np = raw_t.cpu().numpy().astype(np.float32)

        spawn_id = coordinator.choose(state)
        prev_ids = {int(a["agent_id"]) for a in agents}
        actions = []

        for i, agent in enumerate(agents):
            req, _used, mode = compose_action(
                agent_state=agent,
                state=state,
                raw_residual=raw_np[i],
                spawn_agent=(
                    spawn_id is not None
                    and int(agent["agent_id"]) == int(spawn_id)
                ),
            )
            mode_counter[mode] += 1
            actions.append((int(agent["agent_id"]), req))

        state = sim.step(actions)
        next_ids = {
            int(a["agent_id"])
            for a in (state.get("observations", []) or [])
        }
        births += len(next_ids - prev_ids)
        deaths += len(prev_ids - next_ids)
        max_population = max(max_population, int(state["num_agents"]))
        tail_history.append(_snapshot(state))

    result = {
        "seed": int(seed),
        "score": float(state["score"]),
        "time": float(state["sim_time"]),
        "survived_full": float(state["sim_time"]) >= 3000.0,
        "births": int(births),
        "deaths": int(deaths),
        "max_population": int(max_population),
        "final_population": int(state["num_agents"]),
        "modes": mode_counter,
    }
    result.update(_tail_summary(tail_history))
    return result


def parse_seed_file(path: str) -> List[int]:
    seeds: List[int] = []
    seen = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.replace("\t", ",").split(",")[0].strip()
        try:
            seed = int(first)
        except ValueError:
            continue
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return seeds


def build_seed_list(args) -> List[int]:
    if args.seed_file:
        seeds = parse_seed_file(args.seed_file)
        if not seeds:
            raise ValueError(f"No seeds found in {args.seed_file}")
        if args.episodes > 0:
            seeds = seeds[: args.episodes]
        return seeds

    if args.random_seeds:
        rng = random.Random(args.seed_rng)
        return [rng.randint(0, 2**32 - 1) for _ in range(args.episodes)]

    return [args.seed_start + i for i in range(args.episodes)]


def robust_metrics(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def report_results(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    scores = np.asarray([r["score"] for r in results], dtype=np.float64)
    times = np.asarray([r["time"] for r in results], dtype=np.float64)
    births = np.asarray([r["births"] for r in results], dtype=np.float64)
    max_pops = np.asarray([r["max_population"] for r in results], dtype=np.float64)

    sm = robust_metrics(scores)
    tm = robust_metrics(times)

    print("\n=== Evaluation ===")
    print(f"Score mean:      {sm['mean']:.2f}")
    print(f"Score median:    {sm['median']:.2f}")
    print(f"Score std:       {sm['std']:.2f}")
    print(f"Score p10/p25:   {sm['p10']:.2f} / {sm['p25']:.2f}")
    print(f"Score p75/p90:   {sm['p75']:.2f} / {sm['p90']:.2f}")
    print(f"Score min/max:   {sm['min']:.2f} / {sm['max']:.2f}")

    print("\n=== Survival robustness ===")
    print(f"Time mean:       {tm['mean']:.2f}s")
    print(f"Time median:     {tm['median']:.2f}s")
    print(f"Time p10/p25:    {tm['p10']:.2f}s / {tm['p25']:.2f}s")
    print(f"Time min/max:    {tm['min']:.2f}s / {tm['max']:.2f}s")
    print(f"Full survival:   {sum(r['survived_full'] for r in results)}/{len(results)}")
    print(f"Births mean:     {births.mean():.2f}")
    print(f"Max pop mean:    {max_pops.mean():.2f}")

    tail_keys = [
        ("tail_pop", "Tail pop"),
        ("tail_mean_energy", "Tail mean energy"),
        ("tail_old_fraction", "Tail old fraction"),
        ("tail_young_fraction", "Tail young fraction"),
        ("tail_spawnable_fraction", "Tail spawnable fraction"),
        ("tail_fruit_seen_fraction", "Tail fruit-seen fraction"),
        ("tail_tree_seen_fraction", "Tail tree-seen fraction"),
        ("tail_predator_seen_fraction", "Tail predator-seen fraction"),
    ]
    print("\nLate-game diagnostics (mean over each episode's final ~200s):")
    for key, label in tail_keys:
        vals = [float(r.get(key, 0.0)) for r in results]
        value = float(np.mean(vals))
        if "fraction" in key:
            print(f"  {label:26s}: {100.0 * value:6.1f}%")
        else:
            print(f"  {label:26s}: {value:8.2f}")

    total_modes = Counter()
    for result in results:
        total_modes.update(result["modes"])
    print("\nAction-mode counts:")
    total = max(sum(total_modes.values()), 1)
    for mode, count in total_modes.most_common():
        print(f"  {mode:16s} {count:10d}  ({100.0 * count / total:5.1f}%)")

    return {"score": sm, "time": tm}


def save_results_csv(path: str, results: List[Dict]):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "score",
        "time",
        "survived_full",
        "births",
        "deaths",
        "max_population",
        "final_population",
        "tail_pop",
        "tail_mean_energy",
        "tail_min_energy",
        "tail_mean_age",
        "tail_old_fraction",
        "tail_young_fraction",
        "tail_spawnable_fraction",
        "tail_fruit_seen_fraction",
        "tail_tree_seen_fraction",
        "tail_predator_seen_fraction",
    ]

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in fieldnames})


def save_worst_seeds(path: str, results: List[Dict], count: int):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    worst = sorted(results, key=lambda r: float(r["time"]))[: max(1, int(count))]
    lines = ["seed,time,score"]
    lines.extend(
        f"{int(r['seed'])},{float(r['time']):.6f},{float(r['score']):.6f}"
        for r in worst
    )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_checkpoint(
    checkpoint: str,
    seeds: List[int],
    device: torch.device,
    profile_override: str = "",
    heuristic_only: bool = False,
    verbose: bool = True,
):
    policy = None
    if heuristic_only:
        controller_config = (
            get_controller_config(profile_override)
            if profile_override
            else get_controller_config("v4_robust7")
        )
        config_source = f"heuristic profile: {controller_config.name}"
    else:
        policy, controller_config, _ckpt, config_source = load_policy_and_config(
            checkpoint, device, profile_override
        )

    if verbose:
        print(f"Controller: {controller_config.name} ({config_source})")
        if not heuristic_only:
            print(f"Checkpoint: {checkpoint}")
        print(f"Evaluating {len(seeds)} seeds")

    results = []
    for i, seed in enumerate(seeds):
        result = run_episode(
            policy=policy,
            device=device,
            seed=seed,
            controller_config=controller_config,
            heuristic_only=heuristic_only,
        )
        results.append(result)
        if verbose:
            print(
                f"{i+1:03d}/{len(seeds)} seed={seed} "
                f"score={result['score']:.2f} time={result['time']:.1f}s "
                f"births={result['births']} deaths={result['deaths']} "
                f"max_pop={result['max_population']} "
                f"tailE={result.get('tail_mean_energy', 0.0):.1f} "
                f"tailOld={100.0 * result.get('tail_old_fraction', 0.0):.0f}%"
            )

    return results, controller_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/hybrid_mappo_v4_latest.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=10_000)
    p.add_argument("--random-seeds", action="store_true")
    p.add_argument("--seed-rng", type=int, default=2026)
    p.add_argument("--seed-file", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--heuristic-only", action="store_true")
    p.add_argument(
        "--profile",
        choices=("",) + available_profiles(),
        default="",
        help="Override checkpoint controller config; useful for fair ablations.",
    )
    p.add_argument("--save-csv", default="")
    p.add_argument("--worst-seeds-out", default="")
    p.add_argument("--worst-count", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    pygame.init()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device(args.device)

    seeds = build_seed_list(args)
    results, _controller_config = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        seeds=seeds,
        device=device,
        profile_override=args.profile,
        heuristic_only=args.heuristic_only,
        verbose=True,
    )
    report_results(results)
    save_results_csv(args.save_csv, results)
    save_worst_seeds(args.worst_seeds_out, results, args.worst_count)

    if args.save_csv:
        print(f"Saved results to {args.save_csv}")
    if args.worst_seeds_out:
        print(
            f"Saved worst {min(args.worst_count, len(results))} seeds "
            f"to {args.worst_seeds_out}"
        )

    pygame.quit()


if __name__ == "__main__":
    main()
