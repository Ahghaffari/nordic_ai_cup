from __future__ import annotations

# Set before importing pygame/simulator code.
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from controller_config import available_profiles, get_controller_config
from hybrid_controller import SpawnCoordinator, compose_action
from mappo_policy import HybridMAPPO
from src.core import SimulationCore
from v2_encoder import OBS_DIM, encode_state


@dataclass
class TeamTransition:
    team_obs: np.ndarray
    actor_obs: np.ndarray
    raw_residual: np.ndarray
    old_log_prob: np.ndarray
    actor_used: np.ndarray
    value: float
    reward: float
    done: bool
    next_value: float


# -----------------------------------------------------------------------------
# Seed sampling: mostly fresh worlds, small amount of VERIFIED hard-seed replay.
# -----------------------------------------------------------------------------


def parse_seed_file(path: str) -> List[int]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    seeds: List[int] = []
    seen = set()
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.replace("\t", ",").split(",")[0].strip()
        try:
            seed = int(first)
        except ValueError:
            # Header such as seed,score.
            continue
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return seeds


class SeedSampler:
    def __init__(
        self,
        seed: int,
        hard_seeds: Sequence[int],
        hard_probability: float,
    ):
        self.rng = random.Random(int(seed))
        self.hard_seeds = [int(s) for s in hard_seeds]
        self.hard_probability = float(hard_probability)

    def sample(self) -> Tuple[int, str]:
        if self.hard_seeds and self.rng.random() < self.hard_probability:
            return int(self.rng.choice(self.hard_seeds)), "hard"
        return int(self.rng.randint(0, 2**32 - 1)), "fresh"


# -----------------------------------------------------------------------------
# Simulator/reward
# -----------------------------------------------------------------------------


def create_sim(seed: int):
    sim = SimulationCore(seed=int(seed))
    # Organizer local_playground/server primes observations with step([]).
    state = sim.step([])
    return sim, state


def _ids(state: Dict):
    return {
        int(a["agent_id"])
        for a in (state.get("observations", []) or [])
    }


def compute_team_reward(prev_state: Dict, next_state: Dict):
    """Team reward kept deliberately close to the organizer score."""
    score_delta = float(next_state["score"] - prev_state["score"])
    prev_ids = _ids(prev_state)
    next_ids = _ids(next_state)
    deaths = len(prev_ids - next_ids)
    births = len(next_ids - prev_ids)
    population = int(next_state["num_agents"])

    reward = score_delta
    reward -= 0.05 * deaths
    if population == 0:
        reward -= 5.0

    return float(reward), {
        "score_delta": score_delta,
        "deaths": deaths,
        "births": births,
    }


# -----------------------------------------------------------------------------
# Team-level GAE / PPO
# -----------------------------------------------------------------------------


def compute_gae(
    records: Sequence[TeamTransition],
    gamma: float,
    gae_lambda: float,
):
    advantages = np.zeros(len(records), dtype=np.float32)
    returns = np.zeros(len(records), dtype=np.float32)

    gae = 0.0
    for i in reversed(range(len(records))):
        rec = records[i]
        nonterminal = 0.0 if rec.done else 1.0
        delta = rec.reward + gamma * rec.next_value * nonterminal - rec.value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[i] = gae
        returns[i] = gae + rec.value

    return advantages, returns


def pad_team_obs(arrays: Sequence[np.ndarray], device: torch.device):
    batch = len(arrays)
    max_n = max(arr.shape[0] for arr in arrays)
    obs_dim = arrays[0].shape[1]

    padded = np.zeros((batch, max_n, obs_dim), dtype=np.float32)
    mask = np.zeros((batch, max_n), dtype=np.bool_)
    for i, arr in enumerate(arrays):
        n = arr.shape[0]
        padded[i, :n] = arr
        mask[i, :n] = True

    return (
        torch.as_tensor(padded, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def weighted_mean(x: torch.Tensor, w: torch.Tensor):
    return (x * w).sum() / w.sum().clamp_min(1e-8)


def ppo_update(
    policy: HybridMAPPO,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    records: Sequence[TeamTransition],
    advantages: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    actor_batch_size: int,
    critic_batch_size: int,
    epochs: int,
    clip_eps: float,
    entropy_coef: float,
    max_grad_norm: float,
    target_kl: float,
    update_actor: bool,
):
    adv_tick = advantages.astype(np.float32)
    adv_tick = (adv_tick - adv_tick.mean()) / (adv_tick.std() + 1e-8)

    metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "actor_updates": 0,
        "critic_updates": 0,
    }

    # Actor: each simulator tick has total weight 1 regardless of population.
    if update_actor:
        actor_obs_list = []
        raw_list = []
        old_logp_list = []
        actor_adv_list = []
        actor_weight_list = []

        for t, rec in enumerate(records):
            used_idx = np.flatnonzero(rec.actor_used)
            if len(used_idx) == 0:
                continue
            per_agent_weight = 1.0 / float(len(used_idx))
            for i in used_idx:
                actor_obs_list.append(rec.actor_obs[i])
                raw_list.append(rec.raw_residual[i])
                old_logp_list.append(rec.old_log_prob[i])
                actor_adv_list.append(adv_tick[t])
                actor_weight_list.append(per_agent_weight)

        if actor_obs_list:
            actor_obs = torch.as_tensor(
                np.asarray(actor_obs_list), dtype=torch.float32, device=device
            )
            raw = torch.as_tensor(np.asarray(raw_list), dtype=torch.float32, device=device)
            old_logp = torch.as_tensor(
                np.asarray(old_logp_list), dtype=torch.float32, device=device
            )
            actor_adv = torch.as_tensor(
                np.asarray(actor_adv_list), dtype=torch.float32, device=device
            )
            actor_w = torch.as_tensor(
                np.asarray(actor_weight_list), dtype=torch.float32, device=device
            )

            indices = np.arange(actor_obs.shape[0])
            early_stop = False

            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), actor_batch_size):
                    mb = indices[start : start + actor_batch_size]
                    mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)

                    new_logp, entropy = policy.evaluate_actor(
                        actor_obs[mb_t], raw[mb_t]
                    )
                    log_ratio = new_logp - old_logp[mb_t]
                    ratio = log_ratio.exp()

                    adv_mb = actor_adv[mb_t]
                    w_mb = actor_w[mb_t]
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(
                        ratio, 1.0 - clip_eps, 1.0 + clip_eps
                    ) * adv_mb
                    actor_loss = weighted_mean(
                        -torch.minimum(surr1, surr2), w_mb
                    )
                    entropy_mean = weighted_mean(entropy, w_mb)
                    loss = actor_loss - entropy_coef * entropy_mean

                    actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(policy.actor_trunk.parameters())
                        + list(policy.actor_mean.parameters())
                        + [policy.log_std],
                        max_grad_norm,
                    )
                    actor_optimizer.step()

                    with torch.no_grad():
                        approx_kl_per = (ratio - 1.0) - log_ratio
                        clip_per = ((ratio - 1.0).abs() > clip_eps).float()
                        approx_kl = float(
                            weighted_mean(approx_kl_per, w_mb).item()
                        )
                        clip_fraction = float(
                            weighted_mean(clip_per, w_mb).item()
                        )

                    metrics["actor_loss"] += float(actor_loss.detach().item())
                    metrics["entropy"] += float(entropy_mean.detach().item())
                    metrics["approx_kl"] += approx_kl
                    metrics["clip_fraction"] += clip_fraction
                    metrics["actor_updates"] += 1

                    if target_kl > 0.0 and approx_kl > target_kl:
                        early_stop = True
                        break
                if early_stop:
                    break

    # Centralized critic.
    tick_indices = np.arange(len(records))
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    for _ in range(epochs):
        np.random.shuffle(tick_indices)
        for start in range(0, len(records), critic_batch_size):
            mb = tick_indices[start : start + critic_batch_size]
            team_arrays = [records[int(i)].team_obs for i in mb]
            padded, mask = pad_team_obs(team_arrays, device)
            pred = policy.team_value_batch(padded, mask)
            target = returns_t[
                torch.as_tensor(mb, dtype=torch.long, device=device)
            ]

            critic_loss = F.smooth_l1_loss(pred, target)

            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.critic_agent_encoder.parameters())
                + list(policy.critic_head.parameters()),
                max_grad_norm,
            )
            critic_optimizer.step()

            metrics["critic_loss"] += float(critic_loss.detach().item())
            metrics["critic_updates"] += 1

    au = max(int(metrics["actor_updates"]), 1)
    cu = max(int(metrics["critic_updates"]), 1)
    for key in ("actor_loss", "entropy", "approx_kl", "clip_fraction"):
        metrics[key] /= au
    metrics["critic_loss"] /= cu
    return metrics


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    policy: HybridMAPPO,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    total_ticks: int,
    update_idx: int,
    args,
    controller_config,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "actor_optimizer_state": actor_optimizer.state_dict(),
            "critic_optimizer_state": critic_optimizer.state_dict(),
            "obs_dim": OBS_DIM,
            "hidden_dim": args.hidden_dim,
            "critic_embed_dim": args.critic_embed_dim,
            "total_ticks": int(total_ticks),
            "update_idx": int(update_idx),
            "controller_config": controller_config.to_dict(),
            "controller_version": "v4_v2movement_agebalanced_spawn",
            "config": vars(args),
        },
        path,
    )


def snapshot_path(template: str, total_ticks: int) -> Path:
    path = Path(template)
    suffix = path.suffix or ".pt"
    stem = path.stem
    return path.with_name(f"{stem}_tick_{int(total_ticks):09d}{suffix}")


def parse_args():
    p = argparse.ArgumentParser()

    # Resume/fine-tune budget.
    p.add_argument("--total-ticks", type=int, default=250_000)
    p.add_argument("--additional-ticks", type=int, default=0)
    p.add_argument("--rollout-ticks", type=int, default=2048)

    # Neural architecture: intentionally unchanged from V2/V3.
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--critic-embed-dim", type=int, default=128)

    # Conservative fine-tuning defaults for an already-good checkpoint.
    p.add_argument("--actor-lr", type=float, default=2e-5)
    p.add_argument("--critic-lr", type=float, default=5e-5)
    p.add_argument("--gamma", type=float, default=0.9995)
    p.add_argument("--gae-lambda", type=float, default=0.98)
    p.add_argument("--clip-eps", type=float, default=0.10)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--actor-batch-size", type=int, default=2048)
    p.add_argument("--critic-batch-size", type=int, default=256)
    p.add_argument("--entropy-coef", type=float, default=0.0005)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.010)

    # Let critic adapt to changed controller/population dynamics before moving
    # the already-useful actor weights.
    p.add_argument("--actor-warmup-ticks", type=int, default=32768)

    # Fixed, deterministic hard seeds mined by evaluate_hybrid_mappo.py.
    p.add_argument("--hard-seeds-file", type=str, default="")
    p.add_argument("--hard-seed-prob", type=float, default=0.10)

    p.add_argument(
        "--controller-profile",
        choices=available_profiles(),
        default="v4_robust7",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch-threads", type=int, default=1)

    p.add_argument("--resume", type=str, default="")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/hybrid_mappo_v4_latest.pt",
    )
    p.add_argument(
        "--snapshot-template",
        type=str,
        default="checkpoints/hybrid_mappo_v4.pt",
    )
    p.add_argument("--snapshot-every", type=int, default=50_000)
    return p.parse_args()


def main():
    args = parse_args()

    if not (0.0 <= args.hard_seed_prob <= 1.0):
        raise ValueError("--hard-seed-prob must be in [0,1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(args.torch_threads)))

    pygame.init()
    device = torch.device(args.device)
    controller_config = get_controller_config(args.controller_profile)

    print(f"Using device: {device}")
    print(f"Observation dimension: {OBS_DIM} (unchanged)")
    print(f"Controller profile: {controller_config.name}")

    policy = HybridMAPPO(
        OBS_DIM,
        hidden_dim=args.hidden_dim,
        critic_embed_dim=args.critic_embed_dim,
    ).to(device)

    actor_params = (
        list(policy.actor_trunk.parameters())
        + list(policy.actor_mean.parameters())
        + [policy.log_std]
    )
    critic_params = (
        list(policy.critic_agent_encoder.parameters())
        + list(policy.critic_head.parameters())
    )
    actor_optimizer = Adam(actor_params, lr=args.actor_lr, eps=1e-5)
    critic_optimizer = Adam(critic_params, lr=args.critic_lr, eps=1e-5)

    total_ticks = 0
    update_idx = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
        if ckpt_obs_dim != OBS_DIM:
            raise RuntimeError(
                f"Checkpoint obs_dim={ckpt_obs_dim}, current OBS_DIM={OBS_DIM}."
            )

        policy.load_state_dict(ckpt["model_state"])

        # Optimizer moments are reusable because architecture is unchanged, but
        # the NEW conservative learning rates below always override old values.
        try:
            if "actor_optimizer_state" in ckpt:
                actor_optimizer.load_state_dict(ckpt["actor_optimizer_state"])
            if "critic_optimizer_state" in ckpt:
                critic_optimizer.load_state_dict(ckpt["critic_optimizer_state"])
        except (ValueError, KeyError) as exc:
            print(f"Warning: optimizer state not restored: {exc}")

        for group in actor_optimizer.param_groups:
            group["lr"] = float(args.actor_lr)
        for group in critic_optimizer.param_groups:
            group["lr"] = float(args.critic_lr)

        total_ticks = int(ckpt.get("total_ticks", 0))
        update_idx = int(ckpt.get("update_idx", 0))
        print(f"Resumed {args.resume} at recorded tick {total_ticks:,}")

    start_ticks = total_ticks
    target_ticks = (
        start_ticks + int(args.additional_ticks)
        if args.additional_ticks > 0
        else int(args.total_ticks)
    )
    if target_ticks <= start_ticks:
        raise ValueError(
            f"Target ticks {target_ticks:,} <= start {start_ticks:,}. "
            "When resuming, use --additional-ticks, e.g. 200000."
        )

    hard_seeds = parse_seed_file(args.hard_seeds_file)
    print(
        f"Hard seeds: {len(hard_seeds)} | replay probability: "
        f"{args.hard_seed_prob:.0%}"
    )

    sampler = SeedSampler(
        seed=args.seed + start_ticks,
        hard_seeds=hard_seeds,
        hard_probability=args.hard_seed_prob,
    )

    sim_seed, episode_source = sampler.sample()
    sim, state = create_sim(sim_seed)
    coordinator = SpawnCoordinator(controller_config)

    episode_births = 0
    episode_deaths = 0
    episode_max_pop = int(state["num_agents"])

    recent_scores: List[float] = []
    recent_times: List[float] = []
    recent_sources: List[str] = []

    actor_unfreeze_tick = start_ticks + max(0, int(args.actor_warmup_ticks))
    next_snapshot_tick: Optional[int]
    if args.snapshot_every > 0:
        next_snapshot_tick = start_ticks + int(args.snapshot_every)
    else:
        next_snapshot_tick = None

    pbar = tqdm(
        total=target_ticks,
        initial=min(total_ticks, target_ticks),
        desc="Hybrid MAPPO V4",
        unit="tick",
        dynamic_ncols=True,
    )

    try:
        while total_ticks < target_ticks:
            records: List[TeamTransition] = []

            while len(records) < args.rollout_ticks and total_ticks < target_ticks:
                agents, obs_np, _ = encode_state(state)

                if not agents:
                    sim_seed, episode_source = sampler.sample()
                    sim, state = create_sim(sim_seed)
                    coordinator.reset()
                    episode_births = 0
                    episode_deaths = 0
                    episode_max_pop = int(state["num_agents"])
                    continue

                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                raw_t, old_logp_t, _ = policy.act_actor(
                    obs_t, deterministic=False
                )
                raw_np = raw_t.cpu().numpy().astype(np.float32)
                old_logp_np = old_logp_t.cpu().numpy().astype(np.float32)

                with torch.no_grad():
                    value = float(policy.team_value(obs_t).item())

                spawn_id = coordinator.choose(state)
                actions = []
                actor_used = np.zeros(len(agents), dtype=np.bool_)

                for i, agent in enumerate(agents):
                    req, used, _mode = compose_action(
                        agent_state=agent,
                        state=state,
                        raw_residual=raw_np[i],
                        spawn_agent=(
                            spawn_id is not None
                            and int(agent["agent_id"]) == int(spawn_id)
                        ),
                    )
                    actions.append((int(agent["agent_id"]), req))
                    actor_used[i] = used

                prev_state = state
                next_state = sim.step(actions)
                reward, diag = compute_team_reward(prev_state, next_state)

                episode_births += int(diag["births"])
                episode_deaths += int(diag["deaths"])
                episode_max_pop = max(
                    episode_max_pop, int(next_state["num_agents"])
                )

                done = (
                    int(next_state["num_agents"]) == 0
                    or float(next_state["sim_time"]) >= 3000.0
                )

                if done:
                    next_value = 0.0
                else:
                    _, next_obs_np, _ = encode_state(next_state)
                    next_obs_t = torch.as_tensor(
                        next_obs_np, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        next_value = float(policy.team_value(next_obs_t).item())

                records.append(
                    TeamTransition(
                        team_obs=obs_np.copy(),
                        actor_obs=obs_np.copy(),
                        raw_residual=raw_np.copy(),
                        old_log_prob=old_logp_np.copy(),
                        actor_used=actor_used.copy(),
                        value=value,
                        reward=reward,
                        done=done,
                        next_value=next_value,
                    )
                )

                state = next_state
                total_ticks += 1
                pbar.update(1)

                if done:
                    final_score = float(state["score"])
                    final_time = float(state["sim_time"])

                    recent_scores.append(final_score)
                    recent_times.append(final_time)
                    recent_sources.append(episode_source)
                    recent_scores = recent_scores[-30:]
                    recent_times = recent_times[-30:]
                    recent_sources = recent_sources[-30:]

                    tqdm.write(
                        f"EP seed={sim_seed} source={episode_source} "
                        f"score={final_score:.2f} time={final_time:.1f}s "
                        f"births={episode_births} deaths={episode_deaths} "
                        f"max_pop={episode_max_pop}"
                    )

                    sim_seed, episode_source = sampler.sample()
                    sim, state = create_sim(sim_seed)
                    coordinator.reset()
                    episode_births = 0
                    episode_deaths = 0
                    episode_max_pop = int(state["num_agents"])

                postfix = {
                    "src": episode_source,
                    "ep_score": f"{float(state.get('score', 0.0)):.0f}",
                    "ep_time": f"{float(state.get('sim_time', 0.0)):.0f}s",
                    "pop": int(state.get("num_agents", 0)),
                    "births": episode_births,
                    "actor": "warmup" if total_ticks < actor_unfreeze_tick else "on",
                }
                if recent_scores:
                    postfix["train_med"] = f"{np.median(recent_scores):.0f}"
                pbar.set_postfix(**postfix)

            if not records:
                break

            advantages, returns = compute_gae(records, args.gamma, args.gae_lambda)
            update_actor = total_ticks >= actor_unfreeze_tick
            metrics = ppo_update(
                policy=policy,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                records=records,
                advantages=advantages,
                returns=returns,
                device=device,
                actor_batch_size=args.actor_batch_size,
                critic_batch_size=args.critic_batch_size,
                epochs=args.ppo_epochs,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                update_actor=update_actor,
            )

            update_idx += 1
            if recent_scores:
                train_mean = float(np.mean(recent_scores))
                train_med = float(np.median(recent_scores))
                train_p10 = float(np.percentile(recent_scores, 10))
                train_stats = (
                    f"train_n={len(recent_scores)} p10={train_p10:.1f} "
                    f"med={train_med:.1f} mean={train_mean:.1f}"
                )
            else:
                train_stats = (
                    f"train_n=0 live_score={float(state.get('score', 0.0)):.1f} "
                    f"live_time={float(state.get('sim_time', 0.0)):.1f}s"
                )

            tqdm.write(
                f"UPDATE {update_idx:04d} | ticks={total_ticks:,}/{target_ticks:,} "
                f"| rollout={len(records)} | {train_stats} | "
                f"actor_on={update_actor} actor={metrics['actor_loss']:.4f} "
                f"critic={metrics['critic_loss']:.4f} "
                f"entropy={metrics['entropy']:.3f} "
                f"kl={metrics['approx_kl']:.5f} "
                f"clip={metrics['clip_fraction']:.3f}"
            )

            save_checkpoint(
                Path(args.checkpoint),
                policy,
                actor_optimizer,
                critic_optimizer,
                total_ticks,
                update_idx,
                args,
                controller_config,
            )

            if next_snapshot_tick is not None and total_ticks >= next_snapshot_tick:
                snap = snapshot_path(args.snapshot_template, total_ticks)
                save_checkpoint(
                    snap,
                    policy,
                    actor_optimizer,
                    critic_optimizer,
                    total_ticks,
                    update_idx,
                    args,
                    controller_config,
                )
                tqdm.write(f"SNAPSHOT -> {snap}")
                while next_snapshot_tick is not None and next_snapshot_tick <= total_ticks:
                    next_snapshot_tick += int(args.snapshot_every)

    finally:
        pbar.close()
        pygame.quit()

    # Always save exact final state too.
    save_checkpoint(
        Path(args.checkpoint),
        policy,
        actor_optimizer,
        critic_optimizer,
        total_ticks,
        update_idx,
        args,
        controller_config,
    )
    final_snap = snapshot_path(args.snapshot_template, total_ticks)
    save_checkpoint(
        final_snap,
        policy,
        actor_optimizer,
        critic_optimizer,
        total_ticks,
        update_idx,
        args,
        controller_config,
    )
    print(f"Saved latest model to {args.checkpoint}")
    print(f"Saved final snapshot to {final_snap}")
    print("Important: select the deployment checkpoint with deterministic DEV evaluation, not training logs.")


if __name__ == "__main__":
    main()
