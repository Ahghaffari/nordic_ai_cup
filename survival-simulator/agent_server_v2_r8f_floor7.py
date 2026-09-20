from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import Body, FastAPI

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = None
for candidate in (HERE, *HERE.parents):
    if (candidate / "hybrid_controller.py").exists() and (candidate / "src").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.DTOs import StepResponse
from v2_encoder import OBS_DIM, encode_state
from mappo_policy import HybridMAPPO
from hybrid_controller_r8f_floor7 import make_controller


HOST = "0.0.0.0"
PORT = int(os.environ.get("SURVIVAL_PORT", "9052"))
CHECKPOINT_PATH = os.getenv("MODEL_PATH", "checkpoints/hybrid_mappo_best.pt")
# CHECKPOINT_PATH = os.getenv("MODEL_PATH", "checkpoints/hybrid_mappo_adaptive_ft_latest_tick075776.pt")
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

# Spawn controller remains the Floor7 family. Movement can be rolled back to
# original V2 with MOVEMENT_MODE=v2 without changing files.
CONTROLLER_MODE = os.getenv("CONTROLLER_MODE", "r8f_floor7").strip().lower()
MOVEMENT_MODE = os.getenv("MOVEMENT_MODE", "adaptive").strip().lower()

CTRL_DEBUG = os.getenv("CTRL_DEBUG", os.getenv("SPAWN_DEBUG", "0")).strip().lower() in {
    "1", "true", "yes", "on"
}
DEBUG_INTERVAL = float(os.getenv("DEBUG_INTERVAL", "10"))

print(f"Loading model: {CHECKPOINT_PATH}")
print(f"Spawn controller: {CONTROLLER_MODE}")
print(f"Movement mode: {MOVEMENT_MODE}")
print(f"Device: {DEVICE}")
print(f"Controller debug: {CTRL_DEBUG}")

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
ckpt_obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
if ckpt_obs_dim != OBS_DIM:
    raise RuntimeError(f"Checkpoint obs_dim={ckpt_obs_dim}, expected {OBS_DIM}")

policy = HybridMAPPO(
    OBS_DIM,
    int(ckpt.get("hidden_dim", 256)),
    int(ckpt.get("critic_embed_dim", 128)),
).to(DEVICE)
policy.load_state_dict(ckpt["model_state"])
policy.eval()
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

controller = make_controller(
    spawn_mode=CONTROLLER_MODE,
    movement_mode=MOVEMENT_MODE,
)

last_sim_time = -1.0
last_population: Optional[int] = None
last_debug_time = -1e9

app = FastAPI(title="Nordic AI Cup V2 Adaptive Floor7")


class PredictRequest(StepResponse):
    sim_time: float = 0.0
    n_agents: Optional[int] = None


OBS_TYPES = {t.lower(): t for t in ("Fruit", "Tree", "Predator", "Agent", "Edge")}


@app.get("/")
def index():
    return {
        "message": "V2 adaptive Floor7 agent",
        "model": os.path.basename(CHECKPOINT_PATH),
        "spawn_controller": CONTROLLER_MODE,
        "movement": MOVEMENT_MODE,
        "device": str(DEVICE),
        "debug": CTRL_DEBUG,
    }


@app.post("/predict")
def predict(step: PredictRequest = Body(...)):
    global last_sim_time, last_population, last_debug_time

    sim_time = float(step.sim_time)
    if last_sim_time >= 0.0 and sim_time < last_sim_time:
        controller.reset()
        last_population = None
        last_debug_time = -1e9
    last_sim_time = sim_time

    if not step.agent_status:
        return {"actions": []}

    agents = [
        a.model_dump() if hasattr(a, "model_dump") else a.dict()
        for a in step.agent_status
    ]

    # Normalize observation type spelling exactly as the V2 encoder expects.
    for a in agents:
        for o in a.get("observations", []):
            o["type"] = OBS_TYPES.get(str(o.get("type", "")).lower(), o.get("type"))

    pop = int(step.n_agents if step.n_agents is not None else len(agents))
    state = {
        "observations": agents,
        "num_agents": pop,
        "score": float(step.score),
        "sim_time": sim_time,
    }

    # Must happen before spawn/movement so hidden-senescence inference uses the
    # energy change caused by the PREVIOUS tick's recorded action.
    controller.prepare_step(state)

    if CTRL_DEBUG and pop != last_population:
        ages = [round(float(a.get("age", 0.0)), 1) for a in agents]
        energies = [round(float(a.get("energy", 0.0)), 1) for a in agents]
        print(
            f"[POP] t={sim_time:.1f} pop={pop} ages={ages} energies={energies}",
            flush=True,
        )
    last_population = pop

    encoded_agents, obs_np, _ = encode_state(state)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        raw_t, _, _ = policy.act_actor(obs_t, deterministic=True)
    raw_np = raw_t.cpu().numpy().astype(np.float32)

    spawn_id = controller.choose_spawn(state)

    if CTRL_DEBUG and spawn_id is not None:
        parent = next((a for a in agents if int(a["agent_id"]) == int(spawn_id)), None)
        spawn_obj = controller.spawn
        reason = getattr(spawn_obj, "last_reason", "v2")
        target_cap = getattr(spawn_obj, "last_target_cap", 7)
        if parent is not None:
            sen = controller.memory.is_senescent(int(spawn_id))
            print(
                f"[SPAWN] t={sim_time:.1f} pop={pop} target={target_cap} "
                f"reason={reason} parent={spawn_id} age={float(parent['age']):.1f} "
                f"energy={float(parent['energy']):.1f} senescent={int(sen)}",
                flush=True,
            )

    actions = []
    for i, agent in enumerate(encoded_agents):
        req, _, _ = controller.compose_action(
            agent,
            state,
            raw_np[i],
            spawn_agent=(
                spawn_id is not None
                and int(agent["agent_id"]) == int(spawn_id)
            ),
        )
        actions.append(req.model_dump() if hasattr(req, "model_dump") else req.dict())

    if CTRL_DEBUG and sim_time - last_debug_time >= DEBUG_INTERVAL:
        snap = controller.debug_snapshot(state)
        print(
            f"[STAT] t={sim_time:.1f} pop={pop} "
            f"target={snap.get('target_cap')} "
            f"meanE={snap.get('mean_energy')} p25E={snap.get('p25_energy')} "
            f"meanEr={snap.get('mean_energy_ratio', 'na')} "
            f"spawnable={snap.get('spawnable_fraction', 'na')} "
            f"young={snap.get('young', 'na')} senior={snap.get('senior', 'na')} "
            f"senescent={snap.get('senescent')} "
            f"fruitSeen={snap.get('fruit_seen_fraction', 'na')} "
            f"treeSeen={snap.get('tree_seen_fraction', 'na')} "
            f"deaths12={snap.get('recent_deaths_12s', 'na')} "
            f"modes={snap.get('modes', {})}",
            flush=True,
        )
        last_debug_time = sim_time

    return {"actions": actions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
