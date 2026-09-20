from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# V2/V3 observation architecture (UNCHANGED for checkpoint compatibility)
# -----------------------------------------------------------------------------
# self / traits:        10
# species / global:      9
# biome one-hot:         5
# perceived counts:      4
# nearest fruits:     4*4 = 16
# nearest trees:      3*4 = 12
# nearest predators:  3*6 = 18
# nearest agents:     4*6 = 24
# nearest edges:      2*6 = 12
# TOTAL                    110
# -----------------------------------------------------------------------------

OBS_DIM = 110

BIOMES = ["forest", "grassland", "swamp", "desert", "river"]
BIOME_TO_INDEX = {name: i for i, name in enumerate(BIOMES)}

MAX_SPEED = 20.0
MAX_SPRINT_SPEED = 40.0
MAX_HEARING_RADIUS = 100.0
MAX_VISION_RANGE = 400.0
MAX_VISION_ANGLE = math.pi / 2.0
MAX_MAX_ENERGY = 1000.0
MAX_AGE_NORM = 180.0
MAX_POP_NORM = 20.0
MAX_SIM_TIME = 3000.0


def build_global_context(state: Dict) -> Dict[str, float]:
    agents = list(state.get("observations", []) or [])

    if agents:
        energy_ratios = np.asarray(
            [
                max(0.0, float(a["energy"])) / max(float(a["max_energy"]), 1e-6)
                for a in agents
            ],
            dtype=np.float32,
        )
        ages = np.asarray([float(a["age"]) for a in agents], dtype=np.float32)
        energies = np.asarray([float(a["energy"]) for a in agents], dtype=np.float32)

        mean_energy_ratio = float(energy_ratios.mean())
        min_energy_ratio = float(energy_ratios.min())
        max_energy_ratio = float(energy_ratios.max())
        mean_age = float(ages.mean())
        max_age = float(ages.max())
        spawnable_fraction = float(np.mean(energies > 100.0))
        old_fraction = float(np.mean(ages >= 50.0))
    else:
        mean_energy_ratio = 0.0
        min_energy_ratio = 0.0
        max_energy_ratio = 0.0
        mean_age = 0.0
        max_age = 0.0
        spawnable_fraction = 0.0
        old_fraction = 0.0

    return {
        "sim_time": float(state.get("sim_time", 0.0)),
        "population": float(state.get("num_agents", len(agents))),
        "mean_energy_ratio": mean_energy_ratio,
        "min_energy_ratio": min_energy_ratio,
        "max_energy_ratio": max_energy_ratio,
        "mean_age": mean_age,
        "max_age": max_age,
        "spawnable_fraction": spawnable_fraction,
        "old_fraction": old_fraction,
    }


def _object_slots(
    observations: Iterable[Dict],
    tag: str,
    k: int,
    distance_scale: float,
    include_rel_dir: bool = False,
) -> List[float]:
    objs = [o for o in observations if o.get("type") == tag]
    objs.sort(key=lambda o: float(o.get("distance", 1e9)))

    out: List[float] = []
    for i in range(k):
        if i >= len(objs):
            out.extend([0.0, 1.0, 0.0, 0.0])
            if include_rel_dir:
                out.extend([0.0, 0.0])
            continue

        obj = objs[i]
        distance = min(
            max(float(obj.get("distance", distance_scale)), 0.0) / distance_scale,
            1.0,
        )
        angle = float(obj.get("angle", 0.0))
        out.extend([1.0, distance, math.sin(angle), math.cos(angle)])

        if include_rel_dir:
            rel_dir = float(obj.get("rel_dir", 0.0))
            out.extend([math.sin(rel_dir), math.cos(rel_dir)])

    return out


def _closest_point_on_segment(x1: float, y1: float, x2: float, y2: float):
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return x1, y1
    t = -(x1 * dx + y1 * dy) / denom
    t = max(0.0, min(1.0, t))
    return x1 + t * dx, y1 + t * dy


def _edge_slots(observations: Iterable[Dict], k: int, distance_scale: float) -> List[float]:
    encoded: List[Tuple[float, float, float]] = []
    for obj in observations:
        if obj.get("type") != "Edge":
            continue
        coords = obj.get("coords")
        if not coords or len(coords) != 2:
            continue

        (x1, y1), (x2, y2) = coords
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        px, py = _closest_point_on_segment(x1, y1, x2, y2)
        dist = math.hypot(px, py)
        angle = math.atan2(py, px)
        orientation = math.atan2(y2 - y1, x2 - x1)
        encoded.append((dist, angle, orientation))

    encoded.sort(key=lambda item: item[0])

    out: List[float] = []
    for i in range(k):
        if i >= len(encoded):
            out.extend([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            continue

        dist, angle, orientation = encoded[i]
        out.extend(
            [
                1.0,
                min(max(dist / distance_scale, 0.0), 1.0),
                math.sin(angle),
                math.cos(angle),
                math.sin(orientation),
                math.cos(orientation),
            ]
        )
    return out


def encode_agent(agent_state: Dict, global_context: Dict[str, float]) -> np.ndarray:
    observations = list(agent_state.get("observations", []) or [])

    energy = float(agent_state["energy"])
    max_energy = max(float(agent_state["max_energy"]), 1e-6)
    hearing_radius = max(float(agent_state["hearing_radius"]), 1.0)
    vision_range = max(float(agent_state["vision_range"]), 1.0)
    distance_scale = max(hearing_radius, vision_range, 1.0)

    features: List[float] = [
        # Individual status / traits: 10
        np.clip(energy / max_energy, 0.0, 1.5),
        np.clip(energy / 500.0, 0.0, 2.0),
        np.clip((energy - 100.0) / 400.0, -0.5, 2.0),
        np.clip(float(agent_state["age"]) / MAX_AGE_NORM, 0.0, 1.5),
        np.clip(float(agent_state["speed"]) / MAX_SPEED, 0.0, 1.5),
        np.clip(float(agent_state["sprint_speed"]) / MAX_SPRINT_SPEED, 0.0, 1.5),
        np.clip(hearing_radius / MAX_HEARING_RADIUS, 0.0, 1.5),
        np.clip(float(agent_state["vision_angle"]) / MAX_VISION_ANGLE, 0.0, 1.5),
        np.clip(vision_range / MAX_VISION_RANGE, 0.0, 1.5),
        np.clip(max_energy / MAX_MAX_ENERGY, 0.0, 1.5),

        # Species / global context: 9
        np.clip(global_context["sim_time"] / MAX_SIM_TIME, 0.0, 1.0),
        np.clip(global_context["population"] / MAX_POP_NORM, 0.0, 1.5),
        np.clip(global_context["mean_energy_ratio"], 0.0, 1.5),
        np.clip(global_context["min_energy_ratio"], 0.0, 1.5),
        np.clip(global_context["max_energy_ratio"], 0.0, 1.5),
        np.clip(global_context["mean_age"] / MAX_AGE_NORM, 0.0, 1.5),
        np.clip(global_context["max_age"] / MAX_AGE_NORM, 0.0, 1.5),
        np.clip(global_context["spawnable_fraction"], 0.0, 1.0),
        np.clip(global_context["old_fraction"], 0.0, 1.0),
    ]

    # Biome one-hot: 5
    biome_vec = [0.0] * len(BIOMES)
    idx = BIOME_TO_INDEX.get(str(agent_state.get("biome", "")))
    if idx is not None:
        biome_vec[idx] = 1.0
    features.extend(biome_vec)

    # Local object counts: 4
    for tag in ("Fruit", "Tree", "Predator", "Agent"):
        count = sum(1 for o in observations if o.get("type") == tag)
        features.append(min(float(count) / 5.0, 1.0))

    # Nearest-K geometry.
    features.extend(_object_slots(observations, "Fruit", 4, distance_scale, False))
    features.extend(_object_slots(observations, "Tree", 3, distance_scale, False))
    features.extend(_object_slots(observations, "Predator", 3, distance_scale, True))
    features.extend(_object_slots(observations, "Agent", 4, distance_scale, True))
    features.extend(_edge_slots(observations, 2, distance_scale))

    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (OBS_DIM,):
        raise RuntimeError(f"Encoder bug: expected {(OBS_DIM,)}, got {arr.shape}")
    return arr


def encode_state(state: Dict):
    agents = list(state.get("observations", []) or [])
    context = build_global_context(state)
    if not agents:
        return agents, np.empty((0, OBS_DIM), dtype=np.float32), context

    obs = np.stack([encode_agent(agent, context) for agent in agents]).astype(np.float32)
    return agents, obs, context
