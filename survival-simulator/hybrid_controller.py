from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.utils.DTOs import ActionRequest


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _objects(agent_state: Dict, tag: str) -> List[Dict]:
    objs = [o for o in (agent_state.get("observations", []) or []) if o.get("type") == tag]
    objs.sort(key=lambda o: float(o.get("distance", 1e9)))
    return objs


def nearest_object(agent_state: Dict, tag: str) -> Optional[Dict]:
    objs = _objects(agent_state, tag)
    return objs[0] if objs else None


def local_count(agent_state: Dict, tag: str, within: float) -> int:
    return sum(
        1
        for o in (agent_state.get("observations", []) or [])
        if o.get("type") == tag and float(o.get("distance", 1e9)) <= within
    )


def nearest_edge(agent_state: Dict) -> Optional[Tuple[float, float]]:
    best = None
    for obj in agent_state.get("observations", []) or []:
        if obj.get("type") != "Edge":
            continue
        coords = obj.get("coords")
        if not coords or len(coords) != 2:
            continue
        (x1, y1), (x2, y2) = coords
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            px, py = x1, y1
        else:
            t = -(x1 * dx + y1 * dy) / denom
            t = max(0.0, min(1.0, t))
            px, py = x1 + t * dx, y1 + t * dy
        dist = math.hypot(px, py)
        angle = math.atan2(py, px)
        if best is None or dist < best[0]:
            best = (dist, angle)
    return best


@dataclass
class BaseAction:
    move_distance: float
    move_direction: float
    turn_angle: float
    mode: str
    hard_override: bool = False


def _forage_distance(agent_state: Dict) -> float:
    energy = float(agent_state["energy"])
    max_energy = max(float(agent_state["max_energy"]), 1e-6)
    ratio = energy / max_energy

    # Movement in this simulator is per tick, not per second. Asking for 10
    # every tick costs 5 energy/s before passive drain. Deliberately stay well
    # below walking speed except during predator escape.
    if energy < 90.0 or ratio < 0.22:
        return 4.5
    if ratio < 0.45:
        return 4.0
    if ratio < 0.75:
        return 3.4
    return 3.0


def heuristic_base_action(agent_state: Dict, state: Dict) -> BaseAction:
    """Strong deterministic prior. PPO learns residual corrections around this.

    Priority:
      1) flee predators;
      2) collect visible/smelled fruit;
      3) move to a tree and camp near it;
      4) cheap scan/explore cycle;
      5) avoid wasting motion into nearby edges.
    """
    predator = nearest_object(agent_state, "Predator")
    fruit = nearest_object(agent_state, "Fruit")
    tree = nearest_object(agent_state, "Tree")
    edge = nearest_edge(agent_state)

    speed = float(agent_state["speed"])
    sprint_speed = float(agent_state["sprint_speed"])

    # 1) Safety override. Agent speed 20 sprint vs predator 15 default sprint,
    # so use the expensive sprint only when danger exists. Face the predator
    # while moving away: predator logic is less direct when the herbivore is
    # looking toward it, while movement direction itself can be independent.
    if predator is not None:
        pd = float(predator["distance"])
        pa = float(predator["angle"])
        escape_dir = wrap_angle(pa + math.pi)
        if pd < 120.0:
            distance = sprint_speed
        elif pd < 180.0:
            distance = max(speed, speed + 0.65 * (sprint_speed - speed))
        else:
            distance = max(speed, speed + 0.30 * (sprint_speed - speed))
        turn = clip(pa, -0.70, 0.70)
        return BaseAction(distance, escape_dir, turn, "flee", hard_override=True)

    # 2) Fruit is immediate energy. Approach economically; never sprint for food.
    if fruit is not None:
        fd = float(fruit["distance"])
        fa = float(fruit["angle"])
        distance = min(_forage_distance(agent_state), max(0.0, fd))
        turn = clip(0.45 * fa, -0.45, 0.45)
        base = BaseAction(distance, wrap_angle(fa), turn, "fruit", hard_override=False)
    # 3) Trees are future food hotspots. Fruit appears close around mature trees,
    # so once close, stop paying movement cost and scan/camp.
    elif tree is not None:
        td = float(tree["distance"])
        ta = float(tree["angle"])
        if td > 38.0:
            distance = min(3.0, max(0.0, td - 28.0))
            turn = clip(0.35 * ta, -0.35, 0.35)
            base = BaseAction(distance, wrap_angle(ta), turn, "tree_approach", hard_override=False)
        else:
            # Slow scan while remaining near the tree. The turn cost is much
            # cheaper than walking at full speed and exposes the vision cone.
            scan_sign = 1.0 if int(agent_state["agent_id"]) % 2 == 0 else -1.0
            base = BaseAction(0.20, scan_sign * math.pi / 2.0, scan_sign * 0.24, "tree_camp", False)
    else:
        # 4) No resource currently perceived. Alternate cheap 360-degree scans
        # and low-cost displacement. The phase is deterministic and different
        # across agents, so the species does not move as a single clump.
        sim_time = float(state.get("sim_time", 0.0))
        agent_id = int(agent_state["agent_id"])
        phase = (sim_time + 0.73 * agent_id) % 5.0
        sign = 1.0 if agent_id % 2 == 0 else -1.0
        if phase < 2.0:
            base = BaseAction(0.0, 0.0, sign * 0.34, "scan", False)
        else:
            wobble = 0.18 * math.sin(0.7 * sim_time + 1.3 * agent_id)
            base = BaseAction(1.8, wobble, sign * 0.07, "explore", False)

    # 5) Edge protection. The simulator does collision redirection itself, but
    # repeatedly pushing into an obstacle still wastes energy. Only override a
    # non-predator action when the edge is very close and roughly ahead.
    if edge is not None:
        edge_dist, edge_angle = edge
        if edge_dist < 18.0 and abs(edge_angle) < math.pi * 0.55:
            away = wrap_angle(edge_angle + math.pi)
            return BaseAction(min(base.move_distance, 2.0), away, clip(away * 0.25, -0.4, 0.4), "edge_avoid", True)

    return base


class SpawnCoordinator:
    """Coordinated species-level reproduction.

    Spawning is intentionally not learned in V2. It is rare, binary and has a
    delayed benefit across generations, which was the hardest credit-assignment
    failure in V1. This coordinator creates replacement generations without
    allowing population explosion. At most one birth request is emitted at a
    time and births are spaced by a short cooldown.
    """

    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_spawn_time = -1e9

    def reset(self):
        self.last_spawn_time = -1e9

    @staticmethod
    def _trait_quality(agent: Dict) -> float:
        return (
            float(agent["speed"]) / 20.0
            + float(agent["sprint_speed"]) / 40.0
            + float(agent["hearing_radius"]) / 100.0
            + float(agent["vision_range"]) / 400.0
            + float(agent["vision_angle"]) / (math.pi / 2.0)
            + float(agent["max_energy"]) / 1000.0
        ) / 6.0

    @staticmethod
    def _support(agent: Dict) -> Tuple[bool, float, float, float, int]:
        fruit = nearest_object(agent, "Fruit")
        tree = nearest_object(agent, "Tree")
        predator = nearest_object(agent, "Predator")

        fruit_d = float(fruit["distance"]) if fruit is not None else 1e9
        tree_d = float(tree["distance"]) if tree is not None else 1e9
        predator_d = float(predator["distance"]) if predator is not None else 1e9
        crowd = local_count(agent, "Agent", 80.0)

        food_support = fruit_d <= 110.0 or tree_d <= 75.0
        return food_support, fruit_d, tree_d, predator_d, crowd

    def choose(self, state: Dict) -> Optional[int]:
        sim_time = float(state.get("sim_time", 0.0))
        if sim_time - self.last_spawn_time < self.cooldown_seconds:
            return None

        agents = state.get("observations", []) or []
        pop = int(state.get("num_agents", len(agents)))
        if pop <= 0 or pop >= 7:
            return None

        candidates: List[Tuple[float, int]] = []

        for agent in agents:
            energy = float(agent["energy"])
            age = float(agent["age"])
            agent_id = int(agent["agent_id"])

            if energy <= 100.0:
                continue

            food_support, fruit_d, tree_d, predator_d, crowd = self._support(agent)
            # Do not reproduce into immediate danger.
            if predator_d < 180.0:
                continue

            trait = self._trait_quality(agent)
            eligible = False
            base_priority = 0.0

            # Emergency recovery. At very low population we accept more risk,
            # but still avoid leaving a barely-alive parent unless food exists.
            if pop <= 2:
                eligible = energy >= 150.0 and (food_support or energy >= 260.0)
                base_priority = 1000.0
            elif pop == 3:
                eligible = energy >= 175.0 and (food_support or energy >= 280.0)
                base_priority = 900.0
            elif pop == 4:
                eligible = energy >= 200.0 and food_support
                base_priority = 800.0
            elif pop == 5:
                # Replacement before the earliest possible aging penalty at 60s.
                replacement = age >= 45.0 and energy >= 220.0 and food_support
                # Or create one buffer child only when a parent is very healthy.
                buffer_growth = energy >= 320.0 and food_support and crowd <= 2
                eligible = replacement or buffer_growth
                base_priority = 700.0 if replacement else 600.0
            elif pop == 6:
                # Briefly allow seven only as an old-agent replacement generation.
                eligible = age >= 55.0 and energy >= 240.0 and food_support and crowd <= 2
                base_priority = 500.0

            if not eligible:
                continue

            # Favor old, energetic, high-trait parents with nearby actual fruit
            # and less local crowding. This is only ranking among already-safe
            # candidates; it is not an extra birth incentive.
            score = base_priority
            score += 1.2 * age
            score += 0.30 * energy
            score += 80.0 * trait
            score -= 12.0 * crowd
            if fruit_d <= 80.0:
                score += 40.0
            elif tree_d <= 60.0:
                score += 20.0

            candidates.append((score, agent_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        chosen = candidates[0][1]
        self.last_spawn_time = sim_time
        return chosen


def compose_action(
    agent_state: Dict,
    state: Dict,
    raw_residual: np.ndarray,
    spawn_agent: bool,
) -> Tuple[ActionRequest, bool, str]:
    """Combine the strong heuristic prior with a learned residual.

    Returns (request, actor_used, mode). Safety overrides ignore the residual and
    are excluded from actor PPO loss, preventing random exploration from ruining
    predator escape or very-close edge avoidance.
    """
    base = heuristic_base_action(agent_state, state)
    raw = np.asarray(raw_residual, dtype=np.float32)

    if base.hard_override:
        move_distance = base.move_distance
        move_direction = base.move_direction
        turn_angle = base.turn_angle
        actor_used = False
    else:
        # Residual ranges are deliberately modest because the heuristic already
        # encodes high-value survival mechanics. PPO can still shift its latent
        # mean arbitrarily if a larger correction is consistently useful.
        distance_delta = float(np.tanh(raw[0]) * 2.0)
        direction_delta = float(np.tanh(raw[1]) * (math.pi / 3.0))
        turn_delta = float(np.tanh(raw[2]) * 0.35)

        move_distance = base.move_distance + distance_delta
        move_direction = wrap_angle(base.move_direction + direction_delta)
        turn_angle = base.turn_angle + turn_delta
        actor_used = True

    sprint_speed = float(agent_state["sprint_speed"])
    move_distance = clip(move_distance, 0.0, sprint_speed)
    move_direction = wrap_angle(move_direction)
    turn_angle = clip(turn_angle, -math.pi / 2.0, math.pi / 2.0)

    req = ActionRequest(
        agent_id=int(agent_state["agent_id"]),
        move_distance=float(move_distance),
        move_direction=float(move_direction),
        turn_angle=float(turn_angle),
        spawn_agent=bool(spawn_agent and float(agent_state["energy"]) > 100.0),
    )
    return req, actor_used, base.mode
