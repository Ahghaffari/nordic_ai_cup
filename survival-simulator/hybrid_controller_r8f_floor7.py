# from __future__ import annotations

# """V2-R8F Floor7 Proactive reproduction controller.

# Movement remains EXACTLY the original V2 movement stack. This module only
# changes reproduction.

# Design:
#   * pop >= 8: never spawn.
#   * pop == 7: proactively build an 8th replacement buffer before a synchronized
#     late-game death wave. In particular, do not wait until the colony has almost
#     no replacement capacity left.
#   * pop <= 6: hard recovery. Original V2 gets first chance; if it does not spawn,
#     use progressively more aggressive recovery rules so the colony returns to 7
#     quickly instead of lingering at 5-6 and cascading to extinction.

# Interpretation:
#   * 7 = operating floor
#   * 8 = temporary replacement reserve

# The controller can only request a birth when the simulator allows it; if every
# remaining agent is too low-energy to reproduce, no spawn policy can force one.
# """

# import math
# from collections import Counter
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# from hybrid_controller import (
#     SpawnCoordinator as OriginalSpawnCoordinator,
#     compose_action,
#     local_count,
#     nearest_object,
# )


# class R8FFloor7SpawnCoordinator:
#     TARGET_FLOOR = 7
#     HARD_MAX_POPULATION = 8

#     YOUNG_AGE = 30.0
#     OLD_AGE = 45.0
#     SENIOR_AGE = 58.0

#     def __init__(
#         self,
#         cooldown_seconds: float = 2.0,
#         recovery_cooldown_seconds: float = 0.6,
#         same_parent_cooldown_seconds: float = 6.0,
#         hard_max_population: int = 8,
#     ):
#         self.original = OriginalSpawnCoordinator(cooldown_seconds=cooldown_seconds)
#         self.cooldown_seconds = float(cooldown_seconds)
#         self.recovery_cooldown_seconds = float(recovery_cooldown_seconds)
#         self.same_parent_cooldown_seconds = float(same_parent_cooldown_seconds)
#         self.hard_max_population = int(hard_max_population)

#         self.last_recovery_time = -1e9
#         self.last_parent_spawn_time: Dict[int, float] = {}
#         self.last_reason = "none"
#         self.reason_counts: Counter = Counter()

#     def reset(self):
#         self.original.reset()
#         self.last_recovery_time = -1e9
#         self.last_parent_spawn_time.clear()
#         self.last_reason = "none"
#         self.reason_counts.clear()

#     @staticmethod
#     def _support(agent: Dict) -> Tuple[bool, float, float, float, int]:
#         fruit = nearest_object(agent, "Fruit")
#         tree = nearest_object(agent, "Tree")
#         predator = nearest_object(agent, "Predator")

#         fruit_d = float(fruit["distance"]) if fruit is not None else 1e9
#         tree_d = float(tree["distance"]) if tree is not None else 1e9
#         predator_d = float(predator["distance"]) if predator is not None else 1e9
#         crowd = local_count(agent, "Agent", 80.0)

#         # Keep V2's notion of nearby food support.
#         food_support = fruit_d <= 110.0 or tree_d <= 75.0
#         return food_support, fruit_d, tree_d, predator_d, crowd

#     @staticmethod
#     def _trait_quality(agent: Dict) -> float:
#         return (
#             float(agent["speed"]) / 20.0
#             + float(agent["sprint_speed"]) / 40.0
#             + float(agent["hearing_radius"]) / 100.0
#             + float(agent["vision_range"]) / 400.0
#             + float(agent["vision_angle"]) / (math.pi / 2.0)
#             + float(agent["max_energy"]) / 1000.0
#         ) / 6.0

#     @classmethod
#     def _cohort_stats(cls, agents: List[Dict]) -> Dict[str, float]:
#         ages = np.asarray([float(a["age"]) for a in agents], dtype=np.float64)
#         energies = np.asarray([float(a["energy"]) for a in agents], dtype=np.float64)
#         if len(ages) == 0:
#             return {
#                 "young": 0.0,
#                 "old": 0.0,
#                 "senior": 0.0,
#                 "max_age": 0.0,
#                 "mean_age": 0.0,
#                 "mean_energy": 0.0,
#                 "min_energy": 0.0,
#             }
#         return {
#             "young": float(np.sum(ages < cls.YOUNG_AGE)),
#             "old": float(np.sum(ages >= cls.OLD_AGE)),
#             "senior": float(np.sum(ages >= cls.SENIOR_AGE)),
#             "max_age": float(np.max(ages)),
#             "mean_age": float(np.mean(ages)),
#             "mean_energy": float(np.mean(energies)),
#             "min_energy": float(np.min(energies)),
#         }

#     def _record(self, reason: str, sim_time: float, parent_id: int):
#         self.last_reason = reason
#         self.reason_counts[reason] += 1
#         self.last_parent_spawn_time[int(parent_id)] = float(sim_time)

#         # Synchronize with V2's global spawn cooldown after any extra birth.
#         self.original.last_spawn_time = float(sim_time)
#         if reason.startswith("recover"):
#             self.last_recovery_time = float(sim_time)

#     def _recent_parent_penalty(self, agent_id: int, sim_time: float) -> float:
#         last = self.last_parent_spawn_time.get(int(agent_id), -1e9)
#         elapsed = sim_time - float(last)
#         if elapsed >= self.same_parent_cooldown_seconds:
#             return 0.0
#         # Soft penalty only. Never hard-ban the only viable parent.
#         frac = max(0.0, 1.0 - elapsed / self.same_parent_cooldown_seconds)
#         return 140.0 * frac

#     def _recovery_candidate(self, state: Dict) -> Tuple[Optional[int], str]:
#         """Aggressive recovery whenever population is below 7."""
#         agents = state.get("observations", []) or []
#         pop = int(state.get("num_agents", len(agents)))
#         sim_time = float(state.get("sim_time", 0.0))

#         if pop <= 0 or pop >= self.TARGET_FLOOR:
#             return None, "none"

#         if sim_time - self.last_recovery_time < self.recovery_cooldown_seconds:
#             return None, "none"

#         # As population falls, survival risk dominates parent-conservation risk.
#         # Unsupported births still require somewhat more energy than births near
#         # fruit/tree support, but the gap shrinks in emergency states.
#         if pop == 6:
#             food_min_energy = 170.0
#             nofood_min_energy = 205.0
#             min_predator_d = 115.0
#             max_crowd = 5
#             reason = "recover6"
#             base_priority = 1000.0
#         elif pop == 5:
#             food_min_energy = 155.0
#             nofood_min_energy = 180.0
#             min_predator_d = 95.0
#             max_crowd = 6
#             reason = "recover5"
#             base_priority = 1200.0
#         elif pop == 4:
#             food_min_energy = 140.0
#             nofood_min_energy = 160.0
#             min_predator_d = 80.0
#             max_crowd = 8
#             reason = "recover4"
#             base_priority = 1400.0
#         else:  # pop <= 3: extinction emergency
#             food_min_energy = 115.0
#             nofood_min_energy = 125.0
#             min_predator_d = 65.0
#             max_crowd = 99
#             reason = "recover_critical"
#             base_priority = 1650.0

#         candidates: List[Tuple[float, int]] = []
#         for agent in agents:
#             energy = float(agent["energy"])
#             age = float(agent["age"])
#             agent_id = int(agent["agent_id"])

#             # Simulator requires >100 energy to reproduce. Keep a tiny margin.
#             if energy <= 105.0:
#                 continue

#             food_support, fruit_d, tree_d, predator_d, crowd = self._support(agent)
#             if predator_d < min_predator_d or crowd > max_crowd:
#                 continue

#             required_energy = food_min_energy if food_support else nofood_min_energy
#             if energy < required_energy:
#                 continue

#             post_spawn_energy = energy - 100.0
#             score = base_priority
#             score += 1.85 * post_spawn_energy
#             score += 70.0 * self._trait_quality(agent)
#             score -= 8.0 * crowd
#             score -= self._recent_parent_penalty(agent_id, sim_time)

#             # Prefer a parent that can refill soon after paying the 100-energy cost.
#             if fruit_d <= 80.0:
#                 score += 80.0
#             elif tree_d <= 60.0:
#                 score += 35.0

#             # Prefer healthy adults; avoid charging seniors if a safer parent exists.
#             if 20.0 <= age <= 48.0:
#                 score += 35.0
#             elif age >= self.SENIOR_AGE:
#                 score -= 80.0
#             elif age >= 52.0:
#                 score -= 40.0

#             candidates.append((score, agent_id))

#         if not candidates:
#             return None, "none"

#         candidates.sort(reverse=True)
#         return int(candidates[0][1]), reason

#     def _buffer8_candidate(self, state: Dict) -> Tuple[Optional[int], str]:
#         """Proactive 7->8 reserve birth to stagger the age cohort.

#         The old R8/R8F logic often waited until the age skew was already severe.
#         Here we create the reserve earlier whenever the 7-agent colony lacks a
#         replacement cohort or has several agents approaching the high-drain ages.
#         """
#         agents = state.get("observations", []) or []
#         pop = int(state.get("num_agents", len(agents)))
#         if pop != self.TARGET_FLOOR or pop >= self.hard_max_population:
#             return None, "none"

#         stats = self._cohort_stats(agents)
#         young = int(stats["young"])
#         old = int(stats["old"])
#         senior = int(stats["senior"])
#         max_age = float(stats["max_age"])

#         replacement_risk = (
#             young <= 1
#             or senior >= 2
#             or old >= 4
#             or (max_age >= 55.0 and young <= 2)
#         )
#         if not replacement_risk:
#             return None, "none"

#         sim_time = float(state.get("sim_time", 0.0))
#         candidates: List[Tuple[float, int]] = []

#         for agent in agents:
#             energy = float(agent["energy"])
#             age = float(agent["age"])
#             agent_id = int(agent["agent_id"])

#             if age < 20.0:
#                 continue

#             food_support, fruit_d, tree_d, predator_d, crowd = self._support(agent)

#             # Earlier than old R8F, but still conservative enough not to turn
#             # every healthy 7-agent state into an expensive birth.
#             min_energy = 225.0 if food_support else 300.0
#             if energy < min_energy:
#                 continue
#             if predator_d < 140.0 or crowd > 4:
#                 continue

#             post_spawn_energy = energy - 100.0
#             score = 520.0
#             score += 1.65 * post_spawn_energy
#             score += 80.0 * self._trait_quality(agent)
#             score -= 9.0 * crowd
#             score -= self._recent_parent_penalty(agent_id, sim_time)

#             if 22.0 <= age <= 48.0:
#                 score += 40.0
#             elif age >= self.SENIOR_AGE:
#                 score -= 80.0
#             elif age >= 52.0:
#                 score -= 35.0

#             if fruit_d <= 80.0:
#                 score += 65.0
#             elif tree_d <= 60.0:
#                 score += 25.0

#             # Extra urgency when the replacement cohort is almost absent.
#             if young == 0:
#                 score += 70.0
#             elif young == 1:
#                 score += 35.0
#             if senior >= 2:
#                 score += 35.0

#             candidates.append((score, agent_id))

#         if not candidates:
#             return None, "none"

#         candidates.sort(reverse=True)
#         return int(candidates[0][1]), "buffer8_proactive"

#     def choose(self, state: Dict) -> Optional[int]:
#         self.last_reason = "none"
#         sim_time = float(state.get("sim_time", 0.0))
#         agents = state.get("observations", []) or []
#         pop = int(state.get("num_agents", len(agents)))

#         if pop <= 0 or pop >= self.hard_max_population:
#             return None

#         # 1) BELOW FLOOR: V2 gets first chance; otherwise hard recovery takes over.
#         if pop < self.TARGET_FLOOR:
#             original_choice = self.original.choose(state)
#             if original_choice is not None:
#                 chosen = int(original_choice)
#                 self._record("v2", sim_time, chosen)
#                 return chosen

#             # Do not inherit V2's 2-second cooldown here. Recovery uses the short
#             # 0.6-second cooldown so a 5-agent colony can attempt 5->6->7 quickly.
#             chosen, reason = self._recovery_candidate(state)
#             if chosen is not None:
#                 self._record(reason, sim_time, chosen)
#                 return int(chosen)
#             return None

#         # 2) AT FLOOR: proactively maintain an 8th replacement reserve when the
#         # age structure says a death wave could soon push the colony below 7.
#         if pop == self.TARGET_FLOOR:
#             if sim_time - float(self.original.last_spawn_time) < self.cooldown_seconds:
#                 return None

#             chosen, reason = self._buffer8_candidate(state)
#             if chosen is not None:
#                 self._record(reason, sim_time, chosen)
#                 return int(chosen)
#             return None

#         return None


# SpawnCoordinator = R8FFloor7SpawnCoordinator


# def make_spawn_coordinator(mode: str = "r8f_floor7"):
#     mode = str(mode).strip().lower()
#     if mode in {"v2", "original", "v2_original"}:
#         return OriginalSpawnCoordinator()
#     if mode in {
#         "r8f_floor7",
#         "floor7",
#         "r8ff7",
#         "v2_r8f_floor7",
#         "r8f_floor7_proactive",
#         "floor7_proactive",
#     }:
#         return R8FFloor7SpawnCoordinator()
#     raise ValueError(
#         f"Unknown controller mode: {mode!r}; expected 'v2' or 'r8f_floor7'."
#     )


# __all__ = [
#     "OriginalSpawnCoordinator",
#     "R8FFloor7SpawnCoordinator",
#     "SpawnCoordinator",
#     "make_spawn_coordinator",
#     "compose_action",
# ]


# from __future__ import annotations

# """V2-R8F Floor7 reproduction controller with staggered maintenance births.

# Movement remains EXACTLY the original V2 movement stack. This module only
# changes reproduction.

# Population policy
# -----------------
# * pop < 7: hard recovery back to the operating floor.
# * pop == 7: refill the first reserve (8th agent) after a short spacing gap.
# * pop == 8: maintain a rolling 9th reserve on a slower cadence when colony
#   energy is healthy; age-risk can accelerate the birth.
# * pop == 9: create a 10th reserve only under severe replacement risk.
# * pop >= 10: never spawn.

# The important difference from the previous Floor7/max10 controller is that
# reserve births are intentionally SPACED through time.  The controller no
# longer waits only for a late age-risk event and then tries to build several
# reserve agents close together.  The goal is an age ladder rather than a
# synchronized cohort.

# Interpretation
# --------------
# * 7  = hard operating floor
# * 8  = immediate safety reserve
# * 9  = rolling maintenance reserve
# * 10 = severe-risk shock absorber

# Spacing
# -------
# * emergency below 7: ~0.6 s recovery cadence
# * 7 -> 8 refill:     ~8.0 s minimum since the previous birth
# * normal 8 -> 9:    ~11.5 s since the previous birth
# * risk-accelerated reserve birth: ~6.0 s minimum

# These gaps are deliberately much shorter than an agent lifetime, but long
# enough to avoid producing several same-age newborns in a few seconds.
# """

# import math
# from collections import Counter
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# from hybrid_controller import (
#     SpawnCoordinator as OriginalSpawnCoordinator,
#     compose_action,
#     local_count,
#     nearest_object,
# )


# class R8FFloor7SpawnCoordinator:
#     TARGET_FLOOR = 7
#     HARD_MAX_POPULATION = 10

#     YOUNG_AGE = 30.0
#     OLD_AGE = 45.0
#     SENIOR_AGE = 58.0

#     def __init__(
#         self,
#         cooldown_seconds: float = 2.0,
#         recovery_cooldown_seconds: float = 0.6,
#         same_parent_cooldown_seconds: float = 6.0,
#         floor_refill_gap_seconds: float = 8.0,
#         normal_maintenance_gap_seconds: float = 11.5,
#         risk_maintenance_gap_seconds: float = 6.0,
#         hard_max_population: int = 10,
#     ):
#         self.original = OriginalSpawnCoordinator(cooldown_seconds=cooldown_seconds)
#         self.cooldown_seconds = float(cooldown_seconds)
#         self.recovery_cooldown_seconds = float(recovery_cooldown_seconds)
#         self.same_parent_cooldown_seconds = float(same_parent_cooldown_seconds)
#         self.floor_refill_gap_seconds = float(floor_refill_gap_seconds)
#         self.normal_maintenance_gap_seconds = float(normal_maintenance_gap_seconds)
#         self.risk_maintenance_gap_seconds = float(risk_maintenance_gap_seconds)
#         self.hard_max_population = int(hard_max_population)

#         self.last_recovery_time = -1e9
#         self.last_reserve_time = -1e9
#         self.last_parent_spawn_time: Dict[int, float] = {}
#         self.last_reason = "none"
#         self.reason_counts: Counter = Counter()

#     def reset(self):
#         self.original.reset()
#         self.last_recovery_time = -1e9
#         self.last_reserve_time = -1e9
#         self.last_parent_spawn_time.clear()
#         self.last_reason = "none"
#         self.reason_counts.clear()

#     @staticmethod
#     def _support(agent: Dict) -> Tuple[bool, float, float, float, int]:
#         fruit = nearest_object(agent, "Fruit")
#         tree = nearest_object(agent, "Tree")
#         predator = nearest_object(agent, "Predator")

#         fruit_d = float(fruit["distance"]) if fruit is not None else 1e9
#         tree_d = float(tree["distance"]) if tree is not None else 1e9
#         predator_d = float(predator["distance"]) if predator is not None else 1e9
#         crowd = local_count(agent, "Agent", 80.0)

#         food_support = fruit_d <= 110.0 or tree_d <= 75.0
#         return food_support, fruit_d, tree_d, predator_d, crowd

#     @staticmethod
#     def _trait_quality(agent: Dict) -> float:
#         return (
#             float(agent["speed"]) / 20.0
#             + float(agent["sprint_speed"]) / 40.0
#             + float(agent["hearing_radius"]) / 100.0
#             + float(agent["vision_range"]) / 400.0
#             + float(agent["vision_angle"]) / (math.pi / 2.0)
#             + float(agent["max_energy"]) / 1000.0
#         ) / 6.0

#     @classmethod
#     def _cohort_stats(cls, agents: List[Dict]) -> Dict[str, float]:
#         ages = np.asarray([float(a["age"]) for a in agents], dtype=np.float64)
#         energies = np.asarray([float(a["energy"]) for a in agents], dtype=np.float64)

#         if len(ages) == 0:
#             return {
#                 "young": 0.0,
#                 "old": 0.0,
#                 "senior": 0.0,
#                 "youngest_age": 0.0,
#                 "max_age": 0.0,
#                 "mean_age": 0.0,
#                 "mean_energy": 0.0,
#                 "p25_energy": 0.0,
#                 "min_energy": 0.0,
#                 "max_energy": 0.0,
#             }

#         return {
#             "young": float(np.sum(ages < cls.YOUNG_AGE)),
#             "old": float(np.sum(ages >= cls.OLD_AGE)),
#             "senior": float(np.sum(ages >= cls.SENIOR_AGE)),
#             "youngest_age": float(np.min(ages)),
#             "max_age": float(np.max(ages)),
#             "mean_age": float(np.mean(ages)),
#             "mean_energy": float(np.mean(energies)),
#             "p25_energy": float(np.percentile(energies, 25)),
#             "min_energy": float(np.min(energies)),
#             "max_energy": float(np.max(energies)),
#         }

#     def _record(self, reason: str, sim_time: float, parent_id: int):
#         self.last_reason = reason
#         self.reason_counts[reason] += 1
#         self.last_parent_spawn_time[int(parent_id)] = float(sim_time)

#         # Synchronize the V2 global cooldown with every extra birth.  This also
#         # gives us one shared "last birth" clock for the maintenance cadence.
#         self.original.last_spawn_time = float(sim_time)

#         if reason.startswith("recover"):
#             self.last_recovery_time = float(sim_time)
#         else:
#             self.last_reserve_time = float(sim_time)

#     def _recent_parent_penalty(self, agent_id: int, sim_time: float) -> float:
#         last = self.last_parent_spawn_time.get(int(agent_id), -1e9)
#         elapsed = sim_time - float(last)
#         if elapsed >= self.same_parent_cooldown_seconds:
#             return 0.0
#         frac = max(0.0, 1.0 - elapsed / self.same_parent_cooldown_seconds)
#         return 140.0 * frac

#     def _recovery_candidate(self, state: Dict) -> Tuple[Optional[int], str]:
#         """Aggressive recovery whenever population is below 7."""
#         agents = state.get("observations", []) or []
#         pop = int(state.get("num_agents", len(agents)))
#         sim_time = float(state.get("sim_time", 0.0))

#         if pop <= 0 or pop >= self.TARGET_FLOOR:
#             return None, "none"
#         if sim_time - self.last_recovery_time < self.recovery_cooldown_seconds:
#             return None, "none"

#         if pop == 6:
#             food_min_energy = 170.0
#             nofood_min_energy = 205.0
#             min_predator_d = 115.0
#             max_crowd = 5
#             reason = "recover6"
#             base_priority = 1000.0
#         elif pop == 5:
#             food_min_energy = 155.0
#             nofood_min_energy = 180.0
#             min_predator_d = 95.0
#             max_crowd = 6
#             reason = "recover5"
#             base_priority = 1200.0
#         elif pop == 4:
#             food_min_energy = 140.0
#             nofood_min_energy = 160.0
#             min_predator_d = 80.0
#             max_crowd = 8
#             reason = "recover4"
#             base_priority = 1400.0
#         else:  # pop <= 3
#             food_min_energy = 115.0
#             nofood_min_energy = 125.0
#             min_predator_d = 65.0
#             max_crowd = 99
#             reason = "recover_critical"
#             base_priority = 1650.0

#         candidates: List[Tuple[float, int]] = []
#         for agent in agents:
#             energy = float(agent["energy"])
#             age = float(agent["age"])
#             agent_id = int(agent["agent_id"])

#             # Simulator reproduction requires >100 energy. Keep a small margin.
#             if energy <= 105.0:
#                 continue

#             food_support, fruit_d, tree_d, predator_d, crowd = self._support(agent)
#             if predator_d < min_predator_d or crowd > max_crowd:
#                 continue

#             required_energy = food_min_energy if food_support else nofood_min_energy
#             if energy < required_energy:
#                 continue

#             post_spawn_energy = energy - 100.0
#             score = base_priority
#             score += 1.85 * post_spawn_energy
#             score += 70.0 * self._trait_quality(agent)
#             score -= 8.0 * crowd
#             score -= self._recent_parent_penalty(agent_id, sim_time)

#             if fruit_d <= 80.0:
#                 score += 80.0
#             elif tree_d <= 60.0:
#                 score += 35.0

#             if 20.0 <= age <= 48.0:
#                 score += 35.0
#             elif age >= self.SENIOR_AGE:
#                 score -= 80.0
#             elif age >= 52.0:
#                 score -= 40.0

#             candidates.append((score, agent_id))

#         if not candidates:
#             return None, "none"

#         candidates.sort(reverse=True)
#         return int(candidates[0][1]), reason

#     @staticmethod
#     def _risk_level(pop: int, stats: Dict[str, float]) -> int:
#         """Return 0=no age risk, 1=elevated, 2=severe.

#         The cadence itself maintains the reserve.  Age risk only accelerates a
#         reserve birth; it is no longer the sole trigger.
#         """
#         young = int(stats["young"])
#         old = int(stats["old"])
#         senior = int(stats["senior"])
#         max_age = float(stats["max_age"])

#         severe = (
#             young == 0
#             or senior >= 3
#             or (old >= 6 and young <= 1)
#             or (max_age >= 68.0 and young <= 1)
#         )
#         if severe:
#             return 2

#         elevated = (
#             young <= 1
#             or senior >= 2
#             or old >= max(4, pop - 3)
#             or (max_age >= 55.0 and young <= 2)
#         )
#         return 1 if elevated else 0

#     def _reserve_trigger(self, pop: int, stats: Dict[str, float], sim_time: float) -> Tuple[bool, str]:
#         """Decide whether a reserve birth is due, independent of parent choice."""
#         since_birth = sim_time - float(self.original.last_spawn_time)
#         risk = self._risk_level(pop, stats)

#         # 7 is the hard floor.  Refill the first reserve relatively quickly so
#         # a single additional death does not immediately put us into recovery.
#         if pop == 7:
#             if risk >= 1 and since_birth >= self.risk_maintenance_gap_seconds:
#                 return True, "reserve8_risk"
#             if since_birth >= self.floor_refill_gap_seconds:
#                 return True, "reserve8_cadence"
#             return False, "none"

#         # At 8, deliberately space the 9th agent.  This is the core age-ladder
#         # behavior: normal healthy worlds get one birth roughly every 11.5 s,
#         # while an aging cohort can accelerate to the 6 s risk gap.
#         if pop == 8:
#             if risk >= 1 and since_birth >= self.risk_maintenance_gap_seconds:
#                 return True, "reserve9_risk"
#             if since_birth >= self.normal_maintenance_gap_seconds:
#                 return True, "reserve9_cadence"
#             return False, "none"

#         # 10 is not a normal target. Only severe age risk is allowed to fill it.
#         if pop == 9:
#             if risk >= 2 and since_birth >= self.risk_maintenance_gap_seconds:
#                 return True, "reserve10_severe"
#             return False, "none"

#         return False, "none"

#     def _reserve_candidate(self, state: Dict) -> Tuple[Optional[int], str]:
#         """Choose a parent for a staggered 7->8, 8->9, or 9->10 birth."""
#         agents = state.get("observations", []) or []
#         pop = int(state.get("num_agents", len(agents)))
#         sim_time = float(state.get("sim_time", 0.0))

#         if pop < self.TARGET_FLOOR or pop >= self.hard_max_population:
#             return None, "none"
#         if pop not in (7, 8, 9):
#             return None, "none"

#         stats = self._cohort_stats(agents)
#         due, reason = self._reserve_trigger(pop, stats, sim_time)
#         if not due:
#             return None, "none"

#         mean_energy = float(stats["mean_energy"])
#         p25_energy = float(stats["p25_energy"])
#         risk = self._risk_level(pop, stats)

#         # Energy guards are intentionally stricter for the upper reserve layers.
#         # Risk births get a small relaxation, but never bypass the colony-energy
#         # checks entirely; max10 must not become "ten starving agents".
#         if pop == 7:
#             food_min_energy = 215.0 if risk >= 1 else 220.0
#             nofood_min_energy = 280.0 if risk >= 1 else 290.0
#             min_predator_d = 135.0
#             max_crowd = 4
#             min_mean_energy = 135.0
#             min_p25_energy = 90.0
#             base_priority = 540.0
#         elif pop == 8:
#             food_min_energy = 225.0 if risk >= 1 else 235.0
#             nofood_min_energy = 295.0 if risk >= 1 else 305.0
#             min_predator_d = 145.0
#             max_crowd = 4
#             min_mean_energy = 150.0
#             min_p25_energy = 95.0
#             base_priority = 500.0
#         else:  # pop == 9, severe only
#             food_min_energy = 245.0
#             nofood_min_energy = 320.0
#             min_predator_d = 155.0
#             max_crowd = 4
#             min_mean_energy = 165.0
#             min_p25_energy = 105.0
#             base_priority = 450.0

#         if mean_energy < min_mean_energy or p25_energy < min_p25_energy:
#             return None, "none"

#         young = int(stats["young"])
#         senior = int(stats["senior"])

#         candidates: List[Tuple[float, int]] = []
#         for agent in agents:
#             energy = float(agent["energy"])
#             age = float(agent["age"])
#             agent_id = int(agent["agent_id"])

#             # Use an established adult when possible.  Newborn/very-young agents
#             # should not immediately finance the next generation.
#             if age < 20.0:
#                 continue

#             food_support, fruit_d, tree_d, predator_d, crowd = self._support(agent)
#             required_energy = food_min_energy if food_support else nofood_min_energy

#             if energy < required_energy:
#                 continue
#             if predator_d < min_predator_d or crowd > max_crowd:
#                 continue

#             post_spawn_energy = energy - 100.0
#             score = base_priority
#             score += 1.75 * post_spawn_energy
#             score += 80.0 * self._trait_quality(agent)
#             score -= 9.0 * crowd
#             score -= self._recent_parent_penalty(agent_id, sim_time)

#             # Prefer adults with recovery time remaining; avoid financing births
#             # from the oldest members of the cohort when alternatives exist.
#             if 22.0 <= age <= 48.0:
#                 score += 50.0
#             elif age >= self.SENIOR_AGE:
#                 score -= 100.0
#             elif age >= 52.0:
#                 score -= 50.0

#             if fruit_d <= 80.0:
#                 score += 80.0
#             elif tree_d <= 60.0:
#                 score += 30.0

#             if young == 0:
#                 score += 85.0
#             elif young == 1:
#                 score += 45.0
#             if senior >= 2:
#                 score += 30.0

#             candidates.append((score, agent_id))

#         if not candidates:
#             return None, "none"

#         candidates.sort(reverse=True)
#         return int(candidates[0][1]), reason

#     def choose(self, state: Dict) -> Optional[int]:
#         self.last_reason = "none"
#         sim_time = float(state.get("sim_time", 0.0))
#         agents = state.get("observations", []) or []
#         pop = int(state.get("num_agents", len(agents)))

#         if pop <= 0 or pop >= self.hard_max_population:
#             return None

#         # 1) BELOW FLOOR: V2 gets first chance; otherwise hard recovery takes over.
#         if pop < self.TARGET_FLOOR:
#             original_choice = self.original.choose(state)
#             if original_choice is not None:
#                 chosen = int(original_choice)
#                 self._record("v2", sim_time, chosen)
#                 return chosen

#             chosen, reason = self._recovery_candidate(state)
#             if chosen is not None:
#                 self._record(reason, sim_time, chosen)
#                 return int(chosen)
#             return None

#         # 2) RESERVE REGION 7..9: cadence creates an age ladder; cohort risk may
#         # accelerate the next reserve birth, but every birth still passes energy,
#         # predator and parent-quality guards.
#         if pop in (7, 8, 9):
#             # V2's minimum global cooldown is always respected first.
#             if sim_time - float(self.original.last_spawn_time) < self.cooldown_seconds:
#                 return None

#             chosen, reason = self._reserve_candidate(state)
#             if chosen is not None:
#                 self._record(reason, sim_time, chosen)
#                 return int(chosen)
#             return None

#         return None


# # Keep the exact public names expected by the existing R8F/Floor7 server.
# SpawnCoordinator = R8FFloor7SpawnCoordinator


# def make_spawn_coordinator(mode: str = "r8f_floor7"):
#     mode = str(mode).strip().lower()
#     if mode in {"v2", "original", "v2_original"}:
#         return OriginalSpawnCoordinator()
#     if mode in {
#         "r8f_floor7",
#         "floor7",
#         "r8ff7",
#         "v2_r8f_floor7",
#         "r8f_floor7_proactive",
#         "floor7_proactive",
#     }:
#         return R8FFloor7SpawnCoordinator()
#     raise ValueError(
#         f"Unknown controller mode: {mode!r}; expected 'v2' or 'r8f_floor7'."
#     )


# __all__ = [
#     "OriginalSpawnCoordinator",
#     "R8FFloor7SpawnCoordinator",
#     "SpawnCoordinator",
#     "make_spawn_coordinator",
#     "compose_action",
# ]


from __future__ import annotations

"""Adaptive V2 controller for Nordic AI Cup survival simulator.

This keeps the proven V2 actor/checkpoint, but fixes several deterministic
failure modes around it:

1) Threat-aware predator handling
   - Predators only see to 250 and hear to 60 in the organizer simulator.
   - A predator directly chases when the herbivore is looking away OR when it
     is closer than ~90. Otherwise it pivots around the herbivore.
   - We therefore do NOT burn sprint energy merely because a predator is
     visible. We face detected predators, back away economically, and sprint
     only in the true danger zone / multi-predator case.
   - Far or currently-undetecting predators are ignored/monitored instead of
     triggering the old always-flee branch.

2) Hidden-senescence detection
   - max_age is hidden, but once age > max_age the simulator applies an extra
     0.01 * age energy loss EVERY TICK. We infer this from energy residuals
     after subtracting the action cost and passive drain.
   - Detected senescence feeds reproduction risk and can trigger a safe
     last-chance replacement birth.

3) Dynamic carrying capacity
   - 7 remains the hard floor.
   - 8 is the safety reserve.
   - 9 is the normal rolling reserve when energy allows.
   - 10 is used when food/energy are healthy AND there is replacement/death
     pressure (or the world is unusually resource-rich).
   - In a poor seed the controller stops insisting on ten starving agents.

4) Staggered births + death-pressure response
   - Normal births remain spaced to avoid same-age cohorts.
   - Sudden deaths or detected senescence can accelerate a reserve birth.

5) Evolution-aware parent ranking
   - Parent ranking favors traits that directly improve long-run survival:
     cheap walking speed, max energy, hearing, vision, then sprint speed.

6) Conservative biome-aware exploration
   - The learned residual is still used for ordinary resource movement.
   - Only no-resource exploration is nudged harder in river/desert/swamp,
     where staying put is relatively unattractive.

The model checkpoint does NOT need retraining. The PPO residual is preserved
for non-hard-override actions; predator safety remains deterministic.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.utils.DTOs import ActionRequest
from hybrid_controller import (
    BaseAction,
    SpawnCoordinator as OriginalSpawnCoordinator,
    clip,
    heuristic_base_action as v2_heuristic_base_action,
    local_count,
    nearest_edge,
    nearest_object,
    wrap_angle,
)


# ---------------------------------------------------------------------------
# Stateful species memory
# ---------------------------------------------------------------------------


@dataclass
class AgentMemory:
    last_time: float = -1e9
    last_energy: float = 0.0
    last_age: float = 0.0
    expected_action_cost: float = 0.0

    senescence_hits: float = 0.0
    senescent: bool = False
    senescence_detected_time: float = -1e9
    last_fruit_gain_time: float = -1e9

    last_pred_distance: Optional[float] = None
    last_pred_time: float = -1e9
    predator_calm_ticks: int = 0

    last_mode: str = "none"


class SpeciesMemory:
    """Cross-tick memory shared by movement and reproduction."""

    def __init__(self):
        self.agent: Dict[int, AgentMemory] = {}
        self.last_step_time: float = -1e9
        self.known_ids: set[int] = set()
        self.death_times: Deque[float] = deque(maxlen=256)
        self.birth_times: Deque[float] = deque(maxlen=256)
        self.current_modes: Dict[int, str] = {}
        self.cumulative_modes: Counter = Counter()

    def reset(self):
        self.agent.clear()
        self.last_step_time = -1e9
        self.known_ids.clear()
        self.death_times.clear()
        self.birth_times.clear()
        self.current_modes.clear()
        self.cumulative_modes.clear()

    @staticmethod
    def _predicted_action_cost(agent_state: Dict, req: ActionRequest) -> float:
        """Exact deterministic action-energy cost used by the public simulator."""
        speed = max(float(agent_state.get("speed", 0.0)), 0.0)
        sprint_speed = max(float(agent_state.get("sprint_speed", speed)), speed)
        energy = float(agent_state.get("energy", 0.0))
        max_energy = max(float(agent_state.get("max_energy", 1.0)), 1e-6)

        distance = clip(float(req.move_distance), 0.0, sprint_speed)
        # Public simulator limits low-energy sprinting back to walking speed.
        if energy < max_energy / 5.0 and distance > speed:
            distance = speed

        if distance <= speed:
            move_cost = distance * 0.05
        else:
            move_cost = speed * 0.05 + (distance - speed) * 0.5

        turn = float(req.turn_angle)
        turn_cost = min(math.pi, abs(turn)) / (2.0 * math.pi)
        spawn_cost = 100.0 if bool(req.spawn_agent) and energy > 100.0 else 0.0
        return float(move_cost + turn_cost + spawn_cost)

    def prepare_step(self, state: Dict):
        """Update deaths/births and infer hidden senescence once per tick."""
        sim_time = float(state.get("sim_time", 0.0))
        if sim_time < self.last_step_time - 1e-6:
            self.reset()

        # Idempotent: movement/spawn wrappers may both call prepare_step.
        if abs(sim_time - self.last_step_time) < 1e-9:
            return

        agents = state.get("observations", []) or []
        current_ids = {int(a["agent_id"]) for a in agents}

        if self.last_step_time > -1e8:
            for _aid in self.known_ids - current_ids:
                self.death_times.append(sim_time)
            for _aid in current_ids - self.known_ids:
                self.birth_times.append(sim_time)

        self.current_modes = {}

        for a in agents:
            aid = int(a["agent_id"])
            energy = float(a["energy"])
            age = float(a["age"])
            mem = self.agent.setdefault(aid, AgentMemory())

            if mem.last_time > -1e8 and sim_time > mem.last_time:
                dt = sim_time - mem.last_time
                actual_drop = mem.last_energy - energy
                # All published biome classes currently leave energy drain rate
                # at the base 1 energy/s. Extra old-age drain is NOT dt-scaled.
                expected_drop = mem.expected_action_cost + dt * 1.0
                residual_extra = actual_drop - expected_drop

                # Fruit consumption causes a positive energy jump and should not
                # be mistaken for "not senescent" after senescence was detected.
                if actual_drop < -2.0:
                    mem.last_fruit_gain_time = sim_time

                if age >= 52.0 and not mem.senescent:
                    # Hidden old-age penalty is >= ~0.6/tick at the earliest
                    # possible onset, so two clean residual hits are decisive.
                    if residual_extra > 0.32:
                        mem.senescence_hits += 1.0
                    elif residual_extra < -1.0:
                        # Fruit gain: do not punish confidence much.
                        mem.senescence_hits = max(0.0, mem.senescence_hits - 0.10)
                    else:
                        mem.senescence_hits = max(0.0, mem.senescence_hits - 0.25)

                    if mem.senescence_hits >= 2.0:
                        mem.senescent = True
                        mem.senescence_detected_time = sim_time

            mem.last_time = sim_time
            mem.last_energy = energy
            mem.last_age = age
            # This gets overwritten after the action is composed below.
            mem.expected_action_cost = 0.0

        # Keep memory for dead IDs only briefly unnecessary; deleting prevents
        # an ever-growing dictionary in long runs with many generations.
        for aid in list(self.agent.keys()):
            if aid not in current_ids:
                del self.agent[aid]

        self.known_ids = current_ids
        self.last_step_time = sim_time

        while self.death_times and sim_time - self.death_times[0] > 60.0:
            self.death_times.popleft()
        while self.birth_times and sim_time - self.birth_times[0] > 60.0:
            self.birth_times.popleft()

    def record_action(self, agent_state: Dict, req: ActionRequest, mode: str):
        aid = int(agent_state["agent_id"])
        mem = self.agent.setdefault(aid, AgentMemory())
        mem.expected_action_cost = self._predicted_action_cost(agent_state, req)
        mem.last_mode = str(mode)
        self.current_modes[aid] = str(mode)
        self.cumulative_modes[str(mode)] += 1

    def is_senescent(self, agent_id: int) -> bool:
        mem = self.agent.get(int(agent_id))
        return bool(mem.senescent) if mem is not None else False

    def recent_deaths(self, sim_time: float, window: float = 12.0) -> int:
        return sum(1 for t in self.death_times if sim_time - t <= window)

    def recent_births(self, sim_time: float, window: float = 12.0) -> int:
        return sum(1 for t in self.birth_times if sim_time - t <= window)

    def predator_closing_rate(self, agent_state: Dict, sim_time: float, distance: float) -> float:
        """Net closing speed (world units / simulated second) for nearest threat."""
        aid = int(agent_state["agent_id"])
        mem = self.agent.setdefault(aid, AgentMemory())
        rate = 0.0
        if mem.last_pred_distance is not None and sim_time > mem.last_pred_time:
            dt = sim_time - mem.last_pred_time
            if dt <= 0.5:
                rate = (float(mem.last_pred_distance) - float(distance)) / max(dt, 1e-6)
                rate = clip(rate, -250.0, 250.0)

        if abs(rate) < 5.0:
            mem.predator_calm_ticks += 1
        else:
            mem.predator_calm_ticks = 0

        mem.last_pred_distance = float(distance)
        mem.last_pred_time = float(sim_time)
        return float(rate)

    def clear_predator_track(self, agent_id: int):
        mem = self.agent.get(int(agent_id))
        if mem is not None:
            mem.last_pred_distance = None
            mem.last_pred_time = -1e9
            mem.predator_calm_ticks = 0


# ---------------------------------------------------------------------------
# Movement controller
# ---------------------------------------------------------------------------


class AdaptiveMovementController:
    """V2 residual actor with deterministic fixes around the heuristic prior."""

    PREDATOR_HEARING = 60.0
    PREDATOR_VISION = 250.0
    PREDATOR_HALF_CONE = math.pi / 6.0  # predator Creature default cone = pi/3
    DIRECT_CHASE_DISTANCE = 90.0

    def __init__(self, memory: SpeciesMemory, mode: str = "adaptive"):
        self.memory = memory
        self.mode = str(mode).strip().lower()
        if self.mode not in {"adaptive", "v2", "original"}:
            raise ValueError(f"Unknown movement mode: {mode!r}")

    @staticmethod
    def _predators(agent_state: Dict) -> List[Dict]:
        out = [
            o for o in (agent_state.get("observations", []) or [])
            if o.get("type") == "Predator"
        ]
        out.sort(key=lambda o: float(o.get("distance", 1e9)))
        return out

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return abs(wrap_angle(float(a) - float(b)))

    def _predator_can_detect(self, pred: Dict) -> bool:
        d = float(pred.get("distance", 1e9))
        rel = float(pred.get("rel_dir", math.pi))
        if d <= self.PREDATOR_HEARING + 3.0:
            return True
        # Small margin around the exact public half-cone avoids oscillation.
        return d <= self.PREDATOR_VISION + 3.0 and abs(rel) <= self.PREDATOR_HALF_CONE + 0.10

    def _edge_safe_escape(self, agent_state: Dict, escape_dir: float) -> float:
        edge = nearest_edge(agent_state)
        if edge is None:
            return wrap_angle(escape_dir)
        edge_dist, edge_angle = edge
        if edge_dist >= 30.0:
            return wrap_angle(escape_dir)

        # Only alter escape when the requested direction is actually toward the
        # nearby edge. Blend with an away-from-edge vector instead of flipping.
        if self._angle_diff(escape_dir, edge_angle) > 1.05:
            return wrap_angle(escape_dir)

        away_edge = wrap_angle(edge_angle + math.pi)
        vx = math.cos(escape_dir) + 1.35 * math.cos(away_edge)
        vy = math.sin(escape_dir) + 1.35 * math.sin(away_edge)
        if abs(vx) + abs(vy) < 1e-8:
            return away_edge
        return wrap_angle(math.atan2(vy, vx))

    def _predator_action(self, agent_state: Dict, state: Dict) -> Optional[BaseAction]:
        preds = self._predators(agent_state)
        aid = int(agent_state["agent_id"])
        sim_time = float(state.get("sim_time", 0.0))

        if not preds:
            self.memory.clear_predator_track(aid)
            return None

        nearest = preds[0]
        pd = float(nearest.get("distance", 1e9))
        pa = float(nearest.get("angle", 0.0))
        closing = self.memory.predator_closing_rate(agent_state, sim_time, pd)

        detectable = [p for p in preds if self._predator_can_detect(p)]
        near_count = sum(1 for p in preds if float(p.get("distance", 1e9)) <= 170.0)

        # If the predator cannot currently detect us and is not close, do not
        # reproduce V2's expensive "flee every visible predator" behavior.
        # At >250 the predator literally cannot see us in the public simulator.
        if not detectable:
            if pd > 175.0:
                return None

            # Stealth hold: a nearby predator is looking elsewhere. Do not walk
            # into its cone / hearing radius just because fruit is visible.
            # Turn toward it gradually so that if detection happens next tick,
            # the predator will prefer its pivot behavior rather than direct chase.
            turn = clip(pa, -0.35, 0.35)
            if pd <= 95.0:
                # Quietly increase separation without sprinting.
                distance = min(float(agent_state["speed"]) * 0.35, 3.0)
                escape = self._edge_safe_escape(agent_state, wrap_angle(pa + math.pi))
                return BaseAction(distance, escape, turn, "predator_stealth_backoff", True)
            return BaseAction(0.0, 0.0, turn, "predator_stealth_hold", True)

        # Threat vector away from all predators that can currently detect the
        # agent, with extra weight for close predators. This is safer than using
        # only the nearest predator when wolves approach from two directions.
        vx = 0.0
        vy = 0.0
        for p in detectable:
            d = max(float(p.get("distance", 1e9)), 20.0)
            a = float(p.get("angle", 0.0))
            w = 1.0 / (d ** 1.20)
            vx -= w * math.cos(a)
            vy -= w * math.sin(a)
        escape_dir = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-12 else wrap_angle(pa + math.pi)
        escape_dir = self._edge_safe_escape(agent_state, escape_dir)

        speed = max(float(agent_state["speed"]), 0.1)
        sprint = max(float(agent_state["sprint_speed"]), speed)
        facing_error = abs(pa)
        n_detected = len(detectable)

        # Always face the nearest threat. This directly exploits the organizer
        # predator rule: outside 90 units, looking toward it causes a pivot rather
        # than the cheaper straight-line chase.
        turn = clip(pa, -0.72, 0.72)

        # True emergency: close enough that the predator direct-chases regardless
        # of facing, or multiple predators make a single facing direction unsafe.
        if pd < 72.0 or (pd < 95.0 and facing_error > 1.25) or (n_detected >= 2 and pd < 110.0):
            return BaseAction(sprint, escape_dir, turn, "predator_sprint", True)

        # Transition zone. If we are still turning to face the predator, buy a
        # little time with partial sprint. Once facing it, walking is usually
        # enough to counter most radial closing from the pivot path.
        if pd < 100.0:
            if facing_error > 0.95 or closing > 55.0:
                distance = speed + 0.45 * (sprint - speed)
                mode = "predator_run"
            else:
                distance = speed
                mode = "predator_walk"
            return BaseAction(distance, escape_dir, turn, mode, True)

        if pd < 145.0:
            if facing_error > 1.15 or closing > 55.0 or near_count >= 2:
                distance = speed + 0.25 * (sprint - speed)
                mode = "predator_run"
            elif closing > 12.0:
                distance = speed
                mode = "predator_walk"
            elif closing <= 0.0:
                distance = 0.35 * speed
                mode = "predator_coast"
            else:
                distance = 0.75 * speed
                mode = "predator_walk"
            return BaseAction(distance, escape_dir, turn, mode, True)

        if pd < 205.0:
            if n_detected >= 2 or closing > 35.0:
                distance = 0.95 * speed
                mode = "predator_walk"
            elif closing > 8.0:
                distance = 0.70 * speed
                mode = "predator_walk"
            else:
                # Predator may be resting / orbiting away. Keep facing it but
                # stop paying V2's huge >speed sprint cost.
                distance = 0.25 * speed
                mode = "predator_coast"
            return BaseAction(distance, escape_dir, turn, mode, True)

        # Detectable but far. A low-cost backpedal preserves separation while
        # letting the predator burn its own energy. If net distance is already
        # opening, nearly stop.
        if closing > 30.0 or n_detected >= 2:
            distance = 0.60 * speed
        elif closing > 5.0:
            distance = 0.40 * speed
        else:
            distance = 0.15 * speed
        return BaseAction(distance, escape_dir, turn, "predator_far_control", True)

    @staticmethod
    def _copy_without_predators(agent_state: Dict) -> Dict:
        copied = dict(agent_state)
        copied["observations"] = [
            o for o in (agent_state.get("observations", []) or [])
            if o.get("type") != "Predator"
        ]
        return copied

    def _enhanced_base_action(self, agent_state: Dict, state: Dict) -> BaseAction:
        predator_action = self._predator_action(agent_state, state)
        if predator_action is not None:
            return predator_action

        # Important: when a far/stealth predator is intentionally ignored, call
        # the original V2 heuristic WITHOUT predator observations; otherwise its
        # old unconditional predator branch would activate again.
        agent_no_pred = self._copy_without_predators(agent_state)
        base = v2_heuristic_base_action(agent_no_pred, state)

        if base.hard_override:
            return base

        energy = float(agent_state["energy"])
        max_energy = max(float(agent_state["max_energy"]), 1e-6)
        ratio = energy / max_energy
        speed = float(agent_state["speed"])
        biome = str(agent_state.get("biome", "")).lower()
        aid = int(agent_state["agent_id"])

        # Low-energy fruit urgency. Stay at walking cost; simply stop crawling at
        # 4 units/tick when there is a visible fruit and extinction is near.
        if base.mode == "fruit" and (energy < 85.0 or ratio < 0.18):
            fruit = nearest_object(agent_state, "Fruit")
            if fruit is not None:
                fd = float(fruit.get("distance", 1e9))
                urgent = min(max(5.5, 0.75 * speed), speed, fd)
                base = BaseAction(
                    max(base.move_distance, urgent),
                    base.move_direction,
                    base.turn_angle,
                    "fruit_urgent",
                    False,
                )

        # Biome-aware no-resource search. Movement is charged BEFORE the biome
        # movement penalty, so river/swamp are bad places to endlessly scan. We
        # nudge displacement upward only when there is no current resource target.
        if base.mode in {"scan", "explore"}:
            sim_time = float(state.get("sim_time", 0.0))
            sign = 1.0 if aid % 2 == 0 else -1.0
            wobble = 0.22 * math.sin(0.41 * sim_time + 1.17 * aid)
            if biome == "river":
                base = BaseAction(min(3.2, 0.45 * speed), wobble, sign * 0.08, "escape_river", False)
            elif biome == "desert" and base.mode == "explore":
                base = BaseAction(max(base.move_distance, min(2.4, 0.35 * speed)), wobble, base.turn_angle, "explore_desert", False)
            elif biome == "swamp" and base.mode == "explore":
                base = BaseAction(max(base.move_distance, min(2.1, 0.30 * speed)), wobble, base.turn_angle, "explore_swamp", False)

        # Once hidden senescence is detected, expensive blind exploration has
        # low expected return. Stay resource-local and preserve enough energy for
        # a possible replacement birth; visible fruit is still pursued normally.
        if self.memory.is_senescent(aid) and base.mode in {
            "explore", "explore_desert", "explore_swamp", "escape_river"
        }:
            base = BaseAction(
                min(base.move_distance, 1.2),
                base.move_direction,
                base.turn_angle,
                "senescent_conserve",
                False,
            )

        return base

    def compose_action(
        self,
        agent_state: Dict,
        state: Dict,
        raw_residual: np.ndarray,
        spawn_agent: bool,
    ) -> Tuple[ActionRequest, bool, str]:
        self.memory.prepare_step(state)

        if self.mode in {"v2", "original"}:
            # Reimplement the original combine step below as well so action cost
            # memory remains available even in rollback mode.
            base = v2_heuristic_base_action(agent_state, state)
        else:
            base = self._enhanced_base_action(agent_state, state)

        raw = np.asarray(raw_residual, dtype=np.float32)
        if base.hard_override:
            move_distance = base.move_distance
            move_direction = base.move_direction
            turn_angle = base.turn_angle
            actor_used = False
        else:
            # EXACT same residual scaling as V2 training.
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
        self.memory.record_action(agent_state, req, base.mode)
        return req, actor_used, base.mode


# ---------------------------------------------------------------------------
# Reproduction controller
# ---------------------------------------------------------------------------


class R8FFloor7SpawnCoordinator:
    TARGET_FLOOR = 7
    HARD_MAX_POPULATION = 10

    YOUNG_AGE = 30.0
    OLD_AGE = 45.0
    SENIOR_AGE = 58.0

    def __init__(
        self,
        memory: Optional[SpeciesMemory] = None,
        cooldown_seconds: float = 2.0,
        recovery_cooldown_seconds: float = 0.6,
        same_parent_cooldown_seconds: float = 7.0,
        floor_refill_gap_seconds: float = 8.0,
        normal_maintenance_gap_seconds: float = 11.5,
        risk_maintenance_gap_seconds: float = 6.0,
        senescent_salvage_gap_seconds: float = 3.5,
        hard_max_population: int = 10,
    ):
        self.memory = memory if memory is not None else SpeciesMemory()
        self.original = OriginalSpawnCoordinator(cooldown_seconds=cooldown_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.recovery_cooldown_seconds = float(recovery_cooldown_seconds)
        self.same_parent_cooldown_seconds = float(same_parent_cooldown_seconds)
        self.floor_refill_gap_seconds = float(floor_refill_gap_seconds)
        self.normal_maintenance_gap_seconds = float(normal_maintenance_gap_seconds)
        self.risk_maintenance_gap_seconds = float(risk_maintenance_gap_seconds)
        self.senescent_salvage_gap_seconds = float(senescent_salvage_gap_seconds)
        self.hard_max_population = int(hard_max_population)

        self.last_recovery_time = -1e9
        self.last_reserve_time = -1e9
        self.last_parent_spawn_time: Dict[int, float] = {}
        self.last_reason = "none"
        self.last_target_cap = 8
        self.reason_counts: Counter = Counter()

    def reset(self):
        self.original.reset()
        self.last_recovery_time = -1e9
        self.last_reserve_time = -1e9
        self.last_parent_spawn_time.clear()
        self.last_reason = "none"
        self.last_target_cap = 8
        self.reason_counts.clear()

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

    @staticmethod
    def _trait_quality(agent: Dict) -> float:
        """Mechanics-weighted heritable quality in [roughly 0,1]."""
        speed = min(float(agent["speed"]) / 20.0, 1.0)
        sprint = min(float(agent["sprint_speed"]) / 40.0, 1.0)
        hearing = min(float(agent["hearing_radius"]) / 100.0, 1.0)
        vision = min(float(agent["vision_range"]) / 400.0, 1.0)
        angle = min(float(agent["vision_angle"]) / (math.pi / 2.0), 1.0)
        max_energy = min(float(agent["max_energy"]) / 1000.0, 1.0)

        # Walking speed and max_energy are especially valuable because higher
        # speed extends the cheap-movement regime, while max_energy increases
        # reproductive and starvation buffer. Hearing is omnidirectional and is
        # important against predators outside the vision cone.
        return (
            0.25 * speed
            + 0.05 * sprint
            + 0.20 * hearing
            + 0.15 * vision
            + 0.10 * angle
            + 0.25 * max_energy
        )

    @classmethod
    def _cohort_stats(cls, agents: List[Dict], memory: SpeciesMemory) -> Dict[str, float]:
        if not agents:
            return {
                "young": 0.0,
                "old": 0.0,
                "senior": 0.0,
                "senescent": 0.0,
                "youngest_age": 0.0,
                "max_age": 0.0,
                "mean_age": 0.0,
                "mean_energy": 0.0,
                "p25_energy": 0.0,
                "min_energy": 0.0,
                "max_energy": 0.0,
                "mean_energy_ratio": 0.0,
                "p25_energy_ratio": 0.0,
                "spawnable_fraction": 0.0,
            }

        ages = np.asarray([float(a["age"]) for a in agents], dtype=np.float64)
        energies = np.asarray([float(a["energy"]) for a in agents], dtype=np.float64)
        max_energies = np.asarray([max(float(a["max_energy"]), 1e-6) for a in agents], dtype=np.float64)
        ratios = energies / max_energies

        return {
            "young": float(np.sum(ages < cls.YOUNG_AGE)),
            "old": float(np.sum(ages >= cls.OLD_AGE)),
            "senior": float(np.sum(ages >= cls.SENIOR_AGE)),
            "senescent": float(sum(memory.is_senescent(int(a["agent_id"])) for a in agents)),
            "youngest_age": float(np.min(ages)),
            "max_age": float(np.max(ages)),
            "mean_age": float(np.mean(ages)),
            "mean_energy": float(np.mean(energies)),
            "p25_energy": float(np.percentile(energies, 25)),
            "min_energy": float(np.min(energies)),
            "max_energy": float(np.max(energies)),
            "mean_energy_ratio": float(np.mean(ratios)),
            "p25_energy_ratio": float(np.percentile(ratios, 25)),
            "spawnable_fraction": float(np.mean(energies > 100.0)),
        }

    @staticmethod
    def _resource_stats(agents: List[Dict]) -> Dict[str, float]:
        if not agents:
            return {
                "fruit_seen_fraction": 0.0,
                "tree_seen_fraction": 0.0,
                "mean_fruit_count": 0.0,
                "mean_tree_count": 0.0,
                "good_biome_fraction": 0.0,
                "poor_biome_fraction": 0.0,
            }

        fruit_counts = []
        tree_counts = []
        good = 0
        poor = 0
        for a in agents:
            obs = a.get("observations", []) or []
            fruit_counts.append(sum(1 for o in obs if o.get("type") == "Fruit"))
            tree_counts.append(sum(1 for o in obs if o.get("type") == "Tree"))
            biome = str(a.get("biome", "")).lower()
            if biome in {"forest", "grassland"}:
                good += 1
            if biome in {"desert", "river"}:
                poor += 1

        n = float(len(agents))
        return {
            "fruit_seen_fraction": float(sum(c > 0 for c in fruit_counts) / n),
            "tree_seen_fraction": float(sum(c > 0 for c in tree_counts) / n),
            "mean_fruit_count": float(np.mean(fruit_counts)),
            "mean_tree_count": float(np.mean(tree_counts)),
            "good_biome_fraction": float(good / n),
            "poor_biome_fraction": float(poor / n),
        }

    def _record(self, reason: str, sim_time: float, parent_id: int):
        self.last_reason = str(reason)
        self.reason_counts[str(reason)] += 1
        self.last_parent_spawn_time[int(parent_id)] = float(sim_time)
        self.original.last_spawn_time = float(sim_time)

        if str(reason).startswith("recover"):
            self.last_recovery_time = float(sim_time)
        else:
            self.last_reserve_time = float(sim_time)

    def _recent_parent_penalty(self, agent_id: int, sim_time: float) -> float:
        last = self.last_parent_spawn_time.get(int(agent_id), -1e9)
        elapsed = sim_time - float(last)
        if elapsed >= self.same_parent_cooldown_seconds:
            return 0.0
        frac = max(0.0, 1.0 - elapsed / self.same_parent_cooldown_seconds)
        return 160.0 * frac

    def _parent_recent(self, agent_id: int, sim_time: float) -> bool:
        return sim_time - self.last_parent_spawn_time.get(int(agent_id), -1e9) < self.same_parent_cooldown_seconds

    def _risk_level(self, pop: int, stats: Dict[str, float]) -> int:
        young = int(stats["young"])
        old = int(stats["old"])
        senior = int(stats["senior"])
        senescent = int(stats["senescent"])
        max_age = float(stats["max_age"])

        severe = (
            senescent >= 2
            or young == 0
            or senior >= 3
            or (old >= 6 and young <= 1)
            or (max_age >= 68.0 and young <= 1)
        )
        if severe:
            return 2

        elevated = (
            senescent >= 1
            or young <= 1
            or senior >= 2
            or old >= max(4, pop - 3)
            or (max_age >= 55.0 and young <= 2)
        )
        return 1 if elevated else 0

    def _desired_cap(self, state: Dict, stats: Dict[str, float], resources: Dict[str, float]) -> int:
        """Choose 8/9/10 from current carrying capacity and replacement pressure."""
        sim_time = float(state.get("sim_time", 0.0))
        agents = state.get("observations", []) or []
        pop = int(state.get("num_agents", len(agents)))
        risk = self._risk_level(pop, stats)
        recent_deaths = self.memory.recent_deaths(sim_time, window=12.0)

        mean_ratio = float(stats["mean_energy_ratio"])
        p25_ratio = float(stats["p25_energy_ratio"])
        mean_energy = float(stats["mean_energy"])
        p25_energy = float(stats["p25_energy"])
        spawnable = float(stats["spawnable_fraction"])

        fruit_seen = float(resources["fruit_seen_fraction"])
        tree_seen = float(resources["tree_seen_fraction"])
        good_biome = float(resources["good_biome_fraction"])

        # Starvation guard. Do not insist on 9/10 merely to watch all agents run
        # out of energy together on a poor seed.
        fragile = (
            mean_energy < 135.0
            or p25_energy < 72.0
            or mean_ratio < 0.27
            or p25_ratio < 0.14
            or spawnable < 0.40
        )
        resource_poor = fruit_seen < 0.08 and tree_seen < 0.30 and mean_ratio < 0.36
        if fragile or resource_poor:
            return 8

        cap = 9

        food_ok = fruit_seen >= 0.12 or tree_seen >= 0.50 or good_biome >= 0.60
        robust_energy = (
            mean_energy >= 160.0
            and p25_energy >= 90.0
            and mean_ratio >= 0.32
            and p25_ratio >= 0.17
            and spawnable >= 0.55
        )
        replacement_pressure = recent_deaths >= 1 or int(stats["senescent"]) >= 2 or risk >= 2
        abundant = fruit_seen >= 0.25 and mean_ratio >= 0.40 and p25_ratio >= 0.22

        if robust_energy and food_ok and (replacement_pressure or (sim_time >= 90.0 and abundant)):
            cap = 10

        return min(cap, self.hard_max_population)

    def _recovery_candidate(self, state: Dict) -> Tuple[Optional[int], str]:
        agents = state.get("observations", []) or []
        pop = int(state.get("num_agents", len(agents)))
        sim_time = float(state.get("sim_time", 0.0))

        if pop <= 0 or pop >= self.TARGET_FLOOR:
            return None, "none"
        if sim_time - self.last_recovery_time < self.recovery_cooldown_seconds:
            return None, "none"

        if pop == 6:
            food_min_energy, nofood_min_energy = 168.0, 202.0
            min_predator_d, max_crowd = 110.0, 5
            reason, base_priority = "recover6", 1000.0
        elif pop == 5:
            food_min_energy, nofood_min_energy = 152.0, 176.0
            min_predator_d, max_crowd = 92.0, 6
            reason, base_priority = "recover5", 1200.0
        elif pop == 4:
            food_min_energy, nofood_min_energy = 138.0, 156.0
            min_predator_d, max_crowd = 78.0, 8
            reason, base_priority = "recover4", 1400.0
        else:
            food_min_energy, nofood_min_energy = 112.0, 122.0
            min_predator_d, max_crowd = 62.0, 99
            reason, base_priority = "recover_critical", 1650.0

        candidates: List[Tuple[float, int]] = []
        for agent in agents:
            energy = float(agent["energy"])
            age = float(agent["age"])
            aid = int(agent["agent_id"])
            if energy <= 105.0:
                continue

            food_support, fruit_d, tree_d, predator_d, crowd = self._support(agent)
            if predator_d < min_predator_d or crowd > max_crowd:
                continue

            required = food_min_energy if food_support else nofood_min_energy
            if energy < required:
                continue

            post = energy - 100.0
            score = base_priority
            score += 1.85 * post
            score += 110.0 * self._trait_quality(agent)
            score -= 8.0 * crowd
            score -= self._recent_parent_penalty(aid, sim_time)

            if fruit_d <= 80.0:
                score += 80.0
            elif tree_d <= 60.0:
                score += 35.0

            if self.memory.is_senescent(aid):
                # Energy in a senescent parent is perishable; converting some of
                # it into a newborn can be better than losing it to age drain.
                score += 70.0 if energy >= 185.0 else 15.0
            elif 20.0 <= age <= 48.0:
                score += 35.0
            elif age >= self.SENIOR_AGE:
                score -= 35.0

            candidates.append((score, aid))

        if not candidates:
            return None, "none"
        candidates.sort(reverse=True)
        return int(candidates[0][1]), reason

    def _senescent_salvage_candidate(self, state: Dict, target_cap: int) -> Tuple[Optional[int], str]:
        """Safely turn energy from a detected-senescent parent into a replacement."""
        agents = state.get("observations", []) or []
        pop = int(state.get("num_agents", len(agents)))
        sim_time = float(state.get("sim_time", 0.0))
        since_birth = sim_time - float(self.original.last_spawn_time)

        if pop < self.TARGET_FLOOR or pop >= target_cap:
            return None, "none"
        if since_birth < self.senescent_salvage_gap_seconds:
            return None, "none"

        candidates: List[Tuple[float, int]] = []
        for a in agents:
            aid = int(a["agent_id"])
            if not self.memory.is_senescent(aid):
                continue
            if self._parent_recent(aid, sim_time):
                continue

            energy = float(a["energy"])
            if energy <= 100.0:
                continue
            food_support, fruit_d, tree_d, predator_d, crowd = self._support(a)
            required = 195.0 if food_support else 270.0
            if energy < required or predator_d < 125.0 or crowd > 5:
                continue

            score = 700.0
            score += 1.4 * (energy - 100.0)
            score += 130.0 * self._trait_quality(a)
            if fruit_d <= 80.0:
                score += 70.0
            elif tree_d <= 60.0:
                score += 25.0
            candidates.append((score, aid))

        if not candidates:
            return None, "none"
        candidates.sort(reverse=True)
        return int(candidates[0][1]), "senescent_salvage"

    def _reserve_trigger(
        self,
        pop: int,
        stats: Dict[str, float],
        sim_time: float,
        target_cap: int,
    ) -> Tuple[bool, str]:
        since_birth = sim_time - float(self.original.last_spawn_time)
        risk = self._risk_level(pop, stats)
        recent_deaths = self.memory.recent_deaths(sim_time, window=12.0)

        if pop == 7 and target_cap >= 8:
            if (risk >= 1 or recent_deaths >= 1) and since_birth >= self.risk_maintenance_gap_seconds:
                return True, "reserve8_risk"
            if since_birth >= self.floor_refill_gap_seconds:
                return True, "reserve8_cadence"
            return False, "none"

        if pop == 8 and target_cap >= 9:
            if (risk >= 1 or recent_deaths >= 1) and since_birth >= self.risk_maintenance_gap_seconds:
                return True, "reserve9_risk"
            if since_birth >= self.normal_maintenance_gap_seconds:
                return True, "reserve9_cadence"
            return False, "none"

        if pop == 9 and target_cap >= 10:
            if (risk >= 1 or recent_deaths >= 1) and since_birth >= self.risk_maintenance_gap_seconds:
                return True, "reserve10_pressure"
            if since_birth >= self.normal_maintenance_gap_seconds + 2.0:
                return True, "reserve10_abundant"
            return False, "none"

        return False, "none"

    def _reserve_candidate(
        self,
        state: Dict,
        stats: Dict[str, float],
        target_cap: int,
    ) -> Tuple[Optional[int], str]:
        agents = state.get("observations", []) or []
        pop = int(state.get("num_agents", len(agents)))
        sim_time = float(state.get("sim_time", 0.0))

        if pop < self.TARGET_FLOOR or pop >= target_cap or pop not in (7, 8, 9):
            return None, "none"

        due, reason = self._reserve_trigger(pop, stats, sim_time, target_cap)
        if not due:
            return None, "none"

        risk = self._risk_level(pop, stats)
        mean_energy = float(stats["mean_energy"])
        p25_energy = float(stats["p25_energy"])

        if pop == 7:
            food_min_energy = 210.0 if risk >= 1 else 220.0
            nofood_min_energy = 275.0 if risk >= 1 else 290.0
            min_predator_d, max_crowd = 130.0, 4
            min_mean_energy, min_p25_energy = 130.0, 82.0
            base_priority = 550.0
        elif pop == 8:
            food_min_energy = 222.0 if risk >= 1 else 235.0
            nofood_min_energy = 292.0 if risk >= 1 else 305.0
            min_predator_d, max_crowd = 140.0, 4
            min_mean_energy, min_p25_energy = 145.0, 90.0
            base_priority = 510.0
        else:
            food_min_energy = 238.0 if risk >= 1 else 248.0
            nofood_min_energy = 310.0
            min_predator_d, max_crowd = 150.0, 4
            min_mean_energy, min_p25_energy = 158.0, 98.0
            base_priority = 470.0

        if mean_energy < min_mean_energy or p25_energy < min_p25_energy:
            return None, "none"

        young = int(stats["young"])
        candidates: List[Tuple[float, int]] = []
        for a in agents:
            energy = float(a["energy"])
            age = float(a["age"])
            aid = int(a["agent_id"])

            if age < 18.0 or self.memory.is_senescent(aid):
                continue
            if self._parent_recent(aid, sim_time):
                continue

            food_support, fruit_d, tree_d, predator_d, crowd = self._support(a)
            required = food_min_energy if food_support else nofood_min_energy
            if energy < required or predator_d < min_predator_d or crowd > max_crowd:
                continue

            post = energy - 100.0
            score = base_priority
            score += 1.7 * post
            score += 135.0 * self._trait_quality(a)
            score -= 9.0 * crowd

            if 22.0 <= age <= 48.0:
                score += 45.0
            elif age >= self.SENIOR_AGE:
                score -= 75.0
            elif age >= 52.0:
                score -= 35.0

            if fruit_d <= 80.0:
                score += 75.0
            elif tree_d <= 60.0:
                score += 28.0

            if young == 0:
                score += 80.0
            elif young == 1:
                score += 40.0

            candidates.append((score, aid))

        if not candidates:
            return None, "none"
        candidates.sort(reverse=True)
        return int(candidates[0][1]), reason

    def choose(self, state: Dict) -> Optional[int]:
        self.memory.prepare_step(state)
        self.last_reason = "none"

        sim_time = float(state.get("sim_time", 0.0))
        agents = state.get("observations", []) or []
        pop = int(state.get("num_agents", len(agents)))

        if pop <= 0 or pop >= self.hard_max_population:
            self.last_target_cap = self.hard_max_population
            return None

        stats = self._cohort_stats(agents, self.memory)
        resources = self._resource_stats(agents)
        target_cap = self._desired_cap(state, stats, resources)
        self.last_target_cap = int(target_cap)

        # 1) Hard floor recovery. Original V2 gets first chance, then the more
        # aggressive Floor7 recovery rules take over.
        if pop < self.TARGET_FLOOR:
            original_choice = self.original.choose(state)
            if original_choice is not None:
                chosen = int(original_choice)
                self._record("v2", sim_time, chosen)
                return chosen

            chosen, reason = self._recovery_candidate(state)
            if chosen is not None:
                self._record(reason, sim_time, chosen)
                return int(chosen)
            return None

        # Respect the minimum global birth gap in the reserve region.
        if sim_time - float(self.original.last_spawn_time) < self.cooldown_seconds:
            return None

        # 2) Hidden-senescence salvage gets first priority when capacity exists.
        # It converts energy from a parent that has entered the steep age drain
        # into a newborn before that energy disappears.
        chosen, reason = self._senescent_salvage_candidate(state, target_cap)
        if chosen is not None:
            self._record(reason, sim_time, chosen)
            return int(chosen)

        # 3) Normal staggered reserve births up to the DYNAMIC target cap.
        if pop in (7, 8, 9) and pop < target_cap:
            chosen, reason = self._reserve_candidate(state, stats, target_cap)
            if chosen is not None:
                self._record(reason, sim_time, chosen)
                return int(chosen)

        return None

    def debug_snapshot(self, state: Dict) -> Dict[str, float | int | str]:
        agents = state.get("observations", []) or []
        sim_time = float(state.get("sim_time", 0.0))
        stats = self._cohort_stats(agents, self.memory)
        resources = self._resource_stats(agents)
        return {
            "target_cap": int(self.last_target_cap),
            "mean_energy": round(float(stats["mean_energy"]), 1),
            "p25_energy": round(float(stats["p25_energy"]), 1),
            "mean_energy_ratio": round(float(stats["mean_energy_ratio"]), 3),
            "spawnable_fraction": round(float(stats["spawnable_fraction"]), 3),
            "young": int(stats["young"]),
            "senior": int(stats["senior"]),
            "senescent": int(stats["senescent"]),
            "fruit_seen_fraction": round(float(resources["fruit_seen_fraction"]), 3),
            "tree_seen_fraction": round(float(resources["tree_seen_fraction"]), 3),
            "recent_deaths_12s": int(self.memory.recent_deaths(sim_time, 12.0)),
            "last_reason": self.last_reason,
        }


# ---------------------------------------------------------------------------
# Unified controller + compatibility wrappers
# ---------------------------------------------------------------------------


class HybridSpeciesController:
    def __init__(self, spawn_mode: str = "r8f_floor7", movement_mode: str = "adaptive"):
        self.memory = SpeciesMemory()
        self.movement = AdaptiveMovementController(self.memory, mode=movement_mode)

        smode = str(spawn_mode).strip().lower()
        if smode in {"v2", "original", "v2_original"}:
            self.spawn = OriginalSpawnCoordinator()
            self.spawn_mode = "v2"
        elif smode in {
            "r8f_floor7", "floor7", "r8ff7", "v2_r8f_floor7",
            "r8f_floor7_proactive", "floor7_proactive",
        }:
            self.spawn = R8FFloor7SpawnCoordinator(memory=self.memory)
            self.spawn_mode = "r8f_floor7"
        else:
            raise ValueError(f"Unknown spawn/controller mode: {spawn_mode!r}")

        self.movement_mode = str(movement_mode).strip().lower()

    def reset(self):
        self.memory.reset()
        if hasattr(self.spawn, "reset"):
            self.spawn.reset()

    def prepare_step(self, state: Dict):
        self.memory.prepare_step(state)

    def choose_spawn(self, state: Dict) -> Optional[int]:
        self.memory.prepare_step(state)
        return self.spawn.choose(state)

    def compose_action(
        self,
        agent_state: Dict,
        state: Dict,
        raw_residual: np.ndarray,
        spawn_agent: bool,
    ) -> Tuple[ActionRequest, bool, str]:
        return self.movement.compose_action(agent_state, state, raw_residual, spawn_agent)

    def debug_snapshot(self, state: Dict) -> Dict:
        if isinstance(self.spawn, R8FFloor7SpawnCoordinator):
            snap = self.spawn.debug_snapshot(state)
        else:
            agents = state.get("observations", []) or []
            energies = [float(a["energy"]) for a in agents]
            snap = {
                "target_cap": 7,
                "mean_energy": round(float(np.mean(energies)), 1) if energies else 0.0,
                "p25_energy": round(float(np.percentile(energies, 25)), 1) if energies else 0.0,
                "senescent": sum(self.memory.is_senescent(int(a["agent_id"])) for a in agents),
                "last_reason": "v2",
            }
        snap["modes"] = dict(Counter(self.memory.current_modes.values()))
        return snap


def make_controller(
    spawn_mode: str = "r8f_floor7",
    movement_mode: str = "adaptive",
) -> HybridSpeciesController:
    return HybridSpeciesController(spawn_mode=spawn_mode, movement_mode=movement_mode)


# Legacy names for older evaluator/server scripts.
SpawnCoordinator = R8FFloor7SpawnCoordinator


def make_spawn_coordinator(mode: str = "r8f_floor7"):
    mode = str(mode).strip().lower()
    if mode in {"v2", "original", "v2_original"}:
        return OriginalSpawnCoordinator()
    if mode in {
        "r8f_floor7", "floor7", "r8ff7", "v2_r8f_floor7",
        "r8f_floor7_proactive", "floor7_proactive",
    }:
        return R8FFloor7SpawnCoordinator()
    raise ValueError(f"Unknown controller mode: {mode!r}")


# A default stateful movement wrapper keeps old code that imports compose_action
# working. New server code below uses HybridSpeciesController instead.
_DEFAULT_MEMORY = SpeciesMemory()
_DEFAULT_MOVEMENT = AdaptiveMovementController(_DEFAULT_MEMORY, mode="adaptive")


def compose_action(
    agent_state: Dict,
    state: Dict,
    raw_residual: np.ndarray,
    spawn_agent: bool,
) -> Tuple[ActionRequest, bool, str]:
    return _DEFAULT_MOVEMENT.compose_action(agent_state, state, raw_residual, spawn_agent)


def reset_default_controller():
    _DEFAULT_MEMORY.reset()


__all__ = [
    "AgentMemory",
    "SpeciesMemory",
    "AdaptiveMovementController",
    "OriginalSpawnCoordinator",
    "R8FFloor7SpawnCoordinator",
    "SpawnCoordinator",
    "HybridSpeciesController",
    "make_controller",
    "make_spawn_coordinator",
    "compose_action",
    "reset_default_controller",
]