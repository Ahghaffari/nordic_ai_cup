from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass(frozen=True)
class ControllerConfig:
    """Configuration for deterministic ecology logic outside the neural network.

    The MAPPO actor/critic architecture is intentionally unchanged. Keeping these
    values in a serializable dataclass lets training, evaluation and deployment use
    exactly the same controller settings.
    """

    name: str = "v4_robust7"

    # Reproduction pacing.
    normal_cooldown_seconds: float = 8.0
    emergency_cooldown_seconds: float = 2.0
    request_cooldown_seconds: float = 0.6
    max_population: int = 7

    # Age buckets. Official hidden max_age is random in [60, 120].
    young_age: float = 30.0
    old_age: float = 45.0
    senior_age: float = 65.0

    # Minimum parent energy by current species population.
    min_energy_pop2: float = 150.0
    min_energy_pop3: float = 165.0
    min_energy_pop4: float = 180.0
    min_energy_pop5: float = 195.0
    min_energy_pop6: float = 220.0
    min_energy_pop7: float = 250.0

    # If the species is nearly extinct and no food/tree support is perceived,
    # demand a much larger reserve before spending 100 energy on a child.
    emergency_no_food_min_energy: float = 240.0

    # Local support / safety radii.
    fruit_support_distance: float = 110.0
    tree_support_distance: float = 75.0
    predator_block_distance: float = 180.0
    parent_crowd_radius: float = 75.0

    # Population-5 replacement trigger.
    pop5_old_trigger: int = 2
    pop5_young_cap: int = 2
    pop5_heavy_old_trigger: int = 3
    pop5_heavy_young_cap: int = 2
    pop5_max_age_trigger: float = 55.0
    pop5_mean_age_trigger: float = 42.0

    # Population-6 replacement-buffer trigger.
    pop6_old_trigger: int = 3
    pop6_young_cap: int = 1
    pop6_senior_trigger: int = 2
    pop6_max_age_trigger: float = 70.0

    # Population-7 -> 8 is disabled in robust7 because max_population=7. These
    # thresholds are used by the optional v4_buffer8 profile only.
    pop7_old_trigger: int = 4
    pop7_young_cap: int = 1
    pop7_senior_trigger: int = 3
    pop7_max_age_trigger: float = 80.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ControllerConfig":
        if not data:
            return cls()
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**clean)

    def min_parent_energy(self, population: int) -> float:
        pop = int(population)
        if pop <= 2:
            return self.min_energy_pop2
        if pop == 3:
            return self.min_energy_pop3
        if pop == 4:
            return self.min_energy_pop4
        if pop == 5:
            return self.min_energy_pop5
        if pop == 6:
            return self.min_energy_pop6
        return self.min_energy_pop7


PROFILES: Dict[str, ControllerConfig] = {
    # Default: stabilize 5-7 agents; no extra eighth mouth competing for fruit.
    "v4_robust7": ControllerConfig(),

    # Experimental temporary replacement buffer. Population 8 is allowed only
    # when seven agents have a severe old-age imbalance and the selected parent
    # is well-fed/safe. It is NOT the default because more agents can increase
    # resource competition on hard seeds.
    "v4_buffer8": ControllerConfig(
        name="v4_buffer8",
        max_population=8,
        min_energy_pop7=250.0,
    ),

    # A slightly more conservative reproduction profile for ablation/testing.
    "v4_conservative7": ControllerConfig(
        name="v4_conservative7",
        normal_cooldown_seconds=10.0,
        min_energy_pop3=175.0,
        min_energy_pop4=195.0,
        min_energy_pop5=215.0,
        min_energy_pop6=235.0,
        emergency_no_food_min_energy=260.0,
    ),
}


def get_controller_config(name: str) -> ControllerConfig:
    key = str(name).strip()
    if key not in PROFILES:
        raise KeyError(
            f"Unknown controller profile {name!r}. "
            f"Available: {', '.join(sorted(PROFILES))}"
        )
    # Return a detached dataclass instance rather than the global object.
    return ControllerConfig.from_dict(PROFILES[key].to_dict())


def available_profiles():
    return tuple(sorted(PROFILES))
