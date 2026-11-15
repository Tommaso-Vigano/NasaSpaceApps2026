"""Default penalty rules: one selector profile -> Penalty per Datatype.

    from src.app.penalty.penalty_manager_fixed import FixedPenaltyManager
    from src.app.penalty.penalty_config import register_default_penalties

    manager = FixedPenaltyManager()
    register_default_penalties(manager)
    graph = manager.build_graph(HumanProfile(), world_slice)

Thresholds come from src/models/constraints.py (Astronaut for HumanProfile,
RoboticRover for RoverProfile), where each value has its source. The hard_constraints /
path / parameters dicts of src/routing/profiles.py are NOT read: today they hold wrong
values (HumanProfile slope 30 deg instead of 20; speeds in km/h written as m/s:
9.7, 7.5, 15.6). The profile classes are only used to tell walker and rover apart.

ExpeditionProfile -> CombinedPenalty of the participants' penalties (element-wise max:
the most limited member decides).

Unit: "cell equivalents" (1 = crossing one flat cell on rock; FixedPenaltyManager adds
this base cost). A penalty of 0.5 = "this cell costs half a cell more"; inf = forbidden.

Rules (walker | rover):
    TERRAIN_ETA      SoftThreshold(1, 1) = eta - 1   | CutOff(1.5): ripples forbidden [R3]
                     (consistent with "time x eta", TEAM ASSUMPTION option B)
    STEP_HEIGHT      CutOff(0.30 m) [R12]            | CutOff(0.40 m) [R1]
    THERMAL_INERTIA  ZeroPenalty                     | LowerCutOff(100) [R4, TEAM ASSUMPTION]
    LANDER_DISTANCE  CutOff(2000 m) [R11]            | ZeroPenalty
    RETURN_TIME      CutOff(1800 s) SEP warning [R20]| ZeroPenalty
    EVA_TIME         CutOff(21600 s = 6 h) [R11]     | ZeroPenalty
    VISIBILITY       WeightPenalty(0.5) TEAM ASSUMPTION, both
    DUST_TAU         CutOff(2.0)       TEAM ASSUMPTION (as gates.dust_storm_nogo), both
    SOLAR_FLUX       LowerCutOff(100)  TEAM ASSUMPTION (as gates.daylight_window), both
    WIND_SPEED, TEMPERATURE: registered only when PenaltyWeights gives the thresholds,
                     otherwise added to manager.ignored_datatypes (no sourced threshold yet).

The MCD maps (DUST_TAU, SOLAR_FLUX, WIND_SPEED, TEMPERATURE) are uniform over the grid,
so they ONLY use cut-off templates (0 or inf everywhere = GO / NO-GO). A progressive
cost on a uniform map would add the same value to every step, i.e. a cost on the number
of steps, which distorts the path.

The slope weights (2.0 walker, 0.5 rover) stay in FixedPenaltyManager.slope_weights.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from src.app.penalty.implementations.combined_penalty import CombinedPenalty
from src.app.penalty.implementations.cutoff_penalty import CutOffPenalty
from src.app.penalty.implementations.lower_cutoff_penalty import LowerCutOffPenalty
from src.app.penalty.implementations.penalty import Penalty
from src.app.penalty.implementations.soft_threshold_penalty import SoftThresholdPenalty
from src.app.penalty.implementations.weighing_penalty import WeightPenalty
from src.app.penalty.implementations.zero_penalty import ZeroPenalty
from src.app.penalty.penalty_manager_fixed import _kind, _participants
from src.constants import (
    DUST_TAU_MAX,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    SOLAR_FLUX_MIN_W_M2,
    TERRAIN_ETA_SOFT_THRESHOLD,
    TERRAIN_ETA_SOFT_WEIGHT,
    VISIBILITY_WEIGHT,
)
from src.data.types import Datatype
from src.models.constraints import ETA_TERRAIN, Astronaut, RoboticRover
from src.routing.profiles import Profile

# DUST_TAU_MAX and SOLAR_FLUX_MIN_W_M2 are the same constants used as defaults by
# src/models/gates.py (dust_storm_nogo, daylight_window): one source, src/constants.py.


@dataclass
class PenaltyWeights:
    """Weights and optional thresholds that have no published source.

    visibility: TEAM ASSUMPTION. Leaving the lander's line of sight is allowed but
        costs like half a cell more per cell, so the path prefers to stay in view
        without making long detours for it. To be discussed.
    wind_speed_max_ms, temperature_min_k, temperature_max_k: None = no sourced
        threshold yet, the layer is ignored. When given they become cut-offs
        (the MCD maps are uniform: only GO / NO-GO).
    """
    visibility: float = VISIBILITY_WEIGHT
    wind_speed_max_ms: Optional[float] = None
    temperature_min_k: Optional[float] = None
    temperature_max_k: Optional[float] = None


Factory = Callable[[], Penalty]


def _selector(walker: Factory, rover: Factory) -> Callable[[Profile], Penalty]:
    """profile -> Penalty: walker or rover rule; an expedition combines its members."""

    def select(profile: Profile) -> Penalty:
        members = _participants(profile)  # TypeError / ValueError on bad profiles
        penalties = [walker() if _kind(p) == "walker" else rover() for p in members]
        return penalties[0] if len(penalties) == 1 else CombinedPenalty(penalties)

    return select


def _same(factory: Factory) -> Callable[[Profile], Penalty]:
    return _selector(factory, factory)


def register_default_penalties(manager, weights: Optional[PenaltyWeights] = None):
    """Register the default selector of every Datatype on ``manager``
    (a FixedPenaltyManager: it needs ``ignored_datatypes``). Returns the manager."""
    weights = PenaltyWeights() if weights is None else weights
    astronaut, rover = Astronaut(), RoboticRover()

    rover_eta_max = ETA_TERRAIN["sand"] if rover.dunes_nogo else float("inf")
    rules = {
        Datatype.TERRAIN_ETA: _selector(
            lambda: SoftThresholdPenalty(threshold=TERRAIN_ETA_SOFT_THRESHOLD,
                                         weight=TERRAIN_ETA_SOFT_WEIGHT),
            (lambda: CutOffPenalty(rover_eta_max)) if rover.dunes_nogo else ZeroPenalty,
        ),
        Datatype.STEP_HEIGHT: _selector(
            lambda: CutOffPenalty(astronaut.max_step_up_m),
            lambda: CutOffPenalty(rover.max_step_m),
        ),
        Datatype.THERMAL_INERTIA: _selector(
            ZeroPenalty,
            lambda: LowerCutOffPenalty(rover.min_thermal_inertia),
        ),
        Datatype.LANDER_DISTANCE: _selector(
            lambda: CutOffPenalty(astronaut.max_distance_from_lander_m),
            ZeroPenalty,
        ),
        Datatype.RETURN_TIME: _selector(
            lambda: CutOffPenalty(astronaut.sep_return_min * SECONDS_PER_MINUTE),
            ZeroPenalty,
        ),
        Datatype.EVA_TIME: _selector(
            lambda: CutOffPenalty(astronaut.eva_nominal_h * SECONDS_PER_HOUR),
            ZeroPenalty,
        ),
        Datatype.VISIBILITY: _same(lambda: WeightPenalty(weights.visibility)),
        Datatype.DUST_TAU: _same(lambda: CutOffPenalty(DUST_TAU_MAX)),
        Datatype.SOLAR_FLUX: _same(lambda: LowerCutOffPenalty(SOLAR_FLUX_MIN_W_M2)),
    }

    if weights.wind_speed_max_ms is not None:
        rules[Datatype.WIND_SPEED] = _same(lambda: CutOffPenalty(weights.wind_speed_max_ms))
    else:
        manager.ignored_datatypes.add(Datatype.WIND_SPEED)

    t_min, t_max = weights.temperature_min_k, weights.temperature_max_k
    if t_min is not None or t_max is not None:
        def temperature() -> Penalty:
            parts = []
            if t_min is not None:
                parts.append(LowerCutOffPenalty(t_min))
            if t_max is not None:
                parts.append(CutOffPenalty(t_max))
            return parts[0] if len(parts) == 1 else CombinedPenalty(parts)
        rules[Datatype.TEMPERATURE] = _same(temperature)
    else:
        manager.ignored_datatypes.add(Datatype.TEMPERATURE)

    for data_type, selector in rules.items():
        manager.register_penalty(data_type, selector)
        manager.ignored_datatypes.discard(data_type)
    return manager
