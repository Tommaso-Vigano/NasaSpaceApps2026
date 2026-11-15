"""Mobility constraints for the agents moving in Jezero crater.

All quantities are SI (m, s, kg, W, deg for angles). Every constant carries the
ID of its source in square brackets; the full references are in ``SOURCES``
(src/constants.py).
Values marked "TEAM ASSUMPTION" are not taken from a reference.
"""

from dataclasses import dataclass

# Physical constants and the bibliography live in src/constants.py; they are
# re-exported here so that `from src.models.constraints import MARS_RADIUS_M` works.
from src.constants import (  # noqa: F401
    ETA_TERRAIN,
    G_EARTH,
    G_MARS,
    G_REL,
    MARS_RADIUS_M,
    SOURCES,
)


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RoboticRover:
    """Perseverance-class robotic rover."""

    max_slope_deg: float = 30.0  # [R1]
    max_step_m: float = 0.40  # obstacle/step height it can climb [R1]
    speed_ms: float = 0.042  # average AutoNav speed [R2]
    width_m: float = 2.7  # TEAM ASSUMPTION
    min_thermal_inertia: float = 100.0  # J m^-2 K^-1 s^-1/2, loose soil below this [R4, TEAM ASSUMPTION]
    dunes_nogo: bool = True  # dunes are no-go (embedding risk) [R3]


@dataclass(frozen=True)
class CrewedRover:
    """Pressurised / unpressurised crewed rover (LRV and SEV heritage)."""

    max_slope_deg: float = 25.0  # [R7]
    max_step_m: float = 0.30  # [R7]
    speed_ms: float = 10.0 / 3.6  # 10 km/h [R10]
    width_m: float = 3.0  # TEAM ASSUMPTION
    min_thermal_inertia: float = 100.0  # same threshold as RoboticRover [R4, TEAM ASSUMPTION]
    dunes_nogo: bool = True  # [R3]


@dataclass(frozen=True)
class Astronaut:
    """Suited astronaut on EVA."""

    max_slope_deg: float = 20.0  # [R11]
    max_step_up_m: float = 0.30  # [R12]
    max_step_down_m: float = 0.225  # TEAM ASSUMPTION: 75% of max_step_up_m
    v_nominal_ms: float = 3.5 / 3.6  # 3.5 km/h [R13]
    v_long_ms: float = 2.7 / 3.6  # 2.7 km/h, walks longer than 1 h [R13]
    v_max_ms: float = 5.6 / 3.6  # 5.6 km/h, VO2max-limited [R13]
    eva_nominal_h: float = 6.0  # [R11]
    eva_margin_h: float = 2.0  # [R11]
    max_distance_from_lander_m: float = 2000.0  # walkback limit [R11]
    sep_return_min: float = 30.0  # minutes to reach shelter after SEP warning [R20]
    body_mass_kg: float = 80.0  # TEAM ASSUMPTION
    suit_mass_kg: float = 104.0  # [R13]
    walkback_power_W: float = 498.0  # 1700 BTU/h average walkback [R13]
    eye_height_m: float = 1.8  # TEAM ASSUMPTION
