"""Mobility constraints for the agents moving in Jezero crater.

All quantities are SI (m, s, kg, W, deg for angles). Every constant carries the
ID of its source in square brackets; the full references are in ``SOURCES``.
Values marked "TEAM ASSUMPTION" are not taken from a reference.
"""

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Global physical constants
# --------------------------------------------------------------------------
MARS_RADIUS_M = 3_396_190.0  # IAU Mars sphere radius (Mars Trek / USGS) [R31]
G_MARS = 3.721  # m/s^2, commonly used surface gravity (fact sheet: 3.71) [R32]
G_EARTH = 9.80665  # m/s^2, standard gravity (CGPM 1901)
G_REL = G_MARS / G_EARTH  # ~0.379, ratio Mars/Earth gravity

# Terrain coefficient eta for the Pandolf equation.
# TEAM ASSUMPTION, derived from the terrain factors reported in [R18].
ETA_TERRAIN = {
    "rock": 1.0,
    "delta": 1.1,
    "regolith": 1.1,
    "rocky": 1.3,
    "sand": 1.5,
}

SOURCES = {
    "R1": "NASA, Mars 2020 Perseverance Rover Components (mars.nasa.gov)",
    "R2": "Rankin et al. 2023, IEEE Aerospace Conference (Perseverance driving performance)",
    "R3": "NASA/JPL PIA07999, Opportunity at Purgatory Dune",
    "R4": "Morgan et al. 2018, Space Science Reviews 214:104 (InSight landing site thermal inertia)",
    "R5": "Golombek & Rapp 1997, JGR 102(E2):4117 (Mars rock size-frequency distributions)",
    "R7": "NASA SP-4214 / NASA TM X-64623, Lunar Roving Vehicle operations handbook",
    "R9": "MIT 16.89 Space Systems Engineering, 2007",
    "R10": "NASA Space Exploration Vehicle (SEV) fact sheet",
    "R11": "NASA EVA-EXP-0075, Exploration EVA System Concept of Operations",
    "R12": "NASA ICES-2025-294, step height for suited locomotion",
    "R13": "NASA ICES 2026, Mars EVA suit mass and metabolic rates",
    "R15": "Pandolf, Givoni & Goldman 1977, J Appl Physiol 43:577",
    "R18": "Richmond et al. 2015, J Sport Hum Perf 3:1 (terrain factors)",
    "R20": "NASA NESC, space weather / SEP warning and shelter times",
    "R24": "Mars Climate Database v6.2",
    "R27": "Minetti et al. 2002, J Appl Physiol 93:1039 (energy cost on gradients)",
    "R28": "Tobler 1993, Three presentations on geographical analysis (hiking function)",
    "R29": "Horn 1981, Proc IEEE 69:14 (hill shading and slope)",
    "R30": "Riley, DeGloria & Elliot 1999, Terrain Ruggedness Index",
    "R31": (
        "Archinal et al. 2018, Report of the IAU Working Group on Cartographic Coordinates "
        "and Rotational Elements: 2015, Celest. Mech. Dyn. Astr. 130:22 "
        "(Mars sphere R = 3396.19 km, used by Mars Trek / USGS products)"
    ),
    "R32": (
        "NASA NSSDCA Mars Fact Sheet (surface gravity 3.71 m/s^2; "
        "3.721 m/s^2 is the commonly used value)"
    ),
}


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
